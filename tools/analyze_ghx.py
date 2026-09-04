import argparse
import collections
import json
import pathlib
import xml.etree.ElementTree as ET


def direct_items(node):
    items = node.find("items")
    if items is None:
        return []
    return list(items.findall("item"))


def item_values(node, name):
    return [(item.text or "").strip() for item in direct_items(node) if item.get("name") == name]


def item_value(node, name, default=""):
    values = item_values(node, name)
    return values[0] if values else default


def find_child_chunk(node, name):
    chunks = node.find("chunks")
    if chunks is None:
        return None
    for chunk in chunks.findall("chunk"):
        if chunk.get("name") == name:
            return chunk
    return None


def parameter_records(container, kind):
    data = find_child_chunk(container, "ParameterData")
    if data is None:
        return []
    chunks = data.find("chunks")
    if chunks is None:
        return []
    records = []
    for param in chunks.findall("chunk"):
        if param.get("name") != kind:
            continue
        records.append(
            {
                "index": int(param.get("index", "0")),
                "name": item_value(param, "Name"),
                "nickname": item_value(param, "NickName"),
                "guid": item_value(param, "InstanceGuid"),
                "sources": item_values(param, "Source"),
                "source_count": int(item_value(param, "SourceCount", "0") or 0),
                "access": item_value(param, "ScriptParamAccess"),
                "optional": item_value(param, "Optional"),
            }
        )
    return sorted(records, key=lambda record: record["index"])


def analyze(path):
    root = ET.parse(path).getroot()
    definition_objects = None
    for chunk in root.iter("chunk"):
        if chunk.get("name") == "DefinitionObjects":
            definition_objects = chunk
            break
    if definition_objects is None:
        raise RuntimeError("DefinitionObjects chunk not found")

    chunks_node = definition_objects.find("chunks")
    objects = []
    for obj in chunks_node.findall("chunk") if chunks_node is not None else []:
        if obj.get("name") != "Object":
            continue
        container = find_child_chunk(obj, "Container")
        if container is None:
            continue
        attributes = find_child_chunk(container, "Attributes")
        bounds = ""
        if attributes is not None:
            bounds_item = next((x for x in direct_items(attributes) if x.get("name") == "Bounds"), None)
            if bounds_item is not None:
                bounds = {
                    child.tag: float(child.text)
                    for child in list(bounds_item)
                    if child.text is not None
                }
        record = {
            "index": int(obj.get("index", "0")),
            "type_name": item_value(obj, "Name"),
            "type_guid": item_value(obj, "GUID"),
            "name": item_value(container, "Name"),
            "nickname": item_value(container, "NickName"),
            "instance_guid": item_value(container, "InstanceGuid"),
            "description": item_value(container, "Description"),
            "hidden": item_value(container, "Hidden"),
            "locked": item_value(container, "Locked"),
            "bounds": bounds,
            "inputs": parameter_records(container, "InputParam"),
            "outputs": parameter_records(container, "OutputParam"),
            "code": item_value(container, "CodeInput"),
            "group_members": item_values(container, "ID"),
        }
        objects.append(record)
    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ghx", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    objects = analyze(args.ghx)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "objects.json").write_text(
        json.dumps(objects, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    counts = collections.Counter(obj["type_name"] for obj in objects)
    lines = [
        f"Objects: {len(objects)}",
        f"Types: {len(counts)}",
        f"Script components: {sum(bool(obj['code']) for obj in objects)}",
        "",
        "Object types:",
    ]
    lines.extend(f"{count:4d}  {name}" for name, count in counts.most_common())
    lines.extend(["", "Scripts:"])
    scripts_dir = args.output / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    for obj in objects:
        if not obj["code"]:
            continue
        script_name = f"{obj['index']:03d}_{obj['instance_guid']}.py"
        (scripts_dir / script_name).write_text(obj["code"], encoding="utf-8")
        inputs = ", ".join(param["nickname"] or param["name"] for param in obj["inputs"])
        outputs = ", ".join(param["nickname"] or param["name"] for param in obj["outputs"])
        lines.append(
            f"{obj['index']:3d}  {obj['nickname'] or obj['name']}  "
            f"[{obj['instance_guid']}]  in=({inputs}) out=({outputs})"
        )
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
