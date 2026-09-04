# -*- coding: utf-8 -*-
"""Forward-only Rhomb-E motion prototype.

Compatible with both Grasshopper IronPython 2.7 and regular Python 3.
The module deliberately has no Rhino or Grasshopper dependency.
"""

import math
from collections import namedtuple


class Pose(namedtuple(
        "PoseBase",
        "u_left v_left w_left u_right v_right w_right")):
    __slots__ = ()

    def as_tuple(self):
        return tuple(self)


class MotionPlan(object):
    __slots__ = ("poses", "heading_is_fallback", "target", "reached", "message")

    def __init__(self, poses, heading_is_fallback, target, reached, message):
        self.poses = tuple(poses)
        self.heading_is_fallback = tuple(heading_is_fallback)
        self.target = target
        self.reached = bool(reached)
        self.message = message

    def pose_at(self, step):
        index = max(0, min(int(step), len(self.poses) - 1))
        return self.poses[index]

    def heading_is_fallback_at(self, step):
        index = max(0, min(int(step), len(self.heading_is_fallback) - 1))
        return self.heading_is_fallback[index]

    @property
    def last_step(self):
        return len(self.poses) - 1


START_POSE = Pose(0, 0, 1, 2, 0, 1)
MAX_V_STRIDE = 4
MAX_U_STRIDE = 2
MAX_STEPS = 5000
MAX_CACHE_SIZE = 256
_PLAN_CACHE = {}
_CACHE_ORDER = []

# Heading (facing direction, in degrees) when the two feet share the same
# u,v position - e.g. a purely vertical/climbing pose. There is no
# horizontal direction between the feet to derive a heading from in that
# case. Voxel coordinates alone cannot give a correct real-world heading
# either way (the RhomBlock lattice is not a simple orthogonal grid), so
# the actual heading is resolved from real Rhino points - see
# resolve_heading_degrees() below. This constant is only the very first
# fallback, used if a plan's first pose is somehow already degenerate.
DEFAULT_HEADING_DEGREES = 0.0

MAX_HEADING_MEMORY_SIZE = 256
_HEADING_MEMORY = {}
_HEADING_MEMORY_ORDER = []


def _is_heading_undefined(pose):
    return pose.u_left == pose.u_right and pose.v_left == pose.v_right


def _fallback_flags_for(poses):
    return [_is_heading_undefined(pose) for pose in poses]


def resolve_heading_degrees(target_key, is_fallback, point_a, point_b):
    """Real-world heading (degrees) of the line from point_a to point_b.

    point_a/point_b are (x, y, z) tuples in real Rhino world coordinates,
    not voxel coordinates - the RhomBlock lattice is not a simple
    orthogonal grid, so a heading derived from voxel u,v deltas alone does
    not match the true geometry and was the cause of a real, observed
    orientation bug. When the two feet share the same u,v (is_fallback),
    there is no horizontal direction to derive a heading from at all, so
    the last real heading computed for this target is reused instead.
    """
    if not is_fallback:
        delta_x = point_b[0] - point_a[0]
        delta_y = point_b[1] - point_a[1]
        heading = math.degrees(math.atan2(delta_y, delta_x))
        if target_key not in _HEADING_MEMORY:
            _HEADING_MEMORY_ORDER.append(target_key)
            if len(_HEADING_MEMORY_ORDER) > MAX_HEADING_MEMORY_SIZE:
                oldest = _HEADING_MEMORY_ORDER.pop(0)
                _HEADING_MEMORY.pop(oldest, None)
        _HEADING_MEMORY[target_key] = heading
        return heading
    return _HEADING_MEMORY.get(target_key, DEFAULT_HEADING_DEGREES)


class LegGeometry(namedtuple(
        "LegGeometryBase",
        "point_c angle_c_degrees height_c base_a base_b "
        "normal_x normal_y normal_z")):
    __slots__ = ()


MAX_LEG_GEOMETRY_MEMORY_SIZE = 256
_LEG_GEOMETRY_MEMORY = {}
_LEG_GEOMETRY_MEMORY_ORDER = []


def _vector_sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _vector_length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vector_normalized(v):
    length = _vector_length(v)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


MAX_DIRECTION_MEMORY_SIZE = 256
_DIRECTION_MEMORY = {}
_DIRECTION_MEMORY_ORDER = []


def _remember_direction(target_key, unit_direction):
    if target_key not in _DIRECTION_MEMORY:
        _DIRECTION_MEMORY_ORDER.append(target_key)
        if len(_DIRECTION_MEMORY_ORDER) > MAX_DIRECTION_MEMORY_SIZE:
            oldest = _DIRECTION_MEMORY_ORDER.pop(0)
            _DIRECTION_MEMORY.pop(oldest, None)
    _DIRECTION_MEMORY[target_key] = unit_direction


def resolve_point_from_offsets(target_key, point_a, point_b, base_a, height_c):
    """Drop-in replacement for a Grasshopper cluster that built a point by
    offsetting point_a by base_a along the point_a -> point_b direction,
    then by height_c perpendicular to it (the same construction the
    "Build Triangle Points with Vectors" cluster did with Pt_A, Pt_B,
    base_a, heightC).

    Self-contained - does not need voxel u,v state from the pose planner,
    only the real points and the already-computed base_a/height_c values,
    so it can drop straight into that cluster's exact spot without
    creating any new wiring path back into the pose-planner component
    (which is what caused the "Cyclical data stream" error before).

    Robust to point_a/point_b being (near) vertically stacked - the
    horizontal direction between them is undefined in that case, so the
    last real direction seen for this target is reused (the point stays
    put) instead of collapsing onto a fixed world axis, avoiding both a
    crash and a sudden visual jump.
    """
    direction = _vector_sub(point_b, point_a)
    horizontal_length = math.sqrt(direction[0] ** 2 + direction[1] ** 2)

    if horizontal_length > 1e-6:
        unit_direction = _vector_normalized(direction)
        _remember_direction(target_key, unit_direction)
    else:
        unit_direction = _DIRECTION_MEMORY.get(target_key, (1.0, 0.0, 0.0))

    reference = (0.0, 0.0, 1.0) if abs(unit_direction[2]) < 0.9 else (1.0, 0.0, 0.0)
    side = _vector_normalized(_cross(unit_direction, reference))
    normal = _vector_normalized(_cross(side, unit_direction))

    return (
        point_a[0] + base_a * unit_direction[0] + height_c * normal[0],
        point_a[1] + base_a * unit_direction[1] + height_c * normal[1],
        point_a[2] + base_a * unit_direction[2] + height_c * normal[2],
    )


def _triangle_geometry(leg_a, leg_b, side_c):
    """Angle at apex C (degrees), height from C onto side c, and the
    offsets from point_a/point_b to the foot of that altitude, for a
    triangle with sides leg_a (point_a to C), leg_b (point_b to C) and
    side_c (point_a to point_b). Clamps side_c to the range the triangle
    inequality allows, so degenerate near/far foot distances cannot throw
    a math-domain error (this is the same clamp that was added directly
    to the original Grasshopper leg-triangle script).
    """
    c_min = abs(leg_a - leg_b) + 1e-6
    c_max = (leg_a + leg_b) - 1e-6
    c_clamped = max(c_min, min(side_c, c_max))

    cos_c = (c_clamped ** 2 - leg_a ** 2 - leg_b ** 2) / (-2.0 * leg_a * leg_b)
    cos_c = max(-1.0, min(1.0, cos_c))
    angle_c_degrees = math.degrees(math.acos(cos_c))

    s = (leg_a + leg_b + c_clamped) / 2.0
    area = math.sqrt(max(
        0.0, s * (s - leg_a) * (s - leg_b) * (s - c_clamped)))
    height_c = 2.0 * area / c_clamped
    base_a = math.sqrt(max(0.0, leg_a ** 2 - height_c ** 2))
    base_b = math.sqrt(max(0.0, leg_b ** 2 - height_c ** 2))
    return angle_c_degrees, height_c, base_a, base_b


def _remember_leg_geometry(target_key, geometry):
    if target_key not in _LEG_GEOMETRY_MEMORY:
        _LEG_GEOMETRY_MEMORY_ORDER.append(target_key)
        if len(_LEG_GEOMETRY_MEMORY_ORDER) > MAX_LEG_GEOMETRY_MEMORY_SIZE:
            oldest = _LEG_GEOMETRY_MEMORY_ORDER.pop(0)
            _LEG_GEOMETRY_MEMORY.pop(oldest, None)
    _LEG_GEOMETRY_MEMORY[target_key] = geometry


def resolve_leg_geometry(target_key, is_fallback, point_a, point_b, leg_a, leg_b):
    """Full real-world leg-triangle geometry: the shared knee/apex point C,
    the angle at C, the triangle height and base offsets, and a stable
    perpendicular ("normal") direction - everything the original
    Grasshopper leg-triangle script (angleC/heightC/base_a/base_b/normal)
    and the point-C cluster computed separately, in one place.

    Robust to point_a and point_b being vertically stacked (no unique
    perpendicular direction exists in that case - the last real geometry
    computed for this target is reused instead, same fallback strategy as
    resolve_heading_degrees).

    point_a/point_b: real Rhino (x, y, z) tuples for the current left and
    right foot. leg_a/leg_b: fixed physical leg lengths from point_a and
    point_b respectively to the apex point C.
    """
    direction = _vector_sub(point_b, point_a)
    side_c = _vector_length(direction)

    if is_fallback or side_c < 1e-9:
        return _LEG_GEOMETRY_MEMORY.get(target_key, LegGeometry(
            point_a, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

    unit_direction = _vector_normalized(direction)
    angle_c_degrees, height_c, base_a, base_b = _triangle_geometry(
        leg_a, leg_b, side_c)

    # Numerically stable perpendicular direction: pick whichever world
    # axis is least parallel to the point_a -> point_b direction as a
    # reference, avoiding the zero-length-cross-product singularity a
    # single fixed reference axis (e.g. always world Z) runs into when
    # that direction is itself (near) vertical.
    reference = (0.0, 0.0, 1.0) if abs(unit_direction[2]) < 0.9 else (1.0, 0.0, 0.0)
    side = _vector_normalized(_cross(unit_direction, reference))
    normal = _vector_normalized(_cross(side, unit_direction))

    point_c = (
        point_a[0] + base_a * unit_direction[0] + height_c * normal[0],
        point_a[1] + base_a * unit_direction[1] + height_c * normal[1],
        point_a[2] + base_a * unit_direction[2] + height_c * normal[2],
    )
    geometry = LegGeometry(
        point_c, angle_c_degrees, height_c, base_a, base_b,
        normal[0], normal[1], normal[2])
    _remember_leg_geometry(target_key, geometry)
    return geometry


def _integer(value, name):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        raise ValueError(
            "{0} must be a number, received {1!r}".format(name, value)
        )


def _remember(key, plan):
    if key in _PLAN_CACHE:
        return
    _PLAN_CACHE[key] = plan
    _CACHE_ORDER.append(key)
    if len(_CACHE_ORDER) > MAX_CACHE_SIZE:
        oldest = _CACHE_ORDER.pop(0)
        _PLAN_CACHE.pop(oldest, None)


def plan_simple_motion(
        u_left, v_left, w_left, u_right, v_right, w_right):
    """Create the same sequence as the known-working GhPython component."""

    target = Pose(
        _integer(u_left, "u_left"),
        _integer(v_left, "v_left"),
        _integer(w_left, "w_left"),
        _integer(u_right, "u_right"),
        _integer(v_right, "v_right"),
        _integer(w_right, "w_right"),
    )
    cache_key = target.as_tuple()
    cached = _PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    poses = [START_POSE]
    cur_uL, cur_vL, cur_wL = 0, 0, 1
    cur_uR, cur_vR, cur_wR = 2, 0, 1

    def add_step(uL, vL, wL, uR, vR, wR):
        if len(poses) < MAX_STEPS:
            pose = Pose(
                int(uL), int(vL), int(wL),
                int(uR), int(vR), int(wR),
            )
            # The reference code keeps duplicates. Preserve them exactly.
            poses.append(pose)

    # 1) V movement - exact order and W values from the reference component.
    if not (target.v_left < cur_vL or target.v_right < cur_vR):
        while ((cur_vL != target.v_left or cur_vR != target.v_right) and
               len(poses) < MAX_STEPS):
            moved = False
            if cur_vR < target.v_right:
                moved = True
                step_vR = min(4, target.v_right - cur_vR)
                add_step(cur_uL, cur_vL, 1, cur_uR, cur_vR, 2)
                for unused in range(step_vR):
                    cur_vR += 1
                    add_step(cur_uL, cur_vL, 1, cur_uR, cur_vR, 2)
                add_step(cur_uL, cur_vL, 1, cur_uR, cur_vR, 1)

            if cur_vL < target.v_left:
                moved = True
                step_vL = min(4, target.v_left - cur_vL)
                add_step(cur_uL, cur_vL, 2, cur_uR, cur_vR, 1)
                for unused in range(step_vL):
                    cur_vL += 1
                    add_step(cur_uL, cur_vL, 2, cur_uR, cur_vR, 1)
                add_step(cur_uL, cur_vL, 1, cur_uR, cur_vR, 1)

            if not moved:
                break

    # 2) U movement - exact order and W values from the reference component.
    if not (target.u_left < cur_uL or target.u_right < cur_uR):
        while ((cur_uL != target.u_left or cur_uR != target.u_right) and
               len(poses) < MAX_STEPS):
            moved = False
            if cur_uR < target.u_right:
                moved = True
                step_uR = min(2, target.u_right - cur_uR)
                add_step(cur_uL, cur_vL, 1, cur_uR, cur_vR, 2)
                for unused in range(step_uR):
                    cur_uR += 1
                    add_step(cur_uL, cur_vL, 1, cur_uR, cur_vR, 2)
                add_step(cur_uL, cur_vL, 1, cur_uR, cur_vR, 1)

            if cur_uL < target.u_left:
                moved = True
                step_uL = min(2, target.u_left - cur_uL)
                add_step(cur_uL, cur_vL, 2, cur_uR, cur_vR, 1)
                for unused in range(step_uL):
                    cur_uL += 1
                    add_step(cur_uL, cur_vL, 2, cur_uR, cur_vR, 1)
                add_step(cur_uL, cur_vL, 1, cur_uR, cur_vR, 1)

            if not moved:
                break

    # 3) W movement - right foot first, then left foot.
    while cur_wR < target.w_right and len(poses) < MAX_STEPS:
        cur_wR += 1
        add_step(cur_uL, cur_vL, cur_wL, cur_uR, cur_vR, cur_wR)

    while cur_wL < target.w_left and len(poses) < MAX_STEPS:
        cur_wL += 1
        add_step(cur_uL, cur_vL, cur_wL, cur_uR, cur_vR, cur_wR)

    reached = poses[-1] == target
    if len(poses) >= MAX_STEPS and not reached:
        message = "Maximum step count reached"
    elif reached:
        message = "Target reached"
    else:
        message = "Target partly ignored by forward-only reference logic"
    plan = MotionPlan(
        poses,
        _fallback_flags_for(poses),
        target,
        reached,
        message,
    )
    _remember(cache_key, plan)
    return plan
