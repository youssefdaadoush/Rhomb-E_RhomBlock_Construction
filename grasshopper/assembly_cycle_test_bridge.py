"""Test bridge for the complete D1+B2+D2+D3 assembly cycle: pick up a
block, carry it to a target, place it, and return home - matching the
paper's "approaching, gripping, transporting, placing, and returning".

Grasshopper inputs:
    S, stoneU, stoneV, stoneW,
    targetUL, targetVL, targetWL, targetUR, targetVR, targetWR,
    occupiedU, occupiedV,
    project_path, reload_code

Grasshopper outputs:
    uL_out, vL_out, wL_out, uR_out, vR_out, wR_out,
    isCarrying_out, stonePlaced_out, last_step,
    occupiedU_out, occupiedV_out

``targetU/V/WL`` and ``targetU/V/WR`` describe where the robot should be
standing right after placing the block (see complete_assembly_cycle's
docstring in rhombe_motion/work_sequences.py for the exact meaning of the
carrying foot's w there).

``occupiedU``/``occupiedV`` (optional, parallel lists) describe (u, v)
columns already built on by EARLIER assembly cycles - wire the previous
run's ``occupiedU_out``/``occupiedV_out`` back into these two inputs to
chain multiple placements in the same build, so this cycle's pickup,
transport and return legs never try to plant a foot back inside a column
built on earlier. Leave both empty for the very first stone.

``isCarrying_out`` is True/False for the current step S - wire it to a
Boolean toggle / Gate / IF component to show or hide the block attached
to the robot's foot (visible only while actually being carried).

``stonePlaced_out`` is a SEPARATE True/False signal for the current step
S: it becomes True the moment the block reaches its final voxel and
STAYS True for the rest of the cycle (through release and the whole trip
home), unlike isCarrying_out, which goes False right after release. Wire
this to a persistent "keep showing the block resting at its target"
toggle.

``occupiedU_out``/``occupiedV_out`` are this cycle's ``occupiedU``/
``occupiedV`` plus the column this cycle itself just placed a stone in -
feed them into the NEXT assembly-cycle component's ``occupiedU``/
``occupiedV`` inputs to keep building on the same structure.
"""

import importlib
import sys

if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)

import rhombe_motion.primitives as primitives
import rhombe_motion.simple_motion as simple_motion
import rhombe_motion.behaviors as behaviors
import rhombe_motion.work_sequences as work_sequences

if reload_code:
    # Reload in dependency order: primitives -> behaviors -> work_sequences
    # (each imports names from the one before it via "from X import Y").
    # Reloading only work_sequences (as this used to do) re-runs its
    # import statements, but those just re-bind names from whatever
    # primitives/behaviors modules already sit in sys.modules - if THOSE
    # were never reloaded, edits to behaviors.py or primitives.py stay
    # invisible for the rest of the Rhino session no matter how many
    # times reload_code is toggled, until Rhino is fully restarted. This
    # was very likely the cause of the repeated "nothing changed" reports.
    primitives = importlib.reload(primitives)
    simple_motion = importlib.reload(simple_motion)
    behaviors = importlib.reload(behaviors)
    work_sequences = importlib.reload(work_sequences)

target_pose = simple_motion.Pose(
    int(targetUL), int(targetVL), int(targetWL),
    int(targetUR), int(targetVR), int(targetWR))

occupied_in = set()
if occupiedU and occupiedV:
    occupied_in = set(zip((int(u) for u in occupiedU), (int(v) for v in occupiedV)))

poses, is_carrying, stone_placed, occupied_out = work_sequences.complete_assembly_cycle(
    simple_motion.START_POSE,
    int(stoneU), int(stoneV), int(stoneW),
    target_pose,
    occupied=occupied_in)

step = max(0, min(int(S), len(poses) - 1))
pose = poses[step]

uL_out, vL_out, wL_out, uR_out, vR_out, wR_out = pose.as_tuple()
isCarrying_out = is_carrying[step]
stonePlaced_out = stone_placed[step]
last_step = len(poses) - 1

occupiedU_out = [c[0] for c in sorted(occupied_out)]
occupiedV_out = [c[1] for c in sorted(occupied_out)]
