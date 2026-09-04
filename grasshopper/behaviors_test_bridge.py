"""Test bridge for the new primitives-based B1 (Move Forward, No Load).

Grasshopper inputs (identical to the existing pose-planner component -
no new points, no leg lengths, nothing that could create a dependency
cycle):
    S, uL, vL, wL, uR, vR, wR, project_path, reload_code

Grasshopper outputs:
    uL_out, vL_out, wL_out, uR_out, vR_out, wR_out, last_step

Compare these six outputs against the existing, working pose-planner
component's outputs for the same S/uL/vL/wL/uR/vR/wR - they must match
exactly at every step (this is already proven automatically by
tests/test_behaviors.py; this component lets you see it live in
Grasshopper too).
"""

import importlib
import sys

if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)

import rhombe_motion.primitives as primitives
import rhombe_motion.simple_motion as simple_motion
import rhombe_motion.behaviors as behaviors

if reload_code:
    # Reload in dependency order (primitives first, since behaviors
    # imports names from it) - otherwise behaviors.py picking up new code
    # would still bind to a stale, only-ever-imported-once primitives
    # module, and any edit there would silently not take effect until
    # Rhino is fully restarted.
    primitives = importlib.reload(primitives)
    simple_motion = importlib.reload(simple_motion)
    behaviors = importlib.reload(behaviors)

target = simple_motion.Pose(
    int(uL), int(vL), int(wL), int(uR), int(vR), int(wR))
poses = behaviors.move_forward_no_load(simple_motion.START_POSE, target)

step = max(0, min(int(S), len(poses) - 1))
pose = poses[step]

uL_out, vL_out, wL_out, uR_out, vR_out, wR_out = pose.as_tuple()
last_step = len(poses) - 1
