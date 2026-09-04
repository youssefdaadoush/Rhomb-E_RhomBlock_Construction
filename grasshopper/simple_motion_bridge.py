"""Pose-planner Grasshopper Python component - the FIRST of two components.

Only computes the six pose values and diagnostics. Heading and leg
geometry (angleC/heightC/base_a/base_b/normal/point C) live in a SEPARATE
component - see simple_motion_geometry_bridge.py - because they need
pt_A/pt_B, and pt_A/pt_B are themselves computed downstream from this
component's own uL_out/vL_out/wL_out/uR_out/vR_out/wR_out outputs. Feeding
pt_A/pt_B back into this same component creates a genuine dependency
cycle in Grasshopper ("Cyclical data stream detected, parameter pt_A is
recursive") - it cannot be wired around, only avoided by keeping this a
separate, upstream component.

Grasshopper inputs:
    S, uL, vL, wL, uR, vR, wR, project_path, reload_code

Grasshopper outputs:
    uL_out, vL_out, wL_out, uR_out, vR_out, wR_out,
    last_step, reached, status
"""

import importlib
import sys

if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)

import rhombe_motion.simple_motion as simple_motion

if reload_code:
    simple_motion = importlib.reload(simple_motion)

plan = simple_motion.plan_simple_motion(uL, vL, wL, uR, vR, wR)
pose = plan.pose_at(S)

uL_out, vL_out, wL_out, uR_out, vR_out, wR_out = pose.as_tuple()
last_step = plan.last_step
reached = plan.reached
status = plan.message
