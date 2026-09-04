"""Second Grasshopper Python 3 component: heading + leg-triangle geometry.

This MUST be a SEPARATE component from the pose planner
(simple_motion_bridge.py), not merged into it. Reason: pt_A/pt_B (the real
Rhino points for the current feet) are computed downstream FROM the pose
planner's own uL_out/vL_out/wL_out/uR_out/vR_out/wR_out outputs (a block
face point lifted by the resolved w value). If one component both produced
those pose outputs AND consumed pt_A/pt_B, Grasshopper sees a genuine
cycle - this is exactly the observed error "Cyclical data stream detected,
parameter pt_A is recursive".

Grasshopper inputs:
    S, uL, vL, wL, uR, vR, wR, project_path, reload_code,
    pt_A, pt_B, leg_a, leg_b

``S``/``uL``/``vL``/``wL``/``uR``/``vR``/``wR`` must come from the SAME
sliders that feed the pose planner component - NOT from its
uL_out/vL_out/etc. outputs. That keeps this component's target/step
lookup independent of anything pt_A/pt_B depend on, which is what breaks
the cycle. plan_simple_motion() is cached (see simple_motion.py), so
recomputing the plan here is cheap, not duplicated work.

Grasshopper outputs:
    heading_out,
    ptC_x, ptC_y, ptC_z, angleC_out, heightC_out, base_a_out, base_b_out,
    normal_x, normal_y, normal_z

Every output is already fully resolved (including the one case where the
two feet share the same u,v and no real value can be derived) - wire them
directly to whatever previously consumed the native Angle/Degrees chain,
the leg-triangle script, and the point-C cluster. No extra blending or
cross-product wiring needed on the Grasshopper side.
"""

import importlib
import sys

if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)

import rhombe_motion.simple_motion as simple_motion

if reload_code:
    simple_motion = importlib.reload(simple_motion)

plan = simple_motion.plan_simple_motion(uL, vL, wL, uR, vR, wR)
is_fallback = plan.heading_is_fallback_at(S)
target_key = plan.target.as_tuple()
point_a = (pt_A.X, pt_A.Y, pt_A.Z)
point_b = (pt_B.X, pt_B.Y, pt_B.Z)

heading_out = simple_motion.resolve_heading_degrees(
    target_key, is_fallback, point_a, point_b)

geometry = simple_motion.resolve_leg_geometry(
    target_key, is_fallback, point_a, point_b, leg_a, leg_b)
ptC_x, ptC_y, ptC_z = geometry.point_c
angleC_out = geometry.angle_c_degrees
heightC_out = geometry.height_c
base_a_out = geometry.base_a
base_b_out = geometry.base_b
normal_x = geometry.normal_x
normal_y = geometry.normal_y
normal_z = geometry.normal_z
