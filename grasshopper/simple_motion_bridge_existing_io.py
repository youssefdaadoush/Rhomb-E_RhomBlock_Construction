"""Drop-in bridge using the existing GhPython names: S, a, b, c, d, e, f.

Add inputs ``project_path``, ``reload_code``, and two real Rhino points
``pt_A`` / ``pt_B`` for the current left/right foot (not voxel
coordinates). The six existing outputs remain ``a, b, c, d, e, f``.
Optional diagnostic outputs can be added as ``heading``, ``last_step``,
``reached`` and ``status``.

``heading`` is fully resolved already (including the one case where the
two feet share the same u,v and no real heading can be derived from real
points) - wire it directly to whatever previously consumed the native
Angle/Degrees chain, no extra blending needed on the Grasshopper side.
"""

import importlib
import sys

if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)

import rhombe_motion.simple_motion as simple_motion

if reload_code:
    simple_motion = importlib.reload(simple_motion)

plan = simple_motion.plan_simple_motion(a, b, c, d, e, f)
pose = plan.pose_at(S)
is_fallback = plan.heading_is_fallback_at(S)

a, b, c, d, e, f = pose.as_tuple()
heading = simple_motion.resolve_heading_degrees(
    plan.target.as_tuple(),
    is_fallback,
    (pt_A.X, pt_A.Y, pt_A.Z),
    (pt_B.X, pt_B.Y, pt_B.Z),
)
last_step = plan.last_step
reached = plan.reached
status = plan.message
