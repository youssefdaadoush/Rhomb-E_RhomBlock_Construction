# -*- coding: utf-8 -*-
"""Reusable motion primitives (A1, A2 - see Table 1 of the Rhomb-E paper).

Building-block layer of the redesigned motion engine:
Primitives -> Behaviors -> Planner.

A primitive only knows how to move ONE foot by a small amount and report
the intermediate poses it passed through. It has no opinion about which
foot should move when, in what overall order, or why - that decision
belongs to the Behaviors/Planner layers built on top of this module later.

This module is independent of rhombe_motion.simple_motion, which remains
the confirmed-working legacy forward-motion implementation and is left
untouched. It reuses only the Pose type from there.
"""

from rhombe_motion.simple_motion import Pose

FOOT_LEFT = "left"
FOOT_RIGHT = "right"
AXIS_U = "u"
AXIS_V = "v"
AXIS_W = "w"


def _field_name(foot, axis):
    return "{0}_{1}".format(axis, foot)


def get_coordinate(pose, foot, axis):
    """Read one foot's value on one axis (u, v or w) from a Pose."""
    return getattr(pose, _field_name(foot, axis))


def _with_coordinate(pose, foot, axis, value):
    return pose._replace(**{_field_name(foot, axis): value})


def lift_or_lower_foot(pose, foot, target_w):
    """A1 - Lift/Lower Free Foot.

    Moves one foot's w value toward target_w, one voxel unit at a time
    (direction is inferred automatically - this single primitive covers
    both lifting and lowering, matching how the paper's Table 1 treats
    A1 as one action).

    Returns (intermediate_poses, new_pose). intermediate_poses does not
    include the starting pose, only the poses produced by this call, in
    order. If the foot is already at target_w, returns ([], pose).
    """
    current = get_coordinate(pose, foot, AXIS_W)
    if current == target_w:
        return [], pose

    step = 1 if target_w > current else -1
    poses = []
    while current != target_w:
        current += step
        pose = _with_coordinate(pose, foot, AXIS_W, current)
        poses.append(pose)
    return poses, pose


def translate_foot(pose, foot, axis, target, max_stride, adjust_step=None):
    """A2 - Translate Forward Foot.

    Moves one foot along one horizontal axis (u or v) toward target, in a
    single stride capped at max_stride voxel units.

    ``adjust_step`` is an optional hook for landing constraints that go
    beyond a plain distance cap - e.g. the reference implementation
    required a carrying foot to always land on an odd u coordinate, and to
    stay within a maximum distance of the other foot. Rather than baking
    those rules into this primitive, pass a callable:

        adjust_step(pose, foot, axis, naive_step) -> final_step

    which receives the step this primitive would take by default
    (already capped at max_stride, signed toward target) and returns the
    step to actually take. Returning 0 means "do not move this call".
    Leave ``adjust_step`` as None for plain, unconstrained translation.

    Returns (intermediate_poses, new_pose), moving one voxel unit per
    intermediate pose (matching how the physical robot advances one
    module at a time).
    """
    current = get_coordinate(pose, foot, axis)
    remaining = target - current
    if remaining == 0:
        return [], pose

    direction = 1 if remaining > 0 else -1
    naive_step = direction * min(abs(remaining), max_stride)

    step = naive_step
    if adjust_step is not None:
        step = adjust_step(pose, foot, axis, naive_step)
    if step == 0:
        return [], pose

    unit = 1 if step > 0 else -1
    poses = []
    for _ in range(abs(step)):
        current += unit
        pose = _with_coordinate(pose, foot, axis, current)
        poses.append(pose)
    return poses, pose


def rotate_support_foot(
        pose, foot, target_u, target_v, target_w,
        max_u_stride, max_v_stride,
        adjust_u_step=None, adjust_v_step=None):
    """A3 - Rotate Support Foot.

    Table 1 defines A3 as the composition A1 + A2: this primitive is
    literally that composition, applied to one foot across both
    horizontal axes plus its w value, so a foot can pivot to a new u,v
    position (not just move along a single axis like translate_foot)
    while its height changes too.

    Order: lift/lower first (A1), then translate on u (A2), then
    translate on v (A2) - matching the "lift, move, settle" shape every
    other stepping motion in this module uses, just generalized to two
    horizontal axes at once instead of one.

    Returns (intermediate_poses, new_pose).
    """
    poses = []

    w_poses, pose = lift_or_lower_foot(pose, foot, target_w)
    poses.extend(w_poses)

    u_poses, pose = translate_foot(
        pose, foot, AXIS_U, target_u, max_u_stride, adjust_u_step)
    poses.extend(u_poses)

    v_poses, pose = translate_foot(
        pose, foot, AXIS_V, target_v, max_v_stride, adjust_v_step)
    poses.extend(v_poses)

    return poses, pose
