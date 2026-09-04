# -*- coding: utf-8 -*-
"""Work sequences layer (D1-D3, using C1/C2 - Table 1). Composite,
multi-primitive choreographies that do not fit the plain "lift, move,
settle" shape every B-level behavior uses - reaching for a block,
gripping it, placing it, backing off.

Built directly from rhombe_motion.primitives (A1/A2) and reuses
rhombe_motion.behaviors (B1/B2) for the plain transport/return legs.
Grip (C1) and release (C2) are conceptual moments in these sequences
(the physical gripper action) - this module does not model a separate
gripper state on Pose, only documents where they occur, matching how
the reference implementation treated them as instantaneous. Instead,
every function in this module returns a second, parallel list -
``is_carrying`` - alongside the poses, so a caller (e.g. Grasshopper)
knows at every single step whether the robot is currently holding a
block, without having to guess it from the pose numbers. This is what
should drive a "show/hide the carried block" visual toggle.

Numeric defaults (transit height 5, side shift 2, reposition -1, v_shift
1 for release) come directly from the original working reference
implementation for picking up and placing a single block.

Placed stones (``occupied``): once a block is placed at its target
column, that (u, v) column is physically solid ground-up - no foot may
ever rest there again (see complete_assembly_cycle). Every function in
this module and in behaviors.py that decides where a foot may land
accepts an optional ``occupied`` set of (u, v) tuples for exactly this;
passing it is what keeps the planner from trying to plant a foot back
inside a column it (or an earlier cycle) already built on.
"""

from rhombe_motion.primitives import (
    FOOT_LEFT,
    FOOT_RIGHT,
    AXIS_U,
    AXIS_V,
    get_coordinate,
    lift_or_lower_foot,
    translate_foot,
)
from rhombe_motion.behaviors import (
    MAX_U_STRIDE,
    MAX_V_STRIDE,
    SUPPORT_W,
    SUPPORT_W_WITH_LOAD,
    move_forward_no_load,
    move_forward_with_load,
    _white_correction_targets,
    _land_without_collision,
    _land_without_collision_v,
)
from rhombe_motion.simple_motion import START_POSE

# Return-trip transit v: after placing a block, return_home first moves
# to this v (both feet) before making its final approach to home_pose,
# so the return trip does not walk back along v=0 - where the material
# station (the loose-stone pickup point most stone_v values use) sits -
# for the whole trip (user-confirmed: this was causing unnecessary
# conflicts with the material station). Even, matching the white/
# no-load foothold constraint (see _land_on_even_v in behaviors.py).
RETURN_TRANSIT_V = 2


def _stride_to_target(pose, foot, axis, target, max_stride, adjust_step=None):
    """Repeatedly strides (translate_foot calls) along one axis until
    reaching target or making no further progress - unlike a single
    translate_foot call, which only ever covers up to max_stride toward
    target in one go and silently stops short for anything further
    (this was a real bug: pickup_sequence's stone-approach move used a
    single translate_foot call, so a stone more than max_u_stride away
    was never actually reached - the foot would descend at the wrong u,
    short of the real stone). Stops early (without reaching target) if a
    stride returns no poses (blocked - by collision, parity, or the
    physical leg-span limit), same as a single translate_foot call would
    signal "can't move" by returning 0.
    """
    poses = []
    while get_coordinate(pose, foot, axis) != target:
        step_poses, pose = translate_foot(pose, foot, axis, target, max_stride, adjust_step)
        if not step_poses:
            break
        poses.extend(step_poses)
    return poses, pose


def pickup_sequence(
        pose, stone_u, stone_v, stone_w, carrying_foot=FOOT_RIGHT,
        transit_w=5, side_shift=2, reposition=-1,
        max_u_stride=MAX_U_STRIDE, max_v_stride=MAX_V_STRIDE, occupied=None):
    """D1 - Pickup Sequence.

    Reaches over to a block at (stone_u, stone_v, stone_w) with the
    carrying foot, grips it (C1), and settles back into a stable
    with-load stance, shifted sideways (side_shift) and repositioned
    (reposition) so the carrying foot ends on a valid red/odd-sum
    foothold (see _land_on_odd_u in behaviors.py). Composition:
    A1 + A2 + C1 (repeated).

    The free foot never moves during this whole sequence - it stays
    exactly where it started. That means the stone, and the carrying
    foot's whole approach/backoff path, must stay within the robot's
    physical leg-span limit (MAX_AXIS_DISTANCE in behaviors.py) of the
    free foot's CURRENT position throughout. Raises ValueError instead
    of silently gripping the wrong spot if the stone itself, or the
    side_shift/reposition backoff afterward, cannot actually be reached
    given that limit - move the free foot closer first (e.g. via
    move_forward_no_load) if this happens.

    ``occupied`` (optional): a set of (u, v) columns already built on by
    earlier placements (see complete_assembly_cycle) - the backoff/
    reposition landing will never settle there.

    Returns (poses, is_carrying) - two lists of equal length, both
    including the starting pose/state. is_carrying becomes True at the
    pose right after the foot lifts back off the block (the grip
    moment); every pose before that is False (still approaching, empty).
    """
    def land_without_collision(p, f, a, s):
        return _land_without_collision(p, f, a, s, occupied=occupied)

    def land_without_collision_v(p, f, a, s):
        return _land_without_collision_v(p, f, a, s, occupied=occupied)

    poses = [pose]
    is_carrying = [False]

    def run(step_result, carrying):
        step_poses, new_pose = step_result
        poses.extend(step_poses)
        is_carrying.extend([carrying] * len(step_poses))
        return new_pose

    pose = run(lift_or_lower_foot(pose, carrying_foot, transit_w), False)
    pose = run(_stride_to_target(
        pose, carrying_foot, AXIS_U, stone_u, max_u_stride,
        adjust_step=land_without_collision), False)
    # the v-approach was missing entirely in an earlier version of this
    # function - stone_v was accepted as a parameter and documented, but
    # never actually used to move the carrying foot there, so the robot
    # would silently grip whatever was at (stone_u, its ALREADY-current
    # v, stone_w) instead of the real stone whenever stone_v differed
    # from the foot's starting v. Approach it explicitly now, before
    # descending.
    pose = run(_stride_to_target(
        pose, carrying_foot, AXIS_V, stone_v, max_v_stride,
        adjust_step=land_without_collision_v), False)
    reached_u = get_coordinate(pose, carrying_foot, AXIS_U) == stone_u
    reached_v = get_coordinate(pose, carrying_foot, AXIS_V) == stone_v
    if not (reached_u and reached_v):
        # the carrying foot got stuck short of the stone - most likely
        # the physical leg-span limit (MAX_AXIS_DISTANCE, user-
        # confirmed) relative to the free foot's CURRENT position, which
        # pickup_sequence never moves (see "the free foot is untouched"
        # above). Fail loudly instead of silently descending and
        # gripping at the wrong (u, v) as if it were the stone.
        raise ValueError(
            "carrying foot could not reach the stone at (u={0}, v={1}) - "
            "stuck at (u={2}, v={3}) - likely too far from the free "
            "foot's current position given the robot's physical "
            "leg-span limit; reposition the free foot closer before "
            "picking up this stone".format(
                stone_u, stone_v,
                get_coordinate(pose, carrying_foot, AXIS_U),
                get_coordinate(pose, carrying_foot, AXIS_V)))
    pose = run(lift_or_lower_foot(pose, carrying_foot, stone_w), False)
    # --- grip (C1): carrying becomes True starting with the very next pose ---
    pose = run(lift_or_lower_foot(pose, carrying_foot, transit_w), True)
    target_v = get_coordinate(pose, carrying_foot, AXIS_V) + side_shift
    pose = run(_stride_to_target(
        pose, carrying_foot, AXIS_V, target_v, max_v_stride,
        adjust_step=land_without_collision_v), True)
    target_u = get_coordinate(pose, carrying_foot, AXIS_U) + reposition
    pose = run(_stride_to_target(
        pose, carrying_foot, AXIS_U, target_u, max_u_stride,
        adjust_step=land_without_collision), True)
    # side_shift/reposition are tuned (an odd u offset, an even v offset)
    # to always land on a valid red/odd-sum foothold - BUT only if both
    # strides actually reach their full target. Either one can get
    # capped short by the physical leg-span limit relative to the free
    # foot (same class of issue fixed above for the stone approach),
    # silently landing on an invalid (even-sum) foothold instead. Verify
    # directly rather than re-deriving every way that could happen.
    settled_u = get_coordinate(pose, carrying_foot, AXIS_U)
    settled_v = get_coordinate(pose, carrying_foot, AXIS_V)
    if (settled_u + settled_v) % 2 == 0:
        raise ValueError(
            "pickup_sequence's side_shift/reposition backoff landed on "
            "an invalid stance (u={0}, v={1}, even u+v sum - not a red "
            "foothold) instead of the intended (u={2}, v={3}) - likely "
            "capped short by the physical leg-span limit relative to "
            "the free foot's current position; reposition the free foot "
            "closer, or use a smaller side_shift/reposition, before "
            "picking up this stone".format(
                settled_u, settled_v, target_u, target_v))
    pose = run(lift_or_lower_foot(pose, carrying_foot, SUPPORT_W_WITH_LOAD), True)

    return poses, is_carrying


def placement_sequence(
        pose, target_w, carrying_foot=FOOT_RIGHT,
        v_shift=3, max_v_stride=MAX_V_STRIDE, max_u_stride=MAX_U_STRIDE,
        occupied=None):
    """D2 - Placement Sequence.

    Raises the carrying foot to target_w (placing the block at the
    target voxel), then backs off: raises one more unit, shifts sideways
    by v_shift, and lowers back down - releasing (C2) and clearing the
    placed block. Composition: A1 + A2 + C2.

    v_shift defaults to 3, not 1: a placed stone physically occupies its
    own cell plus its immediate left/right/front/back neighbors (user-
    confirmed directly against the Rhino model), so a 1-unit backoff
    still lands right on top of the stone's own footprint. 3 units of
    clearance (matching the user's stated rule) is required to reach a
    foothold that is actually clear of the stone.

    ``occupied`` (optional): a set of (u, v) columns already built on by
    earlier placements (see complete_assembly_cycle) - the backoff
    landing will never settle there. It does NOT need to (and should
    not) include this call's own target_w column - the carrying foot is
    SUPPOSED to land there, that is the placement itself.

    Returns (poses, is_carrying, stone_placed). is_carrying: the starting
    pose and the pose right after reaching target_w are still True (block
    placed but not yet released); every pose after that (the raise-shift-
    lower backoff) is False. stone_placed: False for the starting pose
    (block still mid-air, not yet at its final voxel), then True from the
    moment the carrying foot actually reaches target_w onward - including
    through release and the whole backoff - since the block itself does
    not move again after that, regardless of whether the foot is still
    gripping it. Wire this to a persistent "keep showing the placed
    block" toggle in Grasshopper, separate from is_carrying (which only
    tracks whether the block is currently attached to the foot).
    """
    def land_without_collision_v(p, f, a, s):
        return _land_without_collision_v(p, f, a, s, occupied=occupied)

    poses = [pose]
    is_carrying = [True]
    stone_placed = [False]

    def run(step_result, carrying, placed):
        step_poses, new_pose = step_result
        poses.extend(step_poses)
        is_carrying.extend([carrying] * len(step_poses))
        stone_placed.extend([placed] * len(step_poses))
        return new_pose

    # --- placed: the block reaches its final voxel here, still gripped ---
    pose = run(lift_or_lower_foot(pose, carrying_foot, target_w), True, True)
    # --- release (C2): carrying becomes False starting with the next pose ---
    pose = run(lift_or_lower_foot(pose, carrying_foot, target_w + 1), False, True)
    target_v = get_coordinate(pose, carrying_foot, AXIS_V) + v_shift
    pose = run(translate_foot(
        pose, carrying_foot, AXIS_V, target_v, max_v_stride,
        adjust_step=land_without_collision_v), False, True)
    # the just-released foot can still be resting where exactly one of
    # (u, v) is odd (the red/convex, with-load-only foothold it needed
    # while carrying - see _land_on_odd_u in behaviors.py). EITHER foot
    # needs a no-load/both-even white foothold once it is not carrying -
    # correct whichever of u or v is currently odd now, WHILE STILL
    # ELEVATED (target_w + 1), so it never touches down at target_w on
    # an invalid no-load foothold, even for a single frame.
    corrected_u, corrected_v = _white_correction_targets(
        pose, carrying_foot, max_u_stride, max_v_stride, occupied=occupied)
    if corrected_u != get_coordinate(pose, carrying_foot, AXIS_U):
        pose = run(translate_foot(
            pose, carrying_foot, AXIS_U, corrected_u, max_u_stride), False, True)
    if corrected_v != get_coordinate(pose, carrying_foot, AXIS_V):
        pose = run(translate_foot(
            pose, carrying_foot, AXIS_V, corrected_v, max_v_stride), False, True)
    pose = run(lift_or_lower_foot(pose, carrying_foot, target_w), False, True)

    return poses, is_carrying, stone_placed


def return_home(
        pose, home_pose=START_POSE, carrying_foot=FOOT_RIGHT,
        occupied=None, transit_v=RETURN_TRANSIT_V):
    """D3 - Return Home.

    First settles BOTH feet down to the no-load support height
    (SUPPORT_W) - after placement, one or both feet can be resting at a
    with-load or placement height (e.g. still at the block's placement
    height, or at the with-load support height from the transport leg),
    and move_forward_no_load only lowers a foot as a side effect of
    moving it, so a foot that happens not to need any u/v movement on
    the way home would otherwise stay stranded at that elevated height
    for the whole trip instead of standing normally.

    Then travels home in TWO legs, not one: first to ``transit_v`` (both
    feet, default v=2), then from there to home_pose. The construction
    site's loose-stone material station sits at v=0 for most builds -
    routing the whole return trip straight down v=0 walks the robot
    right back through it (user-confirmed: unnecessary conflicts with
    the material station). Getting to a safe v away from it first, then
    approaching home_pose from there, avoids that lane for all but the
    final short v segment. Leads with the foot that was NOT carrying the
    block (matches the reference implementation, which returns leading
    with the opposite foot from the outbound trip).

    ``occupied`` (optional): a set of (u, v) columns already built on
    (see complete_assembly_cycle) - neither foot will ever land there on
    the way home; this is what stops the return trip from trying to
    plant a foot back inside the column it just placed a stone in.

    Composition: reversed B1 + B3 + B5 + B7.

    Returns (poses, is_carrying); is_carrying is False for every pose -
    the block was already released before returning.
    """
    lead_foot = FOOT_LEFT if carrying_foot == FOOT_RIGHT else FOOT_RIGHT
    other_foot = FOOT_RIGHT if lead_foot == FOOT_LEFT else FOOT_LEFT

    poses = [pose]
    for foot in (lead_foot, other_foot):
        # The foot that was just carrying can still be resting where
        # exactly one of (u, v) is odd (the red/convex, with-load-only
        # foothold - see _land_on_odd_u in behaviors.py). Settling it
        # straight down to the no-load support height here, before its
        # u/v is corrected, would produce exactly the invalid "resting
        # on the red pyramid without a load" pose. move_forward_no_load
        # below already corrects BOTH feet's u/v parity as its very
        # first action (_correct_both_feet_to_white) and settles it
        # properly as part of that - so skip the premature settle here
        # for whichever foot is currently invalid and let it handle that
        # foot instead.
        if (get_coordinate(pose, foot, AXIS_U) % 2 != 0
                or get_coordinate(pose, foot, AXIS_V) % 2 != 0):
            continue
        settle_poses, pose = lift_or_lower_foot(pose, foot, SUPPORT_W)
        poses.extend(settle_poses)

    via_pose = home_pose._replace(v_left=transit_v, v_right=transit_v)
    leg_poses = move_forward_no_load(
        pose, via_pose, lead_foot=lead_foot, occupied=occupied)
    poses.extend(leg_poses[1:])
    pose = leg_poses[-1]

    trip_poses = move_forward_no_load(
        pose, home_pose, lead_foot=lead_foot, occupied=occupied)
    poses.extend(trip_poses[1:])

    return poses, [False] * len(poses)


def _ensure_min_clearance(u, v, other_u, other_v, min_distance=3):
    """Nudges (u, v) further away from (other_u, other_v) along u only,
    until their Manhattan distance is at least min_distance. A placed
    stone physically blocks its own cell plus its immediate left/right/
    front/back neighbors (user-confirmed against the Rhino model - the
    same reasoning behind placement_sequence's v_shift), so the free
    foot's own standing target must not be given a spot too close to
    where the carrying foot will place the stone - the robot cannot
    stand on top of, or immediately next to, the block it just placed.
    Only nudges u (never v) - simplest fix that does not disturb the
    caller's intended v. If already far enough, returns (u, v)
    unchanged.
    """
    distance = abs(u - other_u) + abs(v - other_v)
    if distance >= min_distance:
        return u, v
    direction = 1 if u >= other_u else -1
    return other_u + direction * min_distance, v


def complete_assembly_cycle(
        start_pose, stone_u, stone_v, stone_w, target_pose,
        home_pose=START_POSE, carrying_foot=FOOT_RIGHT, occupied=None):
    """Full pickup -> transport -> place -> return cycle (D1, B2, D2, D3
    chained together), matching the paper's description: "approaching,
    gripping, transporting, placing, and returning".

    ``target_pose`` gives the final voxel position for BOTH feet once
    the block has been placed and the carrying foot has backed off by
    one v unit (see placement_sequence) - i.e. it is the pose the robot
    should be standing in right after placing, not the block's own
    position (that is stone_u/stone_v/stone_w for pickup, and the
    carrying foot's pre-backoff w for placement is read from
    target_pose's own w for that foot).

    The free foot's own target (u_left/v_left when carrying_foot is the
    right foot, or vice versa) is corrected automatically if it was
    given too close to where the block gets placed (see
    _ensure_min_clearance) - the free foot arrives at this target during
    the transport leg, BEFORE the block is placed there, so it would
    otherwise end up standing right where the block (or its blocked
    neighboring cells) needs to go.

    ``occupied`` (optional): a set of (u, v) columns already built on by
    EARLIER calls to this function (this cycle's own stone is added to
    it automatically, internally, before the return trip - the caller
    does not need to include it). Pass the ``occupied`` returned by the
    previous call back in on the next one to keep the whole build's
    footprint - across every stone placed so far, not just this one -
    considered by every later pickup/transport/return leg. Without this,
    each cycle only knows about the ONE stone it places itself, and can
    walk straight back through columns built on by earlier cycles.

    Returns (poses, is_carrying, stone_placed, occupied) - the first
    three are equal-length lists covering the whole cycle, start to
    home; the fourth is the updated occupied set (previous ``occupied``
    plus this cycle's own placement), to feed into the next call. Wire
    is_carrying to a Boolean output to drive a "show/hide the block
    attached to the foot" toggle, and stone_placed to a separate "show
    the block resting at its final voxel" toggle - see placement_sequence
    for why these are not the same signal (is_carrying goes False right
    after release; stone_placed goes True earlier, the moment the block
    reaches target_w, and then stays True for the rest of the cycle,
    including the whole trip home).
    """
    occupied = set(occupied) if occupied else set()

    if carrying_foot == FOOT_RIGHT:
        corrected_u, corrected_v = _ensure_min_clearance(
            target_pose.u_left, target_pose.v_left,
            target_pose.u_right, target_pose.v_right)
        target_pose = target_pose._replace(u_left=corrected_u, v_left=corrected_v)
    else:
        corrected_u, corrected_v = _ensure_min_clearance(
            target_pose.u_right, target_pose.v_right,
            target_pose.u_left, target_pose.v_left)
        target_pose = target_pose._replace(u_right=corrected_u, v_right=corrected_v)

    poses, is_carrying = pickup_sequence(
        start_pose, stone_u, stone_v, stone_w, carrying_foot, occupied=occupied)
    stone_placed = [False] * len(poses)

    carrying_target_w = (
        target_pose.w_right if carrying_foot == FOOT_RIGHT
        else target_pose.w_left)
    carrying_target_u = (
        target_pose.u_right if carrying_foot == FOOT_RIGHT
        else target_pose.u_left)
    carrying_target_v = (
        target_pose.v_right if carrying_foot == FOOT_RIGHT
        else target_pose.v_left)
    transport_target = target_pose._replace(**{
        ("w_right" if carrying_foot == FOOT_RIGHT else "w_left"):
            SUPPORT_W_WITH_LOAD,
    })
    transport_poses = move_forward_with_load(
        poses[-1], transport_target, carrying_foot=carrying_foot, occupied=occupied)
    poses.extend(transport_poses[1:])
    is_carrying.extend([True] * (len(transport_poses) - 1))
    stone_placed.extend([False] * (len(transport_poses) - 1))

    place_poses, place_carrying, place_stone_placed = placement_sequence(
        poses[-1], carrying_target_w, carrying_foot=carrying_foot, occupied=occupied)
    poses.extend(place_poses[1:])
    is_carrying.extend(place_carrying[1:])
    stone_placed.extend(place_stone_placed[1:])

    # The block just placed occupies its column ground-up from here on -
    # every later leg (this cycle's own return trip, and every future
    # cycle's pickup/transport/return) must never plant a foot there.
    occupied = occupied | {(carrying_target_u, carrying_target_v)}

    return_poses, return_carrying = return_home(
        poses[-1], home_pose, carrying_foot, occupied=occupied)
    poses.extend(return_poses[1:])
    is_carrying.extend(return_carrying[1:])
    stone_placed.extend([True] * (len(return_poses) - 1))

    return poses, is_carrying, stone_placed, occupied
