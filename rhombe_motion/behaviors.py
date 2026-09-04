# -*- coding: utf-8 -*-
"""Behaviors layer (B1-B8, D1-D3 - Table 1). Fixed compositions of
primitives with no decision-making of their own about WHICH behavior to
run - that belongs to the planner layer (not built yet).

Only B1 (Move Forward, No Load) is implemented so far. It is built
entirely out of rhombe_motion.primitives (A1 lift/lower, A2 translate)
and is verified, in tests/test_behaviors.py, to reproduce the exact same
pose-by-pose trajectory as the legacy rhombe_motion.simple_motion
implementation for every target that legacy can reach - proving the
refactor into primitives did not change behavior. Unlike legacy, this
version also supports targets that require backward movement on one
axis, which the legacy forward-only implementation silently ignored;
that is intentional and covered by a separate test, not a parity
requirement.
"""

from rhombe_motion.primitives import (
    FOOT_LEFT,
    FOOT_RIGHT,
    AXIS_U,
    AXIS_V,
    AXIS_W,
    get_coordinate,
    lift_or_lower_foot,
    translate_foot,
)

SUPPORT_W = 1
LIFT_W_NO_LOAD = 2
SUPPORT_W_WITH_LOAD = 2
LIFT_W_WITH_LOAD = 3
MAX_V_STRIDE = 4
MAX_U_STRIDE = 2
MAX_BEHAVIOR_POSES = 5000

# Physical leg-span limit (user-confirmed: max 32.9cm between the two
# feet -> at most 4 blocks may lie BETWEEN the two feet, not counting
# the two blocks the feet themselves stand on - i.e. the two feet span
# at most 6 block positions total, a coordinate difference of 5). The
# two feet's u, v, and w values may each never differ by more than
# this, independently per axis - |uL-uR| <= 5, |vL-vR| <= 5,
# |wL-wR| <= 5. w is bounded on its own by the fixed height constants
# above (max difference is LIFT_W_WITH_LOAD - SUPPORT_W = 2), so only u
# and v need active enforcement in the stride functions below.
MAX_AXIS_DISTANCE = 5

# Minimum same-lane separation (user-confirmed: the two feet must never
# stand directly next to each other - at least one block must lie
# between them - or their physical leg mechanisms collide). Only
# matters when the two feet share the OTHER axis's coordinate (the same
# "lane" - see _safe_u_step's same_lane check): a foot's landing u may
# never be within this distance of the other foot's u while their v
# also matches (and symmetrically for v/u).
MIN_SAME_LANE_DISTANCE = 2


def _step_foot_along_axis(
        pose, foot, axis, target, max_stride,
        lift_w=LIFT_W_NO_LOAD, support_w=SUPPORT_W, adjust_step=None,
        already_elevated=False, defer_lower=False):
    """One A1 + A2 + A1 cycle: lift a foot, translate it, set it back
    down. This is the basic "one stride" unit every B-level forward/
    sideways/vertical behavior is built from.

    ``already_elevated`` skips the initial lift (the foot is already at
    lift_w, left there on purpose by a previous call's defer_lower=True -
    see below) - without this, a foot taking several consecutive strides
    in the same direction (e.g. swinging around the other, stationary
    foot during a detour) would lower to support_w and immediately lift
    back to lift_w between every single stride, an unnecessary and
    physically wasteful "stutter" (user-confirmed: the 5th motor already
    handles the pivoting motion continuously - the foot only needs to
    touch down once, at the very end of the whole swing).

    ``defer_lower`` skips the final lower - the caller is responsible
    for lowering once the foot truly has nothing left to do, typically
    by checking whether this axis's target was reached and, if not,
    leaving the foot elevated and calling again with
    already_elevated=True for the next stride.
    """
    poses = []

    if not already_elevated:
        lift_poses, pose = lift_or_lower_foot(pose, foot, lift_w)
        poses.extend(lift_poses)

    move_poses, pose = translate_foot(
        pose, foot, axis, target, max_stride, adjust_step)
    poses.extend(move_poses)

    if not defer_lower:
        lower_poses, pose = lift_or_lower_foot(pose, foot, support_w)
        poses.extend(lower_poses)

    return poses, pose


def _other_foot(foot):
    return FOOT_LEFT if foot == FOOT_RIGHT else FOOT_RIGHT


def _safe_u_step(pose, foot, naive_step, require_parity=None, occupied=None):
    """Rhomb-E is a *relative* robot: the two feet can never occupy the
    same (u, v) voxel - not even for a single transient in-between pose
    while a foot is mid-stride, only as a final resting position. A
    stride is therefore never allowed to step the foot's u through the
    other foot's current u WHILE they also share the same v (a different
    v means a different "lane"; passing under/over a u the other foot
    happens to occupy is fine there - see _step_around_obstacle, which
    relies on exactly this to detour around a blocked straight path).

    ``occupied`` (optional): a set of (u, v) columns where a stone has
    already been placed - the foot may never REST there (it physically
    occupies the ground), so any landing whose (u, v) falls in this set
    is skipped, same as an other-foot collision. Passing over an
    occupied column while airborne mid-stride is unaffected - only the
    final landing is checked, exactly like every other landing rule
    here.

    Finds the largest-magnitude step in naive_step's direction (capped
    at abs(naive_step)) such that:
      - if this foot's v currently equals the other foot's v ("same
        lane"), no unit of the path from the foot's current u to the
        landing comes within MIN_SAME_LANE_DISTANCE of the other foot's
        current u (not just avoids landing exactly on it - the two feet
        would physically collide even standing directly next to each
        other, one block apart, in the same lane - user-confirmed), and
      - the landing satisfies require_parity:
          'even' - landing_u itself must be even (the no-load/white
            constraint - see _land_on_even_u for why this checks u
            directly, on its own, rather than the sum with v).
          'odd' - (landing_u + this foot's current v) must be odd (the
            with-load/red constraint - see _land_on_odd_u for why this
            one IS a sum).
          None - no constraint.
      - the landing does not put this foot's u more than
        MAX_AXIS_DISTANCE away from the other foot's current u (the
        robot's physical leg-span limit - user-confirmed).

    If no such step exists (even a single unit would immediately come
    too close to the other foot while sharing its lane, or no
    parity-valid landing fits before it), returns 0 - this foot simply
    cannot move this round on a straight path; the caller then tries a
    detour (see _step_around_obstacle) or gives the other foot a turn
    instead.
    """
    if naive_step == 0:
        return 0

    other_foot = _other_foot(foot)
    current = get_coordinate(pose, foot, AXIS_U)
    v = get_coordinate(pose, foot, AXIS_V)
    direction = 1 if naive_step > 0 else -1
    max_magnitude = abs(naive_step)

    other_v = get_coordinate(pose, other_foot, AXIS_V)
    same_lane = (v == other_v)
    other_u = get_coordinate(pose, other_foot, AXIS_U)

    safe_max = max_magnitude
    if same_lane:
        for i in range(1, max_magnitude + 1):
            if abs(current + i * direction - other_u) < MIN_SAME_LANE_DISTANCE:
                safe_max = i - 1
                break

    for magnitude in range(safe_max, 0, -1):
        landing = current + magnitude * direction
        if abs(landing - other_u) > MAX_AXIS_DISTANCE:
            continue
        # The symmetric case to the same_lane cap above: this move isn't
        # already in the other foot's v-lane, but landing exactly on the
        # other foot's u would put the two feet in the same U column -
        # if v is already within MIN_SAME_LANE_DISTANCE of the other
        # foot's v there too, that is still a collision even though v
        # itself never moved (see the matching check in _safe_v_step,
        # which found the mirror case live: a v-only move landing
        # exactly on the other foot's v while u was merely close, not
        # equal, was missed until this check was added here too).
        if landing == other_u and abs(v - other_v) < MIN_SAME_LANE_DISTANCE:
            continue
        if require_parity == "even" and landing % 2 != 0:
            continue
        if require_parity == "odd" and (landing + v) % 2 == 0:
            continue
        if occupied and (landing, v) in occupied:
            continue
        return magnitude * direction
    return 0


def _land_on_odd_u(pose, foot, axis, naive_step, occupied=None):
    """Landing rule for the load-carrying foot: the physical red/convex
    (outward-pointing) footholds it must step on while carrying a block
    exist wherever exactly one of (u, v) is odd, i.e. (u + v) is odd -
    the RhomBlock lattice alternates concave/convex, so the color
    depends on the SUM of both coordinates, not on u alone (u alone only
    coincidentally matched at v=0, which is where this was first tested;
    confirmed wrong once v != 0). Never crosses the other foot's current
    u, not even transiently (see _safe_u_step). See _land_on_odd_v for
    the matching v-axis rule.
    """
    if axis != AXIS_U:
        return naive_step
    return _safe_u_step(pose, foot, naive_step, require_parity="odd", occupied=occupied)


def _land_on_even_u(pose, foot, axis, naive_step, occupied=None):
    """Landing rule for the right foot while NOT carrying a block: the
    physical white/concave (inward-pointing) footholds it must rest on
    exist ONLY where u AND v are BOTH independently even - confirmed
    directly against the user's Rhino model (u=2, v=1 was found red even
    though u=2 alone is even; (u, v) both odd is not a foothold at all,
    in any state - the RhomBlock module's four sub-cells are: both-even
    = white/no-load, exactly-one-odd = red/with-load, both-odd = no
    stone/no foothold there). This checks u directly (on its own, NOT
    summed with v) because v is independently kept even by the matching
    v-axis rule (see _land_on_even_v) wherever this foot may settle -
    together the two guarantee "both even", which a sum check alone
    cannot (a sum check alone would also accept the invalid both-odd
    case). Never crosses the other foot's current u, not even
    transiently.
    """
    if axis != AXIS_U:
        return naive_step
    return _safe_u_step(pose, foot, naive_step, require_parity="even", occupied=occupied)


def _land_on_odd_v(pose, foot, axis, naive_step, occupied=None):
    """V-axis counterpart of _land_on_odd_u, for when this foot's V-phase
    stride needs to preserve a valid red (with-load) foothold: (landing
    v + this foot's current u) must be odd. Only applies to the v axis.
    """
    if axis != AXIS_V:
        return naive_step
    return _safe_v_step(pose, foot, axis, naive_step, require_parity="odd", occupied=occupied)


def _land_on_even_v(pose, foot, axis, naive_step, occupied=None):
    """V-axis counterpart of _land_on_even_u: the landing v itself must
    be even (checked directly, not summed with u - see _land_on_even_u
    for why). Only applies to the v axis.
    """
    if axis != AXIS_V:
        return naive_step
    return _safe_v_step(pose, foot, axis, naive_step, require_parity="even", occupied=occupied)


def _land_without_collision(pose, foot, axis, naive_step, occupied=None):
    """Default landing rule for a foot with no foothold-parity
    constraint (the left foot, in the current single-carrying-foot
    model): still may never cross the other foot's current u, even
    transiently (see _safe_u_step), and still never lands more than
    MAX_AXIS_DISTANCE away from the other foot's current u. Only the u
    axis matters here (the two feet sharing a v coordinate is not itself
    a problem).
    """
    if axis != AXIS_U:
        return naive_step
    return _safe_u_step(pose, foot, naive_step, require_parity=None, occupied=occupied)


def _land_without_collision_v(pose, foot, axis, naive_step, occupied=None):
    """V-axis counterpart of _land_without_collision - collision- and
    max-distance-safety only, no foothold-parity constraint. Used for
    legs where no color constraint applies: the left foot's v moves, and
    the v "swing-out" leg of _step_around_obstacle (the u-detour), which
    never settles there so only needs to stay collision- and
    distance-safe, not color-valid.
    """
    if axis != AXIS_V:
        return naive_step
    return _safe_v_step(pose, foot, axis, naive_step, require_parity=None, occupied=occupied)


def _naive_u_step(pose, foot, target_u, max_stride):
    current = get_coordinate(pose, foot, AXIS_U)
    remaining = target_u - current
    if remaining == 0:
        return 0
    direction = 1 if remaining > 0 else -1
    return direction * min(abs(remaining), max_stride)


def _u_move_is_blocked(pose, foot, target_u, max_stride, require_parity, occupied=None):
    naive_step = _naive_u_step(pose, foot, target_u, max_stride)
    if naive_step == 0:
        return False
    return _safe_u_step(pose, foot, naive_step, require_parity, occupied=occupied) == 0


def _safe_v_step(pose, foot, axis, naive_step, require_parity=None, occupied=None):
    """Like _safe_u_step, but for a v-axis move: a same-lane collision
    risk only exists if this foot's u already equals the other foot's u
    (otherwise they cannot come close regardless of v). Used both for
    the "return to original v" leg of _step_around_obstacle
    (collision-only, require_parity left as None - the landing there is
    always a previously-valid v being restored, not a new one, so no
    parity check is needed) and for the V-phase of move_forward_no_load
    / move_forward_with_load, where a foot may need to stop at
    intermediate v values while still resting there validly:
      - require_parity='even': the landing v itself must be even (see
        _land_on_even_v / _land_on_even_u for why this is checked
        directly, not as a sum with u).
      - require_parity='odd': (landing_v + this foot's current u) must
        be odd (see _land_on_odd_v / _land_on_odd_u).
      - None: collision-only, as before.
    Also never lands more than MAX_AXIS_DISTANCE away from the other
    foot's current v, and - while sharing the other foot's u (the same
    "lane") - never within MIN_SAME_LANE_DISTANCE of the other foot's v
    (the two feet would physically collide standing right next to each
    other - user-confirmed).
    """
    if naive_step == 0:
        return 0
    other_foot = _other_foot(foot)
    u = get_coordinate(pose, foot, AXIS_U)
    other_u = get_coordinate(pose, other_foot, AXIS_U)
    same_u = (u == other_u)
    other_v = get_coordinate(pose, other_foot, AXIS_V)

    current_v = get_coordinate(pose, foot, AXIS_V)
    direction = 1 if naive_step > 0 else -1
    max_magnitude = abs(naive_step)
    safe_max = max_magnitude
    if same_u:
        for i in range(1, max_magnitude + 1):
            if abs(current_v + i * direction - other_v) < MIN_SAME_LANE_DISTANCE:
                safe_max = i - 1
                break

    for magnitude in range(safe_max, 0, -1):
        landing = current_v + magnitude * direction
        if abs(landing - other_v) > MAX_AXIS_DISTANCE:
            continue
        # Symmetric case to the same_u cap above - see the matching
        # comment in _safe_u_step. This move isn't already in the other
        # foot's u-lane, but landing exactly on the other foot's v would
        # put the two feet in the same V row - if u is already within
        # MIN_SAME_LANE_DISTANCE of the other foot's u there too, that
        # is still a collision even though u itself never moved (this
        # was the actual observed bug: a v-only move landing exactly on
        # the other foot's v while u was merely close, not equal).
        if landing == other_v and abs(u - other_u) < MIN_SAME_LANE_DISTANCE:
            continue
        if require_parity == "even" and landing % 2 != 0:
            continue
        if require_parity == "odd" and (landing + u) % 2 == 0:
            continue
        if occupied and (u, landing) in occupied:
            continue
        return magnitude * direction
    return 0


def _step_around_obstacle(
        pose, foot, target_u, max_u_stride, max_v_stride,
        lift_w, support_w, adjust_u_step=None, adjust_v_back_step=None,
        already_elevated=False, defer_lower=False, occupied=None):
    """A3 - Rotate Support Foot, used as a detour when a straight u move
    is fully blocked because the other foot occupies the same v (see
    _safe_u_step): a relative robot cannot let its legs cross, but it
    also cannot just wait forever if the other foot has no reason to
    move out of the way either. Instead this foot swings out to a
    different v "lane" first, advances along u past whatever was
    blocking it (safe now, since it no longer shares the other foot's
    v), then swings back to its original v - going around instead of
    through, exactly like a person stepping around something in their
    path rather than into it.

    ``adjust_v_back_step`` controls the swing-back leg specifically: it
    defaults to plain collision-safety (_safe_v_step, no parity), which
    is correct when returning to original_v can never itself be an
    invalid foothold color. But the swing-back can be BLOCKED by the
    other foot occupying that exact spot by the time it runs (the other
    foot can move in between) - if so, and the foot needs a v-parity
    guarantee (see _land_on_even_v / _land_on_odd_v), pass one of those
    here so the swing-back lands on the nearest still-valid v instead of
    stopping short at an invalid one.

    ``already_elevated`` / ``defer_lower`` mirror _step_foot_along_axis:
    when a single detour cycle does not cover the whole rotation (the
    target is further than one max_u_stride away), the caller chains
    several detour calls in a row for the same foot - without these, the
    foot would touch down and lift back off between every single one
    (user-confirmed: physically wasteful, since the pivoting motor
    handles the whole rotation continuously; the foot only needs to
    touch down once at the very end of the entire swing, not after
    every detour cycle along the way).

    Returns (intermediate_poses, new_pose).
    """
    original_v = get_coordinate(pose, foot, AXIS_V)
    other_v = get_coordinate(pose, _other_foot(foot), AXIS_V)
    detour_v = original_v + max_v_stride
    if detour_v == other_v:
        detour_v = original_v - max_v_stride

    poses = []

    if not already_elevated:
        lift_poses, pose = lift_or_lower_foot(pose, foot, lift_w)
        poses.extend(lift_poses)

    v_out_poses, pose = translate_foot(
        pose, foot, AXIS_V, detour_v, max_v_stride,
        adjust_step=lambda p, f, a, s: _land_without_collision_v(p, f, a, s, occupied=occupied))
    poses.extend(v_out_poses)

    u_poses, pose = translate_foot(
        pose, foot, AXIS_U, target_u, max_u_stride, adjust_u_step)
    poses.extend(u_poses)

    if adjust_v_back_step is not None:
        v_back_adjust = adjust_v_back_step
    else:
        v_back_adjust = lambda p, f, a, s: _safe_v_step(p, f, a, s, occupied=occupied)
    v_back_poses, pose = translate_foot(
        pose, foot, AXIS_V, original_v, max_v_stride, adjust_step=v_back_adjust)
    poses.extend(v_back_poses)

    if not defer_lower:
        lower_poses, pose = lift_or_lower_foot(pose, foot, support_w)
        poses.extend(lower_poses)

    return poses, pose


def _naive_v_step(pose, foot, target_v, max_stride):
    current = get_coordinate(pose, foot, AXIS_V)
    remaining = target_v - current
    if remaining == 0:
        return 0
    direction = 1 if remaining > 0 else -1
    return direction * min(abs(remaining), max_stride)


def _v_move_is_blocked(pose, foot, target_v, max_stride, require_parity, occupied=None):
    """V-axis counterpart of _u_move_is_blocked. A straight v move can be
    blocked either by a collision (see _safe_v_step) or - since a
    no-load/white landing must have an even v, and a with-load/red
    landing must satisfy an odd u+v sum - by there being no parity-valid
    landing within reach either, e.g. the only collision-free magnitude
    left lands on the wrong parity (this is a real, observed deadlock:
    the last unit of an approach to a v the other foot already occupies
    at the same u can be blocked by collision, while the next-shortest
    safe magnitude lands on the wrong color).
    """
    naive_step = _naive_v_step(pose, foot, target_v, max_stride)
    if naive_step == 0:
        return False
    return _safe_v_step(pose, foot, AXIS_V, naive_step, require_parity, occupied=occupied) == 0


def _step_around_obstacle_on_v(
        pose, foot, target_v, max_u_stride, max_v_stride,
        lift_w, support_w, adjust_v_step=None, adjust_u_back_step=None,
        already_elevated=False, defer_lower=False, occupied=None):
    """V-axis counterpart of _step_around_obstacle (A3): used when a
    straight v move is fully blocked (see _v_move_is_blocked) - this foot
    swings out to a different u "lane" first, advances along v past
    whatever was blocking it (safe now, since it no longer shares the
    other foot's u), then swings back to its original u.

    ``adjust_u_back_step`` controls the swing-back leg specifically: it
    defaults to plain collision-safety (_land_without_collision, no
    parity). But the swing-back can be BLOCKED by the other foot
    occupying that exact spot by the time it runs - if so, and the foot
    needs a u-parity guarantee (see _land_on_even_u / _land_on_odd_u),
    pass one of those here so the swing-back lands on the nearest still-
    valid u instead of stopping short at an invalid one (this is exactly
    what was observed: the swing-back got stuck 1 unit short of
    original_u, at an odd/invalid u, because the other foot had moved
    into original_u's exact spot in the meantime).

    ``already_elevated`` / ``defer_lower`` mirror _step_foot_along_axis -
    see the matching comment in _step_around_obstacle for why chained
    detour cycles for the same foot should stay elevated between them.

    Returns (intermediate_poses, new_pose).
    """
    original_u = get_coordinate(pose, foot, AXIS_U)
    other_u = get_coordinate(pose, _other_foot(foot), AXIS_U)
    detour_u = original_u + max_u_stride
    if detour_u == other_u:
        detour_u = original_u - max_u_stride

    poses = []

    if not already_elevated:
        lift_poses, pose = lift_or_lower_foot(pose, foot, lift_w)
        poses.extend(lift_poses)

    u_out_poses, pose = translate_foot(
        pose, foot, AXIS_U, detour_u, max_u_stride,
        adjust_step=lambda p, f, a, s: _land_without_collision(p, f, a, s, occupied=occupied))
    poses.extend(u_out_poses)

    v_poses, pose = translate_foot(
        pose, foot, AXIS_V, target_v, max_v_stride, adjust_v_step)
    poses.extend(v_poses)

    if adjust_u_back_step is not None:
        u_back_adjust = adjust_u_back_step
    else:
        u_back_adjust = lambda p, f, a, s: _land_without_collision(p, f, a, s, occupied=occupied)
    u_back_poses, pose = translate_foot(
        pose, foot, AXIS_U, original_u, max_u_stride, adjust_step=u_back_adjust)
    poses.extend(u_back_poses)

    if not defer_lower:
        lower_poses, pose = lift_or_lower_foot(pose, foot, support_w)
        poses.extend(lower_poses)

    return poses, pose


def _nearest_valid_step(safe_step_fn, max_stride):
    """Tries the smallest possible 1-unit nudge in either direction
    first (parity always flips after exactly 1 unit, so this almost
    always succeeds immediately), only falling back to a full
    max_stride-sized step - in whichever direction is available - if a
    1-unit correction is blocked by the other foot in both directions.
    Prefers the smallest disruption to the foot's position, unlike
    ordinary travel (_safe_u_step / _safe_v_step's normal "largest safe
    stride" behavior), which would be right for covering distance but
    would needlessly overshoot for a same-spot parity correction. Works
    for either a white (even) or red (odd) correction - ``safe_step_fn``
    is a closure that already has the desired require_parity baked in.
    Returns the signed step to take, or 0 if no safe correction exists
    in either direction at any tried magnitude.
    """
    for magnitude in (1, max_stride):
        for direction in (1, -1):
            step = safe_step_fn(direction * magnitude)
            if step != 0:
                return step
        if magnitude == max_stride:
            break
    return 0


def _white_correction_targets(pose, foot, max_u_stride, max_v_stride, occupied=None):
    """A valid carrying (red) stance has exactly one of (u, v) odd (see
    _land_on_odd_u). Once a foot is released from carrying (or is simply
    the free foot throughout, in move_forward_no_load where NEITHER
    foot carries), it needs BOTH u and v to become even (white/no-load -
    see _land_on_even_u for why a sum check is not enough), which means
    correcting whichever ONE of the two is currently odd - it could be
    either, depending on which red sub-case the foot was left in.
    Returns (corrected_u, corrected_v): each is the nearest safe even
    value on that axis, or the current (already valid, or
    uncorrectable) value unchanged if that axis is already even.
    """
    u = get_coordinate(pose, foot, AXIS_U)
    corrected_u = u
    if u % 2 != 0:
        step = _nearest_valid_step(
            lambda s: _safe_u_step(pose, foot, s, require_parity="even", occupied=occupied),
            max_u_stride)
        corrected_u = u + step

    v = get_coordinate(pose, foot, AXIS_V)
    corrected_v = v
    if v % 2 != 0:
        step = _nearest_valid_step(
            lambda s: _safe_v_step(pose, foot, AXIS_V, s, require_parity="even", occupied=occupied),
            max_v_stride)
        corrected_v = v + step

    return corrected_u, corrected_v


def _red_correction_targets(pose, carrying_foot, max_u_stride, max_v_stride, occupied=None):
    """A valid carrying (red) stance needs exactly one of (u, v) odd
    (see _land_on_odd_u) - if the carrying foot is currently at an
    invalid stance (both even, or - should it ever happen - both odd),
    nudge ONE axis by the smallest safe amount to make the sum odd.
    Needed because move_forward_with_load can be entered from a stance
    that was never validated as a carrying stance (in real usage this
    should already be valid, coming out of pickup_sequence, but this
    makes the behavior robust regardless of caller assumptions - see the
    matching _white_correction_targets for the no-load case). Returns
    (corrected_u, corrected_v): only one of the two actually changes
    (nudging u is preferred; v is only nudged if u could not be), or
    both are returned unchanged if already valid.
    """
    u = get_coordinate(pose, carrying_foot, AXIS_U)
    v = get_coordinate(pose, carrying_foot, AXIS_V)
    if (u + v) % 2 == 1:
        return u, v

    step = _nearest_valid_step(
        lambda s: _safe_u_step(pose, carrying_foot, s, require_parity="odd", occupied=occupied),
        max_u_stride)
    if step != 0:
        return u + step, v

    step = _nearest_valid_step(
        lambda s: _safe_v_step(pose, carrying_foot, AXIS_V, s, require_parity="odd", occupied=occupied),
        max_v_stride)
    return u, v + step


def _correct_foot_to_white(pose, foot, max_u_stride, max_v_stride, occupied=None):
    """Performs the correction from _white_correction_targets as full
    A1+A2+A1 strides (lift, translate, lower to the ordinary no-load
    support height) - for callers that are settling this foot down to
    its normal standing height anyway. Left uncorrected, every V-phase
    or U-phase stride afterwards would keep re-landing the foot on the
    same invalid foothold (a stride along one axis never touches the
    other), reproducing the invalid stance on every single step -
    exactly the "resting on the red pyramid without a stone" bug. Do
    this immediately, before any V-phase or U-phase movement can begin.
    Applies to EITHER foot: white/no-load footholds require both u and v
    even regardless of which foot is standing on them.
    """
    poses = []
    corrected_u, corrected_v = _white_correction_targets(
        pose, foot, max_u_stride, max_v_stride, occupied=occupied)

    current_u = get_coordinate(pose, foot, AXIS_U)
    if corrected_u != current_u:
        step_poses, pose = _step_foot_along_axis(
            pose, foot, AXIS_U, corrected_u, max_u_stride)
        poses.extend(step_poses)

    current_v = get_coordinate(pose, foot, AXIS_V)
    if corrected_v != current_v:
        step_poses, pose = _step_foot_along_axis(
            pose, foot, AXIS_V, corrected_v, max_v_stride)
        poses.extend(step_poses)

    return poses, pose


def _correct_both_feet_to_white(pose, max_u_stride, max_v_stride, occupied=None):
    """Applies _correct_foot_to_white to both feet in turn - used in
    move_forward_no_load, where NEITHER foot carries a block, so both
    must always be on a valid white (both-even) foothold.
    """
    poses = []
    for foot in (FOOT_LEFT, FOOT_RIGHT):
        step_poses, pose = _correct_foot_to_white(
            pose, foot, max_u_stride, max_v_stride, occupied=occupied)
        poses.extend(step_poses)
    return poses, pose


def _correct_settled_feet_to_white(pose, max_u_stride, max_v_stride, occupied=None):
    """Like _correct_both_feet_to_white, but only touches a foot that is
    currently actually settled (at SUPPORT_W). A foot deliberately left
    elevated mid-swing (see _step_foot_along_axis's already_elevated /
    defer_lower - used so a foot taking several consecutive strides in
    a row, e.g. swinging around the other stationary foot, does not
    touch down and lift back off between every single stride) is still
    in transit, not yet at its final position for this phase - checking
    or correcting its (u, v) now would be meaningless (it is not
    resting there) and could fight the movement still in progress. It
    will be checked once it actually settles.
    """
    poses = []
    for foot in (FOOT_LEFT, FOOT_RIGHT):
        if get_coordinate(pose, foot, AXIS_W) != SUPPORT_W:
            continue
        step_poses, pose = _correct_foot_to_white(
            pose, foot, max_u_stride, max_v_stride, occupied=occupied)
        poses.extend(step_poses)
    return poses, pose


def _correct_carrying_foot_to_red(pose, carrying_foot, max_u_stride, max_v_stride, occupied=None):
    """Performs the correction from _red_correction_targets as a full
    A1+A2+A1 stride (lift, translate, lower to the with-load support
    height) - the carrying-foot counterpart of _correct_foot_to_white.
    See _red_correction_targets for why this is needed.
    """
    corrected_u, corrected_v = _red_correction_targets(
        pose, carrying_foot, max_u_stride, max_v_stride, occupied=occupied)

    current_u = get_coordinate(pose, carrying_foot, AXIS_U)
    if corrected_u != current_u:
        return _step_foot_along_axis(
            pose, carrying_foot, AXIS_U, corrected_u, max_u_stride,
            lift_w=LIFT_W_WITH_LOAD, support_w=SUPPORT_W_WITH_LOAD)

    current_v = get_coordinate(pose, carrying_foot, AXIS_V)
    if corrected_v != current_v:
        return _step_foot_along_axis(
            pose, carrying_foot, AXIS_V, corrected_v, max_v_stride,
            lift_w=LIFT_W_WITH_LOAD, support_w=SUPPORT_W_WITH_LOAD)

    return [], pose


def move_forward_no_load(
        pose, target_pose, lead_foot=FOOT_RIGHT,
        max_v_stride=MAX_V_STRIDE, max_u_stride=MAX_U_STRIDE, occupied=None):
    """B1 - Move Forward (No Load).

    ``occupied`` (optional): a set of (u, v) columns where a stone has
    already been placed (see _safe_u_step) - neither foot will ever land
    (rest) there; the stride search picks the nearest other valid
    landing instead, exactly as it already does for the other foot.

    Moves both feet from ``pose`` to ``target_pose``: first all of V
    (lead foot first, then the other, alternating one stride at a time),
    then all of U the same way, then W settles directly to target (lead
    foot first, then the other). Returns the full list of poses
    including the starting pose.

    ``lead_foot`` defaults to the right foot, matching the legacy
    implementation this behavior is proven identical to (see
    tests/test_behaviors.py). The reference return-trip sequence leads
    with the left foot instead - pass FOOT_LEFT to reproduce that.

    BOTH feet's u AND v landings are kept even (the no-load/white-
    foothold constraint - see _land_on_even_u and _land_on_even_v):
    neither foot carries a block in this behavior, so neither is exempt
    - a white/concave foothold requires both u and v even regardless of
    which foot stands on it (user-confirmed directly: e.g. u=9, v=0 for
    the LEFT foot is just as invalid as it would be for the right foot).

    Raises ValueError up front if either foot actually needs to move on
    u (or v) and its target u (or v) is odd - the even-landing
    constraint makes that target physically unreachable, so this fails
    fast instead of looping forever trying to land on it.
    """
    for foot_name, foot, target_u, target_v in (
            ("left", FOOT_LEFT, target_pose.u_left, target_pose.v_left),
            ("right", FOOT_RIGHT, target_pose.u_right, target_pose.v_right)):
        if get_coordinate(pose, foot, AXIS_U) != target_u and target_u % 2 != 0:
            raise ValueError(
                "{0} foot target u={1} is odd - the white/concave "
                "footholds required while not carrying a block only "
                "exist where u is even, so this target is "
                "unreachable".format(foot_name, target_u))
        if get_coordinate(pose, foot, AXIS_V) != target_v and target_v % 2 != 0:
            raise ValueError(
                "{0} foot target v={1} is odd - the white/concave "
                "footholds required while not carrying a block only "
                "exist where v is even, so this target is "
                "unreachable".format(foot_name, target_v))

    other_foot = FOOT_LEFT if lead_foot == FOOT_RIGHT else FOOT_RIGHT
    poses = [pose]

    def land_on_even_v(p, f, a, s):
        return _land_on_even_v(p, f, a, s, occupied=occupied)

    def land_on_even_u(p, f, a, s):
        return _land_on_even_u(p, f, a, s, occupied=occupied)

    correction_poses, pose = _correct_both_feet_to_white(
        pose, max_u_stride, max_v_stride, occupied=occupied)
    poses.extend(correction_poses)

    while ((pose.v_left != target_pose.v_left or pose.v_right != target_pose.v_right)
           and len(poses) < MAX_BEHAVIOR_POSES):
        moved = False
        for foot in (lead_foot, other_foot):
            axis_target = (target_pose.v_right if foot == FOOT_RIGHT
                            else target_pose.v_left)
            if get_coordinate(pose, foot, AXIS_V) == axis_target:
                continue
            # Stay elevated across consecutive strides/detour cycles for
            # the same foot (e.g. swinging around the other, stationary
            # foot) instead of touching down and lifting back off
            # between every single one - only lower once this axis's
            # target is actually reached (user-confirmed: the pivoting
            # motor already handles the whole rotation continuously, the
            # foot only needs to touch down once at the very end). Only
            # safe when the OTHER foot has already finished ITS move for
            # this phase (won't need a turn of its own) - both feet must
            # never be airborne at the same pose, so if the other foot
            # still has moving left to do, this one settles fully every
            # round instead, preserving the normal one-stride-at-a-time
            # alternation between them.
            other = _other_foot(foot)
            other_axis_target = target_pose.v_right if other == FOOT_RIGHT else target_pose.v_left
            other_done = get_coordinate(pose, other, AXIS_V) == other_axis_target
            already_elevated = get_coordinate(pose, foot, AXIS_W) == LIFT_W_NO_LOAD
            if _v_move_is_blocked(pose, foot, axis_target, max_v_stride, "even", occupied=occupied):
                step_poses, pose = _step_around_obstacle_on_v(
                    pose, foot, axis_target, max_u_stride, max_v_stride,
                    LIFT_W_NO_LOAD, SUPPORT_W, adjust_v_step=land_on_even_v,
                    adjust_u_back_step=land_on_even_u,
                    already_elevated=already_elevated, defer_lower=other_done,
                    occupied=occupied)
            else:
                step_poses, pose = _step_foot_along_axis(
                    pose, foot, AXIS_V, axis_target, max_v_stride,
                    adjust_step=land_on_even_v,
                    already_elevated=already_elevated, defer_lower=other_done)
            poses.extend(step_poses)
            if get_coordinate(pose, foot, AXIS_V) == axis_target:
                lower_poses, pose = lift_or_lower_foot(pose, foot, SUPPORT_W)
                poses.extend(lower_poses)
            moved = True
        if not moved:
            break
        # A v-detour's swing-back-to-original-u leg (see
        # _step_around_obstacle_on_v) can itself be blocked by the other
        # foot occupying that exact u,v by the time the swing-back runs -
        # leaving a foot settled at an invalid odd u for this one round.
        # Correct it after every round, not just once at the very end,
        # so no invalid frame survives even transiently. Only touches
        # feet that are actually settled right now (see
        # _correct_settled_feet_to_white) - a foot deliberately left
        # elevated above is still mid-stride, not a real resting stance
        # to validate yet.
        correction_poses, pose = _correct_settled_feet_to_white(pose, max_u_stride, max_v_stride, occupied=occupied)
        poses.extend(correction_poses)

    # Belt-and-braces: the v-phase above already keeps every v landing
    # even on its own (_land_on_even_v), but if a v move was ever
    # blocked outright (returns 0, foot stays where it started), correct
    # any leftover invalid stance now, before any U-phase logic - which
    # itself assumes a valid starting stance - runs.
    correction_poses, pose = _correct_both_feet_to_white(pose, max_u_stride, max_v_stride, occupied=occupied)
    poses.extend(correction_poses)

    while ((pose.u_left != target_pose.u_left or pose.u_right != target_pose.u_right)
           and len(poses) < MAX_BEHAVIOR_POSES):
        moved = False
        for foot in (lead_foot, other_foot):
            axis_target = (target_pose.u_right if foot == FOOT_RIGHT
                            else target_pose.u_left)
            if get_coordinate(pose, foot, AXIS_U) == axis_target:
                continue
            # See the matching comment in the v-phase above: only stay
            # elevated across rounds while the other foot has nothing
            # left to do this phase.
            other = _other_foot(foot)
            other_axis_target = target_pose.u_right if other == FOOT_RIGHT else target_pose.u_left
            other_done = get_coordinate(pose, other, AXIS_U) == other_axis_target
            already_elevated = get_coordinate(pose, foot, AXIS_W) == LIFT_W_NO_LOAD
            if _u_move_is_blocked(pose, foot, axis_target, max_u_stride, "even", occupied=occupied):
                step_poses, pose = _step_around_obstacle(
                    pose, foot, axis_target, max_u_stride, max_v_stride,
                    LIFT_W_NO_LOAD, SUPPORT_W, adjust_u_step=land_on_even_u,
                    adjust_v_back_step=land_on_even_v,
                    already_elevated=already_elevated, defer_lower=other_done,
                    occupied=occupied)
            else:
                step_poses, pose = _step_foot_along_axis(
                    pose, foot, AXIS_U, axis_target, max_u_stride,
                    adjust_step=land_on_even_u,
                    already_elevated=already_elevated, defer_lower=other_done)
            poses.extend(step_poses)
            if get_coordinate(pose, foot, AXIS_U) == axis_target:
                lower_poses, pose = lift_or_lower_foot(pose, foot, SUPPORT_W)
                poses.extend(lower_poses)
            moved = True
        if not moved:
            break
        correction_poses, pose = _correct_settled_feet_to_white(pose, max_u_stride, max_v_stride, occupied=occupied)
        poses.extend(correction_poses)

    # A _step_around_obstacle detour during the u phase can leave a foot
    # short of fully returning to its original v (see _safe_v_step) if
    # the other foot was in the way at that exact moment. Settle any
    # such leftover v drift now, the same way the v phase above does.
    while ((pose.v_left != target_pose.v_left or pose.v_right != target_pose.v_right)
           and len(poses) < MAX_BEHAVIOR_POSES):
        moved = False
        for foot in (lead_foot, other_foot):
            axis_target = (target_pose.v_right if foot == FOOT_RIGHT
                            else target_pose.v_left)
            if get_coordinate(pose, foot, AXIS_V) == axis_target:
                continue
            other = _other_foot(foot)
            other_axis_target = target_pose.v_right if other == FOOT_RIGHT else target_pose.v_left
            other_done = get_coordinate(pose, other, AXIS_V) == other_axis_target
            already_elevated = get_coordinate(pose, foot, AXIS_W) == LIFT_W_NO_LOAD
            if _v_move_is_blocked(pose, foot, axis_target, max_v_stride, "even", occupied=occupied):
                step_poses, pose = _step_around_obstacle_on_v(
                    pose, foot, axis_target, max_u_stride, max_v_stride,
                    LIFT_W_NO_LOAD, SUPPORT_W, adjust_v_step=land_on_even_v,
                    adjust_u_back_step=land_on_even_u,
                    already_elevated=already_elevated, defer_lower=other_done,
                    occupied=occupied)
            else:
                step_poses, pose = _step_foot_along_axis(
                    pose, foot, AXIS_V, axis_target, max_v_stride,
                    adjust_step=land_on_even_v,
                    already_elevated=already_elevated, defer_lower=other_done)
            poses.extend(step_poses)
            if get_coordinate(pose, foot, AXIS_V) == axis_target:
                lower_poses, pose = lift_or_lower_foot(pose, foot, SUPPORT_W)
                poses.extend(lower_poses)
            moved = True
        if not moved:
            break
        correction_poses, pose = _correct_settled_feet_to_white(pose, max_u_stride, max_v_stride, occupied=occupied)
        poses.extend(correction_poses)

    # Belt-and-braces once more: same reasoning as after the first
    # v-phase above - the cleanup pass's own detours can leave a foot's
    # u invalid for the same reason.
    correction_poses, pose = _correct_both_feet_to_white(pose, max_u_stride, max_v_stride, occupied=occupied)
    poses.extend(correction_poses)

    for foot in (lead_foot, other_foot):
        target_w = target_pose.w_right if foot == FOOT_RIGHT else target_pose.w_left
        w_poses, pose = lift_or_lower_foot(pose, foot, target_w)
        poses.extend(w_poses)

    return poses


def move_forward_with_load(
        pose, target_pose, carrying_foot=FOOT_RIGHT,
        max_v_stride=MAX_V_STRIDE, max_u_stride=MAX_U_STRIDE, occupied=None):
    """B2 - Move Forward (With Load).

    ``occupied`` (optional): a set of (u, v) columns where a stone has
    already been placed (see _safe_u_step) - neither foot will ever land
    (rest) there, including the carrying foot's own red-foothold search.

    Same stepping shape as B1 (V fully, then U fully, alternating feet,
    then W settles), but:
      - ONLY the carrying foot uses the with-load lift/support heights
        (LIFT_W_WITH_LOAD / SUPPORT_W_WITH_LOAD) - it is standing on the
        block (support) or lifted clear of it (lift). The free foot
        keeps using the ordinary no-load heights (LIFT_W_NO_LOAD /
        SUPPORT_W) throughout, exactly as it would without a block,
        because it is not resting on anything different. Bumping BOTH
        feet to the with-load heights - what an earlier version of this
        function did - left the free foot's resting height (2) neither
        truly grounded (1) nor a valid on-block height, which together
        with the carrying foot's own lift height (3) violated the basic
        two-foot-contact invariant every pose must satisfy: at least one
        foot is always at its own true ground-contact height (free foot:
        w=1; carrying foot: w=2, i.e. standing on the block).
      - the carrying foot must land where (u + v) is odd on every U or V
        step, because the red/convex footholds it needs while carrying a
        block only exist there (see _land_on_odd_u / _land_on_odd_v).
        The free foot keeps the same both-even constraint used in
        move_forward_no_load (see _land_on_even_u / _land_on_even_v) -
        white/concave footholds require both u and v even regardless of
        which foot stands on them, so this applies whichever foot is
        free, not only when it happens to be the right foot.

    ``carrying_foot`` defaults to the right foot (matches the reference
    implementation); pass FOOT_LEFT if the left foot is carrying instead.

    Raises ValueError up front if the carrying foot actually needs to
    move on u and its target (u + v) sum is even (unreachable while
    carrying), or if the free foot actually needs to move on u (or v)
    and its target on that axis is odd (unreachable while not carrying)
    - fails fast instead of looping forever trying to land there.
    """
    free_foot = FOOT_LEFT if carrying_foot == FOOT_RIGHT else FOOT_RIGHT

    def target_u_for(foot):
        return target_pose.u_right if foot == FOOT_RIGHT else target_pose.u_left

    def target_v_for(foot):
        return target_pose.v_right if foot == FOOT_RIGHT else target_pose.v_left

    def target_w_for(foot):
        return target_pose.w_right if foot == FOOT_RIGHT else target_pose.w_left

    carrying_u_target = target_u_for(carrying_foot)
    carrying_v_target = target_v_for(carrying_foot)
    if (get_coordinate(pose, carrying_foot, AXIS_U) != carrying_u_target
            and (carrying_u_target + carrying_v_target) % 2 == 0):
        raise ValueError(
            "carrying foot ({0}) target (u={1}, v={2}) has an even u+v "
            "sum - the red/convex footholds required while carrying a "
            "block only exist where u+v is odd, so this target is "
            "unreachable".format(carrying_foot, carrying_u_target, carrying_v_target))
    free_u_target = target_u_for(free_foot)
    free_v_target = target_v_for(free_foot)
    if (get_coordinate(pose, free_foot, AXIS_U) != free_u_target
            and free_u_target % 2 != 0):
        raise ValueError(
            "free foot ({0}) target u={1} is odd - the white/concave "
            "footholds required while not carrying a block only "
            "exist where u is even, so this target is "
            "unreachable".format(free_foot, free_u_target))
    if (get_coordinate(pose, free_foot, AXIS_V) != free_v_target
            and free_v_target % 2 != 0):
        raise ValueError(
            "free foot ({0}) target v={1} is odd - the white/concave "
            "footholds required while not carrying a block only "
            "exist where v is even, so this target is "
            "unreachable".format(free_foot, free_v_target))

    def adjust_for(foot):
        fn = _land_on_odd_u if foot == carrying_foot else _land_on_even_u
        return lambda p, f, a, s, _fn=fn: _fn(p, f, a, s, occupied=occupied)

    def adjust_v_for(foot):
        fn = _land_on_odd_v if foot == carrying_foot else _land_on_even_v
        return lambda p, f, a, s, _fn=fn: _fn(p, f, a, s, occupied=occupied)

    def require_parity_for(foot):
        return "odd" if foot == carrying_foot else "even"

    def lift_w_for(foot):
        return LIFT_W_WITH_LOAD if foot == carrying_foot else LIFT_W_NO_LOAD

    def support_w_for(foot):
        return SUPPORT_W_WITH_LOAD if foot == carrying_foot else SUPPORT_W

    poses = [pose]

    # move_forward_with_load should normally only ever be entered right
    # after pickup_sequence, which already leaves the carrying foot on a
    # valid red stance - but correct it defensively here too (mirroring
    # _correct_foot_to_white for the free foot below), so this behavior
    # is robust even if called directly from a stance that was never
    # validated as a carrying stance.
    correction_poses, pose = _correct_carrying_foot_to_red(
        pose, carrying_foot, max_u_stride, max_v_stride, occupied=occupied)
    poses.extend(correction_poses)
    correction_poses, pose = _correct_foot_to_white(
        pose, free_foot, max_u_stride, max_v_stride, occupied=occupied)
    poses.extend(correction_poses)

    while ((pose.v_left != target_pose.v_left or pose.v_right != target_pose.v_right)
           and len(poses) < MAX_BEHAVIOR_POSES):
        moved = False
        for foot in (FOOT_RIGHT, FOOT_LEFT):
            if get_coordinate(pose, foot, AXIS_V) == target_v_for(foot):
                continue
            # Stay elevated across consecutive strides/detour cycles for
            # the same foot (e.g. swinging around the other, stationary
            # foot) instead of touching down and lifting back off
            # between every single one - only while the other foot has
            # nothing left to do this phase (both feet must never be
            # airborne at once - see the matching comment in
            # move_forward_no_load).
            other = _other_foot(foot)
            other_done = get_coordinate(pose, other, AXIS_V) == target_v_for(other)
            already_elevated = get_coordinate(pose, foot, AXIS_W) == lift_w_for(foot)
            if _v_move_is_blocked(
                    pose, foot, target_v_for(foot), max_v_stride,
                    require_parity_for(foot), occupied=occupied):
                step_poses, pose = _step_around_obstacle_on_v(
                    pose, foot, target_v_for(foot), max_u_stride, max_v_stride,
                    lift_w_for(foot), support_w_for(foot),
                    adjust_v_step=adjust_v_for(foot),
                    adjust_u_back_step=adjust_for(foot),
                    already_elevated=already_elevated, defer_lower=other_done,
                    occupied=occupied)
            else:
                step_poses, pose = _step_foot_along_axis(
                    pose, foot, AXIS_V, target_v_for(foot), max_v_stride,
                    lift_w_for(foot), support_w_for(foot),
                    adjust_step=adjust_v_for(foot),
                    already_elevated=already_elevated, defer_lower=other_done)
            poses.extend(step_poses)
            if get_coordinate(pose, foot, AXIS_V) == target_v_for(foot):
                lower_poses, pose = lift_or_lower_foot(pose, foot, support_w_for(foot))
                poses.extend(lower_poses)
            moved = True
        if not moved:
            break
        # Same reasoning as move_forward_no_load: a v-detour's swing-
        # back-to-original-u leg can itself be blocked by the other foot
        # occupying that exact spot by the time it runs, leaving the
        # free foot at an invalid odd (u, v). Correct it after every
        # round, not just once at the end - but only if it is actually
        # settled right now (see _correct_settled_feet_to_white in
        # move_forward_no_load for why a foot deliberately left elevated
        # mid-swing must not be "corrected" mid-flight).
        if get_coordinate(pose, free_foot, AXIS_W) == SUPPORT_W:
            correction_poses, pose = _correct_foot_to_white(
                pose, free_foot, max_u_stride, max_v_stride, occupied=occupied)
            poses.extend(correction_poses)

    correction_poses, pose = _correct_foot_to_white(
        pose, free_foot, max_u_stride, max_v_stride, occupied=occupied)
    poses.extend(correction_poses)

    while ((pose.u_left != target_pose.u_left or pose.u_right != target_pose.u_right)
           and len(poses) < MAX_BEHAVIOR_POSES):
        moved = False
        for foot in (FOOT_RIGHT, FOOT_LEFT):
            if get_coordinate(pose, foot, AXIS_U) == target_u_for(foot):
                continue
            other = _other_foot(foot)
            other_done = get_coordinate(pose, other, AXIS_U) == target_u_for(other)
            already_elevated = get_coordinate(pose, foot, AXIS_W) == lift_w_for(foot)
            if _u_move_is_blocked(
                    pose, foot, target_u_for(foot), max_u_stride,
                    require_parity_for(foot), occupied=occupied):
                step_poses, pose = _step_around_obstacle(
                    pose, foot, target_u_for(foot), max_u_stride, max_v_stride,
                    lift_w_for(foot), support_w_for(foot),
                    adjust_u_step=adjust_for(foot),
                    adjust_v_back_step=adjust_v_for(foot),
                    already_elevated=already_elevated, defer_lower=other_done,
                    occupied=occupied)
            else:
                step_poses, pose = _step_foot_along_axis(
                    pose, foot, AXIS_U, target_u_for(foot), max_u_stride,
                    lift_w_for(foot), support_w_for(foot), adjust_for(foot),
                    already_elevated=already_elevated, defer_lower=other_done)
            poses.extend(step_poses)
            if get_coordinate(pose, foot, AXIS_U) == target_u_for(foot):
                lower_poses, pose = lift_or_lower_foot(pose, foot, support_w_for(foot))
                poses.extend(lower_poses)
            moved = True
        if not moved:
            break
        if get_coordinate(pose, free_foot, AXIS_W) == SUPPORT_W:
            correction_poses, pose = _correct_foot_to_white(
                pose, free_foot, max_u_stride, max_v_stride, occupied=occupied)
            poses.extend(correction_poses)

    # See the matching comment in move_forward_no_load: a detour during
    # the u phase can leave a foot short of fully returning to its
    # original v. Settle any such leftover drift now.
    while ((pose.v_left != target_pose.v_left or pose.v_right != target_pose.v_right)
           and len(poses) < MAX_BEHAVIOR_POSES):
        moved = False
        for foot in (FOOT_RIGHT, FOOT_LEFT):
            if get_coordinate(pose, foot, AXIS_V) == target_v_for(foot):
                continue
            other = _other_foot(foot)
            other_done = get_coordinate(pose, other, AXIS_V) == target_v_for(other)
            already_elevated = get_coordinate(pose, foot, AXIS_W) == lift_w_for(foot)
            if _v_move_is_blocked(
                    pose, foot, target_v_for(foot), max_v_stride,
                    require_parity_for(foot), occupied=occupied):
                step_poses, pose = _step_around_obstacle_on_v(
                    pose, foot, target_v_for(foot), max_u_stride, max_v_stride,
                    lift_w_for(foot), support_w_for(foot),
                    adjust_v_step=adjust_v_for(foot),
                    adjust_u_back_step=adjust_for(foot),
                    already_elevated=already_elevated, defer_lower=other_done,
                    occupied=occupied)
            else:
                step_poses, pose = _step_foot_along_axis(
                    pose, foot, AXIS_V, target_v_for(foot), max_v_stride,
                    lift_w_for(foot), support_w_for(foot),
                    adjust_step=adjust_v_for(foot),
                    already_elevated=already_elevated, defer_lower=other_done)
            poses.extend(step_poses)
            if get_coordinate(pose, foot, AXIS_V) == target_v_for(foot):
                lower_poses, pose = lift_or_lower_foot(pose, foot, support_w_for(foot))
                poses.extend(lower_poses)
            moved = True
        if not moved:
            break
        if get_coordinate(pose, free_foot, AXIS_W) == SUPPORT_W:
            correction_poses, pose = _correct_foot_to_white(
                pose, free_foot, max_u_stride, max_v_stride, occupied=occupied)
            poses.extend(correction_poses)

    correction_poses, pose = _correct_foot_to_white(
        pose, free_foot, max_u_stride, max_v_stride, occupied=occupied)
    poses.extend(correction_poses)

    for foot in (FOOT_RIGHT, FOOT_LEFT):
        w_poses, pose = lift_or_lower_foot(pose, foot, target_w_for(foot))
        poses.extend(w_poses)

    return poses
