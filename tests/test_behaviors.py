import unittest

from rhombe_motion.simple_motion import START_POSE, Pose, plan_simple_motion
from rhombe_motion.primitives import FOOT_LEFT, FOOT_RIGHT, AXIS_V
from rhombe_motion.behaviors import (
    move_forward_no_load,
    move_forward_with_load,
    SUPPORT_W,
    LIFT_W_WITH_LOAD,
    SUPPORT_W_WITH_LOAD,
    _safe_u_step,
    _safe_v_step,
)


class MoveForwardNoLoadParityTests(unittest.TestCase):
    """B1, built from the new A1/A2 primitives, must reproduce the exact
    same pose-by-pose trajectory as the legacy, already-validated
    simple_motion.plan_simple_motion() for every target legacy can reach
    (i.e. every target that does not require backward movement on an
    axis). This is the proof that refactoring into primitives did not
    change behavior.
    """

    def _assert_matches_legacy(self, target):
        legacy_plan = plan_simple_motion(*target)
        new_poses = move_forward_no_load(START_POSE, Pose(*target))
        self.assertEqual(tuple(new_poses), legacy_plan.poses)

    def test_start_equals_target(self):
        self._assert_matches_legacy((0, 0, 1, 2, 0, 1))

    def test_v_only(self):
        self._assert_matches_legacy((0, 4, 1, 2, 4, 1))

    def test_small_v(self):
        # v_right must be even too, not just u_right (see
        # _land_on_even_v in behaviors.py) - v=1 is not a valid no-load
        # foothold for the right foot, unlike when this test was
        # originally written under the old (u-only) rule.
        self._assert_matches_legacy((0, 2, 1, 2, 2, 1))

    def test_w_increase_only(self):
        self._assert_matches_legacy((0, 0, 3, 2, 0, 4))


class MoveForwardNoLoadImprovementTests(unittest.TestCase):
    """Unlike legacy (which silently ignores any target requiring backward
    movement on an axis and returns just the start pose), the new,
    primitive-based B1 actually performs backward movement. This is an
    intentional improvement, not something legacy parity should hold for.
    """

    def test_handles_backward_target_legacy_could_not(self):
        # u_right=0 is backward from 2, and even (a valid no-load
        # foothold for the right foot). u_left=-2 (not 0, and also
        # moving away from the right foot's path rather than across it -
        # see the "known limitation" test below for why that matters).
        target = Pose(-2, 0, 1, 0, 0, 1)
        legacy_plan = plan_simple_motion(*target)
        self.assertFalse(legacy_plan.reached)  # legacy gives up

        new_poses = move_forward_no_load(START_POSE, target)
        self.assertEqual(new_poses[-1], target)  # new engine gets there

    def test_crossing_paths_resolved_by_detouring_around(self):
        # A plain straight-line stepping loop would deadlock here: left
        # needs to advance past right's current u=2, and right needs to
        # retreat past where left is heading, so neither can move without
        # crossing the other on a straight path (a relative robot's legs
        # can never cross). _step_around_obstacle (A3, "Rotate Support
        # Foot") resolves this by swinging the blocked foot out to a
        # different v lane, advancing past the obstacle there, then
        # swinging back - the two feet's (u, v) must still never coincide
        # at any point, even during the detour.
        target = Pose(4, 0, 1, 0, 0, 1)  # left must pass right's u=2
        poses = move_forward_no_load(START_POSE, target)
        self.assertEqual(poses[-1], target)
        for p in poses:
            self.assertFalse(
                (p.u_left, p.v_left) == (p.u_right, p.v_right), msg=repr(p))

    def test_multi_stride_moves_stay_elevated_between_strides(self):
        # regression/improvement test: these three targets used to be
        # exact-parity tests against legacy, but legacy touches a foot
        # down to SUPPORT_W and immediately lifts it back up between
        # every single max_stride-sized stride even when the same foot
        # needs to keep going the same direction right away (e.g. u=2
        # needs two strides of 2 to reach u=6) - physically wasteful
        # (user-confirmed: the 5th motor already handles a continuous
        # pivot/slide, the foot only needs to touch down once at the
        # very end). move_forward_no_load now stays elevated across
        # consecutive same-direction strides for a foot and only lowers
        # once that axis's target is truly reached - still reaches the
        # exact target, just with fewer touch-downs than legacy.
        for target in (
                Pose(4, 0, 1, 6, 0, 1),      # test_u_only (was)
                Pose(8, 8, 1, 10, 8, 1),     # test_combined_u_v_w (was)
                Pose(20, 16, 1, 22, 16, 1),  # test_distant_target (was)
        ):
            legacy_plan = plan_simple_motion(*target)
            new_poses = move_forward_no_load(START_POSE, target)
            self.assertEqual(new_poses[-1], target, msg=repr(target))

            def touch_downs(poses):
                return sum(
                    1 for prev, cur in zip(poses, poses[1:])
                    if prev.w_left != SUPPORT_W and cur.w_left == SUPPORT_W
                    or prev.w_right != SUPPORT_W and cur.w_right == SUPPORT_W)

            self.assertLessEqual(
                touch_downs(new_poses), touch_downs(legacy_plan.poses),
                msg=repr(target))


class MoveForwardNoLoadFootholdParityTests(unittest.TestCase):
    """The right foot must only ever settle (support height) on an even u
    coordinate while not carrying a block - the white/concave footholds.
    A naive capped-stride walk can settle on an odd intermediate u by
    accident when the total distance is odd; this must never happen.
    """

    def test_right_foot_never_settles_on_an_odd_u(self):
        # right foot starts odd (u=3) - the realistic scenario right
        # after a pickup/carry leg, where the reference implementation's
        # odd-landing rule left it there. Walking home (even target,
        # u=8) means an odd total distance (5); a naive capped-stride
        # walk would settle at u=5 (odd) as an intermediate stop.
        # u_left target=4 (not 0): the physical leg-span limit
        # (MAX_AXIS_DISTANCE=4 - user-confirmed) means u_left and
        # u_right may never differ by more than 4, so the left foot must
        # also advance partway rather than staying put while the right
        # foot travels all the way to u=8.
        start = START_POSE._replace(u_right=3)
        target = Pose(4, 0, 1, 8, 0, 1)
        poses = move_forward_no_load(start, target)
        # skip poses[0]: that is the given starting pose itself (already
        # odd before this function did anything), not something the
        # function settled on.
        for p in poses[1:]:
            if p.w_right == SUPPORT_W:
                self.assertEqual(p.u_right % 2, 0, msg=repr(p))
        self.assertEqual(poses[-1], target)

    def test_odd_target_for_right_foot_raises_immediately(self):
        target = Pose(0, 0, 1, 7, 0, 1)
        with self.assertRaises(ValueError):
            move_forward_no_load(START_POSE, target)


class MoveForwardWithLoadTests(unittest.TestCase):
    def test_reaches_target(self):
        # u_right=9 (odd) - a reachable target for the carrying (right)
        # foot. u_left=4 (even, not 5): the free foot needs a valid
        # white/both-even foothold too (user-confirmed - see
        # test_free_foot_also_needs_a_valid_white_foothold), and stays
        # within the physical leg-span limit (MAX_AXIS_DISTANCE=5 -
        # user-confirmed) of u_right=9.
        target = Pose(4, 8, 1, 9, 8, 1)
        poses = move_forward_with_load(START_POSE, target)
        self.assertEqual(poses[-1], target)

    def test_uses_raised_lift_and_support_heights(self):
        target = Pose(0, 4, 1, 2, 4, 1)
        poses = move_forward_with_load(START_POSE, target)
        w_values_seen = {p.w_right for p in poses}
        # 1 (start), 3 (lift-with-load) and 2 (support-with-load) must
        # appear; the no-load values (2 as lift / 1 as support) must not,
        # except w_right passing through 1 only at the very start pose.
        self.assertIn(3, w_values_seen)
        self.assertIn(2, w_values_seen)

    def test_carrying_foot_always_lands_on_odd_u(self):
        target = Pose(0, 0, 1, 9, 0, 1)
        poses = move_forward_with_load(
            START_POSE, target, carrying_foot=FOOT_RIGHT)
        # w_right passes through the support height (2) twice per step:
        # once transiently while still lifting up to 3 (u_right not yet
        # moved), and once for real right after lowering back down from
        # 3. Only the second case - a genuine landing - must be odd.
        for prev, cur in zip(poses, poses[1:]):
            if prev.w_right == LIFT_W_WITH_LOAD and cur.w_right == SUPPORT_W_WITH_LOAD:
                self.assertEqual(cur.u_right % 2, 1, msg=repr(cur))

    def test_free_foot_also_needs_a_valid_white_foothold(self):
        # regression test: an earlier version of this function left the
        # free foot completely unconstrained whenever it was the left
        # foot (the "even" white/both-even requirement was only ever
        # applied when the free foot happened to be the right foot).
        # User-confirmed directly (reported live: uL=9, vL=0 - an odd u
        # for the LEFT foot - was standing on an invalid foothold): the
        # white/concave foothold rule (both u and v even) applies to
        # WHICHEVER foot is not carrying, regardless of left or right.
        # u_right=9 (odd, reachable for the carrying foot); u_left=4
        # (even) must be reachable exactly since the left foot is free.
        target = Pose(4, 0, 1, 9, 0, 1)
        poses = move_forward_with_load(
            START_POSE, target, carrying_foot=FOOT_RIGHT)
        self.assertEqual(poses[-1], target)
        for p in poses:
            if p.w_left == SUPPORT_W:
                self.assertEqual(p.u_left % 2, 0, msg=repr(p))
                self.assertEqual(p.v_left % 2, 0, msg=repr(p))

    def test_free_left_foot_odd_target_raises_immediately(self):
        # mirror of test_odd_target_for_right_foot_raises_immediately in
        # MoveForwardNoLoadFootholdParityTests, but for the free LEFT
        # foot in a with-load call - it needs the same even-u/even-v
        # constraint as the free right foot does.
        target = Pose(3, 0, 1, 9, 0, 1)
        with self.assertRaises(ValueError):
            move_forward_with_load(START_POSE, target, carrying_foot=FOOT_RIGHT)

    def test_left_foot_can_be_the_carrying_foot(self):
        # u_left target 3 (odd, valid) stays behind u_right's target 6 -
        # left never needs to cross right's path (see the "known
        # limitation" test in MoveForwardNoLoadFootholdParityTests).
        target = Pose(3, 0, 1, 6, 0, 1)
        poses = move_forward_with_load(
            START_POSE, target, carrying_foot=FOOT_LEFT)
        for prev, cur in zip(poses, poses[1:]):
            if prev.w_left == LIFT_W_WITH_LOAD and cur.w_left == SUPPORT_W_WITH_LOAD:
                self.assertEqual(cur.u_left % 2, 1, msg=repr(cur))
        self.assertEqual(poses[-1], target)

    def test_even_target_for_carrying_foot_raises_immediately(self):
        # u_right=8 is even - physically unreachable while carrying (the
        # red/convex footholds only exist on odd u). Must fail fast with
        # a clear error instead of looping forever trying to land there.
        target = Pose(0, 0, 1, 8, 0, 1)
        with self.assertRaises(ValueError):
            move_forward_with_load(START_POSE, target, carrying_foot=FOOT_RIGHT)

    def test_free_foot_only_ever_uses_no_load_heights(self):
        # the free (non-carrying) foot must never be lifted to the
        # with-load lift height (3) - it is not resting on the block, so
        # it never needs to clear it. (Its support height, 1, and its
        # own no-load lift height, 2, are unaffected by carrying - note
        # 2 is also numerically SUPPORT_W_WITH_LOAD, so that value alone
        # cannot distinguish the two; only the unique value 3 can.)
        target = Pose(4, 8, 1, 9, 8, 3)
        poses = move_forward_with_load(
            START_POSE, target, carrying_foot=FOOT_RIGHT)
        w_left_values = {p.w_left for p in poses}
        self.assertNotIn(LIFT_W_WITH_LOAD, w_left_values)

    def test_at_least_one_foot_is_always_at_true_ground_contact(self):
        # invariant: at every single pose, either the free foot is at its
        # true ground height (1) or the carrying foot is standing on the
        # block (2) - never both feet elevated above their own contact
        # height at the same time (that is what "the robot floats" meant).
        target = Pose(4, 8, 1, 9, 8, 3)
        poses = move_forward_with_load(
            START_POSE, target, carrying_foot=FOOT_RIGHT)
        for p in poses:
            grounded = (p.w_left == SUPPORT_W) or (p.w_right == SUPPORT_W_WITH_LOAD)
            self.assertTrue(grounded, msg=repr(p))


class OccupiedColumnAvoidanceTests(unittest.TestCase):
    """A placed stone permanently occupies its (u, v) column - no foot
    may ever rest there again (user-confirmed: "keine Fussposition auf
    einem neu platzierten Stein"). _safe_u_step/_safe_v_step accept an
    ``occupied`` set of (u, v) tuples for exactly this - every landing
    search here already treats it exactly like an other-foot collision:
    skip it, fall back to the next-best magnitude.
    """

    def test_u_step_skips_an_occupied_landing(self):
        pose = START_POSE._replace(u_right=0, v_right=0, u_left=0, v_left=-6)
        # without occupied, the full stride (magnitude 4, landing u=4) is
        # taken - it is a valid white/even foothold and far enough from
        # the other foot.
        self.assertEqual(
            _safe_u_step(pose, FOOT_RIGHT, 4, require_parity="even"), 4)
        # marking (4, 0) as already built on must skip that landing -
        # magnitude 3 lands on u=3 (odd, invalid parity) so it falls all
        # the way back to magnitude 2 (u=2, even, unoccupied).
        self.assertEqual(
            _safe_u_step(
                pose, FOOT_RIGHT, 4, require_parity="even",
                occupied={(4, 0)}),
            2)

    def test_v_step_skips_an_occupied_landing(self):
        pose = START_POSE._replace(u_right=0, v_right=0, u_left=-6, v_left=0)
        self.assertEqual(
            _safe_v_step(pose, FOOT_RIGHT, AXIS_V, 4, require_parity="even"), 4)
        self.assertEqual(
            _safe_v_step(
                pose, FOOT_RIGHT, AXIS_V, 4, require_parity="even",
                occupied={(0, 4)}),
            2)

    def test_occupied_none_or_empty_is_a_no_op(self):
        # default (None) and an explicitly empty set must behave
        # identically to never passing occupied at all - every existing
        # caller/test relies on this.
        pose = START_POSE._replace(u_right=0, v_right=0, u_left=0, v_left=-6)
        plain = _safe_u_step(pose, FOOT_RIGHT, 4, require_parity="even")
        self.assertEqual(
            _safe_u_step(pose, FOOT_RIGHT, 4, require_parity="even", occupied=None),
            plain)
        self.assertEqual(
            _safe_u_step(pose, FOOT_RIGHT, 4, require_parity="even", occupied=set()),
            plain)

    def test_move_forward_no_load_routes_around_an_occupied_column(self):
        # end-to-end: a target reachable in a single straight stride, but
        # whose direct landing column is occupied, must still be reached
        # (via the existing detour machinery) without ever settling a
        # foot on the occupied column at any point in the trajectory.
        target = Pose(0, 0, 1, 4, 4, 1)
        poses = move_forward_no_load(START_POSE, target, occupied={(4, 0)})
        self.assertEqual(poses[-1], target)
        for p in poses:
            self.assertFalse(
                (p.u_right, p.v_right) == (4, 0) and p.w_right == SUPPORT_W,
                msg=repr(p))
            self.assertFalse(
                (p.u_left, p.v_left) == (4, 0) and p.w_left == SUPPORT_W,
                msg=repr(p))


if __name__ == "__main__":
    unittest.main()
