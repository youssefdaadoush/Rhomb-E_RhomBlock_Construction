import unittest

from rhombe_motion.simple_motion import START_POSE, Pose
from rhombe_motion.primitives import FOOT_LEFT, FOOT_RIGHT
from rhombe_motion.behaviors import SUPPORT_W, SUPPORT_W_WITH_LOAD
from rhombe_motion.work_sequences import (
    pickup_sequence,
    placement_sequence,
    return_home,
    complete_assembly_cycle,
    RETURN_TRANSIT_V,
)


def _is_settled(pose, foot, carrying):
    if foot == FOOT_LEFT:
        return pose.w_left == SUPPORT_W
    on_block_height = SUPPORT_W_WITH_LOAD if carrying else SUPPORT_W
    return pose.w_right == on_block_height


class PickupSequenceTests(unittest.TestCase):
    def test_ends_on_stable_odd_carrying_stance(self):
        poses, is_carrying = pickup_sequence(START_POSE, stone_u=4, stone_v=0, stone_w=4)
        final = poses[-1]
        self.assertEqual(final.u_right, 3)   # stone_u(4) + reposition(-1)
        self.assertEqual(final.v_right, 2)   # 0 + side_shift(2)
        self.assertEqual(final.w_right, 2)   # SUPPORT_W_WITH_LOAD
        self.assertEqual(final.u_right % 2, 1)
        self.assertTrue(is_carrying[-1])

    def test_free_foot_is_untouched(self):
        poses, _ = pickup_sequence(START_POSE, stone_u=4, stone_v=0, stone_w=4)
        final = poses[-1]
        self.assertEqual(final.u_left, START_POSE.u_left)
        self.assertEqual(final.v_left, START_POSE.v_left)
        self.assertEqual(final.w_left, START_POSE.w_left)

    def test_descends_onto_the_stone_before_lifting_with_it(self):
        poses, _ = pickup_sequence(START_POSE, stone_u=4, stone_v=0, stone_w=4)
        self.assertTrue(any(p.w_right == 4 and p.u_right == 4 for p in poses))

    def test_left_foot_can_carry_instead(self):
        poses, _ = pickup_sequence(
            START_POSE, stone_u=0, stone_v=0, stone_w=4,
            carrying_foot=FOOT_LEFT)
        final = poses[-1]
        self.assertEqual(final.u_left, -1)  # 0 + reposition(-1)
        self.assertEqual(final.u_right, START_POSE.u_right)

    def test_is_carrying_matches_poses_length_and_starts_false(self):
        poses, is_carrying = pickup_sequence(START_POSE, stone_u=4, stone_v=0, stone_w=4)
        self.assertEqual(len(poses), len(is_carrying))
        self.assertFalse(is_carrying[0])

    def test_is_carrying_becomes_true_only_after_grip(self):
        poses, is_carrying = pickup_sequence(START_POSE, stone_u=4, stone_v=0, stone_w=4)
        # find the pose where the foot is resting on the stone (grip moment)
        grip_index = next(
            i for i, p in enumerate(poses) if p.w_right == 4 and p.u_right == 4)
        self.assertFalse(is_carrying[grip_index])
        self.assertTrue(all(is_carrying[grip_index + 1:]))

    def test_actually_approaches_stone_v_not_just_stone_u(self):
        # regression test: stone_v was accepted as a parameter and
        # documented, but an earlier version of this function never
        # actually used it to move the carrying foot there before
        # descending - it would silently grip at (stone_u, whatever v
        # the foot already happened to be at, stone_w) instead of the
        # real stone whenever stone_v differed from 0 (every existing
        # test used stone_v=0, which coincidentally matched START_POSE's
        # starting v, hiding the bug completely).
        poses, is_carrying = pickup_sequence(START_POSE, stone_u=2, stone_v=2, stone_w=4)
        self.assertTrue(any(
            p.u_right == 2 and p.v_right == 2 and p.w_right == 4 for p in poses))
        final = poses[-1]
        self.assertTrue(is_carrying[-1])
        self.assertEqual((final.u_right + final.v_right) % 2, 1)

    def test_raises_instead_of_gripping_wrong_spot_when_stone_unreachable(self):
        # regression test: if the carrying foot cannot actually reach the
        # stone (here: stone_u=0, stone_v=0 coincides exactly with the
        # free foot's own position at START_POSE - physically impossible
        # to place the carrying foot there too), this must fail loudly
        # instead of silently descending and gripping wherever the
        # approach happened to stop short, as if that were the stone.
        with self.assertRaises(ValueError):
            pickup_sequence(START_POSE, stone_u=0, stone_v=0, stone_w=4)

    def test_raises_instead_of_settling_on_invalid_parity_after_backoff(self):
        # regression test: side_shift/reposition are tuned to always
        # land on a valid red/odd-sum foothold, but only if both strides
        # fully reach their computed target - if either gets capped
        # short (e.g. by the physical leg-span limit relative to the
        # free foot), the old code would silently settle on an invalid
        # (even-sum) stance instead. stone=(4, 4) with the free foot at
        # START_POSE's (0, 0) triggers exactly this.
        with self.assertRaises(ValueError):
            pickup_sequence(START_POSE, stone_u=4, stone_v=4, stone_w=4)


class PlacementSequenceTests(unittest.TestCase):
    def test_ends_at_target_height_shifted_over(self):
        pose = START_POSE._replace(u_right=3, v_right=2, w_right=2)
        poses, is_carrying, stone_placed = placement_sequence(pose, target_w=3)
        final = poses[-1]
        self.assertEqual(final.w_right, 3)
        # v_shift(3) plus the correction step (nudging to the nearest
        # valid white/both-even stance - see _white_correction_targets
        # in behaviors.py) lands at (4, 4): a placed stone physically
        # blocks its own cell plus its immediate left/right/front/back
        # neighbors (user-confirmed against the Rhino model), so the
        # backoff needs enough clearance - at least 3 units Manhattan
        # distance from the placement point (3, 2) - to actually reach a
        # foothold clear of the stone, capped by the physical leg-span
        # limit (MAX_AXIS_DISTANCE=4) from the untouched left foot at
        # (0, 0) - user-confirmed: |uL-uR| and |vL-vR| may never exceed
        # 4.
        self.assertEqual(final.u_right, 4)
        self.assertEqual(final.v_right, 4)
        self.assertGreaterEqual(abs(final.u_right - 3) + abs(final.v_right - 2), 3)
        self.assertLessEqual(abs(final.u_right - final.u_left), 4)
        self.assertLessEqual(abs(final.v_right - final.v_left), 4)
        self.assertFalse(is_carrying[-1])
        self.assertTrue(stone_placed[-1])

    def test_rises_above_target_before_backing_off(self):
        pose = START_POSE._replace(u_right=3, v_right=2, w_right=2)
        poses, _, _ = placement_sequence(pose, target_w=3)
        self.assertTrue(any(p.w_right == 4 for p in poses))

    def test_is_carrying_true_at_placement_then_false_after_release(self):
        pose = START_POSE._replace(u_right=3, v_right=2, w_right=2)
        poses, is_carrying, stone_placed = placement_sequence(pose, target_w=3)
        self.assertTrue(is_carrying[0])
        place_index = next(i for i, p in enumerate(poses) if p.w_right == 3)
        self.assertTrue(is_carrying[place_index])
        self.assertFalse(is_carrying[-1])

    def test_stone_placed_becomes_true_at_touchdown_and_stays_true(self):
        # regression test: stone_placed is a SEPARATE signal from
        # is_carrying - it must go True the moment the block reaches its
        # final voxel (before release), then stay True through release
        # and the whole backoff, since the block itself does not move
        # again after that.
        pose = START_POSE._replace(u_right=3, v_right=2, w_right=2)
        poses, is_carrying, stone_placed = placement_sequence(pose, target_w=3)
        self.assertFalse(stone_placed[0])
        place_index = next(i for i, p in enumerate(poses) if p.w_right == 3)
        self.assertTrue(stone_placed[place_index])
        self.assertTrue(is_carrying[place_index])  # still gripped at touchdown
        self.assertTrue(all(stone_placed[place_index:]))


class ReturnHomeTests(unittest.TestCase):
    def test_reaches_home_pose_exactly(self):
        pose = START_POSE._replace(u_right=9, v_right=8, w_right=1, u_left=8, v_left=8)
        poses, is_carrying = return_home(pose, START_POSE, carrying_foot=FOOT_RIGHT)
        self.assertEqual(poses[-1], START_POSE)
        self.assertFalse(any(is_carrying))

    def test_a_foot_already_at_its_target_uv_still_settles_immediately(self):
        # regression test: a foot left at the with-load height (2) after
        # placement, that does NOT need any u/v movement to reach home,
        # used to stay stranded at height 2 for the entire return trip
        # instead of settling down to the normal standing height (1)
        # right away.
        pose = START_POSE._replace(
            u_left=START_POSE.u_left, v_left=START_POSE.v_left, w_left=2,
            u_right=9, v_right=8, w_right=2)
        poses, _ = return_home(pose, START_POSE, carrying_foot=FOOT_RIGHT)
        # left foot already matches home's u,v - it must settle to w=1
        # within the first couple of poses, not linger at 2.
        settle_index = next(i for i, p in enumerate(poses) if p.w_left == 1)
        self.assertLessEqual(settle_index, 1)


class CompleteAssemblyCycleTests(unittest.TestCase):
    def test_full_cycle_returns_to_home(self):
        target = Pose(0, 0, 1, 3, 2, 3)
        poses, _, _, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        self.assertEqual(poses[-1], START_POSE)

    def test_full_cycle_passes_through_placement_height(self):
        target = Pose(0, 0, 1, 3, 2, 3)
        poses, _, _, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        self.assertTrue(any(p.w_right == 3 for p in poses))

    def test_the_two_feet_never_settle_on_the_same_voxel(self):
        # regression test: the two feet can momentarily share the same
        # (u, v) while one of them is mid-stride (airborne, different w)
        # - that is fine, feet at different heights are not colliding.
        # What must never happen is BOTH feet being settled (resting on
        # their own true ground-contact height) at the same (u, v) at
        # once.
        target = Pose(8, 8, 0, 9, 8, 1)
        poses, is_carrying, _, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        for i, p in enumerate(poses):
            if (p.u_left, p.v_left) == (p.u_right, p.v_right):
                both_settled = (
                    _is_settled(p, FOOT_LEFT, is_carrying[i])
                    and _is_settled(p, FOOT_RIGHT, is_carrying[i]))
                self.assertFalse(both_settled, msg="step {0}: {1}".format(i, p))

    def test_cycle_is_indexable_like_a_normal_motion_plan(self):
        target = Pose(0, 0, 1, 3, 2, 3)
        poses, is_carrying, stone_placed, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        self.assertGreater(len(poses), 10)
        self.assertEqual(poses[0], START_POSE)
        self.assertEqual(len(poses), len(is_carrying))
        self.assertEqual(len(poses), len(stone_placed))

    def test_is_carrying_starts_and_ends_false_with_a_true_middle(self):
        target = Pose(0, 0, 1, 3, 2, 3)
        poses, is_carrying, _, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        self.assertFalse(is_carrying[0])
        self.assertFalse(is_carrying[-1])
        self.assertTrue(any(is_carrying))

    def test_stone_placed_starts_false_and_stays_true_to_the_end(self):
        # regression test: unlike is_carrying (False again by the end),
        # stone_placed must stay True from the moment the block is set
        # down all the way through the return trip home - the block
        # itself never moves again after that.
        target = Pose(0, 0, 1, 3, 2, 3)
        poses, _, stone_placed, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        self.assertFalse(stone_placed[0])
        self.assertTrue(stone_placed[-1])
        self.assertTrue(any(stone_placed))

    def test_occupied_output_contains_the_placed_stones_column(self):
        target = Pose(0, 0, 1, 3, 2, 3)
        _, _, _, occupied = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        self.assertIn((target.u_right, target.v_right), occupied)

    def test_return_trip_never_settles_on_the_just_placed_column(self):
        # regression test: reported directly - after placing at
        # (u=5, v=8, w=4) the right foot was later trying to settle back
        # at (u=5, v=8, w=2) during the return trip so the left foot
        # could keep turning, which is physically impossible once a
        # stone occupies that column. The whole return leg must now
        # avoid that column entirely, not just the exact placement w.
        target = Pose(8, 8, 1, 5, 8, 4)
        poses, _, _, occupied = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        placed_column = (target.u_right, target.v_right)
        self.assertIn(placed_column, occupied)
        for p in poses:
            self.assertFalse(
                (p.u_left, p.v_left) == placed_column and p.w_left == SUPPORT_W,
                msg=repr(p))
            self.assertFalse(
                (p.u_right, p.v_right) == placed_column and p.w_right == SUPPORT_W,
                msg=repr(p))
        self.assertEqual(poses[-1], START_POSE)

    def test_occupied_from_an_earlier_cycle_is_avoided_by_the_next_one(self):
        # a second stone's whole cycle (pickup, transport, return) must
        # never settle a foot on a column an EARLIER cycle already built
        # on, when that earlier column is passed in via ``occupied``.
        first_target = Pose(8, 8, 0, 9, 8, 1)
        _, _, _, occupied = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=first_target)
        self.assertTrue(occupied)

        second_target = Pose(0, 0, 1, 5, 2, 1)
        poses, _, _, occupied2 = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4,
            target_pose=second_target, occupied=occupied)
        for column in occupied:
            for p in poses:
                self.assertFalse(
                    (p.u_left, p.v_left) == column and p.w_left == SUPPORT_W,
                    msg=repr(p))
                self.assertFalse(
                    (p.u_right, p.v_right) == column and p.w_right == SUPPORT_W,
                    msg=repr(p))
        self.assertTrue(occupied <= occupied2)

    def test_return_trip_visits_the_v2_transit_lane_before_home(self):
        # the return leg must route through RETURN_TRANSIT_V (avoiding
        # the v=0 material station lane for as much of the trip as
        # possible) rather than walking straight back down v=0.
        target = Pose(0, 0, 1, 3, 2, 3)
        poses, _, _, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        self.assertTrue(any(
            p.v_left == RETURN_TRANSIT_V and p.v_right == RETURN_TRANSIT_V
            for p in poses))

    def test_right_foot_never_rests_without_load_on_a_red_foothold(self):
        # regression test: after release (placement backoff) and again
        # during return_home, the right foot can still carry an odd
        # (u + v) sum left over from carrying a block (an odd u+v sum is
        # only ever a valid *carrying* foothold - see _land_on_odd_u in
        # behaviors.py; the RhomBlock lattice's red/white pattern
        # alternates on the sum of both coordinates, like a checkerboard,
        # NOT on u alone - confirmed directly against the user's Rhino
        # model, which showed u=2, v=1 as red even though u=2 alone is
        # even). Settling to the no-load support height on a red
        # foothold - even for a single frame - is not physically valid
        # (reported directly: "rechte Fuss ohne Stein darf nicht auf rot
        # Pyramide stehen"). This must never happen, at any step, for any
        # target/stone combination that requires the right foot to carry.
        target = Pose(8, 8, 0, 9, 8, 1)
        poses, is_carrying, _, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        for i, p in enumerate(poses):
            if not is_carrying[i] and p.w_right == SUPPORT_W:
                self.assertEqual(
                    (p.u_right + p.v_right) % 2, 0, msg="step {0}: {1}".format(i, p))
        self.assertEqual(poses[-1], START_POSE)

    def test_right_foot_never_rests_without_load_on_a_red_foothold_nonzero_v_case(self):
        # the exact scenario the user reported as still broken after the
        # first fix: target u_right=7 (odd, valid carrying foothold - see
        # _land_on_odd_u) at v_right=8 (7+8=15, odd - consistent). The
        # return trip afterwards must never rest the free right foot
        # anywhere its (u + v) sum is odd.
        target = Pose(8, 8, 0, 7, 8, 1)
        poses, is_carrying, _, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        for i, p in enumerate(poses):
            if not is_carrying[i] and p.w_right == SUPPORT_W:
                self.assertEqual(
                    (p.u_right + p.v_right) % 2, 0, msg="step {0}: {1}".format(i, p))
        self.assertEqual(poses[-1], START_POSE)

    def test_free_foot_target_is_pushed_clear_of_the_placed_block(self):
        # regression test: the caller gave u_left=8, v_left=8 as the free
        # foot's post-placement standing spot, right next to where the
        # block gets placed (u_right=7, v_right=8 - Manhattan distance 1).
        # The free foot arrives at its target during transport, BEFORE
        # the block is placed there, so a too-close target would leave it
        # standing where the block (or its blocked neighboring cells)
        # needs to go - reported directly: "die Füß kann nicht direkt
        # neben ihm sein, da hat sie keine Platz" / "das geht nicht der
        # roboter kann nicht so stehen". complete_assembly_cycle must
        # push it out to at least Manhattan distance 3 automatically
        # (see _ensure_min_clearance) before doing anything else.
        target = Pose(8, 8, 0, 7, 8, 2)
        poses, is_carrying, _, _ = complete_assembly_cycle(
            START_POSE, stone_u=4, stone_v=0, stone_w=4, target_pose=target)
        # find the free (left) foot's fully-settled resting pose right
        # before release (C2) - the moment is_carrying flips True->False -
        # by which point transport has long finished and the free foot
        # is done moving for good.
        release_index = next(
            i for i in range(1, len(poses)) if is_carrying[i - 1] and not is_carrying[i])
        parked = poses[release_index - 1]
        self.assertGreaterEqual(
            abs(parked.u_left - 7) + abs(parked.v_left - 8), 3, msg=repr(parked))
        self.assertEqual((parked.u_left, parked.v_left), (10, 8))
        self.assertEqual(poses[-1], START_POSE)


if __name__ == "__main__":
    unittest.main()
