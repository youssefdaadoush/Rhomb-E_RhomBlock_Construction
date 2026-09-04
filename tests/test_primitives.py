import unittest

from rhombe_motion.simple_motion import START_POSE
from rhombe_motion.primitives import (
    FOOT_LEFT,
    FOOT_RIGHT,
    AXIS_U,
    AXIS_V,
    get_coordinate,
    lift_or_lower_foot,
    translate_foot,
    rotate_support_foot,
)


class LiftOrLowerFootTests(unittest.TestCase):
    def test_lift_raises_one_unit_at_a_time(self):
        poses, new_pose = lift_or_lower_foot(START_POSE, FOOT_RIGHT, 3)
        self.assertEqual(len(poses), 2)
        self.assertEqual(poses[0].w_right, 2)
        self.assertEqual(poses[1].w_right, 3)
        self.assertEqual(new_pose.w_right, 3)

    def test_lower_decreases_one_unit_at_a_time(self):
        raised = START_POSE._replace(w_left=4)
        poses, new_pose = lift_or_lower_foot(raised, FOOT_LEFT, 1)
        self.assertEqual([p.w_left for p in poses], [3, 2, 1])
        self.assertEqual(new_pose.w_left, 1)

    def test_already_at_target_returns_no_poses(self):
        poses, new_pose = lift_or_lower_foot(START_POSE, FOOT_RIGHT, 1)
        self.assertEqual(poses, [])
        self.assertEqual(new_pose, START_POSE)

    def test_only_touches_the_requested_foot(self):
        poses, new_pose = lift_or_lower_foot(START_POSE, FOOT_RIGHT, 2)
        self.assertEqual(new_pose.u_left, START_POSE.u_left)
        self.assertEqual(new_pose.v_left, START_POSE.v_left)
        self.assertEqual(new_pose.w_left, START_POSE.w_left)
        self.assertEqual(new_pose.u_right, START_POSE.u_right)
        self.assertEqual(new_pose.v_right, START_POSE.v_right)


class TranslateFootTests(unittest.TestCase):
    def test_moves_toward_target_capped_at_max_stride(self):
        poses, new_pose = translate_foot(
            START_POSE, FOOT_RIGHT, AXIS_U, target=10, max_stride=2)
        self.assertEqual([p.u_right for p in poses], [3, 4])
        self.assertEqual(new_pose.u_right, 4)

    def test_moves_negative_direction(self):
        far = START_POSE._replace(u_right=10)
        poses, new_pose = translate_foot(
            far, FOOT_RIGHT, AXIS_U, target=2, max_stride=2)
        self.assertEqual([p.u_right for p in poses], [9, 8])
        self.assertEqual(new_pose.u_right, 8)

    def test_does_not_overshoot_a_close_target(self):
        poses, new_pose = translate_foot(
            START_POSE, FOOT_LEFT, AXIS_V, target=1, max_stride=4)
        self.assertEqual([p.v_left for p in poses], [1])
        self.assertEqual(new_pose.v_left, 1)

    def test_already_at_target_returns_no_poses(self):
        poses, new_pose = translate_foot(
            START_POSE, FOOT_RIGHT, AXIS_U, target=2, max_stride=2)
        self.assertEqual(poses, [])
        self.assertEqual(new_pose, START_POSE)

    def test_adjust_step_can_change_landing_spot(self):
        # Mirrors the reference implementation's "carrying foot must land
        # on an odd u" rule, expressed as an injected policy instead of
        # being hardcoded into the primitive.
        def land_on_odd(pose, foot, axis, naive_step):
            current = get_coordinate(pose, foot, axis)
            landing = current + naive_step
            if landing % 2 == 0:
                landing += 1 if naive_step >= 0 else -1
            return landing - current

        poses, new_pose = translate_foot(
            START_POSE, FOOT_RIGHT, AXIS_U, target=4,
            max_stride=2, adjust_step=land_on_odd)
        self.assertEqual(new_pose.u_right % 2, 1)
        self.assertEqual(new_pose.u_right, 5)
        self.assertEqual([p.u_right for p in poses], [3, 4, 5])

    def test_adjust_step_can_block_movement(self):
        poses, new_pose = translate_foot(
            START_POSE, FOOT_RIGHT, AXIS_U, target=10,
            max_stride=2, adjust_step=lambda pose, foot, axis, step: 0)
        self.assertEqual(poses, [])
        self.assertEqual(new_pose, START_POSE)


class RotateSupportFootTests(unittest.TestCase):
    def test_pivots_both_axes_and_height_together(self):
        poses, new_pose = rotate_support_foot(
            START_POSE, FOOT_LEFT,
            target_u=2, target_v=3, target_w=2,
            max_u_stride=2, max_v_stride=4)
        self.assertEqual(new_pose.u_left, 2)
        self.assertEqual(new_pose.v_left, 3)
        self.assertEqual(new_pose.w_left, 2)
        # order: w changes first, then u, then v (lift, then move, then move)
        self.assertEqual(poses[0].w_left, 2)
        self.assertTrue(any(p.u_left == 2 for p in poses))
        self.assertTrue(any(p.v_left == 3 for p in poses))

    def test_only_touches_the_requested_foot(self):
        poses, new_pose = rotate_support_foot(
            START_POSE, FOOT_RIGHT,
            target_u=4, target_v=4, target_w=2,
            max_u_stride=2, max_v_stride=4)
        self.assertEqual(new_pose.u_left, START_POSE.u_left)
        self.assertEqual(new_pose.v_left, START_POSE.v_left)
        self.assertEqual(new_pose.w_left, START_POSE.w_left)

    def test_no_movement_needed_returns_no_poses(self):
        poses, new_pose = rotate_support_foot(
            START_POSE, FOOT_RIGHT,
            target_u=START_POSE.u_right, target_v=START_POSE.v_right,
            target_w=START_POSE.w_right,
            max_u_stride=2, max_v_stride=4)
        self.assertEqual(poses, [])
        self.assertEqual(new_pose, START_POSE)

    def test_adjust_hooks_apply_per_axis(self):
        def land_on_odd(pose, foot, axis, naive_step):
            current = get_coordinate(pose, foot, axis)
            landing = current + naive_step
            if landing % 2 == 0:
                landing += 1 if naive_step >= 0 else -1
            return landing - current

        poses, new_pose = rotate_support_foot(
            START_POSE, FOOT_RIGHT,
            target_u=4, target_v=4, target_w=2,
            max_u_stride=2, max_v_stride=4,
            adjust_u_step=land_on_odd)
        self.assertEqual(new_pose.u_right % 2, 1)


if __name__ == "__main__":
    unittest.main()
