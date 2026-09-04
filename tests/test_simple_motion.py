import math
import unittest

from rhombe_motion.simple_motion import (
    START_POSE,
    DEFAULT_HEADING_DEGREES,
    plan_simple_motion,
    resolve_heading_degrees,
    resolve_leg_geometry,
    resolve_point_from_offsets,
)


class SimpleMotionTests(unittest.TestCase):
    def test_start_target(self):
        plan = plan_simple_motion(0, 0, 1, 2, 0, 1)
        self.assertTrue(plan.reached)
        self.assertEqual(plan.poses, (START_POSE,))

    def test_forward_target_is_reached(self):
        plan = plan_simple_motion(2, 4, 1, 4, 4, 1)
        self.assertTrue(plan.reached)
        self.assertEqual(plan.poses[-1], plan.target)

    def test_right_foot_moves_first_on_v(self):
        plan = plan_simple_motion(0, 1, 1, 2, 1, 1)
        self.assertEqual(plan.poses[1].w_right, 2)
        self.assertEqual(plan.poses[1].v_left, 0)

    def test_backward_target_is_ignored_like_reference(self):
        plan = plan_simple_motion(0, 0, 1, 1, 0, 1)
        self.assertFalse(plan.reached)
        self.assertEqual(plan.poses, (START_POSE,))
        self.assertIn("partly ignored", plan.message)

    def test_slider_is_clamped(self):
        plan = plan_simple_motion(0, 1, 1, 2, 1, 1)
        self.assertEqual(plan.pose_at(999), plan.poses[-1])
        self.assertEqual(plan.pose_at(-5), plan.poses[0])

    def test_distant_target_alternates_feet(self):
        plan = plan_simple_motion(20, 16, 1, 22, 16, 1)
        self.assertTrue(plan.reached)
        for pose in plan.poses:
            self.assertLessEqual(abs(pose.v_right - pose.v_left), 4)
            self.assertLessEqual(abs(pose.u_right - pose.u_left), 4)

    def test_resolve_heading_uses_real_points_not_voxels(self):
        # Same voxel delta (dx=2) but a real-world direction that is not
        # aligned with the voxel u-axis - a naive voxel-space atan2 would
        # get this wrong; resolve_heading_degrees must use the real points.
        heading = resolve_heading_degrees(
            "target-a", False, point_a=(0.0, 0.0, 0.0), point_b=(0.0, 5.0, 0.0)
        )
        self.assertAlmostEqual(heading, 90.0)

    def test_resolve_heading_falls_back_to_last_real_heading(self):
        target_key = "target-b"
        real_heading = resolve_heading_degrees(
            target_key, False, point_a=(0.0, 0.0, 0.0), point_b=(3.0, 0.0, 0.0)
        )
        fallback_heading = resolve_heading_degrees(
            target_key, True, point_a=(0.0, 0.0, 0.0), point_b=(0.0, 0.0, 5.0)
        )
        self.assertEqual(fallback_heading, real_heading)

    def test_resolve_heading_fallback_without_history_uses_default(self):
        heading = resolve_heading_degrees(
            "target-never-seen-before", True,
            point_a=(0.0, 0.0, 0.0), point_b=(0.0, 0.0, 5.0),
        )
        self.assertEqual(heading, DEFAULT_HEADING_DEGREES)

    def test_resolve_leg_geometry_flat_horizontal_case(self):
        # point_a and point_b 5 apart along X, equal leg lengths of 3.9 ->
        # isoceles triangle, point C sits above the midpoint.
        geometry = resolve_leg_geometry(
            "pc-flat", False,
            point_a=(0.0, 0.0, 0.0), point_b=(5.0, 0.0, 0.0),
            leg_a=3.9, leg_b=3.9,
        )
        point_c = geometry.point_c
        self.assertAlmostEqual(point_c[0], 2.5, places=3)
        self.assertAlmostEqual(point_c[1], 0.0, places=6)
        self.assertGreater(point_c[2], 0.0)
        # distance from point_a to point_c must match leg_a
        dist_a = math.sqrt(sum((point_c[i] - (0, 0, 0)[i]) ** 2 for i in range(3)))
        self.assertAlmostEqual(dist_a, 3.9, places=3)
        self.assertAlmostEqual(geometry.base_a, 2.5, places=3)
        self.assertAlmostEqual(geometry.base_b, 2.5, places=3)
        self.assertGreater(geometry.angle_c_degrees, 0.0)
        self.assertLess(geometry.angle_c_degrees, 180.0)

    def test_resolve_leg_geometry_does_not_crash_when_vertical(self):
        # point_a directly below point_b - no unique perpendicular exists,
        # but this must not raise (the old Grasshopper cross-product logic
        # crashed here with a zero-length-vector error).
        geometry = resolve_leg_geometry(
            "pc-vertical-nohistory", False,
            point_a=(0.0, 0.0, 0.0), point_b=(0.0, 0.0, 5.0),
            leg_a=3.9, leg_b=3.9,
        )
        for value in geometry.point_c:
            self.assertFalse(math.isnan(value))
        self.assertFalse(math.isnan(geometry.angle_c_degrees))

    def test_resolve_leg_geometry_falls_back_to_last_real_geometry(self):
        target_key = "pc-fallback"
        real_geometry = resolve_leg_geometry(
            target_key, False,
            point_a=(0.0, 0.0, 0.0), point_b=(5.0, 0.0, 0.0),
            leg_a=3.9, leg_b=3.9,
        )
        fallback_geometry = resolve_leg_geometry(
            target_key, True,
            point_a=(0.0, 0.0, 0.0), point_b=(0.0, 0.0, 5.0),
            leg_a=3.9, leg_b=3.9,
        )
        self.assertEqual(fallback_geometry, real_geometry)

    def test_resolve_point_from_offsets_flat_case(self):
        point_c = resolve_point_from_offsets(
            "cluster-drop-in-flat",
            point_a=(0.0, 0.0, 0.0), point_b=(5.0, 0.0, 0.0),
            base_a=2.5, height_c=3.0,
        )
        self.assertAlmostEqual(point_c[0], 2.5, places=6)
        self.assertAlmostEqual(point_c[1], 0.0, places=6)
        self.assertAlmostEqual(abs(point_c[2]), 3.0, places=6)

    def test_resolve_point_from_offsets_freezes_direction_when_vertical(self):
        target_key = "cluster-drop-in-vertical"
        first = resolve_point_from_offsets(
            target_key,
            point_a=(0.0, 0.0, 0.0), point_b=(5.0, 0.0, 0.0),
            base_a=2.5, height_c=3.0,
        )
        second = resolve_point_from_offsets(
            target_key,
            point_a=(0.0, 0.0, 10.0), point_b=(0.0, 0.0, 15.0),
            base_a=2.5, height_c=3.0,
        )
        # direction is reused (frozen), but the offset now starts from the
        # new point_a - the point must not simply repeat the old value.
        self.assertNotEqual(first, second)
        self.assertAlmostEqual(second[0] - 0.0, first[0] - 0.0, places=6)

    def test_resolve_point_from_offsets_no_crash_without_history(self):
        point_c = resolve_point_from_offsets(
            "cluster-drop-in-never-seen",
            point_a=(0.0, 0.0, 0.0), point_b=(0.0, 0.0, 5.0),
            base_a=2.5, height_c=3.0,
        )
        for value in point_c:
            self.assertFalse(math.isnan(value))

    def test_heading_is_fallback_flag_matches_shared_u_v(self):
        plan = plan_simple_motion(4, 4, 1, 4, 4, 2)
        for step, pose in enumerate(plan.poses):
            expected = (pose.u_left == pose.u_right
                        and pose.v_left == pose.v_right)
            self.assertEqual(plan.heading_is_fallback_at(step), expected)
        self.assertFalse(plan.heading_is_fallback_at(0))


if __name__ == "__main__":
    unittest.main()
