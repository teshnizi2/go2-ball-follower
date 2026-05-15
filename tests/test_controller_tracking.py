import unittest

from controller import FollowController, INVISIBLE_THRESH


class FollowControllerTrackingTests(unittest.TestCase):
    def test_far_target_keeps_forward_motion_while_turning_hard(self) -> None:
        ctrl = FollowController()
        vx, vyaw = ctrl.compute(
            cx=0,
            cy=120,
            area=800.0,
            conf=1.0,
            img_w=480,
            img_h=360,
            dt=0.02,
            world_dist_m=2.5,
        )
        self.assertGreaterEqual(abs(vyaw), 0.3)
        self.assertGreaterEqual(
            vx,
            0.05,
            "Controller should keep positive forward intent while turning so it keeps trying to close distance.",
        )

    def test_transitions_to_recovering_after_long_loss(self) -> None:
        ctrl = FollowController()
        ctrl.compute(
            cx=240,
            cy=120,
            area=1200.0,
            conf=1.0,
            img_w=480,
            img_h=360,
            dt=0.02,
            world_dist_m=1.2,
        )
        last = (0.0, 0.0)
        for _ in range(INVISIBLE_THRESH + 1):
            last = ctrl.compute(
                cx=None,
                cy=None,
                area=0.0,
                conf=0.0,
                img_w=480,
                img_h=360,
                dt=0.02,
                world_dist_m=1.2,
            )
        vx, vyaw = last
        self.assertAlmostEqual(vx, 0.0, delta=1e-6)
        self.assertGreater(abs(vyaw), 0.2)

    def test_offcenter_target_prioritizes_recentering_over_forward_speed(self) -> None:
        ctrl = FollowController()
        vx_center, vyaw_center = ctrl.compute(
            cx=240,
            cy=120,
            area=800.0,
            conf=1.0,
            img_w=480,
            img_h=360,
            dt=0.02,
            world_dist_m=2.5,
        )
        vx_edge, vyaw_edge = ctrl.compute(
            cx=30,
            cy=120,
            area=800.0,
            conf=1.0,
            img_w=480,
            img_h=360,
            dt=0.02,
            world_dist_m=2.5,
        )
        self.assertGreater(vx_center, vx_edge)
        self.assertGreater(vx_edge, 0.0)
        self.assertGreater(abs(vyaw_edge), abs(vyaw_center))


if __name__ == "__main__":
    unittest.main()
