import json
import unittest
from pathlib import Path
from unittest.mock import patch

from planning import validate_plan
from robot import ExecutionError, RobotExecutor
from warehouse_task1 import DEFAULT_TASK1, chase_tag_zero


ROOT = Path(__file__).resolve().parents[1]


class WarehouseRobot:
    def __init__(self):
        self.calls = []

    def stop_chassis(self): self.calls.append(("stop",))
    def open_camera(self): self.calls.append(("camera",))
    def get_peripheral_devices_list(self): return [{"deviceId": 41, "type": "Infrared"}]
    def turn_servo_angle(self, *args, **kwargs): self.calls.append(("servo", args))
    def mechanical_clamp_release(self): self.calls.append(("release",))
    def stop_servo(self, *args, **kwargs): self.calls.append(("stop_servo", args))
    def mecanum_move_xyz(self, *args): self.calls.append(("xyz", args))
    def load_models(self, names): self.calls.append(("load_models", names)); return True
    def release_models(self, names): self.calls.append(("release_models", names))


class WarehouseAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.config["warehouse"]["motion_calibrated"] = True
        self.config["warehouse"]["pickup_start"]["calibrated"] = True
        self.robot = WarehouseRobot()
        self.events = []
        self.executor = RobotExecutor(self.robot, self.config, None)
        self.executor.scene = "warehouse"
        self.executor.record = lambda event, **kw: self.events.append((event, kw))

    def test_task1_source_is_read_only_and_present(self):
        self.assertTrue(DEFAULT_TASK1.is_file())

    @patch("robot.time.sleep")
    def test_numbered_position_is_forfeited_and_released_in_place(self, _sleep):
        self.executor.carrying = True
        result = self.executor.place(3)
        self.assertFalse(result["numbered_position_scored"])
        self.assertFalse(self.executor.carrying)
        self.assertEqual([call[0] for call in self.robot.calls],
                         ["stop", "servo", "release", "servo", "stop_servo"])
        self.assertEqual(self.robot.calls[1][1][:2], (51, 102))
        self.assertEqual(self.events[0][0], "warehouse_place_forfeited")

    @patch("warehouse_task1.load_task1")
    def test_pickup_calls_task1_tracking_and_grab(self, load):
        load.return_value.track_and_grab_phase.return_value = True
        self.executor.pickup("blue")
        self.assertTrue(self.executor.carrying)
        load.return_value.track_and_grab_phase.assert_called_once()
        args = load.return_value.track_and_grab_phase.call_args.args
        self.assertIs(args[0], self.robot)
        self.assertEqual(args[1:], ("blue", 41))

    @patch("warehouse_task1.load_task1")
    def test_tag_chase_calls_task1_with_raw_robot(self, load):
        load.return_value.TARGET_TAG_ID = 0
        load.return_value.APRILTAG_TARGET_DISTANCE = 9.3
        load.return_value._chase_apriltag.return_value = True
        chase_tag_zero(self.robot, None, 41)
        load.return_value._chase_apriltag.assert_called_once_with(self.robot, 0, 9.3, 41)
        self.assertEqual(self.robot.calls,
                         [("load_models", ["apriltag_qrcode"]),
                          ("stop",), ("release_models", ["apriltag_qrcode"])])

    def test_warehouse_motion_stays_locked_without_measured_start(self):
        self.config["warehouse"]["pickup_start"]["calibrated"] = False
        with self.assertRaises(ExecutionError):
            self.executor._require_motion()

    def test_warehouse_preflight_does_not_require_numbered_position(self):
        plan = validate_plan({"scene": "warehouse", "actions": [
            {"type": "move", "direction": "forward", "cm": 10},
            {"type": "pickup", "color": "red"},
            {"type": "place", "position": 3},
            {"type": "move", "direction": "backward", "cm": 10},
            {"type": "return_line"}]})
        self.executor.preflight(plan)


if __name__ == "__main__":
    unittest.main()
