import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from planning import validate_plan
from robot import ExecutionError, RobotExecutor, listen_text

ROOT = Path(__file__).resolve().parents[1]


class FakeRobot:
    def __init__(self):
        self.calls = []
        self.AUDIO = self
        self.yaw = 0.0
        self.pending_yaw = 0.0
        self.continuous_moving = False
    def mecanum_move_speed_times(self, *args):
        if not isinstance(args[2], int):
            raise TypeError("UGOT travel distance must be an integer")
        self.calls.append(("move", args))
    def mecanum_translate_speed_times(self, *args):
        if not isinstance(args[2], int):
            raise TypeError("UGOT strafe distance must be an integer")
        self.calls.append(("strafe", args))
    def mecanum_turn_speed_times(self, *args):
        if not isinstance(args[2], int):
            raise TypeError("UGOT turn target must be an integer")
        self.calls.append(("turn", args))
        self.pending_yaw += args[2] * (1 if args[0] == 2 else -1)
    def mecanum_move_turn(self, *args):
        self.calls.append(("continuous_turn", args))
        self.continuous_moving = True
    def read_gyro_data(self):
        if self.pending_yaw:
            step = max(-15, min(15, self.pending_yaw))
            self.yaw += step
            self.pending_yaw -= step
        elif self.continuous_moving:
            self.yaw -= 15
        return [0, 0, self.yaw % 360, 0, 0, 0, 0, 0, 9.8]
    def stop_chassis(self):
        self.calls.append(("stop", ()))
        self.continuous_moving = False
    def screen_display_emotion(self, name): self.calls.append(("emotion", (name,)))
    def play_sound(self, name, wait=False): self.calls.append(("sound", (name, wait)))
    def show_light_rgb_effect(self, *args): self.calls.append(("light", args))
    def screen_print_text(self, *args): self.calls.append(("screen", args))
    def play_audio_tts(self, *args, **kwargs): self.calls.append(("tts", args))
    def setAudioTts(self, text, voice):
        self.calls.append(("tts", (text, voice)))
        return SimpleNamespace(msg="success")
    def start_audio_asr(self): return "请讲一个笑话"
    def start_audio_asr_doa(self, duration=15): return ["前方", "请讲一个笑话"]
    def load_models(self, names): self.calls.append(("load_models", tuple(names))); return True
    def release_models(self, names): self.calls.append(("release_models", tuple(names)))
    def get_words_result(self): return "安全第一"
    def get_gesture_result(self): return "点赞"
    def get_traffic_total_info(self): return [["停止"]]
    def get_apriltag_total_info(self): return [[4]]
    def get_color_total_info(self): return ["红色"]
    def open_camera(self): self.calls.append(("camera", ()))
    def mechanical_clamp_release(self): self.calls.append(("clamp_release", ()))
    def turn_servo_angle(self, *args, **kwargs): self.calls.append(("servo", (args, kwargs)))
    def stop_servo(self, *args, **kwargs): self.calls.append(("servo_stop", (args, kwargs)))


class FakeLLM:
    def answer(self, question): return "这是一个简短回答。"


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.robot = FakeRobot()
        self.executor = RobotExecutor(self.robot, self.config, FakeLLM(), reconnect=lambda ip: self.robot)

    def test_uncalibrated_motion_stops_before_command(self):
        with self.assertRaises(ExecutionError): self.executor.move("forward", 10)
        self.assertFalse(any(c[0] == "move" for c in self.robot.calls))

    def test_asr_service_error_is_not_spoken_command(self):
        self.robot.start_audio_asr_doa = lambda duration=15: ["", "Something is error"]
        with self.assertRaisesRegex(ExecutionError, "ASR service"):
            listen_text(self.robot)

    def test_asr_allows_eight_seconds_before_speech(self):
        self.robot.getAsrAndDoa = Mock(return_value=json.dumps({"asr": {"msg": "请帮我"}}))
        self.assertEqual(listen_text(self.robot), "请帮我")
        self.robot.getAsrAndDoa.assert_called_once_with(begin_vad=8000, end_vad=1500, duration=30000)

    @patch("robot.time.sleep")
    def test_boundary_guard(self, _sleep):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        self.executor.x_cm, self.executor.y_cm = 75, 225
        with self.assertRaises(ExecutionError): self.executor.move("forward", 10)
        self.assertFalse(any(c[0] == "move" for c in self.robot.calls))

    @patch("robot.time.sleep")
    def test_math_motion_order(self, _sleep):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        plan = validate_plan({"scene": "math", "actions": [
            {"type": "polygon", "sides": 4, "side_cm": 10.0, "direction": "clockwise", "travel": "forward"},
            {"type": "strafe", "direction": "left", "cm": 10.0},
            {"type": "polygon", "sides": 3, "side_cm": 10, "direction": "counterclockwise", "travel": "forward"},
            {"type": "spin", "direction": "right"}, {"type": "turn", "direction": "left", "degrees": 90.0}]})
        self.executor.run(plan, lambda *a, **kw: None)
        types = [x[0] for x in self.robot.calls if x[0] in {"move", "strafe", "turn"}]
        self.assertEqual(types[:8], ["move", "turn"] * 4)
        self.assertEqual(types[8], "strafe")
        self.assertEqual(types[-2:], ["turn", "turn"])

    def test_polygon_integer_turns_close_for_every_supported_side_count(self):
        for sides in range(3, 9):
            angles = [self.executor._polygon_turn_degrees(sides, index)
                      for index in range(sides)]
            self.assertTrue(all(isinstance(angle, int) for angle in angles))
            self.assertEqual(sum(angles), 360)

    def test_math_preflight_rejects_unsafe_full_route(self):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        self.executor.y_cm = 225
        plan = validate_plan({"scene": "math", "actions": [
            {"type": "move", "direction": "forward", "cm": 30},
            {"type": "turn", "direction": "right", "degrees": 90},
            {"type": "move", "direction": "forward", "cm": 10},
            {"type": "spin", "direction": "right"}, {"type": "arm_wave"}]})
        with self.assertRaises(ExecutionError):
            self.executor.run(plan, lambda *a, **kw: None)
        self.assertFalse(any(c[0] in {"move", "turn"} for c in self.robot.calls))

    def test_math_preflight_checks_turning_footprint_at_polygon_corners(self):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        plan = validate_plan({"scene": "math", "actions": [
            {"type": "polygon", "sides": 4, "side_cm": 50,
             "direction": "clockwise", "travel": "forward"},
            {"type": "turn", "direction": "left", "degrees": 90},
            {"type": "polygon", "sides": 4, "side_cm": 50,
             "direction": "clockwise", "travel": "backward"},
            {"type": "strafe", "direction": "left", "cm": 30},
            {"type": "arm_wave"}]})
        self.executor.scene = "math"
        with self.assertRaisesRegex(ExecutionError, "rotation footprint"):
            self.executor.preflight(plan)
        self.executor.x_cm = 50
        self.executor.y_cm = 85
        self.executor.preflight(plan)

    def test_supervised_math_trial_uses_its_pose_without_unlocking_other_scenes(self):
        self.config["math_supervised_trial"] = True
        self.config["math_trial_pose"] = {
            "calibrated": True, "x_cm": 50, "y_cm": 100, "heading_deg": 0}
        plan = validate_plan({"scene": "math", "actions": [
            {"type": "polygon", "sides": 4, "side_cm": 50,
             "direction": "clockwise", "travel": "forward"},
            {"type": "turn", "direction": "left", "degrees": 90},
            {"type": "polygon", "sides": 4, "side_cm": 50,
             "direction": "clockwise", "travel": "backward"},
            {"type": "strafe", "direction": "left", "cm": 30},
            {"type": "arm_wave"}]})
        self.executor.scene = "math"
        self.executor.x_cm, self.executor.y_cm = 50, 100
        self.executor.preflight(plan)
        self.executor.scene = "warehouse"
        with self.assertRaises(ExecutionError):
            self.executor._require_motion()

    @patch("robot.time.sleep")
    def test_return_line_targets_nearest_segment_point(self, _sleep):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        self.config["warehouse"]["line_return"] = {
            "calibrated": True, "segment_cm": [[40, 100], [110, 100]],
            "tag_id": None, "line_confidence": 0.65,
        }
        with patch.object(self.executor, "_confirm_line") as confirm:
            self.executor.return_line()
        self.assertAlmostEqual(self.executor.x_cm, 75)
        self.assertAlmostEqual(self.executor.y_cm, 100)
        confirm.assert_called_once_with(0.65)

    @patch("robot.time.sleep")
    def test_place_navigates_from_current_pose_to_surveyed_coordinate(self, _sleep):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        self.config["warehouse"]["positions"]["3"] = {
            "calibrated": True, "waypoints_cm": [[120, 90]],
            "approach_cm": [130, 90], "heading_deg": 90, "tag_id": None,
        }
        self.config["warehouse"]["place_arm_pose"]["calibrated"] = True
        self.executor.x_cm, self.executor.y_cm = 105, 75
        self.executor.carrying = True
        with patch.object(self.executor, "_arm_pose"):
            self.executor.place(3)
        self.assertAlmostEqual(self.executor.x_cm, 130, delta=1)
        self.assertAlmostEqual(self.executor.y_cm, 90, delta=1)
        self.assertAlmostEqual(self.executor.heading_deg, 90)
        self.assertIn(("clamp_release", ()), self.robot.calls)

    @patch("robot.time.sleep")
    def test_companion_sdk_order(self, _sleep):
        plan = validate_plan({"scene": "companion", "actions": [
            {"type": "sound", "name": "ambulance"}, {"type": "conversation", "prompt": "询问情况"},
            {"type": "screen_text", "source": "last_reply"}, {"type": "light", "color": "blue"},
            {"type": "emotion", "name": "Smile"}]})
        self.executor.run(plan, lambda *a, **kw: None)
        filtered = [x[0] for x in self.robot.calls
                    if x[0] in {"sound", "screen", "light", "emotion"}
                    and x != ("sound", ("received", True))]
        self.assertEqual(filtered, ["sound", "screen", "light", "emotion"])

    def test_companion_light_service_failure_stops_the_plan(self):
        self.robot.DEVICE = SimpleNamespace(
            showLightEffect=lambda color, effect: SimpleNamespace(code=-1, msg="failed"))
        plan = validate_plan({"scene": "companion", "actions": [
            {"type": "light", "color": "green"}, {"type": "emotion", "name": "Smile"},
            {"type": "sound", "name": "happy"}, {"type": "sound", "name": "tiger"},
            {"type": "emotion", "name": "Love"}]})
        with self.assertRaisesRegex(ExecutionError, "light service rejected"):
            self.executor.run(plan, lambda *a, **kw: None)
        self.assertFalse(any(call[0] == "emotion" for call in self.robot.calls))

    @patch("robot.time.sleep")
    def test_companion_empty_asr_fails_without_model_reply(self, _sleep):
        self.robot.start_audio_asr_doa = lambda duration=15: ["", ""]
        plan = validate_plan({"scene": "companion", "actions": [
            {"type": "conversation", "prompt": "不要代替真实提问"},
            {"type": "emotion", "name": "Smile"}, {"type": "sound", "name": "happy"},
            {"type": "light", "color": "blue"}, {"type": "screen_text", "source": "last_reply"}]})
        with patch.object(self.executor.llm, "answer") as answer:
            with self.assertRaises(ExecutionError):
                self.executor.run(plan, lambda *a, **kw: None)
            answer.assert_not_called()
        self.assertFalse(any(c[0] == "emotion" for c in self.robot.calls))

    def test_companion_rejected_prompt_never_starts_asr(self):
        self.robot.setAudioTts = lambda text, voice: SimpleNamespace(code=-1, msg="Something is error")
        self.robot.start_audio_asr_doa = Mock()
        plan = validate_plan({"scene": "companion", "actions": [
            {"type": "conversation", "prompt": "第一次现场对话"},
            {"type": "emotion", "name": "Smile"}, {"type": "sound", "name": "happy"},
            {"type": "light", "color": "blue"}, {"type": "screen_text", "source": "last_reply"}]})
        with self.assertRaisesRegex(ExecutionError, "TTS service rejected"):
            self.executor.run(plan, lambda *a, **kw: None)
        self.robot.start_audio_asr_doa.assert_not_called()

    @patch("robot.time.sleep")
    def test_recon_one_scan_broadcasts_in_plan_order(self, _sleep):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        order = ("traffic", "color", "tag", "word", "gesture")
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind} for kind in order]})
        events = []
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda event, **details: events.append((event, details)))
        self.assertEqual(len([c for c in self.robot.calls if c[0] == "load_models"]), 1)
        turns = [c for c in self.robot.calls if c[0] == "turn"]
        self.assertEqual([turn[1][2] for turn in turns], [360])
        self.assertEqual(turns[0][1][1], self.config["recon_scan_speed_deg_s"])
        self.assertGreaterEqual(len([1 for event, _ in events if event == "recon_view"]), 3)
        self.assertEqual(len([1 for event, _ in events if event == "recon_scan_complete"]), 1)
        camera_at = next(i for i, c in enumerate(self.robot.calls) if c[0] == "camera")
        models_at = next(i for i, c in enumerate(self.robot.calls) if c[0] == "load_models")
        self.assertLess(camera_at, models_at)
        spoken = [c[1][0] for c in self.robot.calls if c[0] == "tts"]
        self.assertEqual(spoken, ["交通标志，停止。色块，红色。标签，四号。文字，安全第一。手势，点赞。"])

    @patch("robot.time.sleep")
    def test_recon_polls_vision_while_turn_is_running(self, _sleep):
        self.config["recon_start_pose"]["calibrated"] = True
        started = threading.Event()
        saw_poll = threading.Event()
        original_words = self.robot.get_words_result

        def blocking_turn(*args):
            self.robot.calls.append(("turn", args))
            self.robot.pending_yaw -= args[2]
            started.set()
            if not saw_poll.wait(2):
                raise RuntimeError("Vision was not polled during the turn")

        def words_during_turn():
            if not started.is_set():
                return ""
            saw_poll.set()
            return original_words()

        self.robot.mecanum_turn_speed_times = blocking_turn
        self.robot.get_words_result = words_during_turn
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "tag", "traffic", "color", "gesture")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda *a, **kw: None)
        self.assertTrue(saw_poll.is_set())

    @patch("robot.time.sleep")
    def test_recon_turn_failure_stops_without_broadcast(self, _sleep):
        self.config["recon_start_pose"]["calibrated"] = True
        self.robot.mecanum_turn_speed_times = lambda *_args: (_ for _ in ()).throw(RuntimeError("motor RPC failed"))
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "tag", "traffic", "color", "gesture")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            with self.assertRaisesRegex(ExecutionError, "Recon rotation failed"):
                self.executor.run(plan, lambda *a, **kw: None)
        self.assertIn(("stop", ()), self.robot.calls)
        self.assertFalse(any(call[0] == "tts" for call in self.robot.calls))

    @patch("robot.time.sleep")
    def test_recon_accepts_single_valid_apriltag_sighting(self, _sleep):
        self.config["recon_start_pose"]["calibrated"] = True
        sightings = iter([[[4, 100, 100]], [], []])
        self.robot.get_apriltag_total_info = lambda: next(sightings, [])
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "tag", "traffic", "color", "gesture")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda *a, **kw: None)
        self.assertEqual(self.executor._recon_results["tag"], "4")

    def test_immediate_recon_stops_speaks_and_resumes_until_full_turn(self):
        self.config["recon_start_pose"]["calibrated"] = True
        self.config["recon_announce_immediately"] = True
        events = []
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "tag", "traffic", "color", "gesture")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda event, **details: events.append((event, details)))
        announcements = [details for event, details in events if event == "announcement_complete"]
        self.assertEqual({item["kind"] for item in announcements},
                         {"word", "tag", "traffic", "color", "gesture"})
        tts_positions = [i for i, call in enumerate(self.robot.calls) if call[0] == "tts"]
        self.assertTrue(all(self.robot.calls[i - 1][0] == "stop" for i in tts_positions))
        self.assertGreaterEqual(len([call for call in self.robot.calls if call[0] == "continuous_turn"]), 2)
        self.assertEqual(len([event for event, _ in events if event == "recon_scan_start"]), 1)
        complete = [details for event, details in events if event == "recon_scan_complete"]
        self.assertEqual(len(complete), 1)
        self.assertLessEqual(complete[0]["observed_yaw_deg"], -360)
        self.assertEqual(len([call for call in self.robot.calls if call[0] == "tts"]), 5)

    def test_immediate_recon_missing_tag_still_finishes_one_turn(self):
        self.config["recon_start_pose"]["calibrated"] = True
        self.config["recon_announce_immediately"] = True
        self.robot.get_apriltag_total_info = lambda: []
        events = []
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "tag", "traffic", "color", "gesture")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda event, **details: events.append((event, details)))
        self.assertEqual(len([event for event, _ in events if event == "recon_scan_start"]), 1)
        self.assertEqual(len([event for event, _ in events if event == "recon_scan_complete"]), 1)
        self.assertEqual([details["kinds"] for event, details in events if event == "recon_missing"],
                         [["tag"]])
        self.assertEqual(len([call for call in self.robot.calls if call[0] == "tts"]), 4)

    def test_immediate_recon_announces_gesture_seen_twice(self):
        self.config["recon_start_pose"]["calibrated"] = True
        self.config["recon_announce_immediately"] = True
        gestures = iter(["剪刀", "剪刀", "no_gesture_rec"])
        self.robot.get_gesture_result = lambda: next(gestures, "no_gesture_rec")
        events = []
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "tag", "traffic", "color", "gesture")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda event, **details: events.append((event, details)))
        gesture_announcements = [details["text"] for event, details in events
                                 if event == "announcement_complete" and details["kind"] == "gesture"]
        self.assertEqual(gesture_announcements, ["手势，剪刀。"])
        self.assertEqual(self.executor._recon_results["gesture"], "剪刀")
        self.assertEqual(len([event for event, _ in events if event == "recon_scan_complete"]), 1)

    def test_immediate_recon_skips_words_longer_than_five_characters(self):
        self.config["recon_start_pose"]["calibrated"] = True
        self.config["recon_announce_immediately"] = True
        calls = 0

        def words():
            nonlocal calls
            calls += 1
            return "测试文字太长了" if calls <= 3 else "安全第一啊"

        self.robot.get_words_result = words
        events = []
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "tag", "traffic", "color", "gesture")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda event, **details: events.append((event, details)))
        word_announcements = [details["text"] for event, details in events
                              if event == "announcement_complete" and details["kind"] == "word"]
        self.assertEqual(word_announcements, ["文字，安全第一啊。"])
        self.assertEqual(self.executor._recon_results["word"], "安全第一啊")

    def test_immediate_recon_announces_each_distinct_traffic_sign(self):
        self.config["recon_start_pose"]["calibrated"] = True
        self.config["recon_announce_immediately"] = True
        calls = 0

        def traffic_signs():
            nonlocal calls
            calls += 1
            green = ["绿灯", 200, 140, 100, 100, 20000]
            red = ["红灯", 300, 140, 100, 100, 16000]
            return [green] if calls <= 3 else [green, red]

        self.robot.get_traffic_total_info = traffic_signs
        events = []
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "tag", "traffic", "color", "gesture")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda event, **details: events.append((event, details)))
        traffic = [details["text"] for event, details in events
                   if event == "announcement_complete" and details["kind"] == "traffic"]
        self.assertEqual(traffic, ["交通标志，绿灯。", "交通标志，红灯。"])
        self.assertEqual(self.executor._recon_results["traffic"], "绿灯、红灯")
        self.assertEqual(len([event for event, _ in events if event == "recon_scan_complete"]), 1)

    @patch("robot.time.sleep")
    def test_recon_turn_calibration_is_separate_from_translation(self, _sleep):
        self.config["recon_turn_calibrated"] = True
        self.config["recon_start_pose"]["calibrated"] = True
        plan = validate_plan({"scene": "recon", "actions": [
            {"type": "recognize", "kind": kind}
            for kind in ("word", "gesture", "traffic", "tag", "color")]})
        with patch.object(self.executor, "_cube_color_from_yolo", return_value="红色"):
            self.executor.run(plan, lambda *a, **kw: None)
        self.assertFalse(self.config["motion_calibrated"])
        self.assertEqual(self.executor.x_cm, self.config["recon_start_pose"]["x_cm"])
        with self.assertRaises(ExecutionError):
            self.executor.move("forward", 10)

    def test_sdk_english_recognition_labels_are_spoken_in_chinese(self):
        convert = RobotExecutor._recognition_value
        self.assertEqual(convert("gesture", "Scissors"), "剪刀")
        self.assertEqual(convert("traffic", [["green light", 100, 100]]), "绿灯")
        self.assertEqual(convert("color", ["Red", "Square"]), "红色")
        self.assertEqual(convert("color", ["no_color_rec"]), "")
        self.assertEqual(convert("color", ["绿色", "方块", 294, 267, 54, 59, 3186]), "")
        self.assertEqual(convert("color", ["红色", "方块", 371, 277, 99, 88, 8712]), "红色")
        self.assertEqual(convert("traffic", [["斑马线", 214, 275, 41, 80, 3300]]), "")
        self.assertEqual(convert("traffic", [["zebra crossing", 260, 141, 243, 161, 39000]]), "")
        self.assertEqual(convert("traffic", [["绿灯", 260, 141, 243, 161, 39000]]), "绿灯")
        self.assertEqual(convert("tag", [[4, 100, 100]]), "4")
        self.assertEqual(convert("tag", [[5, 100, 100]]), "5")
        self.assertEqual(convert("tag", [[0, 100, 100]]), "0")
        self.assertEqual(convert("tag", [[6, 100, 100]]), "")
        self.assertEqual(convert("tag", [[7, 100, 100]]), "")
        self.assertEqual(convert("tag", [["unknown", 100, 100]]), "")
        self.assertEqual(RobotExecutor._recognition_utterance("tag", "5"), "标签，五号。")

    def test_arm_wave_commands_more_than_three_seconds_of_motion(self):
        with patch.object(self.executor, "_arm_pose"):
            self.executor.arm_wave()
        swings = [call for call in self.robot.calls if call[0] == "servo"]
        self.assertEqual(len(swings), 6)
        self.assertGreaterEqual(sum(call[1][0][2] for call in swings), 3000)
        self.assertTrue(all(call[1][1] == {"wait": True} for call in swings))
        self.assertEqual(self.robot.calls[-1][0], "servo_stop")

    def test_late_turn_rpc_uses_verified_imu_after_confirmed_stop(self):
        self.executor.scene = "recon"
        self.config["recon_start_pose"]["calibrated"] = True
        def late_rpc(_command, _expected, observe=None):
            self.robot.yaw = -60
            raise ExecutionError("Motion exceeded watchdog deadline; chassis stopped")
        with patch.object(self.executor, "_bounded_motion", side_effect=late_rpc):
            self.executor.turn("right", 60)
        self.assertEqual(self.executor.heading_deg, 60)

    def test_sdk_error_after_verified_turn_and_stop_is_accepted(self):
        self.executor.scene = "recon"
        self.config["recon_start_pose"]["calibrated"] = True

        def late_sdk(*_args):
            self.robot.yaw = -60
            raise RuntimeError("SDK deadline")

        self.robot.mecanum_turn_speed_times = late_sdk
        self.executor.turn("right", 60)
        self.assertEqual(self.executor.heading_deg, 60)
        self.assertIn(("stop", ()), self.robot.calls)

    def test_late_turn_rpc_without_matching_imu_still_fails(self):
        self.executor.scene = "recon"
        self.config["recon_start_pose"]["calibrated"] = True
        with patch.object(self.executor, "_bounded_motion",
                          side_effect=ExecutionError("Motion exceeded watchdog deadline; chassis stopped")):
            with self.assertRaises(ExecutionError):
                self.executor.turn("right", 60)

    @patch("robot.time.sleep")
    def test_failed_motion_does_not_advance_predicted_pose(self, _sleep):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        self.robot.mecanum_move_speed_times = lambda *args: (_ for _ in ()).throw(RuntimeError("RPC failed"))
        with self.assertRaises(RuntimeError):
            self.executor.move("forward", 10)
        self.assertEqual((self.executor.x_cm, self.executor.y_cm, self.executor.path_cm), (75, 80, 0))
        self.assertIn(("stop", ()), self.robot.calls)

    @patch("robot.time.sleep")
    def test_imu_turn_mismatch_does_not_advance_heading(self, _sleep):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        self.robot.mecanum_turn_speed_times = lambda *args: self.robot.calls.append(("turn", args))
        with self.assertRaisesRegex(ExecutionError, "IMU mismatch"):
            self.executor.turn("left", 90)
        self.assertEqual(self.executor.heading_deg, 0)
        self.assertIn(("stop", ()), self.robot.calls)

    def test_blocking_full_turn_still_accumulates_imu_yaw(self):
        self.config["motion_calibrated"] = True
        self.config["start_pose"]["calibrated"] = True
        state = {"yaw": 0.0}
        self.robot.read_gyro_data = lambda: [0, 0, state["yaw"] % 360]
        def blocking_turn(*_args):
            for _ in range(6):
                state["yaw"] -= 60
                time.sleep(0.12)
        self.robot.mecanum_turn_speed_times = blocking_turn
        with patch.object(self.executor, "_bounded_motion", side_effect=lambda command, *_args, **_kw: command()):
            self.executor.turn("right", 360)
        self.assertEqual(self.executor.heading_deg, 0)


if __name__ == "__main__": unittest.main()
