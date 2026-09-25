import unittest
from unittest.mock import patch
import json

from planning import CloudLLM, PlanError, parse_recon_plan, validate_plan


class PlanningTests(unittest.TestCase):
    def test_rule_examples(self):
        samples = [
            {"scene": "warehouse", "actions": [
                {"type": "pickup", "color": "red"}, {"type": "place", "position": 3},
                {"type": "move", "direction": "backward", "cm": 10},
                {"type": "turn", "direction": "left", "degrees": 90}, {"type": "return_line"}]},
            {"scene": "companion", "actions": [
                {"type": "conversation", "prompt": "聊聊天"}, {"type": "emotion", "name": "Anger"},
                {"type": "sound", "name": "elephant"}, {"type": "sound", "name": "tiger"},
                {"type": "light", "color": "red"}]},
            {"scene": "recon", "actions": [
                {"type": "recognize", "kind": kind}
                for kind in ("word", "gesture", "traffic", "tag", "color")]},
            {"scene": "math", "actions": [
                {"type": "polygon", "sides": 4, "side_cm": 30, "direction": "clockwise", "travel": "forward"},
                {"type": "strafe", "direction": "right", "cm": 30},
                {"type": "polygon", "sides": 5, "side_cm": 30, "direction": "counterclockwise", "travel": "forward"},
                {"type": "arm_wave"}, {"type": "spin", "direction": "right"}]},
        ]
        for sample in samples:
            with self.subTest(sample["scene"]):
                self.assertEqual(validate_plan(sample).as_dict(), sample)

    def test_no_extra_action_or_code(self):
        sample = {"scene": "math", "actions": [{"type": "spin", "direction": "right"}] * 6}
        with self.assertRaises(PlanError):
            validate_plan(sample)
        sample["actions"] = [{"type": "spin", "direction": "right", "code": "eval('x')"}] * 5
        with self.assertRaises(PlanError):
            validate_plan(sample)

    def test_recon_order_preserved(self):
        order = ("traffic", "color", "tag", "word", "gesture")
        data = {"scene": "recon", "actions": [{"type": "recognize", "kind": x} for x in order]}
        self.assertEqual([a.args["kind"] for a in validate_plan(data).actions], list(order))

    def test_recon_spoken_order_ignores_prop_list_and_expected_values(self):
        command = ("你是一名侦探，面前摆放有色块、标签、文字、手势、交通标志，"
                   "请识别并按照文字（你好）、手势（剪刀）、交通标志（绿灯）、"
                   "标签（4号）、色块颜色（红色）的顺序播报。")
        plan = parse_recon_plan(command)
        self.assertEqual([action.args["kind"] for action in plan.actions],
                         ["word", "gesture", "traffic", "tag", "color"])

    def test_recon_spoken_order_without_answer_values(self):
        command = ("你现在是一名侦探，需要识别并播报你看到的交通标志，"
                   "识别并播报手势、文字、标签号和色块颜色。")
        plan = parse_recon_plan(command)
        self.assertEqual([action.args["kind"] for action in plan.actions],
                         ["traffic", "gesture", "word", "tag", "color"])

    def test_recon_rule_example_does_not_require_literal_recognize_word(self):
        command = ("你是一名侦探，面前摆放有色块、标签、文字、手势、交通标志，"
                   "现在把看到的内容按照文字、手势、交通标志、标签、色块颜色的顺序播报。")
        plan = parse_recon_plan(command)
        self.assertEqual([action.args["kind"] for action in plan.actions],
                         ["word", "gesture", "traffic", "tag", "color"])

    def test_recon_live_asr_handbook_homophone_preserves_five_step_order(self):
        command = "你是一名侦探，请识别到你看到的内容，并按照文字标签交通标志色块手册的顺序播报。"
        plan = parse_recon_plan(command)
        self.assertEqual([action.args["kind"] for action in plan.actions],
                         ["word", "tag", "traffic", "color", "gesture"])

    def test_recon_handbook_alone_is_not_a_gesture(self):
        command = "你是一名侦探，请识别并播报这本手册上的文字和标签。"
        self.assertIsNone(parse_recon_plan(command))

    def test_invalid_motion_rejected(self):
        data = {"scene": "math", "actions": [{"type": "turn", "direction": "left", "degrees": 400}] * 5}
        with self.assertRaises(PlanError):
            validate_plan(data)

    def test_warehouse_pickup_precedes_place(self):
        data = {"scene": "warehouse", "actions": [
            {"type": "place", "position": 3}, {"type": "pickup", "color": "red"},
            {"type": "move", "direction": "backward", "cm": 10},
            {"type": "turn", "direction": "left", "degrees": 90}, {"type": "return_line"}]}
        with self.assertRaises(PlanError): validate_plan(data)

    def test_cloud_result_goes_through_local_validator(self):
        with patch.dict("os.environ", {"TASK4_LLM_BASE_URL": "https://example.invalid/v1", "TASK4_LLM_MODEL": "test", "TASK4_LLM_API_KEY": "test"}):
            llm = CloudLLM()
        bad = {"scene": "warehouse", "actions": [{"type": "pickup", "color": "red", "code": "unsafe"}] * 5}
        with patch.object(llm, "chat", return_value=json.dumps(bad)):
            with self.assertRaises(PlanError):
                llm.plan("抓取红色")

    def test_math_instruction_uses_math_scoring_prompt(self):
        with patch.dict("os.environ", {"TASK4_LLM_BASE_URL": "https://example.invalid/v1", "TASK4_LLM_MODEL": "test", "TASK4_LLM_API_KEY": "test"}):
            llm = CloudLLM()
        plan = {"scene": "math", "actions": [
            {"type": "polygon", "sides": 4, "side_cm": 30, "direction": "clockwise", "travel": "forward"},
            {"type": "strafe", "direction": "left", "cm": 10},
            {"type": "strafe", "direction": "right", "cm": 10},
            {"type": "polygon", "sides": 4, "side_cm": 30, "direction": "clockwise", "travel": "forward"},
            {"type": "arm_wave"}]}
        with patch.object(llm, "chat", return_value=json.dumps(plan)) as chat:
            self.assertEqual(llm.plan("走一个四边形，左右平移，再走四边形，动动手").as_dict(), plan)
        system = chat.call_args.args[0]
        self.assertIn('scene to "math"', system)
        self.assertIn("arm_wave EACH count", system)


if __name__ == "__main__":
    unittest.main()
