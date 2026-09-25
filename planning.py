"""Turn a spoken Task 4 instruction into a bounded, ordered action plan."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests


class PlanError(ValueError):
    pass


ALLOWED = {
    "warehouse": {"move", "turn", "spin", "pickup", "place", "return_line", "strafe", "recognize"},
    "companion": {"conversation", "emotion", "sound", "light", "screen_text"},
    "recon": {"recognize"},
    "math": {"polygon", "move", "turn", "spin", "strafe", "arm_wave"},
}
COLORS = {"red", "green", "blue"}
SOUNDS = {
    "bear", "bird", "chicken", "cow", "dog", "elephant", "giraffe", "horse",
    "lion", "monkey", "pig", "rhinoceros", "sealions", "tiger", "walrus",
    "ambulance", "car horn", "police car 1", "police car 2", "robot",
    "happy", "angry", "surprise", "received", "complete",
}
EMOTIONS = {"WakeUp", "Smile", "Doubt", "Search", "Breathe", "Blink", "Resist", "Love", "Anger", "Proud", "Ticklish", "Weakness", "Sleep", "SleepCirculate", "Switch"}
RECOGNITIONS = {"word", "gesture", "traffic", "tag", "color"}
ACTION_KEYS = {
    "move": {"direction", "cm"}, "strafe": {"direction", "cm"},
    "turn": {"direction", "degrees"}, "spin": {"direction"},
    "pickup": {"color"}, "place": {"position"}, "return_line": set(),
    "conversation": {"prompt"}, "emotion": {"name"}, "sound": {"name"},
    "light": {"color"}, "screen_text": {"source"},
    "recognize": {"kind"}, "polygon": {"sides", "side_cm", "direction", "travel"},
    "arm_wave": set(),
}


@dataclass(frozen=True)
class Action:
    kind: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Plan:
    scene: str
    actions: tuple[Action, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"scene": self.scene, "actions": [{"type": a.kind, **a.args} for a in self.actions]}


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise PlanError(f"{name} must be in [{low}, {high}]")
    return float(value)


def validate_plan(data: Any) -> Plan:
    if not isinstance(data, dict) or set(data) != {"scene", "actions"}:
        raise PlanError("Plan must contain only scene and actions")
    scene = data["scene"]
    if scene not in ALLOWED:
        raise PlanError(f"Unsupported scene: {scene}")
    raw = data["actions"]
    if not isinstance(raw, list) or len(raw) != 5:
        raise PlanError("Exactly five scored actions are required")
    actions = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise PlanError(f"Action {index} is not an object")
        kind = item.get("type")
        if kind not in ALLOWED[scene]:
            raise PlanError(f"Action {index}: {kind} is not allowed in {scene}")
        expected = ACTION_KEYS[kind] | {"type"}
        if set(item) != expected:
            raise PlanError(f"Action {index}: expected fields {sorted(expected)}")
        args = {k: v for k, v in item.items() if k != "type"}
        if kind == "move":
            if args["direction"] not in {"forward", "backward"}: raise PlanError("Invalid move direction")
            _number(args["cm"], "cm", 1, 50)
        elif kind == "strafe":
            if args["direction"] not in {"left", "right"}: raise PlanError("Invalid strafe direction")
            _number(args["cm"], "cm", 1, 50)
        elif kind == "turn":
            if args["direction"] not in {"left", "right"}: raise PlanError("Invalid turn direction")
            _number(args["degrees"], "degrees", 1, 360)
        elif kind == "spin":
            if args["direction"] not in {"left", "right"}: raise PlanError("Invalid spin direction")
        elif kind == "pickup":
            if args["color"] not in COLORS: raise PlanError("Invalid pickup color")
        elif kind == "place":
            if isinstance(args["position"], bool) or not isinstance(args["position"], int) or not 1 <= args["position"] <= 6:
                raise PlanError("Position must be an integer from 1 to 6")
        elif kind == "conversation":
            if not isinstance(args["prompt"], str) or not 1 <= len(args["prompt"]) <= 160:
                raise PlanError("Invalid conversation prompt")
        elif kind == "emotion":
            if args["name"] not in EMOTIONS: raise PlanError("Unsupported SDK emotion")
        elif kind == "sound":
            if args["name"] not in SOUNDS: raise PlanError("Unsupported SDK sound")
        elif kind == "light":
            if args["color"] not in COLORS: raise PlanError("Unsupported light color")
        elif kind == "screen_text":
            if args["source"] != "last_reply": raise PlanError("Screen text must show last model reply")
        elif kind == "recognize":
            if args["kind"] not in RECOGNITIONS: raise PlanError("Unsupported recognition kind")
        elif kind == "polygon":
            if isinstance(args["sides"], bool) or not isinstance(args["sides"], int) or not 3 <= args["sides"] <= 8:
                raise PlanError("Polygon must have 3 to 8 sides")
            _number(args["side_cm"], "side_cm", 1, 50)
            if args["direction"] not in {"clockwise", "counterclockwise"} or args["travel"] not in {"forward", "backward"}:
                raise PlanError("Invalid polygon direction or travel")
        actions.append(Action(kind, args))
    if scene == "recon" and {a.args["kind"] for a in actions} != RECOGNITIONS:
        raise PlanError("Recon must recognize each of the five categories exactly once")
    if scene == "warehouse":
        kinds = [a.kind for a in actions]
        if kinds.count("pickup") != 1 or kinds.count("place") != 1 or kinds.index("pickup") > kinds.index("place"):
            raise PlanError("Warehouse requires one pickup before one place")
    if scene == "companion":
        seen_reply = False
        for action in actions:
            if action.kind == "conversation": seen_reply = True
            if action.kind == "screen_text" and not seen_reply:
                raise PlanError("Screen text needs a previous conversation reply")
    return Plan(scene, tuple(actions))


def parse_recon_plan(command: str) -> Plan | None:
    """Read the five recognition categories in their spoken scoring order."""
    if not re.search(r"侦探|侦察|侦查", command) or "播报" not in command:
        return None
    explicit_orders = re.findall(r"按(?:照)?(.{0,160}?)的?顺序", command)
    ordered_text = explicit_orders[-1] if explicit_orders else command
    # On-site ASR transcribed the final category 手势 as 手册. Only correct it
    # inside an explicit five-category reconnaissance order.
    if explicit_orders and "手势" not in ordered_text and "手册" in ordered_text:
        other_terms = (r"文字|文本", r"交通标志|交通信号", r"标签|AprilTag", r"色块|颜色")
        if all(re.search(pattern, ordered_text, re.IGNORECASE) for pattern in other_terms):
            ordered_text = ordered_text.replace("手册", "手势")
    terms = {
        "word": r"文字|文本",
        "gesture": r"手势",
        "traffic": r"交通标志|交通信号",
        "tag": r"标签|AprilTag",
        "color": r"色块|颜色",
    }
    positions = []
    for kind, pattern in terms.items():
        match = re.search(pattern, ordered_text, re.IGNORECASE)
        if match is None:
            return None
        positions.append((match.start(), kind))
    order = [kind for _, kind in sorted(positions)]
    return validate_plan({"scene": "recon", "actions": [
        {"type": "recognize", "kind": kind} for kind in order]})


SYSTEM_PROMPT = """You parse a Chinese UGOT robot competition command into exactly five scored actions.
Return ONLY a JSON object with scene and actions in the spoken scoring order. No markdown, no prose.
Every action object MUST have a "type" key naming the action; never use "action", "kind", or "name" as its discriminator.
Exact example for a five-step warehouse command:
{"scene":"warehouse","actions":[{"type":"move","direction":"forward","cm":10},{"type":"pickup","color":"red"},{"type":"place","position":3},{"type":"move","direction":"backward","cm":10},{"type":"return_line"}]}.
scene: warehouse | companion | recon | math.
Actions and exact fields:
move(direction: forward|backward, cm: 1..50), strafe(direction: left|right, cm: 1..50),
turn(direction: left|right, degrees: 1..360), spin(direction: left|right),
pickup(color: red|green|blue), place(position: integer 1..6), return_line(),
conversation(prompt: short topic), emotion(name: SDK name), sound(name: SDK sound),
light(color: red|green|blue), screen_text(source: last_reply),
recognize(kind: word|gesture|traffic|tag|color),
polygon(sides: 3..8, side_cm: 1..50, direction: clockwise|counterclockwise, travel: forward|backward), arm_wave().
SDK emotions: WakeUp, Smile, Doubt, Search, Breathe, Blink, Resist, Love, Anger, Proud, Ticklish, Weakness, Sleep, SleepCirculate, Switch.
SDK sounds: bear, bird, chicken, cow, dog, elephant, giraffe, horse, lion, monkey, pig, rhinoceros, sealions, tiger, walrus, ambulance, car horn, police car 1, police car 2, robot, happy, angry, surprise, received, complete.
Use two conversation actions if the command asks for two dialogues. Never put expected recognition content into recognize; the robot must observe it. A 360-degree circle is spin. Do not invent actions. If the instruction does not determine five supported scored actions, return {\"error\":\"reason\"}."""


MATH_PROMPT = SYSTEM_PROMPT + """
This command describes the intelligent math scene. Set scene to "math" and preserve the five scored actions in spoken order.
For this scene, polygon, move, strafe, turn, spin, and arm_wave EACH count as one scored action.
"动动手", "动动小手", "动一动发财小手", and a three-second custom arm movement mean arm_wave; never omit this scored action.
"左平移 Z，右平移 Z" means two separate strafe actions, in that order. If the command then repeats the polygon and ends with an arm movement, the five actions are polygon, strafe left, strafe right, polygon, arm_wave.
"倒退走一个顺时针四边形" means polygon(sides:4, direction:clockwise, travel:backward).
"转个圈圈" means spin; if no direction is given for this spin, use right. Use the spoken polygon side count, edge length, travel direction, and action order, not numbers from any example.
If the spoken command does not determine exactly five supported math actions, return an error object instead of inventing actions."""


class CloudLLM:
    def __init__(self) -> None:
        self.base_url = os.environ.get("TASK4_LLM_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("TASK4_LLM_MODEL", "")
        self.key = os.environ.get("TASK4_LLM_API_KEY", "")
        if not all((self.base_url, self.model, self.key)):
            raise PlanError("Set TASK4_LLM_BASE_URL, TASK4_LLM_MODEL and TASK4_LLM_API_KEY")

    def chat(self, system: str, user: str, *, temperature: float = 0.0,
             json_mode: bool = False) -> str:
        payload: dict[str, Any] = {"model": self.model, "temperature": temperature,
                                   "messages": [{"role": "system", "content": system},
                                                {"role": "user", "content": user}]}
        if urlparse(self.base_url).hostname == "api.deepseek.com":
            # DeepSeek enables reasoning by default. With a bounded JSON reply,
            # reasoning can consume every token and leave message.content empty.
            payload["thinking"] = {"type": "disabled"}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            payload["max_tokens"] = 1024
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(5, 25),
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise PlanError("Model response exceeded its output token limit")
        content = choice["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise PlanError("Model returned an empty response")
        return content

    def plan(self, command: str) -> Plan:
        recon = parse_recon_plan(command)
        if recon is not None:
            return recon
        system = MATH_PROMPT if re.search(r"边形|正方形|长方形|三角形", command) else SYSTEM_PROMPT
        result = self.chat(system, command, json_mode=True)
        result = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip(), flags=re.IGNORECASE)
        try:
            data = json.loads(result)
        except json.JSONDecodeError as error:
            raise PlanError("Model returned invalid JSON") from error
        if isinstance(data, dict) and "error" in data:
            raise PlanError(str(data["error"]))
        return validate_plan(data)

    def answer(self, question: str) -> str:
        answer = self.chat("你是友善的陪伴机器人。请用简短、自然的中文回答，最多60字。", question, temperature=0.4).strip()
        if not answer or len(answer) > 180:
            raise PlanError("Invalid companion answer")
        return answer
