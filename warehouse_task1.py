"""Warehouse-only adapter for the untouched, field-tested Task 1 routines.

The original Task 1 source is loaded read-only. Its YOLO tracking, grab pose,
AprilTag 0 chase, and lane perception remain the source of control logic.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
SIBLING_TASK1 = (HERE.parent / "TASK1_code_d8ad863_队友交接_2026-09-22"
                 / "TASK1_code_d8ad863" / "task1_country.py")
BUNDLED_TASK1 = HERE / "vendor" / "task1" / "task1_country.py"
DEFAULT_TASK1 = SIBLING_TASK1 if SIBLING_TASK1.is_file() else BUNDLED_TASK1


class WarehouseTask1Error(RuntimeError):
    pass


def load_task1(source: str | Path | None = None) -> Any:
    path = Path(source) if source else DEFAULT_TASK1
    if not path.is_file():
        raise WarehouseTask1Error(f"Task 1 source is missing: {path}")
    spec = importlib.util.spec_from_file_location("_task4_readonly_task1", path)
    if spec is None or spec.loader is None:
        raise WarehouseTask1Error(f"Cannot load Task 1 source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Keep the source untouched, but use Task 4's copied detector asset.
    module.MODEL_PATH = str(HERE / "models" / "best.pt")
    module.LANE_MODEL_PATH = HERE / "models" / "line_seg_mnv3.onnx"
    return module


def track_and_grab(robot: Any, source: str | Path | None, sensor_id: int,
                   color: str) -> None:
    task1 = load_task1(source)
    # Call the verified Task 1 routine with the original UGOT object. Task 1
    # owns its camera/control threads, live preview, and emergency stop here.
    reached = task1.track_and_grab_phase(robot, color, sensor_id)
    if reached is not True:
        raise WarehouseTask1Error(f"Task 1 did not confirm grabbing the {color} cube")


def chase_tag_zero(robot: Any, source: str | Path | None, sensor_id: int) -> None:
    task1 = load_task1(source)
    if not robot.load_models(["apriltag_qrcode"]):
        raise WarehouseTask1Error("Could not load UGOT AprilTag model")
    try:
        reached = task1._chase_apriltag(
            robot, task1.TARGET_TAG_ID, task1.APRILTAG_TARGET_DISTANCE, sensor_id)
        if reached is not True:
            raise WarehouseTask1Error("Task 1 did not confirm reaching AprilTag 0")
    finally:
        robot.stop_chassis()
        robot.release_models(["apriltag_qrcode"])


def follow_pickup_line(robot: Any, frame: Callable[[], Any], source: str | Path | None,
                       *, timeout_s: float = 12, max_travel_cm: float = 15,
                       min_travel_cm: float = 4, confidence: float = 0.55) -> float:
    """Use Task 1 lane perception/controller for a short, bounded line handoff."""
    task1 = load_task1(source)
    perception = task1.LanePerception(model_path=HERE / "models" / "line_seg_mnv3.onnx")
    follower = task1.LaneFollower()
    start = time.monotonic()
    last = start
    travelled = 0.0
    stable = 0
    try:
        while time.monotonic() - start < timeout_s and travelled < max_travel_cm:
            observation = perception.analyze(frame())
            now = time.monotonic()
            travelled += min(0.5, now - last) * min(12, max(0, follower.last_command.forward_speed))
            last = now
            good = not observation.lost and observation.confidence >= confidence
            stable = stable + 1 if good else 0
            if stable >= 3 and travelled >= min_travel_cm:
                return travelled
            command = follower.step(observation)
            if command.mode != task1.DriveMode.TRACK:
                robot.stop_chassis()
            else:
                # Task 1 limits motor commands to 5 Hz to avoid camera/RPC contention.
                robot.mecanum_move_xyz(0, min(12, command.forward_speed),
                                       max(-18, min(18, command.yaw_speed)))
            time.sleep(0.2)
    finally:
        robot.stop_chassis()
    raise WarehouseTask1Error("Pickup line was not confirmed within the bounded handoff")
