"""One supervised chassis movement and an optional measured-result record."""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

from robot import connect
from telemetry import AngleAccumulator

ROOT = Path(__file__).resolve().parent
WHEEL_IDS = (11, 31, 41, 61)


def wheel_angles(robot: object) -> dict[int, float]:
    return {sid: float(robot.read_motor_angle(sid)[str(sid)]) for sid in WHEEL_IDS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run exactly one UGOT straight or strafe calibration move")
    parser.add_argument("--axis", choices=("straight", "strafe"), required=True)
    parser.add_argument("--direction", choices=("forward", "backward", "left", "right"), required=True)
    parser.add_argument("--cm", type=int, choices=(10, 30, 50), required=True)
    parser.add_argument("--speed", type=int, default=12)
    parser.add_argument("--ip", help="UGOT IP; defaults to config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.axis == "straight" and args.direction not in {"forward", "backward"}:
        parser.error("straight requires forward or backward")
    if args.axis == "strafe" and args.direction not in {"left", "right"}:
        parser.error("strafe requires left or right")
    if not 5 <= args.speed <= 40:
        parser.error("--speed must be 5..40 cm/s")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    ip = args.ip or cfg["robot_ip"]
    if args.axis == "straight":
        method = "mecanum_move_speed_times"
        sdk_args = (0 if args.direction == "forward" else 1, args.speed, args.cm, 1)
    else:
        method = "mecanum_translate_speed_times"
        sdk_args = (-90 if args.direction == "left" else 90, args.speed, args.cm, 1)
    print(f"UGOT={ip}; {method}{sdk_args}")
    if args.dry_run:
        return 0
    print("Place the robot in a clear area. Mark its starting center and keep a hand near emergency stop.")
    if input("Type MOVE to run exactly one movement: ").strip() != "MOVE":
        print("Cancelled; no movement command sent")
        return 1
    robot = connect(ip)
    start_wheels = wheel_angles(robot)
    wheels = {sid: AngleAccumulator(value) for sid, value in start_wheels.items()}
    yaw_start = float(robot.read_gyro_data()[2])
    observe_s = max(2.5, args.cm / args.speed + 1.5)
    timeout = observe_s + 3
    expired = threading.Event()
    def emergency_stop() -> None:
        expired.set()
        robot.stop_chassis()
    watchdog = threading.Timer(timeout, emergency_stop)
    watchdog.daemon = True
    watchdog.start()
    started = time.monotonic()
    try:
        getattr(robot, method)(*sdk_args)
        deadline = started + observe_s
        while time.monotonic() < deadline and not expired.is_set():
            for sid, angle in wheel_angles(robot).items():
                wheels[sid].add(angle)
            time.sleep(0.1)
        if expired.is_set():
            raise RuntimeError("Movement watchdog stopped chassis")
    finally:
        watchdog.cancel()
        robot.stop_chassis()
    for sid, angle in wheel_angles(robot).items():
        wheels[sid].add(angle)
    yaw_end = float(robot.read_gyro_data()[2])
    print("Wheel angle changes (degrees, diagnostic only):", {sid: round(w.total, 1) for sid, w in wheels.items()})
    print(f"IMU yaw before/after: {yaw_start:.1f}° / {yaw_end:.1f}°")
    measured = input("Measured displacement in cm (blank to record later): ").strip()
    actual = None if not measured else float(measured)
    row = {"time": datetime.now().astimezone().isoformat(), "ip": ip,
           "axis": args.axis, "direction": args.direction, "command_cm": args.cm,
           "speed_cm_s": args.speed, "actual_cm": actual, "watchdog_s": timeout,
           "wheel_delta_deg": {str(sid): round(w.total, 2) for sid, w in wheels.items()},
           "imu_yaw_start_deg": yaw_start, "imu_yaw_end_deg": yaw_end}
    path = ROOT / "logs" / "motion_calibration.jsonl"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Recorded in {path}. Config calibration flags remain unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
