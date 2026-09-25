"""Pre-competition single-turn calibration; never used by task4.py."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="One bounded UGOT turn for measuring real angle")
    parser.add_argument("--direction", choices=("left", "right"), required=True)
    parser.add_argument("--degrees", type=int, choices=(45, 90, 180, 360), required=True)
    parser.add_argument("--speed", type=int, default=20, help="Degrees per second, 5 to 40")
    parser.add_argument("--ip", help="UGOT IP; defaults to config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 5 <= args.speed <= 40:
        parser.error("--speed must be between 5 and 40 degrees per second")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    ip = args.ip or cfg["robot_ip"]
    sdk_args = (2 if args.direction == "left" else 3, args.speed, args.degrees, 2)
    print(f"UGOT={ip}; direction={args.direction}; target={args.degrees}°; speed={args.speed}°/s")
    print(f"SDK mecanum_turn_speed_times{sdk_args}; final 2 means degrees")
    if args.dry_run:
        return 0
    print("Place the robot at the center of a clear test area. Mark its initial heading.")
    if input("Type TURN to run exactly one turn: ").strip() != "TURN":
        print("Cancelled; no movement command sent")
        return 1
    robot = connect(ip)
    yaw_start = float(robot.read_gyro_data()[2])
    yaw = AngleAccumulator(yaw_start)
    observe_s = max(2.5, args.degrees / args.speed + 1.5)
    expired = threading.Event()
    def emergency_stop() -> None:
        expired.set()
        robot.stop_chassis()
    watchdog = threading.Timer(observe_s + 3, emergency_stop)
    watchdog.daemon = True
    watchdog.start()
    started = time.monotonic()
    sampler_stop = threading.Event()
    sampler_error: list[Exception] = []
    def sample_during_blocking_rpc() -> None:
        while not sampler_stop.wait(0.1):
            try:
                yaw.add(robot.read_gyro_data()[2])
            except Exception as error:
                sampler_error.append(error)
                robot.stop_chassis()
                return
    sampler = threading.Thread(target=sample_during_blocking_rpc, daemon=True)
    sampler.start()
    try:
        try:
            robot.mecanum_turn_speed_times(*sdk_args)
        finally:
            sampler_stop.set()
            sampler.join(timeout=1)
        deadline = started + observe_s
        while time.monotonic() < deadline and not expired.is_set():
            yaw.add(robot.read_gyro_data()[2])
            time.sleep(0.1)
        if sampler_error:
            raise RuntimeError("IMU sampling failed during turn") from sampler_error[0]
        if expired.is_set():
            raise RuntimeError("Turn watchdog stopped chassis")
    finally:
        sampler_stop.set()
        sampler.join(timeout=1)
        watchdog.cancel()
        robot.stop_chassis()
    yaw.add(robot.read_gyro_data()[2])
    print(f"IMU yaw start={yaw_start:.1f}°; accumulated change={yaw.total:.1f}°")
    print("Compare the IMU result with the start heading mark; record any visible slip or drift.")
    row = {"time": datetime.now().astimezone().isoformat(), "ip": ip, "direction": args.direction,
           "command_deg": args.degrees, "speed_deg_s": args.speed, "imu_delta_deg": round(yaw.total, 2)}
    path = ROOT / "logs" / "turn_calibration.jsonl"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Recorded in {path}; motion calibration flags remain unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
