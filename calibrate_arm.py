"""Run exactly one stationary arm wave for the math scene."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from robot import RobotExecutor, connect

ROOT = Path(__file__).resolve().parent
SERVO_IDS = (51, 52, 53)


def read_angles(robot: object) -> dict[str, int]:
    result = {}
    for sid in SERVO_IDS:
        raw = robot.read_servo_angle(sid) or {}
        if str(sid) not in raw:
            raise RuntimeError(f"Servo {sid} readback unavailable")
        result[str(sid)] = int(raw[str(sid)])
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", help="UGOT IP; defaults to config.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    ip = args.ip or cfg["robot_ip"]
    pose = cfg["grasp"]["carry_arm_pose"]
    print(f"UGOT={ip}; carry pose={pose}; wave servo 51 by ±12° for three 600 ms cycles")
    if args.dry_run:
        return 0
    print("Clear objects and hands from the mechanical arm's full reach. Keep the robot stationary.")
    if input("Type ARM to run exactly one wave: ").strip() != "ARM":
        print("Cancelled; no servo command sent")
        return 1
    robot = connect(ip)
    try:
        before = read_angles(robot)
        started = time.monotonic()
        RobotExecutor(robot, cfg, None).arm_wave()
        elapsed = time.monotonic() - started
        after = read_angles(robot)
        row = {"time": datetime.now().astimezone().isoformat(), "ip": ip,
               "before_deg": before, "after_deg": after, "elapsed_s": round(elapsed, 2),
               "commanded_wave_motion_s": 3.6}
        path = ROOT / "logs" / "arm_calibration.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False))
        print(f"Recorded in {path}")
        return 0
    finally:
        robot.stop_servo(51, lock=True)
        robot.stop_chassis()


if __name__ == "__main__":
    raise SystemExit(main())
