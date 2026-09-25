"""Inspect one UGOT reconnaissance model without moving or speaking."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from robot import RobotExecutor, connect

ROOT = Path(__file__).resolve().parent
MODELS = {
    "word": ("word_recognition", "get_words_result"),
    "gesture": ("gesture", "get_gesture_result"),
    "traffic": ("traffic_sign", "get_traffic_total_info"),
    "tag": ("apriltag_qrcode", "get_apriltag_total_info"),
    "color": ("color_recognition", "get_color_total_info"),
}


def bounded_cleanup(label: str, command, timeout_s: float = 4) -> bool:
    """Do not let an unresponsive SDK cleanup RPC hold the test open."""
    finished = threading.Event()
    errors: list[Exception] = []

    def run() -> None:
        try:
            command()
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    threading.Thread(target=run, daemon=True).start()
    if not finished.wait(timeout_s):
        print(f"Warning: {label} did not finish within {timeout_s:g}s", file=sys.stderr)
        return False
    if errors:
        print(f"Warning: {label} failed: {errors[0]!r}", file=sys.stderr)
        return False
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=MODELS, required=True)
    parser.add_argument("--ip", help="UGOT IP; defaults to config.json")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--save-frame", action="store_true", help="Save the camera view beside the sample log")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 20:
        parser.error("--seconds must be 1..20")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    ip = args.ip or cfg["robot_ip"]
    name, getter_name = MODELS[args.kind]
    robot = connect(ip)
    path = ROOT / "logs" / "recon_static.jsonl"
    path.parent.mkdir(exist_ok=True)
    sample_count = 0
    cleanup_ok = True
    try:
        robot.stop_chassis()
        robot.open_camera()
        if not robot.load_models([name]):
            raise RuntimeError(f"Could not load {name}")
        if args.save_frame:
            import cv2
            import numpy as np
            frame_bytes = robot.read_camera_data()
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR) if frame_bytes else None
            if frame is None:
                raise RuntimeError("Could not decode reconnaissance camera frame")
            frame_path = ROOT / "logs" / f"recon_{args.kind}_live_view.png"
            frame_path.parent.mkdir(exist_ok=True)
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError("Could not encode reconnaissance camera frame")
            encoded.tofile(str(frame_path))
            print(f"Saved camera view in {frame_path}")
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            raw = getattr(robot, getter_name)()
            value = RobotExecutor._recognition_value(args.kind, raw)
            row = {"elapsed_s": round(args.seconds - (deadline - time.monotonic()), 2),
                   "kind": args.kind, "value": value, "raw": raw}
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time": datetime.now().astimezone().isoformat(),
                                         "ip": ip, **row}, ensure_ascii=False, default=str) + "\n")
            sample_count += 1
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
            time.sleep(0.25)
    finally:
        stopped = bounded_cleanup("stop_chassis", robot.stop_chassis)
        released = bounded_cleanup("release_models", lambda: robot.release_models([name]))
        cleanup_ok = stopped and released
    print(f"Recorded {sample_count} samples in {path}")
    return 0 if cleanup_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
