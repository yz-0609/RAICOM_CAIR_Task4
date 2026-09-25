"""Read-only Tag 0 and infrared comparison for the Task 4 warehouse setup."""

from __future__ import annotations

import argparse
import json
import time

from robot import connect
from task4 import load_config, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "tmp" / "warehouse_trial.json"))
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    from pathlib import Path

    cfg = load_config(Path(args.config))
    robot = connect(cfg["robot_ip"])
    sensors = [int(item["deviceId"]) for item in robot.get_peripheral_devices_list()
               if item.get("type") == "Infrared"]
    if not sensors:
        raise RuntimeError("No infrared sensor found")
    sensor_id = sensors[0]
    robot.load_models(["apriltag_qrcode"])
    try:
        for _ in range(args.samples):
            tags = robot.get_apriltag_total_info() or []
            tag = next((item for item in tags if item and item[0] == 0), None)
            row = {"infrared_cm": robot.read_distance_data(sensor_id),
                   "tag0": None if tag is None else {
                       "center_x": tag[1], "center_y": tag[2],
                       "width_px": tag[4], "area_px2": tag[5],
                       "vision_distance_5cm": tag[6],
                       "vision_distance_7cm": tag[7],
                       "vision_distance_10cm": tag[8]}}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(0.3)
    finally:
        robot.release_models(["apriltag_qrcode"])


if __name__ == "__main__":
    main()
