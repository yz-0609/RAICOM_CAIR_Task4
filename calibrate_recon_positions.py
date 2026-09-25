"""Pause at reference headings for placing props; the live scan rotates continuously."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from robot import RobotExecutor, connect


ROOT = Path(__file__).resolve().parent


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tmp" / "recon_site_verified.json")
    parser.add_argument("--dry-run", action="store_true", help="Show stops without connecting or moving")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    headings = cfg["recon_scan_headings_deg"]
    if (not isinstance(headings, list) or len(headings) < 3 or headings[0] != 0
            or headings[-1] != 360 or any(not 0 < b - a <= 90 for a, b in zip(headings, headings[1:]))):
        parser.error("Invalid reconnaissance scan headings")
    print("摆位参考角度（正式侦查会连续扫描）：" + " → ".join(f"{angle}°" for angle in headings))
    print("每次停下后可摆放道具；按 Enter 才会进入下一段，输入 q 则结束。")
    if args.dry_run:
        return 0

    robot = None
    executor = None
    log_path = ROOT / "logs" / f"recon_positions_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    def record(event: str, **details: object) -> None:
        row = {"time": datetime.now().astimezone().isoformat(), "event": event, **details}
        line = json.dumps(row, ensure_ascii=False, default=str)
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    try:
        for previous, target in zip(headings, headings[1:]):
            answer = input(f"当前目标位置 {previous}°。检查道具位置后按 Enter 转到 {target}°（q 结束）：").strip().lower()
            if answer == "q":
                print("已结束；未发送下一段转向指令。")
                return 0
            if answer:
                print("未识别输入；已结束。")
                return 1
            if robot is None:
                robot = connect(cfg["robot_ip"])
                executor = RobotExecutor(robot, cfg, llm=None)
                executor.scene = "recon"
                pose = cfg["recon_start_pose"]
                executor.x_cm = float(pose["x_cm"])
                executor.y_cm = float(pose["y_cm"])
                executor.heading_deg = float(pose["heading_deg"])
                executor.record = record
            executor.turn("right", target - previous)
            record("position_reached", heading_deg=target)
        print("已回到起始朝向；道具摆位完成。")
        return 0
    except KeyboardInterrupt:
        print("已中断；正在停止底盘。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"转向失败：{error!r}", file=sys.stderr)
        return 1
    finally:
        if executor is not None:
            executor.stop()
        elif robot is not None:
            robot.stop_chassis()


if __name__ == "__main__":
    raise SystemExit(main())
