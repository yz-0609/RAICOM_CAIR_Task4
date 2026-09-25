"""UGOT national competition Task 4 entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from planning import CloudLLM, PlanError, validate_plan
from robot import ExecutionError, RobotExecutor, connect, listen_text

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    cfg["robot_ip"] = os.environ.get("TASK4_ROBOT_IP", cfg["robot_ip"])
    return cfg


def make_recorder(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def record(event: str, **details) -> None:
        row = {"time": datetime.now().astimezone().isoformat(), "event": event, **details}
        line = json.dumps(row, ensure_ascii=False, default=str)
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    return record


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="UGOT Task 4, four scenes")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--text", help="Use typed speech transcript (diagnostic)")
    parser.add_argument("--plan-json", type=Path, help="Load a stored action plan (diagnostic)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate without robot movement")
    args = parser.parse_args(argv)
    record = make_recorder(ROOT / "logs" / f"task4_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    robot = None
    executor = None
    try:
        load_dotenv(ROOT / ".env", override=False)
        cfg = load_config(args.config)
        llm = None if args.plan_json else CloudLLM()
        if args.plan_json:
            plan = validate_plan(json.loads(args.plan_json.read_text(encoding="utf-8")))
            raw = args.text or "<stored-plan>"
        else:
            if args.text:
                raw = args.text
            else:
                robot = connect(cfg["robot_ip"])
                robot.play_sound("received", wait=True)
                raw = listen_text(robot)
            if not raw:
                raise PlanError("ASR returned empty text")
            record("asr", raw=raw)
            plan = llm.plan(raw)
        record("plan", raw=raw, plan=plan.as_dict())
        if cfg.get("math_supervised_trial", False) and plan.scene != "math":
            raise PlanError("This supervised trial configuration permits only intelligent math")
        if args.dry_run:
            record("dry_run_complete")
            return 0
        if robot is None:
            robot = connect(cfg["robot_ip"])
        if llm is None and plan.scene == "companion":
            llm = CloudLLM()
        executor = RobotExecutor(robot, cfg, llm)
        executor.run(plan, record)
        record("run_complete")
        return 0
    except Exception as error:
        record("run_failed", error=repr(error))
        logging.exception("Task 4 failed")
        return 1
    except KeyboardInterrupt:
        record("interrupted")
        return 130
    finally:
        if executor is not None:
            executor.stop()
        elif robot is not None:
            try:
                robot.stop_chassis()
            except Exception:
                logging.exception("Emergency stop failed")


if __name__ == "__main__":
    sys.exit(main())
