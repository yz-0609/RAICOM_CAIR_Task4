"""Bounded UGOT execution for the four competition scenes.

Physical dimensions and numbered warehouse paths must be calibrated on site.
"""

from __future__ import annotations

import json
import logging
import math
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from planning import Action, Plan
from telemetry import AngleAccumulator

ROOT = Path(__file__).resolve().parent


class ExecutionError(RuntimeError):
    pass


def wait_port(ip: str, port: int = 50051, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((ip, port), timeout=1):
                return
        except OSError:
            time.sleep(0.3)
    if port == 50051:
        try:
            with socket.create_connection((ip, 9090), timeout=1):
                raise ExecutionError(
                    f"{ip}:50051 is unavailable, but :9090 is open. "
                    "This address appears to be the Yanshee robot. "
                    "Read the UGOT IP from its main controller screen and use --ip with that address."
                )
        except OSError:
            pass
    raise ExecutionError(f"UGOT {ip}:{port} is unavailable; check the UGOT screen IP and PC network")


def connect(ip: str) -> Any:
    from ugot import ugot

    wait_port(ip)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            robot = ugot.UGOT()
            # The verified Task 1 run found an unbounded getLanguage RPC. Bound it.
            robot._UGOT__initialize_modules(f"{ip}:50051")
            robot._UGOT__initialize_http_client(ip)
            original = robot.DEVICE.client.getLanguage
            robot.DEVICE.client.getLanguage = lambda request: original(request, timeout=5)
            try:
                robot._UGOT__configure_language()
            except Exception as error:
                logging.warning("UGOT language query failed; using Chinese SDK messages: %r", error)
                get_language = robot.DEVICE.getLanguage
                try:
                    robot.DEVICE.getLanguage = lambda: SimpleNamespace(code=0, lang="china")
                    robot._UGOT__configure_language()
                finally:
                    robot.DEVICE.getLanguage = get_language
            robot.set_volume(100)
            return robot
        except Exception as error:
            last_error = error
            logging.warning("UGOT connection attempt %s failed: %r", attempt + 1, error)
            time.sleep(1)
    raise ExecutionError(f"UGOT initialization failed: {last_error}")


def listen_text(robot: Any, duration_s: int = 30) -> str:
    """Listen for real speech, allowing eight seconds to begin speaking."""
    audio = getattr(robot, "AUDIO", None)
    if audio is not None and hasattr(audio, "getAsrAndDoa"):
        raw = audio.getAsrAndDoa(begin_vad=8000, end_vad=1500, duration=duration_s * 1000)
        try:
            data = json.loads(raw) if raw else {}
            text = str(data.get("asr", {}).get("msg", "") or "").strip()
        except (TypeError, ValueError, AttributeError) as error:
            raise ExecutionError("UGOT ASR returned malformed JSON") from error
    else:
        result = robot.start_audio_asr_doa(duration_s)
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise ExecutionError("UGOT ASR returned an invalid result")
        text = str(result[1] or "").strip()
    if text.casefold() in {"something is error", "asr error", "error"}:
        raise ExecutionError(f"UGOT ASR service returned an error: {text}")
    return text


class RobotExecutor:
    def __init__(self, robot: Any, config: dict[str, Any], llm: Any, reconnect: Any = None) -> None:
        self.robot = robot
        self.reconnect = reconnect or connect
        self.cfg = config
        self.llm = llm
        self.path_cm = 0.0
        self.last_reply = ""
        self.carrying = False
        self._models: set[str] = set()
        self._yolo: Any = None
        self._infrared_id: int | None = None
        self._recon_results: dict[str, str] = {}
        self._recon_scan_done = False
        self.scene = ""
        self.record: Any = lambda *args, **kwargs: None
        start = config["start_pose"]
        self.x_cm = float(start["x_cm"])
        self.y_cm = float(start["y_cm"])
        self.heading_deg = float(start["heading_deg"])

    def stop(self, *, strict: bool = False) -> None:
        try:
            self.robot.stop_chassis()
        except Exception as error:
            logging.exception("Could not stop chassis")
            if strict:
                raise ExecutionError("Could not confirm chassis stop") from error

    def _require_motion(self, action: str = "translation") -> None:
        if self.cfg.get("motion_calibrated", False) and self.cfg["start_pose"].get("calibrated"):
            return
        if self.scene == "warehouse" and self.cfg.get("warehouse", {}).get("task1_adapter"):
            warehouse = self.cfg["warehouse"]
            if warehouse.get("motion_calibrated") and warehouse["pickup_start"].get("calibrated"):
                return
            raise ExecutionError("Verify warehouse pickup start and motion before moving")
        if self.scene == "math" and self.cfg.get("math_supervised_trial", False):
            if self.cfg.get("math_trial_pose", {}).get("calibrated"):
                return
            raise ExecutionError("Set a verified math trial start pose before moving")
        if self.scene == "recon" and action in {"turn", "recon_scan"} and self.cfg.get("recon_turn_calibrated", False):
            if self.cfg.get("recon_start_pose", {}).get("calibrated"):
                return
            raise ExecutionError("Set a verified reconnaissance start pose before rotating")
        raise ExecutionError("Set motion_calibrated only after real distance, turn and boundary tests")

    def _check_pose(self, x: float, y: float) -> None:
        width, height = self.cfg["field_cm"]
        margin = self.cfg["robot_clearance_cm"] + self.cfg.get("motion_error_margin_cm", 0)
        if self.scene == "recon":
            margin = max(margin, self.cfg.get("recon_rotation_clearance_cm", margin))
        if not (margin <= x <= width - margin and margin <= y <= height - margin):
            raise ExecutionError(f"Predicted chassis footprint leaves field: ({x:.1f}, {y:.1f})")

    def _project(self, direction_deg: float, cm: float) -> tuple[float, float]:
        angle = math.radians(direction_deg)
        return self.x_cm + math.sin(angle) * cm, self.y_cm + math.cos(angle) * cm

    def _spend_path(self, cm: float) -> None:
        self._require_motion()
        if cm <= 0 or cm > self.cfg["max_single_move_cm"]:
            raise ExecutionError(f"Single move outside calibrated limit: {cm} cm")
        if self.path_cm + cm > self.cfg["max_total_path_cm"]:
            raise ExecutionError("Total path budget exhausted; stop before field boundary")
    def _bounded_motion(self, command: Any, expected_s: float, observe: Any = None,
                        timeout_extra_s: float | None = None) -> None:
        """Stop even if a movement RPC or its expected duration stalls."""
        # The SDK's distance/angle mode stops itself. Allow acceleration and
        # braking time before sending our redundant stop command.
        observe_s = max(2.5, expected_s + 1.5)
        extra_s = (self.cfg.get("motion_timeout_extra_s", 3)
                   if timeout_extra_s is None else timeout_extra_s)
        deadline_s = observe_s + float(extra_s)
        timed_out = threading.Event()
        def emergency_stop() -> None:
            timed_out.set()
            self.stop()
        watchdog = threading.Timer(deadline_s, emergency_stop)
        watchdog.daemon = True
        watchdog.start()
        started = time.monotonic()
        try:
            command()
            remaining = max(0.0, observe_s - (time.monotonic() - started))
            while remaining > 0 and not timed_out.is_set():
                if observe is not None:
                    observe()
                pause = min(0.2, remaining)
                time.sleep(pause)
                remaining -= pause
            if timed_out.is_set():
                raise ExecutionError("Motion exceeded watchdog deadline; chassis stopped")
        finally:
            watchdog.cancel()
            self.stop(strict=True)

    def move(self, direction: str, cm: float) -> None:
        cm = int(math.floor(cm + 0.5))
        self._spend_path(cm)
        x, y = self._project(self.heading_deg + (180 if direction == "backward" else 0), cm)
        self._check_pose(x, y)
        self._bounded_motion(lambda: self.robot.mecanum_move_speed_times(
                0 if direction == "forward" else 1,
                self.cfg["movement_speed_cm_s"], cm, 1,
            ), cm / self.cfg["movement_speed_cm_s"])
        self.x_cm, self.y_cm = x, y
        self.path_cm += cm

    def strafe(self, direction: str, cm: float) -> None:
        cm = int(math.floor(cm + 0.5))
        self._spend_path(cm)
        x, y = self._project(self.heading_deg + (-90 if direction == "left" else 90), cm)
        self._check_pose(x, y)
        self._bounded_motion(lambda: self.robot.mecanum_translate_speed_times(
                -90 if direction == "left" else 90,
                self.cfg["movement_speed_cm_s"], cm, 1,
            ), cm / self.cfg["movement_speed_cm_s"])
        self.x_cm, self.y_cm = x, y
        self.path_cm += cm

    def turn(self, direction: str, degrees: float, *, speed_deg_s: float | None = None,
             timeout_extra_s: float | None = None) -> None:
        self._require_motion("turn")
        self._check_pose(self.x_cm, self.y_cm)
        speed = speed_deg_s if speed_deg_s is not None else self.cfg["turn_speed_deg_s"]
        if not 5 <= speed <= 40:
            raise ExecutionError("Turn speed must be 5..40 degrees per second")
        if not 1 <= degrees <= 360:
            raise ExecutionError("Turn angle must be 1..360 degrees")
        # The UGOT protobuf target_angle and rotate_speed fields are integers.
        # Keep the commanded angle, IMU target, and predicted heading aligned.
        speed = int(math.floor(speed + 0.5))
        degrees = int(math.floor(degrees + 0.5))
        imu = AngleAccumulator(self.robot.read_gyro_data()[2])
        sampler_stop = threading.Event()
        sampler_error: list[Exception] = []
        def sample_during_blocking_rpc() -> None:
            while not sampler_stop.wait(0.1):
                try:
                    imu.add(self.robot.read_gyro_data()[2])
                except Exception as error:
                    sampler_error.append(error)
                    self.stop()
                    return
        sampler = threading.Thread(target=sample_during_blocking_rpc, daemon=True)
        sampler.start()
        def command() -> None:
            try:
                self.robot.mecanum_turn_speed_times(
                    2 if direction == "left" else 3,
                    speed, degrees, 2,
                )
            finally:
                sampler_stop.set()
                sampler.join(timeout=1)
        expected_yaw = degrees if direction == "left" else -degrees
        try:
            options = {"timeout_extra_s": timeout_extra_s} if timeout_extra_s is not None else {}
            self._bounded_motion(command, degrees / speed,
                                 observe=lambda: imu.add(self.robot.read_gyro_data()[2]),
                                 **options)
        except Exception as error:
            try:
                imu.add(self.robot.read_gyro_data()[2])
                self.record("turn_interrupted_imu", direction=direction, target_deg=degrees,
                            observed_yaw_deg=round(imu.total, 2))
            except Exception:
                logging.exception("Could not read IMU after interrupted turn")
                raise
            late_sdk_error = not isinstance(error, ExecutionError)
            watchdog_expired = (isinstance(error, ExecutionError)
                                and "Motion exceeded watchdog deadline" in str(error))
            if not ((late_sdk_error or watchdog_expired)
                    and abs(imu.total - expected_yaw) <= self.cfg.get("turn_tolerance_deg", 15)):
                raise
            # _bounded_motion confirmed a stop even if its SDK RPC failed.
            # Accept that late error only when IMU independently verifies the turn.
            self.record("turn_late_rpc_verified", direction=direction, target_deg=degrees,
                        observed_yaw_deg=round(imu.total, 2), error=repr(error))
        finally:
            sampler_stop.set()
            sampler.join(timeout=1)
        if sampler_error:
            raise ExecutionError("IMU sampling failed during turn") from sampler_error[0]
        imu.add(self.robot.read_gyro_data()[2])
        self.record("turn_imu", direction=direction, target_deg=degrees, observed_yaw_deg=round(imu.total, 2))
        if abs(imu.total - expected_yaw) > self.cfg.get("turn_tolerance_deg", 15):
            raise ExecutionError(f"Turn IMU mismatch: target {expected_yaw:.1f}°, observed {imu.total:.1f}°")
        self.heading_deg = (self.heading_deg + (degrees if direction == "right" else -degrees)) % 360

    def _load_model(self, name: str) -> None:
        if name not in self._models:
            if not self.robot.load_models([name]):
                raise ExecutionError(f"Could not load UGOT model {name}")
            self._models.add(name)
            time.sleep(0.4)

    def _release_models(self) -> None:
        for name in tuple(self._models):
            try:
                self.robot.release_models([name])
            finally:
                self._models.discard(name)

    def _frame(self) -> Any:
        import cv2
        import numpy as np

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            data = self.robot.read_camera_data()
            if data:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
            time.sleep(0.1)
        raise ExecutionError("Camera stream did not provide a decodable frame within 3 seconds")

    def _get_yolo(self) -> Any:
        if self._yolo is None:
            from ultralytics import YOLO

            path = ROOT / "models" / "best.pt"
            if not path.exists():
                raise ExecutionError(f"Missing verified cube model: {path}")
            self._yolo = YOLO(str(path))
        return self._yolo

    def _find_cube(self, color: str) -> tuple[float, float] | None:
        model = self._get_yolo()
        frame = self._frame()
        result = model(frame, imgsz=640, conf=0.5, verbose=False)[0]
        boxes = result.boxes
        candidates: list[tuple[float, float]] = []
        if boxes is not None and boxes.xyxy is not None:
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                # Class order comes from the verified Task 1 project.
                if cls >= 3 or ("red", "green", "blue")[cls] != color:
                    continue
                x1, _, x2, _ = boxes.xyxy[i].tolist()
                candidates.append((float(x2 - x1), (x1 + x2) / 2 - frame.shape[1] / 2))
        return max(candidates, default=None)

    def _sensor_id(self) -> int:
        if self._infrared_id is None:
            devices = self.robot.get_peripheral_devices_list() or []
            sensors = [int(d["deviceId"]) for d in devices if d.get("type") == "Infrared"]
            if not sensors:
                raise ExecutionError("No infrared sensor found; pickup cannot be verified")
            self._infrared_id = sensors[0]
        return self._infrared_id

    def _arm_pose(self, pose: dict[str, int], duration_ms: int = 1200) -> None:
        # Extracted from verified Task 1: use absolute poses and readback.
        for sid in (51, 52, 53):
            current = self.robot.read_servo_angle(sid) or {}
            value = current.get(str(sid))
            if value is None:
                raise ExecutionError(f"Servo {sid} readback unavailable")
            target = int(pose[str(sid)])
            if abs(target - int(value)) > 180:
                # Firmware may take the long way when sent an absolute target
                # across +/-180. Use the verified Task 1 low-speed crossing.
                shortest = (target - int(value) + 180) % 360 - 180
                speed = 12 if shortest > 0 else -12
                deadline = time.monotonic() + 2.5
                self.robot.turn_servo_speed(sid, speed)
                try:
                    while time.monotonic() < deadline:
                        time.sleep(0.05)
                        current = (self.robot.read_servo_angle(sid) or {}).get(str(sid))
                        if current is not None and abs(target - int(current)) <= 180:
                            break
                    else:
                        raise ExecutionError(f"Servo {sid} could not safely cross angle wrap")
                finally:
                    self.robot.stop_servo(sid, lock=True)
        for sid in (51, 52, 53):
            self.robot.turn_servo_angle(sid, int(pose[str(sid)]), duration_ms, wait=False)
        time.sleep(duration_ms / 1000 + 0.2)
        for sid in (51, 52, 53):
            actual = (self.robot.read_servo_angle(sid) or {}).get(str(sid))
            if actual is None or abs((int(actual) - int(pose[str(sid)]) + 180) % 360 - 180) > 5:
                raise ExecutionError(f"Servo {sid} did not reach target")

    def pickup(self, color: str) -> None:
        self._require_motion()
        if self.scene == "warehouse" and self.cfg["warehouse"].get("task1_adapter"):
            from warehouse_task1 import track_and_grab

            warehouse = self.cfg["warehouse"]
            self.robot.open_camera()
            track_and_grab(self.robot, warehouse.get("task1_source"), self._sensor_id(), color)
            self.carrying = True
            self.record("warehouse_grab_confirmed", color=color, source="task1")
            return
        cfg = self.cfg["grasp"]
        self.robot.open_camera()
        sid = self._sensor_id()
        seen = 0
        deadline = time.monotonic() + cfg["search_timeout_s"]
        while time.monotonic() < deadline:
            found = self._find_cube(color)
            if found is not None:
                seen += 1
                if seen >= 2:
                    break
            else:
                seen = 0
            self.turn("left", 5)
        else:
            raise ExecutionError(f"Could not find {color} cube")
        stable = 0
        deadline = time.monotonic() + cfg["approach_timeout_s"]
        while time.monotonic() < deadline:
            found = self._find_cube(color)
            distance = self.robot.read_distance_data(sid)
            if found is None or distance <= 0:
                raise ExecutionError("Cube or distance sensor lost during approach")
            _, offset = found
            if abs(offset) > cfg["center_tolerance_px"]:
                self.turn("right" if offset > 0 else "left", 3)
                stable = 0
            elif abs(distance - cfg["distance_cm"]) <= cfg["tolerance_cm"]:
                stable += 1
                if stable >= cfg["stable_frames"]:
                    break
            elif distance > cfg["distance_cm"]:
                self.move("forward", min(2, max(1, distance - cfg["distance_cm"])))
                stable = 0
            else:
                self.move("backward", 1)
                stable = 0
        else:
            raise ExecutionError("Cube approach timed out")
        self._arm_pose(cfg["carry_arm_pose"])
        self.robot.mechanical_clamp_release()
        time.sleep(0.3)
        self._arm_pose(cfg["grab_arm_pose"], 2000)
        self.robot.mechanical_clamp_close()
        time.sleep(0.5)
        self._arm_pose(cfg["carry_arm_pose"], 2200)
        self.carrying = True

    def _navigate_to(self, target_x: float, target_y: float) -> None:
        """Navigate between surveyed field points, splitting each leg at the SDK limit."""
        self._require_motion()
        self._check_pose(target_x, target_y)
        distance = math.hypot(target_x - self.x_cm, target_y - self.y_cm)
        if distance <= 1:
            return
        desired = math.degrees(math.atan2(target_x - self.x_cm, target_y - self.y_cm)) % 360
        delta = (desired - self.heading_deg + 180) % 360 - 180
        if abs(delta) >= 1:
            self.turn("right" if delta > 0 else "left", abs(delta))
        while distance > 1:
            step = min(self.cfg["max_single_move_cm"], distance)
            self.move("forward", step)
            distance = math.hypot(target_x - self.x_cm, target_y - self.y_cm)

    def _confirm_tag(self, tag_id: int) -> None:
        self._load_model("apriltag_qrcode")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            tags = self.robot.get_apriltag_total_info() or []
            if any(isinstance(tag, (list, tuple)) and len(tag) > 0 and tag[0] == tag_id for tag in tags):
                return
            time.sleep(0.2)
        raise ExecutionError(f"Expected AprilTag {tag_id} not visible")

    def place(self, position: int) -> None:
        if not self.carrying:
            raise ExecutionError("Cannot place without a completed pickup")
        if self.scene == "warehouse" and self.cfg["warehouse"].get("task1_adapter"):
            # The team deliberately forfeits the numbered destination point.
            # Preserve its place action in the five-step order, but release in
            # place without navigating to the spoken Y position.
            pose = self.cfg["grasp"]["carry_arm_pose"]
            target = int(self.cfg["warehouse"]["drop_servo_51_deg"])
            self.stop(strict=True)
            try:
                self.robot.turn_servo_angle(51, target, 600, wait=True)
                self.robot.mechanical_clamp_release()
                time.sleep(0.4)
                self.robot.turn_servo_angle(51, int(pose["51"]), 600, wait=True)
            finally:
                self.robot.stop_servo(51, lock=True)
            self.carrying = False
            self.record("warehouse_place_forfeited", requested_position=position,
                        released_at="current_position", servo_51_deg=target)
            return {"numbered_position_scored": False, "requested_position": position}
        cfg = self.cfg["warehouse"]
        target = cfg["positions"].get(str(position))
        pose = cfg["place_arm_pose"]
        if not target or not target.get("calibrated") or not pose.get("calibrated"):
            raise ExecutionError(f"Position {position} or place arm pose is not calibrated")
        if "approach_cm" not in target:
            raise ExecutionError(f"Position {position} has no surveyed approach coordinate")
        for point in [*target.get("waypoints_cm", []), target["approach_cm"]]:
            if not isinstance(point, list) or len(point) != 2:
                raise ExecutionError("Warehouse waypoint must be an [x_cm, y_cm] pair")
            self._navigate_to(float(point[0]), float(point[1]))
        if "heading_deg" in target:
            delta = (float(target["heading_deg"]) - self.heading_deg + 180) % 360 - 180
            if abs(delta) >= 1:
                self.turn("right" if delta > 0 else "left", abs(delta))
        if target.get("tag_id") is not None:
            self._confirm_tag(int(target["tag_id"]))
        self._arm_pose(pose["angles"])
        self.robot.mechanical_clamp_release()
        time.sleep(0.4)
        self._arm_pose(self.cfg["grasp"]["carry_arm_pose"])
        self.carrying = False

    def return_line(self) -> None:
        if self.scene == "warehouse" and self.cfg["warehouse"].get("task1_adapter"):
            from warehouse_task1 import chase_tag_zero, follow_pickup_line

            cfg = self.cfg["warehouse"]
            self._require_motion()
            self.robot.open_camera()
            chase_tag_zero(self.robot, cfg.get("task1_source"), self._sensor_id())
            self.record("warehouse_tag_reached", tag_id=0)
            self.turn("left", 90)
            distance = follow_pickup_line(
                self.robot, self._frame, cfg.get("task1_source"),
                timeout_s=float(cfg["line_timeout_s"]),
                max_travel_cm=float(cfg["line_max_travel_cm"]),
                confidence=float(cfg["line_confidence"]),
            )
            self.record("warehouse_pickup_line_confirmed", travelled_cm=round(distance, 1))
            return
        cfg = self.cfg["warehouse"]["line_return"]
        if not cfg.get("calibrated") or not isinstance(cfg.get("segment_cm"), list) or len(cfg["segment_cm"]) != 2:
            raise ExecutionError("Return-to-line segment is not calibrated")
        (x1, y1), (x2, y2) = cfg["segment_cm"]
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq <= 0:
            raise ExecutionError("Invalid line segment geometry")
        t = max(0.0, min(1.0, ((self.x_cm - x1) * dx + (self.y_cm - y1) * dy) / length_sq))
        target_x, target_y = x1 + t * dx, y1 + t * dy
        self._navigate_to(target_x, target_y)
        if cfg.get("tag_id") is not None:
            self._confirm_tag(int(cfg["tag_id"]))
        self.robot.open_camera()
        self._confirm_line(float(cfg["line_confidence"]))

    def _confirm_line(self, threshold: float) -> None:
        import cv2
        import numpy as np
        import onnxruntime as ort

        path = ROOT / "models" / "line_seg_mnv3.onnx"
        if not path.exists():
            raise ExecutionError("Missing verified lane model")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        frame = self._frame()
        roi = frame[int(frame.shape[0] * 0.30):]
        rgb = cv2.cvtColor(cv2.resize(roi, (320, 192)), cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        tensor = tensor.transpose(2, 0, 1)[None]
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if not np.isfinite(output).all():
            raise ExecutionError("Lane model returned invalid values")
        prob = 1 / (1 + np.exp(-np.clip(output.squeeze(), -30, 30)))
        lower = prob[96:]
        mask = lower >= 0.5
        coverage = float(mask.mean())
        mean_prob = float(lower[mask].mean()) if mask.any() else 0
        confidence = 0.75 * mean_prob + 0.25 * min(1, coverage / 0.06)
        if coverage < 0.006 or confidence < threshold:
            raise ExecutionError("Pickup-area line not confirmed")

    def recognize(self, kind: str) -> str:
        if self.scene != "recon":
            return self._recognize_single(kind)
        if not self._recon_scan_done:
            self._scan_recon()
            self._recon_scan_done = True
        if self.cfg.get("recon_announce_immediately", False):
            return self._recon_results.get(kind, "")
        return self._recon_results[kind]

    def _announce_recon(self, plan: Plan) -> None:
        text = "".join(self._recognition_utterance(a.args["kind"], self._recon_results[a.args["kind"]])
                       for a in plan.actions)
        self.stop(strict=True)
        self.robot = self.reconnect(self.cfg["robot_ip"])
        self.record("announcement_start", text=text)
        self._tts_checked(text)
        self.record("announcement_complete", text=text)

    def _tts_checked(self, text: str) -> None:
        for attempt in range(2):
            response = self.robot.AUDIO.setAudioTts(text, 0)
            self.record("tts_response", attempt=attempt + 1,
                        code=getattr(response, "code", None), message=getattr(response, "msg", None))
            if getattr(response, "msg", "") == "success":
                return
            if attempt == 0 and getattr(response, "code", None) == -1:
                self.robot = self.reconnect(self.cfg["robot_ip"])
                continue
            raise ExecutionError(f"UGOT TTS service rejected speech: {response!r}")

    def _scan_recon(self) -> None:
        """Continuously observe while making one slow, bounded revolution."""
        if self.cfg.get("recon_announce_immediately", False):
            return self._scan_recon_immediate()
        self._require_motion("recon_scan")
        speed = self.cfg.get("recon_scan_speed_deg_s", 7)
        if not isinstance(speed, (int, float)) or not 5 <= speed <= 20:
            raise ExecutionError("Recon scan speed must be 5..20 degrees per second")
        timeout_extra_s = self.cfg.get("recon_scan_timeout_extra_s", 30)
        if not isinstance(timeout_extra_s, (int, float)) or not 3 <= timeout_extra_s <= 30:
            raise ExecutionError("Recon scan timeout allowance must be 3..30 seconds")
        names = {"word": "word_recognition", "gesture": "gesture", "traffic": "traffic_sign",
                 "tag": "apriltag_qrcode", "color": "color_recognition"}
        getters = {"word": self.robot.get_words_result, "gesture": self.robot.get_gesture_result,
                   "traffic": self.robot.get_traffic_total_info, "tag": self.robot.get_apriltag_total_info,
                   "color": self.robot.get_color_total_info}
        self._release_models()
        self.robot.open_camera()
        if not self.robot.load_models(list(names.values())):
            raise ExecutionError("Could not load reconnaissance vision models")
        self._models.update(names.values())
        previous = {kind: "" for kind in names}
        counts = {kind: 0 for kind in names}
        traffic_evidence: dict[str, float] = {}
        last_color_check: dict[str, float] = {}
        ignored_traffic: set[str] = set()
        turn_done = threading.Event()
        turn_errors: list[Exception] = []

        def rotate() -> None:
            try:
                self.turn("right", 360, speed_deg_s=speed, timeout_extra_s=timeout_extra_s)
            except Exception as error:
                turn_errors.append(error)
            finally:
                turn_done.set()

        turn_thread = threading.Thread(target=rotate, daemon=True)
        deadline = time.monotonic() + 360 / speed + 1.5 + timeout_extra_s + 2
        self.record("recon_scan_start", speed_deg_s=speed, timeout_extra_s=timeout_extra_s)
        turn_thread.start()
        try:
            scan_pass = 0
            while not turn_done.is_set() or scan_pass < 3:
                if turn_errors:
                    raise ExecutionError("Recon rotation failed") from turn_errors[0]
                if not turn_done.is_set() and time.monotonic() > deadline:
                    raise ExecutionError("Recon rotation exceeded deadline")
                candidates: dict[str, str] = {}
                for kind, getter in getters.items():
                    if turn_done.is_set() and scan_pass >= 3:
                        break
                    if kind in self._recon_results and kind != "traffic":
                        continue
                    raw = getter()
                    value = self._recognition_value(kind, raw)
                    if kind == "tag" and raw and not value:
                        self.record("tag_rejected", raw=raw, scan_pass=scan_pass)
                    if (kind == "traffic" and not value and isinstance(raw, list) and raw
                            and isinstance(raw[0], (list, tuple)) and raw[0]):
                        label = str(raw[0][0]).casefold()
                        if label in {"斑马线", "zebra crossing"} and label not in ignored_traffic:
                            ignored_traffic.add(label)
                            self.record("traffic_ignored", value=label, scan_pass=scan_pass)
                    if value:
                        candidates[kind] = value
                    if kind == "tag" and value and kind not in self._recon_results:
                        # A decoded AprilTag ID has error correction; a single
                        # valid 0/4 sighting may be all a moving camera gets.
                        self._recon_results[kind] = value
                        self.record("recognition", kind=kind, value=value, scan_pass=scan_pass)
                        continue
                    if value and value == previous[kind]:
                        counts[kind] += 1
                        if counts[kind] >= 2:
                            if kind == "traffic":
                                area = float(raw[0][5]) if isinstance(raw, list) and raw and len(raw[0]) >= 6 else 1.0
                                if area > traffic_evidence.get(value, 0):
                                    traffic_evidence[value] = area
                                    self.record("traffic_candidate", value=value, area_px=area, scan_pass=scan_pass)
                            elif kind == "color":
                                now = time.monotonic()
                                if now - last_color_check.get(value, float("-inf")) >= 1:
                                    last_color_check[value] = now
                                    cube = self._cube_color_from_yolo()
                                    if cube == value:
                                        self._recon_results[kind] = value
                                        self.record("recognition", kind=kind, value=value,
                                                    scan_pass=scan_pass, cube_model=cube)
                                    else:
                                        self.record("color_rejected", value=value, cube_model=cube,
                                                    scan_pass=scan_pass)
                            else:
                                self._recon_results[kind] = value
                                self.record("recognition", kind=kind, value=value, scan_pass=scan_pass)
                    else:
                        previous[kind], counts[kind] = value, 0
                self.record("recon_view", scan_pass=scan_pass, candidates=candidates)
                scan_pass += 1
                if not turn_done.is_set():
                    time.sleep(0.15)
            turn_thread.join(timeout=0.5)
            if turn_errors:
                raise ExecutionError("Recon rotation failed") from turn_errors[0]
            self.record("recon_scan_complete", scan_passes=scan_pass)
            if traffic_evidence:
                best = max(traffic_evidence, key=traffic_evidence.get)
                self._recon_results["traffic"] = best
                self.record("recognition", kind="traffic", value=best,
                            area_px=traffic_evidence[best], alternatives=traffic_evidence)
            missing = sorted(set(names) - self._recon_results.keys())
            if missing:
                raise ExecutionError(f"Recon scan incomplete: {', '.join(missing)}")
        finally:
            if turn_thread.is_alive():
                self.stop(strict=True)
                turn_thread.join(timeout=1)
            self._release_models()

    def _scan_recon_immediate(self) -> None:
        """Diagnostic scan: stop, speak each confirmed observation, then resume."""
        self._require_motion("recon_scan")
        self._check_pose(self.x_cm, self.y_cm)
        speed = self.cfg.get("recon_scan_speed_deg_s", 7)
        if not isinstance(speed, (int, float)) or not 5 <= speed <= 20:
            raise ExecutionError("Recon scan speed must be 5..20 degrees per second")
        names = {"word": ("word_recognition", "get_words_result"),
                 "tag": ("apriltag_qrcode", "get_apriltag_total_info"),
                 "traffic": ("traffic_sign", "get_traffic_total_info"),
                 "color": ("color_recognition", "get_color_total_info"),
                 "gesture": ("gesture", "get_gesture_result")}
        self._release_models()
        self.robot.open_camera()
        if not self.robot.load_models([name for name, _ in names.values()]):
            raise ExecutionError("Could not load reconnaissance vision models")
        self._models.update(name for name, _ in names.values())

        imu = AngleAccumulator(self.robot.read_gyro_data()[2])
        imu_lock = threading.Lock()
        control_lock = threading.RLock()
        sampler_stop = threading.Event()
        rotation_complete = threading.Event()
        timed_out = threading.Event()
        moving = threading.Event()
        sampler_errors: list[Exception] = []
        def angle() -> float:
            with imu_lock:
                return imu.total

        def brake() -> None:
            moving.clear()
            with control_lock:
                self.stop(strict=True)

        def resume() -> None:
            with control_lock:
                if rotation_complete.is_set():
                    return
            # Keep the potentially blocking command outside the stop lock.
            self.robot.mecanum_move_turn(0, 0, 3, speed)
            if rotation_complete.is_set():
                brake()
            else:
                moving.set()

        def sample_imu() -> None:
            last_progress_at = time.monotonic()
            last_progress_yaw = 0.0
            while not sampler_stop.wait(0.1):
                try:
                    reading = self.robot.read_gyro_data()[2]
                    with imu_lock:
                        observed = imu.add(reading)
                    now = time.monotonic()
                    if not moving.is_set() or abs(observed - last_progress_yaw) >= 1:
                        last_progress_at, last_progress_yaw = now, observed
                    elif now - last_progress_at > 8:
                        raise ExecutionError("Recon rotation stalled for 8 seconds")
                    if observed <= -360 and not rotation_complete.is_set():
                        rotation_complete.set()
                        brake()
                except Exception as error:
                    sampler_errors.append(error)
                    rotation_complete.set()
                    try:
                        brake()
                    except Exception as stop_error:
                        sampler_errors.append(stop_error)
                    return

        def deadline_stop() -> None:
            timed_out.set()
            rotation_complete.set()
            try:
                brake()
            except Exception as error:
                sampler_errors.append(error)

        # The independent sampler and timer can stop the chassis even if a
        # vision RPC takes longer than expected.
        # Include TTS pauses and slow real motor response; the sampler still
        # stops at 360° and catches any eight-second motor stall.
        max_wall_s = 360 / speed * 2.5 + 80
        watchdog = threading.Timer(max_wall_s, deadline_stop)
        watchdog.daemon = True
        sampler = threading.Thread(target=sample_imu, daemon=True)
        streaks: dict[tuple[str, str], int] = {}
        last_seen: dict[tuple[str, str], int] = {}
        announced: set[tuple[str, str]] = set()
        reported_values: dict[str, list[str]] = {kind: [] for kind in names}
        last_color_check: dict[str, float] = {}
        scan_pass = 0

        def announce(kind: str, value: str) -> None:
            announced.add((kind, value))
            reported_values[kind].append(value)
            self._recon_results[kind] = "、".join(reported_values[kind])
            self.record("recognition", kind=kind, value=value, scan_pass=scan_pass,
                        observed_yaw_deg=round(angle(), 2))
            phrase = self._recognition_utterance(kind, value)
            self.record("announcement_start", kind=kind, text=phrase)
            self._tts_checked(phrase)
            self.record("announcement_complete", kind=kind, text=phrase)

        self.record("recon_scan_start", speed_deg_s=speed, mode="immediate")
        try:
            sampler.start()
            watchdog.start()
            resume()
            while not rotation_complete.is_set():
                if sampler_errors:
                    raise ExecutionError("Recon IMU or stop failed") from sampler_errors[0]
                candidates: dict[str, list[str]] = {}
                for kind, (_, getter_name) in names.items():
                    if rotation_complete.is_set():
                        break
                    raw = getattr(self.robot, getter_name)()
                    items = raw if kind in {"tag", "traffic"} and isinstance(raw, list) else [raw]
                    values: list[str] = []
                    for item in items:
                        model_result = [item] if kind in {"tag", "traffic"} else item
                        value = self._recognition_value(kind, model_result)
                        if value and value not in values:
                            values.append(value)
                    if values:
                        candidates[kind] = values
                    if kind == "tag" and raw and not values:
                        self.record("tag_rejected", raw=raw, scan_pass=scan_pass)
                    confirmed: list[str] = []
                    for value in values:
                        key = (kind, value)
                        if key in announced:
                            continue
                        streaks[key] = streaks.get(key, 0) + 1 if last_seen.get(key) == scan_pass - 1 else 1
                        last_seen[key] = scan_pass
                        if kind == "tag" or streaks[key] >= 3:
                            confirmed.append(value)
                    if kind == "color":
                        now = time.monotonic()
                        confirmed = [value for value in confirmed
                                     if now - last_color_check.get(value, float("-inf")) >= 1]
                    if not confirmed:
                        continue
                    brake()
                    if kind == "color":
                        cube = self._cube_color_from_yolo()
                        for value in confirmed:
                            last_color_check[value] = now
                            if cube != value:
                                self.record("color_rejected", value=value, cube_model=cube,
                                            scan_pass=scan_pass)
                        confirmed = [value for value in confirmed if cube == value]
                    for value in confirmed:
                        announce(kind, value)
                    if not rotation_complete.is_set():
                        resume()
                self.record("recon_view", scan_pass=scan_pass, candidates=candidates,
                            observed_yaw_deg=round(angle(), 2))
                scan_pass += 1
                if not rotation_complete.is_set():
                    time.sleep(0.15)
            if sampler_errors:
                raise ExecutionError("Recon IMU or stop failed") from sampler_errors[0]
            if timed_out.is_set():
                raise ExecutionError("Recon rotation exceeded deadline")
            brake()
            observed = angle()
            if abs(observed + 360) > self.cfg.get("turn_tolerance_deg", 15):
                raise ExecutionError(f"Recon rotation IMU mismatch: {observed:.1f}°")
            self.heading_deg = self.heading_deg % 360
            self.record("recon_scan_complete", scan_passes=scan_pass,
                        observed_yaw_deg=round(observed, 2), mode="immediate")
            missing = sorted(set(names) - self._recon_results.keys())
            if missing:
                self.record("recon_missing", kinds=missing)
        finally:
            watchdog.cancel()
            sampler_stop.set()
            brake()
            sampler.join(timeout=1)
            self._release_models()

    def _cube_color_from_yolo(self) -> str:
        model = self._get_yolo()
        result = model.predict(self._frame(), conf=0.5, verbose=False)[0]
        if not len(result.boxes):
            return ""
        box = max(result.boxes, key=lambda item: float(item.conf[0]))
        name = str(model.names[int(box.cls[0])]).casefold()
        return {"red": "红色", "green": "绿色", "blue": "蓝色"}.get(name, "")

    def _recognize_single(self, kind: str) -> str:
        model = {"word": "word_recognition", "gesture": "gesture", "traffic": "traffic_sign",
                 "tag": "apriltag_qrcode", "color": "color_recognition"}[kind]
        self._release_models()
        self._load_model(model)
        getter = {
            "word": self.robot.get_words_result,
            "gesture": self.robot.get_gesture_result,
            "traffic": self.robot.get_traffic_total_info,
            "tag": self.robot.get_apriltag_total_info,
            "color": self.robot.get_color_total_info,
        }[kind]
        previous = ""
        stable = 0
        for _ in range(24):
            for _ in range(3):
                raw = getter()
                value = self._recognition_value(kind, raw)
                if value and value == previous:
                    stable += 1
                    if stable >= 2:
                        self.robot.play_audio_tts(self._recognition_utterance(kind, value), 0, wait=True)
                        return value
                else:
                    previous, stable = value, 0
                time.sleep(0.15)
            self.turn("right", 15)
        raise ExecutionError(f"Recognition timed out: {kind}")

    @staticmethod
    def _recognition_value(kind: str, raw: Any) -> str:
        if kind == "traffic" and isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) >= 6:
            # On the field, the traffic model classified the small "你好" text
            # patch as a zebra crossing (about 3,300 px²). The actual sign
            # occupied about 39,000 px² in the same 640×480 camera image.
            try:
                if float(raw[0][5]) < 8000:
                    return ""
            except (TypeError, ValueError):
                return ""
        if kind == "color" and isinstance(raw, list):
            if len(raw) > 1 and str(raw[1]).casefold() not in {"方块", "square", "cube", "block"}:
                return ""
            if len(raw) >= 7:
                # The green-light screen produced a 3,200 px² color patch;
                # the real red cube at its competition distance was 8,500+.
                try:
                    if float(raw[6]) < 6000:
                        return ""
                except (TypeError, ValueError):
                    return ""
        if kind in {"word", "gesture"}:
            value = str(raw or "").strip()
        elif kind in {"traffic", "tag"}:
            value = str(raw[0][0]).strip() if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)) and raw[0] else ""
        elif kind == "color":
            value = str(raw[0]).strip() if isinstance(raw, list) and raw and raw[0] else ""
        else:
            return ""
        if value.casefold() in {"no_color_rec", "no_gesture_rec", "no_traffic_rec",
                                "no_word_rec", "no_words_rec", "no_apriltag_rec",
                                "none", "something is error"}:
            return ""
        if kind == "word" and len(value) > 5:
            return ""
        if kind == "tag":
            return value if value in {"0", "4"} else ""
        names = {
            "gesture": {"rock": "石头", "scissors": "剪刀", "paper": "布", "ok": "OK", "thumbs up": "点赞"},
            "traffic": {"green light": "绿灯", "red light": "红灯", "yellow light": "黄灯",
                        "left turn": "左转", "right turn": "右转", "horn": "鸣笛",
                        "zebra crossing": "斑马线", "children": "注意儿童",
                        "no long-time parking": "禁止长时间停车", "enter tunnel": "进入隧道"},
            "color": {"red": "红色", "green": "绿色", "blue": "蓝色"},
        }
        normalized = names.get(kind, {}).get(value.casefold(), value)
        if kind == "traffic" and normalized == "斑马线":
            # The map's black route is often reported as this absent prop.
            return ""
        if kind == "color" and normalized not in {"红色", "绿色", "蓝色"}:
            return ""
        return normalized

    @staticmethod
    def _kind_chinese(kind: str) -> str:
        return {"word": "文字", "gesture": "手势", "traffic": "交通标志", "tag": "标签", "color": "色块"}[kind]

    @classmethod
    def _recognition_utterance(cls, kind: str, value: str) -> str:
        spoken = {"0": "零号", "4": "四号"}.get(value, value) if kind == "tag" else value
        return f"{cls._kind_chinese(kind)}，{spoken}。"

    def polygon(self, sides: int, side_cm: float, direction: str, travel: str) -> None:
        self._require_motion()
        side_cm = int(math.floor(side_cm + 0.5))
        if sides * side_cm + self.path_cm > self.cfg["max_total_path_cm"]:
            raise ExecutionError("Polygon exceeds calibrated path budget")
        for index in range(sides):
            self.move(travel, side_cm)
            self.turn("right" if direction == "clockwise" else "left",
                      self._polygon_turn_degrees(sides, index))

    @staticmethod
    def _polygon_turn_degrees(sides: int, index: int) -> int:
        # UGOT requires an integer target angle. Distribute rounding across
        # corners so every supported polygon still turns exactly 360° overall.
        return round((index + 1) * 360 / sides) - round(index * 360 / sides)

    def arm_wave(self) -> None:
        pose = self.cfg["grasp"]["carry_arm_pose"]
        # The rule requires at least three seconds of actual arm motion.
        # Three 600 ms cycles provide 3.6 seconds of commanded movement.
        try:
            self._arm_pose(pose)
            for _ in range(3):
                self.robot.turn_servo_angle(51, int(pose["51"]) + 12, 600, wait=True)
                self.robot.turn_servo_angle(51, int(pose["51"]) - 12, 600, wait=True)
            self._arm_pose(pose)
        finally:
            self.robot.stop_servo(51, lock=True)

    def execute(self, action: Action) -> Any:
        kind, args = action.kind, action.args
        if kind == "move": return self.move(**args)
        if kind == "strafe": return self.strafe(**args)
        if kind == "turn": return self.turn(**args)
        if kind == "spin": return self.turn(args["direction"], 360)
        if kind == "polygon": return self.polygon(**args)
        if kind == "pickup": return self.pickup(args["color"])
        if kind == "place": return self.place(args["position"])
        if kind == "return_line": return self.return_line()
        if kind == "recognize": return self.recognize(args["kind"])
        if kind == "conversation":
            self._tts_checked("请说，我在听。")
            self.robot.play_sound("received", wait=True)
            question = listen_text(self.robot)
            if not question:
                raise ExecutionError("Companion ASR returned empty text")
            self.record("companion_asr", raw=question)
            self.last_reply = self.llm.answer(question)
            self._tts_checked(self.last_reply)
            return self.last_reply
        if kind == "emotion": return self.robot.screen_display_emotion(args["name"])
        if kind == "sound": return self.robot.play_sound(args["name"], wait=True)
        if kind == "light":
            rgb = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}[args["color"]]
            device = getattr(self.robot, "DEVICE", None)
            if device is not None and hasattr(device, "showLightEffect"):
                packed_rgb = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
                response = device.showLightEffect(packed_rgb, 2)
                code = getattr(response, "code", None)
                self.record("light_response", color=args["color"], effect="breathing",
                            code=code, message=getattr(response, "msg", None))
                if code != 0:
                    raise ExecutionError(f"UGOT light service rejected breathing effect: {response!r}")
            else:
                self.robot.show_light_rgb_effect(*rgb, 2)
            # The rule requires three seconds; allow more time to see a full cycle.
            time.sleep(5)
            return None
        if kind == "screen_text":
            if not self.last_reply: raise ExecutionError("No companion reply to display")
            return self.robot.screen_print_text(self.last_reply, 1)
        if kind == "arm_wave": return self.arm_wave()
        raise ExecutionError(f"Unimplemented action: {kind}")

    def run(self, plan: Plan, record: Any) -> None:
        self.scene = plan.scene
        if plan.scene == "warehouse" and self.cfg["warehouse"].get("task1_adapter"):
            pose = self.cfg["warehouse"]["pickup_start"]
            if not pose.get("calibrated"):
                raise ExecutionError("Verify the warehouse pickup start pose before moving")
            self.x_cm = float(pose["x_cm"])
            self.y_cm = float(pose["y_cm"])
            self.heading_deg = float(pose["heading_deg"])
        elif plan.scene == "recon":
            pose = self.cfg.get("recon_start_pose", self.cfg["start_pose"])
            self.x_cm = float(pose["x_cm"])
            self.y_cm = float(pose["y_cm"])
            self.heading_deg = float(pose["heading_deg"])
        elif plan.scene == "math" and self.cfg.get("math_supervised_trial", False):
            pose = self.cfg.get("math_trial_pose", {})
            if not pose.get("calibrated"):
                raise ExecutionError("Set a verified math trial start pose before moving")
            self.x_cm = float(pose["x_cm"])
            self.y_cm = float(pose["y_cm"])
            self.heading_deg = float(pose["heading_deg"])
        self.preflight(plan)
        self.record = record
        self._recon_results.clear()
        self._recon_scan_done = False
        try:
            for index, action in enumerate(plan.actions, 1):
                record("action_start", index=index, action={"type": action.kind, **action.args})
                try:
                    result = self.execute(action)
                except Exception as error:
                    record("action_failed", index=index, error=repr(error))
                    raise
                record("action_complete", index=index, result=result)
            if plan.scene == "recon" and not self.cfg.get("recon_announce_immediately", False):
                self._announce_recon(plan)
        finally:
            self.stop()
            self._release_models()

    def preflight(self, plan: Plan) -> None:
        motion_actions = {"move", "strafe", "turn", "spin", "polygon", "pickup", "place", "return_line", "recognize"}
        if any(a.kind in motion_actions for a in plan.actions):
            self._require_motion("recon_scan" if plan.scene == "recon" else "translation")
        if plan.scene == "recon":
            self._check_pose(self.x_cm, self.y_cm)
        if plan.scene == "warehouse":
            warehouse = self.cfg["warehouse"]
            if not warehouse["pickup_start"].get("calibrated"):
                raise ExecutionError("Warehouse pickup start has not been calibrated")
            if warehouse.get("task1_adapter"):
                from warehouse_task1 import DEFAULT_TASK1

                source = Path(warehouse.get("task1_source") or DEFAULT_TASK1)
                if not source.is_file():
                    raise ExecutionError(f"Verified Task 1 source is missing: {source}")
                if any(a.kind == "pickup" for a in plan.actions) and not (ROOT / "models" / "best.pt").exists():
                    raise ExecutionError("Missing Task 1 cube detector model")
                if any(a.kind == "return_line" for a in plan.actions) and not (ROOT / "models" / "line_seg_mnv3.onnx").exists():
                    raise ExecutionError("Missing Task 1 lane model")
                if not 70 <= int(warehouse["drop_servo_51_deg"]) <= 110:
                    raise ExecutionError("Warehouse servo 51 drop angle is outside configured safe range")
                return
            for action in plan.actions:
                if action.kind == "place":
                    pos = warehouse["positions"].get(str(action.args["position"]))
                    if not pos or not pos.get("calibrated") or "approach_cm" not in pos or not warehouse["place_arm_pose"].get("calibrated"):
                        raise ExecutionError("Warehouse destination and arm placement need calibration")
                if action.kind == "return_line" and not warehouse["line_return"].get("calibrated"):
                    raise ExecutionError("Return-to-line path needs calibration")
            if any(a.kind == "pickup" for a in plan.actions) and not (ROOT / "models" / "best.pt").exists():
                raise ExecutionError("Missing cube detector model")
        if plan.scene == "math":
            x, y, heading = self.x_cm, self.y_cm, self.heading_deg
            total = 0.0
            self._check_pose(x, y)
            def check_turn_clearance() -> None:
                margin = max(self.cfg["robot_clearance_cm"] + self.cfg.get("motion_error_margin_cm", 0),
                             self.cfg.get("math_rotation_clearance_cm", 30))
                width, height = self.cfg["field_cm"]
                if not (margin <= x <= width - margin and margin <= y <= height - margin):
                    raise ExecutionError(f"Math rotation footprint leaves field: ({x:.1f}, {y:.1f})")
            def move_preview(direction: float, distance: float) -> None:
                nonlocal x, y, total
                x += math.sin(math.radians(direction)) * distance
                y += math.cos(math.radians(direction)) * distance
                total += distance
                self._check_pose(x, y)
            for action in plan.actions:
                a = action.args
                if action.kind == "move":
                    move_preview(heading + (180 if a["direction"] == "backward" else 0),
                                 int(math.floor(a["cm"] + 0.5)))
                elif action.kind == "strafe":
                    move_preview(heading + (-90 if a["direction"] == "left" else 90),
                                 int(math.floor(a["cm"] + 0.5)))
                elif action.kind == "turn":
                    check_turn_clearance()
                    degrees = int(math.floor(a["degrees"] + 0.5))
                    heading += degrees * (1 if a["direction"] == "right" else -1)
                elif action.kind == "spin":
                    check_turn_clearance()
                elif action.kind == "polygon":
                    side_cm = int(math.floor(a["side_cm"] + 0.5))
                    for index in range(a["sides"]):
                        move_preview(heading + (180 if a["travel"] == "backward" else 0), side_cm)
                        check_turn_clearance()
                        heading += self._polygon_turn_degrees(a["sides"], index) * (
                            1 if a["direction"] == "clockwise" else -1)
            if total > self.cfg["max_total_path_cm"]:
                raise ExecutionError("Math plan exceeds calibrated path budget")
