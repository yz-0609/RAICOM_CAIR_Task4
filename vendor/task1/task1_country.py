# ============================================================
# task1_country.py — 国赛任务一：双指令 → 两次取货、运输与卸货
# ============================================================

import argparse
import re
import sys
import time
import socket
import threading
import tomllib
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ugot import ugot
from ultralytics import YOLO
from loguru import logger as _core_logger

# ============================================================
# 配置 & 日志（内联 config.py / logger.py）
# ============================================================

_CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(_CONFIG_PATH, "rb") as _f:
    _data = tomllib.load(_f)
ROBOT_IP = _data["network"]["robot_ip"]
CONSOLE_LEVEL = _data["logging"]["console_level"]
ROBOT_INITIALIZE_RPC_TIMEOUT_SECONDS = 6.0
VOICE_BEGIN_VAD_MS = 8000
# 国赛指令包含两组任务，实车 12 秒窗口曾截掉第二组目的地区域。
# 同时放宽停顿判定，允许两组指令之间有自然停顿。
VOICE_END_VAD_MS = 2500
VOICE_LISTEN_DURATION_MS = 18000
COUNTRY_CONFIRM_SETTLE_SECONDS = 2.5

_logger_configured = False


def get_logger(script_name=None):
    global _logger_configured
    if not _logger_configured:
        _core_logger.remove()
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _core_logger.add(
            sys.stderr,
            format=lambda r: (
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level.name: <8}</level> | "
                "<level>{message}</level>"
                + (
                    " | "
                    + " ".join(
                        f"<cyan>{k}</cyan>=<level>{str(v).replace('{', '{{').replace('}', '}}')}</level>"
                        for k, v in r["extra"].items()
                    )
                    if r["extra"]
                    else ""
                )
            )
            + "\n",
            level=CONSOLE_LEVEL,
            colorize=True,
        )
        _core_logger.add(
            str(Path("logs") / f"{ts}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS!UTC} | {level.name} | {message}",
            level="TRACE",
            serialize=True,
            rotation="10 MB",
            retention="7 days",
        )
        _logger_configured = True
    return _core_logger.bind(script=script_name) if script_name else _core_logger


_log = get_logger("task1_country")

# ============================================================
# 常量
# ============================================================

# ── 语音指令 (voice_command.py) ──
COLOR_MAP = {"红色": "red", "绿色": "green", "蓝色": "blue"}
_SHORT_COLORS = ["红", "绿", "蓝"]
COLOR_CN = {"red": "红色", "green": "绿色", "blue": "蓝色"}
COUNTRY_BROADCAST_ENABLED = False
COUNTRY_BROADCAST_CHANNEL = 10
COUNTRY_BROADCAST_REPEATS = 3
COUNTRY_BROADCAST_INTERVAL_SECONDS = 0.30

# ── 自定义视觉巡线 ──
LANE_MODEL_PATH = Path(__file__).parent / "line_seg_mnv3.onnx"
LANE_REQUIRE_MODEL = True
LANE_INPUT_WIDTH = 320
LANE_INPUT_HEIGHT = 192
LANE_ROI_TOP = 0.30
LANE_MASK_THRESHOLD = 0.50
LANE_CROSS_ENTER = 0.65
LANE_CROSS_EXIT = 0.35
# 实车转弯后出现过 286～538 ms 的短暂帧延迟。
LANE_FRAME_STALE_SECONDS = 0.60
LANE_INITIAL_FRAME_TIMEOUT_SECONDS = 3.0
LANE_FRAME_RECOVERY_TIMEOUT_SECONDS = 2.5
LANE_LOST_STOP_SECONDS = 1.0
LANE_STATIC_REACQUIRE_SECONDS = 5.0
LANE_STATIC_REACQUIRE_FRAMES = 2
LANE_STATIC_REACQUIRE_CONFIDENCE = 0.50
# 1→2 是地图上唯一的长弯道。实车可能在弯道末端因视角差异短暂丢线；
# 该路段允许低速、限幅保持最后转向并持续取帧，避免原地停车后永远看不回线。
LANE_ROLLING_REACQUIRE_SPEED = 5
LANE_ROLLING_REACQUIRE_MAX_YAW = 10
LANE_ROLLING_REACQUIRE_SECONDS = 2.5
# AprilTag 将车交接到 3 号点并左转后，先静止确认 3→2 引导线。
LANE_TAG_HANDOFF_STABLE_SECONDS = 6.0
LANE_TAG_HANDOFF_STABLE_FRAMES = 3
LANE_TAG_HANDOFF_MIN_CONFIDENCE = 0.55
LANE_TAG_HANDOFF_SPEED = 12
# AprilTag 左转后车可能不在引导线正中。此处只确认“稳定看见且
# 方向可纠正”，不要求静止的车已经居中；起步后由 LaneFollower 低速纠偏。
LANE_TAG_HANDOFF_MAX_LATERAL = 1.05
LANE_TAG_HANDOFF_MAX_HEADING = 1.00
LANE_NORMAL_SPEED = 30   #20
LANE_CURVE_SPEED = 13
LANE_UNCERTAIN_SPEED = 8
LANE_MAX_YAW = 45
LANE_KP_LATERAL = 32.0
LANE_KP_HEADING = 24.0
LANE_KD_LATERAL = 3.0

# 实车验证过的路口转向状态机参数。巡线帧率约 10 FPS，底盘指令
# 限制为 5 Hz，避免摄像头与底盘 RPC 争用机器人服务。
LANE_CONTROL_HZ = 5.0
LANE_LIVE_PREVIEW = True
LANE_PREVIEW_WINDOW = "UGOT Task 1 - Live Lane Recognition"
LANE_INTERSECTION_ENTER = 0.82
LANE_INTERSECTION_EXIT = 0.68
LANE_APPROACH_SPEED = 6
LANE_TURN_ENTRY_Y = 0.90
LANE_APPROACH_TIMEOUT = 8.0
# 路口已经被多帧确认后，允许在超时时使用整段接近过程中的最强几何证据。
# 这用于摄像头安装略高、路口横线无法到达 y=0.88 的实车，不降低初次识别门槛。
LANE_APPROACH_RELAXED_ENTRY_Y = 0.68
LANE_APPROACH_RELAXED_CONFIDENCE = 0.60
LANE_APPROACH_RELAXED_CROSS = 0.65
# 已确认十字路口后，横线接近画面底部并突然消失，通常表示横线已经
# 从摄像头下方通过。实车 B 路线记录的最后有效位置为 0.776。
LANE_APPROACH_LOST_ENTRY_Y = 0.68
LANE_APPROACH_LOST_MIN_SECONDS = 2.0
# 6→4、7→5 是独立的返程短路段。离开 6/7 号旧路口后，
# 依靠前进距离+弱视觉证据提交转向，不在丢线后静止等待稳定巡线。
LANE_RETURN_SHORT_SPEED = 8
LANE_RETURN_SHORT_LOST_SPEED = 5
LANE_RETURN_SHORT_MAX_YAW = 8
LANE_RETURN_SHORT_MIN_COMMIT_CM = 4.0
LANE_RETURN_SHORT_LOST_COMMIT_SECONDS = 0.30
LANE_RETURN_SHORT_WEAK_CROSS = 0.55
LANE_RETURN_SHORT_WEAK_WINDOW = 4
LANE_RETURN_SHORT_WEAK_REQUIRED = 2
LANE_RETURN_SHORT_FALLBACK_CM = 28.0
LANE_RETURN_SHORT_BLIND_SPEED = 8
LANE_RETURN_SHORT_BLIND_OFFSET_CM_BY_ZONE = {"A": 15.0, "B": 15.0}
LANE_CENTERING_SPEED = 15
LANE_FIXED_TURN_SPEED =45    #28
LANE_TURN_SEARCH_TIMEOUT = 8.0
LANE_POST_TURN_SPEED = 8
LANE_POST_TURN_GRACE = 3.0
# 未知路线的保守默认值；正式任务使用下方的地图路段配置。
LANE_DEFAULT_SEGMENT_GUARD_SECONDS = 3.0
LANE_DEFAULT_PIVOT_OFFSET_CM = 25.0
LANE_PIVOT_OFFSET_BY_TURN = {3: 25.0}
LANE_PICKUP_ENTRY_SPEED = 13
LANE_PICKUP_ENTRY_DISTANCE_CM = 37   #40
LANE_PICKUP_ENTRY_SETTLE_SECONDS = 1.2

# 标定后将下列三项替换为实测值。None 表示保持原图/ROI 坐标。
LANE_CAMERA_MATRIX = None
LANE_DIST_COEFFS = None
LANE_BIRDSEYE_H = None

# ── 卸货区导航 (goto_zone.py) ──
STOP_DISTANCE = 8  # 巡线时到达目的地距离 cm
UNLOAD_ENTRY_SPEED = 13
UNLOAD_ENTRY_DISTANCE_CM = 26
UNLOAD_ENTRY_SETTLE_SECONDS = 0.8
TURN_SPEED = 40  # 路口转弯速度
TURN_ANGLE = 90  # 路口转弯角度
RETURN_TURN_SPEED = 38    #28
RETURN_TURN_ANGLE = 180
RETURN_TURN_SETTLE_SECONDS = 0.8

# ── AprilTag 追踪 (goto_zone.py) ──
APRILTAG_SEARCH_SPEED = 15  # 搜索旋转速度  15
APRILTAG_KP, APRILTAG_KI, APRILTAG_KD = 0.12, 0, 0.10  # 追踪 PID   0.12, 0, 0.10
APRILTAG_CHASE_SPEED = 20  # 追踪前进速度 cm/s
APRILTAG_TURN_SPEED_MAX = 30  # 追踪转弯速度上限
APRILTAG_LOCK_LOST_GRACE_SECONDS = 0.60  # 锁定后短暂丢帧不立即恢复自旋
APRILTAG_TARGET_DISTANCE = 9.3  # 追踪目标距离 cm
APRILTAG_STOP_DISTANCE = 9.5  # 追踪停止距离 cm
APRILTAG_SLOW_DISTANCE = 15  # 追踪减速距离 cm
TARGET_TAG_ID = 0  # 目标 AprilTag ID

# ── 舵机 (control_servo.py) ──
SERVO_IDS = [51, 52, 53]
DEFAULT_DURATION = 800
# 2026-09-20 实车重新标定的稳定夹取姿态。
GRAB_ARM_POSE = {51: 92, 52: 148, 53: 165}
# 夹紧后回到这一抬起姿态，避免机械臂遮挡摄像头。
ARM_CAMERA_CLEAR_POSE = {51: 90, 52: 44, 53: 144}
# 第一次卸货后转身前继续抬高肩关节，避免张开的夹爪/前臂刮到刚放下的物块。
# 52=20 来自工程旧版已使用的“抬起”位置；其余关节保持当前运输姿态。
RETURN_TURN_ARM_CLEAR_POSE = {51: 90, 52: 20, 53: 144}
RETURN_TURN_ARM_DURATION_MS = 1200
RETURN_TURN_ARM_SETTLE_SECONDS = 0.3
SERVO_TARGET_TOLERANCE = 5
SERVO_WRAP_SPEED = 12
SERVO_WRAP_TIMEOUT_SECONDS = 2.5

# ── PT 追踪 (pt_cube_chase.py) ──
SEARCH_SPEED = 30
TRACK_KP, TRACK_KI, TRACK_KD = 0.25, 0, 0.05
CHASE_SPEED = 15
DISTANCE_KP, DISTANCE_KI, DISTANCE_KD = 1.5, 0, 0.1   #2.0, 0, 0.1
BACKWARD_SPEED = 7

# ── 抓取稳定判定 (track_and_grab.py) ──
GRAB_DISTANCE_THRESHOLD = 9.3
GRAB_DISTANCE_TOLERANCE = 0.5
GRAB_OFFSET_THRESHOLD = 20
GRAB_STABLE_FRAMES = 10

# ── YOLO (pt_cube_detector.py) ──
CLASS_NAMES = ["red", "green", "blue"]
MODEL_PATH = "best.pt"
YOLO_CONF = 0.5
YOLO_IMGSZ = 640

SEP2 = "─" * 10

# ============================================================
# 工具函数
# ============================================================


def wait_port(ip, port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((ip, port), timeout=2)
            s.close()
            return True
        except OSError:
            time.sleep(1)
    return False


def parse_command(text):
    color = None
    for cn, en in COLOR_MAP.items():
        if cn in text:
            color = en
            break
    if color is None:
        for sc, en in zip(_SHORT_COLORS, ["red", "green", "blue"]):
            if sc in text:
                color = en
                break
    zone = None
    m = re.search(r"[ABab]", text)
    if m:
        zone = m.group().upper()
    return color, zone


@dataclass(frozen=True, slots=True)
class DeliveryOrder:
    color: str
    zone: str

    def __post_init__(self):
        if self.color not in COLOR_CN:
            raise ValueError(f"不支持的色块颜色: {self.color}")
        if self.zone not in ("A", "B"):
            raise ValueError(f"不支持的存储区: {self.zone}")


@dataclass(frozen=True, slots=True)
class CountryTask:
    first: DeliveryOrder
    second: DeliveryOrder
    raw_text: str = ""

    def to_official_payload(self):
        """生成官方人形机器人示例兼容的四元素广播字符串。"""
        return repr([
            COLOR_CN[self.first.color],
            self.first.zone,
            COLOR_CN[self.second.color],
            self.second.zone,
        ])


_FULLWIDTH_TRANSLATION = str.maketrans({"Ａ": "A", "Ｂ": "B", "ａ": "A", "ｂ": "B"})


def _parse_delivery_fragment(fragment):
    normalized = str(fragment).translate(_FULLWIDTH_TRANSLATION).upper()
    color = None
    for cn, en in COLOR_MAP.items():
        if cn in normalized:
            color = en
            break
    if color is None:
        for short_cn, en in zip(_SHORT_COLORS, ("red", "green", "blue")):
            if short_cn in normalized:
                color = en
                break

    zone_match = re.search(r"([AB])(?:号)?(?:存储)?区", normalized)
    if zone_match is not None:
        zone = zone_match.group(1)
    else:
        short_zone_match = re.search(r"[AB]", normalized)
        zone = short_zone_match.group(0) if short_zone_match else None
    if color is None or zone is None:
        return None
    return DeliveryOrder(color=color, zone=zone)


def parse_country_command(text):
    """从同一次 ASR 文本中按顺序解析两组国赛搬运任务。"""
    normalized = str(text or "").strip().translate(_FULLWIDTH_TRANSLATION)
    if not normalized:
        return None

    marked = re.search(
        r"第一次(?P<first>.+?)第二次(?P<second>.+)",
        normalized,
        flags=re.DOTALL,
    )
    if marked:
        first = _parse_delivery_fragment(marked.group("first"))
        second = _parse_delivery_fragment(marked.group("second"))
        if first is not None and second is not None:
            return CountryTask(first=first, second=second, raw_text=normalized)

    color_hits = []
    for match in re.finditer(r"红色?|绿色?|蓝色?", normalized):
        parsed = _parse_delivery_fragment(match.group(0) + "A区")
        color_hits.append(parsed.color)
    zone_hits = [
        match.group(1).upper()
        for match in re.finditer(r"([ABabＡＢａｂ])(?:号)?(?:存储)?区", normalized)
    ]
    if len(color_hits) != 2 or len(zone_hits) != 2:
        return None
    return CountryTask(
        first=DeliveryOrder(color_hits[0], zone_hits[0]),
        second=DeliveryOrder(color_hits[1], zone_hits[1]),
        raw_text=normalized,
    )


def transmit_country_task(
    robot,
    task,
    *,
    enabled=COUNTRY_BROADCAST_ENABLED,
    repeats=COUNTRY_BROADCAST_REPEATS,
    sleep_fn=time.sleep,
):
    """向官方频道10发送双任务；默认关闭但接口可直接启用。"""
    payload = task.to_official_payload()
    if not enabled:
        _log.bind(channel=COUNTRY_BROADCAST_CHANNEL, payload=payload).warning(
            "国赛任务通信当前未启用；双次搬运可运行，但尚不满足人形机器人信息传递规则"
        )
        return False

    successes = 0
    try:
        robot.enable_broadcast()
        robot.set_broadcast_channel(COUNTRY_BROADCAST_CHANNEL)
        for attempt in range(1, repeats + 1):
            try:
                robot.send_broadcast_message(payload)
                successes += 1
                _log.bind(
                    channel=COUNTRY_BROADCAST_CHANNEL,
                    attempt=attempt,
                    repeats=repeats,
                    payload=payload,
                ).info("已发送国赛双任务广播")
            except Exception as error:
                _log.bind(attempt=attempt, error=repr(error)).warning(
                    "国赛双任务广播发送失败"
                )
            if attempt < repeats:
                sleep_fn(COUNTRY_BROADCAST_INTERVAL_SECONDS)
    except Exception as error:
        _log.bind(error=repr(error)).error("无法启用国赛双任务广播")
    finally:
        try:
            robot.disable_broadcast()
        except Exception:
            pass
    return successes == repeats


def set_servo_position(got, servo_id, angle, duration_ms=DEFAULT_DURATION, wait=True):
    _log.bind(servo_id=servo_id, angle=angle, duration_ms=duration_ms).debug("舵机移动")
    got.turn_servo_angle(servo_id, angle, duration_ms, wait=wait)
    if wait:
        time.sleep(duration_ms / 1000.0 + 0.1)


def set_all_servo_positions(got, a1, a2, a3, duration_ms=DEFAULT_DURATION, wait=True):
    angles = [a1, a2, a3]
    _log.bind(joints=dict(zip(SERVO_IDS, angles)), duration_ms=duration_ms).debug(
        "多舵机移动"
    )
    for sid, ang in zip(SERVO_IDS, angles):
        got.turn_servo_angle(sid, ang, duration_ms, wait=False)
    if wait:
        time.sleep(duration_ms / 1000.0 + 0.1)


def move_servos_to_actual_angles(
    got,
    targets,
    duration_ms=DEFAULT_DURATION,
    tolerance=SERVO_TARGET_TOLERANCE,
):
    """按实机固件行为下发目标角度，并用硬件读数校验到位。"""
    current = {}
    for servo_id in SERVO_IDS:
        result = got.read_servo_angle(servo_id) or {}
        value = result.get(str(servo_id))
        if value is None:
            raise RuntimeError(f"无法读取舵机 {servo_id} 当前角度")
        current[servo_id] = int(value)

    _log.bind(current=current, targets=targets).info(
        "机械臂移动到实测绝对姿态"
    )

    # 编码器角度使用 [-180, 180]。例如 -171 -> +144 的
    # 物理最短路径只有 -45 度，直接下发 +144 时某些
    # 固件会按数值差走 +315 度长路径，导致机械臂向后绕行。
    # 先低速跨过编码器边界，再使用原有绝对目标。
    for servo_id in SERVO_IDS:
        target = int(targets[servo_id])
        if abs(target - current[servo_id]) <= 180:
            continue

        shortest_delta = (target - current[servo_id] + 180) % 360 - 180
        speed = SERVO_WRAP_SPEED if shortest_delta > 0 else -SERVO_WRAP_SPEED
        _log.bind(
            servo_id=servo_id,
            current=current[servo_id],
            target=target,
            shortest_delta=shortest_delta,
            speed=speed,
        ).warning("舵机目标跨过 ±180°，低速走最短路径")

        deadline = time.monotonic() + SERVO_WRAP_TIMEOUT_SECONDS
        got.turn_servo_speed(servo_id, speed)
        crossed_angle = current[servo_id]
        try:
            while time.monotonic() < deadline:
                time.sleep(0.05)
                result = got.read_servo_angle(servo_id) or {}
                value = result.get(str(servo_id))
                if value is None:
                    continue
                crossed_angle = int(value)
                if abs(target - crossed_angle) <= 180:
                    break
            else:
                raise RuntimeError(
                    f"舵机 {servo_id} 未在限时内跨过角度边界: "
                    f"current={current[servo_id]}, actual={crossed_angle}, target={target}"
                )
        finally:
            got.stop_servo(servo_id, lock=True)
        current[servo_id] = crossed_angle
        _log.bind(servo_id=servo_id, actual=crossed_angle).success(
            "已跨过舵机角度边界"
        )

    for servo_id in SERVO_IDS:
        target = int(targets[servo_id])
        got.turn_servo_angle(servo_id, target, duration_ms, wait=False)
    time.sleep(duration_ms / 1000.0 + 0.15)

    actual = {}
    for servo_id in SERVO_IDS:
        result = got.read_servo_angle(servo_id) or {}
        value = result.get(str(servo_id))
        if value is None:
            raise RuntimeError(f"无法校验舵机 {servo_id} 到位角度")
        actual[servo_id] = int(value)

    errors = {
        servo_id: (
            actual[servo_id] - int(targets[servo_id]) + 180
        ) % 360 - 180
        for servo_id in SERVO_IDS
    }
    if any(abs(error) > tolerance for error in errors.values()):
        raise RuntimeError(
            f"机械臂未到达目标姿态: target={targets}, actual={actual}, "
            f"error={errors}"
        )
    _log.bind(actual=actual, errors=errors).success("机械臂姿态已到位")
    return actual


def _discover_infrared_id(robot):
    devices = robot.get_peripheral_devices_list()
    for dev in devices:
        if dev.get("type") == "Infrared":
            return int(dev.get("deviceId"))
    _log.warning("未发现红外传感器，使用默认 ID 41")
    return 41


# ============================================================
# 自定义赛道感知与闭环控制
# ============================================================


class DriveMode(str, Enum):
    TRACK = "TRACK"
    TURN = "TURN"
    RECOVER = "RECOVER"
    STOP = "STOP"


class RouteAction(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT = "STRAIGHT"
    ARRIVE = "ARRIVE"
    RIGHT_AND_ARRIVE = "RIGHT_AND_ARRIVE"


@dataclass(frozen=True, slots=True)
class LaneSegmentProfile:
    """两个地图路口之间的巡线策略。

    min_event_seconds 表示离开上一个路口后，最早何时允许
    消费下一个路口动作。post_turn_preclaim 用于 2→3、2→4、
    4→6、5→7 这类右转后路口分数连续、无法依赖低分重置的路段；
    stabilize_before_start
    用于 AprilTag 把车交接到 3 号点后，静止稳定捕获 3→2 引导线。
    speed_limit 只限制该路段的普通巡线速度。
    reacquire_seconds 可为容易短暂丢线的单一路段延长静止重捕。
    rolling_reacquire_seconds 仅用于地图明确且前方无岔路的路段；丢线时
    低速保持最后方向继续取帧，超时后仍会停车报错。
    """

    name: str
    min_event_seconds: float
    post_turn_preclaim: bool = False
    stabilize_before_start: bool = False
    speed_limit: int | None = None
    reacquire_seconds: float | None = None
    rolling_reacquire_seconds: float | None = None
    # 只用于地图上明确的“直行后紧邻下一路口”路段。
    # 允许用横线从画面底部回到远处的几何变化重新启用检测，
    # 不强制分类分数必须先连续降低。
    straight_exit_relaxed: bool = False
    # 仅用于返程 6→4、7→5；字母同时选择 A/B 独立盲走标定。
    short_turn_zone: str | None = None


@dataclass(frozen=True, slots=True)
class ShortRouteTurnProfile:
    """6→4 / 7→5 专用的提交与盲走参数。"""

    zone: str
    segment_name: str
    min_commit_cm: float = LANE_RETURN_SHORT_MIN_COMMIT_CM
    fallback_cm: float = LANE_RETURN_SHORT_FALLBACK_CM
    blind_offset_cm: float = 15.0

    @property
    def blind_seconds(self):
        return self.blind_offset_cm / LANE_RETURN_SHORT_BLIND_SPEED


SHORT_ROUTE_TURN_PROFILE_BY_ZONE = {
    zone: ShortRouteTurnProfile(
        zone=zone,
        segment_name="6->4" if zone == "A" else "7->5",
        blind_offset_cm=offset,
    )
    for zone, offset in LANE_RETURN_SHORT_BLIND_OFFSET_CM_BY_ZONE.items()
}


@dataclass(slots=True)
class ShortRouteProgress:
    """用已发送的前进速度和单调时钟累计短路段距离。"""

    profile: ShortRouteTurnProfile
    started_at: float
    distance_cm: float = 0.0
    last_update: float | None = None
    commanded_speed: float = 0.0
    lost_since: float | None = None
    commit_reason: str | None = None
    weak_window: deque = field(
        default_factory=lambda: deque(maxlen=LANE_RETURN_SHORT_WEAK_WINDOW)
    )

    def __post_init__(self):
        if self.last_update is None:
            self.last_update = self.started_at

    def advance(self, now):
        elapsed = max(0.0, now - self.last_update)
        self.distance_cm += max(0.0, self.commanded_speed) * elapsed
        self.last_update = now

    def set_commanded_speed(self, speed, now):
        self.advance(now)
        self.commanded_speed = max(0.0, float(speed))

    def pause(self, now):
        self.set_commanded_speed(0.0, now)

    def resume(self, now):
        self.last_update = now
        self.commanded_speed = 0.0

    def observe(self, observation, now):
        self.weak_window.append(
            not observation.lost
            and observation.confidence >= 0.50
            and observation.cross_score >= LANE_RETURN_SHORT_WEAK_CROSS
        )
        if observation.lost:
            if self.lost_since is None:
                self.lost_since = now
        else:
            self.lost_since = None

    def lost_seconds(self, now):
        return 0.0 if self.lost_since is None else max(0.0, now - self.lost_since)

    def evaluate_commit(self, now):
        if self.distance_cm < self.profile.min_commit_cm:
            return None
        if sum(self.weak_window) >= LANE_RETURN_SHORT_WEAK_REQUIRED:
            return "weak_visual"
        if self.lost_seconds(now) >= LANE_RETURN_SHORT_LOST_COMMIT_SECONDS:
            return "expected_loss"
        if self.distance_cm >= self.profile.fallback_cm:
            return "distance_fallback"
        return None

    @property
    def weak_frames(self):
        return sum(self.weak_window)


# 省赛任务一的显式路线表。识别到十字路口只负责消费下一个动作，
# 绝不在感知层默认“见路口就右转”。
START_TO_PICKUP_ROUTE = (
    RouteAction.RIGHT,
    RouteAction.RIGHT,
    RouteAction.RIGHT_AND_ARRIVE,
)
UNLOAD_ROUTE_BY_ZONE = {
    "A": (
        RouteAction.RIGHT,
        RouteAction.RIGHT,
        RouteAction.ARRIVE,
    ),
    "B": (
        RouteAction.RIGHT,
        RouteAction.STRAIGHT,
        RouteAction.RIGHT,
        RouteAction.ARRIVE,
    ),
}

# 第一次卸货后从 A/B 高台反向返回 3 号取货区。
RETURN_TO_PICKUP_ROUTE_BY_ZONE = {
    "A": (
        RouteAction.STRAIGHT,
        RouteAction.LEFT,
        RouteAction.LEFT,
        RouteAction.RIGHT_AND_ARRIVE,
    ),
    "B": (
        RouteAction.STRAIGHT,
        RouteAction.LEFT,
        RouteAction.STRAIGHT,
        RouteAction.LEFT,
        RouteAction.RIGHT_AND_ARRIVE,
    ),
}

# 配置顺序与上方动作顺序一一对应：每项表示“到达并执行
# 该动作所在路口”的入边。时间来自赛道地图与实车日志，
# 同时依然需要多帧路口感知确认。
# 2026-09-20：用户确认当前整套任务流程已完成一次实车成功运行。
ROUTE_SEGMENTS_BY_NAME = {
    "start-to-pickup": (
        LaneSegmentProfile("start->1", 0.0),
        LaneSegmentProfile(
            "1->2",
            8.0,
            rolling_reacquire_seconds=LANE_ROLLING_REACQUIRE_SECONDS,
        ),
        LaneSegmentProfile("2->3", 0.4, post_turn_preclaim=True),
    ),
    "pickup-to-A": (
        LaneSegmentProfile(
            "3->2", 0.0,
            stabilize_before_start=True,
            speed_limit=LANE_TAG_HANDOFF_SPEED,
        ),
        LaneSegmentProfile("2->4", 2.5, post_turn_preclaim=True),
        LaneSegmentProfile("4->6", 2.0, post_turn_preclaim=True),
    ),
    "pickup-to-B": (
        LaneSegmentProfile(
            "3->2", 0.0,
            stabilize_before_start=True,
            speed_limit=LANE_TAG_HANDOFF_SPEED,
        ),
        LaneSegmentProfile("2->4", 2.5, post_turn_preclaim=True),
        LaneSegmentProfile(
            "4->5",
            2.5,
            reacquire_seconds=6.0,
            straight_exit_relaxed=True,
        ),
        LaneSegmentProfile("5->7", 2.0, post_turn_preclaim=True),
    ),
    "return-A-to-pickup": (
        LaneSegmentProfile(
            "A->6", 0.0,
            stabilize_before_start=True,
            speed_limit=LANE_TAG_HANDOFF_SPEED,
        ),
        LaneSegmentProfile(
            "6->4",
            1.0,
            speed_limit=LANE_RETURN_SHORT_SPEED,
            straight_exit_relaxed=True,
            short_turn_zone="A",
        ),
        LaneSegmentProfile("4->2", 2.5, post_turn_preclaim=True),
        LaneSegmentProfile("2->3", 0.4, post_turn_preclaim=True),
    ),
    "return-B-to-pickup": (
        LaneSegmentProfile(
            "B->7", 0.0,
            stabilize_before_start=True,
            speed_limit=LANE_TAG_HANDOFF_SPEED,
        ),
        LaneSegmentProfile(
            "7->5",
            1.0,
            speed_limit=LANE_RETURN_SHORT_SPEED,
            straight_exit_relaxed=True,
            short_turn_zone="B",
        ),
        # B 返程在 5 号口左转完成后，实车中 4 号口会立即进入画面，
        # 并在约 2.6 s 内驶出。原 2.5 s 屏蔽期会漏掉整个 4 号口，
        # 导致把 2 号当成 4 号、再把 1 号当成 2 号。
        LaneSegmentProfile("5->4", 1.0, post_turn_preclaim=True),
        LaneSegmentProfile("4->2", 2.5, straight_exit_relaxed=True),
        LaneSegmentProfile("2->3", 0.4, post_turn_preclaim=True),
    ),
}


def _segment_target_intersection(segment):
    """从地图路段名中返回图上标注的目标十字路口编号。"""
    try:
        return int(segment.name.rsplit("->", 1)[1])
    except (AttributeError, IndexError, ValueError):
        return None


def _post_turn_can_preclaim(segment, next_action, elapsed, positive_frames):
    """只在地图明确标记的连续路口段上提前准备下一动作。"""
    if segment is None or not segment.post_turn_preclaim:
        return False
    if elapsed < segment.min_event_seconds or positive_frames < 2:
        return False
    return next_action in (
        RouteAction.LEFT,
        RouteAction.RIGHT,
        RouteAction.STRAIGHT,
        RouteAction.RIGHT_AND_ARRIVE,
        RouteAction.ARRIVE,
    )


def _turn_direction_for_action(action):
    """把地图转向动作映射为 UGOT SDK 的方向编号。"""
    if action == RouteAction.LEFT:
        return 2
    if action in (RouteAction.RIGHT, RouteAction.RIGHT_AND_ARRIVE):
        return 3
    raise ValueError(f"动作不是转向动作: {action}")


@dataclass(slots=True)
class LaneObservation:
    timestamp: float
    confidence: float
    lateral_error: float
    heading_error: float
    curvature: float
    cross_score: float
    intersection_y: float
    lost: bool
    source: str
    mask: np.ndarray | None = None
    # Keep the exact camera frame used for this observation so the formal task
    # can display what the controller saw, rather than a newer unrelated frame.
    frame: np.ndarray | None = None


@dataclass(slots=True)
class DriveCommand:
    forward_speed: int
    yaw_speed: int
    mode: DriveMode
    cross_event: bool = False


class LatestFrameReader:
    """Continuously read the robot camera while retaining only the newest frame."""

    def __init__(self, robot):
        self.robot = robot
        self._condition = threading.Condition()
        self._latest = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="lane-camera-reader", daemon=True
        )

    def start(self):
        self.robot.open_camera()
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                data = self.robot.read_camera_data()
                if not data:
                    self._stop.wait(0.01)
                    continue
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                item = (time.monotonic(), frame)
                with self._condition:
                    self._latest = item
                    self._condition.notify_all()
            except Exception:
                _log.opt(exception=True).warning("巡线摄像头读取异常")
                self._stop.wait(0.03)

    def get_latest(self, after_timestamp=0.0, timeout=0.25):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stop.is_set():
                if self._latest is not None and self._latest[0] > after_timestamp:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def stop(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class LanePerception:
    """ONNX lane segmentation with deterministic geometric post-processing."""

    def __init__(self, model_path=LANE_MODEL_PATH, require_model=LANE_REQUIRE_MODEL):
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = None
        self.output_names = []
        self.previous_center = LANE_INPUT_WIDTH / 2
        self.last_latency_ms = 0.0

        if self.model_path.exists():
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) // 2))
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [item.name for item in self.session.get_outputs()]
            # Warm up graph optimization before timestamps become safety-critical.
            warmup = np.zeros(
                (1, 3, LANE_INPUT_HEIGHT, LANE_INPUT_WIDTH), dtype=np.float32
            )
            for _ in range(3):
                self.session.run(self.output_names, {self.input_name: warmup})
            _log.bind(
                model=str(self.model_path), outputs=self.output_names
            ).success("自定义巡线 ONNX 模型加载完成")
        elif require_model:
            raise FileNotFoundError(
                f"缺少比赛巡线模型: {self.model_path}. "
                "请先用 lane_training/train.py 训练并导出模型。"
            )
        else:
            _log.warning("未找到 ONNX 模型，仅启用低速 OpenCV 降级感知")

    @property
    def model_ready(self):
        return self.session is not None

    @staticmethod
    def _sigmoid(value):
        value = np.clip(value, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-value))

    def _prepare_frame(self, frame):
        if LANE_CAMERA_MATRIX is not None and LANE_DIST_COEFFS is not None:
            frame = cv2.undistort(
                frame,
                np.asarray(LANE_CAMERA_MATRIX, dtype=np.float32),
                np.asarray(LANE_DIST_COEFFS, dtype=np.float32),
            )
        roi_top = int(round(frame.shape[0] * LANE_ROI_TOP))
        roi = frame[roi_top:, :]
        resized = cv2.resize(
            roi, (LANE_INPUT_WIDTH, LANE_INPUT_HEIGHT), interpolation=cv2.INTER_AREA
        )
        return resized

    def _run_model(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        started = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        mask_output = outputs[0]
        for name, value in zip(self.output_names, outputs):
            if "mask" in name.lower():
                mask_output = value
                break
        mask_logits = np.asarray(mask_output).squeeze()
        if mask_logits.shape != (LANE_INPUT_HEIGHT, LANE_INPUT_WIDTH):
            mask_logits = cv2.resize(
                mask_logits,
                (LANE_INPUT_WIDTH, LANE_INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        probabilities = self._sigmoid(mask_logits).astype(np.float32)

        cross_probability = 0.0
        for name, value in zip(self.output_names, outputs):
            if "cross" not in name.lower():
                continue
            logits = np.asarray(value).reshape(-1)
            if logits.size == 1:
                cross_probability = float(self._sigmoid(logits[0]))
            elif logits.size >= 2:
                shifted = logits - logits.max()
                exp = np.exp(shifted)
                cross_probability = float(exp[-1] / exp.sum())
            break
        return probabilities, cross_probability

    @staticmethod
    def _opencv_fallback(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.bitwise_and(adaptive, otsu)
        return mask.astype(np.float32) / 255.0

    def _select_lane_component(self, binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, 8)
        if count <= 1:
            return np.zeros_like(binary)

        height, width = binary.shape
        best_label = 0
        best_score = -1e9
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < height * width * 0.002 or area > height * width * 0.60:
                continue
            bottom_gap = height - (y + h)
            bottom_score = max(0.0, 1.0 - bottom_gap / max(1.0, height * 0.35))
            centre_distance = abs(float(centroids[label][0]) - self.previous_center) / width
            area_score = min(1.0, area / (height * width * 0.08))
            score = 2.2 * bottom_score + area_score - 1.4 * centre_distance
            if score > best_score:
                best_score = score
                best_label = label
        return ((labels == best_label).astype(np.uint8) * 255) if best_label else np.zeros_like(binary)

    @staticmethod
    def _row_runs(xs):
        if xs.size == 0:
            return []
        split_at = np.where(np.diff(xs) > 3)[0] + 1
        return [part for part in np.split(xs, split_at) if part.size]

    def _geometry(self, lane_mask, probability_mask):
        if LANE_BIRDSEYE_H is not None:
            lane_mask = cv2.warpPerspective(
                lane_mask,
                np.asarray(LANE_BIRDSEYE_H, dtype=np.float32),
                (LANE_INPUT_WIDTH, LANE_INPUT_HEIGHT),
                flags=cv2.INTER_NEAREST,
            )
            probability_mask = cv2.warpPerspective(
                probability_mask,
                np.asarray(LANE_BIRDSEYE_H, dtype=np.float32),
                (LANE_INPUT_WIDTH, LANE_INPUT_HEIGHT),
                flags=cv2.INTER_LINEAR,
            )

        height, width = lane_mask.shape
        sample_rows = np.linspace(height - 8, int(height * 0.30), 12).astype(int)
        points = []
        widths = []
        predicted_x = self.previous_center

        for row in sample_rows:
            y1, y2 = max(0, row - 2), min(height, row + 3)
            xs = np.where(np.any(lane_mask[y1:y2] > 0, axis=0))[0]
            runs = self._row_runs(xs)
            if not runs:
                continue
            chosen = min(runs, key=lambda run: abs(float(np.median(run)) - predicted_x))
            run_center = float(np.median(chosen))
            total_width = float(xs[-1] - xs[0] + 1)
            widths.append((row, total_width / width))
            # Wide rows are the horizontal arm of an intersection; preserve the
            # incoming-line prediction instead of pulling the fit sideways.
            if total_width / width < 0.42:
                points.append((float(row), run_center))
                predicted_x = run_center

        if len(points) < 4:
            return 0.0, 0.0, 0.0, 0.0, 0.0, True

        pts = np.asarray(points, dtype=np.float64)
        initial = np.polyfit(pts[:, 0], pts[:, 1], 2)
        residual = np.abs(np.polyval(initial, pts[:, 0]) - pts[:, 1])
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median))) + 1e-6
        keep = residual <= median + 3.5 * mad
        if int(keep.sum()) >= 4:
            pts = pts[keep]
        weights = np.linspace(1.5, 0.7, len(pts))
        coeff = np.polyfit(pts[:, 0], pts[:, 1], 2, w=weights)

        near_y = height * 0.86
        lookahead_y = height * 0.58
        near_x = float(np.polyval(coeff, near_y))
        slope = float(2.0 * coeff[0] * lookahead_y + coeff[1])
        lateral_error = float(np.clip((near_x - width / 2) / (width / 2), -1.5, 1.5))
        # Image y grows toward the robot, hence the minus sign.
        heading_error = float(np.clip(-np.arctan(slope) / (np.pi / 4), -1.5, 1.5))
        curvature = float(np.clip(abs(2.0 * coeff[0]) * height, 0.0, 1.0))
        self.previous_center = float(np.clip(near_x, 0, width - 1))

        max_width = max((ratio for _, ratio in widths), default=0.0)
        geometry_cross = float(np.clip((max_width - 0.28) / 0.50, 0.0, 1.0))
        intersection_y = 0.0
        if widths:
            widest_row = max(widths, key=lambda item: item[1])[0]
            intersection_y = float(widest_row / height)

        selected = lane_mask > 0
        mean_probability = float(probability_mask[selected].mean()) if selected.any() else 0.0
        row_coverage = min(1.0, len(points) / 9.0)
        confidence = float(np.clip(0.65 * mean_probability + 0.35 * row_coverage, 0, 1))
        return (
            lateral_error,
            heading_error,
            curvature,
            geometry_cross,
            intersection_y,
            confidence < 0.18,
        )

    def analyze(self, frame, frame_timestamp=None):
        image = self._prepare_frame(frame)
        if self.session is not None:
            probabilities, model_cross = self._run_model(image)
            source = "onnx"
        else:
            probabilities = self._opencv_fallback(image)
            model_cross = 0.0
            source = "opencv-fallback"
            self.last_latency_ms = 0.0

        binary = (probabilities >= LANE_MASK_THRESHOLD).astype(np.uint8) * 255
        lane_mask = self._select_lane_component(binary)
        lateral, heading, curvature, geometry_cross, intersection_y, lost = (
            self._geometry(lane_mask, probabilities)
        )
        if self.session is None:
            cross_score = geometry_cross
        else:
            cross_score = 0.6 * model_cross + 0.4 * geometry_cross

        selected = lane_mask > 0
        probability_confidence = (
            float(probabilities[selected].mean()) if selected.any() else 0.0
        )
        coverage = min(1.0, np.count_nonzero(lane_mask) / (lane_mask.size * 0.06))
        confidence = float(np.clip(0.75 * probability_confidence + 0.25 * coverage, 0, 1))
        if source != "onnx":
            confidence = min(confidence, 0.55)

        return LaneObservation(
            timestamp=frame_timestamp if frame_timestamp is not None else time.monotonic(),
            confidence=confidence,
            lateral_error=lateral,
            heading_error=heading,
            curvature=curvature,
            cross_score=float(np.clip(cross_score, 0, 1)),
            intersection_y=intersection_y,
            lost=lost,
            source=source,
            mask=lane_mask,
            frame=frame,
        )


class LaneFollower:
    def __init__(self):
        self.cross_window = deque(maxlen=5)
        self.cross_exit_window = deque(maxlen=4)
        self.cross_armed = True
        self.lost_since = None
        self.last_time = None
        self.last_lateral = 0.0
        self.filtered_lateral = 0.0
        self.last_command = DriveCommand(0, 0, DriveMode.STOP)

    def reset_after_intersection(self):
        self.cross_window.clear()
        self.cross_exit_window.clear()
        self.cross_armed = False

    def step(self, observation, route_state=""):
        now = time.monotonic()
        if now - observation.timestamp > LANE_FRAME_STALE_SECONDS:
            self.last_command = DriveCommand(0, 0, DriveMode.STOP)
            return self.last_command

        self.cross_window.append(observation.cross_score >= LANE_CROSS_ENTER)
        self.cross_exit_window.append(observation.cross_score < LANE_CROSS_EXIT)
        cross_event = False
        if self.cross_armed and len(self.cross_window) == 5 and sum(self.cross_window) >= 3:
            cross_event = True
            self.cross_armed = False
        elif (
            not self.cross_armed
            and len(self.cross_exit_window) == self.cross_exit_window.maxlen
            and all(self.cross_exit_window)
        ):
            self.cross_armed = True
            self.cross_window.clear()

        if observation.lost:
            if self.lost_since is None:
                self.lost_since = now
            lost_duration = now - self.lost_since
            if lost_duration >= LANE_LOST_STOP_SECONDS:
                command = DriveCommand(0, 0, DriveMode.STOP, cross_event)
            else:
                command = DriveCommand(0, 0, DriveMode.RECOVER, cross_event)
            self.last_command = command
            return command

        self.lost_since = None
        dt = max(0.02, min(0.20, now - self.last_time)) if self.last_time else 0.05
        self.filtered_lateral = 0.65 * self.filtered_lateral + 0.35 * observation.lateral_error
        derivative = (self.filtered_lateral - self.last_lateral) / dt
        yaw = -(
            LANE_KP_LATERAL * self.filtered_lateral
            + LANE_KP_HEADING * observation.heading_error
            + LANE_KD_LATERAL * derivative
        )
        yaw = int(round(np.clip(yaw, -LANE_MAX_YAW, LANE_MAX_YAW)))

        if observation.source != "onnx" or observation.confidence < 0.45:
            speed = LANE_UNCERTAIN_SPEED
        elif observation.curvature > 0.20 or abs(observation.heading_error) > 0.35:
            speed = LANE_CURVE_SPEED
        else:
            speed = LANE_NORMAL_SPEED
        if observation.cross_score >= 0.45:
            speed = min(speed, 10)

        self.last_time = now
        self.last_lateral = self.filtered_lateral
        self.last_command = DriveCommand(speed, yaw, DriveMode.TRACK, cross_event)
        return self.last_command


class LaneRouteState(str, Enum):
    """正式任务的巡线路由状态。

    路口上的具体动作仍由 RouteAction 决定；该状态机只负责安全地
    完成一次需要左/右转的路口，不会将所有十字路口默认为转向。
    """

    TRACK = "TRACK"
    APPROACH = "APPROACH"
    SHORT_ROUTE_APPROACH = "SHORT_ROUTE_APPROACH"
    SHORT_BLIND_CENTER = "SHORT_BLIND_CENTER"
    CENTER_PIVOT = "CENTER_PIVOT"
    TURN_90 = "TURN_90"
    SEARCH_LANE = "SEARCH_LANE"
    POST_TURN = "POST_TURN"


class IntersectionLatch:
    """多帧确认路口，并在离开当前路口前禁止重复触发。"""

    def __init__(self, enter=LANE_INTERSECTION_ENTER, exit_threshold=LANE_INTERSECTION_EXIT):
        self.enter = enter
        self.exit = exit_threshold
        self.window = deque(maxlen=6)
        self.exit_window = deque(maxlen=6)
        self.armed = True
        self.cooldown_until = 0.0
        self.straight_exit = False
        self.bottom_frames = 0
        self.saw_bottom = False
        self.departed_frames = 0

    def update(
        self,
        observation,
        allow_trigger=True,
        relaxed_straight_exit=False,
        enter_threshold=None,
        positive_frames_required=4,
    ):
        now = time.monotonic()
        active_enter = self.enter if enter_threshold is None else enter_threshold
        positive = (
            allow_trigger
            and not observation.lost
            and observation.confidence >= 0.60
            and observation.cross_score >= active_enter
        )
        self.window.append(positive)
        self.exit_window.append(observation.cross_score < self.exit)

        if not self.armed:
            # 直行时横线连续到达底部，再退到远处/消失，表示已驶离。
            # 分类头在相邻路口间可能持续高分，因此同时检查几何变化。
            valid = not observation.lost and observation.confidence >= 0.60
            if self.straight_exit:
                self.bottom_frames = self.bottom_frames + 1 if (
                    valid and observation.intersection_y >= 0.88
                ) else 0
                bottom_frames_required = 1 if relaxed_straight_exit else 2
                heading_limit = 0.70 if relaxed_straight_exit else 0.40
                self.saw_bottom |= self.bottom_frames >= bottom_frames_required
                self.departed_frames = self.departed_frames + 1 if (
                    self.saw_bottom and valid
                    and observation.intersection_y < 0.55
                    and abs(observation.heading_error) < heading_limit
                ) else 0
            if (
                now >= self.cooldown_until
                and (
                    (len(self.exit_window) == self.exit_window.maxlen
                     and all(self.exit_window))
                    or (self.straight_exit and self.departed_frames >= 3)
                )
            ):
                self.armed = True
                self.window.clear()
                if self.straight_exit:
                    _log.info("已确认驶离直行路口，重新启用下一路口检测")
                self.straight_exit = False
            return False

        if (
            len(self.window) == self.window.maxlen
            and sum(self.window) >= positive_frames_required
        ):
            self.armed = False
            self.window.clear()
            return True
        return False

    def completed_turn(self):
        self.armed = False
        self.window.clear()
        self.exit_window.clear()
        self.cooldown_until = time.monotonic() + 1.2
        self.straight_exit = False
        self.bottom_frames = 0
        self.saw_bottom = False
        self.departed_frames = 0

    def started_straight(self, observation=None):
        """锁定当前直行路口，同时保留横线已经到达画面底部的证据。

        4→2 等短路段中，路线动作可能正好在横线即将离开画面时才被消费。
        若这里把 saw_bottom 清零，分类头持续高分时锁会一直关闭，直到越过
        下一个路口。保留当前帧的底部几何后，仍需三帧“退回远处”才能重开，
        因而不会把当前路口重复消费。
        """
        self.completed_turn()
        self.straight_exit = True
        if (
            observation is not None
            and not observation.lost
            and observation.confidence >= 0.60
            and observation.intersection_y >= 0.88
        ):
            self.saw_bottom = True
            self.bottom_frames = 1

def _turn_mask_alignment(mask):
    """固定 90° 转向后，用较短的掩膜线段辅助重新捕获赛道。"""
    if mask is None or mask.ndim != 2 or not np.any(mask):
        return False, 0.0, 0.0

    height, width = mask.shape
    sample_rows = np.linspace(height - 5, int(height * 0.25), 24).astype(int)
    points = []
    centre_x = width / 2.0
    for row in sample_rows:
        y1, y2 = max(0, row - 2), min(height, row + 3)
        xs = np.where(np.any(mask[y1:y2] > 0, axis=0))[0]
        if xs.size == 0:
            continue
        split_at = np.where(np.diff(xs) > 3)[0] + 1
        runs = [run for run in np.split(xs, split_at) if run.size]
        chosen = min(runs, key=lambda run: abs(float(np.median(run)) - centre_x))
        points.append((float(row), float(np.median(chosen))))

    if len(points) < 2:
        return False, 0.0, 0.0
    pts = np.asarray(points, dtype=np.float64)
    if float(np.ptp(pts[:, 0])) < height * 0.08:
        return False, 0.0, 0.0

    slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
    near_x = float(slope * (height * 0.86) + intercept)
    lateral = float(np.clip((near_x - centre_x) / centre_x, -1.5, 1.5))
    heading = float(np.clip(-np.arctan(slope) / (np.pi / 4), -1.5, 1.5))
    return True, lateral, heading


# ============================================================
# 阶段 1 — 语音指令
# ============================================================


def voice_command_phase(robot):
    _log.info("正在监听语音...")
    try:
        resp = robot.AUDIO.setAudioAsr(
            begin_vad=VOICE_BEGIN_VAD_MS,
            end_vad=VOICE_END_VAD_MS,
            duration=VOICE_LISTEN_DURATION_MS,
        )
        _log.bind(code=resp.code, msg=resp.msg, data=resp.data).info("ASR 原始响应")
        text = resp.data.strip() if resp.code == 0 and resp.data else ""
    except Exception:
        _log.opt(exception=True).error("语音识别异常")
        return None, None

    if not text:
        _log.warning("未识别到语音")
        robot.play_audio_tts("未识别到语音，请重试", 0, wait=True)
        return None, None

    _log.bind(raw=text).success("语音识别结果")
    color, zone = parse_command(text)
    _log.bind(color=color, zone=zone).info("解析结果")
    return color, zone


def country_voice_command_phase(robot):
    """监听一次完整国赛双任务指令，不与其他轮次拼接。"""
    _log.info("正在监听国赛双任务语音...")
    try:
        response = robot.AUDIO.setAudioAsr(
            begin_vad=VOICE_BEGIN_VAD_MS,
            end_vad=VOICE_END_VAD_MS,
            duration=VOICE_LISTEN_DURATION_MS,
        )
        _log.bind(
            code=response.code,
            msg=response.msg,
            data=response.data,
        ).info("ASR 原始响应")
        text = response.data.strip() if response.code == 0 and response.data else ""
    except Exception:
        _log.opt(exception=True).error("国赛双任务语音识别异常")
        return None

    if not text:
        _log.warning("未识别到国赛双任务语音")
        return None
    task = parse_country_command(text)
    if task is None:
        _log.bind(raw=text).warning("同一次语音中未解析出两组完整搬运任务")
        return None
    _log.bind(
        raw=text,
        first_color=task.first.color,
        first_zone=task.first.zone,
        second_color=task.second.color,
        second_zone=task.second.zone,
    ).success("国赛双任务语音解析完成")
    return task


def collect_country_task(robot, max_attempts=2):
    """最多重听整句一次；不会把两次残缺 ASR 结果拼成任务。"""
    for attempt in range(1, max_attempts + 1):
        task = country_voice_command_phase(robot)
        if task is not None:
            return task
        if attempt < max_attempts:
            robot.play_audio_tts("未完整识别两次搬运任务，请重新说完整指令", 0, wait=True)
            time.sleep(0.5)
    return None


def country_confirmation_text(task):
    return (
        f"好的，我第一次去搬运{COLOR_CN[task.first.color]}色块放在"
        f"{task.first.zone}号存储区，第二次去搬运"
        f"{COLOR_CN[task.second.color]}色块放在{task.second.zone}号存储区"
    )


def speak_country_confirmation(robot, task, sleep_fn=time.sleep):
    """完整播报两次任务，并留出固件音频尾部播放时间。"""
    confirmation = country_confirmation_text(task)
    robot.play_audio_tts(confirmation, 0, wait=True)
    sleep_fn(COUNTRY_CONFIRM_SETTLE_SECONDS)
    _log.success(confirmation)
    return confirmation


# ============================================================
# 阶段 2 — 自定义视觉巡线
# ============================================================


def render_lane_live_preview(
    frame,
    observation,
    *,
    route_name,
    route_state,
    segment_name,
    map_intersection,
    action_index,
    action_count,
    inference_ms,
    note="",
):
    """绘制正式任务中控制器实际使用的巡线识别结果。"""
    output = frame.copy()
    height, width = output.shape[:2]
    roi_top = int(round(height * LANE_ROI_TOP))
    roi_height = max(1, height - roi_top)

    if observation.mask is not None:
        mask = cv2.resize(
            observation.mask,
            (width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay = output[roi_top:].copy()
        mask_colour = (20, 20, 230) if observation.lost else (40, 230, 40)
        overlay[mask > 0] = mask_colour
        output[roi_top:] = cv2.addWeighted(
            output[roi_top:], 0.60, overlay, 0.40, 0
        )

    centre_x = width // 2
    detected_x = int(round(centre_x + observation.lateral_error * centre_x))
    detected_x = int(np.clip(detected_x, 0, width - 1))
    cv2.line(output, (centre_x, roi_top), (centre_x, height - 1), (255, 255, 255), 1)
    cv2.line(output, (detected_x, roi_top), (detected_x, height - 1), (255, 180, 0), 2)
    cv2.line(output, (0, roi_top), (width - 1, roi_top), (180, 180, 180), 1)

    if observation.intersection_y > 0:
        cross_y = roi_top + int(round(observation.intersection_y * roi_height))
        cross_y = int(np.clip(cross_y, roi_top, height - 1))
        cv2.line(output, (0, cross_y), (width - 1, cross_y), (0, 220, 255), 2)

    state_text = getattr(route_state, "value", str(route_state))
    status_colour = (20, 20, 255) if observation.lost else (40, 230, 40)
    frame_age_ms = max(0.0, (time.monotonic() - observation.timestamp) * 1000.0)
    lines = (
        f"route={route_name}  state={state_text}  segment={segment_name}",
        f"MAP INTERSECTION={map_intersection or '?'}  route_step={action_index}/{action_count}  source={observation.source}  LOST={observation.lost}",
        f"conf={observation.confidence:.3f}  cross={observation.cross_score:.3f}  cross_y={observation.intersection_y:.3f}",
        f"lateral={observation.lateral_error:+.3f}  heading={observation.heading_error:+.3f}  curve={observation.curvature:.3f}",
        f"inference={inference_ms:.1f}ms  frame_age={frame_age_ms:.0f}ms  {note}",
        "Green=selected lane  Yellow=cross position  Q/Esc=EMERGENCY STOP",
    )
    panel_bottom = 12 + len(lines) * 24
    cv2.rectangle(output, (8, 8), (min(width - 8, 790), panel_bottom), (0, 0, 0), -1)
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (18, 31 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            status_colour if index == 1 else (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return output


def _read_lane_observation(reader, perception, after_timestamp, timeout=0.30):
    item = reader.get_latest(after_timestamp=after_timestamp, timeout=timeout)
    if item is None:
        return None, after_timestamp
    frame_timestamp, frame = item
    return perception.analyze(frame, frame_timestamp), frame_timestamp


def _wait_fresh_lane_observation(reader, perception, after_timestamp, timeout):
    """在总等待期限内跳过旧帧；调用方须先停车。"""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, after_timestamp
        observation, after_timestamp = _read_lane_observation(
            reader, perception, after_timestamp, timeout=remaining
        )
        if observation is None:
            return None, after_timestamp
        if time.monotonic() - observation.timestamp <= LANE_FRAME_STALE_SECONDS:
            return observation, after_timestamp


def _reacquire_lane_statically(
    robot,
    reader,
    perception,
    after_timestamp,
    timeout_seconds=None,
    preview_callback=None,
):
    """丢线时停车，以连续新帧确认恢复；绝不盲目前进搜索。"""
    robot.stop_chassis()
    timeout_seconds = (
        LANE_STATIC_REACQUIRE_SECONDS
        if timeout_seconds is None
        else float(timeout_seconds)
    )
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    saw_fresh_frame = False
    while time.monotonic() < deadline:
        observation, after_timestamp = _wait_fresh_lane_observation(
            reader, perception, after_timestamp,
            timeout=max(0.0, deadline - time.monotonic()),
        )
        if observation is None:
            if saw_fresh_frame:
                break
            raise RuntimeError("静止重捕期间视频未恢复，已保持停车")
        saw_fresh_frame = True
        if preview_callback is not None:
            preview_callback(observation, "STATIC REACQUIRE")
        stable = stable + 1 if (
            not observation.lost
            and observation.confidence >= LANE_STATIC_REACQUIRE_CONFIDENCE
        ) else 0
        if stable >= LANE_STATIC_REACQUIRE_FRAMES:
            return observation, after_timestamp
    raise RuntimeError(
        "画面有效但持续未检测到引导线，"
        f"静止重捕 {timeout_seconds:.1f}s 超时，已保持停车"
    )


def _reacquire_lane_while_rolling(
    robot,
    reader,
    perception,
    after_timestamp,
    timeout_seconds,
    last_yaw,
    preview_callback=None,
):
    """在已标定的安全路段低速前进，并用连续新帧重新捕获引导线。"""
    deadline = time.monotonic() + float(timeout_seconds)
    stable = 0
    saw_fresh_frame = False
    last_motion_sent = float("-inf")
    yaw = int(np.clip(
        last_yaw,
        -LANE_ROLLING_REACQUIRE_MAX_YAW,
        LANE_ROLLING_REACQUIRE_MAX_YAW,
    ))
    control_period = 1.0 / LANE_CONTROL_HZ

    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_motion_sent >= control_period:
                robot.mecanum_move_xyz(
                    0,
                    LANE_ROLLING_REACQUIRE_SPEED,
                    yaw,
                )
                last_motion_sent = now

            observation, after_timestamp = _wait_fresh_lane_observation(
                reader,
                perception,
                after_timestamp,
                timeout=max(0.0, deadline - time.monotonic()),
            )
            if observation is None:
                if saw_fresh_frame:
                    break
                raise RuntimeError("滚动重捕期间视频未恢复")
            saw_fresh_frame = True
            if preview_callback is not None:
                preview_callback(observation, "ROLLING REACQUIRE")
            stable = stable + 1 if (
                not observation.lost
                and observation.confidence >= LANE_STATIC_REACQUIRE_CONFIDENCE
            ) else 0
            if stable >= LANE_STATIC_REACQUIRE_FRAMES:
                return observation, after_timestamp
    finally:
        robot.stop_chassis()

    raise RuntimeError(
        "画面有效但滚动重捕仍未检测到引导线，"
        f"已低速搜索 {timeout_seconds:.1f}s 并停车"
    )


def _stabilize_tag_handoff_lane(
    robot,
    reader,
    perception,
    after_timestamp,
    preview_callback=None,
    lane_name="3->2",
):
    """静止确认一条路线起步引导线，避免转向后盲目前进。"""
    robot.stop_chassis()
    deadline = time.monotonic() + LANE_TAG_HANDOFF_STABLE_SECONDS
    stable = 0
    latest = None
    while time.monotonic() < deadline:
        observation, after_timestamp = _wait_fresh_lane_observation(
            reader,
            perception,
            after_timestamp,
            timeout=max(0.0, deadline - time.monotonic()),
        )
        if observation is None:
            break
        latest = observation
        if preview_callback is not None:
            preview_callback(observation, "TAG HANDOFF STABILIZE")
        safe_to_start = (
            not observation.lost
            and observation.confidence >= LANE_TAG_HANDOFF_MIN_CONFIDENCE
            and abs(observation.lateral_error) <= LANE_TAG_HANDOFF_MAX_LATERAL
            and abs(observation.heading_error) <= LANE_TAG_HANDOFF_MAX_HEADING
        )
        stable = stable + 1 if safe_to_start else 0
        if stable >= LANE_TAG_HANDOFF_STABLE_FRAMES:
            robot.stop_chassis()
            _log.bind(
                lane=lane_name,
                confidence=round(observation.confidence, 3),
                lateral=round(observation.lateral_error, 3),
                heading=round(observation.heading_error, 3),
                cross=round(observation.cross_score, 3),
                frames=stable,
            ).success("路线起步引导线已稳定，低速起步纠偏")
            return observation, after_timestamp

    robot.stop_chassis()
    details = "无有效画面" if latest is None else (
        f"confidence={latest.confidence:.3f}, "
        f"lateral={latest.lateral_error:.3f}, "
        f"heading={latest.heading_error:.3f}, lost={latest.lost}"
    )
    raise RuntimeError(
        f"未能稳定捕获路线起步引导线 {lane_name}，"
        f"已保持停车（{details}）"
    )


def _approach_loss_means_cross_passed(max_intersection_y, peak_cross, elapsed):
    """判断接近已确认路口时的丢线是否来自横线移出画面。"""
    return (
        max_intersection_y >= LANE_APPROACH_LOST_ENTRY_Y
        and peak_cross >= LANE_INTERSECTION_ENTER
        and elapsed >= LANE_APPROACH_LOST_MIN_SECONDS
    )


def _approach_ready_to_center(
    observation, elapsed, max_intersection_y, peak_cross
):
    """返回（可进入定距居中，是否使用了接近超时兜底）。

    正常路径仍要求横线到达画面底部。只有路口已被多帧确认、持续接近达到
    硬超时，且有效引导线与累计几何证据都足够时，才使用较宽松的兜底。
    """
    normal_reached = (
        not observation.lost
        and observation.confidence >= 0.55
        and observation.intersection_y >= LANE_TURN_ENTRY_Y - 0.02
    )
    if normal_reached:
        return True, False

    relaxed_reached = (
        elapsed >= LANE_APPROACH_TIMEOUT
        and not observation.lost
        and observation.confidence >= LANE_APPROACH_RELAXED_CONFIDENCE
        and max_intersection_y >= LANE_APPROACH_RELAXED_ENTRY_Y
        and peak_cross >= LANE_APPROACH_RELAXED_CROSS
        and abs(observation.lateral_error) <= LANE_TAG_HANDOFF_MAX_LATERAL
        and abs(observation.heading_error) <= LANE_TAG_HANDOFF_MAX_HEADING
    )
    return relaxed_reached, relaxed_reached


def _enter_pickup_zone(robot, route_name):
    """最后一次右转后前进进入取货区。

    取货区入口在固定 90° 右转后没有可见引导线，因此这个显式的
    RIGHT_AND_ARRIVE 动作使用已有的 40 cm 入区标定，不执行普通右转的
    静止重捕。
    """
    robot.stop_chassis()
    _log.bind(
        route=route_name,
        speed=LANE_PICKUP_ENTRY_SPEED,
        distance_cm=LANE_PICKUP_ENTRY_DISTANCE_CM,
    ).info("最终右转完成，前进进入取货区")
    robot.mecanum_move_speed_times(
        0,
        LANE_PICKUP_ENTRY_SPEED,
        LANE_PICKUP_ENTRY_DISTANCE_CM,
        1,
    )
    time.sleep(LANE_PICKUP_ENTRY_SETTLE_SECONDS)
    robot.stop_chassis()
    _log.bind(route=route_name).success("已进入取货区并停车")


def _enter_unload_zone(robot, route_name):
    """通过最后一个到达路口，前进到卸货点后停车。"""
    robot.stop_chassis()
    _log.bind(
        route=route_name,
        speed=UNLOAD_ENTRY_SPEED,
        distance_cm=UNLOAD_ENTRY_DISTANCE_CM,
    ).info("通过最后路口，前进到卸货点")
    robot.mecanum_move_speed_times(
        0,
        UNLOAD_ENTRY_SPEED,
        UNLOAD_ENTRY_DISTANCE_CM,
        1,
    )
    time.sleep(UNLOAD_ENTRY_SETTLE_SECONDS)
    robot.stop_chassis()
    _log.bind(route=route_name).success("路线已完成")


def _validate_short_route_left(action, profile):
    """短路段只允许消费地图中的左转动作。"""
    if action != RouteAction.LEFT:
        raise RuntimeError(
            f"返程短路段 {profile.segment_name} 的下一动作必须是 LEFT，"
            f"实际为 {getattr(action, 'value', action)}"
        )


def _follow_lane_route(
    robot,
    perception,
    route_name,
    route_actions,
    sensor_id=None,
    use_distance_stop=False,
    trip_index=None,
):
    route_segments = ROUTE_SEGMENTS_BY_NAME.get(route_name)
    if route_segments is None:
        route_segments = tuple(
            LaneSegmentProfile(
                f"unknown->{index + 1}",
                0.0 if index == 0 else LANE_DEFAULT_SEGMENT_GUARD_SECONDS,
            )
            for index in range(len(route_actions))
        )
    if len(route_segments) != len(route_actions):
        raise RuntimeError(
            f"路线 {route_name} 的动作数与地图路段数不一致: "
            f"actions={len(route_actions)}, segments={len(route_segments)}"
        )

    reader = LatestFrameReader(robot).start()
    follower = LaneFollower()
    latch = IntersectionLatch()
    route_state = LaneRouteState.TRACK
    state_started = time.monotonic()
    active_action = None
    active_pivot_offset_cm = LANE_DEFAULT_PIVOT_OFFSET_CM
    approach_max_intersection_y = 0.0
    approach_peak_cross = 0.0
    stable_reacquired = 0
    completed_turns = 0
    post_turn_started = None
    post_turn_lost_since = None
    short_progress = None
    short_commit_reason = None
    post_turn_cross_window = deque(maxlen=4)
    action_index = 0
    segment_profile = route_segments[0]
    segment_started = state_started
    next_track_intersection_at = (
        segment_started + segment_profile.min_event_seconds
    )
    last_timestamp = 0.0
    last_log = 0.0
    last_motion_sent = 0.0
    control_period = 1.0 / LANE_CONTROL_HZ
    preview_active = LANE_LIVE_PREVIEW
    preview_opened = False

    def show_live_preview(observation, note=""):
        nonlocal preview_active, preview_opened
        if not preview_active or observation.frame is None:
            return
        active = active_action.value if active_action is not None else "waiting"
        if active_action is not None and action_index > 0:
            target_segment = route_segments[action_index - 1]
        else:
            target_segment = segment_profile
        map_intersection = _segment_target_intersection(target_segment)
        detail = f"next={active}"
        if short_progress is not None and route_state in (
            LaneRouteState.SHORT_ROUTE_APPROACH,
            LaneRouteState.SHORT_BLIND_CENTER,
        ):
            detail = (
                f"short={short_progress.profile.segment_name} "
                f"dist={short_progress.distance_cm:.1f}cm "
                f"weak={short_progress.weak_frames}/"
                f"{LANE_RETURN_SHORT_WEAK_WINDOW} "
                f"lost={short_progress.lost_seconds(time.monotonic()):.2f}s "
                f"reason={short_commit_reason or '-'} "
                f"offset={short_progress.profile.blind_offset_cm:.1f}cm  {detail}"
            )
        if trip_index is not None:
            detail = f"trip={trip_index}/2  {detail}"
        if note:
            detail = f"{note}  {detail}"
        try:
            preview = render_lane_live_preview(
                observation.frame,
                observation,
                route_name=route_name,
                route_state=route_state,
                segment_name=segment_profile.name,
                map_intersection=map_intersection,
                action_index=action_index,
                action_count=len(route_actions),
                inference_ms=perception.last_latency_ms,
                note=detail,
            )
            cv2.imshow(LANE_PREVIEW_WINDOW, preview)
            preview_opened = True
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as error:
            preview_active = False
            _log.bind(error=str(error)).warning(
                "OpenCV 实时窗口不可用，已关闭画面预览，安全控制继续"
            )
            return
        if key in (ord("q"), ord("Q"), 27):
            robot.stop_chassis()
            _log.warning("实时识别窗口收到 Q/Esc，已紧急停车")
            raise KeyboardInterrupt

    def claim_next_action():
        nonlocal action_index
        if action_index >= len(route_actions):
            robot.stop_chassis()
            raise RuntimeError(f"路线 {route_name} 检测到计划外路口")
        action = route_actions[action_index]
        claimed_segment = route_segments[action_index]
        map_intersection = _segment_target_intersection(claimed_segment)
        action_index += 1
        _log.bind(
            trip=trip_index,
            route=route_name,
            route_step=action_index,
            map_intersection=map_intersection,
            action=action.value,
            segment=claimed_segment.name,
        ).info("确认地图十字路口并查询路线动作")
        return action

    try:
        _log.bind(
            trip=trip_index,
            route=route_name,
            actions=[a.value for a in route_actions],
            segments=[segment.name for segment in route_segments],
            apriltag_handoff=segment_profile.stabilize_before_start,
        ).info(
            "开始自定义视觉巡线"
        )
        if segment_profile.stabilize_before_start:
            # 转向交接后先静止捕获新方向；路口锁保持 armed，
            # 出发后看到的第一条横线就是当前路线的首个目标路口。
            _, last_timestamp = _stabilize_tag_handoff_lane(
                robot,
                reader,
                perception,
                last_timestamp,
                preview_callback=show_live_preview,
                lane_name=segment_profile.name,
            )
            state_started = time.monotonic()
            segment_started = state_started
            next_track_intersection_at = (
                segment_started + segment_profile.min_event_seconds
            )
        while True:
            frame_timeout = (
                LANE_INITIAL_FRAME_TIMEOUT_SECONDS
                if last_timestamp == 0.0
                else 0.30
            )
            observation, last_timestamp = _read_lane_observation(
                reader,
                perception,
                last_timestamp,
                timeout=frame_timeout,
            )
            if observation is None:
                paused_at = time.monotonic()
                if short_progress is not None:
                    short_progress.pause(paused_at)
                robot.stop_chassis()
                if last_timestamp == 0.0:
                    raise RuntimeError(
                        f"摄像头启动后 {LANE_INITIAL_FRAME_TIMEOUT_SECONDS:.1f}s "
                        "内未收到首帧，已安全停车"
                    )
                _log.bind(
                    initial_gap_ms=300,
                    recovery_timeout=LANE_FRAME_RECOVERY_TIMEOUT_SECONDS,
                ).warning("摄像头帧短暂中断，底盘已停车并等待视频恢复")
                observation, last_timestamp = _wait_fresh_lane_observation(
                    reader,
                    perception,
                    last_timestamp,
                    timeout=LANE_FRAME_RECOVERY_TIMEOUT_SECONDS,
                )
                if observation is None:
                    raise RuntimeError(
                        "摄像头视频流持续中断超过 "
                        f"{LANE_FRAME_RECOVERY_TIMEOUT_SECONDS:.1f}s，已安全停车"
                    )
                recovered_at = time.monotonic()
                if route_state == LaneRouteState.SHORT_BLIND_CENTER:
                    state_started += recovered_at - paused_at
                if short_progress is not None:
                    short_progress.resume(recovered_at)
                _log.success("摄像头视频流已恢复，继续巡线")

            if use_distance_stop and sensor_id is not None:
                distance = robot.read_distance_data(sensor_id)
                if 0 < distance < STOP_DISTANCE:
                    robot.stop_chassis()
                    _log.bind(distance=distance).success("距离传感器确认到达目的地")
                    return True

            # 相机线程可能在上面的设备 RPC 期间继续更新画面。
            # 如果当前 observation 因 RPC 阻塞而过期，先停车并取一帧
            # 真正的新画面，不把一次短暂延迟误判成永久丢线。
            frame_age = time.monotonic() - observation.timestamp
            if frame_age > LANE_FRAME_STALE_SECONDS:
                paused_at = time.monotonic()
                if short_progress is not None:
                    short_progress.pause(paused_at)
                robot.stop_chassis()
                _log.bind(
                    frame_age_ms=round(frame_age * 1000),
                    recovery_timeout=LANE_FRAME_RECOVERY_TIMEOUT_SECONDS,
                ).warning("巡线画面过期，已停车重取最新帧")
                observation, last_timestamp = _wait_fresh_lane_observation(
                    reader,
                    perception,
                    last_timestamp,
                    timeout=LANE_FRAME_RECOVERY_TIMEOUT_SECONDS,
                )
                if observation is None:
                    raise RuntimeError("巡线画面持续过期，已安全停车")
                recovered_at = time.monotonic()
                if route_state == LaneRouteState.SHORT_BLIND_CENTER:
                    state_started += recovered_at - paused_at
                if short_progress is not None:
                    short_progress.resume(recovered_at)
                _log.success("已取得最新巡线画面，继续执行路线")

            show_live_preview(observation)

            now = time.monotonic()
            if route_state == LaneRouteState.APPROACH and not observation.lost:
                approach_max_intersection_y = max(
                    approach_max_intersection_y, observation.intersection_y
                )
                approach_peak_cross = max(approach_peak_cross, observation.cross_score)

            if (
                observation.lost
                and route_state == LaneRouteState.APPROACH
                and active_action in (
                    RouteAction.LEFT,
                    RouteAction.RIGHT,
                    RouteAction.RIGHT_AND_ARRIVE,
                )
                and _approach_loss_means_cross_passed(
                    approach_max_intersection_y,
                    approach_peak_cross,
                    now - state_started,
                )
            ):
                turn_number = completed_turns + 1
                active_pivot_offset_cm = LANE_PIVOT_OFFSET_BY_TURN.get(
                    turn_number, LANE_DEFAULT_PIVOT_OFFSET_CM
                )
                route_state = LaneRouteState.CENTER_PIVOT
                state_started = now
                _log.bind(
                    trip=trip_index,
                    route=route_name,
                    turn=turn_number,
                    max_intersection_y=round(approach_max_intersection_y, 3),
                    peak_cross=round(approach_peak_cross, 3),
                    offset_cm=active_pivot_offset_cm,
                ).warning(
                    "路口横线接近画面底部后消失，"
                    "按已驶过摄像头视野继续居中转向"
                )

            # 直行通过 6/7 号后，连续的低横线分（包括预期丢线帧）
            # 只用于确认旧路口已离开。只有这一刻才开始短路段计程。
            short_latch_updated = False
            if (
                route_state == LaneRouteState.TRACK
                and segment_profile.short_turn_zone is not None
                and active_action == RouteAction.STRAIGHT
                and not latch.armed
            ):
                short_latch_updated = True
                was_armed = latch.armed
                latch.update(
                    observation,
                    allow_trigger=False,
                    relaxed_straight_exit=True,
                )
                if not was_armed and latch.armed:
                    profile = SHORT_ROUTE_TURN_PROFILE_BY_ZONE[
                        segment_profile.short_turn_zone
                    ]
                    short_progress = ShortRouteProgress(profile, now)
                    short_commit_reason = None
                    active_action = None
                    route_state = LaneRouteState.SHORT_ROUTE_APPROACH
                    state_started = now
                    _log.bind(
                        trip=trip_index,
                        route=route_name,
                        segment=profile.segment_name,
                        zone=profile.zone,
                        min_commit_cm=profile.min_commit_cm,
                        fallback_cm=profile.fallback_cm,
                        blind_offset_cm=profile.blind_offset_cm,
                    ).warning(
                        "已确认驶离 6/7 号旧路口，"
                        "进入返程短路段计程提交模式"
                    )

            clearing_short_old_cross = (
                route_state == LaneRouteState.TRACK
                and segment_profile.short_turn_zone is not None
                and active_action == RouteAction.STRAIGHT
                and not latch.armed
            )

            if (
                observation.lost
                and not clearing_short_old_cross
                and route_state in (
                    LaneRouteState.TRACK, LaneRouteState.APPROACH,
                    LaneRouteState.POST_TURN,
                )
            ):
                rolling_reacquire = (
                    route_state == LaneRouteState.TRACK
                    and segment_profile.rolling_reacquire_seconds is not None
                )
                reacquire_seconds = (
                    segment_profile.rolling_reacquire_seconds
                    if rolling_reacquire
                    else (
                        segment_profile.reacquire_seconds
                        if segment_profile.reacquire_seconds is not None
                        else LANE_STATIC_REACQUIRE_SECONDS
                    )
                )
                _log.bind(
                    route=route_name, state=route_state.value,
                    segment=segment_profile.name,
                    route_step=action_index,
                    next_map_intersection=(
                        _segment_target_intersection(route_segments[action_index])
                        if action_index < len(route_segments) else None
                    ),
                    next_action=(route_actions[action_index].value
                                 if action_index < len(route_actions) else None),
                    guard_remaining=round(max(0, next_track_intersection_at - now), 2),
                    latch_armed=latch.armed,
                    reacquire_seconds=reacquire_seconds,
                    rolling=rolling_reacquire,
                ).warning(
                    "未检测到引导线，低速滚动等待连续新帧重捕"
                    if rolling_reacquire
                    else "未检测到引导线，停车等待连续新帧重捕"
                )
                paused_at = now
                if rolling_reacquire:
                    observation, last_timestamp = _reacquire_lane_while_rolling(
                        robot,
                        reader,
                        perception,
                        last_timestamp,
                        timeout_seconds=reacquire_seconds,
                        last_yaw=follower.last_command.yaw_speed,
                        preview_callback=show_live_preview,
                    )
                else:
                    observation, last_timestamp = _reacquire_lane_statically(
                        robot,
                        reader,
                        perception,
                        last_timestamp,
                        timeout_seconds=reacquire_seconds,
                        preview_callback=show_live_preview,
                    )
                now = time.monotonic()
                if not rolling_reacquire:
                    paused_seconds = now - paused_at
                    # 静止等待不计入接近路口/过渡段的运动时间。
                    state_started += paused_seconds
                    segment_started += paused_seconds
                    next_track_intersection_at += paused_seconds
                    if post_turn_started is not None:
                        post_turn_started += paused_seconds
                post_turn_lost_since = None
                post_turn_cross_window.clear()
                latch.window.clear()
                latch.exit_window.clear()
                follower = LaneFollower()
                last_motion_sent = 0.0
                _log.bind(
                    route=route_name,
                    segment=segment_profile.name,
                    rolling=rolling_reacquire,
                ).success("连续新帧已重捕引导线，恢复当前路线动作")
            previous_state = route_state
            forward = 0
            yaw = 0

            if clearing_short_old_cross and observation.lost:
                # 旧路口横线离开视野时丢线是预期现象；不静止重捕，
                # 低速保持最后方向，直到路口锁确认释放。
                forward = LANE_RETURN_SHORT_LOST_SPEED
                yaw = int(np.clip(
                    follower.last_command.yaw_speed,
                    -LANE_RETURN_SHORT_MAX_YAW,
                    LANE_RETURN_SHORT_MAX_YAW,
                ))

            elif route_state == LaneRouteState.TRACK:
                command = follower.step(observation, route_name)
                if command.mode == DriveMode.STOP:
                    robot.stop_chassis()
                    raise RuntimeError(
                        f"巡线控制停止: lost={observation.lost}, "
                        f"frame_age_ms={(time.monotonic() - observation.timestamp) * 1000:.0f}"
                    )

                forward = command.forward_speed
                yaw = command.yaw_speed
                if segment_profile.speed_limit is not None:
                    forward = min(forward, segment_profile.speed_limit)
                if active_action == RouteAction.STRAIGHT and not latch.armed:
                    forward = min(forward, LANE_APPROACH_SPEED)
                    yaw = int(np.clip(yaw, -18, 18))
                intersection_triggered = (
                    False
                    if short_latch_updated
                    else latch.update(
                        observation,
                        allow_trigger=now >= next_track_intersection_at,
                        relaxed_straight_exit=segment_profile.straight_exit_relaxed,
                    )
                )
                if intersection_triggered:
                    action = claim_next_action()
                    if action == RouteAction.STRAIGHT:
                        active_action = action
                        latch.started_straight(observation)
                        if action_index < len(route_segments):
                            segment_profile = route_segments[action_index]
                            segment_started = now
                            next_track_intersection_at = (
                                now + segment_profile.min_event_seconds
                            )
                        forward = min(LANE_APPROACH_SPEED, forward)
                        yaw = int(np.clip(yaw, -18, 18))
                        _log.bind(
                            route=route_name,
                            route_step=action_index,
                            map_intersection=_segment_target_intersection(
                                route_segments[action_index - 1]
                            ),
                        ).info(
                            "当前路口动作为直行，不执行转向"
                        )
                    elif action == RouteAction.ARRIVE:
                        _enter_unload_zone(robot, route_name)
                        return True
                    else:
                        active_action = action
                        route_state = LaneRouteState.APPROACH
                        state_started = now
                        approach_max_intersection_y = observation.intersection_y
                        approach_peak_cross = observation.cross_score
                        forward = LANE_APPROACH_SPEED
                        yaw = int(np.clip(yaw, -18, 18))
                        _log.bind(
                            route=route_name,
                            route_step=action_index,
                            map_intersection=_segment_target_intersection(
                                route_segments[action_index - 1]
                            ),
                            score=round(observation.cross_score, 2),
                        ).warning("路线要求转向，低速驶向路口中心")

            elif route_state == LaneRouteState.SHORT_ROUTE_APPROACH:
                if short_progress is None:
                    robot.stop_chassis()
                    raise RuntimeError("返程短路段状态缺少计程器")
                short_progress.advance(now)
                short_progress.observe(observation, now)
                commit_reason = short_progress.evaluate_commit(now)

                if observation.lost:
                    forward = LANE_RETURN_SHORT_LOST_SPEED
                    yaw = int(np.clip(
                        follower.last_command.yaw_speed,
                        -LANE_RETURN_SHORT_MAX_YAW,
                        LANE_RETURN_SHORT_MAX_YAW,
                    ))
                else:
                    command = follower.step(observation, "return-short-approach")
                    if command.mode == DriveMode.STOP:
                        robot.stop_chassis()
                        raise RuntimeError("返程短路段视觉帧已过期")
                    forward = min(command.forward_speed, LANE_RETURN_SHORT_SPEED)
                    yaw = int(np.clip(
                        command.yaw_speed,
                        -LANE_RETURN_SHORT_MAX_YAW,
                        LANE_RETURN_SHORT_MAX_YAW,
                    ))

                if commit_reason is not None:
                    short_progress.pause(now)
                    short_progress.commit_reason = commit_reason
                    short_commit_reason = commit_reason
                    next_action = claim_next_action()
                    try:
                        _validate_short_route_left(next_action, short_progress.profile)
                    except RuntimeError:
                        robot.stop_chassis()
                        raise
                    active_action = next_action
                    active_pivot_offset_cm = short_progress.profile.blind_offset_cm
                    route_state = LaneRouteState.SHORT_BLIND_CENTER
                    state_started = now
                    forward = LANE_RETURN_SHORT_BLIND_SPEED
                    yaw = 0
                    _log.bind(
                        trip=trip_index,
                        route=route_name,
                        segment=short_progress.profile.segment_name,
                        zone=short_progress.profile.zone,
                        reason=commit_reason,
                        distance_cm=round(short_progress.distance_cm, 2),
                        weak_frames=short_progress.weak_frames,
                        weak_window=LANE_RETURN_SHORT_WEAK_WINDOW,
                        lost_seconds=round(short_progress.lost_seconds(now), 3),
                        blind_offset_cm=active_pivot_offset_cm,
                        blind_speed=LANE_RETURN_SHORT_BLIND_SPEED,
                    ).warning(
                        "返程短路段已提交下一路口左转，"
                        "盲走到车体旋转中心"
                    )

            elif route_state == LaneRouteState.APPROACH:
                command = follower.step(observation, "intersection-approach")
                if command.mode == DriveMode.STOP:
                    robot.stop_chassis()
                    raise RuntimeError("接近路口过程中丢线或画面过期")
                forward = LANE_APPROACH_SPEED
                yaw = int(np.clip(command.yaw_speed, -18, 18))

                approach_elapsed = now - state_started
                reached, relaxed_timeout_reached = _approach_ready_to_center(
                    observation,
                    approach_elapsed,
                    approach_max_intersection_y,
                    approach_peak_cross,
                )
                if reached:
                    turn_number = completed_turns + 1
                    active_pivot_offset_cm = LANE_PIVOT_OFFSET_BY_TURN.get(
                        turn_number, LANE_DEFAULT_PIVOT_OFFSET_CM
                    )
                    route_state = LaneRouteState.CENTER_PIVOT
                    state_started = now
                    forward = LANE_CENTERING_SPEED
                    yaw = 0
                    _log.bind(
                        route=route_name,
                        turn=turn_number,
                        intersection_y=round(observation.intersection_y, 2),
                        max_intersection_y=round(approach_max_intersection_y, 2),
                        peak_cross=round(approach_peak_cross, 2),
                        relaxed_timeout=relaxed_timeout_reached,
                        offset_cm=active_pivot_offset_cm,
                        seconds=round(
                            active_pivot_offset_cm / LANE_CENTERING_SPEED, 2
                        ),
                    ).info("路口横线已到达画面底部，盲走到车体旋转中心")
                elif approach_elapsed >= LANE_APPROACH_TIMEOUT:
                    robot.stop_chassis()
                    raise RuntimeError("接近路口超时，未到达安全转向位置")

            elif route_state in (
                LaneRouteState.CENTER_PIVOT,
                LaneRouteState.SHORT_BLIND_CENTER,
            ):
                short_blind = route_state == LaneRouteState.SHORT_BLIND_CENTER
                centering_speed = (
                    LANE_RETURN_SHORT_BLIND_SPEED
                    if short_blind else LANE_CENTERING_SPEED
                )
                forward = centering_speed
                yaw = 0
                centering_seconds = active_pivot_offset_cm / centering_speed
                if now - state_started >= centering_seconds:
                    robot.stop_chassis()
                    route_state = LaneRouteState.TURN_90
                    state_started = time.monotonic()
                    forward = yaw = 0
                    turn_left = active_action == RouteAction.LEFT
                    turn_direction = _turn_direction_for_action(active_action)
                    turn_name = "左转" if turn_left else "右转"
                    _log.bind(
                        route=route_name,
                        offset_cm=active_pivot_offset_cm,
                        action=active_action.value,
                        direction=turn_direction,
                        angle=90,
                        speed=LANE_FIXED_TURN_SPEED,
                        short_route=short_blind,
                        commit_reason=short_commit_reason,
                    ).info(f"车体旋转中心已到达路口，执行固定{turn_name} 90 度")
                    robot.mecanum_turn_speed_times(
                        turn_direction, LANE_FIXED_TURN_SPEED, 90, 2
                    )

            elif route_state == LaneRouteState.TURN_90:
                forward = yaw = 0
                fixed_turn_seconds = 90.0 / LANE_FIXED_TURN_SPEED + 0.40
                if now - state_started >= fixed_turn_seconds:
                    robot.stop_chassis()
                    if active_action == RouteAction.RIGHT_AND_ARRIVE:
                        _enter_pickup_zone(robot, route_name)
                        return True
                    route_state = LaneRouteState.SEARCH_LANE
                    state_started = now
                    stable_reacquired = 0
                    _log.bind(
                        route=route_name,
                        action=active_action.value,
                        timeout=LANE_TURN_SEARCH_TIMEOUT,
                    ).info("固定 90 度转向完成，底盘静止搜索新赛道")

            elif route_state == LaneRouteState.SEARCH_LANE:
                forward = yaw = 0
                elapsed = now - state_started
                geometry_reacquired = (
                    not observation.lost
                    and observation.confidence >= 0.30
                    and abs(observation.lateral_error) < 1.05
                    and abs(observation.heading_error) < 1.35
                )
                turn_candidate, candidate_lateral, candidate_heading = (
                    _turn_mask_alignment(observation.mask)
                )
                short_mask_reacquired = (
                    observation.confidence >= 0.25
                    and turn_candidate
                    and abs(candidate_lateral) < 1.10
                    and abs(candidate_heading) < 1.40
                )
                reacquired = geometry_reacquired or short_mask_reacquired
                stable_reacquired = stable_reacquired + 1 if reacquired else 0
                if stable_reacquired >= 2:
                    robot.stop_chassis()
                    completed_turns += 1
                    follower = LaneFollower()
                    latch.completed_turn()
                    route_state = LaneRouteState.POST_TURN
                    state_started = now
                    post_turn_started = now
                    active_action = None
                    if action_index < len(route_segments):
                        segment_profile = route_segments[action_index]
                    segment_started = now
                    next_track_intersection_at = (
                        now + segment_profile.min_event_seconds
                    )
                    post_turn_lost_since = None
                    post_turn_cross_window.clear()
                    _log.bind(
                        route=route_name,
                        turns=completed_turns,
                        seconds=round(elapsed, 2),
                        speed=LANE_POST_TURN_SPEED,
                        segment=segment_profile.name,
                        next_intersection_guard_seconds=segment_profile.min_event_seconds,
                        post_turn_preclaim=segment_profile.post_turn_preclaim,
                    ).success("已捕获转向后的引导线，进入低速过渡段")
                elif elapsed >= LANE_TURN_SEARCH_TIMEOUT:
                    robot.stop_chassis()
                    raise RuntimeError("固定转向后搜索新赛道超时")

            elif route_state == LaneRouteState.POST_TURN:
                # 普通路段必须先用 latch 确认驶离当前路口，
                # 避免把 2 号路口的残留横线当成 4 号。
                latch.update(observation, allow_trigger=False)
                post_turn_elapsed = (
                    now - post_turn_started if post_turn_started is not None else 0.0
                )
                pending_action = (
                    route_actions[action_index]
                    if action_index < len(route_actions)
                    else None
                )
                if post_turn_elapsed >= segment_profile.min_event_seconds:
                    post_turn_cross_window.append(
                        not observation.lost
                        and observation.confidence >= 0.50
                        and observation.cross_score >= 0.70
                    )
                else:
                    post_turn_cross_window.clear()
                next_intersection = _post_turn_can_preclaim(
                    segment_profile,
                    pending_action,
                    post_turn_elapsed,
                    sum(post_turn_cross_window),
                )

                if next_intersection:
                    next_action = claim_next_action()
                    post_turn_cross_window.clear()
                    post_turn_lost_since = None
                    if next_action == RouteAction.STRAIGHT:
                        command = follower.step(observation, "post-turn-straight")
                        if command.mode == DriveMode.STOP:
                            raise RuntimeError("转弯后直行通过路口时丢线")
                        forward = LANE_APPROACH_SPEED
                        yaw = int(np.clip(command.yaw_speed, -18, 18))
                        route_state = LaneRouteState.TRACK
                        state_started = now
                        active_action = RouteAction.STRAIGHT
                        latch.started_straight(observation)
                        if action_index < len(route_segments):
                            segment_profile = route_segments[action_index]
                            segment_started = now
                            next_track_intersection_at = (
                                now + segment_profile.min_event_seconds
                            )
                        _log.bind(
                            route=route_name,
                            route_step=action_index,
                            map_intersection=_segment_target_intersection(
                                route_segments[action_index - 1]
                            ),
                        ).info(
                            "紧邻路口的路线动作为直行，"
                            "锁定路口检测直至完全驶离当前路口"
                        )
                    elif next_action == RouteAction.ARRIVE:
                        _enter_unload_zone(robot, route_name)
                        return True
                    else:
                        active_action = next_action
                        route_state = LaneRouteState.APPROACH
                        state_started = now
                        approach_max_intersection_y = observation.intersection_y
                        approach_peak_cross = observation.cross_score
                        command = follower.step(
                            observation, "post-turn-intersection"
                        )
                        if command.mode == DriveMode.STOP:
                            raise RuntimeError("转弯后接近紧邻路口时丢线")
                        forward = LANE_APPROACH_SPEED
                        yaw = int(np.clip(command.yaw_speed, -18, 18))
                        _log.bind(
                            route=route_name,
                            route_step=action_index,
                            map_intersection=_segment_target_intersection(
                                route_segments[action_index - 1]
                            ),
                            action=next_action.value,
                        ).warning("过渡段确认下一个转向路口")
                elif observation.lost:
                    if post_turn_lost_since is None:
                        post_turn_lost_since = now
                        robot.stop_chassis()
                        _log.info("转弯后引导线结束，停车确认")
                    if now - post_turn_lost_since >= 0.30:
                        robot.stop_chassis()
                        if active_action == RouteAction.RIGHT_AND_ARRIVE:
                            _log.bind(route=route_name).success(
                                "已沿短引导线进入取货区"
                            )
                            return True
                        raise RuntimeError("普通转向后持续丢线，不得误判为到达")
                else:
                    post_turn_lost_since = None
                    command = follower.step(observation, "post-turn-guide")
                    if command.mode == DriveMode.STOP:
                        raise RuntimeError("转弯后过渡段画面过期")
                    forward = LANE_POST_TURN_SPEED
                    yaw = int(np.clip(command.yaw_speed, -18, 18))

                if (
                    route_state == LaneRouteState.POST_TURN
                    and post_turn_started is not None
                    and post_turn_elapsed >= max(
                        LANE_POST_TURN_GRACE,
                        segment_profile.min_event_seconds,
                    )
                ):
                    route_state = LaneRouteState.TRACK
                    state_started = now
                    active_action = None
                    post_turn_lost_since = None
                    _log.bind(
                        route=route_name,
                        segment=segment_profile.name,
                        latch_armed=latch.armed,
                    ).info(
                        "转向后过渡结束，按地图路段恢复巡线"
                    )

            if forward or yaw:
                state_changed = route_state != previous_state
                if state_changed or now - last_motion_sent >= control_period:
                    robot.mecanum_move_xyz(0, int(forward), int(yaw))
                    last_motion_sent = time.monotonic()
                    if route_state == LaneRouteState.SHORT_ROUTE_APPROACH:
                        short_progress.set_commanded_speed(
                            int(forward), last_motion_sent
                        )

            if time.monotonic() - last_log >= 0.5:
                _log.bind(
                    trip=trip_index,
                    route=route_name,
                    state=route_state.value,
                    confidence=round(observation.confidence, 3),
                    lateral=round(observation.lateral_error, 3),
                    heading=round(observation.heading_error, 3),
                    curvature=round(observation.curvature, 3),
                    cross=round(observation.cross_score, 3),
                    intersection_y=round(observation.intersection_y, 3),
                    lost=observation.lost,
                    frame_age_ms=round((time.monotonic() - observation.timestamp) * 1000),
                    latch_armed=latch.armed,
                    guard_remaining=round(max(0, next_track_intersection_at - now), 2),
                    segment=segment_profile.name,
                    target_map_intersection=_segment_target_intersection(
                        segment_profile
                    ),
                    source=observation.source,
                    inference_ms=round(perception.last_latency_ms, 1),
                    short_distance_cm=(
                        round(short_progress.distance_cm, 2)
                        if short_progress is not None else None
                    ),
                    short_weak_frames=(
                        short_progress.weak_frames
                        if short_progress is not None else None
                    ),
                    short_lost_seconds=(
                        round(short_progress.lost_seconds(now), 3)
                        if short_progress is not None else None
                    ),
                    short_commit_reason=short_commit_reason,
                    short_blind_offset_cm=(
                        short_progress.profile.blind_offset_cm
                        if short_progress is not None else None
                    ),
                ).debug("巡线状态")
                last_log = time.monotonic()
    finally:
        reader.stop()
        if preview_opened:
            try:
                cv2.destroyWindow(LANE_PREVIEW_WINDOW)
                cv2.waitKey(1)
            except cv2.error:
                pass
        try:
            robot.stop_chassis()
        except Exception:
            pass


def line_follow_phase(robot, perception, trip_index=1):
    return _follow_lane_route(
        robot,
        perception,
        "start-to-pickup",
        START_TO_PICKUP_ROUTE,
        trip_index=trip_index,
    )


def _turn_around_for_return(robot, zone, sleep_fn=time.sleep):
    """在卸货区开阔位置左转180°，让摄像头面向返程引导线。"""
    if zone not in RETURN_TO_PICKUP_ROUTE_BY_ZONE:
        raise ValueError(f"不支持的返程起点: {zone}")
    robot.stop_chassis()
    _log.bind(
        zone=zone,
        direction=2,
        speed=RETURN_TURN_SPEED,
        angle=RETURN_TURN_ANGLE,
    ).info("第一次卸货完成，原地左转180度准备返航")
    robot.mecanum_turn_speed_times(
        2, RETURN_TURN_SPEED, RETURN_TURN_ANGLE, 2
    )
    sleep_fn(RETURN_TURN_ANGLE / RETURN_TURN_SPEED + RETURN_TURN_SETTLE_SECONDS)
    robot.stop_chassis()


def _raise_arm_for_return_turn(robot, sleep_fn=time.sleep):
    """第一次卸货后继续抬臂，为原地 180° 转身留出物块间隙。"""
    robot.stop_chassis()
    sleep_fn(RETURN_TURN_ARM_SETTLE_SECONDS)
    _log.bind(pose=RETURN_TURN_ARM_CLEAR_POSE).info(
        "第一次卸货完成，继续抬高机械臂避让已放下物块"
    )
    move_servos_to_actual_angles(
        robot,
        RETURN_TURN_ARM_CLEAR_POSE,
        duration_ms=RETURN_TURN_ARM_DURATION_MS,
    )
    _log.success("返程转身避障姿态已到位")


def return_to_pickup_phase(
    robot, zone, perception, sleep_fn=time.sleep, trip_index=1
):
    """从第一次存储区返回3号取货区。"""
    _turn_around_for_return(robot, zone, sleep_fn=sleep_fn)
    route_name = f"return-{zone}-to-pickup"
    route = RETURN_TO_PICKUP_ROUTE_BY_ZONE[zone]
    return _follow_lane_route(
        robot, perception, route_name, route, trip_index=trip_index
    )


# ============================================================
# 阶段 3 — YOLO 追踪 + 稳定抓取 + 舵机夹方块
# ============================================================


def track_and_grab_phase(robot, color, sensor_id, execute_grab=True):
    _log.bind(color=color).info("开始 YOLO 追踪")

    state = {"offset": None, "area": 0, "found": False, "frame": None}
    lock = threading.Lock()
    stop_event = threading.Event()
    grab_event = threading.Event()

    pid_h = robot.create_pid_controller()
    pid_h.set_pid(TRACK_KP, TRACK_KI, TRACK_KD)
    _log.bind(pid="horizontal", kp=TRACK_KP, ki=TRACK_KI, kd=TRACK_KD).info("水平 PID")

    pid_dist = robot.create_pid_controller()
    pid_dist.set_pid(DISTANCE_KP, DISTANCE_KI, DISTANCE_KD)
    _log.bind(
        pid="distance",
        kp=DISTANCE_KP,
        ki=DISTANCE_KI,
        kd=DISTANCE_KD,
        target=GRAB_DISTANCE_THRESHOLD,
    ).info("距离 PID")

    # ── 控制线程 ──
    def control_loop():
        stable_counter = 0
        while not stop_event.is_set():
            distance = robot.read_distance_data(sensor_id)
            if distance <= 0:
                for _ in range(3):
                    stop_event.wait(0.05)
                    distance = robot.read_distance_data(sensor_id)
                    if distance > 0:
                        break
                else:
                    _log.bind(sensor=sensor_id, val=distance).critical(
                        "距离传感器连续无数据"
                    )
                    stop_event.set()
                    return

            dist_error = round(pid_dist.update(distance - GRAB_DISTANCE_THRESHOLD))

            with lock:
                found = state["found"]
                offset = state["offset"]

            if not found:
                stable_counter = 0
                robot.mecanum_move_turn(0, 0, 2, SEARCH_SPEED)
                _log.bind(state="searching", dist=distance).trace("搜索旋转")
            else:
                dic = round(pid_h.update(offset))

                if distance > 25:
                    max_turn = 40
                elif distance > 15:
                    max_turn = 25
                else:
                    max_turn = 15
                turn_speed = min(abs(dic), max_turn)

                if dist_error < 0:
                    dist_forward = int(min(-dist_error, CHASE_SPEED))
                elif dist_error > 0:
                    bwd = int(min(dist_error, BACKWARD_SPEED))
                    robot.mecanum_move_speed(1, bwd)
                    stable_counter = 0
                    _log.bind(state="backward", dist=distance, speed=bwd).trace("后退")
                    stop_event.wait(0.05)
                    continue
                else:
                    dist_forward = 0

                if dist_forward == 0:
                    if turn_speed < 3:
                        robot.stop_chassis()
                        if (
                            distance
                            <= GRAB_DISTANCE_THRESHOLD + GRAB_DISTANCE_TOLERANCE
                            and abs(offset) < GRAB_OFFSET_THRESHOLD
                        ):
                            stable_counter += 1
                            _log.bind(
                                state="idle",
                                dist=distance,
                                offset=offset,
                                stable=stable_counter,
                                need=GRAB_STABLE_FRAMES,
                            ).trace("待命就绪")
                            if stable_counter >= GRAB_STABLE_FRAMES:
                                _log.bind(dist=distance, offset=offset).success(
                                    "已对准，准备抓取"
                                )
                                grab_event.set()
                                stop_event.set()
                                return
                        else:
                            stable_counter = 0
                            _log.bind(state="idle", dist=distance, offset=offset).trace(
                                "待命"
                            )
                    elif dic < 0:
                        stable_counter = 0
                        robot.mecanum_move_turn(0, 0, 3, turn_speed)
                        _log.bind(
                            state="turn_right",
                            dist=distance,
                            offset=offset,
                            turn=turn_speed,
                        ).trace("右转")
                    else:
                        stable_counter = 0
                        robot.mecanum_move_turn(0, 0, 2, turn_speed)
                        _log.bind(
                            state="turn_left",
                            dist=distance,
                            offset=offset,
                            turn=turn_speed,
                        ).trace("左转")
                elif turn_speed < 3:
                    if (
                        distance <= GRAB_DISTANCE_THRESHOLD + GRAB_DISTANCE_TOLERANCE
                        and abs(offset) < GRAB_OFFSET_THRESHOLD
                    ):
                        stable_counter += 1
                        _log.bind(
                            state="forward_grab",
                            dist=distance,
                            offset=offset,
                            stable=stable_counter,
                            need=GRAB_STABLE_FRAMES,
                        ).trace("微调前进已就绪")
                        if stable_counter >= GRAB_STABLE_FRAMES:
                            _log.bind(dist=distance, offset=offset).success(
                                "已对准，准备抓取"
                            )
                            grab_event.set()
                            stop_event.set()
                            return
                    else:
                        stable_counter = 0
                    robot.mecanum_move_speed(0, dist_forward)
                    _log.bind(state="forward", dist=distance, speed=dist_forward).trace(
                        "前进"
                    )
                elif dic < 0:
                    stable_counter = 0
                    robot.mecanum_move_turn(0, dist_forward, 3, turn_speed)
                    _log.bind(
                        state="fwd_right",
                        dist=distance,
                        fwd=dist_forward,
                        turn=turn_speed,
                    ).trace("前进右转")
                else:
                    stable_counter = 0
                    robot.mecanum_move_turn(0, dist_forward, 2, turn_speed)
                    _log.bind(
                        state="fwd_left",
                        dist=distance,
                        fwd=dist_forward,
                        turn=turn_speed,
                    ).trace("前进左转")

            stop_event.wait(0.05)

    # ── 视觉线程 (YOLO) ──
    def vision_loop():
        try:
            model = YOLO(MODEL_PATH)
            _log.success("YOLO 模型加载成功")
        except Exception:
            _log.opt(exception=True).error("YOLO 加载失败")
            stop_event.set()
            return

        model(
            np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8),
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            verbose=False,
        )
        _log.success("模型预热完成")

        while not stop_event.is_set():
            try:
                data = robot.read_camera_data()
            except Exception:
                stop_event.wait(0.01)
                continue
            if data is None:
                stop_event.wait(0.01)
                continue

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            fh, fw = frame.shape[:2]
            cx = fw // 2

            results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)

            cubes = []
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cls_id = int(boxes.cls[i])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else ""
                    if cls_name != color:
                        continue
                    x, y = int(round(x1)), int(round(y1))
                    bw, bh = int(round(x2 - x1)), int(round(y2 - y1))
                    if bw <= 0 or bh <= 0:
                        continue
                    cubes.append((x, y, bw, bh, bw * bh))

            largest = max(cubes, key=lambda c: c[4]) if cubes else None

            with lock:
                if largest is not None:
                    x, y, bw, bh, area = largest
                    state["offset"] = (x + bw // 2) - cx
                    state["area"] = area
                    state["found"] = True
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.line(frame, (cx, 0), (cx, fh), (255, 255, 0), 1)
                    cv2.putText(
                        frame,
                        color,
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    state["found"] = False
                state["frame"] = frame

    with lock:
        state["frame"] = np.zeros((480, 640, 3), dtype=np.uint8)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    vis_thread = threading.Thread(target=vision_loop, daemon=True)
    ctrl_thread.start()
    vis_thread.start()

    _log.info("进入追踪主循环")
    try:
        while (
            vis_thread.is_alive() and ctrl_thread.is_alive() and not stop_event.is_set()
        ):
            with lock:
                disp = state["frame"]
            if disp is not None:
                cv2.imshow(f"YOLO Track - {color}", disp)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    robot.stop_chassis()
                    _log.warning("YOLO 追踪窗口收到 Q/Esc，已紧急停车")
                    raise KeyboardInterrupt
            else:
                stop_event.wait(0.05)
    finally:
        stop_event.set()
        try:
            robot.stop_chassis()
        except Exception:
            pass
        cv2.destroyAllWindows()

    if not grab_event.is_set():
        _log.info("未触发抓取")
        return False

    if not execute_grab:
        _log.bind(color=color).success(
            "已稳定靠近并对准目标，跳过自动抓取以进入机械臂标定"
        )
        return True

    # ── 抓取序列 ──
    _log.info("执行实车标定抓取序列")
    _log.bind(pose=ARM_CAMERA_CLEAR_POSE).info("机械臂进入抬起准备姿态")
    move_servos_to_actual_angles(
        robot,
        ARM_CAMERA_CLEAR_POSE,
        duration_ms=1200,
    )
    _log.info("夹爪张开")
    robot.mechanical_clamp_release()
    time.sleep(0.3)

    _log.bind(pose=GRAB_ARM_POSE).info("机械臂进入标定夹取姿态")
    move_servos_to_actual_angles(robot, GRAB_ARM_POSE, duration_ms=2000)

    _log.info("夹爪闭合")
    robot.mechanical_clamp_close()
    time.sleep(0.5)

    _log.bind(pose=ARM_CAMERA_CLEAR_POSE).info("夹取完成，抬起机械臂避让摄像头")
    move_servos_to_actual_angles(
        robot,
        ARM_CAMERA_CLEAR_POSE,
        duration_ms=2200,
    )

    _log.success("抓取并抬臂完成，摄像头视野已避让")
    return True


# ============================================================
# 阶段 4 — 从取货区导航到 A/B 卸货区
# ============================================================


def _get_target_tag(tags, target_id):
    """从 AprilTag 列表中查找目标 Tag"""
    for tag in tags:
        if tag[0] == target_id:
            return tag
    return None


def _apriltag_control_mode(tag, last_seen_at, now):
    """返回 track / hold / search，防止已锁定后因单帧丢失突然自旋。"""
    if tag is not None:
        return "track"
    if (
        last_seen_at is not None
        and now < last_seen_at + APRILTAG_LOCK_LOST_GRACE_SECONDS
    ):
        return "hold"
    return "search"


def _chase_apriltag(robot, target_id, target_dist, sensor_id):
    """追踪 AprilTag 到达目标距离

    Returns: True 到达目标距离，False 被中断
    """
    _log.bind(tag_id=target_id, target_dist=target_dist).info("开始追踪 AprilTag")

    state = {"tag": None, "last_seen_at": None}
    lock = threading.Lock()
    stop_event = threading.Event()
    reached = False

    pid = robot.create_pid_controller()
    pid.set_pid(APRILTAG_KP, APRILTAG_KI, APRILTAG_KD)
    _log.bind(pid="horizontal", kp=APRILTAG_KP, ki=APRILTAG_KI, kd=APRILTAG_KD).info(
        "水平 PID 配置"
    )

    def control_loop():
        nonlocal reached
        while not stop_event.is_set():
            with lock:
                tag = state["tag"]
                last_seen_at = state["last_seen_at"]

            control_mode = _apriltag_control_mode(
                tag, last_seen_at, time.monotonic()
            )
            if control_mode == "search":
                robot.mecanum_move_xyz(0, 0, APRILTAG_SEARCH_SPEED)
                _log.bind(state="searching").trace("搜索旋转")
            elif control_mode == "hold":
                # 底盘会持续执行上一条搜索旋转命令，因此必须
                # 显式停车，避免 Tag 单帧丢失时左旋继续干扰追踪。
                robot.stop_chassis()
                _log.bind(state="lock_grace").trace(
                    "Tag 短暂丢帧，保持停车等待重新锁定"
                )
            else:
                _id, cx, cy = tag[:3]

                distance = robot.read_distance_data(sensor_id)
                if distance <= 0:
                    for _ in range(3):
                        stop_event.wait(0.05)
                        distance = robot.read_distance_data(sensor_id)
                        if distance > 0:
                            break
                    else:
                        _log.bind(sensor_id=sensor_id, value=distance).critical(
                            "距离传感器连续无数据"
                        )
                        stop_event.set()
                        return

                offset_px = cx - (640 // 2)
                dic = round(pid.update(offset_px))
                z_speed = dic

                if distance < APRILTAG_STOP_DISTANCE:
                    y_speed = 0
                elif distance < APRILTAG_SLOW_DISTANCE:
                    y_speed = int(
                        np.clip(
                            (distance - APRILTAG_STOP_DISTANCE) * 3,
                            5,
                            APRILTAG_CHASE_SPEED,
                        )
                    )
                else:
                    y_speed = APRILTAG_CHASE_SPEED

                if abs(y_speed) < 1 and abs(z_speed) < 3:
                    robot.stop_chassis()
                    _log.bind(
                        state="idle", distance=distance, offset_px=offset_px
                    ).trace("待命")
                    if distance < APRILTAG_STOP_DISTANCE:
                        reached = True
                        break
                else:
                    z_speed = int(
                        np.clip(
                            z_speed, -APRILTAG_TURN_SPEED_MAX, APRILTAG_TURN_SPEED_MAX
                        )
                    )
                    robot.mecanum_move_xyz(0, y_speed, z_speed)
                    _log.bind(
                        state="chase",
                        distance=distance,
                        offset_px=offset_px,
                        y_speed=y_speed,
                        z_speed=z_speed,
                    ).trace("追踪")

            stop_event.wait(0.05)

    def vision_loop():
        while not stop_event.is_set():
            try:
                tags = robot.get_apriltag_total_info()
            except Exception:
                _log.opt(exception=True).warning("AprilTag 推理异常")
                stop_event.wait(0.05)
                continue

            target = _get_target_tag(tags, target_id) if tags else None
            with lock:
                state["tag"] = target
                if target is not None:
                    state["last_seen_at"] = time.monotonic()

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    vis_thread = threading.Thread(target=vision_loop, daemon=True)

    try:
        ctrl_thread.start()
        vis_thread.start()
        while (
            vis_thread.is_alive() and ctrl_thread.is_alive() and not stop_event.is_set()
        ):
            stop_event.wait(0.05)
    finally:
        stop_event.set()
        try:
            robot.stop_chassis()
        except Exception:
            pass

    return reached


def unload_phase(robot, zone, sensor_id, perception, trip_index=None):
    """从取货区导航到 A/B 卸货区

    子阶段：
      4a: 追踪 AprilTag 到达卸货区附近
      4b: 巡线进入卸货区
    """
    _log.bind(zone=zone).info("开始导航到卸货区")

    # ── 子阶段 4a: 追踪 AprilTag ──
    _log.info("子阶段 4a: 追踪 AprilTag")
    robot.load_models(["apriltag_qrcode"])
    _log.success("AprilTag 模型加载完成")
    time.sleep(1)

    reached = _chase_apriltag(robot, TARGET_TAG_ID, APRILTAG_TARGET_DISTANCE, sensor_id)
    if not reached:
        _log.warning("未追踪到 AprilTag")
        return False

    # 左转进入巡线区域
    _log.bind(action="turn_left", speed=TURN_SPEED, angle=TURN_ANGLE).debug(
        "左转进入巡线区域"
    )
    robot.mecanum_turn_speed_times(2, TURN_SPEED, TURN_ANGLE, 2)
    time.sleep(1)
    robot.stop_chassis()
    time.sleep(0.5)

    # ── 子阶段 4b: 巡线进入卸货区 ──
    _log.info("子阶段 4b: 巡线进入卸货区")
    route_completed = False
    try:
        robot.release_models(["apriltag_qrcode"])
        if zone not in UNLOAD_ROUTE_BY_ZONE:
            raise ValueError(f"不支持的卸货区: {zone}")
        route = UNLOAD_ROUTE_BY_ZONE[zone]
        route_completed = _follow_lane_route(
            robot,
            perception,
            f"pickup-to-{zone}",
            route,
            sensor_id=sensor_id,
            # A/B 的到达由明确的 ARRIVE 路口与最后 24 cm
            # 入区动作确定。红外测距只要遇到场地物体就可能
            # 提前结束路线，所以此处不将其作为卸货到达条件。
            use_distance_stop=False,
            trip_index=trip_index,
        )

        if not route_completed:
            _log.warning("卸货路线未完成，夹爪保持闭合")
            return False

        _log.bind(action="clamp_release").info("已到达卸货点，夹手张开")
        robot.mechanical_clamp_release()
        _log.success("卸货完成")
        return True
    finally:
        try:
            robot.stop_chassis()
        except Exception:
            pass
        if not route_completed:
            _log.warning("卸货区导航未完成，未执行放货")


def _country_task_failure(robot, message, **context):
    robot.stop_chassis()
    _log.bind(**context).error(message)
    robot.play_audio_tts(message, 0, wait=True)
    return False


def execute_country_task(robot, task, sensor_id, perception):
    """执行国赛两次搬运；只有第二次卸货成功才播报完成。"""
    transmit_country_task(robot, task)

    _log.bind(trip=1, route="start-to-pickup").info(
        "国赛阶段 1/7：启动区巡线到取货区"
    )
    if not line_follow_phase(robot, perception, trip_index=1):
        return _country_task_failure(
            robot, "去程巡线未完成，程序退出", trip=1, phase="start-to-pickup"
        )

    robot.open_camera()
    time.sleep(1)
    orders = (task.first, task.second)
    for trip, order in enumerate(orders, start=1):
        _log.bind(
            trip=trip,
            color=order.color,
            zone=order.zone,
        ).info(f"国赛第 {trip} 次搬运：识别并抓取目标色块")
        if not track_and_grab_phase(robot, order.color, sensor_id):
            return _country_task_failure(
                robot,
                f"第{trip}次抓取失败，程序退出",
                trip=trip,
                color=order.color,
                phase="grab",
            )

        _log.bind(
            trip=trip,
            color=order.color,
            zone=order.zone,
            route=f"pickup-to-{order.zone}",
        ).info(f"国赛第 {trip} 次搬运：导航并卸货")
        if not unload_phase(
            robot, order.zone, sensor_id, perception, trip_index=trip
        ):
            return _country_task_failure(
                robot,
                f"第{trip}次卸货失败，程序退出",
                trip=trip,
                zone=order.zone,
                phase="unload",
            )

        if trip == 1:
            _log.bind(
                trip=trip,
                zone=order.zone,
                route=f"return-{order.zone}-to-pickup",
            ).info("第一次卸货成功，开始返回取货区")
            _raise_arm_for_return_turn(robot)
            if not return_to_pickup_phase(
                robot, order.zone, perception, trip_index=trip
            ):
                return _country_task_failure(
                    robot,
                    "第一次卸货后返程失败，程序退出",
                    trip=trip,
                    zone=order.zone,
                    phase="return-to-pickup",
                )

    transmit_country_task(robot, task)
    robot.play_audio_tts("任务一已完成", 0, wait=True)
    _log.bind(
        first_zone=task.first.zone,
        second_zone=task.second.zone,
    ).success("国赛任务一两次搬运均已完成")
    return True


# ============================================================
# 连接辅助
# ============================================================


def _initialize_robot_with_deadline(ip):
    """复现 SDK 初始化流程，但为无超时的 getLanguage RPC 加截止时间。"""
    robot = ugot.UGOT()
    address = f"{ip}:50051"
    print(address)

    # UGOT.initialize() 的最后一步 getLanguage() 没有 deadline，机器人服务
    # 半响应时会永久阻塞。当前 SDK 没有公开的超时参数，只能在模块创建后
    # 给该 unary RPC 包一层有界调用，其余初始化步骤保持 SDK 原样。
    robot._UGOT__initialize_modules(address)
    robot._UGOT__initialize_http_client(ip)
    language_rpc = robot.DEVICE.client.getLanguage

    def get_language_with_deadline(request):
        return language_rpc(
            request,
            timeout=ROBOT_INITIALIZE_RPC_TIMEOUT_SECONDS,
        )

    robot.DEVICE.client.getLanguage = get_language_with_deadline
    robot._UGOT__configure_language()
    return robot


def _connect_robot():
    ip = ROBOT_IP
    if ip:
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
            _log.bind(ip=ip).error("无效 IP 地址")
            return None
        _log.bind(ip=ip, source="config").info("使用指定 IP")
    else:
        scanner = ugot.UGOT()
        _log.info("扫描 UGOT 设备...")
        devices = scanner.scan_device()
        if not devices:
            _log.error("未找到设备")
            return None
        name = list(devices.keys())[0]
        ip = list(devices.values())[0]
        _log.bind(device=name, ip=ip).info("发现设备")

    _log.bind(port=50051).info("检测端口...")
    if not wait_port(ip, 50051, timeout=15):
        _log.bind(ip=ip, port=50051).error("端口不可达")
        return None
    _log.success("端口连通")

    _log.info("初始化 SDK...")
    for attempt in range(3):
        try:
            # 每次使用全新对象，避免失败通道污染下一次重试。
            robot = _initialize_robot_with_deadline(ip)
            _log.success("初始化成功")
            return robot
        except Exception as error:
            _log.bind(
                attempt=attempt + 1,
                error=repr(error),
                timeout=ROBOT_INITIALIZE_RPC_TIMEOUT_SECONDS,
            ).warning("初始化失败")
            if attempt < 2:
                time.sleep(2)
    _log.error("连续 3 次初始化失败")
    return None


# ============================================================
# Main
# ============================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="UGOT 国赛任务一双次搬运")
    parser.add_argument(
        "--voice-test",
        action="store_true",
        help="只识别并解析国赛双任务语音，不移动机器人",
    )
    parser.add_argument(
        "--return-route-test",
        choices=("A", "B", "a", "b"),
        help="从指定存储区测试180度调头及返程，不执行抓取卸货",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _log.success("=" * 48)
    _log.success("UGOT 国赛任务一：双指令 → 两次取货、运输与卸货")
    _log.success("=" * 48)

    robot = _connect_robot()
    if robot is None:
        return

    try:
        robot.set_volume(100)
        time.sleep(0.5)

        if args.return_route_test:
            try:
                lane_perception = LanePerception()
            except Exception:
                _log.opt(exception=True).critical("巡线模型初始化失败，机器人不会移动")
                return
            zone = args.return_route_test.upper()
            _log.bind(zone=zone).warning(
                "返程诊断模式：请确认机器人已放在对应卸货位置且前方可安全左转180度"
            )
            completed = return_to_pickup_phase(robot, zone, lane_perception)
            _log.bind(zone=zone, completed=completed).success("返程诊断结束")
            return

        robot.play_audio_tts(
            "请说出第一次和第二次搬运的完整指令",
            0,
            wait=True,
        )
        task = collect_country_task(robot)
        if task is None:
            robot.play_audio_tts("仍未完整识别两次搬运任务，程序退出", 0, wait=True)
            return

        speak_country_confirmation(robot, task)
        if args.voice_test:
            _log.bind(payload=task.to_official_payload()).success(
                "语音诊断完成，未初始化视觉且未移动机器人"
            )
            return

        try:
            lane_perception = LanePerception()
        except Exception:
            _log.opt(exception=True).critical("巡线模型初始化失败，机器人不会移动")
            return

        sensor_id = _discover_infrared_id(robot)
        _log.bind(sensor_id=sensor_id).info("红外传感器 ID")
        execute_country_task(robot, task, sensor_id, lane_perception)

    except KeyboardInterrupt:
        _log.info("收到停止信号")
    except Exception as error:
        _log.opt(exception=True).bind(error=repr(error)).error("运行异常")
    finally:
        try:
            robot.stop_chassis()
        except Exception:
            pass
        _log.success("=" * 48)
        _log.success("任务结束")
        _log.success("=" * 48)


if __name__ == "__main__":
    main()
