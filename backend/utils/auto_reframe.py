"""
智能竖屏重构 - 主体（人脸）跟踪算法模块

职责：
1. 对视频按固定间隔采样关键帧，使用 OpenCV DNN（YuNet）人脸检测，
   得到主体水平中心点（归一化 0~1）的时间序列。
2. 对中心点序列做时间平滑（滑动平均 + 限速位移），避免裁剪框抖动。
3. 提供降级判定：有效人脸采样不足时返回 track 失败，由上层走居中/模糊兜底。

设计要点：
- 纯函数（smooth_centers / compute_crop_x / detect_success / build_segments）
  不依赖 cv2，便于单元测试。
- 人脸检测层（compute_focus_track）在缺少 cv2 或模型文件时优雅降级，
  返回空轨迹而非抛异常，由上层进入降级路径。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..core.shared_config import (
    FACE_SAMPLE_INTERVAL,
    SMOOTH_WINDOW,
    MAX_CENTER_SHIFT_RATIO,
    REFRAME_FACE_MIN_CONFIDENCE,
    REFRAME_MIN_TRACK_HIT_RATIO,
    REFRAME_DETECT_TIMEOUT_SECONDS,
    REFRAME_FACE_MODEL_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class FocusPoint:
    """单个采样点的主体焦点信息。"""
    t: float            # 采样时间（秒）
    x_ratio: float      # 主体水平中心，归一化到 [0, 1]
    confidence: float   # 检测置信度（无主体时为 0）
    hit: bool           # 该采样点是否检测到有效主体


@dataclass
class CropSegment:
    """一段时间内固定的裁剪窗口（用于分段 crop 表达式）。"""
    start: float        # 段开始时间（秒）
    end: float          # 段结束时间（秒）
    x_ratio: float      # 该段裁剪窗口中心 x 归一化值 [0, 1]


def clamp(value: float, low: float, high: float) -> float:
    """将 value 限制到 [low, high]。"""
    if value < low:
        return low
    if value > high:
        return high
    return value


def smooth_centers(
    x_ratios: List[float],
    window: int = SMOOTH_WINDOW,
    max_shift: float = MAX_CENTER_SHIFT_RATIO,
) -> List[float]:
    """
    对中心点 x 归一化序列做时间平滑。

    两步：
    1. 奇数窗口滑动平均，抑制高频抖动。
    2. 限速：相邻输出点位移不超过 max_shift，避免突跳。

    纯函数，便于测试。空输入返回空列表。
    """
    if not x_ratios:
        return []
    n = len(x_ratios)
    if window < 1:
        window = 1
    half = window // 2

    # 1) 滑动平均
    averaged: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        averaged.append(sum(x_ratios[lo:hi]) / (hi - lo))

    # 2) 限速位移
    smoothed: List[float] = [clamp(averaged[0], 0.0, 1.0)]
    for i in range(1, n):
        prev = smoothed[-1]
        target = averaged[i]
        delta = target - prev
        if delta > max_shift:
            delta = max_shift
        elif delta < -max_shift:
            delta = -max_shift
        smoothed.append(clamp(prev + delta, 0.0, 1.0))
    return smoothed


def detect_success(
    track: List[FocusPoint],
    min_hit_ratio: float = REFRAME_MIN_TRACK_HIT_RATIO,
) -> bool:
    """
    判定主体跟踪是否成功：有效命中采样占比 >= 阈值。

    纯函数。空轨迹判为失败（触发降级）。
    """
    if not track:
        return False
    hits = sum(1 for p in track if p.hit)
    return (hits / len(track)) >= min_hit_ratio


def compute_crop_x(
    x_ratio: float,
    src_width: int,
    src_height: int,
    target_w: int,
    target_h: int,
) -> int:
    """
    根据主体中心归一化 x 与目标比例，计算 crop 窗口左上角 x（像素，整数）。

    裁剪窗口宽度 = src_height * target_w / target_h（保持目标宽高比、铺满源高度），
    水平跟随主体中心并做边界 clamp。纯函数，便于测试。
    """
    crop_w = int(round(src_height * target_w / target_h))
    crop_w = min(crop_w, src_width)
    center_px = x_ratio * src_width
    x = int(round(center_px - crop_w / 2))
    x = int(clamp(x, 0, max(0, src_width - crop_w)))
    return x


def build_segments(
    track: List[FocusPoint],
    smoothed_x: List[float],
    duration: float,
    min_change: float = 0.01,
) -> List[CropSegment]:
    """
    将平滑后的中心点序列压缩为若干 CropSegment，
    相邻变化小于 min_change 的采样点合并到同一段，减少 ffmpeg 分段数量。

    纯函数。track 为空时返回单段居中（x_ratio=0.5）。
    """
    if not track or not smoothed_x:
        return [CropSegment(start=0.0, end=max(duration, 0.0), x_ratio=0.5)]

    segments: List[CropSegment] = []
    seg_start = track[0].t
    seg_x = smoothed_x[0]
    for i in range(1, len(track)):
        if abs(smoothed_x[i] - seg_x) >= min_change:
            segments.append(CropSegment(start=seg_start, end=track[i].t, x_ratio=seg_x))
            seg_start = track[i].t
            seg_x = smoothed_x[i]
    segments.append(CropSegment(start=seg_start, end=max(duration, track[-1].t), x_ratio=seg_x))
    return segments


def _load_face_detector(width: int, height: int):
    """
    加载 OpenCV YuNet 人脸检测器。失败（无 cv2 / 无模型 / 版本不支持）返回 None。
    """
    model_path = Path(REFRAME_FACE_MODEL_PATH)
    if not model_path.exists():
        logger.warning("人脸检测模型不存在，竖屏将走降级路径: %s", model_path)
        return None
    try:
        import cv2  # 延迟导入，避免无 CV 依赖时影响其它功能
    except ImportError:
        logger.warning("未安装 opencv，竖屏将走降级路径")
        return None
    try:
        detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (width, height),
            score_threshold=REFRAME_FACE_MIN_CONFIDENCE,
        )
        return detector
    except Exception as e:  # noqa: BLE001
        logger.warning("初始化 YuNet 人脸检测器失败，走降级路径: %s", e)
        return None


def compute_focus_track(
    input_video: Path,
    duration: float,
    sample_interval: float = FACE_SAMPLE_INTERVAL,
    timeout_seconds: float = REFRAME_DETECT_TIMEOUT_SECONDS,
) -> List[FocusPoint]:
    """
    采样关键帧并检测人脸，返回主体中心点时间序列。

    任意环节失败（无 cv2 / 无模型 / 读帧失败 / 超时）均返回已采集的部分轨迹，
    由上层 detect_success 判定是否降级。不抛异常。
    """
    try:
        import cv2
    except ImportError:
        logger.warning("未安装 opencv，返回空轨迹（降级）")
        return []

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        logger.warning("无法打开视频进行人脸检测，降级: %s", input_video)
        return []

    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if src_w <= 0 or src_h <= 0:
            return []

        detector = _load_face_detector(src_w, src_h)
        if detector is None:
            return []

        track: List[FocusPoint] = []
        t = 0.0
        started = time.monotonic()
        # 至少采一帧（极短片段 < 1s 时也产出单点）
        while True:
            if time.monotonic() - started > timeout_seconds:
                logger.warning("人脸检测超时，使用已采集 %d 个采样点", len(track))
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            x_ratio, conf, hit = _detect_largest_face_center(detector, frame, src_w)
            track.append(FocusPoint(t=t, x_ratio=x_ratio, confidence=conf, hit=hit))

            t += sample_interval
            if t >= duration:
                break
        return track
    finally:
        cap.release()


def _detect_largest_face_center(detector, frame, src_w: int):
    """
    在单帧上检测最大人脸，返回 (x_ratio, confidence, hit)。
    无人脸返回 (0.5, 0.0, False)。
    """
    try:
        _, faces = detector.detect(frame)
    except Exception as e:  # noqa: BLE001
        logger.debug("单帧人脸检测异常: %s", e)
        return 0.5, 0.0, False
    if faces is None or len(faces) == 0:
        return 0.5, 0.0, False

    # faces[i] = [x, y, w, h, ...landmarks..., score]; 取面积最大者
    best = None
    best_area = -1.0
    for f in faces:
        w = float(f[2])
        h = float(f[3])
        area = w * h
        if area > best_area:
            best_area = area
            best = f
    if best is None:
        return 0.5, 0.0, False

    cx = float(best[0]) + float(best[2]) / 2.0
    score = float(best[-1]) if len(best) > 0 else 0.0
    x_ratio = clamp(cx / src_w, 0.0, 1.0) if src_w > 0 else 0.5
    hit = score >= REFRAME_FACE_MIN_CONFIDENCE
    return x_ratio, score, hit
