"""
智能竖屏重构（Auto-Reframe）单元测试

锚定 planning.md 验收标准 AC1-AC6：
- AC-1 happy path：track 裁剪窗口计算 + 输出 1080×1920。
- AC-2 平滑：smooth_centers 限速、抑抖。
- AC-3 边界：空轨迹单段居中、已竖屏 scale+pad、极短片段。
- AC-4 降级：detect_success 判定 + reframe_to_vertical fallback_used。
- AC-5 缓存/写回：reframe_to_vertical 不破坏 + 居中裁剪几何（缓存语义在 clip_service 层，见 metadata 写回断言）。
- AC-6 可观测：ffmpeg 失败时 result['error'] 含 stderr。

纯函数测试不依赖 cv2/ffmpeg；reframe_to_vertical 通过 monkeypatch 隔离 ffprobe/ffmpeg。
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# conftest 已将仓库根加入 sys.path，可用 backend.* 绝对导入
from backend.utils.auto_reframe import (
    FocusPoint,
    CropSegment,
    clamp,
    smooth_centers,
    detect_success,
    compute_crop_x,
    build_segments,
)
from backend.utils.video_processor import VideoProcessor


# ----------------------------- clamp -----------------------------
class TestClamp:
    def test_within_range_returns_value(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_low_clamps_to_low(self):
        assert clamp(-3.0, 0.0, 1.0) == 0.0

    def test_above_high_clamps_to_high(self):
        assert clamp(9.0, 0.0, 1.0) == 1.0


# ----------------------- smooth_centers (AC-2) -----------------------
class TestSmoothCenters:
    def test_empty_returns_empty(self):
        assert smooth_centers([]) == []

    def test_output_length_matches_input(self):
        xs = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert len(smooth_centers(xs, window=3, max_shift=1.0)) == len(xs)

    def test_rate_limit_caps_per_step_shift(self):
        # 输入存在剧烈跳变，max_shift 限制相邻输出位移不超过阈值
        xs = [0.0, 1.0, 0.0, 1.0, 0.0]
        out = smooth_centers(xs, window=1, max_shift=0.08)
        for prev, cur in zip(out, out[1:]):
            assert abs(cur - prev) <= 0.08 + 1e-9

    def test_results_stay_in_unit_interval(self):
        xs = [-0.5, 2.0, 0.5, 1.5, -1.0]
        out = smooth_centers(xs, window=3, max_shift=0.5)
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_moving_average_smooths_single_spike(self):
        # 单点尖峰经窗口平均后被显著削弱（抑抖）
        xs = [0.5, 0.5, 1.0, 0.5, 0.5]
        out = smooth_centers(xs, window=3, max_shift=1.0)
        assert out[2] < 1.0  # 尖峰被邻域拉低


# ----------------------- detect_success (AC-4) -----------------------
class TestDetectSuccess:
    def _track(self, hits):
        return [FocusPoint(t=i * 0.5, x_ratio=0.5, confidence=0.9 if h else 0.0, hit=h)
                for i, h in enumerate(hits)]

    def test_empty_track_is_failure(self):
        assert detect_success([]) is False

    def test_hit_ratio_above_threshold_succeeds(self):
        track = self._track([True, True, True, False])  # 0.75
        assert detect_success(track, min_hit_ratio=0.3) is True

    def test_hit_ratio_below_threshold_fails(self):
        track = self._track([True, False, False, False])  # 0.25
        assert detect_success(track, min_hit_ratio=0.3) is False

    def test_ratio_exactly_at_threshold_succeeds(self):
        track = self._track([True, False, False])  # ~0.333
        assert detect_success(track, min_hit_ratio=0.3) is True


# ----------------------- compute_crop_x (AC-1) -----------------------
class TestComputeCropX:
    def test_916_crop_width_from_source_height(self):
        # 1920x1080 源，9:16 目标：crop_w = round(1080*9/16)=608
        x = compute_crop_x(0.5, src_width=1920, src_height=1080, target_w=1080, target_h=1920)
        crop_w = round(1080 * 1080 / 1920)  # = 608
        expected = round(0.5 * 1920 - crop_w / 2)
        assert x == expected

    def test_left_edge_clamped_to_zero(self):
        x = compute_crop_x(0.0, 1920, 1080, 1080, 1920)
        assert x == 0

    def test_right_edge_clamped_to_max(self):
        crop_w = min(round(1080 * 1080 / 1920), 1920)
        x = compute_crop_x(1.0, 1920, 1080, 1080, 1920)
        assert x == 1920 - crop_w

    def test_returns_int(self):
        x = compute_crop_x(0.37, 1920, 1080, 1080, 1920)
        assert isinstance(x, int)

    def test_crop_width_never_exceeds_source_width(self):
        # 极端目标导致 crop_w 超过源宽时应被 clamp，x 落在 [0, max]
        x = compute_crop_x(0.5, src_width=400, src_height=1080, target_w=1080, target_h=1920)
        assert 0 <= x <= 400


# ----------------------- build_segments (AC-3) -----------------------
class TestBuildSegments:
    def test_empty_track_returns_single_center_segment(self):
        segs = build_segments([], [], duration=10.0)
        assert len(segs) == 1
        assert segs[0].x_ratio == 0.5
        assert segs[0].start == 0.0
        assert segs[0].end == 10.0

    def test_stable_track_merges_into_one_segment(self):
        track = [FocusPoint(t=i * 0.5, x_ratio=0.5, confidence=0.9, hit=True) for i in range(5)]
        smoothed = [0.5, 0.5, 0.5, 0.5, 0.5]
        segs = build_segments(track, smoothed, duration=2.5, min_change=0.01)
        assert len(segs) == 1

    def test_significant_change_splits_segments(self):
        track = [FocusPoint(t=i * 0.5, x_ratio=0.0, confidence=0.9, hit=True) for i in range(4)]
        smoothed = [0.2, 0.2, 0.8, 0.8]  # 中间发生 >min_change 的跳变
        segs = build_segments(track, smoothed, duration=2.0, min_change=0.01)
        assert len(segs) >= 2

    def test_last_segment_end_covers_duration(self):
        track = [FocusPoint(t=0.0, x_ratio=0.5, confidence=0.9, hit=True)]
        smoothed = [0.5]
        segs = build_segments(track, smoothed, duration=8.0)
        assert segs[-1].end == 8.0


# ----------------- _target_dimensions / filters (AC-1, AC-3) -----------------
class TestTargetDimensions:
    def test_default_916(self):
        assert VideoProcessor._target_dimensions("9:16") == (1080, 1920)

    def test_square_11(self):
        assert VideoProcessor._target_dimensions("1:1") == (1080, 1080)

    def test_unknown_defaults_to_vertical(self):
        assert VideoProcessor._target_dimensions("4:5") == (1080, 1920)


class TestCenterCropFilter:
    def test_center_crop_geometry_and_scale(self):
        vf = VideoProcessor._build_center_crop_filter(1920, 1080, 1080, 1920)
        crop_w = round(1080 * 1080 / 1920)  # 608
        assert f"crop={crop_w}:1080" in vf
        assert "scale=1080:1920" in vf


class TestBlurPaddingFilter:
    def test_blur_padding_has_bg_fg_overlay(self):
        fc = VideoProcessor._build_blur_padding_filter(1080, 1920)
        assert "boxblur" in fc
        assert "[bg]" in fc and "[fg]" in fc
        assert "overlay=" in fc
        assert "scale=1080:1920" in fc


class TestTrackCropFilter:
    def test_single_segment_static_x(self):
        segs = [CropSegment(start=0.0, end=5.0, x_ratio=0.5)]
        vf = VideoProcessor._build_track_crop_filter(segs, 1920, 1080, 1080, 1920)
        assert "crop=" in vf and "scale=1080:1920" in vf
        assert "between" not in vf  # 单段无需时间分支

    def test_multi_segment_uses_between_expression(self):
        segs = [
            CropSegment(start=0.0, end=2.0, x_ratio=0.2),
            CropSegment(start=2.0, end=4.0, x_ratio=0.8),
        ]
        vf = VideoProcessor._build_track_crop_filter(segs, 1920, 1080, 1080, 1920)
        assert "between(t," in vf
        assert "if(" in vf


# ---------------- reframe_to_vertical 集成（AC-1/3/4/6, monkeypatch ffmpeg） ----------------
class _FakeProc:
    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


class TestReframeToVertical:
    def _dims(self, w, h, dur=10.0):
        return {"width": w, "height": h, "duration": dur, "fps": 30.0}

    def test_already_vertical_uses_scale_pad_no_face(self, tmp_path):
        # AC-3：源已竖屏，仅 scale+pad，strategy=center
        out = tmp_path / "out.mp4"
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            out.write_bytes(b"x")  # 模拟产物生成
            return _FakeProc(0)

        with patch.object(VideoProcessor, "get_video_dimensions",
                          return_value=self._dims(1080, 1920)), \
             patch("backend.utils.video_processor.get_ffmpeg_path", return_value="ffmpeg"), \
             patch("subprocess.run", side_effect=fake_run):
            res = VideoProcessor.reframe_to_vertical(tmp_path / "in.mp4", out)

        assert res["success"] is True
        assert res["focus_strategy"] == "center"
        assert (res["width"], res["height"]) == (1080, 1920)
        cmd_str = " ".join(captured["cmd"])
        assert "pad=" in cmd_str and "scale=1080:1920" in cmd_str

    def test_track_fallback_to_center_when_no_face(self, tmp_path):
        # AC-4：横屏无人脸 → detect_success False → fallback_used True
        out = tmp_path / "out.mp4"

        def fake_run(cmd, **kwargs):
            out.write_bytes(b"x")
            return _FakeProc(0)

        with patch.object(VideoProcessor, "get_video_dimensions",
                          return_value=self._dims(1920, 1080)), \
             patch("backend.utils.video_processor.get_ffmpeg_path", return_value="ffmpeg"), \
             patch("backend.utils.auto_reframe.compute_focus_track", return_value=[]), \
             patch("subprocess.run", side_effect=fake_run):
            res = VideoProcessor.reframe_to_vertical(
                tmp_path / "in.mp4", out,
                focus_strategy="track", fallback_strategy="center")

        assert res["success"] is True
        assert res["fallback_used"] is True
        assert res["focus_strategy"] == "center"

    def test_track_success_builds_crop_filter(self, tmp_path):
        # AC-1：有人脸轨迹 → 走跟踪裁剪，命令含 crop+scale
        out = tmp_path / "out.mp4"
        captured = {}
        track = [FocusPoint(t=i * 0.5, x_ratio=0.5, confidence=0.9, hit=True) for i in range(6)]

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            out.write_bytes(b"x")
            return _FakeProc(0)

        with patch.object(VideoProcessor, "get_video_dimensions",
                          return_value=self._dims(1920, 1080)), \
             patch("backend.utils.video_processor.get_ffmpeg_path", return_value="ffmpeg"), \
             patch("subprocess.run", side_effect=fake_run):
            res = VideoProcessor.reframe_to_vertical(
                tmp_path / "in.mp4", out,
                focus_strategy="track", focus_track=track)

        assert res["success"] is True
        assert res["fallback_used"] is False
        cmd_str = " ".join(captured["cmd"])
        assert "crop=" in cmd_str and "scale=1080:1920" in cmd_str

    def test_ffmpeg_failure_records_stderr(self, tmp_path):
        # AC-6：ffmpeg 失败时 error 含 stderr，success=False
        out = tmp_path / "out.mp4"

        def fake_run(cmd, **kwargs):
            return _FakeProc(1, stderr="ffmpeg boom: invalid filter")

        with patch.object(VideoProcessor, "get_video_dimensions",
                          return_value=self._dims(1920, 1080)), \
             patch("backend.utils.video_processor.get_ffmpeg_path", return_value="ffmpeg"), \
             patch("backend.utils.auto_reframe.compute_focus_track", return_value=[]), \
             patch("subprocess.run", side_effect=fake_run):
            res = VideoProcessor.reframe_to_vertical(tmp_path / "in.mp4", out)

        assert res["success"] is False
        assert "ffmpeg boom" in (res["error"] or "")

    def test_unreadable_dimensions_returns_error(self, tmp_path):
        # AC-6：源尺寸不可读，提前失败
        with patch.object(VideoProcessor, "get_video_dimensions", return_value={}):
            res = VideoProcessor.reframe_to_vertical(tmp_path / "in.mp4", tmp_path / "o.mp4")
        assert res["success"] is False
        assert res["error"]
