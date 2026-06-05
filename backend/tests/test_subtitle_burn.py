"""烧录字幕单元测试 - 覆盖 AC1-AC6（纯函数 + 命令构造，不依赖 libass 端到端）。

AC-1 extract_clip_subtitles：过滤/归零对齐/裁剪/≤0.3s 漂移
AC-2 to_ass：限宽换行≤max_lines + 底部安全区 MarginV + 合法 ASS 结构
AC-3 三套模板产出不同 ASS 样式；非法 template_id -> 报错
AC-4 片段时间窗无字幕 -> 优雅空结果，不报错
AC-5 clip_service 缓存复用 + subtitle_variants 写回 + 缺 SRT -> 报错
AC-6 burn_subtitles 命令构造：libx264 / 非 -c:v copy / ass 滤镜，并在失败时记录 stderr
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保可 import backend.*
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.utils.subtitle_processor import SubtitleProcessor
from backend.utils.video_processor import VideoProcessor
from backend.core import shared_config


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def srt_3seg(tmp_path):
    """3 段 SRT：[1,5] [5,10] [20,23]。"""
    srt = tmp_path / "input.srt"
    srt.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:05,000\n"
        "第一段字幕\n\n"
        "2\n"
        "00:00:05,000 --> 00:00:10,000\n"
        "第二段字幕\n\n"
        "3\n"
        "00:00:20,000 --> 00:00:23,000\n"
        "第三段字幕\n\n",
        encoding="utf-8",
    )
    return srt


@pytest.fixture
def processor():
    return SubtitleProcessor()


# --------------------------------------------------------------------------
# AC-1 extract_clip_subtitles
# --------------------------------------------------------------------------

class TestExtractClipSubtitles:
    def test_filters_only_overlapping_segments(self, processor, srt_3seg):
        """窗口 [0.5, 7.0] 只命中前两段，排除 20s 段。"""
        segs = processor.extract_clip_subtitles(srt_3seg, 0.5, 7.0)
        assert len(segs) == 2
        assert [s["text"] for s in segs] == ["第一段字幕", "第二段字幕"]

    def test_zero_alignment(self, processor, srt_3seg):
        """归零对齐：减去片段起点。"""
        segs = processor.extract_clip_subtitles(srt_3seg, 0.5, 7.0)
        # seg1 原 [1,5] -> [0.5,4.5]
        assert abs(segs[0]["startTime"] - 0.5) <= 0.3
        assert abs(segs[0]["endTime"] - 4.5) <= 0.3
        # seg2 原 [5,10] -> [4.5, min(6.5,5)] = [4.5,6.5]
        assert abs(segs[1]["startTime"] - 4.5) <= 0.3

    def test_clips_tail_overflow_to_duration(self, processor, srt_3seg):
        """尾部溢出裁剪到 duration。窗口 [0.5,7.0] duration=6.5，seg2 末端裁到 6.5。"""
        segs = processor.extract_clip_subtitles(srt_3seg, 0.5, 7.0)
        duration = 7.0 - 0.5
        assert segs[-1]["endTime"] <= duration + 1e-6
        assert abs(segs[-1]["endTime"] - duration) <= 0.3

    def test_head_overflow_clamped_to_zero(self, processor, srt_3seg):
        """头部溢出裁剪到 0：窗口 [2,8]，seg1 原 [1,5] -> [0,3]。"""
        segs = processor.extract_clip_subtitles(srt_3seg, 2.0, 8.0)
        assert segs[0]["startTime"] == 0.0
        assert abs(segs[0]["endTime"] - 3.0) <= 0.3

    def test_missing_file_returns_empty(self, processor, tmp_path):
        assert processor.extract_clip_subtitles(tmp_path / "nope.srt", 0, 5) == []

    def test_invalid_window_returns_empty(self, processor, srt_3seg):
        assert processor.extract_clip_subtitles(srt_3seg, 5.0, 5.0) == []
        assert processor.extract_clip_subtitles(srt_3seg, 6.0, 5.0) == []


# --------------------------------------------------------------------------
# AC-4 空字幕优雅处理
# --------------------------------------------------------------------------

class TestEmptySubtitleWindow:
    def test_no_overlap_returns_empty_list(self, processor, srt_3seg):
        """窗口 [11,19] 落在 seg2 与 seg3 之间，无命中。"""
        segs = processor.extract_clip_subtitles(srt_3seg, 11.0, 19.0)
        assert segs == []


# --------------------------------------------------------------------------
# AC-2 to_ass 换行 / 安全区 / 合法结构
# --------------------------------------------------------------------------

class TestToAss:
    def test_valid_ass_structure(self, processor):
        tpl = shared_config.SUBTITLE_TEMPLATES["classic_white"]
        ass = processor.to_ass(
            [{"startTime": 0.0, "endTime": 2.0, "text": "测试字幕"}],
            tpl, 1080, 1920,
        )
        assert "[Script Info]" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass
        assert "PlayResX: 1080" in ass
        assert "PlayResY: 1920" in ass
        assert "Style: Default," in ass
        assert "Dialogue: 0," in ass

    def test_safe_margin_v_proportional_to_height(self, processor):
        tpl = shared_config.SUBTITLE_TEMPLATES["classic_white"]
        ass = processor.to_ass(
            [{"startTime": 0.0, "endTime": 1.0, "text": "x"}], tpl, 1080, 1920,
        )
        expected_mv = int(round(1920 * shared_config.SUBTITLE_SAFE_MARGIN_PCT))
        # Style 行最后几个字段含 MarginV
        style_line = [l for l in ass.splitlines() if l.startswith("Style: Default")][0]
        assert str(expected_mv) in style_line

    def test_wrap_limits_to_max_lines(self, processor):
        tpl = shared_config.SUBTITLE_TEMPLATES["classic_white"]
        long_text = "字" * 100  # 远超 max_lines * max_chars
        ass = processor.to_ass(
            [{"startTime": 0.0, "endTime": 2.0, "text": long_text}], tpl, 1080, 1920,
        )
        dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue")][0]
        text_field = dialogue.split(",,", 1)[1]
        # 行数 = \N 数 + 1，不应超过 max_lines
        line_count = text_field.count("\\N") + 1
        assert line_count <= shared_config.SUBTITLE_MAX_LINES
        # 溢出应有省略号
        assert "…" in text_field

    def test_wrap_short_text_single_line(self, processor):
        wrapped = processor._wrap_text("短句", shared_config.SUBTITLE_MAX_CHARS_PER_LINE,
                                       shared_config.SUBTITLE_MAX_LINES)
        assert "\n" not in wrapped

    def test_font_size_scales_with_canvas_height(self, processor):
        tpl = shared_config.SUBTITLE_TEMPLATES["classic_white"]
        ass_tall = processor.to_ass([{"startTime": 0, "endTime": 1, "text": "a"}], tpl, 1080, 1920)
        ass_wide = processor.to_ass([{"startTime": 0, "endTime": 1, "text": "a"}], tpl, 1920, 1080)
        size_tall = int(round(1920 * tpl["base_font_pct"]))
        size_wide = int(round(1080 * tpl["base_font_pct"]))
        assert str(size_tall) in ass_tall
        assert str(size_wide) in ass_wide
        assert size_tall != size_wide

    def test_ass_time_format(self, processor):
        assert processor._seconds_to_ass_time(0.0) == "0:00:00.00"
        assert processor._seconds_to_ass_time(65.5) == "0:01:05.50"
        assert processor._seconds_to_ass_time(3661.0) == "1:01:01.00"


# --------------------------------------------------------------------------
# AC-3 三套模板差异 + 非法模板
# --------------------------------------------------------------------------

class TestTemplates:
    def test_three_templates_exist(self):
        for tid in ("classic_white", "bold_box", "bright_accent"):
            assert tid in shared_config.SUBTITLE_TEMPLATES

    def test_templates_produce_distinct_styles(self, processor):
        seg = [{"startTime": 0.0, "endTime": 2.0, "text": "样式"}]
        styles = {}
        for tid, tpl in shared_config.SUBTITLE_TEMPLATES.items():
            ass = processor.to_ass(seg, tpl, 1080, 1920)
            styles[tid] = [l for l in ass.splitlines() if l.startswith("Style: Default")][0]
        # classic_white(border_style 1) vs bold_box(border_style 3) 不同
        assert styles["classic_white"] != styles["bold_box"]
        # bright_accent 主色为黄，与 classic_white 白色不同
        assert styles["bright_accent"] != styles["classic_white"]
        assert "&H0000F0FF" in styles["bright_accent"]
        assert "&H80000000" in styles["bold_box"]  # 半透明底色块

    def test_invalid_template_raises_in_service(self):
        from backend.services.clip_service import ClipService
        svc = ClipService.__new__(ClipService)  # 跳过 __init__（无需 DB）
        with pytest.raises(ValueError, match="无效的字幕模板"):
            svc.burn_subtitle_clip("clip-x", "not_a_template")


# --------------------------------------------------------------------------
# AC-6 burn_subtitles 命令构造
# --------------------------------------------------------------------------

class TestBurnSubtitlesCommand:
    def _run(self, tmp_path, returncode=0, make_output=True):
        inp = tmp_path / "in.mp4"
        inp.write_bytes(b"x")
        ass = tmp_path / "s.ass"
        ass.write_text("[Events]\n", encoding="utf-8")
        out = tmp_path / "out.mp4"

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            if make_output and returncode == 0:
                out.write_bytes(b"y")
            m = MagicMock()
            m.returncode = returncode
            m.stderr = "ffmpeg boom" if returncode != 0 else ""
            return m

        with patch("backend.utils.video_processor.subprocess.run", side_effect=fake_run), \
             patch("backend.utils.video_processor.get_ffmpeg_path", return_value="ffmpeg"):
            ok = VideoProcessor.burn_subtitles(inp, out, ass)
        return ok, captured.get("cmd", [])

    def test_uses_libx264_not_copy(self, tmp_path):
        ok, cmd = self._run(tmp_path)
        assert ok is True
        # 必须重编码：含 libx264，且不得 -c:v copy
        assert "libx264" in cmd
        cv_idx = [i for i, a in enumerate(cmd) if a == "-c:v"]
        assert cv_idx, "缺少 -c:v 参数"
        for i in cv_idx:
            assert cmd[i + 1] != "copy"

    def test_includes_ass_filter_and_audio(self, tmp_path):
        ok, cmd = self._run(tmp_path)
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1].startswith("ass=")
        assert "aac" in cmd
        assert "+faststart" in cmd

    def test_failure_returns_false_and_logs_stderr(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            ok, cmd = self._run(tmp_path, returncode=1, make_output=False)
        assert ok is False
        assert any("ffmpeg boom" in r.message or "烧录字幕失败" in r.message
                   for r in caplog.records)


# --------------------------------------------------------------------------
# AC-5 clip_service 缓存复用 / 写回 / 缺 SRT
# --------------------------------------------------------------------------

class TestBurnSubtitleClipService:
    def _make_service(self, clip):
        from backend.services.clip_service import ClipService
        svc = ClipService.__new__(ClipService)
        svc.get = MagicMock(return_value=clip)
        svc.update = MagicMock()
        return svc

    def _make_clip(self, tmp_path, video_name="clip1.mp4", metadata=None):
        video = tmp_path / video_name
        video.write_bytes(b"v")
        clip = MagicMock()
        clip.id = "clip1"
        clip.project_id = "proj1"
        clip.video_path = str(video)
        clip.start_time = 0.5
        clip.end_time = 7.0
        clip.clip_metadata = metadata or {}
        return clip

    def _patch_paths(self, projects_dir):
        return patch("backend.core.path_utils.get_projects_directory",
                     return_value=Path(projects_dir))

    def test_missing_srt_raises(self, tmp_path):
        clip = self._make_clip(tmp_path)
        svc = self._make_service(clip)
        proj_root = tmp_path / "projects"
        (proj_root / "proj1" / "raw").mkdir(parents=True)
        # 不创建 input.srt
        with self._patch_paths(proj_root):
            with pytest.raises(ValueError, match="SRT"):
                svc.burn_subtitle_clip("clip1", "classic_white")

    def test_cache_reuse_skips_burn(self, tmp_path):
        # 已存在产物文件
        cached_video = tmp_path / "clip1_sub_classic_white_source.mp4"
        cached_video.write_bytes(b"cached")
        meta = {
            "subtitle_variants": [{
                "template_id": "classic_white",
                "aspect_ratio": "source",
                "video_path": str(cached_video),
                "thumbnail_path": None,
            }]
        }
        clip = self._make_clip(tmp_path, metadata=meta)
        svc = self._make_service(clip)
        proj_root = tmp_path / "projects"
        raw = proj_root / "proj1" / "raw"
        raw.mkdir(parents=True)
        (raw / "input.srt").write_text(
            "1\n00:00:01,000 --> 00:00:05,000\n你好\n\n", encoding="utf-8")

        with self._patch_paths(proj_root), \
             patch.object(VideoProcessor, "burn_subtitles") as mock_burn:
            result = svc.burn_subtitle_clip("clip1", "classic_white")

        assert result["status"] == "completed"
        assert result["cached"] is True
        mock_burn.assert_not_called()

    def test_empty_window_graceful_skip(self, tmp_path):
        clip = self._make_clip(tmp_path)
        clip.start_time = 11.0
        clip.end_time = 19.0  # 落在字幕间隙
        svc = self._make_service(clip)
        proj_root = tmp_path / "projects"
        raw = proj_root / "proj1" / "raw"
        raw.mkdir(parents=True)
        (raw / "input.srt").write_text(
            "1\n00:00:01,000 --> 00:00:05,000\n第一段\n\n"
            "2\n00:00:20,000 --> 00:00:23,000\n第三段\n\n", encoding="utf-8")

        with self._patch_paths(proj_root), \
             patch.object(VideoProcessor, "burn_subtitles") as mock_burn:
            result = svc.burn_subtitle_clip("clip1", "classic_white")

        assert result["status"] == "empty"
        mock_burn.assert_not_called()

    def test_successful_burn_writes_variant(self, tmp_path):
        clip = self._make_clip(tmp_path)
        svc = self._make_service(clip)
        proj_root = tmp_path / "projects"
        raw = proj_root / "proj1" / "raw"
        raw.mkdir(parents=True)
        (raw / "input.srt").write_text(
            "1\n00:00:01,000 --> 00:00:05,000\n你好世界\n\n", encoding="utf-8")

        def fake_burn(inp, out, ass, fonts=None):
            Path(out).write_bytes(b"burned")
            return True

        with self._patch_paths(proj_root), \
             patch.object(VideoProcessor, "burn_subtitles", side_effect=fake_burn), \
             patch.object(VideoProcessor, "get_video_resolution", return_value=(1920, 1080)), \
             patch.object(VideoProcessor, "extract_thumbnail", return_value=False):
            result = svc.burn_subtitle_clip("clip1", "classic_white")

        assert result["status"] == "completed"
        assert result["cached"] is False
        # update 被调用并写回 subtitle_variants
        svc.update.assert_called_once()
        _, kwargs = svc.update.call_args
        meta = kwargs.get("clip_metadata") or {}
        variants = meta.get("subtitle_variants", [])
        assert len(variants) == 1
        assert variants[0]["template_id"] == "classic_white"
        assert variants[0]["aspect_ratio"] == "source"

    def test_burn_failure_raises_runtime_error(self, tmp_path):
        clip = self._make_clip(tmp_path)
        svc = self._make_service(clip)
        proj_root = tmp_path / "projects"
        raw = proj_root / "proj1" / "raw"
        raw.mkdir(parents=True)
        (raw / "input.srt").write_text(
            "1\n00:00:01,000 --> 00:00:05,000\n你好\n\n", encoding="utf-8")

        with self._patch_paths(proj_root), \
             patch.object(VideoProcessor, "burn_subtitles", return_value=False), \
             patch.object(VideoProcessor, "get_video_resolution", return_value=(1920, 1080)):
            with pytest.raises(RuntimeError, match="字幕烧录失败"):
                svc.burn_subtitle_clip("clip1", "classic_white")
