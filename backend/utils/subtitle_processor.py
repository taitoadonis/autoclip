import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pysrt
from pysrt import SubRipItem, SubRipTime

logger = logging.getLogger(__name__)

class SubtitleProcessor:
    """字幕处理器 - 支持字粒度的字幕解析和处理"""
    
    def __init__(self):
        # 分词分隔符：中文标点（，。！？；：“”‘’（）【】、）加空白字符。
        # 使用双引号原始字符串，避免内嵌单引号导致字面量被提前截断/隐式拼接，
        # 同时让 \s 保持原义、不触发 SyntaxWarning。
        self.word_separators = r"[，。！？；：“”‘’（）【】、\s]+"
    
    def parse_srt_to_word_level(self, srt_path: Path) -> List[Dict]:
        """
        将SRT字幕解析为字粒度的数据结构
        
        Args:
            srt_path: SRT文件路径
            
        Returns:
            字粒度字幕数据列表
        """
        if not srt_path.exists():
            logger.error(f"SRT文件不存在: {srt_path}")
            return []
        
        try:
            subs = pysrt.open(str(srt_path), encoding='utf-8')
            word_level_data = []
            
            for sub in subs:
                segment_data = self._process_subtitle_segment(sub)
                word_level_data.append(segment_data)
            
            logger.info(f"成功解析SRT文件，共 {len(word_level_data)} 个字幕段")
            return word_level_data
            
        except Exception as e:
            logger.error(f"解析SRT文件失败: {e}")
            return []
    
    def _process_subtitle_segment(self, sub: SubRipItem) -> Dict:
        """
        处理单个字幕段，将其分解为字粒度数据
        
        Args:
            sub: pysrt字幕项
            
        Returns:
            字粒度字幕数据
        """
        # 转换时间格式
        start_seconds = self._srt_time_to_seconds(sub.start)
        end_seconds = self._srt_time_to_seconds(sub.end)
        
        # 分解文本为单词
        words = self._split_text_to_words(sub.text, start_seconds, end_seconds)
        
        return {
            'id': str(uuid.uuid4()),
            'startTime': start_seconds,
            'endTime': end_seconds,
            'text': sub.text.strip(),
            'words': words,
            'index': sub.index
        }
    
    def _split_text_to_words(self, text: str, start_time: float, end_time: float) -> List[Dict]:
        """
        将文本分解为单词，并分配时间戳
        
        Args:
            text: 字幕文本
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            单词列表，每个单词包含时间戳
        """
        # 清理文本
        clean_text = text.strip()
        if not clean_text:
            return []
        
        # 按标点符号和空格分割
        word_parts = re.split(self.word_separators, clean_text)
        word_parts = [part.strip() for part in word_parts if part.strip()]
        
        if not word_parts:
            return []
        
        # 计算每个单词的时间分配
        total_duration = end_time - start_time
        words_count = len(word_parts)
        
        # 简单的时间分配策略：平均分配
        word_duration = total_duration / words_count
        
        words = []
        for i, word_text in enumerate(word_parts):
            word_start = start_time + (i * word_duration)
            word_end = word_start + word_duration
            
            words.append({
                'id': str(uuid.uuid4()),
                'text': word_text,
                'startTime': word_start,
                'endTime': word_end
            })
        
        return words
    
    def _srt_time_to_seconds(self, srt_time: SubRipTime) -> float:
        """
        将SRT时间格式转换为秒数
        
        Args:
            srt_time: pysrt时间对象
            
        Returns:
            秒数
        """
        return srt_time.hours * 3600 + srt_time.minutes * 60 + srt_time.seconds + srt_time.milliseconds / 1000
    
    def _seconds_to_srt_time_object(self, time_str: str) -> SubRipTime:
        """
        将时间字符串转换为pysrt时间对象
        
        Args:
            time_str: 时间字符串 (如 "00:01:25,140")
            
        Returns:
            pysrt时间对象
        """
        # 处理逗号和点的格式
        time_str = time_str.replace(',', '.')
        
        # 解析时间
        time_parts = time_str.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        
        # 处理秒和毫秒
        seconds_part = time_parts[2]
        if '.' in seconds_part:
            seconds, milliseconds = seconds_part.split('.')
            seconds = int(seconds)
            milliseconds = int(milliseconds.ljust(3, '0')[:3])  # 确保3位毫秒
        else:
            seconds = int(seconds_part)
            milliseconds = 0
        
        return SubRipTime(hours, minutes, seconds, milliseconds)
    
    def create_edit_operations(self, deleted_segments: List[str], 
                             original_data: List[Dict]) -> List[Dict]:
        """
        根据删除的字幕段创建编辑操作
        
        Args:
            deleted_segments: 要删除的字幕段ID列表
            original_data: 原始字幕数据
            
        Returns:
            编辑操作列表
        """
        operations = []
        
        for segment_id in deleted_segments:
            segment = next((s for s in original_data if s['id'] == segment_id), None)
            if segment:
                operation = {
                    'type': 'delete',
                    'segmentIds': [segment_id],
                    'timestamp': segment['startTime'],
                    'metadata': {
                        'originalText': segment['text'],
                        'timeRange': {
                            'start': segment['startTime'],
                            'end': segment['endTime']
                        }
                    }
                }
                operations.append(operation)
        
        return operations
    
    def generate_edited_video_timeline(self, original_data: List[Dict], 
                                     deleted_segments: List[str]) -> List[Tuple[float, float]]:
        """
        生成编辑后的视频时间轴
        
        Args:
            original_data: 原始字幕数据
            deleted_segments: 要删除的字幕段ID列表
            
        Returns:
            保留片段的时间范围列表 [(start, end), ...]
        """
        deleted_ids = set(deleted_segments)
        timeline = []
        
        for segment in original_data:
            if segment['id'] not in deleted_ids:
                timeline.append((segment['startTime'], segment['endTime']))
        
        # 合并相邻的时间段
        if timeline:
            merged_timeline = [timeline[0]]
            for current_start, current_end in timeline[1:]:
                last_start, last_end = merged_timeline[-1]
                
                # 如果当前段与上一段相邻或重叠，则合并
                if current_start <= last_end + 0.1:  # 允许0.1秒的间隔
                    merged_timeline[-1] = (last_start, max(last_end, current_end))
                else:
                    merged_timeline.append((current_start, current_end))
            
            return merged_timeline
        
        return []
    
    def export_edited_srt(self, original_data: List[Dict], 
                         deleted_segments: List[str], 
                         output_path: Path) -> bool:
        """
        导出编辑后的SRT文件
        
        Args:
            original_data: 原始字幕数据
            deleted_segments: 要删除的字幕段ID列表
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        try:
            deleted_ids = set(deleted_segments)
            edited_segments = []
            
            for segment in original_data:
                if segment['id'] not in deleted_ids:
                    edited_segments.append(segment)
            
            # 重新编号
            for i, segment in enumerate(edited_segments, 1):
                segment['index'] = i
            
            # 写入SRT文件
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in edited_segments:
                    start_time = self._seconds_to_srt_time(segment['startTime'])
                    end_time = self._seconds_to_srt_time(segment['endTime'])
                    
                    f.write(f"{segment['index']}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text']}\n\n")
            
            logger.info(f"编辑后的SRT文件已保存: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出编辑后的SRT文件失败: {e}")
            return False
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        将秒数转换为SRT时间格式
        
        Args:
            seconds: 秒数
            
        Returns:
            SRT时间格式字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def get_subtitle_statistics(self, data: List[Dict]) -> Dict:
        """
        获取字幕统计信息
        
        Args:
            data: 字幕数据
            
        Returns:
            统计信息
        """
        if not data:
            return {
                'totalDuration': 0,
                'wordCount': 0,
                'segmentCount': 0,
                'averageWordsPerSegment': 0
            }
        
        total_duration = max(seg['endTime'] for seg in data) - min(seg['startTime'] for seg in data)
        word_count = sum(len(seg['words']) for seg in data)
        segment_count = len(data)
        
        return {
            'totalDuration': total_duration,
            'wordCount': word_count,
            'segmentCount': segment_count,
            'averageWordsPerSegment': word_count / segment_count if segment_count > 0 else 0
        }

    # ============================================================
    # 烧录字幕：片段字幕提取与 ASS 生成
    # ============================================================

    def extract_clip_subtitles(self, srt_path: Path, start: float, end: float) -> List[Dict]:
        """从项目 SRT 中提取某片段时间窗内的整句字幕，并归零对齐。

        Args:
            srt_path: 项目 SRT 文件路径
            start: 片段起点（秒，相对原视频）
            end: 片段终点（秒，相对原视频）

        Returns:
            片段级整句字幕列表，每项 {startTime, endTime, text}，
            时间已减去片段起点并裁剪到 [0, duration]。无重叠时返回空列表。
        """
        srt_path = Path(srt_path)
        if not srt_path.exists():
            logger.error(f"SRT文件不存在: {srt_path}")
            return []

        if end <= start:
            logger.warning(f"无效的片段时间窗: start={start}, end={end}")
            return []

        duration = end - start

        try:
            subs = pysrt.open(str(srt_path), encoding='utf-8')
        except Exception as e:
            logger.error(f"解析SRT文件失败: {e}")
            return []

        clip_segments: List[Dict] = []
        for sub in subs:
            seg_start = self._srt_time_to_seconds(sub.start)
            seg_end = self._srt_time_to_seconds(sub.end)

            # 过滤：与片段时间窗有重叠的字幕段
            if seg_end <= start or seg_start >= end:
                continue

            text = (sub.text or "").strip()
            if not text:
                continue

            # 归零对齐 + 裁剪首尾溢出到 [0, duration]
            aligned_start = max(0.0, seg_start - start)
            aligned_end = min(duration, seg_end - start)

            # 裁剪后若无有效时长则跳过
            if aligned_end <= aligned_start:
                continue

            clip_segments.append({
                'startTime': aligned_start,
                'endTime': aligned_end,
                'text': text,
            })

        logger.info(
            f"提取片段字幕: 窗口[{start:.2f},{end:.2f}] 命中 {len(clip_segments)} 段"
        )
        return clip_segments

    def to_ass(self, segments: List[Dict], template: Dict,
               canvas_w: int, canvas_h: int) -> str:
        """将片段整句字幕转换为 ASS 字幕字符串。

        Args:
            segments: extract_clip_subtitles 产出的整句字幕列表
            template: shared_config.SUBTITLE_TEMPLATES 中的某个模板字典
            canvas_w: 目标画布宽度（像素）
            canvas_h: 目标画布高度（像素）

        Returns:
            完整 ASS 文件文本（可写入 .ass 供 libass 渲染）。
        """
        from ..core import shared_config

        safe_margin_pct = getattr(shared_config, "SUBTITLE_SAFE_MARGIN_PCT", 0.12)
        max_lines = getattr(shared_config, "SUBTITLE_MAX_LINES", 2)
        max_chars = getattr(shared_config, "SUBTITLE_MAX_CHARS_PER_LINE", 18)

        # 字号按画布高度比例计算，保证横竖屏一致观感
        base_font_pct = template.get("base_font_pct", 0.05)
        font_size = max(12, int(round(canvas_h * base_font_pct)))

        # 底部安全区边距（像素），用于 ASS MarginV
        margin_v = int(round(canvas_h * safe_margin_pct))
        margin_h = int(round(canvas_w * 0.05))

        font_name = template.get("font_name", "Noto Sans CJK SC")
        primary = template.get("primary_colour", "&H00FFFFFF")
        outline_c = template.get("outline_colour", "&H00000000")
        back_c = template.get("back_colour", "&H00000000")
        bold = int(template.get("bold", 1))
        border_style = int(template.get("border_style", 1))
        outline = template.get("outline", 2)
        shadow = template.get("shadow", 0)
        alignment = int(template.get("alignment", 2))

        # ASS 头部 + 样式定义
        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {canvas_w}\n"
            f"PlayResY: {canvas_h}\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{font_name},{font_size},{primary},&H000000FF,"
            f"{outline_c},{back_c},{bold},0,0,0,100,100,0,0,"
            f"{border_style},{outline},{shadow},{alignment},"
            f"{margin_h},{margin_h},{margin_v},1\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text\n"
        )

        lines = [header]
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            wrapped = self._wrap_text(text, max_chars, max_lines)
            ass_text = wrapped.replace("\n", "\\N")
            start_ts = self._seconds_to_ass_time(seg["startTime"])
            end_ts = self._seconds_to_ass_time(seg["endTime"])
            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{ass_text}\n"
            )

        return "".join(lines)

    def _wrap_text(self, text: str, max_chars: int, max_lines: int) -> str:
        """按最大字符数对整句限宽换行，最多 max_lines 行，溢出截断加省略号。"""
        if max_chars <= 0:
            return text
        words = text.split()
        # 中文整句通常无空格：退化为按字符切分
        if len(words) <= 1:
            chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        else:
            chunks = []
            current = ""
            for w in words:
                candidate = (current + " " + w).strip()
                if len(candidate) > max_chars and current:
                    chunks.append(current)
                    current = w
                else:
                    current = candidate
            if current:
                chunks.append(current)

        if len(chunks) > max_lines:
            chunks = chunks[:max_lines]
            # 末行溢出标记
            last = chunks[-1]
            if len(last) >= max_chars:
                last = last[: max(0, max_chars - 1)] + "…"
            else:
                last = last + "…"
            chunks[-1] = last

        return "\n".join(chunks)

    def _seconds_to_ass_time(self, seconds: float) -> str:
        """将秒数转换为 ASS 时间格式 H:MM:SS.cc（厘秒）。"""
        if seconds < 0:
            seconds = 0.0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int(round((seconds - int(seconds)) * 100))
        if centiseconds >= 100:
            centiseconds = 99
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
