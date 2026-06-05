"""
视频处理工具
"""
import subprocess
import json
import logging
import re
from typing import List, Dict, Optional
from pathlib import Path
from .ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path

# 修复导入问题
try:
    from ..core.shared_config import CLIPS_DIR, COLLECTIONS_DIR
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    backend_path = Path(__file__).parent.parent
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    from ..core.shared_config import CLIPS_DIR, COLLECTIONS_DIR

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理工具类"""
    
    def __init__(self, clips_dir: Optional[str] = None, collections_dir: Optional[str] = None):
        # 强制使用传入的项目特定路径，不使用全局路径作为后备
        if not clips_dir:
            raise ValueError("clips_dir 参数是必需的，不能使用全局路径")
        if not collections_dir:
            raise ValueError("collections_dir 参数是必需的，不能使用全局路径")
        
        self.clips_dir = Path(clips_dir)
        self.collections_dir = Path(collections_dir)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        清理文件名，移除或替换不合法的字符
        
        Args:
            filename: 原始文件名
            
        Returns:
            清理后的文件名
        """
        # 移除或替换不合法的字符
        # Windows和Unix系统都不允许的字符: < > : " | ? * \ /
        # 替换为下划线
        sanitized = re.sub(r'[<>:"|?*\\/]', '_', filename)
        
        # 移除前后空格和点
        sanitized = sanitized.strip(' .')
        
        # 限制长度，避免文件名过长
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        # 确保文件名不为空
        if not sanitized:
            sanitized = "untitled"
            
        return sanitized
    
    @staticmethod
    def convert_srt_time_to_ffmpeg_time(srt_time: str) -> str:
        """
        将SRT时间格式转换为FFmpeg时间格式
        
        Args:
            srt_time: SRT时间格式 (如 "00:00:06,140" 或 "00:00:06.140")
            
        Returns:
            FFmpeg时间格式 (如 "00:00:06.140")
        """
        # 将逗号替换为点
        return srt_time.replace(',', '.')
    
    @staticmethod
    def convert_seconds_to_ffmpeg_time(seconds: float) -> str:
        """
        将秒数转换为FFmpeg时间格式
        
        Args:
            seconds: 秒数
            
        Returns:
            FFmpeg时间格式 (如 "00:00:06.140")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    @staticmethod
    def convert_ffmpeg_time_to_seconds(time_str: str) -> float:
        """
        将FFmpeg时间格式转换为秒数
        
        Args:
            time_str: FFmpeg时间格式 (如 "00:00:06.140")
            
        Returns:
            秒数
        """
        try:
            # 处理毫秒部分
            if '.' in time_str:
                time_part, ms_part = time_str.split('.')
                milliseconds = int(ms_part)
            else:
                time_part = time_str
                milliseconds = 0
            
            # 解析时分秒
            h, m, s = map(int, time_part.split(':'))
            
            return h * 3600 + m * 60 + s + milliseconds / 1000
        except Exception as e:
            logger.error(f"时间格式转换失败: {time_str}, 错误: {e}")
            return 0.0
    
    @staticmethod
    def extract_clip(input_video: Path, output_path: Path, 
                    start_time: str, end_time: str) -> bool:
        """
        从视频中提取指定时间段的片段
        
        Args:
            input_video: 输入视频路径
            output_path: 输出视频路径
            start_time: 开始时间 (格式: "00:01:25,140")
            end_time: 结束时间 (格式: "00:02:53,500")
            
        Returns:
            是否成功
        """
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换时间格式：从SRT格式转换为FFmpeg格式
            ffmpeg_start_time = VideoProcessor.convert_srt_time_to_ffmpeg_time(start_time)
            ffmpeg_end_time = VideoProcessor.convert_srt_time_to_ffmpeg_time(end_time)
            
            # 计算持续时间
            start_seconds = VideoProcessor.convert_ffmpeg_time_to_seconds(ffmpeg_start_time)
            end_seconds = VideoProcessor.convert_ffmpeg_time_to_seconds(ffmpeg_end_time)
            duration = end_seconds - start_seconds
            
            # 构建优化的FFmpeg命令
            # 使用 -ss 在输入前进行精确定位，使用 -t 指定持续时间
            ffmpeg_bin = get_ffmpeg_path()
            cmd = [
                ffmpeg_bin,
                '-ss', ffmpeg_start_time,  # 在输入前定位，更精确
                '-i', str(input_video),
                '-t', str(duration),  # 使用持续时间而不是绝对结束时间
                '-c:v', 'copy',  # 复制视频流
                '-c:a', 'copy',  # 复制音频流
                '-avoid_negative_ts', 'make_zero',
                '-y',  # 覆盖输出文件
                str(output_path)
            ]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                logger.info(f"成功提取视频片段: {output_path} ({ffmpeg_start_time} -> {ffmpeg_end_time}, 时长: {duration:.2f}秒)")
                return True
            else:
                logger.error(f"提取视频片段失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"视频处理异常: {str(e)}")
            return False
    
    @staticmethod
    def create_collection(clips_list: List[Path], output_path: Path) -> bool:
        """
        将多个视频片段拼接成合集
        
        Args:
            clips_list: 视频片段路径列表
            output_path: 输出合集路径
            
        Returns:
            是否成功
        """
        try:
            # 验证输入参数
            if not clips_list:
                logger.error("clips_list为空，无法创建合集")
                return False
            
            # 验证所有视频文件是否存在
            valid_clips = []
            for clip_path in clips_list:
                if not clip_path.exists():
                    logger.warning(f"视频文件不存在，跳过: {clip_path}")
                    continue
                valid_clips.append(clip_path)
            
            if not valid_clips:
                logger.error("没有有效的视频文件，无法创建合集")
                return False
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建concat文件
            concat_file = output_path.parent / "concat_list.txt"
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for clip_path in valid_clips:
                    # 使用绝对路径并转义单引号
                    abs_path = clip_path.absolute()
                    escaped_path = str(abs_path).replace("'", "'\"'\"'")
                    f.write(f"file '{escaped_path}'\n")
            
            # 验证concat文件内容
            if concat_file.stat().st_size == 0:
                logger.error("concat文件为空，无法创建合集")
                concat_file.unlink(missing_ok=True)
                return False
            
            # 构建FFmpeg命令 - 使用H.264编码确保兼容性
            ffmpeg_bin = get_ffmpeg_path()
            cmd = [
                ffmpeg_bin,
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',  # 使用H.264视频编码
                '-preset', 'ultrafast',  # 使用最快的编码预设
                '-crf', '28',  # 稍微降低质量以加快编码速度
                '-c:a', 'aac',  # 使用AAC音频编码
                '-b:a', '128k',  # 音频比特率
                '-movflags', '+faststart',  # 优化网络播放
                '-y',
                str(output_path)
            ]
            
            logger.info(f"执行FFmpeg命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            # 清理临时文件
            concat_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                logger.info(f"成功创建合集: {output_path}")
                return True
            else:
                logger.error(f"创建合集失败: {result.stderr}")
                logger.error(f"FFmpeg stdout: {result.stdout}")
                return False
                
        except Exception as e:
            logger.error(f"视频拼接异常: {str(e)}")
            return False
    
    @staticmethod
    def get_video_dimensions(video_path: Path) -> Dict:
        """
        读取视频宽、高、时长、帧率。

        Returns:
            {'width': int, 'height': int, 'duration': float, 'fps': float}
            失败返回空字典。
        """
        try:
            ffprobe_bin = get_ffprobe_path()
            cmd = [
                ffprobe_bin,
                '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,avg_frame_rate:format=duration',
                '-print_format', 'json',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                logger.error(f"读取视频尺寸失败: {result.stderr}")
                return {}
            info = json.loads(result.stdout)
            stream = (info.get('streams') or [{}])[0]
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))
            # 解析帧率 "30000/1001"
            fps = 0.0
            rate = stream.get('avg_frame_rate', '0/1')
            try:
                num, den = rate.split('/')
                den_f = float(den)
                fps = float(num) / den_f if den_f else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            duration = float(info.get('format', {}).get('duration', 0.0) or 0.0)
            return {'width': width, 'height': height, 'duration': duration, 'fps': fps}
        except Exception as e:
            logger.error(f"读取视频尺寸异常: {str(e)}")
            return {}

    @staticmethod
    def _target_dimensions(aspect_ratio: str) -> tuple:
        """根据比例返回 (target_w, target_h)。"""
        from ..core.shared_config import VERTICAL_WIDTH, VERTICAL_HEIGHT, SQUARE_SIZE
        if aspect_ratio == "1:1":
            return SQUARE_SIZE, SQUARE_SIZE
        return VERTICAL_WIDTH, VERTICAL_HEIGHT

    @staticmethod
    def _build_track_crop_filter(segments, src_w: int, src_h: int,
                                 target_w: int, target_h: int) -> str:
        """
        由分段裁剪窗口构造跟踪裁剪滤镜：
        crop 宽度固定（=src_h*target_w/target_h，clamp 到 src_w），
        x 用按时间分段的 if/between 表达式跟随主体中心，再 scale 到目标尺寸。
        """
        from .auto_reframe import compute_crop_x
        crop_w = int(round(src_h * target_w / target_h))
        crop_w = min(crop_w, src_w)

        if not segments:
            x_expr = str(int(round((src_w - crop_w) / 2)))
        else:
            # 构造嵌套 if(between(t,s,e), x, ...) 表达式
            expr = str(compute_crop_x(segments[-1].x_ratio, src_w, src_h, target_w, target_h))
            for seg in reversed(segments[:-1]):
                x_px = compute_crop_x(seg.x_ratio, src_w, src_h, target_w, target_h)
                expr = f"if(between(t,{seg.start:.3f},{seg.end:.3f}),{x_px},{expr})"
            x_expr = expr

        return (
            f"crop={crop_w}:{src_h}:x='{x_expr}':y=0,"
            f"scale={target_w}:{target_h}"
        )

    @staticmethod
    def _build_blur_padding_filter(target_w: int, target_h: int) -> str:
        """
        模糊背景兜底滤镜：放大模糊原画铺底 + 居中叠加完整缩放原画。
        """
        from ..core.shared_config import REFRAME_BLUR_SIGMA
        return (
            f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h},boxblur={REFRAME_BLUR_SIGMA}[bg];"
            f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )

    @staticmethod
    def _build_center_crop_filter(src_w: int, src_h: int,
                                  target_w: int, target_h: int) -> str:
        """居中裁剪兜底滤镜。"""
        crop_w = int(round(src_h * target_w / target_h))
        crop_w = min(crop_w, src_w)
        x = int(round((src_w - crop_w) / 2))
        return f"crop={crop_w}:{src_h}:x={x}:y=0,scale={target_w}:{target_h}"

    @staticmethod
    def reframe_to_vertical(input_video: Path, output_path: Path,
                            aspect_ratio: str = "9:16",
                            focus_strategy: str = "track",
                            fallback_strategy: str = "center",
                            focus_track: Optional[List] = None) -> Dict:
        """
        将横屏片段重构为竖屏（或 1:1）成品。

        Args:
            input_video: 源视频路径
            output_path: 输出路径
            aspect_ratio: "9:16" | "1:1"
            focus_strategy: "track"（人脸跟踪）| "center"（居中）| "blur_padding"
            fallback_strategy: track 失败时降级策略 "center" | "blur_padding"
            focus_track: 预计算的 FocusPoint 列表（None 则内部计算）

        Returns:
            {'success': bool, 'fallback_used': bool, 'width': int, 'height': int,
             'focus_strategy': str, 'error': Optional[str]}
        """
        from .auto_reframe import (
            compute_focus_track, detect_success, smooth_centers, build_segments,
        )
        from ..core.shared_config import REFRAME_VIDEO_CRF, REFRAME_VIDEO_PRESET

        result = {
            'success': False, 'fallback_used': False,
            'width': 0, 'height': 0,
            'focus_strategy': focus_strategy, 'error': None,
        }
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dims = VideoProcessor.get_video_dimensions(input_video)
            if not dims or dims.get('width', 0) <= 0 or dims.get('height', 0) <= 0:
                result['error'] = "无法读取源视频尺寸"
                logger.error(f"竖屏重构失败：{result['error']} ({input_video})")
                return result

            src_w, src_h = dims['width'], dims['height']
            duration = dims.get('duration', 0.0)
            target_w, target_h = VideoProcessor._target_dimensions(aspect_ratio)
            result['width'], result['height'] = target_w, target_h

            # 源已是竖屏（高 >= 宽）：仅 scale + pad，保比铺满
            if src_h >= src_w:
                vf = (f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                      f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2")
                filter_args = ['-vf', vf]
                result['focus_strategy'] = 'center'
            else:
                use_track = (focus_strategy == "track")
                segments = None
                if use_track:
                    track = focus_track if focus_track is not None else \
                        compute_focus_track(input_video, duration)
                    if detect_success(track):
                        smoothed = smooth_centers([p.x_ratio for p in track])
                        segments = build_segments(track, smoothed, duration)
                    else:
                        result['fallback_used'] = True
                        result['focus_strategy'] = fallback_strategy
                        use_track = False

                if use_track and segments is not None:
                    vf = VideoProcessor._build_track_crop_filter(
                        segments, src_w, src_h, target_w, target_h)
                    filter_args = ['-vf', vf]
                elif (result['fallback_used'] and fallback_strategy == "blur_padding") \
                        or focus_strategy == "blur_padding":
                    fc = VideoProcessor._build_blur_padding_filter(target_w, target_h)
                    filter_args = ['-filter_complex', fc]
                else:
                    # 居中裁剪（显式 center 或 center 降级）
                    vf = VideoProcessor._build_center_crop_filter(
                        src_w, src_h, target_w, target_h)
                    filter_args = ['-vf', vf]

            ffmpeg_bin = get_ffmpeg_path()
            cmd = [ffmpeg_bin, '-i', str(input_video)] + filter_args + [
                '-c:v', 'libx264',
                '-preset', REFRAME_VIDEO_PRESET,
                '-crf', str(REFRAME_VIDEO_CRF),
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                '-y', str(output_path)
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding='utf-8', errors='ignore')
            if proc.returncode == 0 and output_path.exists():
                result['success'] = True
                logger.info(
                    f"竖屏重构成功: {output_path} ({target_w}x{target_h}, "
                    f"strategy={result['focus_strategy']}, fallback={result['fallback_used']})")
            else:
                result['error'] = proc.stderr
                logger.error(
                    f"竖屏重构 ffmpeg 失败 clip={input_video.name}: {proc.stderr}")
            return result
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"竖屏重构异常 clip={input_video}: {e}")
            return result

    @staticmethod
    def extract_thumbnail(video_path: Path, output_path: Path, time_offset: int = 5) -> bool:
        """
        从视频中提取缩略图
        
        Args:
            video_path: 视频文件路径
            output_path: 输出缩略图路径
            time_offset: 提取时间点（秒）
            
        Returns:
            是否成功
        """
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 构建FFmpeg命令
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(time_offset),
                '-vframes', '1',
                '-q:v', '2',  # 高质量
                '-y',  # 覆盖输出文件
                str(output_path)
            ]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"成功提取缩略图: {output_path}")
                return True
            else:
                logger.error(f"提取缩略图失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"提取缩略图异常: {str(e)}")
            return False
    
    @staticmethod
    def get_video_info(video_path: Path) -> Dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        try:
            ffprobe_bin = get_ffprobe_path()
            cmd = [
                ffprobe_bin,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return {
                    'duration': float(info['format']['duration']),
                    'size': int(info['format']['size']),
                    'bitrate': int(info['format']['bit_rate']),
                    'streams': info['streams']
                }
            else:
                logger.error(f"获取视频信息失败: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"获取视频信息异常: {str(e)}")
            return {}
    
    def batch_extract_clips(self, input_video: Path, clips_data: List[Dict]) -> List[Path]:
        """
        批量提取视频片段
        
        Args:
            input_video: 输入视频路径
            clips_data: 片段数据列表，每个元素包含id、title、start_time、end_time
            
        Returns:
            成功提取的片段路径列表
        """
        successful_clips = []
        
        for clip_data in clips_data:
            clip_id = clip_data['id']
            title = clip_data.get('title', f"片段_{clip_id}")
            start_time = clip_data['start_time']
            end_time = clip_data['end_time']
            
            # 处理时间格式 - 如果是秒数，转换为SRT格式
            if isinstance(start_time, (int, float)):
                start_time = VideoProcessor.convert_seconds_to_ffmpeg_time(start_time)
            if isinstance(end_time, (int, float)):
                end_time = VideoProcessor.convert_seconds_to_ffmpeg_time(end_time)
            
            # 使用标题作为文件名，并清理不合法的字符
            # 在文件名中包含clip_id，便于后续合集拼接时查找
            safe_title = VideoProcessor.sanitize_filename(title)
            output_path = self.clips_dir / f"{clip_id}_{safe_title}.mp4"
            
            logger.info(f"提取切片 {clip_id}: {start_time} -> {end_time}, 输出: {output_path}")
            
            if VideoProcessor.extract_clip(input_video, output_path, start_time, end_time):
                successful_clips.append(output_path)
                logger.info(f"切片 {clip_id} 提取成功")
            else:
                logger.error(f"切片 {clip_id} 提取失败")
        
        return successful_clips
    
    def create_collections_from_metadata(self, collections_data: List[Dict]) -> List[Dict]:
        """
        根据元数据创建合集
        
        Args:
            collections_data: 合集数据列表
            
        Returns:
            成功创建的合集信息列表，包含视频路径和缩略图路径
        """
        successful_collections = []
        
        for collection_data in collections_data:
            collection_id = collection_data['id']
            collection_title = collection_data.get('collection_title', f'合集_{collection_id}')
            clip_ids = collection_data['clip_ids']
            
            # 构建片段路径列表
            clips_list = []
            for clip_id in clip_ids:
                # 查找对应的切片文件
                # 新的文件名格式是: {clip_id}_{title}.mp4
                clip_path = self.clips_dir / f"{clip_id}_*.mp4"
                found_clips = list(self.clips_dir.glob(f"{clip_id}_*.mp4"))
                
                if found_clips:
                    found_clip = found_clips[0]  # 取第一个匹配的文件
                    clips_list.append(found_clip)
                    logger.info(f"找到合集 {collection_id} 的切片: {found_clip.name}")
                else:
                    logger.warning(f"未找到合集 {collection_id} 的切片 {clip_id}")
            
            if clips_list:
                # 使用collection_title作为文件名，并清理不合法的字符
                safe_title = VideoProcessor.sanitize_filename(collection_title)
                output_path = self.collections_dir / f"{safe_title}.mp4"
                
                if VideoProcessor.create_collection(clips_list, output_path):
                    # 生成合集缩略图
                    thumbnail_path = None
                    try:
                        thumbnail_filename = f"{collection_id}_{safe_title}_thumbnail.jpg"
                        thumbnail_path = self.collections_dir / thumbnail_filename
                        
                        # 从视频中提取缩略图（第2秒的帧）
                        thumbnail_success = VideoProcessor.extract_thumbnail(output_path, thumbnail_path, time_offset=2)
                        if thumbnail_success:
                            logger.info(f"合集 {collection_id} 缩略图生成成功: {thumbnail_path}")
                        else:
                            logger.warning(f"合集 {collection_id} 缩略图生成失败")
                            thumbnail_path = None
                    except Exception as e:
                        logger.error(f"生成合集 {collection_id} 缩略图时出错: {e}")
                        thumbnail_path = None
                    
                    # 返回包含视频路径和缩略图路径的信息
                    collection_info = {
                        'collection_id': collection_id,
                        'video_path': str(output_path),
                        'thumbnail_path': str(thumbnail_path) if thumbnail_path else None,
                        'title': collection_title
                    }
                    successful_collections.append(collection_info)
                    logger.info(f"成功创建合集 {collection_id}: {output_path}")
            else:
                logger.warning(f"合集 {collection_id} 没有找到任何有效的切片文件")
        
        return successful_collections