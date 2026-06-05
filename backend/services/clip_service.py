"""
切片服务
提供切片相关的业务逻辑操作
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..services.base import BaseService
from ..repositories.clip_repository import ClipRepository
from ..models.clip import Clip
from ..schemas.clip import ClipCreate, ClipUpdate, ClipResponse, ClipListResponse, ClipFilter
from ..schemas.base import PaginationParams, PaginationResponse


class ClipService(BaseService[Clip, ClipCreate, ClipUpdate, ClipResponse]):
    """Clip service with business logic."""
    
    def __init__(self, db: Session):
        repository = ClipRepository(db)
        super().__init__(repository)
        self.db = db
    
    def create_clip(self, clip_data: ClipCreate) -> Clip:
        """Create a new clip with business logic."""
        data = clip_data.model_dump()
        orm_data = {
            "project_id": data["project_id"],
            "title": data["title"],
            "description": data["description"],
            "start_time": int(data["start_time"]) if data["start_time"] is not None else 0,
            "end_time": int(data["end_time"]) if data["end_time"] is not None else 0,
            "duration": int(data["duration"]) if data["duration"] is not None else 0,
            "score": data.get("score"),
            "clip_metadata": data.get("clip_metadata", {}),
            "tags": data.get("tags", [])
        }
        return self.create(**orm_data)
    
    def update_clip(self, clip_id: str, clip_data: ClipUpdate) -> Optional[Clip]:
        """Update a clip with business logic."""
        update_data = {k: v for k, v in clip_data.model_dump().items() if v is not None}
        if not update_data:
            return self.get(clip_id)
        
        return self.update(clip_id, **update_data)
    
    def get_clips_by_project(self, project_id: str, skip: int = 0, limit: int = 100) -> List[Clip]:
        """Get clips by project ID."""
        return self.repository.find_by(project_id=project_id)

    def burn_subtitle_clip(
        self,
        clip_id: str,
        template_id: str,
        aspect_ratio: Optional[str] = None,
    ) -> Dict[str, Any]:
        """为单个片段烧录整句字幕。

        Args:
            clip_id: 片段ID
            template_id: 字幕样式模板ID（见 shared_config.SUBTITLE_TEMPLATES）
            aspect_ratio: 目标画面比例（如 "9:16"/"16:9"），缺省按源视频比例

        Returns:
            结果字典，含 status、video_path、thumbnail_path、subtitle_variant 等。

        Raises:
            ValueError: 模板无效、片段不存在或缺少 SRT。
        """
        import logging
        from pathlib import Path
        from datetime import datetime, timezone

        from ..core import shared_config
        from ..core.path_utils import get_projects_directory
        from ..utils.subtitle_processor import SubtitleProcessor
        from ..utils.video_processor import VideoProcessor

        logger = logging.getLogger(__name__)

        templates = getattr(shared_config, "SUBTITLE_TEMPLATES", {})
        if template_id not in templates:
            raise ValueError(f"无效的字幕模板: {template_id}")
        template = templates[template_id]

        clip = self.get(clip_id)
        if not clip:
            raise ValueError("片段不存在")

        project_id = str(clip.project_id)

        # 解析片段源视频
        source_video = getattr(clip, "video_path", None)
        if not source_video or not Path(source_video).exists():
            raise ValueError("片段源视频不存在，无法烧录字幕")
        source_video = Path(source_video)

        # 解析项目 SRT
        project_dir = get_projects_directory() / project_id
        srt_path = project_dir / "raw" / "input.srt"
        if not srt_path.exists():
            raise ValueError("项目字幕（SRT）不存在，请先生成字幕")

        # 归一化 aspect_ratio 标签（用于缓存键与文件名）
        ar_label = (aspect_ratio or "source").replace(":", "x")

        # 缓存检查：相同 template_id + aspect_ratio 且文件存在则复用
        existing_meta = dict(getattr(clip, "clip_metadata", {}) or {})
        variants = list(existing_meta.get("subtitle_variants", []) or [])
        for v in variants:
            if (
                v.get("template_id") == template_id
                and v.get("aspect_ratio") == (aspect_ratio or "source")
                and v.get("video_path")
                and Path(v["video_path"]).exists()
            ):
                logger.info(f"复用已有字幕烧录产物: clip={clip_id}, template={template_id}")
                return {
                    "status": "completed",
                    "cached": True,
                    "clip_id": clip_id,
                    "template_id": template_id,
                    "video_path": v["video_path"],
                    "thumbnail_path": v.get("thumbnail_path"),
                    "subtitle_variant": v,
                }

        # 片段时间窗（秒）
        start = float(getattr(clip, "start_time", 0) or 0)
        end = float(getattr(clip, "end_time", 0) or 0)
        duration = end - start

        processor = SubtitleProcessor()
        segments = processor.extract_clip_subtitles(srt_path, start, end)

        if not segments:
            # 无重叠字幕：优雅跳过，不报错（等价于不烧录）
            logger.info(f"片段无重叠字幕，跳过烧录: clip={clip_id}")
            return {
                "status": "empty",
                "cached": False,
                "clip_id": clip_id,
                "template_id": template_id,
                "message": "该片段时间窗内无字幕，已跳过烧录",
                "video_path": None,
                "thumbnail_path": None,
                "subtitle_variant": None,
            }

        # 解析画布分辨率
        resolution = VideoProcessor.get_video_resolution(source_video)
        if resolution:
            src_w, src_h = resolution
        else:
            src_w, src_h = (1920, 1080)

        canvas_w, canvas_h = self._resolve_canvas(src_w, src_h, aspect_ratio)

        # 生成 ASS 到临时文件
        ass_text = processor.to_ass(segments, template, canvas_w, canvas_h)
        clips_dir = source_video.parent
        clips_dir.mkdir(parents=True, exist_ok=True)
        ass_path = clips_dir / f"{clip_id}_sub_{template_id}_{ar_label}.ass"
        try:
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_text)
        except Exception as e:
            raise ValueError(f"写入 ASS 字幕文件失败: {e}")

        # 输出路径（不覆盖原片段）
        stem = source_video.stem
        output_path = clips_dir / f"{stem}_sub_{template_id}_{ar_label}.mp4"

        fonts_dir = getattr(shared_config, "SUBTITLE_FONTS_DIR", None)
        ok = VideoProcessor.burn_subtitles(source_video, output_path, ass_path, fonts_dir)
        if not ok or not output_path.exists():
            logger.error(
                f"字幕烧录失败: clip_id={clip_id}, template_id={template_id}, ass={ass_path}"
            )
            raise RuntimeError(
                f"字幕烧录失败 (clip_id={clip_id}, template_id={template_id})，详见 ffmpeg 日志"
            )

        # 生成缩略图
        thumbnail_path = clips_dir / f"{stem}_sub_{template_id}_{ar_label}_thumb.jpg"
        thumb_ok = VideoProcessor.extract_thumbnail(output_path, thumbnail_path, time_offset=1)
        thumb_str = str(thumbnail_path) if thumb_ok and thumbnail_path.exists() else None

        variant = {
            "template_id": template_id,
            "aspect_ratio": aspect_ratio or "source",
            "video_path": str(output_path),
            "thumbnail_path": thumb_str,
            "width": canvas_w,
            "height": canvas_h,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # 去重写回（同 template_id + aspect_ratio 覆盖）
        deduped = [
            v for v in variants
            if not (
                v.get("template_id") == template_id
                and v.get("aspect_ratio") == (aspect_ratio or "source")
            )
        ]
        deduped.append(variant)
        existing_meta["subtitle_variants"] = deduped

        # 落盘优先已完成（视频文件已写入），再写库
        try:
            self.update(clip_id, clip_metadata=existing_meta)
        except Exception as e:
            logger.error(f"写回 subtitle_variants 失败: {e}")
            raise

        logger.info(f"字幕烧录完成: clip={clip_id}, template={template_id} -> {output_path}")
        return {
            "status": "completed",
            "cached": False,
            "clip_id": clip_id,
            "template_id": template_id,
            "video_path": str(output_path),
            "thumbnail_path": thumb_str,
            "subtitle_variant": variant,
        }

    @staticmethod
    def _resolve_canvas(src_w: int, src_h: int, aspect_ratio: Optional[str]) -> tuple:
        """根据目标比例解析画布尺寸；缺省沿用源尺寸。"""
        if not aspect_ratio:
            return (src_w, src_h)
        try:
            wr, hr = aspect_ratio.split(":")
            wr, hr = float(wr), float(hr)
        except Exception:
            return (src_w, src_h)
        if wr <= 0 or hr <= 0:
            return (src_w, src_h)
        # 标准竖屏/横屏画布
        if abs(wr / hr - 9 / 16) < 0.01:
            return (1080, 1920)
        if abs(wr / hr - 16 / 9) < 0.01:
            return (1920, 1080)
        if abs(wr / hr - 1.0) < 0.01:
            return (1080, 1080)
        # 其他比例：按源高度推算宽度
        new_w = int(round(src_h * (wr / hr)))
        return (new_w, src_h)
    
    def get_clips_paginated(
        self, 
        pagination: PaginationParams,
        filters: Optional[ClipFilter] = None
    ) -> ClipListResponse:
        """Get paginated clips with filtering."""
        filter_dict = {}
        if filters:
            filter_data = filters.model_dump()
            filter_dict = {k: v for k, v in filter_data.items() if v is not None}
        
        items, pagination_response = self.get_paginated(pagination, filter_dict)
        
        # Convert to response schemas (simplified)
        clip_responses = []
        for clip in items:
            status_obj = getattr(clip, 'status', None)
            status_value = status_obj.value if hasattr(status_obj, 'value') else 'pending'
            
            clip_responses.append(ClipResponse(
                id=str(clip.id),
                project_id=str(clip.project_id),
                title=str(clip.title),
                description=str(clip.description) if clip.description else None,
                start_time=getattr(clip, 'start_time', 0),
                end_time=getattr(clip, 'end_time', 0),
                duration=int(getattr(clip, 'duration', 0)),
                score=getattr(clip, 'score', None),
                status=status_value,
                video_path=getattr(clip, 'video_path', None),
                tags=getattr(clip, 'tags', []) or [],
                clip_metadata=getattr(clip, 'clip_metadata', {}) or {},
                created_at=getattr(clip, 'created_at', None) if isinstance(getattr(clip, 'created_at', None), (type(None), __import__('datetime').datetime)) else None,
                updated_at=getattr(clip, 'updated_at', None) if isinstance(getattr(clip, 'updated_at', None), (type(None), __import__('datetime').datetime)) else None,
                collection_ids=[]
            ))
        
        return ClipListResponse(
            items=clip_responses,
            pagination=pagination_response
        ) 