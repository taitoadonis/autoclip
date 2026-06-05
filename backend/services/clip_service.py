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

    def _resolve_clip_video_path(self, clip: Clip):
        """解析片段源视频文件路径，复用合集生成端点的查找逻辑。"""
        from pathlib import Path
        from ..core.path_utils import get_project_directory

        if clip.video_path and Path(clip.video_path).exists():
            return Path(clip.video_path)

        project_dir = get_project_directory(str(clip.project_id))
        clips_dir = project_dir / "output" / "clips"
        matches = list(clips_dir.glob(f"{clip.id}_*.mp4"))
        if matches:
            return matches[0]
        for name in (f"clip_{clip.id}.mp4", f"{clip.id}.mp4"):
            candidate = clips_dir / name
            if candidate.exists():
                return candidate
        return None

    def reframe_clip(
        self,
        clip_id: str,
        aspect_ratio: str = "9:16",
        focus_strategy: str = "track",
        fallback_strategy: str = "center",
    ) -> Dict[str, Any]:
        """
        将单个片段重构为竖屏（或 1:1）并把变体写回 clip_metadata。

        幂等：若已存在相同 aspect_ratio 且文件仍在的变体，直接复用缓存。

        Returns:
            {'success': bool, 'cached': bool, 'variant': dict|None, 'error': str|None}
        """
        from datetime import datetime
        from pathlib import Path
        from ..utils.video_processor import VideoProcessor
        from ..core.path_utils import get_project_directory

        out = {'success': False, 'cached': False, 'variant': None, 'error': None}

        clip = self.get(clip_id)
        if not clip:
            out['error'] = "切片不存在"
            return out

        metadata = dict(getattr(clip, 'clip_metadata', {}) or {})
        variants = list(metadata.get('vertical_variants', []) or [])

        # 缓存复用：相同比例且文件仍存在
        for v in variants:
            if v.get('aspect_ratio') == aspect_ratio and \
                    v.get('video_path') and Path(v['video_path']).exists():
                out.update({'success': True, 'cached': True, 'variant': v})
                return out

        source = self._resolve_clip_video_path(clip)
        if source is None or not source.exists():
            out['error'] = f"切片源视频文件不存在: {clip_id}"
            return out

        project_dir = get_project_directory(str(clip.project_id))
        clips_dir = project_dir / "output" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        suffix = aspect_ratio.replace(':', 'x')  # 9:16 -> 9x16
        safe_title = VideoProcessor.sanitize_filename(str(clip.title) or f"clip_{clip_id}")
        output_path = clips_dir / f"{clip_id}_{safe_title}_{suffix}.mp4"

        res = VideoProcessor.reframe_to_vertical(
            input_video=source,
            output_path=output_path,
            aspect_ratio=aspect_ratio,
            focus_strategy=focus_strategy,
            fallback_strategy=fallback_strategy,
        )
        if not res.get('success'):
            out['error'] = res.get('error') or "竖屏重构失败"
            return out

        # 竖屏缩略图
        thumb_path = clips_dir / f"{clip_id}_{safe_title}_{suffix}_thumb.jpg"
        thumb_ok = VideoProcessor.extract_thumbnail(output_path, thumb_path, time_offset=1)

        variant = {
            'aspect_ratio': aspect_ratio,
            'video_path': str(output_path),
            'thumbnail_path': str(thumb_path) if thumb_ok else None,
            'focus_strategy': res.get('focus_strategy', focus_strategy),
            'fallback_used': bool(res.get('fallback_used', False)),
            'width': res.get('width', 0),
            'height': res.get('height', 0),
            'created_at': datetime.now().isoformat(),
        }
        # 去重同比例旧记录后写入
        variants = [v for v in variants if v.get('aspect_ratio') != aspect_ratio]
        variants.append(variant)
        metadata['vertical_variants'] = variants

        self.update(clip_id, clip_metadata=metadata)
        out.update({'success': True, 'cached': False, 'variant': variant})
        return out
