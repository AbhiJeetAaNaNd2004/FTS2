from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
import logging
import cv2
import io
import asyncio
import json

from backend.config.database import get_db
from backend.core.security import require_permissions, get_current_user
from backend.models.database import User
from backend.models.enums import Permission

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/cameras")
async def list_cameras(
    current_user: User = Depends(require_permissions([Permission.VIEW_CAMERA_STREAM])),
    db: Session = Depends(get_db)
):
    """List available cameras."""
    try:
        from backend.core.face_tracking_system import get_face_tracking_system
        
        face_tracking_system = get_face_tracking_system()
        cameras = face_tracking_system._get_camera_configs()
        
        return {
            "cameras": [
                {
                    "camera_id": cam.camera_id,
                    "camera_type": cam.camera_type,
                    "resolution": cam.resolution,
                    "fps": cam.fps,
                    "gpu_id": cam.gpu_id
                }
                for cam in cameras
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve camera list"
        )


@router.get("/camera/{camera_id}/frame")
async def get_camera_frame(
    camera_id: int,
    current_user: User = Depends(require_permissions([Permission.VIEW_CAMERA_STREAM])),
    db: Session = Depends(get_db)
):
    """Get current frame from camera."""
    try:
        from backend.core.face_tracking_system import get_face_tracking_system
        
        face_tracking_system = get_face_tracking_system()
        frame = face_tracking_system.get_current_frame(camera_id)
        
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera {camera_id} not found or not active"
            )
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera frame: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve camera frame"
        )


# Note: WebSocket streaming is handled in main.py
# This is a placeholder for future streaming enhancements