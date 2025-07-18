from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging

from backend.config.database import get_db
from backend.core.security import require_admin_permissions, get_current_user
from backend.models.database import User, SystemLog

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class SystemLogResponse(BaseModel):
    id: int
    log_level: str
    message: str
    component: Optional[str]
    user_id: Optional[int]
    employee_id: Optional[str]
    camera_id: Optional[int]
    timestamp: str
    
    class Config:
        from_attributes = True


@router.get("/logs", response_model=List[SystemLogResponse])
async def get_system_logs(
    skip: int = 0,
    limit: int = 100,
    log_level: Optional[str] = None,
    component: Optional[str] = None,
    current_user: User = Depends(require_admin_permissions()),
    db: Session = Depends(get_db)
):
    """Get system logs."""
    try:
        query = db.query(SystemLog)
        
        if log_level:
            query = query.filter(SystemLog.log_level == log_level)
        
        if component:
            query = query.filter(SystemLog.component == component)
        
        logs = query.order_by(SystemLog.timestamp.desc()).offset(skip).limit(limit).all()
        
        return [
            SystemLogResponse(
                id=log.id,
                log_level=log.log_level,
                message=log.message,
                component=log.component,
                user_id=log.user_id,
                employee_id=log.employee_id,
                camera_id=log.camera_id,
                timestamp=log.timestamp.isoformat()
            )
            for log in logs
        ]
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system logs"
        )


@router.get("/system-status")
async def get_system_status(
    current_user: User = Depends(require_admin_permissions()),
    db: Session = Depends(get_db)
):
    """Get overall system status."""
    try:
        from backend.core.face_tracking_system import get_face_tracking_system
        
        # Get face tracking system status
        face_tracking_system = get_face_tracking_system()
        
        # Get database stats
        from backend.models.database import Employee, FaceEmbedding, AttendanceRecord
        
        total_employees = db.query(Employee).filter(Employee.is_active == True).count()
        total_embeddings = db.query(FaceEmbedding).filter(FaceEmbedding.is_active == True).count()
        total_attendance_records = db.query(AttendanceRecord).filter(AttendanceRecord.is_valid == True).count()
        
        # Get present employees count
        present_employees = len(face_tracking_system.get_present_employees())
        
        return {
            "face_tracking_active": True,
            "database_connected": True,
            "statistics": {
                "total_employees": total_employees,
                "total_embeddings": total_embeddings,
                "total_attendance_records": total_attendance_records,
                "present_employees": present_employees
            },
            "cameras": len(face_tracking_system._get_camera_configs()),
            "system_health": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status"
        )


@router.post("/reload-system")
async def reload_face_tracking_system(
    current_user: User = Depends(require_admin_permissions()),
    db: Session = Depends(get_db)
):
    """Reload face tracking system embeddings."""
    try:
        from backend.core.face_tracking_system import get_face_tracking_system
        
        face_tracking_system = get_face_tracking_system()
        face_tracking_system.reload_embeddings_and_rebuild_index()
        
        logger.info(f"Face tracking system reloaded by admin user: {current_user.username}")
        
        return {"message": "Face tracking system reloaded successfully"}
        
    except Exception as e:
        logger.error(f"Error reloading face tracking system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload face tracking system"
        )