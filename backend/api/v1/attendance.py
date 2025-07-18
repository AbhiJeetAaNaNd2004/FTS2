from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date
import logging

from backend.config.database import get_db
from backend.core.security import require_permissions, get_current_user
from backend.models.database import AttendanceRecord, Employee, User
from backend.models.enums import Permission

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class AttendanceRecordResponse(BaseModel):
    id: int
    employee_id: str
    employee_name: str
    camera_id: int
    event_type: str
    timestamp: str
    confidence_score: Optional[float]
    work_status: str
    notes: Optional[str]
    
    class Config:
        from_attributes = True


class PresentEmployeeResponse(BaseModel):
    employee_id: str
    employee_name: str
    department: Optional[str]
    check_in_time: str
    camera_id: int


@router.get("/records", response_model=List[AttendanceRecordResponse])
async def get_attendance_records(
    employee_id: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    event_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get attendance records with filtering."""
    try:
        # Check permissions - employees can only see their own records
        if current_user.role.role_name == "employee":
            if not current_user.employee_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Employee account not linked to employee record"
                )
            employee_id = current_user.employee_id
        else:
            # Admin/Super Admin can view all records or filter by employee
            from backend.core.security import AuthService
            auth_service = AuthService(db)
            if not auth_service.has_permission(current_user, Permission.VIEW_ALL_ATTENDANCE):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions to view all attendance records"
                )
        
        # Build query
        query = db.query(AttendanceRecord, Employee.employee_name).join(
            Employee, AttendanceRecord.employee_id == Employee.id
        ).filter(AttendanceRecord.is_valid == True)
        
        if employee_id:
            query = query.filter(AttendanceRecord.employee_id == employee_id)
        
        if start_date:
            query = query.filter(AttendanceRecord.timestamp >= start_date)
        
        if end_date:
            query = query.filter(AttendanceRecord.timestamp <= end_date)
        
        if event_type:
            query = query.filter(AttendanceRecord.event_type == event_type)
        
        # Order by timestamp descending
        query = query.order_by(AttendanceRecord.timestamp.desc())
        
        # Apply pagination
        records = query.offset(skip).limit(limit).all()
        
        return [
            AttendanceRecordResponse(
                id=record.AttendanceRecord.id,
                employee_id=record.AttendanceRecord.employee_id,
                employee_name=record.employee_name,
                camera_id=record.AttendanceRecord.camera_id,
                event_type=record.AttendanceRecord.event_type,
                timestamp=record.AttendanceRecord.timestamp.isoformat(),
                confidence_score=record.AttendanceRecord.confidence_score,
                work_status=record.AttendanceRecord.work_status,
                notes=record.AttendanceRecord.notes
            )
            for record in records
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting attendance records: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve attendance records"
        )


@router.get("/present", response_model=List[PresentEmployeeResponse])
async def get_present_employees(
    current_user: User = Depends(require_permissions([Permission.VIEW_PRESENT_EMPLOYEES])),
    db: Session = Depends(get_db)
):
    """Get list of currently present employees."""
    try:
        from backend.core.face_tracking_system import get_face_tracking_system
        
        # Get present employees from face tracking system
        face_tracking_system = get_face_tracking_system()
        present_employees = face_tracking_system.get_present_employees()
        
        return [
            PresentEmployeeResponse(
                employee_id=emp["employee_id"],
                employee_name=emp["employee_name"],
                department=emp.get("department"),
                check_in_time=emp["check_in_time"],
                camera_id=emp["camera_id"]
            )
            for emp in present_employees
        ]
        
    except Exception as e:
        logger.error(f"Error getting present employees: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve present employees"
        )


@router.get("/employee/{employee_id}", response_model=List[AttendanceRecordResponse])
async def get_employee_attendance(
    employee_id: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get attendance records for specific employee."""
    try:
        # Check permissions
        if current_user.role.role_name == "employee":
            if current_user.employee_id != employee_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot access other employee's attendance records"
                )
        else:
            from backend.core.security import AuthService
            auth_service = AuthService(db)
            if not auth_service.has_permission(current_user, Permission.VIEW_ALL_ATTENDANCE):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions to view employee attendance"
                )
        
        # Verify employee exists
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        # Build query
        query = db.query(AttendanceRecord).filter(
            AttendanceRecord.employee_id == employee_id,
            AttendanceRecord.is_valid == True
        )
        
        if start_date:
            query = query.filter(AttendanceRecord.timestamp >= start_date)
        
        if end_date:
            query = query.filter(AttendanceRecord.timestamp <= end_date)
        
        # Order by timestamp descending
        query = query.order_by(AttendanceRecord.timestamp.desc())
        
        # Apply pagination
        records = query.offset(skip).limit(limit).all()
        
        return [
            AttendanceRecordResponse(
                id=record.id,
                employee_id=record.employee_id,
                employee_name=employee.employee_name,
                camera_id=record.camera_id,
                event_type=record.event_type,
                timestamp=record.timestamp.isoformat(),
                confidence_score=record.confidence_score,
                work_status=record.work_status,
                notes=record.notes
            )
            for record in records
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting employee attendance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve employee attendance"
        )


@router.get("/stats")
async def get_attendance_stats(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_user: User = Depends(require_permissions([Permission.VIEW_ALL_ATTENDANCE])),
    db: Session = Depends(get_db)
):
    """Get attendance statistics."""
    try:
        from sqlalchemy import func, and_
        
        # Build base query
        query_filter = [AttendanceRecord.is_valid == True]
        
        if start_date:
            query_filter.append(AttendanceRecord.timestamp >= start_date)
        
        if end_date:
            query_filter.append(AttendanceRecord.timestamp <= end_date)
        
        # Get total check-ins and check-outs
        check_ins = db.query(func.count(AttendanceRecord.id)).filter(
            and_(AttendanceRecord.event_type == 'check_in', *query_filter)
        ).scalar()
        
        check_outs = db.query(func.count(AttendanceRecord.id)).filter(
            and_(AttendanceRecord.event_type == 'check_out', *query_filter)
        ).scalar()
        
        # Get unique employees with attendance
        unique_employees = db.query(func.count(func.distinct(AttendanceRecord.employee_id))).filter(
            and_(*query_filter)
        ).scalar()
        
        # Get currently present count
        present_count = len(get_face_tracking_system().get_present_employees())
        
        return {
            "total_check_ins": check_ins,
            "total_check_outs": check_outs,
            "unique_employees": unique_employees,
            "currently_present": present_count,
            "period": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting attendance stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve attendance statistics"
        )