from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging

from backend.config.database import get_db
from backend.core.security import require_permissions, require_admin_permissions, get_current_user
from backend.models.database import Employee, User
from backend.models.enums import Permission

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class EmployeeCreate(BaseModel):
    id: str
    employee_name: str
    department: Optional[str] = None
    designation: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class EmployeeUpdate(BaseModel):
    employee_name: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


class EmployeeResponse(BaseModel):
    id: str
    employee_name: str
    department: Optional[str]
    designation: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    is_active: bool
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


@router.get("/", response_model=List[EmployeeResponse])
async def list_employees(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    department: Optional[str] = Query(None),
    active_only: bool = Query(True),
    current_user: User = Depends(require_permissions([Permission.MANAGE_EMPLOYEES])),
    db: Session = Depends(get_db)
):
    """List employees with pagination and filtering."""
    try:
        query = db.query(Employee)
        
        if active_only:
            query = query.filter(Employee.is_active == True)
        
        if department:
            query = query.filter(Employee.department.ilike(f"%{department}%"))
        
        employees = query.offset(skip).limit(limit).all()
        
        return [EmployeeResponse(
            id=emp.id,
            employee_name=emp.employee_name,
            department=emp.department,
            designation=emp.designation,
            email=emp.email,
            phone=emp.phone,
            is_active=emp.is_active,
            created_at=emp.created_at.isoformat(),
            updated_at=emp.updated_at.isoformat()
        ) for emp in employees]
        
    except Exception as e:
        logger.error(f"Error listing employees: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve employees"
        )


@router.get("/{employee_id}", response_model=EmployeeResponse)
async def get_employee(
    employee_id: str,
    current_user: User = Depends(require_permissions([Permission.MANAGE_EMPLOYEES])),
    db: Session = Depends(get_db)
):
    """Get employee details by ID."""
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        return EmployeeResponse(
            id=employee.id,
            employee_name=employee.employee_name,
            department=employee.department,
            designation=employee.designation,
            email=employee.email,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at.isoformat(),
            updated_at=employee.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting employee {employee_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve employee"
        )


@router.post("/", response_model=EmployeeResponse)
async def create_employee(
    employee: EmployeeCreate,
    current_user: User = Depends(require_permissions([Permission.MANAGE_EMPLOYEES])),
    db: Session = Depends(get_db)
):
    """Create a new employee."""
    try:
        # Check if employee already exists
        existing = db.query(Employee).filter(Employee.id == employee.id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Employee with this ID already exists"
            )
        
        # Create new employee
        new_employee = Employee(
            id=employee.id,
            employee_name=employee.employee_name,
            department=employee.department,
            designation=employee.designation,
            email=employee.email,
            phone=employee.phone
        )
        
        db.add(new_employee)
        db.commit()
        db.refresh(new_employee)
        
        logger.info(f"Employee created: {employee.id} by user {current_user.username}")
        
        return EmployeeResponse(
            id=new_employee.id,
            employee_name=new_employee.employee_name,
            department=new_employee.department,
            designation=new_employee.designation,
            email=new_employee.email,
            phone=new_employee.phone,
            is_active=new_employee.is_active,
            created_at=new_employee.created_at.isoformat(),
            updated_at=new_employee.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating employee: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create employee"
        )


@router.put("/{employee_id}", response_model=EmployeeResponse)
async def update_employee(
    employee_id: str,
    employee_update: EmployeeUpdate,
    current_user: User = Depends(require_permissions([Permission.MANAGE_EMPLOYEES])),
    db: Session = Depends(get_db)
):
    """Update employee information."""
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        # Update fields if provided
        update_data = employee_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(employee, field, value)
        
        db.commit()
        db.refresh(employee)
        
        logger.info(f"Employee updated: {employee_id} by user {current_user.username}")
        
        return EmployeeResponse(
            id=employee.id,
            employee_name=employee.employee_name,
            department=employee.department,
            designation=employee.designation,
            email=employee.email,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at.isoformat(),
            updated_at=employee.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating employee {employee_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update employee"
        )


@router.delete("/{employee_id}")
async def delete_employee(
    employee_id: str,
    current_user: User = Depends(require_permissions([Permission.MANAGE_EMPLOYEES])),
    db: Session = Depends(get_db)
):
    """Delete employee and all associated data."""
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        # Delete employee (cascade will handle related records)
        db.delete(employee)
        db.commit()
        
        logger.info(f"Employee deleted: {employee_id} by user {current_user.username}")
        
        return {"message": f"Employee {employee_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting employee {employee_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete employee"
        )


@router.get("/{employee_id}/embeddings")
async def get_employee_embeddings(
    employee_id: str,
    current_user: User = Depends(require_permissions([Permission.MANAGE_EMPLOYEES])),
    db: Session = Depends(get_db)
):
    """Get employee's face embeddings."""
    try:
        from backend.models.database import FaceEmbedding
        
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        embeddings = db.query(FaceEmbedding).filter(
            FaceEmbedding.employee_id == employee_id,
            FaceEmbedding.is_active == True
        ).all()
        
        return {
            "employee_id": employee_id,
            "employee_name": employee.employee_name,
            "embeddings": [
                {
                    "id": emb.id,
                    "embedding_type": emb.embedding_type,
                    "quality_score": emb.quality_score,
                    "source_image_path": emb.source_image_path,
                    "created_at": emb.created_at.isoformat()
                }
                for emb in embeddings
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting embeddings for employee {employee_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve employee embeddings"
        )


@router.delete("/{employee_id}/embeddings/{embedding_id}")
async def delete_employee_embedding(
    employee_id: str,
    embedding_id: int,
    current_user: User = Depends(require_permissions([Permission.DELETE_EMBEDDINGS])),
    db: Session = Depends(get_db)
):
    """Delete a specific employee embedding."""
    try:
        from backend.models.database import FaceEmbedding
        
        embedding = db.query(FaceEmbedding).filter(
            FaceEmbedding.id == embedding_id,
            FaceEmbedding.employee_id == employee_id
        ).first()
        
        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Embedding not found"
            )
        
        # Soft delete by setting is_active to False
        embedding.is_active = False
        db.commit()
        
        # Reload face tracking system embeddings
        from backend.core.face_tracking_system import get_face_tracking_system
        face_tracking_system = get_face_tracking_system()
        face_tracking_system.reload_embeddings_and_rebuild_index()
        
        logger.info(f"Embedding deleted: {embedding_id} for employee {employee_id} by user {current_user.username}")
        
        return {"message": f"Embedding {embedding_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting embedding {embedding_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete embedding"
        )