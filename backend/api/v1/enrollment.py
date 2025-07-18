from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import tempfile
import shutil

from backend.config.database import get_db
from backend.core.security import require_permissions, get_current_user
from backend.models.database import Employee, User
from backend.models.enums import Permission
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# Pydantic models
class EnrollmentRequest(BaseModel):
    employee_id: str
    employee_name: str
    department: Optional[str] = None
    designation: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    update_existing: bool = False


class EnrollmentResponse(BaseModel):
    message: str
    employee_id: str
    employee_name: str
    embeddings_created: int
    total_embeddings: int


@router.post("/enroll", response_model=EnrollmentResponse)
async def enroll_employee_faces(
    employee_id: str = Form(...),
    employee_name: str = Form(...),
    department: Optional[str] = Form(None),
    designation: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    update_existing: bool = Form(False),
    files: List[UploadFile] = File(...),
    current_user: User = Depends(require_permissions([Permission.ENROLL_FACES])),
    db: Session = Depends(get_db)
):
    """Enroll employee faces from uploaded images."""
    try:
        # Validate files
        if len(files) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Minimum 3 images required for enrollment"
            )
        
        # Validate file types
        allowed_extensions = settings.allowed_image_extensions
        for file in files:
            if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
                )
        
        # Check if employee exists
        existing_employee = db.query(Employee).filter(Employee.id == employee_id).first()
        
        if existing_employee and not update_existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Employee already exists. Use update_existing=true to add more images."
            )
        
        # Create employee if doesn't exist
        if not existing_employee:
            new_employee = Employee(
                id=employee_id,
                employee_name=employee_name,
                department=department,
                designation=designation,
                email=email,
                phone=phone
            )
            db.add(new_employee)
            db.commit()
            db.refresh(new_employee)
        
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        try:
            for i, file in enumerate(files):
                # Create temporary file
                temp_path = os.path.join(temp_dir, f"{employee_id}_{i}_{file.filename}")
                
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                image_paths.append(temp_path)
            
            # Initialize face enroller
            from backend.core.face_tracking_system import get_face_tracking_system
            face_tracking_system = get_face_tracking_system()
            
            # Import the face enroller from the legacy system
            import sys
            sys.path.append('legacy')
            from backend.core.face_enroller import FaceEnroller
            
            face_enroller = FaceEnroller(tracking_system=face_tracking_system)
            
            # Enroll faces
            success = face_enroller.enroll_from_images(
                employee_id=employee_id,
                employee_name=employee_name,
                image_paths=image_paths,
                min_faces=3,
                update_existing=update_existing,
                rebuild_index=True
            )
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Face enrollment failed. Please check image quality and try again."
                )
            
            # Get embedding count
            from backend.models.database import FaceEmbedding
            embedding_count = db.query(FaceEmbedding).filter(
                FaceEmbedding.employee_id == employee_id,
                FaceEmbedding.is_active == True
            ).count()
            
            action = "updated" if update_existing else "enrolled"
            new_embeddings = len(image_paths)  # Assuming all images were processed successfully
            
            logger.info(f"Employee {action}: {employee_id} with {new_embeddings} images by user {current_user.username}")
            
            return EnrollmentResponse(
                message=f"Employee {action} successfully",
                employee_id=employee_id,
                employee_name=employee_name,
                embeddings_created=new_embeddings,
                total_embeddings=embedding_count
            )
            
        finally:
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during enrollment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Face enrollment failed"
        )


@router.post("/add-images/{employee_id}")
async def add_employee_images(
    employee_id: str,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(require_permissions([Permission.ENROLL_FACES])),
    db: Session = Depends(get_db)
):
    """Add additional images for existing employee."""
    try:
        # Check if employee exists
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        # Validate files
        if len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one image required"
            )
        
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        
        try:
            for i, file in enumerate(files):
                temp_path = os.path.join(temp_dir, f"{employee_id}_add_{i}_{file.filename}")
                
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                image_paths.append(temp_path)
            
            # Add images using face enroller
            from backend.core.face_tracking_system import get_face_tracking_system
            face_tracking_system = get_face_tracking_system()
            
            from backend.core.face_enroller import FaceEnroller
            face_enroller = FaceEnroller(tracking_system=face_tracking_system)
            
            successful_additions = 0
            for image_path in image_paths:
                try:
                    success = face_enroller.add_embedding(
                        employee_id=employee_id,
                        image_path=image_path,
                        rebuild_index=False  # Rebuild once at the end
                    )
                    if success:
                        successful_additions += 1
                except Exception as e:
                    logger.warning(f"Failed to add image {image_path}: {e}")
            
            # Rebuild index once after all additions
            if successful_additions > 0:
                face_tracking_system.reload_embeddings_and_rebuild_index()
            
            logger.info(f"Added {successful_additions} images for employee {employee_id} by user {current_user.username}")
            
            return {
                "message": f"Successfully added {successful_additions} out of {len(files)} images",
                "employee_id": employee_id,
                "images_added": successful_additions,
                "total_files": len(files)
            }
            
        finally:
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding images for employee {employee_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add employee images"
        )


@router.get("/status/{employee_id}")
async def get_enrollment_status(
    employee_id: str,
    current_user: User = Depends(require_permissions([Permission.ENROLL_FACES])),
    db: Session = Depends(get_db)
):
    """Get enrollment status for employee."""
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        # Get embedding information
        from backend.models.database import FaceEmbedding
        embeddings = db.query(FaceEmbedding).filter(
            FaceEmbedding.employee_id == employee_id,
            FaceEmbedding.is_active == True
        ).all()
        
        enroll_embeddings = [e for e in embeddings if e.embedding_type == 'enroll']
        update_embeddings = [e for e in embeddings if e.embedding_type == 'update']
        
        return {
            "employee_id": employee_id,
            "employee_name": employee.employee_name,
            "is_enrolled": len(enroll_embeddings) > 0,
            "total_embeddings": len(embeddings),
            "enroll_embeddings": len(enroll_embeddings),
            "update_embeddings": len(update_embeddings),
            "enrollment_date": employee.created_at.isoformat(),
            "last_updated": employee.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting enrollment status for {employee_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get enrollment status"
        )


@router.delete("/{employee_id}/embeddings")
async def delete_all_employee_embeddings(
    employee_id: str,
    current_user: User = Depends(require_permissions([Permission.DELETE_EMBEDDINGS])),
    db: Session = Depends(get_db)
):
    """Delete all embeddings for an employee."""
    try:
        employee = db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        # Soft delete all embeddings
        from backend.models.database import FaceEmbedding
        embeddings = db.query(FaceEmbedding).filter(
            FaceEmbedding.employee_id == employee_id,
            FaceEmbedding.is_active == True
        ).all()
        
        for embedding in embeddings:
            embedding.is_active = False
        
        db.commit()
        
        # Reload face tracking system
        from backend.core.face_tracking_system import get_face_tracking_system
        face_tracking_system = get_face_tracking_system()
        face_tracking_system.reload_embeddings_and_rebuild_index()
        
        logger.info(f"All embeddings deleted for employee {employee_id} by user {current_user.username}")
        
        return {
            "message": f"All embeddings deleted for employee {employee_id}",
            "deleted_count": len(embeddings)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting embeddings for {employee_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete employee embeddings"
        )