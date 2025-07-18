from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import logging

from backend.config.database import get_db
from backend.core.security import require_super_admin_permissions, get_current_user, get_password_hash
from backend.models.database import User, Role
from backend.models.enums import UserRole

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    employee_id: Optional[str] = None
    password: str
    role_name: str
    is_active: bool = True


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    employee_id: Optional[str] = None
    role_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str]
    full_name: Optional[str]
    employee_id: Optional[str]
    is_active: bool
    role_name: str
    created_at: str
    last_login_time: Optional[str]
    
    class Config:
        from_attributes = True


@router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(require_super_admin_permissions()),
    db: Session = Depends(get_db)
):
    """List all users."""
    try:
        users = db.query(User).offset(skip).limit(limit).all()
        
        return [
            UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                employee_id=user.employee_id,
                is_active=user.is_active,
                role_name=user.role.role_name,
                created_at=user.created_at.isoformat(),
                last_login_time=user.last_login_time.isoformat() if user.last_login_time else None
            )
            for user in users
        ]
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_super_admin_permissions()),
    db: Session = Depends(get_db)
):
    """Create a new user."""
    try:
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == user_data.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email already exists
        if user_data.email:
            existing_email = db.query(User).filter(User.email == user_data.email).first()
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already exists"
                )
        
        # Get role
        role = db.query(Role).filter(Role.role_name == user_data.role_name).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role"
            )
        
        # Create user
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            employee_id=user_data.employee_id,
            password_hash=get_password_hash(user_data.password),
            is_active=user_data.is_active,
            role_id=role.id
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"User created: {user_data.username} by super admin {current_user.username}")
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            full_name=new_user.full_name,
            employee_id=new_user.employee_id,
            is_active=new_user.is_active,
            role_name=new_user.role.role_name,
            created_at=new_user.created_at.isoformat(),
            last_login_time=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(require_super_admin_permissions()),
    db: Session = Depends(get_db)
):
    """Update user information."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields if provided
        update_data = user_update.dict(exclude_unset=True)
        
        # Handle role update
        if "role_name" in update_data:
            role = db.query(Role).filter(Role.role_name == update_data["role_name"]).first()
            if not role:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid role"
                )
            user.role_id = role.id
            del update_data["role_name"]
        
        # Update other fields
        for field, value in update_data.items():
            setattr(user, field, value)
        
        db.commit()
        db.refresh(user)
        
        logger.info(f"User updated: {user.username} by super admin {current_user.username}")
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            employee_id=user.employee_id,
            is_active=user.is_active,
            role_name=user.role.role_name,
            created_at=user.created_at.isoformat(),
            last_login_time=user.last_login_time.isoformat() if user.last_login_time else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_super_admin_permissions()),
    db: Session = Depends(get_db)
):
    """Delete a user."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent self-deletion
        if user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        db.delete(user)
        db.commit()
        
        logger.info(f"User deleted: {user.username} by super admin {current_user.username}")
        
        return {"message": f"User {user.username} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.get("/roles")
async def list_roles(
    current_user: User = Depends(require_super_admin_permissions()),
    db: Session = Depends(get_db)
):
    """List all available roles."""
    try:
        roles = db.query(Role).filter(Role.is_active == True).all()
        
        return {
            "roles": [
                {
                    "id": role.id,
                    "role_name": role.role_name,
                    "display_name": role.display_name,
                    "description": role.description,
                    "permissions": role.permissions
                }
                for role in roles
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing roles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve roles"
        )