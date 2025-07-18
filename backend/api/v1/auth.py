from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging

from backend.config.database import get_db
from backend.core.security import (
    AuthService, get_current_user, get_client_info, rate_limit
)
from backend.models.database import User
from backend.models.enums import UserRole

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    session_token: str
    token_type: str
    user: 'UserInfo'


class UserInfo(BaseModel):
    id: int
    username: str
    email: Optional[str]
    full_name: Optional[str]
    role: str
    employee_id: Optional[str]
    is_active: bool
    
    class Config:
        from_attributes = True


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class RefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str


class LogoutRequest(BaseModel):
    session_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# Update LoginResponse to reference UserInfo properly
LoginResponse.model_rebuild()


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db),
    _rate_limit: bool = Depends(rate_limit(max_requests=5, window_seconds=300))
):
    """
    Authenticate user and return access tokens.
    Rate limited to 5 attempts per 5 minutes per IP.
    """
    try:
        # Get client information
        client_info = get_client_info(request)
        
        # Create auth service
        auth_service = AuthService(db)
        
        # Authenticate user
        user = auth_service.authenticate_user(
            username=login_data.username,
            password=login_data.password,
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"]
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Create user session
        tokens = auth_service.create_user_session(
            user=user,
            ip_address=client_info["ip_address"],
            user_agent=client_info["user_agent"]
        )
        
        # Return response with user info
        return LoginResponse(
            **tokens,
            user=UserInfo(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role.role_name,
                employee_id=user.employee_id,
                is_active=user.is_active
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    try:
        auth_service = AuthService(db)
        
        tokens = auth_service.refresh_access_token(refresh_data.refresh_token)
        
        return RefreshTokenResponse(**tokens)
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout(
    logout_data: LogoutRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout user and revoke session."""
    try:
        auth_service = AuthService(db)
        
        # Revoke session
        auth_service.revoke_session(
            session_token=logout_data.session_token,
            user_id=current_user.id
        )
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/logout-all")
async def logout_all(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Logout user from all sessions."""
    try:
        auth_service = AuthService(db)
        
        # Revoke all user sessions
        auth_service.revoke_all_user_sessions(current_user.id)
        
        return {"message": "Logged out from all sessions successfully"}
        
    except Exception as e:
        logger.error(f"Logout all error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserInfo)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserInfo(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role.role_name,
        employee_id=current_user.employee_id,
        is_active=current_user.is_active
    )


@router.post("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user password."""
    try:
        from backend.core.security import verify_password, get_password_hash
        
        # Verify current password
        if not verify_password(password_data.current_password, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password (add your own validation rules)
        if len(password_data.new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters long"
            )
        
        # Update password
        current_user.password_hash = get_password_hash(password_data.new_password)
        db.commit()
        
        # Revoke all sessions to force re-login
        auth_service = AuthService(db)
        auth_service.revoke_all_user_sessions(current_user.id)
        
        return {"message": "Password changed successfully. Please log in again."}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.get("/permissions")
async def get_user_permissions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's permissions."""
    try:
        auth_service = AuthService(db)
        permissions = auth_service.get_user_permissions(current_user)
        
        return {
            "permissions": permissions,
            "role": current_user.role.role_name
        }
        
    except Exception as e:
        logger.error(f"Get permissions error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get permissions"
        )


@router.get("/check-session")
async def check_session(current_user: User = Depends(get_current_user)):
    """Check if current session is valid."""
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "role": current_user.role.role_name
    }