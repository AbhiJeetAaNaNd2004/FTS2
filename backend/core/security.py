from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import logging
from sqlalchemy.orm import Session

from backend.config.settings import get_settings
from backend.config.database import get_db
from backend.models.database import User, Role, UserSession, AuditLog
from backend.models.enums import Permission, UserRole

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Custom authentication error."""
    pass


class AuthorizationError(Exception):
    """Custom authorization error."""
    pass


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        raise AuthenticationError("Invalid token")


def generate_session_token() -> str:
    """Generate a secure session token."""
    return secrets.token_urlsafe(32)


class AuthService:
    """Authentication and authorization service."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def authenticate_user(self, username: str, password: str, ip_address: str = None, 
                         user_agent: str = None) -> Optional[User]:
        """Authenticate user with username and password."""
        try:
            # Get user from database
            user = self.db.query(User).filter(User.username == username).first()
            
            if not user:
                logger.warning(f"Authentication failed: User not found - {username}")
                return None
            
            # Check if user is active
            if not user.is_active:
                logger.warning(f"Authentication failed: User inactive - {username}")
                return None
            
            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.utcnow():
                logger.warning(f"Authentication failed: Account locked - {username}")
                return None
            
            # Verify password
            if not verify_password(password, user.password_hash):
                # Increment failed login attempts
                user.failed_login_attempts += 1
                
                # Lock account after 5 failed attempts
                if user.failed_login_attempts >= 5:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                    logger.warning(f"Account locked due to failed attempts: {username}")
                
                self.db.commit()
                logger.warning(f"Authentication failed: Invalid password - {username}")
                return None
            
            # Reset failed login attempts on successful authentication
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login_time = datetime.utcnow()
            self.db.commit()
            
            # Log successful authentication
            self._log_audit_event(
                user.id, "login", "user", str(user.id),
                ip_address=ip_address, user_agent=user_agent
            )
            
            logger.info(f"User authenticated successfully: {username}")
            return user
            
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return None
    
    def create_user_session(self, user: User, ip_address: str = None, 
                           user_agent: str = None) -> Dict[str, str]:
        """Create a new user session with tokens."""
        try:
            # Generate tokens
            access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
            refresh_token_expires = timedelta(days=settings.refresh_token_expire_days)
            
            access_token = create_access_token(
                data={"sub": str(user.id), "username": user.username, "role": user.role.role_name},
                expires_delta=access_token_expires
            )
            
            refresh_token = create_refresh_token(
                data={"sub": str(user.id)},
                expires_delta=refresh_token_expires
            )
            
            # Generate session token
            session_token = generate_session_token()
            
            # Store session in database
            user_session = UserSession(
                user_id=user.id,
                session_token=session_token,
                refresh_token=refresh_token,
                expires_at=datetime.utcnow() + access_token_expires,
                refresh_expires_at=datetime.utcnow() + refresh_token_expires,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.db.add(user_session)
            self.db.commit()
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "session_token": session_token,
                "token_type": "bearer"
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user session: {e}")
            raise AuthenticationError("Failed to create session")
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token."""
        try:
            # Verify refresh token
            payload = verify_token(refresh_token)
            
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid token type")
            
            user_id = payload.get("sub")
            if not user_id:
                raise AuthenticationError("Invalid token payload")
            
            # Get user session from database
            session = self.db.query(UserSession).filter(
                UserSession.refresh_token == refresh_token,
                UserSession.is_active == True,
                UserSession.refresh_expires_at > datetime.utcnow()
            ).first()
            
            if not session:
                raise AuthenticationError("Invalid or expired refresh token")
            
            # Get user
            user = self.db.query(User).filter(User.id == session.user_id).first()
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            # Create new access token
            access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
            access_token = create_access_token(
                data={"sub": str(user.id), "username": user.username, "role": user.role.role_name},
                expires_delta=access_token_expires
            )
            
            # Update session expiry
            session.expires_at = datetime.utcnow() + access_token_expires
            session.last_accessed_at = datetime.utcnow()
            self.db.commit()
            
            return {
                "access_token": access_token,
                "token_type": "bearer"
            }
            
        except JWTError:
            raise AuthenticationError("Invalid refresh token")
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            raise AuthenticationError("Failed to refresh token")
    
    def revoke_session(self, session_token: str, user_id: int = None):
        """Revoke a user session."""
        try:
            query = self.db.query(UserSession).filter(UserSession.session_token == session_token)
            
            if user_id:
                query = query.filter(UserSession.user_id == user_id)
            
            session = query.first()
            if session:
                session.is_active = False
                self.db.commit()
                
                # Log logout
                self._log_audit_event(
                    session.user_id, "logout", "user", str(session.user_id)
                )
                
                logger.info(f"Session revoked: {session_token}")
            
        except Exception as e:
            logger.error(f"Error revoking session: {e}")
    
    def revoke_all_user_sessions(self, user_id: int):
        """Revoke all sessions for a user."""
        try:
            sessions = self.db.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            ).all()
            
            for session in sessions:
                session.is_active = False
            
            self.db.commit()
            logger.info(f"All sessions revoked for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error revoking all sessions: {e}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            expired_sessions = self.db.query(UserSession).filter(
                UserSession.refresh_expires_at < datetime.utcnow()
            ).all()
            
            for session in expired_sessions:
                self.db.delete(session)
            
            self.db.commit()
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
    
    def get_user_permissions(self, user: User) -> List[str]:
        """Get user permissions based on role."""
        if not user.role:
            return []
        
        permissions = user.role.permissions or {}
        return list(permissions.keys()) if isinstance(permissions, dict) else permissions
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user_permissions = self.get_user_permissions(user)
        return permission.value in user_permissions
    
    def require_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has all required permissions."""
        user_permissions = self.get_user_permissions(user)
        required_perms = [p.value for p in permissions]
        return all(perm in user_permissions for perm in required_perms)
    
    def _log_audit_event(self, user_id: int, action: str, resource_type: str, 
                        resource_id: str, old_values: Dict = None, new_values: Dict = None,
                        ip_address: str = None, user_agent: str = None):
        """Log audit event."""
        try:
            audit_log = AuditLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                old_values=old_values,
                new_values=new_values,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.db.add(audit_log)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    try:
        # Verify token
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise AuthenticationError("Invalid token payload")
        
        # Get user from database
        user = db.query(User).filter(User.id == int(user_id)).first()
        
        if not user:
            raise AuthenticationError("User not found")
        
        if not user.is_active:
            raise AuthenticationError("User inactive")
        
        # Check if session is still valid
        session = db.query(UserSession).filter(
            UserSession.user_id == user.id,
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow()
        ).first()
        
        if not session:
            raise AuthenticationError("Session expired")
        
        return user
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_permissions(permissions: List[Permission]):
    """Decorator to require specific permissions."""
    def permission_checker(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        auth_service = AuthService(db)
        
        if not auth_service.require_permissions(current_user, permissions):
            missing_perms = [p.value for p in permissions if not auth_service.has_permission(current_user, p)]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Missing: {', '.join(missing_perms)}"
            )
        
        return current_user
    
    return permission_checker


def require_role(allowed_roles: List[UserRole]):
    """Decorator to require specific roles."""
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role.role_name not in [role.value for role in allowed_roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join([r.value for r in allowed_roles])}"
            )
        
        return current_user
    
    return role_checker


# Convenience functions for common permission checks
def require_admin_permissions():
    """Require admin-level permissions."""
    return require_role([UserRole.ADMIN, UserRole.SUPER_ADMIN])


def require_super_admin_permissions():
    """Require super admin permissions."""
    return require_role([UserRole.SUPER_ADMIN])


def require_employee_access():
    """Require any authenticated user access."""
    return get_current_user


# Rate limiting decorator (simple implementation)
class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self):
        self.requests = {}
    
    def check_rate_limit(self, key: str, max_requests: int = 10, window_seconds: int = 60) -> bool:
        """Check if request is within rate limit."""
        now = datetime.utcnow()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside the window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if (now - req_time).total_seconds() < window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[key]) >= max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """Rate limiting decorator."""
    def rate_limit_checker(request: Request):
        # Use IP address as key for rate limiting
        client_ip = request.client.host
        
        if not rate_limiter.check_rate_limit(client_ip, max_requests, window_seconds):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return True
    
    return rate_limit_checker


def get_client_info(request: Request) -> Dict[str, str]:
    """Extract client information from request."""
    return {
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent", "")
    }