from enum import Enum


class UserRole(str, Enum):
    """User role enumeration."""
    EMPLOYEE = "employee"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class AttendanceEventType(str, Enum):
    """Attendance event types."""
    CHECK_IN = "check_in"
    CHECK_OUT = "check_out"


class WorkStatus(str, Enum):
    """Work status enumeration."""
    WORKING = "working"
    BREAK = "break"
    MEETING = "meeting"
    OFFLINE = "offline"


class CameraType(str, Enum):
    """Camera type enumeration."""
    ENTRY = "entry"
    EXIT = "exit"
    GENERAL = "general"


class EmbeddingType(str, Enum):
    """Face embedding type enumeration."""
    ENROLL = "enroll"
    UPDATE = "update"


class TrackingState(str, Enum):
    """Tracking state enumeration."""
    ACTIVE = "active"
    LOST = "lost"
    OCCLUDED = "occluded"


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditAction(str, Enum):
    """Audit action enumeration."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ENROLL = "enroll"
    VIEW_STREAM = "view_stream"


class Permission(str, Enum):
    """Permission enumeration for role-based access control."""
    # Employee permissions
    VIEW_OWN_ATTENDANCE = "view_own_attendance"
    VIEW_PRESENT_EMPLOYEES = "view_present_employees"
    
    # Admin permissions
    VIEW_ALL_ATTENDANCE = "view_all_attendance"
    MANAGE_EMPLOYEES = "manage_employees"
    ENROLL_FACES = "enroll_faces"
    DELETE_EMBEDDINGS = "delete_embeddings"
    VIEW_CAMERA_STREAM = "view_camera_stream"
    MANAGE_CAMERA_CONFIG = "manage_camera_config"
    VIEW_SYSTEM_LOGS = "view_system_logs"
    
    # Super Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    SYSTEM_SETTINGS = "system_settings"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    DATABASE_BACKUP = "database_backup"


# Default role permissions mapping
DEFAULT_ROLE_PERMISSIONS = {
    UserRole.EMPLOYEE: [
        Permission.VIEW_OWN_ATTENDANCE,
        Permission.VIEW_PRESENT_EMPLOYEES,
    ],
    UserRole.ADMIN: [
        Permission.VIEW_OWN_ATTENDANCE,
        Permission.VIEW_PRESENT_EMPLOYEES,
        Permission.VIEW_ALL_ATTENDANCE,
        Permission.MANAGE_EMPLOYEES,
        Permission.ENROLL_FACES,
        Permission.DELETE_EMBEDDINGS,
        Permission.VIEW_CAMERA_STREAM,
        Permission.MANAGE_CAMERA_CONFIG,
        Permission.VIEW_SYSTEM_LOGS,
    ],
    UserRole.SUPER_ADMIN: [
        # Super admin has all permissions
        Permission.VIEW_OWN_ATTENDANCE,
        Permission.VIEW_PRESENT_EMPLOYEES,
        Permission.VIEW_ALL_ATTENDANCE,
        Permission.MANAGE_EMPLOYEES,
        Permission.ENROLL_FACES,
        Permission.DELETE_EMBEDDINGS,
        Permission.VIEW_CAMERA_STREAM,
        Permission.MANAGE_CAMERA_CONFIG,
        Permission.VIEW_SYSTEM_LOGS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_ROLES,
        Permission.SYSTEM_SETTINGS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.DATABASE_BACKUP,
    ]
}