from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey, LargeBinary, JSON, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.config.database import Base
import datetime


class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(String, primary_key=True, index=True)
    employee_name = Column(String, nullable=False)
    department = Column(String)
    designation = Column(String)
    email = Column(String, unique=True)
    phone = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    embeddings = relationship("FaceEmbedding", back_populates="employee", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="employee")
    tracking_records = relationship("TrackingRecord", back_populates="employee")
    
    # Indexes
    __table_args__ = (
        Index('idx_employee_active', 'is_active'),
        Index('idx_employee_name', 'employee_name'),
        Index('idx_employee_email', 'email'),
    )


class FaceEmbedding(Base):
    __tablename__ = 'face_embeddings'
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, ForeignKey('employees.id', ondelete='CASCADE'), nullable=False)
    embedding_data = Column(LargeBinary, nullable=False)
    embedding_type = Column(String, default='enroll')  # 'enroll', 'update'
    quality_score = Column(Float)
    source_image_path = Column(String)
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    employee = relationship("Employee", back_populates="embeddings")
    
    # Indexes
    __table_args__ = (
        Index('idx_embedding_employee_active', 'employee_id', 'is_active'),
        Index('idx_embedding_type', 'embedding_type'),
        Index('idx_embedding_created', 'created_at'),
    )


class AttendanceRecord(Base):
    __tablename__ = 'attendance_records'
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, ForeignKey('employees.id'), nullable=False)
    camera_id = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False)  # 'check_in', 'check_out'
    timestamp = Column(DateTime, default=func.now())
    confidence_score = Column(Float)
    work_status = Column(String, default='working')
    is_valid = Column(Boolean, default=True)
    notes = Column(Text)
    
    # Relationships
    employee = relationship("Employee", back_populates="attendance_records")
    
    # Indexes
    __table_args__ = (
        Index('idx_attendance_employee_timestamp', 'employee_id', 'timestamp'),
        Index('idx_attendance_timestamp', 'timestamp'),
        Index('idx_attendance_event_type', 'event_type'),
        Index('idx_attendance_valid', 'is_valid'),
    )


class TrackingRecord(Base):
    __tablename__ = 'tracking_records'
    
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, ForeignKey('employees.id'), nullable=False)
    camera_id = Column(Integer, nullable=False)
    position_x = Column(Float)
    position_y = Column(Float)
    confidence_score = Column(Float)
    quality_metrics = Column(JSON)
    timestamp = Column(DateTime, default=func.now())
    tracking_state = Column(String, default='active')
    
    # Relationships
    employee = relationship("Employee", back_populates="tracking_records")
    
    # Indexes
    __table_args__ = (
        Index('idx_tracking_employee_camera', 'employee_id', 'camera_id'),
        Index('idx_tracking_timestamp', 'timestamp'),
    )


class CameraConfig(Base):
    __tablename__ = 'camera_configs'
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, unique=True, nullable=False)
    camera_name = Column(String, nullable=False)
    camera_type = Column(String, default='entry')  # 'entry', 'exit'
    resolution_width = Column(Integer, default=1920)
    resolution_height = Column(Integer, default=1080)
    fps = Column(Integer, default=30)
    gpu_id = Column(Integer, default=0)
    tripwire_config = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_camera_active', 'is_active'),
        Index('idx_camera_type', 'camera_type'),
    )


class Role(Base):
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True, index=True)
    role_name = Column(String, unique=True, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(Text)
    permissions = Column(JSON)  # Flexible permissions storage as JSON
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    users = relationship("User", back_populates="role")
    
    # Indexes
    __table_args__ = (
        Index('idx_role_name', 'role_name'),
        Index('idx_role_active', 'is_active'),
    )


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    employee_id = Column(String, ForeignKey('employees.id'), nullable=True)  # Link to employee record for employees
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    last_login_time = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    role_id = Column(Integer, ForeignKey('roles.id'), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    role = relationship("Role", back_populates="users")
    employee = relationship("Employee")
    user_sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_username', 'username'),
        Index('idx_user_email', 'email'),
        Index('idx_user_active', 'is_active'),
        Index('idx_user_employee', 'employee_id'),
    )


class UserSession(Base):
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    session_token = Column(String, unique=True, nullable=False, index=True)
    refresh_token = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    refresh_expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String)
    user_agent = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_accessed_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="user_sessions")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_token', 'session_token'),
        Index('idx_session_refresh_token', 'refresh_token'),
        Index('idx_session_expires', 'expires_at'),
        Index('idx_session_user_active', 'user_id', 'is_active'),
    )


class SystemLog(Base):
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    log_level = Column(String, default='INFO')
    message = Column(Text, nullable=False)
    component = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    employee_id = Column(String, ForeignKey('employees.id'))
    camera_id = Column(Integer)
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=func.now())
    additional_data = Column(JSON)
    
    # Relationships
    user = relationship("User")
    employee = relationship("Employee")
    
    # Indexes
    __table_args__ = (
        Index('idx_log_timestamp', 'timestamp'),
        Index('idx_log_level', 'log_level'),
        Index('idx_log_component', 'component'),
        Index('idx_log_user', 'user_id'),
        Index('idx_log_employee', 'employee_id'),
    )


class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)  # 'employee', 'user', 'attendance', etc.
    resource_id = Column(String, nullable=False)
    old_values = Column(JSON)
    new_values = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_action', 'action'),
    )


class SystemSettings(Base):
    __tablename__ = 'system_settings'
    
    id = Column(Integer, primary_key=True, index=True)
    setting_key = Column(String, unique=True, nullable=False, index=True)
    setting_value = Column(JSON, nullable=False)
    description = Column(Text)
    is_encrypted = Column(Boolean, default=False)
    updated_by = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    updated_by_user = relationship("User")