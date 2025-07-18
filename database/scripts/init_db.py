#!/usr/bin/env python3
"""
Database initialization script for Facial Recognition SPA.
Creates tables, default roles, and admin user.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from backend.config.database import engine, SessionLocal, create_tables
from backend.models.database import Role, User, Employee
from backend.models.enums import UserRole, DEFAULT_ROLE_PERMISSIONS
from backend.core.security import get_password_hash
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_default_roles(db: Session):
    """Create default roles with permissions."""
    logger.info("Creating default roles...")
    
    for role_name, permissions in DEFAULT_ROLE_PERMISSIONS.items():
        # Check if role already exists
        existing_role = db.query(Role).filter(Role.role_name == role_name.value).first()
        
        if not existing_role:
            # Create permission dict
            permissions_dict = {perm.value: True for perm in permissions}
            
            # Create role
            role = Role(
                role_name=role_name.value,
                display_name=role_name.value.replace('_', ' ').title(),
                description=f"Default {role_name.value.replace('_', ' ')} role",
                permissions=permissions_dict
            )
            
            db.add(role)
            logger.info(f"Created role: {role_name.value}")
        else:
            logger.info(f"Role already exists: {role_name.value}")
    
    db.commit()


def create_admin_user(db: Session):
    """Create default admin user."""
    logger.info("Creating default admin user...")
    
    # Check if admin user already exists
    existing_admin = db.query(User).filter(User.username == "admin").first()
    
    if not existing_admin:
        # Get super admin role
        super_admin_role = db.query(Role).filter(Role.role_name == UserRole.SUPER_ADMIN.value).first()
        
        if not super_admin_role:
            logger.error("Super admin role not found. Please create roles first.")
            return
        
        # Create admin user
        admin_user = User(
            username="admin",
            email="admin@company.com",
            full_name="System Administrator",
            password_hash=get_password_hash("admin123"),  # Change this in production!
            is_active=True,
            is_superuser=True,
            role_id=super_admin_role.id
        )
        
        db.add(admin_user)
        db.commit()
        
        logger.info("Created default admin user:")
        logger.info("  Username: admin")
        logger.info("  Password: admin123")
        logger.warning("IMPORTANT: Change the default password in production!")
    else:
        logger.info("Admin user already exists")


def create_sample_employee(db: Session):
    """Create a sample employee for testing."""
    logger.info("Creating sample employee...")
    
    # Check if sample employee already exists
    existing_employee = db.query(Employee).filter(Employee.id == "EMP001").first()
    
    if not existing_employee:
        sample_employee = Employee(
            id="EMP001",
            employee_name="John Doe",
            department="Engineering",
            designation="Software Developer",
            email="john.doe@company.com",
            phone="+1-555-0123"
        )
        
        db.add(sample_employee)
        db.commit()
        
        logger.info("Created sample employee: EMP001 - John Doe")
    else:
        logger.info("Sample employee already exists")


def create_employee_user(db: Session):
    """Create a sample employee user for testing."""
    logger.info("Creating sample employee user...")
    
    # Check if employee user already exists
    existing_user = db.query(User).filter(User.username == "employee").first()
    
    if not existing_user:
        # Get employee role
        employee_role = db.query(Role).filter(Role.role_name == UserRole.EMPLOYEE.value).first()
        
        if not employee_role:
            logger.error("Employee role not found. Please create roles first.")
            return
        
        # Get sample employee
        sample_employee = db.query(Employee).filter(Employee.id == "EMP001").first()
        
        # Create employee user
        employee_user = User(
            username="employee",
            email="employee@company.com",
            full_name="John Doe",
            employee_id="EMP001" if sample_employee else None,
            password_hash=get_password_hash("employee123"),
            is_active=True,
            role_id=employee_role.id
        )
        
        db.add(employee_user)
        db.commit()
        
        logger.info("Created sample employee user:")
        logger.info("  Username: employee")
        logger.info("  Password: employee123")
    else:
        logger.info("Employee user already exists")


def main():
    """Main initialization function."""
    logger.info("Starting database initialization...")
    
    try:
        # Create tables
        logger.info("Creating database tables...")
        create_tables()
        
        # Create database session
        db = SessionLocal()
        
        try:
            # Create default roles
            create_default_roles(db)
            
            # Create admin user
            create_admin_user(db)
            
            # Create sample employee
            create_sample_employee(db)
            
            # Create employee user
            create_employee_user(db)
            
            logger.info("Database initialization completed successfully!")
            
            logger.info("\n" + "="*50)
            logger.info("DEFAULT CREDENTIALS:")
            logger.info("  Admin - Username: admin, Password: admin123")
            logger.info("  Employee - Username: employee, Password: employee123")
            logger.info("="*50)
            logger.warning("IMPORTANT: Change default passwords in production!")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()