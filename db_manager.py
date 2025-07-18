from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from db_config import get_db_session, close_db_session
from db_models import Employee, FaceEmbedding, AttendanceRecord, Role, TrackingRecord, SystemLog, User
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import threading
class DatabaseManager:
    def __init__(self):
        self.session_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    def get_session(self) -> Session:
        return get_db_session()
    def close_session(self, session: Session):
        close_db_session(session)
    def create_employee(self, employee_id: str, employee_name: str, department: str = None, designation: str = None, email: str = None, phone: str = None) -> bool:
        session = self.get_session()
        try:
            existing_employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if existing_employee:
                return False
            employee = Employee(
                id=employee_id,
                employee_name=employee_name,
                department=department,
                designation=designation,
                email=email,
                phone=phone)
            session.add(employee)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error creating employee {employee_id}: {e}")
            return False
        finally:
            self.close_session(session)
    def get_employee(self, employee_id: str) -> Optional[Employee]:
        session = self.get_session()
        try:
            return session.query(Employee).filter(Employee.id == employee_id).first()
        except Exception as e:
            self.logger.error(f"Error getting employee {employee_id}: {e}")
            return None
        finally:
            self.close_session(session)
    def get_all_employees(self) -> List[Employee]:
        session = self.get_session()
        try:
            return session.query(Employee).filter(Employee.is_active == True).all()
        except Exception as e:
            self.logger.error(f"Error getting all employees: {e}")
            return []
        finally:
            self.close_session(session)
    def store_face_embedding(self, employee_id: str, embedding: np.ndarray, embedding_type: str = 'enroll', quality_score: float = 0.0, source_image_path: str = None) -> bool:
        session = self.get_session()
        try:
            embedding_data = pickle.dumps(embedding)
            face_embedding = FaceEmbedding(
                employee_id=employee_id,
                embedding_data=embedding_data,
                embedding_type=embedding_type,
                quality_score=quality_score,
                source_image_path=source_image_path)
            session.add(face_embedding)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing face embedding for {employee_id}: {e}")
            return False
        finally:
            self.close_session(session)
    def get_face_embeddings(self, employee_id: str = None, embedding_type: str = None, limit: int = None) -> List[Tuple[str, np.ndarray]]:
        session = self.get_session()
        try:
            query = session.query(FaceEmbedding).filter(FaceEmbedding.is_active == True)
            if employee_id:
                query = query.filter(FaceEmbedding.employee_id == employee_id)
            if embedding_type:
                query = query.filter(FaceEmbedding.embedding_type == embedding_type)
            query = query.order_by(desc(FaceEmbedding.created_at))
            if limit:
                query = query.limit(limit)
            results = []
            for embedding_record in query.all():
                embedding_data = pickle.loads(embedding_record.embedding_data)
                results.append((embedding_record.employee_id, embedding_data))
            return results
        except Exception as e:
            self.logger.error(f"Error getting face embeddings: {e}")
            return []
        finally:
            self.close_session(session)
    def get_all_active_embeddings(self) -> Tuple[List[np.ndarray], List[str]]:
        session = self.get_session()
        try:
            embeddings = []
            labels = []
            enroll_embeddings = session.query(FaceEmbedding).filter(
                and_(FaceEmbedding.is_active == True, FaceEmbedding.embedding_type == 'enroll')
            ).all()
            for emb_record in enroll_embeddings:
                embedding_data = pickle.loads(emb_record.embedding_data)
                embeddings.append(embedding_data)
                labels.append(emb_record.employee_id)
            update_embeddings = session.query(FaceEmbedding).filter(
                and_(FaceEmbedding.is_active == True, FaceEmbedding.embedding_type == 'update')
            ).order_by(desc(FaceEmbedding.created_at)).all()
            employee_update_count = {}
            for emb_record in update_embeddings:
                emp_id = emb_record.employee_id
                if emp_id not in employee_update_count:
                    employee_update_count[emp_id] = 0
                if employee_update_count[emp_id] < 3:
                    embedding_data = pickle.loads(emb_record.embedding_data)
                    embeddings.append(embedding_data)
                    labels.append(emb_record.employee_id)
                    employee_update_count[emp_id] += 1
            return embeddings, labels
        except Exception as e:
            self.logger.error(f"Error getting all active embeddings: {e}")
            return [], []
        finally:
            self.close_session(session)
    def log_attendance(self, employee_id: str, camera_id: int, event_type: str, confidence_score: float = 0.0, work_status: str = 'working', notes: str = None) -> bool:
        session = self.get_session()
        try:
            attendance_record = AttendanceRecord(
                employee_id=employee_id,
                camera_id=camera_id,
                event_type=event_type,
                confidence_score=confidence_score,
                work_status=work_status,
                notes=notes)
            session.add(attendance_record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error logging attendance for {employee_id}: {e}")
            return False
        finally:
            self.close_session(session)
    def get_attendance_records(self, employee_id: str = None, start_date: datetime = None, end_date: datetime = None, limit: int = 100) -> List[AttendanceRecord]:
        session = self.get_session()
        try:
            query = session.query(AttendanceRecord).filter(AttendanceRecord.is_valid == True)
            if employee_id:
                query = query.filter(AttendanceRecord.employee_id == employee_id)
            if start_date:
                query = query.filter(AttendanceRecord.timestamp >= start_date)
            if end_date:
                query = query.filter(AttendanceRecord.timestamp <= end_date)
            query = query.order_by(desc(AttendanceRecord.timestamp))
            if limit:
                query = query.limit(limit)
            return query.all()
        except Exception as e:
            self.logger.error(f"Error getting attendance records: {e}")
            return []
        finally:
            self.close_session(session)
    def get_latest_attendance_by_employee(self, employee_id: str, hours_back: int = 10) -> Optional[AttendanceRecord]:
        session = self.get_session()
        try:
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            return session.query(AttendanceRecord).filter(
                and_(
                    AttendanceRecord.employee_id == employee_id,
                    AttendanceRecord.timestamp >= time_threshold,
                    AttendanceRecord.is_valid == True)
            ).order_by(desc(AttendanceRecord.timestamp)).first()
        except Exception as e:
            self.logger.error(f"Error getting latest attendance for {employee_id}: {e}")
            return None
        finally:
            self.close_session(session)
    def store_tracking_record(self, employee_id: str, camera_id: int, position_x: float, position_y: float, confidence_score: float, quality_metrics: dict = None) -> bool:
        session = self.get_session()
        try:
            tracking_record = TrackingRecord(
                employee_id=employee_id,
                camera_id=camera_id,
                position_x=position_x,
                position_y=position_y,
                confidence_score=confidence_score,
                quality_metrics=quality_metrics or {})
            session.add(tracking_record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing tracking record for {employee_id}: {e}")
            return False
        finally:
            self.close_session(session)
    def cleanup_old_embeddings(self, employee_id: str, max_embeddings: int = 25):
        session = self.get_session()
        try:
            update_embeddings = session.query(FaceEmbedding).filter(
                and_(
                    FaceEmbedding.employee_id == employee_id,
                    FaceEmbedding.embedding_type == 'update',
                    FaceEmbedding.is_active == True
                )
            ).order_by(desc(FaceEmbedding.created_at)).all()
            if len(update_embeddings) > max_embeddings:
                embeddings_to_deactivate = update_embeddings[max_embeddings:]
                for embedding in embeddings_to_deactivate:
                    embedding.is_active = False
                session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error cleaning up old embeddings for {employee_id}: {e}")
        finally:
            self.close_session(session)
    def log_system_event(self, message: str, log_level: str = 'INFO', component: str = None, employee_id: str = None, camera_id: int = None, additional_data: dict = None) -> bool:
        session = self.get_session()
        try:
            system_log = SystemLog(
                log_level=log_level,
                message=message,
                component=component,
                employee_id=employee_id,
                camera_id=camera_id,
                additional_data=additional_data or {})
            session.add(system_log)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error logging system event: {e}")
            return False
        finally:
            self.close_session(session)
    def get_employee_work_status(self, employee_id: str) -> bool:
        latest_record = self.get_latest_attendance_by_employee(employee_id)
        if latest_record:
            return latest_record.event_type == 'check_in'
        return False
    def create_role(self, role_name: str, permissions: dict = None) -> bool:
        session = self.get_session()
        try:
            existing = session.query(Role).filter_by(role_name=role_name).first()
            if existing:
                return False
            role = Role(
                role_name=role_name,
                permissions=permissions or {}
            )
            session.add(role)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error creating role {role_name}: {e}")
            return False
        finally:
            self.close_session(session)
    def get_role(self, role_name: str) -> Role:
        session = self.get_session()
        try:
            return session.query(Role).filter_by(role_name=role_name).first()
        except Exception as e:
            self.logger.error(f"Error fetching role {role_name}: {e}")
            return None
        finally:
            self.close_session(session)
    def create_user(self, username: str, password_hash: str, role_name: str) -> bool:
        session = self.get_session()
        try:
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                return False
            role = session.query(Role).filter_by(role_name=role_name).first()
            if not role:
                self.logger.error(f"Role {role_name} does not exist")
                return False
            user = User(
                username=username,
                password_hash=password_hash,
                role_id=role.id
            )
            session.add(user)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error creating user {username}: {e}")
            return False
        finally:
            self.close_session(session)
    def get_user(self, username: str) -> User:
        session = self.get_session()
        try:
            return session.query(User).filter_by(username=username).first()
        except Exception as e:
            self.logger.error(f"Error fetching user {username}: {e}")
            return None
        finally:
            self.close_session(session)