import os
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Union, Optional
from insightface.app import FaceAnalysis
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.db_manager import DatabaseManager
from db.db_models import FaceEmbedding
from core.fts_system import FaceTrackingPipeline
class FaceEnrollmentError(Exception):
    pass
class EmployeeNotFoundError(FaceEnrollmentError):
    pass
class DatabaseOperationError(FaceEnrollmentError):
    pass
class ImageProcessingError(FaceEnrollmentError):
    pass
class FaceEnroller:
    ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg')
    def __init__(self, tracking_system=None):
        self.db_manager = DatabaseManager()
        self.tracking_system = tracking_system
        self.face_app = FaceAnalysis(name='antelopev2',
                                     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(416, 416))
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self._batch_mode = False
    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        return isinstance(embedding, np.ndarray) and embedding.dtype == np.float32 and len(embedding.shape) == 1
    def _validate_quality_score(self, score: float) -> bool:
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0
    def set_batch_mode(self, enabled: bool):
        self._batch_mode = enabled
    def enroll_from_images(self, employee_id: str,
                           employee_name: str,
                           image_paths: Union[List[str], str],
                           min_faces: int = 3,
                           update_existing: bool = False,
                           rebuild_index: bool = True) -> bool:
        if not employee_id or not employee_name:
            self.logger.error("Employee ID and name cannot be empty")
            raise ValueError("Employee ID and name cannot be empty")
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                image_paths = [
                    os.path.join(image_paths, f)
                    for f in os.listdir(image_paths)
                    if f.lower().endswith(self.ALLOWED_EXTENSIONS)]
            else:
                image_paths = [image_paths]
        if not image_paths:
            self.logger.error("No valid image files provided")
            raise ValueError("No valid image files provided")
        existing_employee = self.db_manager.get_employee(employee_id)
        if existing_employee:
            if not update_existing:
                self.logger.error(f"Employee {employee_id} already exists (use update_existing=True)")
                raise ValueError(f"Employee {employee_id} already exists")
            self.logger.info(f"Updating existing employee {employee_name} ({employee_id})")
        else:
            created = self.db_manager.create_employee(employee_id, employee_name)
            if not created:
                self.logger.error(f"Error creating employee {employee_id} in database")
                raise DatabaseOperationError(f"Failed to create employee {employee_id}")
            self.logger.info(f"Created new employee {employee_name} ({employee_id}) in database")
        valid_count = 0
        for img_path in image_paths:
            if not os.path.exists(img_path):
                self.logger.warning(f"Image not found - {img_path}")
                continue
            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.logger.warning(f"Could not read image - {img_path}")
                    continue
                faces = self.face_app.get(img)
                if len(faces) != 1:
                    self.logger.warning(f"Found {len(faces)} faces in {img_path} (expected 1)")
                    continue
                face = faces[0]
                if not self._validate_embedding(face.embedding):
                    self.logger.error(f"Invalid embedding format from {img_path}")
                    continue
                if not self._validate_quality_score(face.det_score):
                    self.logger.warning(f"Invalid quality score from {img_path}, using default")
                    face.det_score = 0.5
                stored = self.db_manager.store_face_embedding(
                    employee_id,
                    face.embedding,
                    embedding_type='enroll' if not update_existing else 'update',
                    quality_score=face.det_score,
                    source_image_path=img_path)
                if not stored:
                    self.logger.error(f"Error storing embedding for {employee_id} from {img_path}")
                    continue
                valid_count += 1
                self.logger.info(f"Processed {img_path} - Face detected and embedding stored in DB")
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {str(e)}")
                continue
        if valid_count >= min_faces:
            action = "Updated" if update_existing else "Enrolled"
            self.logger.info(f"{action} {employee_name} ({employee_id}) with {valid_count} images")
            if rebuild_index and not self._batch_mode and self.tracking_system:
                self.tracking_system.reload_embeddings_and_rebuild_index()
            return True
        else:
            self.logger.error(f"Only {valid_count} valid faces found (minimum {min_faces} required)")
            raise ValueError(f"Insufficient valid faces: {valid_count} < {min_faces}")
    def add_embedding(self, employee_id: str, image_path: str, rebuild_index: bool = True) -> bool:
        existing_employee = self.db_manager.get_employee(employee_id)
        if not existing_employee:
            self.logger.error(f"Employee {employee_id} not found")
            raise EmployeeNotFoundError(f"Employee {employee_id} not found")
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found - {image_path}")
            raise FileNotFoundError(f"Image not found - {image_path}")
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Could not read image - {image_path}")
                raise ImageProcessingError(f"Could not read image - {image_path}")
            faces = self.face_app.get(img)
            if len(faces) != 1:
                self.logger.error(f"Found {len(faces)} faces in image (expected 1)")
                raise ImageProcessingError(f"Expected 1 face, found {len(faces)}")
            face = faces[0]
            if not self._validate_embedding(face.embedding):
                raise ImageProcessingError(f"Invalid embedding format from {image_path}")
            if not self._validate_quality_score(face.det_score):
                self.logger.warning(f"Invalid quality score from {image_path}, using default")
                face.det_score = 0.5
            stored = self.db_manager.store_face_embedding(
                employee_id,
                face.embedding,
                embedding_type='update',
                quality_score=face.det_score,
                source_image_path=image_path
            )
            if stored:
                self.logger.info(f"Added new embedding for {employee_id} from {image_path}")
                if rebuild_index and not self._batch_mode and self.tracking_system:
                    self.tracking_system.reload_embeddings_and_rebuild_index()
                return True
            else:
                self.logger.error(f"Error storing embedding for {employee_id} from {image_path}")
                raise DatabaseOperationError(f"Failed to store embedding for {employee_id}")
        except (ImageProcessingError, DatabaseOperationError):
            raise
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise ImageProcessingError(f"Error processing image: {str(e)}")
    def update_embeddings(self, employee_id: str, image_paths: List[str], rebuild_index: bool = True) -> bool:
        self.set_batch_mode(True)
        try:
            if not self.remove_all_embeddings(employee_id, rebuild_index=False):
                return False
            success = self.enroll_from_images(
                employee_id,
                self.db_manager.get_employee(employee_id).employee_name,
                image_paths,
                update_existing=True,
                rebuild_index=False
            )
            if success and rebuild_index and self.tracking_system:
                self.tracking_system.reload_embeddings_and_rebuild_index()
            return success
        finally:
            self.set_batch_mode(False)
    def delete_employee_embedding(self, embedding_id: int, rebuild_index: bool = True) -> bool:
        try:
            success = self.db_manager.remove_embedding(embedding_id)
            if success:
                self.logger.info(f"Deleted embedding ID {embedding_id}")
                if rebuild_index and not self._batch_mode and self.tracking_system:
                    self.tracking_system.reload_embeddings_and_rebuild_index()
            else:
                self.logger.error(f"Error deleting embedding ID {embedding_id}")
                raise DatabaseOperationError(f"Failed to delete embedding ID {embedding_id}")
            return success
        except DatabaseOperationError:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting embedding: {str(e)}")
            raise DatabaseOperationError(f"Error deleting embedding: {str(e)}")
    def remove_all_embeddings(self, employee_id: str, rebuild_index: bool = True) -> bool:
        try:
            success = self.db_manager.delete_embeddings(employee_id)
            if success:
                self.logger.info(f"Deleted all embeddings for {employee_id}")
                if rebuild_index and not self._batch_mode and self.tracking_system:
                    self.tracking_system.reload_embeddings_and_rebuild_index()
                return True
            else:
                self.logger.error(f"Error deleting embeddings for {employee_id}")
                raise DatabaseOperationError(f"Failed to delete embeddings for {employee_id}")
        except DatabaseOperationError:
            raise
        except Exception as e:
            self.logger.error(f"Error removing embeddings: {str(e)}")
            raise DatabaseOperationError(f"Error removing embeddings: {str(e)}")
    def archive_all_embeddings(self, employee_id: str, rebuild_index: bool = True) -> bool:
        try:
            success = self.db_manager.archive_embeddings(employee_id)
            if success:
                self.logger.info(f"Archived all embeddings for {employee_id}")
                if rebuild_index and not self._batch_mode and self.tracking_system:
                    self.tracking_system.reload_embeddings_and_rebuild_index()
                return True
            else:
                self.logger.error(f"Error archiving embeddings for {employee_id}")
                raise DatabaseOperationError(f"Failed to archive embeddings for {employee_id}")
        except DatabaseOperationError:
            raise
        except Exception as e:
            self.logger.error(f"Error archiving embeddings: {str(e)}")
            raise DatabaseOperationError(f"Error archiving embeddings: {str(e)}")
    def delete_employee(self, employee_id: str, rebuild_index: bool = True) -> bool:
        try:
            success = self.db_manager.delete_employee(employee_id)
            if success:
                self.logger.info(f"Deleted employee {employee_id} from database")
                if rebuild_index and not self._batch_mode and self.tracking_system:
                    self.tracking_system.reload_embeddings_and_rebuild_index()
            else:
                self.logger.error(f"Error deleting employee {employee_id} from database")
                raise DatabaseOperationError(f"Failed to delete employee {employee_id}")
            return success
        except DatabaseOperationError:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting employee: {str(e)}")
            raise DatabaseOperationError(f"Error deleting employee: {str(e)}")
if __name__ == "__main__":
    enroller = FaceEnroller()
    emp_id = input("Enter employee ID: ").strip()
    emp_name = input("Enter employee name: ").strip()
    img_dir = input("Enter image directory path: ").strip()
    enroller.enroll_from_images(emp_id, emp_name, img_dir)