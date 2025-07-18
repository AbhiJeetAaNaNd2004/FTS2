import cv2
import os
import numpy as np
import faiss
import torch
import time
import threading
import csv
import pickle
import sys
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from insightface.app import FaceAnalysis
import requests
import json
from datetime import datetime, timedelta
import queue
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from concurrent.futures import ThreadPoolExecutor
import websockets

from backend.config.database import get_db
from backend.models.database import Employee, FaceEmbedding, AttendanceRecord, TrackingRecord
from backend.config.settings import get_settings

settings = get_settings()


@dataclass
class TripwireConfig:
    position: float
    spacing: float
    direction: str
    name: str


@dataclass
class CameraConfig:
    camera_id: int
    gpu_id: int
    camera_type: str
    tripwires: List[TripwireConfig]
    resolution: tuple
    fps: int


@dataclass
class GlobalTrack:
    employee_id: str
    last_seen_time: float
    last_camera_id: int
    embedding_history: deque
    confidence_score: float = 0.0
    work_status: str = "working"


@dataclass
class EmployeeMetadata:
    employee_id: str
    employee_name: str
    enrollment_date: str
    embedding_count: int
    source_images: List[str]


@dataclass
class FaceQualityMetrics:
    sharpness_score: float
    brightness_score: float
    angle_score: float
    size_score: float
    overall_quality: float


@dataclass
class TrackingState:
    position_history: List[Tuple[int, int]]
    velocity: Tuple[float, float]
    predicted_position: Tuple[int, int]
    confidence_history: List[float]
    quality_history: List[FaceQualityMetrics]


class DatabaseManager:
    """Enhanced database manager for the web application."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_all_active_embeddings(self) -> Tuple[List[np.ndarray], List[str]]:
        """Get all active embeddings from database."""
        try:
            db = next(get_db())
            embeddings = []
            labels = []
            
            # Get enrollment embeddings
            enroll_embeddings = db.query(FaceEmbedding).filter(
                FaceEmbedding.is_active == True,
                FaceEmbedding.embedding_type == 'enroll'
            ).all()
            
            for emb_record in enroll_embeddings:
                embedding_data = pickle.loads(emb_record.embedding_data)
                embeddings.append(embedding_data)
                labels.append(emb_record.employee_id)
            
            # Get recent update embeddings (max 3 per employee)
            update_embeddings = db.query(FaceEmbedding).filter(
                FaceEmbedding.is_active == True,
                FaceEmbedding.embedding_type == 'update'
            ).order_by(FaceEmbedding.created_at.desc()).all()
            
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
            db.close()
    
    def store_face_embedding(self, employee_id: str, embedding: np.ndarray, 
                           embedding_type: str = 'enroll', quality_score: float = 0.0, 
                           source_image_path: str = None) -> bool:
        """Store face embedding in database."""
        try:
            db = next(get_db())
            embedding_data = pickle.dumps(embedding)
            
            face_embedding = FaceEmbedding(
                employee_id=employee_id,
                embedding_data=embedding_data,
                embedding_type=embedding_type,
                quality_score=quality_score,
                source_image_path=source_image_path
            )
            
            db.add(face_embedding)
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing face embedding for {employee_id}: {e}")
            return False
        finally:
            db.close()
    
    def log_attendance(self, employee_id: str, camera_id: int, event_type: str, 
                      confidence_score: float = 0.0, work_status: str = 'working', 
                      notes: str = None) -> bool:
        """Log attendance record."""
        try:
            db = next(get_db())
            attendance_record = AttendanceRecord(
                employee_id=employee_id,
                camera_id=camera_id,
                event_type=event_type,
                confidence_score=confidence_score,
                work_status=work_status,
                notes=notes
            )
            
            db.add(attendance_record)
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error logging attendance for {employee_id}: {e}")
            return False
        finally:
            db.close()
    
    def get_latest_attendance_by_employee(self, employee_id: str, hours_back: int = 10) -> Optional[AttendanceRecord]:
        """Get latest attendance record for employee."""
        try:
            db = next(get_db())
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            
            return db.query(AttendanceRecord).filter(
                AttendanceRecord.employee_id == employee_id,
                AttendanceRecord.timestamp >= time_threshold,
                AttendanceRecord.is_valid == True
            ).order_by(AttendanceRecord.timestamp.desc()).first()
            
        except Exception as e:
            self.logger.error(f"Error getting latest attendance for {employee_id}: {e}")
            return None
        finally:
            db.close()
    
    def store_tracking_record(self, employee_id: str, camera_id: int, position_x: float, 
                            position_y: float, confidence_score: float, quality_metrics: dict = None) -> bool:
        """Store tracking record."""
        try:
            db = next(get_db())
            tracking_record = TrackingRecord(
                employee_id=employee_id,
                camera_id=camera_id,
                position_x=position_x,
                position_y=position_y,
                confidence_score=confidence_score,
                quality_metrics=quality_metrics or {}
            )
            
            db.add(tracking_record)
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error storing tracking record for {employee_id}: {e}")
            return False
        finally:
            db.close()
    
    def cleanup_old_embeddings(self, employee_id: str, max_embeddings: int = 25):
        """Cleanup old embeddings for employee."""
        try:
            db = next(get_db())
            update_embeddings = db.query(FaceEmbedding).filter(
                FaceEmbedding.employee_id == employee_id,
                FaceEmbedding.embedding_type == 'update',
                FaceEmbedding.is_active == True
            ).order_by(FaceEmbedding.created_at.desc()).all()
            
            if len(update_embeddings) > max_embeddings:
                embeddings_to_deactivate = update_embeddings[max_embeddings:]
                for embedding in embeddings_to_deactivate:
                    embedding.is_active = False
                db.commit()
                
        except Exception as e:
            db.rollback()
            self.logger.error(f"Error cleaning up old embeddings for {employee_id}: {e}")
        finally:
            db.close()
    
    def get_employee_work_status(self, employee_id: str) -> bool:
        """Check if employee is currently working."""
        latest_record = self.get_latest_attendance_by_employee(employee_id)
        if latest_record:
            return latest_record.event_type == 'check_in'
        return False
    
    def get_employee_name(self, employee_id: str) -> str:
        """Get employee name by ID."""
        try:
            db = next(get_db())
            employee = db.query(Employee).filter(Employee.id == employee_id).first()
            if employee:
                return employee.employee_name
            return employee_id
        except Exception as e:
            self.logger.error(f"Error getting employee name for {employee_id}: {e}")
            return employee_id
        finally:
            db.close()


class WebSocketManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        self.connections: Dict[str, set] = {
            'camera_feed': set(),
            'attendance_updates': set(),
            'tracking_updates': set()
        }
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket, connection_type: str):
        """Add a new WebSocket connection."""
        if connection_type in self.connections:
            self.connections[connection_type].add(websocket)
            self.logger.info(f"New {connection_type} connection: {websocket.remote_address}")
    
    async def disconnect(self, websocket, connection_type: str):
        """Remove a WebSocket connection."""
        if connection_type in self.connections:
            self.connections[connection_type].discard(websocket)
            self.logger.info(f"Disconnected {connection_type}: {websocket.remote_address}")
    
    async def broadcast(self, message: dict, connection_type: str):
        """Broadcast message to all connections of a specific type."""
        if connection_type not in self.connections:
            return
        
        disconnected = set()
        for websocket in self.connections[connection_type].copy():
            try:
                await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                self.logger.error(f"Error sending message to {websocket.remote_address}: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected connections
        for websocket in disconnected:
            self.connections[connection_type].discard(websocket)


class FaceTrackingSystem:
    """Core face tracking system with web integration."""
    
    def __init__(self):
        # Core tracking data
        self.embeddings = []
        self.labels = []
        self.employee_metadata = {}
        self.index = None
        self.apps = {}
        self.trackers = {}
        self.global_tracks = {}
        self.track_identities = {}
        self.track_lifetimes = {}
        self.track_positions = {}
        self.last_embedding_update = {}
        
        # Frame and detection management
        self.frame_locks = {}
        self.latest_frames = {}
        self.latest_faces = {}
        self.face_detection_threads = {}
        self.embedding_cache = {}
        
        # Tracking state management
        self.next_global_track_id = 1
        self.last_faces_reload = time.time()
        self.faces_reload_interval = 30
        self.frame_skip_counter = {}
        self.detection_interval = {}
        self.identity_tracks = {}
        self.identity_last_seen = {}
        self.identity_cameras = {}
        self.identity_positions = {}
        self.identity_trip_logged = {}
        self.identity_crossing_state = {}
        self.identity_zone_state = {}
        self.kalman_trackers = {}
        self.tracking_states = {}
        
        # System management
        self.shutdown_flag = threading.Event()
        self.camera_threads = []
        self.embedding_update_worker = None
        self.embedding_update_queue = queue.Queue()
        
        # Locks for thread safety
        self.global_tracks_lock = threading.RLock()
        self.embedding_update_lock = threading.RLock()
        self.identity_tracks_lock = threading.RLock()
        self.embedding_cache_lock = threading.RLock()
        self.faiss_index_lock = threading.RLock()
        self.metadata_lock = threading.RLock()
        
        # Configuration from settings
        self.batch_update_threshold = 5
        self.updates_since_last_rebuild = 0
        self.max_updates_before_rebuild = 20
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # WebSocket manager for real-time updates
        self.websocket_manager = WebSocketManager()
        
        # Initialize system
        self.logger = logging.getLogger(__name__)
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the face tracking system."""
        try:
            # Start embedding update worker
            self.embedding_update_worker = threading.Thread(
                target=self._embedding_update_worker, daemon=True
            )
            self.embedding_update_worker.start()
            
            # Load known faces and metadata
            self._load_known_faces()
            self._load_employee_metadata()
            
            # Initialize FAISS index
            self._initialize_faiss()
            
            # Initialize multi-GPU InsightFace
            self._initialize_multi_gpu_insightface()
            
            # Initialize cameras
            self._initialize_cameras()
            
            self.logger.info("Face tracking system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing face tracking system: {e}")
            raise e
    
    def _load_known_faces(self):
        """Load known faces from database."""
        try:
            embeddings_list, labels_list = self.db_manager.get_all_active_embeddings()
            if embeddings_list:
                self.embeddings = np.array(embeddings_list).astype('float32')
                self.labels = labels_list
                faiss.normalize_L2(self.embeddings)
            
            unique_employees = len(set(labels_list)) if labels_list else 0
            self.logger.info(f"Loaded {unique_employees} employees with {len(embeddings_list)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to load known faces: {e}")
            self.embeddings = []
            self.labels = []
    
    def _load_employee_metadata(self):
        """Load employee metadata from database."""
        try:
            db = next(get_db())
            employees = db.query(Employee).filter(Employee.is_active == True).all()
            
            for employee in employees:
                self.employee_metadata[employee.id] = {
                    'employee_name': employee.employee_name,
                    'department': employee.department,
                    'designation': employee.designation,
                    'email': employee.email,
                    'phone': employee.phone
                }
            
            self.logger.info(f"Loaded metadata for {len(employees)} employees")
            
        except Exception as e:
            self.logger.error(f"Failed to load employee metadata: {e}")
        finally:
            db.close()
    
    def _initialize_faiss(self):
        """Initialize FAISS index for fast similarity search."""
        try:
            if len(self.embeddings) > 0:
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(self.embeddings)
                self.logger.info(f"FAISS index initialized with {len(self.embeddings)} embeddings")
            else:
                self.logger.warning("No embeddings available for FAISS index")
                
        except Exception as e:
            self.logger.error(f"Error initializing FAISS index: {e}")
    
    def _initialize_multi_gpu_insightface(self):
        """Initialize InsightFace for each GPU."""
        try:
            # Get available cameras and their GPU assignments
            cameras = self._get_camera_configs()
            gpu_ids = set()
            
            for camera in cameras:
                gpu_ids.add(camera.gpu_id)
            
            # Initialize InsightFace app for each GPU
            for gpu_id in gpu_ids:
                try:
                    app = FaceAnalysis(
                        name='antelopev2',
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    app.prepare(ctx_id=gpu_id, det_size=(416, 416))
                    self.apps[gpu_id] = app
                    self.logger.info(f"InsightFace initialized for GPU {gpu_id}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize InsightFace for GPU {gpu_id}: {e}")
                    # Fallback to CPU
                    app = FaceAnalysis(
                        name='antelopev2',
                        providers=['CPUExecutionProvider']
                    )
                    app.prepare(ctx_id=-1, det_size=(416, 416))
                    self.apps[gpu_id] = app
                    self.logger.info(f"InsightFace initialized for GPU {gpu_id} (CPU fallback)")
                    
        except Exception as e:
            self.logger.error(f"Error initializing multi-GPU InsightFace: {e}")
    
    def _get_camera_configs(self) -> List[CameraConfig]:
        """Get camera configurations from database or default config."""
        # For now, return default configuration
        # This should be loaded from database in production
        return [
            CameraConfig(
                camera_id=0,
                gpu_id=0,
                camera_type="entry",
                tripwires=[
                    TripwireConfig(position=0.755551, spacing=0.01, direction="horizontal", name="EntryDetection")
                ],
                resolution=(1280, 720),
                fps=15
            ),
            CameraConfig(
                camera_id=1,
                gpu_id=0,
                camera_type="exit",
                tripwires=[
                    TripwireConfig(position=0.5, spacing=0.01, direction="vertical", name="EntryDetection")
                ],
                resolution=(1280, 720),
                fps=15
            )
        ]
    
    def _initialize_cameras(self):
        """Initialize camera tracking threads."""
        try:
            cameras = self._get_camera_configs()
            
            for camera_config in cameras:
                self.frame_locks[camera_config.camera_id] = threading.Lock()
                self.latest_frames[camera_config.camera_id] = None
                self.latest_faces[camera_config.camera_id] = []
                self.frame_skip_counter[camera_config.camera_id] = 0
                self.detection_interval[camera_config.camera_id] = 3
            
            self.logger.info(f"Initialized {len(cameras)} cameras")
            
        except Exception as e:
            self.logger.error(f"Error initializing cameras: {e}")
    
    def _embedding_update_worker(self):
        """Worker thread for processing embedding updates."""
        pending_updates = []
        
        while not self.shutdown_flag.is_set():
            try:
                # Collect updates for batch processing
                try:
                    update = self.embedding_update_queue.get(timeout=1.0)
                    if update is None:  # Shutdown signal
                        break
                    pending_updates.append(update)
                except queue.Empty:
                    pass
                
                # Process batch if threshold reached or timeout
                if (len(pending_updates) >= self.batch_update_threshold or 
                    (pending_updates and time.time() - pending_updates[0][2] > 5.0)):
                    
                    self._process_pending_updates(pending_updates)
                    pending_updates.clear()
                    
            except Exception as e:
                self.logger.error(f"Error in embedding update worker: {e}")
    
    def _process_pending_updates(self, pending_updates):
        """Process batch of pending embedding updates."""
        with self.embedding_update_lock:
            new_embeddings = []
            new_labels = []
            
            for identity, embedding, timestamp in pending_updates:
                # Store in database
                success = self.db_manager.store_face_embedding(
                    employee_id=identity,
                    embedding=embedding,
                    embedding_type='update',
                    quality_score=0.0,
                    source_image_path=None
                )
                
                if success:
                    new_embeddings.append(embedding)
                    new_labels.append(identity)
                    # Cleanup old embeddings
                    self.db_manager.cleanup_old_embeddings(identity, max_embeddings=15)
            
            # Update FAISS index if we have new embeddings
            if new_embeddings:
                self._rebuild_faiss_index()
                self.logger.info(f"Processed {len(new_embeddings)} embedding updates")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index with latest embeddings."""
        try:
            with self.faiss_index_lock:
                embeddings_list, labels_list = self.db_manager.get_all_active_embeddings()
                
                if embeddings_list:
                    self.embeddings = np.array(embeddings_list).astype('float32')
                    self.labels = labels_list
                    faiss.normalize_L2(self.embeddings)
                    
                    # Rebuild FAISS index
                    dimension = self.embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dimension)
                    self.index.add(self.embeddings)
                    
                    self.updates_since_last_rebuild = 0
                    self.logger.info(f"FAISS index rebuilt with {len(embeddings_list)} embeddings")
                    
        except Exception as e:
            self.logger.error(f"Error rebuilding FAISS index: {e}")
    
    def get_employee_name(self, employee_id: str) -> str:
        """Get employee name with caching."""
        with self.metadata_lock:
            if employee_id in self.employee_metadata:
                return self.employee_metadata[employee_id]['employee_name']
            
            # Fallback to database
            return self.db_manager.get_employee_name(employee_id)
    
    def identify_face(self, face_embedding: np.ndarray, threshold: float = None) -> Tuple[Optional[str], float]:
        """Identify face using FAISS index."""
        if threshold is None:
            threshold = settings.face_match_threshold
        
        try:
            with self.faiss_index_lock:
                if self.index is None or len(self.embeddings) == 0:
                    return None, 0.0
                
                # Normalize embedding
                embedding_norm = np.linalg.norm(face_embedding)
                if embedding_norm > 0:
                    face_embedding = face_embedding / embedding_norm
                else:
                    return None, 0.0
                
                # Search in FAISS index
                face_embedding = face_embedding.reshape(1, -1).astype('float32')
                scores, indices = self.index.search(face_embedding, 1)
                
                if len(scores[0]) > 0 and scores[0][0] > threshold:
                    best_idx = indices[0][0]
                    confidence = scores[0][0]
                    employee_id = self.labels[best_idx]
                    return employee_id, confidence
                
                return None, 0.0
                
        except Exception as e:
            self.logger.error(f"Error in face identification: {e}")
            return None, 0.0
    
    def reload_embeddings_and_rebuild_index(self):
        """Reload embeddings from database and rebuild FAISS index."""
        try:
            self._load_known_faces()
            self._load_employee_metadata()
            self._rebuild_faiss_index()
            self.logger.info("Embeddings and index reloaded successfully")
        except Exception as e:
            self.logger.error(f"Error reloading embeddings: {e}")
    
    def get_current_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Get current frame from camera."""
        with self.frame_locks.get(camera_id, threading.Lock()):
            return self.latest_frames.get(camera_id)
    
    def start_multi_camera_tracking(self):
        """Start tracking for all configured cameras."""
        try:
            cameras = self._get_camera_configs()
            
            for camera_config in cameras:
                thread = threading.Thread(
                    target=self.process_camera,
                    args=(camera_config,),
                    daemon=True
                )
                thread.start()
                self.camera_threads.append(thread)
                time.sleep(0.5)  # Stagger camera starts
            
            self.logger.info(f"Started tracking for {len(cameras)} cameras")
            
        except Exception as e:
            self.logger.error(f"Error starting multi-camera tracking: {e}")
    
    def process_camera(self, camera_config: CameraConfig):
        """Process individual camera feed."""
        camera_id = camera_config.camera_id
        
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera {camera_id}")
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, camera_config.fps)
            
            self.logger.info(f"Camera {camera_id} started successfully")
            
            while not self.shutdown_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Update latest frame
                with self.frame_locks[camera_id]:
                    self.latest_frames[camera_id] = frame.copy()
                
                # Process frame for face detection and tracking
                self._process_frame(frame, camera_config)
                
                # Control frame rate
                time.sleep(1.0 / camera_config.fps)
                
        except Exception as e:
            self.logger.error(f"Error processing camera {camera_id}: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
    
    def _process_frame(self, frame: np.ndarray, camera_config: CameraConfig):
        """Process individual frame for face detection and tracking."""
        camera_id = camera_config.camera_id
        
        try:
            # Skip frames for performance
            self.frame_skip_counter[camera_id] += 1
            if self.frame_skip_counter[camera_id] % self.detection_interval[camera_id] != 0:
                return
            
            # Get InsightFace app for this camera's GPU
            app = self.apps.get(camera_config.gpu_id)
            if app is None:
                return
            
            # Detect faces
            faces = app.get(frame)
            
            # Update latest faces
            with self.frame_locks[camera_id]:
                self.latest_faces[camera_id] = faces
            
            # Process each detected face
            for face in faces:
                if face.det_score < settings.face_detection_threshold:
                    continue
                
                # Identify face
                employee_id, confidence = self.identify_face(face.embedding)
                
                if employee_id and confidence > settings.face_match_threshold:
                    # Update tracking and attendance
                    self._update_tracking(employee_id, camera_id, face, confidence)
                    
                    # Broadcast real-time update
                    asyncio.create_task(self._broadcast_tracking_update(
                        employee_id, camera_id, confidence
                    ))
                
        except Exception as e:
            self.logger.error(f"Error processing frame for camera {camera_id}: {e}")
    
    def _update_tracking(self, employee_id: str, camera_id: int, face, confidence: float):
        """Update tracking information for identified employee."""
        current_time = time.time()
        
        try:
            # Update global tracking
            with self.global_tracks_lock:
                if employee_id not in self.global_tracks:
                    self.global_tracks[employee_id] = GlobalTrack(
                        employee_id=employee_id,
                        last_seen_time=current_time,
                        last_camera_id=camera_id,
                        embedding_history=deque(maxlen=settings.embedding_history_size),
                        confidence_score=confidence
                    )
                
                track = self.global_tracks[employee_id]
                track.last_seen_time = current_time
                track.last_camera_id = camera_id
                track.confidence_score = max(track.confidence_score, confidence)
                track.embedding_history.append(face.embedding)
            
            # Check for attendance logging
            self._check_attendance_logging(employee_id, camera_id, confidence)
            
            # Store tracking record
            bbox = face.bbox
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            self.db_manager.store_tracking_record(
                employee_id=employee_id,
                camera_id=camera_id,
                position_x=center_x,
                position_y=center_y,
                confidence_score=confidence
            )
            
            # Update embeddings if needed
            self._update_embeddings(employee_id, face.embedding)
            
        except Exception as e:
            self.logger.error(f"Error updating tracking for {employee_id}: {e}")
    
    def _check_attendance_logging(self, employee_id: str, camera_id: int, confidence: float):
        """Check if attendance should be logged based on camera type and employee status."""
        try:
            # Get camera configuration
            cameras = self._get_camera_configs()
            camera_config = next((c for c in cameras if c.camera_id == camera_id), None)
            
            if not camera_config:
                return
            
            # Check if this is an entry/exit camera
            if camera_config.camera_type not in ['entry', 'exit']:
                return
            
            # Check if employee has recent attendance log
            latest_record = self.db_manager.get_latest_attendance_by_employee(employee_id, hours_back=1)
            
            # Determine event type based on camera type and current status
            event_type = None
            
            if camera_config.camera_type == 'entry':
                if not latest_record or latest_record.event_type == 'check_out':
                    event_type = 'check_in'
            elif camera_config.camera_type == 'exit':
                if latest_record and latest_record.event_type == 'check_in':
                    event_type = 'check_out'
            
            # Log attendance if appropriate
            if event_type:
                success = self.db_manager.log_attendance(
                    employee_id=employee_id,
                    camera_id=camera_id,
                    event_type=event_type,
                    confidence_score=confidence
                )
                
                if success:
                    employee_name = self.get_employee_name(employee_id)
                    self.logger.info(f"Logged {event_type} for {employee_name} ({employee_id}) at camera {camera_id}")
                    
                    # Broadcast attendance update
                    asyncio.create_task(self._broadcast_attendance_update(
                        employee_id, employee_name, event_type, camera_id
                    ))
                
        except Exception as e:
            self.logger.error(f"Error checking attendance logging: {e}")
    
    def _update_embeddings(self, identity: str, embedding: np.ndarray):
        """Update embeddings for continuous learning."""
        current_time = time.time()
        
        with self.embedding_update_lock:
            if identity in self.last_embedding_update:
                if current_time - self.last_embedding_update[identity] < settings.max_embedding_update_cooldown:
                    return False
            
            # Normalize embedding
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                embedding = embedding / emb_norm
            else:
                return False
            
            try:
                self.embedding_update_queue.put((identity, embedding, current_time), timeout=0.1)
                self.last_embedding_update[identity] = current_time
                return True
            except queue.Full:
                return False
    
    async def _broadcast_tracking_update(self, employee_id: str, camera_id: int, confidence: float):
        """Broadcast tracking update via WebSocket."""
        try:
            message = {
                'type': 'tracking_update',
                'employee_id': employee_id,
                'employee_name': self.get_employee_name(employee_id),
                'camera_id': camera_id,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.websocket_manager.broadcast(message, 'tracking_updates')
            
        except Exception as e:
            self.logger.error(f"Error broadcasting tracking update: {e}")
    
    async def _broadcast_attendance_update(self, employee_id: str, employee_name: str, 
                                         event_type: str, camera_id: int):
        """Broadcast attendance update via WebSocket."""
        try:
            message = {
                'type': 'attendance_update',
                'employee_id': employee_id,
                'employee_name': employee_name,
                'event_type': event_type,
                'camera_id': camera_id,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.websocket_manager.broadcast(message, 'attendance_updates')
            
        except Exception as e:
            self.logger.error(f"Error broadcasting attendance update: {e}")
    
    def get_present_employees(self) -> List[Dict[str, Any]]:
        """Get list of currently present employees."""
        try:
            db = next(get_db())
            present_employees = []
            
            # Get all employees
            employees = db.query(Employee).filter(Employee.is_active == True).all()
            
            for employee in employees:
                latest_record = self.db_manager.get_latest_attendance_by_employee(
                    employee.id, hours_back=12
                )
                
                if latest_record and latest_record.event_type == 'check_in':
                    present_employees.append({
                        'employee_id': employee.id,
                        'employee_name': employee.employee_name,
                        'department': employee.department,
                        'check_in_time': latest_record.timestamp.isoformat(),
                        'camera_id': latest_record.camera_id
                    })
            
            return present_employees
            
        except Exception as e:
            self.logger.error(f"Error getting present employees: {e}")
            return []
        finally:
            db.close()
    
    def shutdown(self):
        """Shutdown the face tracking system."""
        try:
            self.logger.info("Shutting down face tracking system...")
            
            # Set shutdown flag
            self.shutdown_flag.set()
            
            # Stop embedding update worker
            if self.embedding_update_worker and self.embedding_update_worker.is_alive():
                self.embedding_update_queue.put(None)  # Shutdown signal
                self.embedding_update_worker.join(timeout=5)
            
            # Stop camera threads
            for thread in self.camera_threads:
                if thread.is_alive():
                    thread.join(timeout=2)
            
            self.logger.info("Face tracking system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global instance for the application
face_tracking_system = None


def get_face_tracking_system() -> FaceTrackingSystem:
    """Get the global face tracking system instance."""
    global face_tracking_system
    if face_tracking_system is None:
        face_tracking_system = FaceTrackingSystem()
    return face_tracking_system


def start_face_tracking_system():
    """Start the face tracking system."""
    system = get_face_tracking_system()
    system.start_multi_camera_tracking()
    return system


def stop_face_tracking_system():
    """Stop the face tracking system."""
    global face_tracking_system
    if face_tracking_system:
        face_tracking_system.shutdown()
        face_tracking_system = None