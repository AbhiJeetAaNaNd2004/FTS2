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
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from insightface.app import FaceAnalysis
from bytetracker.byte_tracker import BYTETracker
import requests
import json
from datetime import datetime
import queue
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from db.db_manager import DatabaseManager
from db.db_config import create_tables
from db.db_models import Employee, FaceEmbedding, AttendanceRecord
from datetime import timedelta
from utils.logging import get_logger

# Global variables for Django integration
system_instance = None
is_tracking_running = False
log_buffer = []
latest_faces = {}
latest_attendance = deque(maxlen=100)
system_stats = {
    "uptime": 0,
    "cam_count": 0,
    "load": 0,
    "faces_detected": 0,
    "attendance_count": 0
}
present_users_by_department = defaultdict(list)
start_time = None

def log_message(msg):
    """Log messages to buffer instead of stdout"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    log_buffer.append(log_entry)
    # Keep last 1000 log entries
    if len(log_buffer) > 1000:
        log_buffer.pop(0)

# Replace all print statements with log_message
print = log_message

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

known_faces_dir = r"D:\Python Course\SEDL AI\insightface-env\known_faces"
THRESHOLD = 0.6
DET_THRESH = 0.5
MATCH_THRESH = 0.8
MAX_LIFETIME = 60
EMBED_UPDATE_COOLDOWN = 10
FRAME_INTERVAL = 1 / 10
GLOBAL_TRACK_TIMEOUT = 300
EMBEDDING_HISTORY_SIZE = 5
TRACK_BUFFER_SIZE = 30
log_file_path = "attendance_log.csv"
ENHANCED_CONFIG = {'face_quality_threshold': 0.65}

API_CONFIG = {
    'base_url': 'https://people.zoho.in/people/api',
    'attendance_endpoint': '/attendance',
    'token_url': 'https://accounts.zoho.in/oauth/v2/token',
    'access_token': '1000.93fd75fc041f34356a438da21926bbdf.283b9eb8d352b27e94cdff51a306b77f',  #ACTUAL TOKEN
    'refresh_token': '1000.2ccc8b029434d7f248b3b653c2e23e81.3772752c9f190137ef70db291d0a6c46',    #ACTUAL REFRESH TOKEN
    'client_id': '1000.K9X96PSA5YII5NNRIWKUNWANDFFCKL',
    'client_secret': '4750df6f1d6c575fe5b9c433b2b73147b5481dc167',
    'timeout': 10,
    'max_retries': 3
}

CAMERAS = [
    CameraConfig(
        camera_id=0,
        gpu_id=0,
        camera_type="entry",
        tripwires=[
            TripwireConfig(position=0.755551, spacing=0.01, direction="horizontal", name="EntryDetection")],
        resolution=(1280, 720),
        fps=15),
    CameraConfig(
        camera_id=1,
        gpu_id=0,
        camera_type="exit",
        tripwires=[
            TripwireConfig(position=0.5, spacing=0.01, direction="vertical", name="EntryDetection")],
        resolution=(1280, 720),
        fps=15)]

def load_employee_metadata(employee_id: str) -> Optional[EmployeeMetadata]:
    emp_folder = os.path.join(known_faces_dir, employee_id)
    metadata_path = os.path.join(emp_folder, "metadata.pkl")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return EmployeeMetadata(
                        employee_id=data.get('employee_id', employee_id),
                        employee_name=data.get('employee_name', employee_id),
                        enrollment_date=data.get('enrollment_date', ''),
                        embedding_count=data.get('embedding_count', 0),
                        source_images=data.get('source_images', []))
                elif isinstance(data, EmployeeMetadata):
                    return data
                else:
                    return None
        except:
            return None
    return None

def save_employee_metadata(metadata: EmployeeMetadata):
    emp_folder = os.path.join(known_faces_dir, metadata.employee_id)
    os.makedirs(emp_folder, exist_ok=True)
    metadata_path = os.path.join(emp_folder, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix =  np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = ENHANCED_CONFIG.get('kalman_process_noise', 0.1) * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = ENHANCED_CONFIG.get('kalman_measurement_noise', 0.1) * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, center_x: int, center_y: int) -> Tuple[int, int]:
        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        if not self.initialized:
            self.kalman.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32).reshape(4, 1)
            self.kalman.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32).reshape(4, 1)
            self.initialized = True
        prediction = self.kalman.predict()
        self.kalman.correct(measurement)
        return int(prediction[0]), int(prediction[1])

class APILogger:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        retry_strategy = Retry(
            total=config.get('max_retries', 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.token_lock = threading.Lock()
        self.access_token = config.get('access_token', '')
        self.refresh_token = config.get('refresh_token', '')
        self.client_id = config.get('client_id', '')
        self.client_secret = config.get('client_secret', '')
        self.token_expiry = 0  # Force immediate refresh
        self._refresh_token()
        self.lock = threading.Lock()
        self.retry_thread = threading.Thread(target=self._retry_failed_logs_worker, daemon=True)
        self.retry_thread.start()
        self.api_queue = queue.Queue(maxsize=1000)
        self.api_worker_thread = threading.Thread(target=self._api_worker, daemon=True)
        self.api_worker_thread.start()

    def _refresh_token(self):
        """Refresh Zoho access token using refresh token"""
        with self.token_lock:
            if time.time() < self.token_expiry - 300:  # 5-minute buffer
                return True
            log_message("[TOKEN] Refreshing access token...")
            params = {
                'refresh_token': self.refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'refresh_token'}
            try:
                response = requests.post(
                    self.config['token_url'],
                    params=params,
                    timeout=self.config.get('timeout', 10))
                response.raise_for_status()
                data = response.json()
                self.access_token = data['access_token']
                self.token_expiry = time.time() + data.get('expires_in', 3600)
                log_message(f"[TOKEN] Refreshed access token. Expires in {data.get('expires_in', 3600)}s")
                return True
            except requests.exceptions.RequestException as e:
                log_message(f"[TOKEN ERROR] Refresh failed: {str(e)}")
                if hasattr(e, 'response') and e.response:
                    log_message(f"Status code: {e.response.status_code}")
                    log_message(f"Response: {e.response.text}")
                self.access_token = ""
                self.token_expiry = 0
                return False

    def _send_attendance_to_zoho(self, emp_id: str, event_type: str):
        """Send attendance record to Zoho People API"""
        if not self.access_token or time.time() >= self.token_expiry - 300:
            if not self._refresh_token():
                log_message("[ZOHO] Skipping API call - no valid access token")
                return False
        params = {
            'empId': emp_id,
            'dateFormat': 'dd-MM-yyyy HH:mm:ss'}
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        if event_type == "check_in":
            params['checkIn'] = current_time
        else:
            params['checkOut'] = current_time
        url = self.config['base_url'] + self.config['attendance_endpoint']
        try:
            response = self.session.post(
                url,
                params=params,
                headers={'Authorization': f'Bearer {self.access_token}'},
                timeout=self.config.get('timeout', 10))
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if isinstance(response_data, list):
                        for item in response_data:
                            if isinstance(item, dict) and item.get('response') == 'success':
                                log_message(f"[ZOHO SUCCESS] Logged {event_type} for {emp_id}")
                                return True
                        log_message(f"[ZOHO ERROR] List response did not contain success: {response_data}")
                        return False
                    elif isinstance(response_data, dict):
                        if response_data.get('response') == 'success':
                            log_message(f"[ZOHO SUCCESS] Logged {event_type} for {emp_id}")
                            return True
                        else:
                            log_message(f"[ZOHO ERROR] API returned failure: {response_data}")
                            return False
                    else:
                        log_message(f"[ZOHO ERROR] Unexpected response type: {type(response_data)} - {response_data}")
                        return False
                except json.JSONDecodeError:
                    log_message(f"[ZOHO ERROR] Invalid JSON response: {response.text}")
                    return False
            else:
                if response.status_code == 401:
                    log_message("[ZOHO] Token expired, refreshing...")
                    if self._refresh_token():
                        response = self.session.post(
                            url,
                            params=params,
                            headers={'Authorization': f'Bearer {self.access_token}'},
                            timeout=self.config.get('timeout', 10))
                        if response.status_code == 200:
                            try:
                                response_data = response.json()
                                if isinstance(response_data, list):
                                    for item in response_data:
                                        if isinstance(item, dict) and item.get('response') == 'success':
                                            log_message(f"[ZOHO SUCCESS] Logged {event_type} for {emp_id}")
                                            return True
                                elif isinstance(response_data, dict) and response_data.get('response') == 'success':
                                    log_message(f"[ZOHO SUCCESS] Logged {event_type} for {emp_id}")
                                    return True
                            except json.JSONDecodeError:
                                pass
                log_message(f"[ZOHO ERROR] HTTP {response.status_code}: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            log_message(f"[ZOHO ERROR] Request failed: {str(e)}")
            return False

    def _fallback_log(self, emp_id, event_type, timestamp=None):
        log_entry = {
            "emp_id": emp_id,
            "event_type": event_type,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        try:
            with self.lock:
                with open("failed_logs.jsonl", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            log_message(f"[FALLBACK] Log backed up for retry: {log_entry}")
        except Exception as e:
            log_message(f"[FALLBACK ERROR] Could not write fallback log: {e}")

    def _retry_failed_logs_worker(self):
        while True:
            time.sleep(60)  # Retry every minute
            try:
                if not os.path.exists("failed_logs.jsonl"):
                    continue
                with self.lock:
                    with open("failed_logs.jsonl", "r") as f:
                        lines = f.readlines()
                    open("failed_logs.jsonl", "w").close()  # Clear for re-adds
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        emp_id = entry['emp_id']
                        event_type = entry['event_type']
                        log_message(f"[RETRY] Attempting retry for {emp_id} ({event_type})")
                        success = self._send_attendance_to_zoho(emp_id, event_type)
                        if not success:
                            self._fallback_log(emp_id, event_type, entry.get("timestamp"))
                    except Exception as e:
                        log_message(f"[RETRY ERROR] {e}")
                        with self.lock:
                            with open("failed_logs.jsonl", "a") as f:
                                f.write(line)  # Put it back
            except Exception as e:
                log_message(f"[RETRY WORKER ERROR] {e}")

    def _api_worker(self):
        while True:
            try:
                task = self.api_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                emp_id, event_type = task
                success = self._send_attendance_to_zoho(emp_id, event_type)
                if not success:
                    log_message(f"[API WORKER] Failed to log attendance for {emp_id}")
                    self._fallback_log(emp_id, event_type)
                self.api_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log_message(f"[API WORKER ERROR] {str(e)}")
                import traceback
                log_message(traceback.format_exc())

    def log_attendance_async(self, emp_id: str, event_type: str):
        try:
            self.api_queue.put((emp_id, event_type), timeout=1)
        except queue.Full:
            log_message(f"[WARNING] API queue full - dropping attendance log for {emp_id}")

    def shutdown(self):
        self.api_queue.put(None)
        self.api_worker_thread.join(timeout=5)

class FaceTrackingSystem:
    def __init__(self, face_app):
        self.face_app = face_app
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
        self.frame_locks = {}
        self.latest_frames = {}
        self.latest_faces = {}
        self.face_detection_threads = {}
        self.embedding_cache = {}
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
        self.logger = get_logger(__name__)
        self.api_logger = APILogger(API_CONFIG)
        self.enable_csv_backup = True
        self.global_tracks_lock = threading.RLock()
        self.embedding_update_lock = threading.RLock()
        self.identity_tracks_lock = threading.RLock()
        self.embedding_cache_lock = threading.RLock()
        self.faiss_index_lock = threading.RLock()
        self.metadata_lock = threading.RLock()
        self.embedding_update_queue = queue.Queue()
        self.shutdown_flag = threading.Event()
        self.embedding_update_worker = None
        self.batch_update_threshold = 5
        self.updates_since_last_rebuild = 0
        self.max_updates_before_rebuild = 20
        self.db_manager = DatabaseManager()
        create_tables()
        self.embedding_update_worker = threading.Thread(target=self._embedding_update_worker, daemon=True)
        self.embedding_update_worker.start()
        if self.enable_csv_backup:
            self._prepare_csv()
        self.camera_threads = []
        self._load_known_faces()
        self._load_employee_metadata()
        self._initialize_faiss()
        self._initialize_multi_gpu_insightface()
        self._initialize_cameras()
        self._prepare_csv()
        
        # Start stats updater thread
        self.stats_thread = threading.Thread(target=self._update_stats, daemon=True)
        self.stats_thread.start()

    def _update_stats(self):
        """Periodically update system statistics"""
        while not self.shutdown_flag.is_set():
            # Update present users by department
            with self.global_tracks_lock:
                active_tracks = [
                    track for track in self.global_tracks.values() 
                    if time.time() - track.last_seen_time < GLOBAL_TRACK_TIMEOUT
                ]
                
                # Reset department map
                global present_users_by_department
                present_users_by_department = defaultdict(list)
                
                # Update with active users
                for track in active_tracks:
                    emp_id = track.employee_id
                    with self.metadata_lock:
                        if emp_id in self.employee_metadata:
                            dept = self.employee_metadata[emp_id].get('department', 'Unknown')
                            present_users_by_department[dept].append(emp_id)
            
            # Update other stats
            system_stats["cam_count"] = len(CAMERAS)
            system_stats["faces_detected"] = len(active_tracks)
            system_stats["attendance_count"] = len(latest_attendance)
            
            time.sleep(5)  # Update every 5 seconds

    def _load_known_faces(self):
        try:
            embeddings_list, labels_list = self.db_manager.get_all_active_embeddings()
            if embeddings_list:
                self.embeddings = np.array(embeddings_list).astype('float32')
                self.labels = labels_list
                faiss.normalize_L2(self.embeddings)
            log_message(f"[INIT] Loaded {len(set(labels_list)) if labels_list else 0} employees with {len(embeddings_list)} embeddings from database")
        except Exception as e:
            log_message(f"[ERROR] Failed to load known faces from database: {e}")
            self.embeddings = []
            self.labels = []

    def _load_employee_metadata(self):
        try:
            employees = self.db_manager.get_all_employees()
            for employee in employees:
                self.employee_metadata[employee.id] = {
                    'employee_name': employee.employee_name,
                    'department': employee.department,
                    'designation': employee.designation,
                    'email': employee.email,
                    'phone': employee.phone
                }
            log_message(f"[INIT] Loaded metadata for {len(employees)} employees from database")
        except Exception as e:
            log_message(f"[ERROR] Failed to load employee metadata from database: {e}")

    def get_employee_name(self, employee_id: str) -> str:
        with self.metadata_lock:
            if employee_id in self.employee_metadata:
                return self.employee_metadata[employee_id]['employee_name']
            employee = self.db_manager.get_employee(employee_id)
            if employee:
                return employee.employee_name
            return employee_id

    def _check_employee_work_status(self, employee_id: str) -> bool:
        try:
            latest_record = self.db_manager.get_latest_attendance_by_employee(employee_id, hours_back=10)
            if latest_record:
                is_working = latest_record.event_type == 'check_in'
                log_message(f"[STATUS CHECK] {employee_id} is {'working' if is_working else 'not working'} (last {latest_record.event_type} at {latest_record.timestamp})")
                return is_working
            else:
                log_message(f"[STATUS CHECK] No recent attendance records found for {employee_id}")
                return False
        except Exception as e:
            log_message(f"[STATUS CHECK] Database error for {employee_id}: {str(e)}")
            return False

    def _update_embeddings(self, identity: str, embedding: np.ndarray):
        current_time = time.time()
        with self.embedding_update_lock:
            if identity in self.last_embedding_update:
                if current_time - self.last_embedding_update[identity] < EMBED_UPDATE_COOLDOWN:
                    return False
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

    def _cleanup_old_embeddings(self, identity: str, max_embeddings: int = 25):
        try:
            self.db_manager.cleanup_old_embeddings(identity, max_embeddings)
        except Exception as e:
            log_message(f"[WARNING] Could not cleanup old embeddings for {identity}: {e}")

    def _log_attendance(self, employee_id: str, event_type: str, camera_id: str, confidence: float, 
                       position: tuple, zone: str = None, metadata: dict = None):
        try:
            employee_name = self.get_employee_name(employee_id)
            success = self.db_manager.create_attendance_record(
                employee_id=employee_id,
                event_type=event_type,
                camera_id=camera_id,
                confidence=confidence,
                position_x=position[0] if position else None,
                position_y=position[1] if position else None,
                zone=zone,
                metadata=metadata)
            if success:
                log_message(f"[ATTENDANCE] {employee_name} ({employee_id}) - {event_type} - Camera: {camera_id} - Confidence: {confidence:.2f}")
                if hasattr(self, 'api_logger') and self.api_logger:
                    self.api_logger.log_attendance(
                        employee_id=employee_id,
                        employee_name=employee_name,
                        event_type=event_type,
                        camera_id=camera_id,
                        confidence=confidence,
                        zone=zone,
                        metadata=metadata)
                if self.enable_csv_backup:
                    self._log_to_csv(employee_id, employee_name, event_type, camera_id, confidence, zone)
                
                # Add to latest attendance buffer
                attendance_event = {
                    "employee_id": employee_id,
                    "employee_name": employee_name,
                    "event_type": event_type,
                    "timestamp": datetime.now().isoformat(),
                    "camera_id": camera_id,
                    "confidence": confidence,
                    "zone": zone
                }
                latest_attendance.append(attendance_event)
                
            else:
                log_message(f"[ERROR] Failed to log attendance for {employee_id}")
        except Exception as e:
            log_message(f"[ERROR] Exception in _log_attendance: {e}")

    def get_attendance_history(self, employee_id: str = None, start_date: str = None, 
                              end_date: str = None, limit: int = 100):
        """Get attendance history from database"""
        try:
            records = self.db_manager.get_attendance_records(
                employee_id=employee_id,
                start_date=start_date,
                end_date=end_date,
                limit=limit)
            history = []
            for record in records:
                employee_name = self.get_employee_name(record.employee_id)
                history.append({
                    'employee_id': record.employee_id,
                    'employee_name': employee_name,
                    'event_type': record.event_type,
                    'timestamp': record.timestamp,
                    'camera_id': record.camera_id,
                    'confidence': record.confidence,
                    'zone': record.zone,
                    'metadata': record.metadata})
            return history
        except Exception as e:
            log_message(f"[ERROR] Failed to get attendance history: {e}")
            return []

    def register_employee(self, employee_id: str, employee_name: str, department: str = None,
                         designation: str = None, email: str = None, phone: str = None):
        """Register a new employee in the database"""
        try:
            success = self.db_manager.create_employee(
                employee_id=employee_id,
                employee_name=employee_name,
                department=department,
                designation=designation,
                email=email,
                phone=phone)
            if success:
                with self.metadata_lock:
                    self.employee_metadata[employee_id] = {
                        'employee_name': employee_name,
                        'department': department,
                        'designation': designation,
                        'email': email,
                        'phone': phone}
                log_message(f"[REGISTER] Successfully registered employee: {employee_name} ({employee_id})")
                return True
            else:
                log_message(f"[ERROR] Failed to register employee: {employee_id}")
                return False
        except Exception as e:
            log_message(f"[ERROR] Exception in register_employee: {e}")
            return False

    def add_employee_face(self, employee_id: str, image_path: str, embedding: np.ndarray = None):
        """Add face embedding for an employee"""
        try:
            if embedding is None:
                log_message("[WARNING] Embedding generation from image not implemented")
                return False
            embedding = embedding / np.linalg.norm(embedding)
            success = self.db_manager.store_face_embedding(
                employee_id=employee_id,
                embedding=embedding,
                embedding_type='registration',
                quality_score=1.0,
                source_image_path=image_path)
            if success:
                with self.faiss_index_lock:
                    if len(self.embeddings) > 0:
                        self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
                    else:
                        self.embeddings = embedding.reshape(1, -1)
                    self.labels.append(employee_id)
                    faiss.normalize_L2(self.embeddings)
                    self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                    self.index.add(self.embeddings)
                log_message(f"[FACE ADD] Successfully added face for employee: {employee_id}")
                return True
            else:
                log_message(f"[ERROR] Failed to store face embedding for: {employee_id}")
                return False
        except Exception as e:
            log_message(f"[ERROR] Exception in add_employee_face: {e}")
            return False

    def cleanup_database(self):
        """Clean up old records and optimize database"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            deleted_count = self.db_manager.cleanup_old_attendance_records(cutoff_date)
            log_message(f"[CLEANUP] Deleted {deleted_count} old attendance records")
            employees = self.db_manager.get_all_employees()
            for employee in employees:
                self.db_manager.cleanup_old_embeddings(employee.id, max_embeddings=15)
            log_message("[CLEANUP] Database cleanup completed")
        except Exception as e:
            log_message(f"[ERROR] Exception in cleanup_database: {e}")

    def shutdown(self):
        """Properly shutdown the system"""
        try:
            log_message("[SHUTDOWN] Initiating shutdown...")
            self.shutdown_flag.set()
            if self.embedding_update_worker and self.embedding_update_worker.is_alive():
                log_message("[SHUTDOWN] Waiting for embedding update worker...")
                self.embedding_update_worker.join(timeout=5)
            for thread in self.camera_threads:
                if thread.is_alive():
                    thread.join(timeout=2)
            if hasattr(self, 'db_manager'):
                self.db_manager.close()
            log_message("[SHUTDOWN] Shutdown completed")
        except Exception as e:
            log_message(f"[ERROR] Exception during shutdown: {e}")

    def get_database_stats(self):
        """Get database statistics"""
        try:
            stats = {
                'total_employees': self.db_manager.get_employee_count(),
                'total_embeddings': self.db_manager.get_embedding_count(),
                'total_attendance_records': self.db_manager.get_attendance_count(),
                'active_employees': len(set(self.labels)) if self.labels else 0,
                'loaded_embeddings': len(self.embeddings) if len(self.embeddings) > 0 else 0}
            return stats
        except Exception as e:
            log_message(f"[ERROR] Failed to get database stats: {e}")
            return {}

    def _initialize_faiss(self):
        if len(self.embeddings) > 0:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        else:
            self.index = None
    def reload_embeddings_and_rebuild_index(self):
        """Reload embeddings from DB and rebuild FAISS index."""
        with self.faiss_index_lock:
            self.embeddings, self.labels = self.db_manager.get_all_active_embeddings()
            if self.embeddings:
                self.embeddings = np.array(self.embeddings).astype('float32')
                faiss.normalize_L2(self.embeddings)
                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.index.add(self.embeddings)
            else:
                self.embeddings = []
                self.labels = []
                self.index = None
            print("[INDEX REBUILD] FAISS index rebuilt with current active embeddings.")

    def _initialize_multi_gpu_insightface(self):
        gpu_ids = list(set([cam.gpu_id for cam in CAMERAS]))
        for gpu_id in gpu_ids:
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                providers = [('CUDAExecutionProvider', {'device_id': gpu_id}), 'CPUExecutionProvider']
            else:
                log_message(f"[GPU WARNING] GPU ID {gpu_id} unavailable, using CPU")
            self.apps[gpu_id] = FaceAnalysis(name='antelopev2',
                providers=providers,
                allowed_modules=['detection', 'recognition'])
            self.apps[gpu_id].prepare(ctx_id=gpu_id, det_size=(416, 416), det_thresh=DET_THRESH)

    def _initialize_cameras(self):
        for cam_config in CAMERAS:
            cam_id = cam_config.camera_id
            self.trackers[cam_id] = BYTETracker(
                frame_rate=cam_config.fps,
                track_buffer=TRACK_BUFFER_SIZE,
                match_thresh=MATCH_THRESH)
            self.frame_locks[cam_id] = threading.Lock()
            self.latest_frames[cam_id] = None
            self.latest_faces[cam_id] = []
            self.track_identities[cam_id] = {}
            self.track_lifetimes[cam_id] = {}
            self.track_positions[cam_id] = {}
            self.frame_skip_counter[cam_id] = 0
            self.detection_interval[cam_id] = 3
            self.track_identities[cam_id] = {}

    def _prepare_csv(self):
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "EmployeeID", "EmployeeName", "CameraID", "Event", "Status"])

    def _enhance_frame_for_cctv(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_frame = cv2.GaussianBlur(enhanced_frame, (3, 3), 0.5)
        return enhanced_frame

    def _adaptive_detection_interval(self, camera_id: int, num_faces: int):
        if num_faces == 0:
            self.detection_interval[camera_id] = min(5, self.detection_interval[camera_id] + 1)
        elif num_faces > 3:
            self.detection_interval[camera_id] = max(2, self.detection_interval[camera_id] - 1)
        else:
            self.detection_interval[camera_id] = 3

    def _face_detection_thread(self, camera_id: int, gpu_id: int):
        while not self.shutdown_flag.is_set():
            try:
                current_time = time.time()
                if current_time - self.last_faces_reload > 300:
                    if self._reload_known_faces_and_metadata():
                        self.last_faces_reload = current_time
                with self.frame_locks[camera_id]:
                    if self.latest_frames[camera_id] is None:
                        time.sleep(0.02)
                        continue
                    frame_copy = self.latest_frames[camera_id].copy()
                self.frame_skip_counter[camera_id] += 1
                if self.frame_skip_counter[camera_id] < self.detection_interval[camera_id]:
                    time.sleep(0.01)
                    continue
                self.frame_skip_counter[camera_id] = 0
                enhanced_frame = self._enhance_frame_for_cctv(frame_copy)
                height, width = enhanced_frame.shape[:2]
                scale_factor = 1.0
                if width > 960:
                    scale_factor = 960 / width
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    enhanced_frame = cv2.resize(enhanced_frame, (new_width, new_height))
                faces = self.apps[gpu_id].get(enhanced_frame)
                if scale_factor != 1.0:
                    for face in faces:
                        face.bbox = face.bbox / scale_factor
                with self.frame_locks[camera_id]:
                    self.latest_faces[camera_id] = faces
                    
                    # Update latest_faces for Django
                    latest_faces[camera_id] = [
                        {
                            "identity": face.get('identity', 'unknown'),
                            "bbox": face.bbox.tolist(),
                            "confidence": face.det_score
                        }
                        for face in faces
                    ]
                    
                self._adaptive_detection_interval(camera_id, len(faces))
            except Exception as e:
                if not self.shutdown_flag.is_set():
                    log_message(f"[ERROR] Face detection thread {camera_id}: {e}")
                time.sleep(0.1)

    def detect_faces(self, frame):
        """
        Detect faces in a frame using InsightFace.
        
        Args:
            frame: numpy array of BGR image
            
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        if frame is None:
            return []
            
        try:
            # Get faces from InsightFace
            faces = self.face_app.get(frame)
            if faces is None:
                return []
                
            return faces
            
        except Exception as e:
            self.logger.error(f"Error in face detection: {e}")
            return []

    def _compute_embedding_similarity(self, embedding: np.ndarray) -> Tuple[str, float]:
        emb_hash = hash(embedding.tobytes()[:100])
        with self.embedding_cache_lock:
            if emb_hash in self.embedding_cache:
                return self.embedding_cache[emb_hash]
        with self.faiss_index_lock:
            if self.index is None or not hasattr(self.index, 'ntotal') or self.index.ntotal == 0 or len(self.labels) == 0:
                return "unknown", 0.0
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                embedding = embedding / emb_norm
            else:
                return "unknown", 0.0
            try:
                k = min(3, len(self.labels))
                D, I = self.index.search(embedding.reshape(1, -1), k)
                if len(D[0]) > 0 and len(I[0]) > 0:
                    best_scores = D[0]
                    best_indices = I[0]
                    weighted_scores = {}
                    for score, idx in zip(best_scores, best_indices):
                        if score > THRESHOLD and idx < len(self.labels):
                            identity = self.labels[idx]
                            if identity in weighted_scores:
                                weighted_scores[identity] = max(weighted_scores[identity], score)
                            else:
                                weighted_scores[identity] = score
                    if weighted_scores:
                        best_identity = max(weighted_scores.items(), key=lambda x: x[1])
                        result = (best_identity[0], float(best_identity[1]))
                    else:
                        result = ("unknown", 0.0)
                else:
                    result = ("unknown", 0.0)
            except Exception as e:
                log_message(f"[ERROR] FAISS search failed: {e}")
                result = ("unknown", 0.0)
        with self.embedding_cache_lock:
            if len(self.embedding_cache) < 1000:
                self.embedding_cache[emb_hash] = result
        return result

    def _temporal_smoothing(self, identity: str, score: float, camera_id: int) -> Tuple[str, float]:
        current_time = time.time()
        track_key = f"{camera_id}_{identity}"
        if track_key not in self.track_identities:
            self.track_identities[camera_id][track_key] = {
                'votes': deque(maxlen=5),
                'last_update': current_time}
        track_data = self.track_identities[camera_id][track_key]
        if current_time - track_data['last_update'] > 2.0:
            track_data['votes'].clear()
        track_data['votes'].append((identity, score))
        track_data['last_update'] = current_time
        if len(track_data['votes']) >= 3:
            identity_counts = {}
            total_score = 0
            for vote_identity, vote_score in track_data['votes']:
                if vote_identity in identity_counts:
                    identity_counts[vote_identity] = max(identity_counts[vote_identity], vote_score)
                else:
                    identity_counts[vote_identity] = vote_score
                total_score += vote_score
            if identity_counts:
                best_identity = max(identity_counts.items(), key=lambda x: x[1])
                avg_score = total_score / len(track_data['votes'])
                return best_identity[0], min(best_identity[1], avg_score)
        return identity, score

    def _quality_filter(self, face, frame_width: int, frame_height: int) -> Tuple[bool, FaceQualityMetrics]:
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        size_score = min(1.0, (face_width * face_height) / (100 * 100))
        if face_width < 50 or face_height < 50:
            size_score = 0.0
        if face_width > frame_width * 0.8 or face_height > frame_height * 0.8:
            size_score = 0.0
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        distance_from_center = np.sqrt((center_x - frame_width/2)**2 + (center_y - frame_height/2)**2)
        max_distance = np.sqrt((frame_width/2)**2 + (frame_height/2)**2)
        position_score = 1.0 - (distance_from_center / max_distance)
        det_score = face.det_score if hasattr(face, 'det_score') else 0.5
        brightness_score = self._compute_brightness_score(face, bbox)
        sharpness_score = self._compute_sharpness_score(face, bbox)
        angle_score = self._compute_face_angle_score(face)
        overall_quality = (
            0.3 * size_score +
            0.2 * position_score +
            0.2 * det_score +
            0.1 * brightness_score +
            0.1 * sharpness_score +
            0.1 * angle_score)
        quality_metrics = FaceQualityMetrics(
            sharpness_score=sharpness_score,
            brightness_score=brightness_score,
            angle_score=angle_score,
            size_score=size_score,
            overall_quality=overall_quality)
        is_valid = overall_quality >= ENHANCED_CONFIG['face_quality_threshold']
        return is_valid, quality_metrics

    def _adaptive_threshold(self, identity: str, base_score: float) -> float:
        if identity in self.global_tracks:
            track = self.global_tracks[identity]
            recent_scores = list(track.embedding_history)[-10:] if hasattr(track, 'embedding_history') else []
            if len(recent_scores) >= 5:
                avg_recent_score = np.mean([self._compute_embedding_similarity(emb)[1] for emb in recent_scores])
                if avg_recent_score > 0.8:
                    return THRESHOLD * 0.9
                elif avg_recent_score < 0.6:
                    return THRESHOLD * 1.1
        return THRESHOLD

    def _compute_brightness_score(self, face, bbox) -> float:
        try:
            if hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106
                if landmarks is not None and len(landmarks) > 0:
                    avg_intensity = np.mean(landmarks) / 255.0
                    return min(1.0, max(0.0, 1.0 - abs(avg_intensity - 0.5) * 2))
            return 0.5
        except:
            return 0.5

    def _compute_sharpness_score(self, face, bbox) -> float:
        try:
            if hasattr(face, 'embedding'):
                embedding_var = np.var(face.embedding)
                normalized_var = min(1.0, embedding_var / 0.1)
                return normalized_var
            return 0.5
        except:
            return 0.5

    def _compute_face_angle_score(self, face) -> float:
        try:
            if hasattr(face, 'pose'):
                yaw, pitch, roll = face.pose
                angle_penalty = (abs(yaw) + abs(pitch) + abs(roll)) / 90.0
                return max(0.0, 1.0 - angle_penalty)
            return 0.8
        except:
            return 0.8

    def _embedding_update_worker(self):
        pending_updates = []
        while not self.shutdown_flag.is_set():
            try:
                try:
                    update = self.embedding_update_queue.get(timeout=1.0)
                    if update is None:
                        break
                    pending_updates.append(update)
                except queue.Empty:
                    if pending_updates:
                        self._process_pending_updates(pending_updates)
                        pending_updates = []
                    continue
                if len(pending_updates) >= self.batch_update_threshold:
                    self._process_pending_updates(pending_updates)
                    pending_updates = []
            except Exception as e:
                log_message(f"[ERROR] Embedding update worker: {e}")
                time.sleep(0.1)
        if pending_updates:
            self._process_pending_updates(pending_updates)

    def _process_pending_updates(self, pending_updates):
        with self.embedding_update_lock:
            new_embeddings = []
            new_labels = []
            for identity, embedding, timestamp in pending_updates:
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
                    self.db_manager.cleanup_old_embeddings(identity, max_embeddings=15)
            if new_embeddings:
                self.updates_since_last_rebuild += len(new_embeddings)
                if self.updates_since_last_rebuild >= self.max_updates_before_rebuild:
                    with self.faiss_index_lock:
                        self.embeddings = np.vstack([self.embeddings] + new_embeddings)
                        self.labels.extend(new_labels)
                        faiss.normalize_L2(self.embeddings)
                        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                        self.index.add(self.embeddings)
                    self.updates_since_last_rebuild = 0
                    with self.embedding_cache_lock:
                        self.embedding_cache.clear()
                else:
                    with self.faiss_index_lock:
                        self.embeddings = np.vstack([self.embeddings] + new_embeddings)
                        self.labels.extend(new_labels)
                        faiss.normalize_L2(self.embeddings)
                        for i, emb in enumerate(new_embeddings):
                            self.index.add(emb.reshape(1, -1))

    def _reload_known_faces_and_metadata(self):
        try:
            with self.faiss_index_lock:
                old_employee_count = len(set(self.labels)) if self.labels else 0
            embeddings_list, labels_list = self.db_manager.get_all_active_embeddings()
            employees = self.db_manager.get_all_employees()
            employee_metadata = {}
            for employee in employees:
                employee_metadata[employee.id] = {
                    'employee_name': employee.employee_name,
                    'department': employee.department,
                    'designation': employee.designation,
                    'email': employee.email,
                    'phone': employee.phone}
            with self.faiss_index_lock:
                if embeddings_list:
                    self.embeddings = np.array(embeddings_list).astype('float32')
                    self.labels = labels_list
                    faiss.normalize_L2(self.embeddings)
                    self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                    self.index.add(self.embeddings)
                else:
                    self.embeddings = []
                    self.labels = []
                    self.index = None
                new_employee_count = len(set(labels_list)) if labels_list else 0
            with self.metadata_lock:
                self.employee_metadata = employee_metadata         
            with self.embedding_cache_lock:
                self.embedding_cache.clear()
            log_message(f"[RELOAD] Reloaded faces and metadata: {old_employee_count} -> {new_employee_count} employees")
            self.last_faces_reload = time.time()
        except Exception as e:
            log_message(f"[ERROR] Failed to reload known faces and metadata: {e}")

    def _get_consistent_track_id(self, identity: str, camera_id: int) -> str:
        current_time = time.time()
        if identity == "unknown":
            return f"unknown_{camera_id}_{int(current_time)}"
        with self.identity_tracks_lock:
            if identity in self.identity_tracks:
                self.identity_last_seen[identity] = current_time
                self.identity_cameras[identity] = camera_id
                return self.identity_tracks[identity]
            track_id = identity
            self.identity_tracks[identity] = track_id
            self.identity_last_seen[identity] = current_time
            self.identity_cameras[identity] = camera_id
            self.identity_positions[identity] = {}
            self.identity_trip_logged[identity] = set()
            self.identity_crossing_state[identity] = {}
            self.identity_zone_state[identity] = {}
            return track_id

    def _update_work_status(self, identity: str, camera_id: int, direction: str):
        if identity not in self.global_tracks:
            return
        track = self.global_tracks[identity]
        if camera_id == 0 and direction == "left->right":
            track.work_status = "working"
            self._log_event(identity, camera_id, "WorkAreaEntry", track.work_status)
        elif camera_id == 1 and direction == "right->left":
            track.work_status = "working"
            self._log_event(identity, camera_id, "WorkAreaEntry", track.work_status)
        elif camera_id == 0 and direction == "right->left":
            track.work_status = "on_break"
            self._log_event(identity, camera_id, "WorkAreaExit", track.work_status)
        elif camera_id == 1 and direction == "left->right":
            track.work_status = "on_break"
            self._log_event(identity, camera_id, "WorkAreaExit", track.work_status)
        elif camera_id == 0 and direction == "top->bottom":
            track.work_status = "working"
            self._log_event(identity, camera_id, "WorkAreaEntry", track.work_status)
        elif camera_id == 1 and direction == "bottom->top":
            track.work_status = "working"
            self._log_event(identity, camera_id, "WorkAreaEntry", track.work_status)
        elif camera_id == 0 and direction == "bottom->top":
            track.work_status = "on_break"
            self._log_event(identity, camera_id, "WorkAreaExit", track.work_status)
        elif camera_id == 1 and direction == "top->bottom":
            track.work_status = "on_break"
            self._log_event(identity, camera_id, "WorkAreaExit", track.work_status)

    def _log_event(self, identity: str, camera_id: int, event: str):
        event_type = "check_in" if event in ["entry", "WorkAreaEntry"] else "check_out"
        if event_type == "check_out":
            if not self._check_employee_work_status(identity):
                log_message(f"[EVENT BLOCKED] {identity} not currently working - exit not logged")
                return
        employee_name = self.get_employee_name(identity)
        self.api_logger.log_attendance_async(identity, event_type)
        if self.enable_csv_backup:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._write_csv_backup(
                timestamp,
                identity,
                employee_name,
                camera_id,
                event_type,
                "logged")
        log_message(f"[EVENT] Camera {camera_id}: {identity} - {event_type}")
    def _write_csv_backup(self, timestamp, identity, employee_name, camera_id, event, status):
        try:
            with open(log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, identity, employee_name, camera_id, event, status])
        except Exception as e:
            log_message(f"[WARNING] CSV backup failed: {e}")
    def _check_tripwire_crossing(self, identity: str, center_x: int, center_y: int, camera_config: CameraConfig, frame_width: int, frame_height: int):
        current_time = time.time()
        camera_key = f"{camera_config.camera_id}"
        if identity not in self.identity_crossing_state:
            self.identity_crossing_state[identity] = {}
        if camera_key not in self.identity_crossing_state[identity]:
            self.identity_crossing_state[identity][camera_key] = {}
        for tripwire in camera_config.tripwires:
            tripwire_key = f"{tripwire.name}"
            if tripwire_key not in self.identity_crossing_state[identity][camera_key]:
                self.identity_crossing_state[identity][camera_key][tripwire_key] = {
                    'state': 'none',
                    'last_position': center_x if tripwire.direction == 'vertical' else center_y,
                    'direction': None}
            state_info = self.identity_crossing_state[identity][camera_key][tripwire_key]
            if tripwire.direction == 'vertical':
                tripwire1_pos = int(frame_width * (tripwire.position - tripwire.spacing/2))
                tripwire2_pos = int(frame_width * (tripwire.position + tripwire.spacing/2))
                current_pos = center_x
                if state_info['state'] == 'none':
                    if current_pos < tripwire1_pos:
                        state_info['state'] = 'left_zone'
                        state_info['direction'] = 'left->right'
                    elif current_pos > tripwire2_pos:
                        state_info['state'] = 'right_zone'
                        state_info['direction'] = 'right->left'
                elif state_info['state'] == 'left_zone' and state_info['direction'] == 'left->right':
                    if current_pos > tripwire2_pos:
                        self._log_event(identity, camera_config.camera_id, camera_config.camera_type)
                        state_info['state'] = 'none'
                        state_info['direction'] = None
                elif state_info['state'] == 'right_zone' and state_info['direction'] == 'right->left':
                    if current_pos < tripwire1_pos:
                        self._log_event(identity, camera_config.camera_id, camera_config.camera_type)
                        state_info['state'] = 'none'
                        state_info['direction'] = None
            else:
                tripwire1_pos = int(frame_height * (tripwire.position - tripwire.spacing/2))
                tripwire2_pos = int(frame_height * (tripwire.position + tripwire.spacing/2))
                current_pos = center_y
                if state_info['state'] == 'none':
                    if current_pos < tripwire1_pos:
                        state_info['state'] = 'top_zone'
                        state_info['direction'] = 'top->bottom'
                    elif current_pos > tripwire2_pos:
                        state_info['state'] = 'bottom_zone'
                        state_info['direction'] = 'bottom->top'
                elif state_info['state'] == 'top_zone' and state_info['direction'] == 'top->bottom':
                    if current_pos > tripwire2_pos:
                        self._log_event(identity, camera_config.camera_id, camera_config.camera_type)
                        state_info['state'] = 'none'
                        state_info['direction'] = None
                elif state_info['state'] == 'bottom_zone' and state_info['direction'] == 'bottom->top':
                    if current_pos < tripwire1_pos:
                        self._log_event(identity, camera_config.camera_id, camera_config.camera_type)
                        state_info['state'] = 'none'
                        state_info['direction'] = None
            state_info['last_position'] = current_pos
    def draw_tripwires(self, frame, camera_config: CameraConfig):
        frame_height, frame_width = frame.shape[:2]
        for tripwire in camera_config.tripwires:
            if tripwire.direction == 'vertical':
                tripwire1_x = int(frame_width * (tripwire.position - tripwire.spacing/2))
                tripwire2_x = int(frame_width * (tripwire.position + tripwire.spacing/2))
                cv2.line(frame, (tripwire1_x, 0), (tripwire1_x, frame_height), (0, 255, 255), 2)
                cv2.line(frame, (tripwire2_x, 0), (tripwire2_x, frame_height), (255, 0, 255), 2)
                cv2.putText(frame, tripwire.name, (tripwire1_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                tripwire1_y = int(frame_height * (tripwire.position - tripwire.spacing/2))
                tripwire2_y = int(frame_height * (tripwire.position + tripwire.spacing/2))
                cv2.line(frame, (0, tripwire1_y), (frame_width, tripwire1_y), (0, 255, 255), 2)
                cv2.line(frame, (0, tripwire2_y), (frame_width, tripwire2_y), (255, 0, 255), 2)
                cv2.putText(frame, tripwire.name, (10, tripwire1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    def process_camera(self, camera_config: CameraConfig):
        cap = cv2.VideoCapture(camera_config.camera_id)
        if not cap.isOpened():
            log_message(f"[ERROR] Cannot open camera {camera_config.camera_id}")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, camera_config.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.resolution[1])
        detection_thread = threading.Thread(
            target=self._face_detection_thread,
            args=(camera_config.camera_id, camera_config.gpu_id),
            daemon=True)
        detection_thread.start()
        self.face_detection_threads[camera_config.camera_id] = detection_thread
        prev_time = 0
        frame_count = 0
        while not self.shutdown_flag.is_set():
            current_time = time.time()
            if current_time - prev_time < FRAME_INTERVAL:
                time.sleep(0.005)
                continue
            prev_time = current_time
            ret, frame = cap.read()
            if not ret:
                log_message(f"[WARNING] Failed to read frame from camera {camera_config.camera_id}")
                time.sleep(0.1)
                continue
            frame_count += 1
            frame_height, frame_width = frame.shape[:2]
            with self.frame_locks[camera_config.camera_id]:
                self.latest_frames[camera_config.camera_id] = frame.copy()
                faces = self.latest_faces[camera_config.camera_id][:]
            face_centers = {}
            valid_faces = []
            for face in faces:
                is_valid, quality_metrics = self._quality_filter(face, frame_width, frame_height)
                if not is_valid:
                    continue  # Skip low-quality faces
                valid_faces.append(face)
            for i, face in enumerate(valid_faces):
                bbox = face.bbox.astype(int)
                embedding = face.embedding.astype('float32')
                identity, score = self._compute_embedding_similarity(embedding)
                if identity != "unknown":
                    adaptive_thresh = self._adaptive_threshold(identity, score)
                    if score >= adaptive_thresh:
                        identity, score = self._temporal_smoothing(identity, score, camera_config.camera_id)
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        if identity not in self.kalman_trackers:
                            self.kalman_trackers[identity] = KalmanTracker()
                        smoothed_position = self.kalman_trackers[identity].update(center_x, center_y)
                        face_centers[identity] = smoothed_position
                        with self.global_tracks_lock:
                            if identity not in self.global_tracks:
                                self.global_tracks[identity] = GlobalTrack(
                                    employee_id=identity,
                                    last_seen_time=current_time,
                                    last_camera_id=camera_config.camera_id,
                                    embedding_history=deque(maxlen=EMBEDDING_HISTORY_SIZE),
                                    work_status="working")
                            track = self.global_tracks[identity]
                            track.last_seen_time = current_time
                            track.last_camera_id = camera_config.camera_id
                            track.confidence_score = score
                            track.embedding_history.append(embedding)
                            state = self.tracking_states.get(identity, TrackingState(
                                position_history=[], velocity=(0, 0),
                                predicted_position=(0, 0), confidence_history=[], quality_history=[]))
                            state.position_history.append((center_x, center_y))
                            state.confidence_history.append(score)
                            state.quality_history.append(quality_metrics)
                            if len(state.position_history) >= 2:
                                dx = state.position_history[-1][0] - state.position_history[-2][0]
                                dy = state.position_history[-1][1] - state.position_history[-2][1]
                                state.velocity = (dx, dy)
                            self.tracking_states[identity] = state
                            self._check_tripwire_crossing(identity, center_x, center_y, camera_config, frame_width, frame_height)
                        if score > 0.8:
                            self._update_embeddings(identity, embedding)
                    else:
                        identity = "unknown"
                consistent_track_id = self._get_consistent_track_id(identity, camera_config.camera_id)
                color = (0, 255, 0) if identity != "unknown" else (0, 0, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                quality_text = f"Q:{quality_metrics.overall_quality:.2f}"
                cv2.putText(frame, quality_text, (bbox[0], bbox[1] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                metadata = load_employee_metadata(identity) if identity != "unknown" else None
                if metadata:
                    if isinstance(metadata, dict):
                        employee_name = metadata.get('employee_name', identity)
                    else:
                        employee_name = metadata.employee_name
                    label = f"{employee_name} ({score:.2f})"
                else:
                    label = f"{consistent_track_id} ({score:.2f})"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            self.draw_tripwires(frame, camera_config)
        cap.release()
    def start_multi_camera_tracking(self):
        try:
            for camera_config in CAMERAS:
                thread = threading.Thread(
                    target=self.process_camera,
                    args=(camera_config,),
                    daemon=True)
                thread.start()
                self.camera_threads.append(thread)
                time.sleep(1)
            while not self.shutdown_flag.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown_flag.set()
        finally:
            for thread in self.camera_threads:
                if thread.is_alive():
                    thread.join(timeout=2)
    def shutdown(self):
        if self.embedding_update_worker and self.embedding_update_worker.is_alive():
            self.embedding_update_queue.put(None)
            self.embedding_update_worker.join(timeout=5)
        self.shutdown_flag.set()
        self.api_logger.shutdown()
        for thread in self.camera_threads:
            if thread.is_alive():
                thread.join(timeout=2)
    def get_identity_info(self, face):
        embedding = face.embedding.astype('float32')
        return self._compute_embedding_similarity(embedding)
    def is_active(self):
        return not self.shutdown_flag.is_set() and len(self.camera_threads) > 0
    def get_latest_frame(self, camera_id: int):
        """Get the latest frame from the specified camera"""
        with self.frame_locks.get(camera_id, threading.Lock()):
            if camera_id in self.latest_frames:
                return self.latest_frames[camera_id]
            return None
class FaceTrackingPipeline:
    def __init__(self):
        # self.system = FaceTrackingSystem(self.face_app)
        # self.pipeline_thread = None
        # self.logger = get_logger(__name__)
        # Initialize InsightFace
        self.face_app = FaceAnalysis(name='antelopev2')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.system = FaceTrackingSystem(self.face_app)
        self.pipeline_thread = None
        self.logger = get_logger(__name__)
        self.face_app = self.system.face_app
    def start(self):
        """Start the face tracking pipeline"""
        if self.pipeline_thread is None or not self.pipeline_thread.is_alive():
            self.pipeline_thread = threading.Thread(
                target=self.system.start_multi_camera_tracking,
                daemon=True)
            self.pipeline_thread.start()
            log_message("[Pipeline] Face Tracking Pipeline started.")
        else:
            log_message("[Pipeline] Pipeline is already running")
    def stop(self):
        """Stop the face tracking pipeline"""
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.system.shutdown()
            self.pipeline_thread.join(timeout=5)
            log_message("[Pipeline] Face Tracking Pipeline stopped.")
        else:
            log_message("[Pipeline] Pipeline is not running")
    def get_camera_frame(self, camera_id: int):
        """Get the latest frame from the specified camera"""
        return self.system.get_latest_frame(camera_id)
    def get_all_employees(self):
        """Get all registered employees"""
        return self.system.db_manager.get_all_employees()
    def get_latest_attendance(self, limit=100):
        """Get latest attendance records"""
        return self.system.get_attendance_history(limit=limit)
    def get_active_employees(self):
        """Get currently active employees (in the building)"""
        active = []
        for emp in self.get_all_employees():
            if self.system._check_employee_work_status(emp.id):
                active.append(emp)
        return active
    def get_last_seen_location(self, employee_id):
        """Get last seen location for an employee"""
        record = self.system.db_manager.get_latest_attendance_by_employee(employee_id)
        return record.camera_id if record else None
def start_tracking_service():
    """Start the face tracking system as a service"""
    global system_instance, is_tracking_running, start_time
    if is_tracking_running:
        log_message("Tracking service is already running")
        return
    log_message("Starting tracking service...")
    face_app = FaceAnalysis(name='antelopev2')
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    system_instance = FaceTrackingSystem(face_app)
    tracking_thread = threading.Thread(
        target=system_instance.start_multi_camera_tracking, 
        daemon=True)
    tracking_thread.start()
    is_tracking_running = True
    start_time = time.time()
    log_message("Tracking service started")
def shutdown_tracking_service():
    """Shutdown the face tracking service"""
    global system_instance, is_tracking_running
    if system_instance and is_tracking_running:
        log_message("Shutting down tracking service...")
        system_instance.shutdown()
        is_tracking_running = False
        log_message("Tracking service stopped")
    else:
        log_message("Tracking service is not running")
def get_system_status():
    """Get current system status and statistics"""
    global system_stats, start_time
    if start_time:
        system_stats["uptime"] = time.time() - start_time
    return system_stats
def get_live_faces():
    """Get latest detected faces"""
    return latest_faces
def get_attendance_data():
    """Get latest attendance records"""
    return list(latest_attendance)
def get_logs(n=100):
    """Get recent logs from buffer"""
    return log_buffer[-n:]
def generate_mjpeg(camera_id: int):
    """Yield MJPEG stream for FastAPI."""
    while is_tracking_running:
        if system_instance:
            frame = system_instance.get_latest_frame(camera_id)
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)