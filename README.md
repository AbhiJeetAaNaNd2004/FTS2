# Facial Recognition SPA

A comprehensive Single Page Application (SPA) for employee attendance tracking using facial recognition technology. Built with FastAPI backend, React frontend, and PostgreSQL database.

## 🚀 Features

### Core Functionality
- **Real-time Face Recognition**: Advanced face detection and recognition using InsightFace
- **Employee Attendance Tracking**: Automatic check-in/check-out logging
- **Multi-camera Support**: Support for multiple camera feeds with GPU acceleration
- **Live Video Streaming**: Real-time camera feeds with WebSocket streaming
- **Face Enrollment System**: Easy enrollment of new employees with multiple images

### User Roles & Permissions
- **Regular Employee**: View own attendance logs and present employees
- **Admin**: Full employee management, face enrollment, camera access
- **Super Admin**: User management, role assignments, system settings

### Technical Features
- **JWT Authentication**: Secure token-based authentication with refresh tokens
- **Role-based Access Control**: Granular permissions system
- **Real-time Updates**: WebSocket connections for live updates
- **Database Integration**: PostgreSQL with SQLAlchemy ORM
- **Containerized Deployment**: Docker and Docker Compose ready
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Audit Logging**: Comprehensive system and user activity logging

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **Core System**: Face tracking pipeline (preserved from existing system)
- **API Layer**: RESTful endpoints with role-based access control
- **Authentication**: JWT-based security with session management
- **Database**: PostgreSQL with connection pooling
- **Real-time**: WebSocket support for live updates

### Frontend (React)
- **Component-based**: Modular React components with hooks
- **State Management**: Context API for global state
- **Real-time**: WebSocket integration for live updates
- **Responsive Design**: Mobile-friendly minimal UI
- **API Integration**: Axios-based API client

### Database Schema
- Enhanced employee and face embedding storage
- User authentication and authorization tables
- Attendance and tracking records
- System logs and audit trails

## 📋 Prerequisites

### System Requirements
- Linux server (Ubuntu 20.04+ recommended)
- Docker and Docker Compose
- NVIDIA GPU (optional, for better performance)
- Camera(s) connected to the system

### Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU, 50GB storage
- **Recommended**: 16GB RAM, 8-core CPU, 100GB storage, NVIDIA GPU

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd facial-recognition-spa

# Copy environment configuration
cp .env.example .env
# Edit .env with your configuration
nano .env
```

### 2. Environment Configuration

Update `.env` file with your settings:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=face_tracking_spa
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Security Configuration
SECRET_KEY=your_super_secret_key_here_please_change_in_production

# Face Recognition Configuration
FACE_DETECTION_THRESHOLD=0.5
FACE_MATCH_THRESHOLD=0.6

# Camera Configuration (adjust device paths as needed)
DEFAULT_CAMERA_RESOLUTION_WIDTH=1280
DEFAULT_CAMERA_RESOLUTION_HEIGHT=720
```

### 3. Deploy with Docker

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### 4. Initialize Database

```bash
# Run database initialization
docker-compose exec backend python database/scripts/init_db.py
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Default Credentials
- **Admin**: Username: `admin`, Password: `admin123`
- **Employee**: Username: `employee`, Password: `employee123`

**⚠️ IMPORTANT: Change default passwords in production!**

## 🔧 Development Setup

### Backend Development

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH=$PWD
export DB_PASSWORD=your_password
export SECRET_KEY=your_secret_key

# Run database initialization
python database/scripts/init_db.py

# Start development server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 📚 API Documentation

### Authentication Endpoints
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout
- `POST /api/v1/auth/refresh` - Refresh access token
- `GET /api/v1/auth/me` - Get current user info

### Employee Management
- `GET /api/v1/employees` - List employees
- `POST /api/v1/employees` - Create employee
- `GET /api/v1/employees/{id}` - Get employee details
- `PUT /api/v1/employees/{id}` - Update employee
- `DELETE /api/v1/employees/{id}` - Delete employee

### Face Enrollment
- `POST /api/v1/enrollment/enroll` - Enroll new face
- `POST /api/v1/enrollment/add-images` - Add images to existing employee
- `DELETE /api/v1/enrollment/{employee_id}/embeddings/{embedding_id}` - Delete embedding

### Attendance Tracking
- `GET /api/v1/attendance/records` - Get attendance records
- `GET /api/v1/attendance/present` - Get currently present employees
- `GET /api/v1/attendance/employee/{id}` - Get employee attendance

### Live Streaming
- `GET /api/v1/streaming/cameras` - List available cameras
- `WebSocket /ws/camera_feed` - Live camera feed
- `WebSocket /ws/attendance_updates` - Real-time attendance updates

## 🎛️ User Guide

### Admin Tasks

#### Enrolling a New Employee
1. Navigate to **Admin Dashboard** → **Enrollment**
2. Fill in employee details (ID, name, department, etc.)
3. Upload multiple clear face images (minimum 3 recommended)
4. Click **Enroll Employee**
5. System will process images and create face embeddings

#### Managing Employees
1. Go to **Employee Management**
2. View list of all employees with their status
3. Edit employee information as needed
4. Delete employees (removes all associated data)

#### Viewing Live Camera Feed
1. Access **Live Streaming** section
2. Select camera from dropdown
3. Monitor real-time face detection and recognition
4. View attendance events as they occur

### Employee Tasks

#### Viewing Own Attendance
1. Login with employee credentials
2. Dashboard shows personal attendance summary
3. View detailed logs in **My Attendance**
4. Check current work status

#### Viewing Present Colleagues
1. Navigate to **Present Employees**
2. See who is currently in the office
3. View check-in times and departments

## 🔒 Security Features

### Authentication & Authorization
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- Session management with automatic expiry
- Account lockout after failed login attempts
- Rate limiting on authentication endpoints

### Data Protection
- Password hashing with bcrypt
- SQL injection prevention with SQLAlchemy ORM
- CORS protection with configurable origins
- Input validation and sanitization
- Audit logging for all user actions

### Face Recognition Security
- Encrypted storage of face embeddings
- Configurable confidence thresholds
- Embedding cleanup and rotation
- Quality-based face validation

## 🛠️ Configuration

### Camera Configuration
Update camera settings in the database or environment variables:

```env
# Camera resolution
DEFAULT_CAMERA_RESOLUTION_WIDTH=1280
DEFAULT_CAMERA_RESOLUTION_HEIGHT=720

# Face recognition thresholds
FACE_DETECTION_THRESHOLD=0.5
FACE_MATCH_THRESHOLD=0.6

# Performance settings
MAX_EMBEDDING_UPDATE_COOLDOWN=10
GLOBAL_TRACK_TIMEOUT=300
```

### Database Configuration
Configure PostgreSQL connection:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=face_tracking_spa
DB_USER=postgres
DB_PASSWORD=your_secure_password
```

### Security Configuration
Set up JWT and encryption:

```env
SECRET_KEY=your_super_secret_key_change_in_production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

## 📊 Monitoring & Maintenance

### Health Checks
- `/health` endpoint provides system status
- Database connectivity check
- Face tracking system status
- Component-level health reporting

### Logging
- Structured logging with configurable levels
- Audit trail for all user actions
- System event logging
- Error tracking and reporting

### Database Maintenance
```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres face_tracking_spa > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres face_tracking_spa < backup.sql

# Clean up old sessions
docker-compose exec backend python -c "
from backend.core.security import AuthService
from backend.config.database import SessionLocal
auth_service = AuthService(SessionLocal())
auth_service.cleanup_expired_sessions()
"
```

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check camera devices
ls /dev/video*

# Test camera access
docker-compose exec backend python -c "
import cv2
cap = cv2.VideoCapture(0)
print('Camera accessible:', cap.isOpened())
cap.release()
"
```

#### Database Connection Issues
```bash
# Check database status
docker-compose logs postgres

# Test database connection
docker-compose exec backend python -c "
from backend.config.database import engine
try:
    conn = engine.connect()
    print('Database connected successfully')
    conn.close()
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

#### Face Recognition Not Working
1. Check GPU availability and drivers
2. Verify InsightFace model downloads
3. Check camera permissions and access
4. Verify face detection thresholds

### Performance Optimization

#### GPU Acceleration
```bash
# Check NVIDIA GPU availability
nvidia-smi

# Verify CUDA in container
docker-compose exec backend python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA devices:', torch.cuda.device_count())
"
```

#### Memory Management
- Monitor memory usage with `docker stats`
- Adjust embedding cleanup frequency
- Configure appropriate batch sizes
- Set camera resolution based on hardware

## 📞 Support

### Error Reporting
When reporting issues, please include:
1. System specifications and OS version
2. Docker container logs
3. Error messages and stack traces
4. Steps to reproduce the issue

### Performance Issues
For performance optimization:
1. Check hardware specifications
2. Monitor resource usage
3. Review camera configuration
4. Analyze database query performance

## 📝 License

This project preserves all core facial recognition functionality from the original system while providing a modern web-based interface. The system is designed for internal company use with proper security measures and role-based access control.

---

**Note**: This system contains advanced facial recognition technology. Ensure compliance with local privacy laws and regulations when deploying in production environments. Always inform employees about facial recognition usage and obtain necessary consents.