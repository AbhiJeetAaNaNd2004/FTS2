# Facial Recognition SPA - Project Structure

## Overview
This document outlines the complete file structure for the facial recognition Single Page Application (SPA) system. The application is built with FastAPI backend, PostgreSQL database, and React frontend, designed for internal company use with role-based authentication.

## Project Directory Structure

```
facial-recognition-spa/
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── setup.py
├── run.sh
│
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py            # Application configuration
│   │   └── database.py            # Database configuration
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── face_tracking_system.py # Core face recognition pipeline (from API_experimentation.py)
│   │   ├── face_enroller.py       # Face enrollment logic
│   │   ├── security.py            # JWT authentication and authorization
│   │   ├── exceptions.py          # Custom exception handlers
│   │   └── middleware.py          # Custom middleware
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py            # SQLAlchemy models (enhanced from db_models.py)
│   │   ├── schemas.py             # Pydantic models for API
│   │   └── enums.py               # Application enums
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                # Dependency injection
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── auth.py            # Authentication endpoints
│   │       ├── employees.py       # Employee management
│   │       ├── attendance.py      # Attendance logs
│   │       ├── enrollment.py      # Face enrollment endpoints
│   │       ├── streaming.py       # Video streaming endpoints
│   │       ├── admin.py           # Admin-only endpoints
│   │       └── users.py           # User management (super admin)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py        # Authentication business logic
│   │   ├── employee_service.py    # Employee management logic
│   │   ├── attendance_service.py  # Attendance processing
│   │   ├── enrollment_service.py  # Face enrollment processing
│   │   ├── streaming_service.py   # Video streaming logic
│   │   └── admin_service.py       # Admin operations
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── face_utils.py          # Face processing utilities
│   │   ├── image_utils.py         # Image processing helpers
│   │   ├── db_utils.py            # Database utilities
│   │   └── validation.py          # Input validation helpers
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py            # Test configuration
│       ├── test_auth.py
│       ├── test_employees.py
│       ├── test_attendance.py
│       ├── test_enrollment.py
│       └── test_streaming.py
│
├── frontend/
│   ├── package.json
│   ├── package-lock.json
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── manifest.json
│   │
│   └── src/
│       ├── index.js               # React entry point
│       ├── App.js                 # Main App component
│       ├── App.css
│       │
│       ├── components/
│       │   ├── common/
│       │   │   ├── Layout.js      # Main layout component
│       │   │   ├── Navbar.js      # Navigation bar
│       │   │   ├── Sidebar.js     # Sidebar navigation
│       │   │   ├── LoadingSpinner.js
│       │   │   ├── ErrorBoundary.js
│       │   │   └── ProtectedRoute.js
│       │   │
│       │   ├── auth/
│       │   │   ├── LoginForm.js
│       │   │   └── LogoutButton.js
│       │   │
│       │   ├── dashboard/
│       │   │   ├── EmployeeDashboard.js
│       │   │   ├── AdminDashboard.js
│       │   │   ├── SuperAdminDashboard.js
│       │   │   └── StatsCards.js
│       │   │
│       │   ├── attendance/
│       │   │   ├── AttendanceList.js
│       │   │   ├── AttendanceTable.js
│       │   │   ├── AttendanceFilters.js
│       │   │   └── PresentEmployees.js
│       │   │
│       │   ├── employees/
│       │   │   ├── EmployeeList.js
│       │   │   ├── EmployeeCard.js
│       │   │   ├── EmployeeForm.js
│       │   │   └── EmployeeDetails.js
│       │   │
│       │   ├── enrollment/
│       │   │   ├── EnrollmentForm.js
│       │   │   ├── ImageUpload.js
│       │   │   ├── FacePreview.js
│       │   │   └── EnrollmentWizard.js
│       │   │
│       │   ├── streaming/
│       │   │   ├── CameraFeed.js
│       │   │   ├── CameraGrid.js
│       │   │   ├── CameraControls.js
│       │   │   └── StreamViewer.js
│       │   │
│       │   └── admin/
│       │       ├── UserManagement.js
│       │       ├── RoleManagement.js
│       │       ├── SystemLogs.js
│       │       └── Settings.js
│       │
│       ├── hooks/
│       │   ├── useAuth.js          # Authentication hook
│       │   ├── useApi.js           # API interaction hook
│       │   ├── useWebSocket.js     # WebSocket connection hook
│       │   └── useLocalStorage.js  # Local storage hook
│       │
│       ├── services/
│       │   ├── api.js              # API client configuration
│       │   ├── auth.js             # Authentication API calls
│       │   ├── employees.js        # Employee API calls
│       │   ├── attendance.js       # Attendance API calls
│       │   ├── enrollment.js       # Enrollment API calls
│       │   └── streaming.js        # Streaming API calls
│       │
│       ├── utils/
│       │   ├── constants.js        # Application constants
│       │   ├── helpers.js          # Utility functions
│       │   ├── validation.js       # Form validation
│       │   └── formatters.js       # Data formatting helpers
│       │
│       ├── context/
│       │   ├── AuthContext.js      # Authentication context
│       │   └── AppContext.js       # Global application context
│       │
│       └── styles/
│           ├── globals.css         # Global styles
│           ├── components.css      # Component-specific styles
│           └── themes.css          # Theme definitions
│
├── database/
│   ├── migrations/
│   │   ├── 001_initial_schema.sql
│   │   ├── 002_add_users_roles.sql
│   │   └── 003_add_indexes.sql
│   │
│   ├── seeds/
│   │   ├── default_roles.sql      # Default roles and permissions
│   │   └── admin_user.sql         # Default admin user
│   │
│   └── scripts/
│       ├── init_db.py             # Database initialization
│       └── backup.py              # Database backup utility
│
├── deployment/
│   ├── docker/
│   │   ├── backend.Dockerfile
│   │   ├── frontend.Dockerfile
│   │   └── nginx.Dockerfile
│   │
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml
│   │   ├── backend-deployment.yaml
│   │   ├── frontend-deployment.yaml
│   │   ├── postgres-deployment.yaml
│   │   ├── nginx-deployment.yaml
│   │   └── ingress.yaml
│   │
│   └── scripts/
│       ├── deploy.sh              # Deployment script
│       ├── backup.sh              # Backup script
│       └── restore.sh             # Restore script
│
├── docs/
│   ├── API.md                     # API documentation
│   ├── DEPLOYMENT.md              # Deployment guide
│   ├── USER_GUIDE.md              # User guide
│   └── DEVELOPMENT.md             # Development setup
│
└── scripts/
    ├── start_dev.sh               # Development startup script
    ├── test.sh                    # Test runner script
    ├── lint.sh                    # Code linting script
    └── build.sh                   # Build script
```

## Key Components Description

### Backend Architecture

1. **Core System (`backend/core/`)**
   - `face_tracking_system.py`: Main face recognition pipeline (preserved from API_experimentation.py)
   - `face_enroller.py`: Face enrollment system (enhanced from face_enroller.py)
   - `security.py`: JWT-based authentication and role authorization

2. **API Layer (`backend/api/v1/`)**
   - RESTful endpoints for all functionality
   - WebSocket endpoints for real-time streaming
   - Role-based access control on all endpoints

3. **Services Layer (`backend/services/`)**
   - Business logic separated from API controllers
   - Database operations and face processing logic
   - Integration with existing enrollment system

### Frontend Architecture

1. **Component Structure**
   - Role-based dashboard components
   - Reusable UI components with minimal, clean design
   - Real-time streaming components for camera feeds

2. **State Management**
   - React Context for global state
   - Custom hooks for API interactions
   - Real-time updates via WebSocket connections

### Database Schema

Enhanced from existing db_models.py with additional tables for:
- User authentication and authorization
- Role-based permissions
- System audit logs
- Camera configurations

### Security Features

1. **Authentication**
   - JWT-based token authentication
   - Role-based access control (Employee, Admin, Super Admin)
   - Session management and token refresh

2. **Authorization**
   - Fine-grained permissions per role
   - API endpoint protection
   - Resource-level access control

### Deployment

1. **Containerization**
   - Docker containers for all services
   - Docker Compose for local development
   - Production-ready container configurations

2. **Scalability**
   - Horizontal scaling support
   - Load balancing configuration
   - Database connection pooling

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL, OpenCV, InsightFace
- **Frontend**: React, Axios, Material-UI (minimal theme)
- **Authentication**: JWT tokens
- **Streaming**: WebSocket, Server-Sent Events
- **Deployment**: Docker, Docker Compose, Nginx
- **Database**: PostgreSQL with connection pooling
- **Testing**: pytest (backend), Jest (frontend)

## Preserved Functionality

All core face recognition functionality from the existing system is preserved:
- Face detection and tracking pipeline
- Embedding generation and storage
- Database integration
- Employee enrollment system
- Attendance logging

The GUI components from API_experimentation.py are replaced with web-based interfaces while maintaining all backend logic intact.