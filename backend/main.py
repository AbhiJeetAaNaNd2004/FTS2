import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException
import uvicorn

from backend.config.settings import get_settings
from backend.config.database import create_tables
from backend.core.face_tracking_system import start_face_tracking_system, stop_face_tracking_system
from backend.api.v1 import auth, employees, attendance, enrollment, streaming, admin, users
from backend.core.security import AuthenticationError, AuthorizationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Facial Recognition SPA...")
    
    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created/verified")
        
        # Start face tracking system
        face_tracking_system = start_face_tracking_system()
        logger.info("Face tracking system started")
        
        # Store reference in app state
        app.state.face_tracking_system = face_tracking_system
        
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down Facial Recognition SPA...")
        try:
            stop_face_tracking_system()
            logger.info("Face tracking system stopped")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Facial Recognition Single Page Application for Employee Attendance Tracking",
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure this properly for production
)


# Custom exception handlers
@app.exception_handler(AuthenticationError)
async def authentication_exception_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc), "type": "authentication_error"}
    )


@app.exception_handler(AuthorizationError)
async def authorization_exception_handler(request: Request, exc: AuthorizationError):
    return JSONResponse(
        status_code=403,
        content={"detail": str(exc), "type": "authorization_error"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"}
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for debugging and monitoring."""
    start_time = asyncio.get_event_loop().time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = asyncio.get_event_loop().time() - start_time
    
    # Log request details
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s - "
        f"Client: {request.client.host}"
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check database connection
        from backend.config.database import DatabaseManager
        db_manager = DatabaseManager()
        db_healthy = db_manager.health_check()
        
        # Check face tracking system
        face_tracking_healthy = hasattr(app.state, 'face_tracking_system') and app.state.face_tracking_system is not None
        
        status = "healthy" if db_healthy and face_tracking_healthy else "unhealthy"
        
        return {
            "status": status,
            "database": "healthy" if db_healthy else "unhealthy",
            "face_tracking": "healthy" if face_tracking_healthy else "unhealthy",
            "version": settings.app_version,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "version": settings.app_version
            }
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


# Include API routers
app.include_router(
    auth.router,
    prefix=f"{settings.api_v1_str}/auth",
    tags=["authentication"]
)

app.include_router(
    employees.router,
    prefix=f"{settings.api_v1_str}/employees",
    tags=["employees"]
)

app.include_router(
    attendance.router,
    prefix=f"{settings.api_v1_str}/attendance",
    tags=["attendance"]
)

app.include_router(
    enrollment.router,
    prefix=f"{settings.api_v1_str}/enrollment",
    tags=["enrollment"]
)

app.include_router(
    streaming.router,
    prefix=f"{settings.api_v1_str}/streaming",
    tags=["streaming"]
)

app.include_router(
    admin.router,
    prefix=f"{settings.api_v1_str}/admin",
    tags=["admin"]
)

app.include_router(
    users.router,
    prefix=f"{settings.api_v1_str}/users",
    tags=["users"]
)


# WebSocket endpoint for real-time updates
@app.websocket("/ws/{connection_type}")
async def websocket_endpoint(websocket, connection_type: str):
    """WebSocket endpoint for real-time updates."""
    try:
        await websocket.accept()
        
        # Get face tracking system from app state
        if hasattr(app.state, 'face_tracking_system'):
            face_tracking_system = app.state.face_tracking_system
            await face_tracking_system.websocket_manager.connect(websocket, connection_type)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                await face_tracking_system.websocket_manager.disconnect(websocket, connection_type)
        else:
            await websocket.close(code=1000, reason="Service unavailable")
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        await websocket.close(code=1000, reason="Connection error")


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.reload,
        workers=settings.workers,
        log_level=settings.log_level.lower()
    )