from typing import List, Optional
from pydantic import BaseSettings, validator
import os
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    app_name: str = "Facial Recognition SPA"
    app_version: str = "1.0.0"
    debug: bool = False
    api_v1_str: str = "/api/v1"
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "face_tracking_spa"
    db_user: str = "postgres"
    db_password: str
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Face Recognition
    face_detection_threshold: float = 0.5
    face_match_threshold: float = 0.6
    max_embedding_update_cooldown: int = 10
    global_track_timeout: int = 300
    embedding_history_size: int = 5
    
    # Camera Configuration
    default_camera_resolution_width: int = 1280
    default_camera_resolution_height: int = 720
    default_camera_fps: int = 15
    video_stream_quality: float = 0.8
    
    # File Storage
    upload_dir: str = "uploads"
    max_upload_size: int = 10485760  # 10MB
    allowed_image_extensions: List[str] = [".jpg", ".jpeg", ".png"]
    
    # CORS
    backend_cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    @validator("backend_cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # External API Configuration (Zoho - Optional)
    zoho_api_base_url: Optional[str] = None
    zoho_access_token: Optional[str] = None
    zoho_refresh_token: Optional[str] = None
    zoho_client_id: Optional[str] = None
    zoho_client_secret: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file_path: str = "logs/app.log"
    
    # Development
    reload: bool = False
    workers: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)