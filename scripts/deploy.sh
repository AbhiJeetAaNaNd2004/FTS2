#!/bin/bash

# Facial Recognition SPA Deployment Script
# This script automates the deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
ENVIRONMENT="production"
SKIP_BUILD=false
INIT_DB=true
START_SERVICES=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --no-db-init)
            INIT_DB=false
            shift
            ;;
        --no-start)
            START_SERVICES=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --env ENV          Set environment (production|development) [default: production]"
            echo "  --skip-build       Skip Docker image building"
            echo "  --no-db-init       Skip database initialization"
            echo "  --no-start         Don't start services after setup"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "Starting Facial Recognition SPA deployment..."
log_info "Environment: $ENVIRONMENT"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_warning "Running as root. Consider using a non-root user for security."
fi

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "docker-compose.yml" ]]; then
        log_error "docker-compose.yml not found. Please run this script from the project root directory."
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        log_warning ".env file not found. Creating from .env.example..."
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            log_info ".env file created. Please review and update the configuration."
        else
            log_error ".env.example file not found. Cannot create .env file."
            exit 1
        fi
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        log_warning "Less than 10GB disk space available. Consider freeing up space."
    fi
    
    log_success "System requirements check completed."
}

# Generate secure secret key if not set
generate_secret_key() {
    log_info "Checking secret key configuration..."
    
    if grep -q "your_super_secret_key" .env; then
        log_info "Generating secure secret key..."
        SECRET_KEY=$(openssl rand -hex 32)
        sed -i "s/SECRET_KEY=your_super_secret_key_here_please_change_in_production/SECRET_KEY=$SECRET_KEY/g" .env
        log_success "Secret key generated and updated in .env file."
    else
        log_info "Secret key already configured."
    fi
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p uploads logs
    
    # Set appropriate permissions
    chmod 755 uploads logs
    
    log_success "Directories created and configured."
}

# Stop existing services
stop_services() {
    log_info "Stopping existing services..."
    docker-compose down --remove-orphans || true
    log_success "Existing services stopped."
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping Docker image build (--skip-build flag provided)."
        return
    fi
    
    log_info "Building Docker images..."
    
    # Build backend image
    log_info "Building backend image..."
    docker-compose build backend
    
    # Build frontend image
    log_info "Building frontend image..."
    docker-compose build frontend
    
    log_success "Docker images built successfully."
}

# Start services
start_services() {
    if [[ "$START_SERVICES" == "false" ]]; then
        log_info "Skipping service startup (--no-start flag provided)."
        return
    fi
    
    log_info "Starting services..."
    
    # Start database first
    log_info "Starting database..."
    docker-compose up -d postgres
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U postgres; then
            break
        fi
        if [[ $i -eq 30 ]]; then
            log_error "Database failed to start within 30 seconds."
            exit 1
        fi
        sleep 1
    done
    log_success "Database is ready."
    
    # Start remaining services
    log_info "Starting all services..."
    docker-compose up -d
    
    log_success "All services started."
}

# Initialize database
initialize_database() {
    if [[ "$INIT_DB" == "false" ]]; then
        log_info "Skipping database initialization (--no-db-init flag provided)."
        return
    fi
    
    log_info "Initializing database..."
    
    # Wait for backend to be ready
    log_info "Waiting for backend service to be ready..."
    for i in {1..60}; do
        if docker-compose exec -T backend curl -f http://localhost:8000/health &> /dev/null; then
            break
        fi
        if [[ $i -eq 60 ]]; then
            log_error "Backend failed to start within 60 seconds."
            exit 1
        fi
        sleep 1
    done
    
    # Run database initialization
    log_info "Running database initialization script..."
    docker-compose exec -T backend python database/scripts/init_db.py
    
    log_success "Database initialized successfully."
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check service status
    if ! docker-compose ps | grep -q "Up"; then
        log_error "Some services are not running properly."
        docker-compose ps
        exit 1
    fi
    
    # Check health endpoints
    log_info "Checking health endpoints..."
    
    # Backend health check
    if ! curl -f http://localhost:8000/health &> /dev/null; then
        log_error "Backend health check failed."
        exit 1
    fi
    
    # Frontend check
    if ! curl -f http://localhost:3000 &> /dev/null; then
        log_error "Frontend health check failed."
        exit 1
    fi
    
    log_success "All health checks passed."
}

# Display access information
display_access_info() {
    log_success "Deployment completed successfully!"
    echo
    echo "=========================================="
    echo "         ACCESS INFORMATION"
    echo "=========================================="
    echo "Frontend URL:      http://localhost:3000"
    echo "API Documentation: http://localhost:8000/docs"
    echo "Health Check:      http://localhost:8000/health"
    echo
    echo "Default Credentials:"
    echo "  Admin:     Username: admin,    Password: admin123"
    echo "  Employee:  Username: employee, Password: employee123"
    echo
    log_warning "IMPORTANT: Change default passwords in production!"
    echo
    echo "Useful Commands:"
    echo "  View logs:        docker-compose logs -f"
    echo "  Stop services:    docker-compose down"
    echo "  Restart:          docker-compose restart"
    echo "  Update:           git pull && $0"
    echo "=========================================="
}

# Cleanup function for error handling
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Deployment failed. Cleaning up..."
        docker-compose down --remove-orphans || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main deployment sequence
main() {
    check_requirements
    generate_secret_key
    setup_directories
    stop_services
    build_images
    start_services
    initialize_database
    verify_deployment
    display_access_info
}

# Run main function
main

log_success "Deployment script completed successfully!"