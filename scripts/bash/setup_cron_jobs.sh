#!/bin/bash
"""
Setup Cron Jobs for Automated Data Pipeline

This script sets up cron jobs to ensure continuous data freshness
and information management for your Polymarket database.

Usage:
    ./setup_cron_jobs.sh     # Interactive setup
    ./setup_cron_jobs.sh -y  # Non-interactive setup with defaults

Cron Jobs Created:
    - Data Pipeline: Every 10 minutes (full pipeline)
    - Market Refresh: Every 5 minutes (data only)
    - Information Management: Every 30 minutes (cleanup/summarization)
    - Health Check: Every hour (monitoring/alerts)
    - Database Backup: Daily at 2 AM
"""

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$PROJECT_DIR/scripts"
LOG_DIR="$PROJECT_DIR/logs"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
BACKUP_DIR="$PROJECT_DIR/backups"

# Default intervals (in minutes)
DATA_PIPELINE_INTERVAL=10      # Full pipeline every 10 minutes
MARKET_REFRESH_INTERVAL=5      # Market data refresh every 5 minutes
INFO_MGMT_INTERVAL=30          # Information management every 30 minutes
HEALTH_CHECK_INTERVAL=60       # Health checks every hour
BACKUP_INTERVAL_DAILY=1440     # Database backup daily

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
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

check_dependencies() {
    log_info "Checking dependencies..."

    # Check if Python is available
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check if cron is available
    if ! command -v crontab &> /dev/null; then
        log_error "Cron is not installed or not in PATH"
        exit 1
    fi

    # Check if project directory exists
    if [ ! -d "$PROJECT_DIR" ]; then
        log_error "Project directory does not exist: $PROJECT_DIR"
        exit 1
    fi

    # Check if data pipeline script exists
    if [ ! -f "$SCRIPT_DIR/python/data_pipeline.py" ]; then
        log_error "Data pipeline script not found: $SCRIPT_DIR/python/data_pipeline.py"
        exit 1
    fi

    log_success "All dependencies satisfied"
}

setup_directories() {
    log_info "Setting up directories..."

    # Create log directory
    mkdir -p "$LOG_DIR"
    log_success "Created log directory: $LOG_DIR"

    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    log_success "Created backup directory: $BACKUP_DIR"
}

create_backup_script() {
    local backup_script="$SCRIPT_DIR/bash/backup_database.sh"

    cat > "$backup_script" << 'EOF'
#!/bin/bash
# Database Backup Script

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="$PROJECT_DIR/backups"
DATA_DIR="$PROJECT_DIR/data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup filename
BACKUP_FILE="$BACKUP_DIR/markets_backup_$TIMESTAMP.db"

# Perform backup
if cp "$DATA_DIR/markets.db" "$BACKUP_FILE"; then
    echo "$(date): Database backup created: $BACKUP_FILE" >> "$PROJECT_DIR/logs/backup.log"

    # Clean up old backups (keep last 7 days)
    find "$BACKUP_DIR" -name "markets_backup_*.db" -mtime +7 -delete
    echo "$(date): Old backups cleaned up" >> "$PROJECT_DIR/logs/backup.log"
else
    echo "$(date): ERROR - Database backup failed" >> "$PROJECT_DIR/logs/backup.log"
    exit 1
fi
EOF

    chmod +x "$backup_script"
    log_success "Created backup script: $backup_script"
}

setup_cron_jobs() {
    log_info "Setting up cron jobs..."

    # Get current crontab
    current_crontab=$(crontab -l 2>/dev/null || echo "")

    # Remove existing Polymarket cron jobs
    cleaned_crontab=$(echo "$current_crontab" | grep -v "polymarket\|data_pipeline\|markets_backup" || echo "")

    # Add new cron jobs
    new_crontab="$cleaned_crontab"

    # Data Pipeline - Full pipeline every 10 minutes
    new_crontab="$new_crontab
*/$DATA_PIPELINE_INTERVAL * * * * cd $PROJECT_DIR && python scripts/python/data_pipeline.py >> $PIPELINE_LOG 2>&1"

    # Market Refresh - Data only every 5 minutes
    if [ "$DATA_PIPELINE_INTERVAL" != "$MARKET_REFRESH_INTERVAL" ]; then
        new_crontab="$new_crontab
*/$MARKET_REFRESH_INTERVAL * * * * cd $PROJECT_DIR && python scripts/python/data_pipeline.py --data-only >> $PIPELINE_LOG 2>&1"
    fi

    # Information Management - Every 30 minutes
    new_crontab="$new_crontab
*/$INFO_MGMT_INTERVAL * * * * cd $PROJECT_DIR && python scripts/python/data_pipeline.py --info-management-only >> $PIPELINE_LOG 2>&1"

    # Health Check - Every hour
    new_crontab="$new_crontab
0 * * * * cd $PROJECT_DIR && python scripts/python/data_pipeline.py --continuous --interval 3600 >> $PIPELINE_LOG 2>&1"

    # Database Backup - Daily at 2 AM
    new_crontab="$new_crontab
0 2 * * * $SCRIPT_DIR/bash/backup_database.sh"

    # Install new crontab
    echo "$new_crontab" | crontab -

    log_success "Cron jobs installed successfully"
}

display_cron_jobs() {
    log_info "Current Polymarket cron jobs:"
    echo
    crontab -l | grep -E "(polymarket|data_pipeline|markets_backup)" || echo "No Polymarket cron jobs found"
    echo
}

test_cron_setup() {
    log_info "Testing cron setup..."

    # Test data pipeline script
    if cd "$PROJECT_DIR" && python scripts/python/data_pipeline.py --help > /dev/null 2>&1; then
        log_success "Data pipeline script is accessible"
    else
        log_error "Data pipeline script test failed"
        return 1
    fi

    # Test backup script
    if [ -x "$SCRIPT_DIR/bash/backup_database.sh" ]; then
        log_success "Backup script is executable"
    else
        log_error "Backup script is not executable"
        return 1
    fi

    return 0
}

cleanup_old_backups() {
    log_info "Cleaning up old backup files..."

    # Remove old backup files (keep last 7)
    if [ -d "$BACKUP_DIR" ]; then
        backup_count=$(find "$BACKUP_DIR" -name "markets_backup_*.db" | wc -l)
        if [ "$backup_count" -gt 7 ]; then
            find "$BACKUP_DIR" -name "markets_backup_*.db" -mtime +7 -delete
            log_success "Cleaned up old backup files"
        else
            log_info "No old backup files to clean up"
        fi
    fi
}

main() {
    local non_interactive=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -y|--yes)
                non_interactive=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Usage: $0 [-y|--yes]"
                exit 1
                ;;
        esac
    done

    echo
    log_info "🔄 Polymarket Cron Job Setup"
    echo
    log_info "This will set up automated data pipeline and information management."
    log_info "The following cron jobs will be created:"
    echo "  - Data Pipeline: Every $DATA_PIPELINE_INTERVAL minutes"
    echo "  - Market Refresh: Every $MARKET_REFRESH_INTERVAL minutes"
    echo "  - Info Management: Every $INFO_MGMT_INTERVAL minutes"
    echo "  - Health Checks: Every hour"
    echo "  - Database Backup: Daily at 2 AM"
    echo

    if [ "$non_interactive" = false ]; then
        read -p "Continue with setup? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Setup cancelled by user"
            exit 0
        fi
    fi

    # Run setup steps
    check_dependencies
    setup_directories
    create_backup_script
    cleanup_old_backups
    setup_cron_jobs

    # Test setup
    if test_cron_setup; then
        log_success "🎉 Cron job setup completed successfully!"
        echo
        display_cron_jobs

        echo
        log_info "📋 Next Steps:"
        echo "  1. Monitor logs: tail -f $PIPELINE_LOG"
        echo "  2. Check cron: crontab -l"
        echo "  3. Test manually: python scripts/python/data_pipeline.py"
        echo "  4. View backups: ls -la $BACKUP_DIR/"
        echo
        log_info "Your database will now be continuously updated and managed!"
    else
        log_error "❌ Setup completed with errors. Please check the logs."
        exit 1
    fi
}

# Run main function
main "$@"
