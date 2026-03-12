#!/bin/bash
# Import a RelBench dataset into PostgreSQL in full (no sampling).
# Tables are created WITHOUT a _full suffix; only the database name carries it.
# Database name defaults to <dataset_short>_full (e.g., f1_full, amazon_full).
#
# Usage:
#   ./import_relbench_full.sh <dataset-name> [options]
#
# Examples:
#   ./import_relbench_full.sh rel-f1
#   ./import_relbench_full.sh rel-amazon --db-name amazon_full
#   ./import_relbench_full.sh rel-hm --keep-data

set -eo pipefail

CONTAINER_NAME="relau2ml-postgres-1"
DB_USER="mlflow"
DB_PASSWORD="mlflow"

# Thiết lập file log
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/import_full_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

log_error() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$message" | tee -a "$LOG_FILE" >&2
}

log_step() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] STEP: $1"
    echo "$message" | tee -a "$LOG_FILE"
}

print_header() {
    local header="========================================="
    echo "" | tee -a "$LOG_FILE"
    echo "$header" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "$header" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

print_info() {
    log "INFO: $1"
}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cleanup() {
    log_step "Starting cleanup process"

    if [ "$KEEP_DATA" != "true" ]; then
        print_header "Cleanup"

        print_info "Removing CSV/SQL files from container..."
        if docker exec $CONTAINER_NAME bash -c "rm -f /tmp/*.csv /tmp/import_*_full.sql" 2>/dev/null; then
            log "Files removed from container"
        else
            log "Nothing to remove from container (or container unreachable)"
        fi

        if [ -d "$DATA_DIR" ]; then
            print_info "Removing local data directory: $DATA_DIR"
            rm -rf "$DATA_DIR"
            log "Data directory removed: $DATA_DIR"
        fi

        if [ "$REMOVE_RELBENCH_CACHE" = "true" ]; then
            RELBENCH_CACHE="$HOME/.cache/relbench"
            if [ -d "$RELBENCH_CACHE" ]; then
                print_info "Removing RelBench cache: $RELBENCH_CACHE"
                rm -rf "$RELBENCH_CACHE"
                log "RelBench cache removed"
            fi
        fi

        echo "Cleanup complete" | tee -a "$LOG_FILE"
    else
        print_info "Keeping data files (KEEP_DATA=true)"
    fi

    log_step "Cleanup completed"
}

trap 'log_error "Script failed on line $LINENO with exit code $?"; cleanup; exit 1' ERR
trap 'cleanup' EXIT

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
show_usage() {
    cat << EOF
Usage: $0 <dataset-name> [options]

Arguments:
  dataset-name          RelBench dataset name (e.g., rel-f1, rel-amazon, rel-hm)

Options:
  --db-name NAME        Target database name (default: <dataset_short>_full)
  --db-password PASS    Database password (default: mlflow)
  --output-dir DIR      Output directory for CSV and SQL files
  --keep-data           Keep CSV and SQL files after import
  --remove-cache        Remove RelBench cache after import
  --help                Show this help message

Description:
  Downloads the full RelBench dataset and imports ALL rows into PostgreSQL.
  No sampling is performed. Table names do NOT carry a _full suffix — only
  the database name does (e.g., stack_full contains a table called votes).

Examples:
  $0 rel-f1
  $0 rel-amazon --db-name amazon_full
  $0 rel-hm --keep-data
  $0 rel-stack --output-dir ./stack_full_data

Supported datasets:
  rel-f1, rel-amazon, rel-hm, rel-stack, rel-trial,
  rel-event, rel-avito, rel-salt, rel-arxiv, rel-ratebeer

Logs are saved to: $LOG_DIR/

EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
log "=========================================="
log "Script started: $0"
log "Arguments: $*"
log "=========================================="

if [ $# -eq 0 ]; then
    show_usage
fi

DATASET_NAME=""
DB_NAME=""
OUTPUT_DIR=""
KEEP_DATA="false"
REMOVE_RELBENCH_CACHE="false"

log_step "Parsing command line arguments"

while [ $# -gt 0 ]; do
    case "$1" in
        --db-name)
            DB_NAME="$2"
            log "Setting DB_NAME=$DB_NAME"
            shift 2
            ;;
        --db-password)
            DB_PASSWORD="$2"
            log "Setting DB_PASSWORD=[HIDDEN]"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            log "Setting OUTPUT_DIR=$OUTPUT_DIR"
            shift 2
            ;;
        --keep-data)
            KEEP_DATA="true"
            log "Setting KEEP_DATA=true"
            shift
            ;;
        --remove-cache)
            REMOVE_RELBENCH_CACHE="true"
            log "Setting REMOVE_RELBENCH_CACHE=true"
            shift
            ;;
        --help)
            show_usage
            ;;
        -*)
            log_error "Unknown option: $1"
            show_usage
            ;;
        *)
            if [ -z "$DATASET_NAME" ]; then
                DATASET_NAME="$1"
                log "Setting DATASET_NAME=$DATASET_NAME"
            else
                log_error "Multiple dataset names provided"
                show_usage
            fi
            shift
            ;;
    esac
done

if [ -z "$DATASET_NAME" ]; then
    log_error "Dataset name is required"
    show_usage
fi

DATASET_SHORT=$(echo "$DATASET_NAME" | sed 's/^rel-//')
log "DATASET_SHORT=$DATASET_SHORT"

if [ -z "$DB_NAME" ]; then
    DB_NAME="${DATASET_SHORT}_full"
    log "DB_NAME set to default: $DB_NAME"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./${DATASET_SHORT}_full_data"
    log "OUTPUT_DIR set to default: $OUTPUT_DIR"
fi

DATA_DIR="$OUTPUT_DIR"
SQL_FILE="$DATA_DIR/import_${DATASET_SHORT}_full.sql"

print_header "RelBench Full Import - $DATASET_NAME"

echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Dataset:       $DATASET_NAME"          | tee -a "$LOG_FILE"
echo "  Database:      $DB_NAME"               | tee -a "$LOG_FILE"
echo "  Container:     $CONTAINER_NAME"        | tee -a "$LOG_FILE"
echo "  Data dir:      $DATA_DIR"              | tee -a "$LOG_FILE"
echo "  Sampling:      DISABLED (full import)" | tee -a "$LOG_FILE"
echo "  Table suffix:  none (DB name carries _full)" | tee -a "$LOG_FILE"
echo "  Log file:      $LOG_FILE"              | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Check Python & dependencies
# ---------------------------------------------------------------------------
log_step "Checking Python installation"
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    exit 1
fi
log "Python 3 is available: $(python3 --version)"

log_step "Checking Python dependencies"

for pkg in relbench pandas; do
    if ! python3 -c "import $pkg" 2>/dev/null; then
        print_info "Installing $pkg..."
        if pip install $pkg >> "$LOG_FILE" 2>&1; then
            log "$pkg installed successfully"
        else
            log_error "Failed to install $pkg"
            exit 1
        fi
    else
        log "$pkg is already installed"
    fi
done
echo "Python dependencies OK" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Step 1: Verify PostgreSQL container
# ---------------------------------------------------------------------------
print_header "Step 1: Verify PostgreSQL Container"
log_step "Verifying PostgreSQL container"
print_info "Checking container: $CONTAINER_NAME"

if ! docker ps | grep -q "$CONTAINER_NAME"; then
    log_error "Container '$CONTAINER_NAME' is not running"
    print_info "Start it with: docker start $CONTAINER_NAME"
    exit 1
fi
log "Container $CONTAINER_NAME is running"

# ---------------------------------------------------------------------------
# Step 2: Prepare database
# ---------------------------------------------------------------------------
print_header "Step 2: Prepare Database"
log_step "Preparing database: $DB_NAME"

PGPASSWORD="$DB_PASSWORD" docker exec -e PGPASSWORD="$DB_PASSWORD" $CONTAINER_NAME \
    psql -U "$DB_USER" -d postgres -tc \
        "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" \
    | grep -q 1 && DB_EXISTS=true || DB_EXISTS=false

if [ "$DB_EXISTS" = "false" ]; then
    print_info "Database '$DB_NAME' does not exist, creating..."
    if PGPASSWORD="$DB_PASSWORD" docker exec -e PGPASSWORD="$DB_PASSWORD" $CONTAINER_NAME \
           psql -U "$DB_USER" -d postgres -c "CREATE DATABASE \"$DB_NAME\";" >> "$LOG_FILE" 2>&1; then
        log "Database '$DB_NAME' created"
    else
        log_error "Failed to create database '$DB_NAME'"
        exit 1
    fi
else
    log "Database '$DB_NAME' already exists – existing tables will be dropped and recreated"
fi

# ---------------------------------------------------------------------------
# Step 3: Download dataset and generate full CSV + SQL
# ---------------------------------------------------------------------------
print_header "Step 3: Download Dataset and Generate SQL"
log_step "Running generate_relbench_sql_full.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/generate_relbench_sql_full.py" ]; then
    log_error "generate_relbench_sql_full.py not found in $SCRIPT_DIR"
    exit 1
fi

print_info "Downloading $DATASET_NAME dataset and exporting full CSVs..."
echo "" | tee -a "$LOG_FILE"
if PYTHONUNBUFFERED=1 python3 "$SCRIPT_DIR/generate_relbench_sql_full.py" \
        "$DATASET_NAME" --output-dir "$DATA_DIR" 2>&1 | tee -a "$LOG_FILE"; then
    log "Python script executed successfully"
else
    log_error "Python script failed – check $LOG_FILE for details"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    log_error "Data directory not created – Python script may have failed"
    exit 1
fi

if [ ! -f "$SQL_FILE" ]; then
    log_error "SQL script not found: $SQL_FILE"
    exit 1
fi

CSV_COUNT=$(ls -1 "$DATA_DIR"/*.csv 2>/dev/null | wc -l)
print_info "Found $CSV_COUNT CSV files"
log "CSV count: $CSV_COUNT"

# ---------------------------------------------------------------------------
# Step 4: Copy files to container
# ---------------------------------------------------------------------------
print_header "Step 4: Copy Files to Container"
log_step "Copying CSV files to container"

for csv_file in "$DATA_DIR"/*.csv; do
    if [ -f "$csv_file" ]; then
        filename=$(basename "$csv_file")
        print_info "Copying $filename..."
        if docker cp "$csv_file" "$CONTAINER_NAME":/tmp/ 2>&1 | tee -a "$LOG_FILE"; then
            log "Copied $filename"
        else
            log_error "Failed to copy $filename"
            exit 1
        fi
    fi
done

print_info "Copying SQL script..."
if docker cp "$SQL_FILE" "$CONTAINER_NAME":/tmp/ >> "$LOG_FILE" 2>&1; then
    log "SQL script copied"
else
    log_error "Failed to copy SQL script"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 5: Execute SQL import inside the container
# ---------------------------------------------------------------------------
print_header "Step 5: Execute Full Import into PostgreSQL"
log_step "Running import SQL in database $DB_NAME"

print_info "Importing full dataset – this may take a while for large datasets..."
if PGPASSWORD="$DB_PASSWORD" docker exec -e PGPASSWORD="$DB_PASSWORD" $CONTAINER_NAME \
       psql -U "$DB_USER" -d "$DB_NAME" -f "/tmp/import_${DATASET_SHORT}_full.sql" \
       2>&1 | tee -a "$LOG_FILE"; then
    log "Full import completed successfully"
else
    log_error "Import failed – check $LOG_FILE for details"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 6: Verify row counts
# ---------------------------------------------------------------------------
print_header "Step 6: Verify Row Counts"
log_step "Querying row counts"

print_info "Row counts in database $DB_NAME:"
PGPASSWORD="$DB_PASSWORD" docker exec -e PGPASSWORD="$DB_PASSWORD" $CONTAINER_NAME \
    psql -U "$DB_USER" -d "$DB_NAME" -c "
SELECT
    relname AS table_name,
    to_char(n_live_tup, 'FM999,999,999') AS estimated_rows
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;

SELECT
    to_char(SUM(n_live_tup), 'FM999,999,999') AS total_rows,
    COUNT(*) AS total_tables
FROM pg_stat_user_tables;
" 2>&1 | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print_header "Import Complete"
echo "Dataset:   $DATASET_NAME"     | tee -a "$LOG_FILE"
echo "Database:  $DB_NAME"          | tee -a "$LOG_FILE"
echo "Tables:    no suffix (DB name carries _full)" | tee -a "$LOG_FILE"
echo "Log file:  $LOG_FILE"         | tee -a "$LOG_FILE"
echo ""                             | tee -a "$LOG_FILE"
echo "Connect:"                     | tee -a "$LOG_FILE"
echo "  docker exec -it $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
