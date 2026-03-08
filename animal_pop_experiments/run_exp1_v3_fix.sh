#!/bin/bash
# Experiment 1 Runner (V3 Fix Only)
set -e

BINARY="./animal_query_bin"
LOG_CONFIG="./log_config.yml"
CONFIG_BASE="deployment_exp1/configs"
ALL_LOGS_DIR="all_logs_exp1"
mkdir -p "$ALL_LOGS_DIR"
mkdir -p log log_outputs

run_group() {
  local dir=$1
  local group=$2
  echo ""
  echo "========================================"
  echo "Running Group: $group"
  echo "========================================"
  
  if [ ! -d "$dir" ]; then
    echo "Skipping $group (Directory not found: $dir)"
    return
  fi
  
  for f in $(ls $dir/*.json); do
    echo "  Config: $(basename $f)"
    
    pgrep -f animal_query_bin | xargs kill -9 2>/dev/null || true
    sleep 1
    
    cp "$f" tmp_config.json
    # Run
    set +e
    time RUST_BACKTRACE=1 AQUIFER_BUDGET_CALCULATION_VERSION=2 $BINARY tmp_config.json $LOG_CONFIG > /dev/null
    ret=$?
    set -e
    echo "    Exit code: $ret"
    # Move Logs
    cfg_name=$(basename $f .json)
    mkdir -p "$ALL_LOGS_DIR/$group"
    rm -rf "$ALL_LOGS_DIR/$group/$cfg_name"
    if [ -d log_outputs ]; then mv log_outputs "$ALL_LOGS_DIR/$group/$cfg_name"; fi
    mkdir -p log_outputs
    if [ -f log/output.log ]; then mv log/output.log "$ALL_LOGS_DIR/$group/$cfg_name.log"; fi
  done
}

# ONLY RUN V3
run_group "$CONFIG_BASE/v3_all_acc_ge_next" "v3_all_acc_ge_next"
