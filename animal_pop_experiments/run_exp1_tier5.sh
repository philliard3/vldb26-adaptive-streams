#!/bin/bash
# Exp1 Tier 5 (Showcase) Runner
# Static script (not generated)

set -e

BINARY="./animal_query_bin"
LOG_CONFIG="./log_config.yml"
TIMESTAMP=$(date +%b%d_%I%M%p)
ALL_LOGS_DIR="all_logs_exp1_tier5_$TIMESTAMP"
mkdir -p "$ALL_LOGS_DIR"

# Ensure we can find the shared libraries
export LD_LIBRARY_PATH=$(pwd)/deps:$LD_LIBRARY_PATH

run_config_group() {
  local grp=$1
  local pattern=$2
  
  echo "Running Group: $grp..."
  
  for cfg_file in $pattern; do
      [ -e "$cfg_file" ] || continue
      echo "  Running $cfg_file"
      
      pgrep -f animal_query_bin | xargs kill -9 2>/dev/null || true
      sleep 1
      
      cp "$cfg_file" tmp_config.json
      mkdir -p log_outputs
      
      # Determine output log name based on config file
      bname=$(basename "$cfg_file" .json)
      run_log="target_debugging_outputs/output_${bname}.log"
      mkdir -p target_debugging_outputs

      # Run Binary with timing
      # Note: REROUTE_ZERO_OPTION is now handled via config file, not env var.
      start_time=$(date +%s)
      RUST_BACKTRACE=1 AQUIFER_BUDGET_CALCULATION_VERSION=2 $BINARY tmp_config.json $LOG_CONFIG > "$run_log" 2>&1
      exit_code=$?
      end_time=$(date +%s)
      duration=$((end_time - start_time))
      
      echo "    Exit Code: $exit_code | Time: ${duration}s"
      
      # Move Logs
      dest_grp="$ALL_LOGS_DIR/$grp"
      mkdir -p "$dest_grp"
      
      # Move compact logs (log_outputs directory content)
      # Legacy Exp1 convention: folder name is just the config name
      if [ -d log_outputs ] && [ "$(ls -A log_outputs)" ]; then
          mv log_outputs "$dest_grp/$bname"
      fi
      rm -rf log_outputs
      
      # Move the main output log
      if [ -f "$run_log" ]; then
          mv "$run_log" "$dest_grp/${bname}.log"
      fi
      
      # Move log/output.log if it exists (legacy behavior, but good to capture)
      if [ -f log/output.log ]; then
           mv log/output.log "$dest_grp/legacy_output_$bname.log"
      fi
  done
}

# Configs are in ./deployment_exp1_tier5/configs/
# Naming pattern: showcase_{subset}_{reroute}_{rate}ips_w{window}.json

# We can group by subset and reroute option for organization
run_config_group "all_first" "./deployment_exp1_tier5/configs/showcase_all_first_*.json"
run_config_group "all_ignored" "./deployment_exp1_tier5/configs/showcase_all_ignored_*.json"
run_config_group "nano_small_medium_first" "./deployment_exp1_tier5/configs/showcase_nano_small_medium_first_*.json"
run_config_group "nano_small_medium_ignored" "./deployment_exp1_tier5/configs/showcase_nano_small_medium_ignored_*.json"
run_config_group "nano_xlarge_first" "./deployment_exp1_tier5/configs/showcase_nano_xlarge_first_*.json"
run_config_group "nano_xlarge_ignored" "./deployment_exp1_tier5/configs/showcase_nano_xlarge_ignored_*.json"
run_config_group "small_large_first" "./deployment_exp1_tier5/configs/showcase_small_large_first_*.json"
run_config_group "small_large_ignored" "./deployment_exp1_tier5/configs/showcase_small_large_ignored_*.json"
