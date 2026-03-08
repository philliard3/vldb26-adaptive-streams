#!/bin/bash
# Exp2 Revised Runner (Feb 19)
# Static script (not generated)

set -e

BINARY="./animal_query_bin"
LOG_CONFIG="./log_config.yml"
TIMESTAMP=$(date +%b%d_%I%M%p)
ALL_LOGS_DIR="all_logs_exp2_revised_$TIMESTAMP"
mkdir -p "$ALL_LOGS_DIR"

# Ensure we can find the shared libraries
export LD_LIBRARY_PATH=$(pwd)/deps:$LD_LIBRARY_PATH

run_config_group() {
  local grp=$1
  local pattern=$2
  
  echo "Running Group: $grp..."

    shopt -s nullglob
    local matched=0
  
  # Bash glob expansion happens here
  # If no files match, the loop might run once with the pattern as the value if nullglob is off (default).
  # We check [ -e ... ] just in case.
  for cfg_file in $pattern; do
      [ -e "$cfg_file" ] || continue
            matched=1
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
    set +e
      RUST_BACKTRACE=1 AQUIFER_BUDGET_CALCULATION_VERSION=2 $BINARY tmp_config.json $LOG_CONFIG > "$run_log" 2>&1
      exit_code=$?
    set -e

      # Detect runtime panics/errors that may not propagate as non-zero process status.
      if grep -Eq "panicked at|Main thread panicked|Error creating ONNX session" "$run_log"; then
          echo "    Detected panic/ONNX initialization failure in $run_log"
          exit_code=99
      fi

      # Optional visibility: report how many items were read if the binary logged it.
      items_read=$(grep -E "items read:" "$run_log" | tail -n 1 | sed -E 's/.*items read: ([0-9]+).*/\1/' || true)
      end_time=$(date +%s)
      duration=$((end_time - start_time))
      
      echo "    Exit Code: $exit_code | Time: ${duration}s"
      if [ -n "$items_read" ]; then
          echo "    Items Read: $items_read"
      fi

      if [ "$exit_code" -ne 0 ]; then
          echo "    ERROR: run failed for $cfg_file (see $run_log)"
      fi
      
      # Move Logs
      dest="$ALL_LOGS_DIR/$grp/log/compact_log_$bname"
      mkdir -p "$dest"
      if [ -d log_outputs ] && [ "$(ls -A log_outputs)" ]; then
         mv log_outputs/* "$dest/"
      fi
      rm -rf log_outputs

      # Move the main output log
      if [ -f "$run_log" ]; then
          mv "$run_log" "$dest/output_$bname.log"
      fi

      # Move log/output.log if it exists
      if [ -f log/output.log ]; then
          mv log/output.log "$dest/legacy_output_$bname.log"
      fi

      # Stop early on hard failures to avoid silently producing partial/invalid runs.
      if [ "$exit_code" -ne 0 ]; then
          echo "Stopping runner due to failure in $cfg_file"
          exit "$exit_code"
      fi
  done
  shopt -u nullglob

  if [ "$matched" -eq 0 ]; then
      echo "  No configs matched for group (skipped)."
  fi
}

# Configs are in ./deployment_exp2/configs/
# Structure: configs/{subset}/{adaptive|baseline}/{...}

# We simply define groups that cover all likely generated configs.
# Even if some aren't present in a specific deployment (e.g. only subset=nano_xlarge),
# the glob will just match nothing and the function handles it gracefully.

MODELS=("nano_xlarge" "small_large" "nano_small_medium" "all")
STRATEGIES=("greedy" "optimal")
REROUTES=("ignored" "first")
BASELINES=("eddies" "always_nano" "always_small" "always_medium" "always_large" "always_xlarge")

# 1. Adaptive Strategies
for subset in "${MODELS[@]}"; do
    for strat in "${STRATEGIES[@]}"; do
        for reroute in "${REROUTES[@]}"; do
             # Group: {subset}_{strat}_{reroute}
             grp="${subset}_${strat}_${reroute}"
             # Pattern: configs/{subset}/adaptive_{strat}/{strat}_*_{reroute}_*.json
             pattern="./deployment_exp2/configs/${subset}/adaptive_${strat}/${strat}_*_${reroute}_*.json"
             run_config_group "$grp" "$pattern"
        done
    done
done

# 2. Baselines
for subset in "${MODELS[@]}"; do
    for base in "${BASELINES[@]}"; do
        # Group: {subset}_baseline_{base}
        grp="${subset}_baseline_${base}"
        # Path: configs/{subset}/baselines/{base}_*.json
        pattern="./deployment_exp2/configs/${subset}/baselines/${base}_*.json"
        run_config_group "$grp" "$pattern"
    done
done
