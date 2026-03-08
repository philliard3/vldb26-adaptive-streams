#!/bin/bash
# Exp3 Revised Wave Runner (Feb25_1238pm)

set -e

BINARY="./animal_query_bin"
LOG_CONFIG="./log_config.yml"
TIMESTAMP=$(date +%b%d_%I%M%S%p)
ALL_LOGS_DIR="all_logs_exp3_revised_wave_feb25_1238pm_$TIMESTAMP"
mkdir -p "$ALL_LOGS_DIR"

export LD_LIBRARY_PATH=$(pwd)/deps:$LD_LIBRARY_PATH

run_config_group() {
  local grp=$1
  local pattern=$2

  echo "Running Group: $grp..."

  shopt -s nullglob
  local matched=0

  for cfg_file in $pattern; do
      [ -e "$cfg_file" ] || continue
      matched=1
      echo "  Running $cfg_file"

      pgrep -f animal_query_bin | xargs kill -9 2>/dev/null || true
      sleep 1

      cp "$cfg_file" tmp_config.json
      mkdir -p log_outputs

      bname=$(basename "$cfg_file" .json)
      run_log="target_debugging_outputs/output_${bname}.log"
      mkdir -p target_debugging_outputs

      start_time=$(date +%s)
      set +e
      RUST_BACKTRACE=1 AQUIFER_BUDGET_CALCULATION_VERSION=2 $BINARY tmp_config.json $LOG_CONFIG > "$run_log" 2>&1
      exit_code=$?
      set -e

      if grep -Eq "panicked at|Main thread panicked|Error creating ONNX session" "$run_log"; then
          echo "    Detected panic/ONNX initialization failure in $run_log"
          exit_code=99
      fi

      items_read=$(grep -E "items read:" "$run_log" | tail -n 1 | sed -E 's/.*items read: ([0-9]+).*/\1/' || true)
      end_time=$(date +%s)
      duration=$((end_time - start_time))

      echo "    Exit Code: $exit_code | Time: ${duration}s"
      if [ -n "$items_read" ]; then
          echo "    Items Read: $items_read"
      fi

      dest="$ALL_LOGS_DIR/$grp/log/compact_log_$bname"
      mkdir -p "$dest"
      if [ -d log_outputs ] && [ "$(ls -A log_outputs)" ]; then
         mv log_outputs/* "$dest/"
      fi
      rm -rf log_outputs

      if [ -f "$run_log" ]; then
          mv "$run_log" "$dest/output_$bname.log"
      fi

      if [ -f log/output.log ]; then
          mv log/output.log "$dest/legacy_output_$bname.log"
      fi

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

MODELS=("nano_xlarge" "small_large" "nano_small_medium" "all")

# Adaptive configs
for subset in "${MODELS[@]}"; do
    pattern="./deployment_exp3_wave_feb25_1238pm/configs/${subset}/adaptive_*/*.json"
    run_config_group "${subset}_adaptive" "$pattern"
done

# Baselines
for subset in "${MODELS[@]}"; do
    pattern="./deployment_exp3_wave_feb25_1238pm/configs/${subset}/baselines/*.json"
    run_config_group "${subset}_baselines" "$pattern"
done


echo "Completed. Logs in: $ALL_LOGS_DIR"
