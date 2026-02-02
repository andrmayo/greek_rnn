#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/four_strats.log"

echo "=== Running master script ===" | tee -a "$LOG_FILE"
echo "Date: $(date +%Y%m%d_%H%M%S)" >>"$LOG_FILE"

SCRIPTS=(
  "grk_rnn_train_1.sh"
  "grk_rnn_train_2.sh"
  "grk_rnn_train_3.sh"
  "grk_rnn_train_4.sh"
)

for i in "${!SCRIPTS[@]}"; do
  script="${SCRIPTS[$i]}"
  log_file="$SCRIPT_DIR/grk_rnn_train_$((i + 1)).log"
  err_file="$SCRIPT_DIR/grk_rnn_train_$((i + 1))_err.log"
  echo "=== Stdout for $script ===" >"$log_file"
  echo "=== Stderr for $script ===" >"$err_file"
  echo "Date: $(date +%Y%m%d_%H%M%S)" >>"$log_file"
  echo "Date: $(date +%Y%m%d_%H%M%S)" >>"$err_file"
  echo "=== Running $script, logging script output to $log_file ===" | tee -a "$LOG_FILE"
  bash "$SCRIPT_DIR/$script" >"$log_file" 2>>"$err_file"
  status=$?
  if [ $status -ne 0 ]; then
    echo "$script failed with exit code $status. Check $log_file and $err_file" | tee -a "$LOG_FILE"
  else
    echo "$script completed successfully. Stdout saved to $log_file, stderr to $err_file" | tee -a "$LOG_FILE"
  fi
done
