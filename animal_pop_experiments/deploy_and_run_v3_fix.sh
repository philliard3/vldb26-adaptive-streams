#!/bin/bash
# Deploy + launch Experiment 1 (V3 Fix Only)
# Usage: ./deploy_and_run_v3_fix.sh

set -euo pipefail

# info in current_vm.txt
# SERVER_IP="${INSTANCE_DNS:-}"
INSTANCE_DNS=$(cat current_vm.txt)
SERVER_IP="${INSTANCE_DNS:-}"
USER="ubuntu"
KEY="adaptive_streams__time_series_pair.pem"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=60"

RUNTIME_DIR="~/animal_query_artifacts"

# Local Paths
LOCAL_PRECLASSIFIERS="animal_preclassifiers_feb17_exp1"
LOCAL_RUNNER="run_exp1_v3_fix.sh"

if [ -z "$SERVER_IP" ]; then
  echo "ERROR: INSTANCE_DNS is not set"; exit 1
fi
chmod 400 "$KEY"

echo "Deploying Experiment 1 V3 Fix to $USER@$SERVER_IP"

# 1. Sync Pre-classifiers (V3 Only)
ssh $SSH_OPTS "$USER@$SERVER_IP" "mkdir -p $RUNTIME_DIR/exp1_preclassifiers_v3"
rsync -avz -e "ssh $SSH_OPTS" "$LOCAL_PRECLASSIFIERS/" "$USER@$SERVER_IP:$RUNTIME_DIR/exp1_preclassifiers_v3/"

# 2. Runner
scp $SSH_OPTS "$LOCAL_RUNNER" "$USER@$SERVER_IP:$RUNTIME_DIR/"
ssh $SSH_OPTS "$USER@$SERVER_IP" "chmod +x $RUNTIME_DIR/$LOCAL_RUNNER"

# 3. Launch (Attach mode for quick feedback)
echo "Launching V3 fix runner..."
ssh $SSH_OPTS "$USER@$SERVER_IP" "cd $RUNTIME_DIR && env LD_LIBRARY_PATH=\$PWD/deps:\$LD_LIBRARY_PATH ./$LOCAL_RUNNER"

echo "V3 Fix Complete."
