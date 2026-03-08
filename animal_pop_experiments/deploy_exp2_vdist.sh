#!/bin/bash
# Deploy Exp2 VDist (Feb 20)
# Usage mirrors deploy_exp2_revised.sh with optional dataset sync.

set -e

DEPLOYMENT_BASE="deployment_exp2_vdist"

SYNC_DATASET=0
DATASET_METADATA_LOCAL=""
DATASET_METADATA_REMOTE=""
DATASET_IMAGES_LOCAL=""
DATASET_IMAGES_REMOTE_DIR=""
CONFIG_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --sync-dataset)
            SYNC_DATASET=1
            shift
            ;;
        --dataset-metadata-local)
            DATASET_METADATA_LOCAL="$2"
            shift 2
            ;;
        --dataset-metadata-remote)
            DATASET_METADATA_REMOTE="$2"
            shift 2
            ;;
        --dataset-images-local)
            DATASET_IMAGES_LOCAL="$2"
            shift 2
            ;;
        --dataset-images-remote-dir)
            DATASET_IMAGES_REMOTE_DIR="$2"
            shift 2
            ;;
        *)
            CONFIG_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Generating Exp2 VDist Configs with args: ${CONFIG_ARGS[*]}"
rm -rf "$DEPLOYMENT_BASE/configs" "$DEPLOYMENT_BASE/queries"
rm -f exp2_vdist_transfers.txt
python3 make_exp2_vdist_configs.py "${CONFIG_ARGS[@]}"

RUNTIME_DIR="~/animal_query_artifacts"

if [ -f "current_vm.txt" ]; then
    INSTANCE_DNS=$(cat current_vm.txt | tr -d '[:space:]')
else
    echo "ERROR: current_vm.txt not found."
    exit 1
fi

KEY="adaptive_streams__time_series_pair.pem"
USER="ubuntu"
SERVER="$USER@$INSTANCE_DNS"

echo "Deploying to $SERVER (Runtime: $RUNTIME_DIR)..."

echo "Stopping potentially conflicting remote processes..."
ssh -i "$KEY" "$SERVER" 'pkill -x animal_query_bin 2>/dev/null || true; pkill -f "^/bin/bash ./run_exp2_vdist.sh$" 2>/dev/null || true'

echo "Syncing code and building..."
rsync -avz -e "ssh -i $KEY" \
    --include='Cargo.toml' \
    --include='Cargo.lock' \
    --include='watershed_shared/***' \
    --include='animal_query/***' \
    --exclude='*' \
    ./ "$SERVER:~/adaptive-ml-experiments/"

echo "Building and installing artifacts..."
ssh -i "$KEY" "$SERVER" << EOF
    set -e
    source ~/.cargo/env
    cd adaptive-ml-experiments
    RUSTFLAGS='-C target-cpu=native' cargo build --release --bin animal_query

    mkdir -p $RUNTIME_DIR/deps
    rm -f $RUNTIME_DIR/animal_query_bin
    rm -f $RUNTIME_DIR/deps/*.so $RUNTIME_DIR/deps/*.so.* 2>/dev/null || true
    cp target/release/animal_query $RUNTIME_DIR/animal_query_bin
    cp target/release/*.so* $RUNTIME_DIR/deps/ 2>/dev/null || true
    cp target/release/deps/*.so* $RUNTIME_DIR/deps/ 2>/dev/null || true
EOF

echo "Uploading configs & queries..."
ssh -i "$KEY" "$SERVER" "mkdir -p $RUNTIME_DIR/deployment_exp2_vdist/configs $RUNTIME_DIR/deployment_exp2_vdist/queries"
rsync -avz --delete -e "ssh -i $KEY" ./deployment_exp2_vdist/ "$SERVER:$RUNTIME_DIR/deployment_exp2_vdist/"

echo "Uploading pre-classifiers..."
ssh -i "$KEY" "$SERVER" "mkdir -p $RUNTIME_DIR/exp2_preclassifiers"

if [ -f "exp2_vdist_transfers.txt" ]; then
    count=0
    while IFS="|" read -r local server; do
        clean_server=${server#./}
        count=$((count + 1))
        if (( count % 5 == 0 )); then
            echo "  Uploading $local -> $clean_server (and others...)"
        fi
        rsync -avz -e "ssh -i $KEY" "$local" "$SERVER:$RUNTIME_DIR/$clean_server" > /dev/null
    done < exp2_vdist_transfers.txt
    echo "  Uploaded $count pre-classifiers."
else
    echo "WARNING: exp2_vdist_transfers.txt not found. No pre-classifiers uploaded."
fi

echo "Uploading runner..."
rsync -avz -e "ssh -i $KEY" run_exp2_vdist.sh "$SERVER:$RUNTIME_DIR/"
rsync -avz -e "ssh -i $KEY" animal_query/log_config.yml "$SERVER:$RUNTIME_DIR/"
ssh -i "$KEY" "$SERVER" "chmod +x $RUNTIME_DIR/run_exp2_vdist.sh"

if [ "$SYNC_DATASET" -eq 1 ]; then
    echo "Syncing dataset artifacts..."

    if [ -n "$DATASET_METADATA_LOCAL" ]; then
        if [ ! -f "$DATASET_METADATA_LOCAL" ]; then
            echo "ERROR: Dataset metadata file not found: $DATASET_METADATA_LOCAL"
            exit 1
        fi
        if [ -z "$DATASET_METADATA_REMOTE" ]; then
            DATASET_METADATA_REMOTE="$(basename "$DATASET_METADATA_LOCAL")"
        fi
        remote_meta_dir=$(dirname "$DATASET_METADATA_REMOTE")
        ssh -i "$KEY" "$SERVER" "mkdir -p $RUNTIME_DIR/$remote_meta_dir"
        rsync -avz -e "ssh -i $KEY" "$DATASET_METADATA_LOCAL" "$SERVER:$RUNTIME_DIR/$DATASET_METADATA_REMOTE"
        echo "  Metadata synced: $DATASET_METADATA_LOCAL -> $RUNTIME_DIR/$DATASET_METADATA_REMOTE"
    fi

    if [ -n "$DATASET_IMAGES_LOCAL" ]; then
        if [ ! -d "$DATASET_IMAGES_LOCAL" ]; then
            echo "ERROR: Dataset images directory not found: $DATASET_IMAGES_LOCAL"
            exit 1
        fi
        if [ -z "$DATASET_IMAGES_REMOTE_DIR" ]; then
            DATASET_IMAGES_REMOTE_DIR="animal_images"
        fi
        ssh -i "$KEY" "$SERVER" "mkdir -p $RUNTIME_DIR/$DATASET_IMAGES_REMOTE_DIR"
        rsync -avz --delete -e "ssh -i $KEY" "$DATASET_IMAGES_LOCAL/" "$SERVER:$RUNTIME_DIR/$DATASET_IMAGES_REMOTE_DIR/"
        echo "  Images synced: $DATASET_IMAGES_LOCAL -> $RUNTIME_DIR/$DATASET_IMAGES_REMOTE_DIR"
    fi
fi

if [ -n "$DATASET_METADATA_REMOTE" ]; then
    echo "Running dataset preflight checks..."
    ssh -i "$KEY" "$SERVER" "python3 - << 'PY'
import json
import os
import sys

runtime_dir = os.path.expanduser('$RUNTIME_DIR')
meta_rel = '$DATASET_METADATA_REMOTE'
meta_path = os.path.join(runtime_dir, meta_rel)

if not os.path.isfile(meta_path):
    print(f'ERROR: metadata missing: {meta_path}')
    sys.exit(1)

with open(meta_path, 'r') as f:
    md = json.load(f)

items = md.get('image_data', [])
if not items:
    print(f'ERROR: metadata has no image_data entries: {meta_path}')
    sys.exit(1)

first_rel = items[0].get('relative_path', '')
if not first_rel:
    print('ERROR: first metadata entry missing relative_path')
    sys.exit(1)

first_img = os.path.join(runtime_dir, first_rel)
if not os.path.isfile(first_img):
    print(f'ERROR: first referenced image missing: {first_img}')
    sys.exit(1)

print(f'Preflight OK: {meta_rel} | first image: {first_rel}')
PY"
fi

echo "=========================================="
echo "Deployment Complete!"
echo "To run the experiment:"
echo "  ssh -i $KEY $SERVER"
echo "  cd $RUNTIME_DIR"
echo "  nohup ./run_exp2_vdist.sh > exp2_vdist.log 2>&1 &"
echo "=========================================="
