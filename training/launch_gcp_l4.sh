#!/usr/bin/env bash
# Ephemeral Google Cloud GPU Training Launcher (NVIDIA L4 / Tesla V100) for NullspanForExtraction v2.0
set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"

PROJECT_ID="${GCP_PROJECT:-hemanth-hm}"
INSTANCE_NAME="nullspan-gpu-trainer-$(date +%s)"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
BUNDLE_TAR="${TMP_DIR}/nullspan_train_bundle.tar.gz"
RESULTS_TAR="${TMP_DIR}/nullspan_artifacts.tar.gz"
ACTIVE_ZONE=""
GPU_MODE=""

cleanup() {
  echo "[GCP] Cleaning up temporary files and deleting ephemeral VM (${INSTANCE_NAME})..."
  rm -rf "${TMP_DIR}" || true
  if [[ -n "${ACTIVE_ZONE}" ]]; then
    gcloud compute instances delete "${INSTANCE_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${ACTIVE_ZONE}" \
      --quiet >/dev/null 2>&1 || true
    echo "[GCP] Ephemeral VM ${INSTANCE_NAME} (${ACTIVE_ZONE}) deleted."
  fi
}
trap cleanup EXIT

wait_for_ssh() {
  for attempt in $(seq 1 30); do
    if gcloud compute ssh "${INSTANCE_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ACTIVE_ZONE}" \
        --command="echo SSH_READY" \
        --quiet 2>/dev/null; then
      return 0
    fi
    sleep 4
  done
  return 1
}

echo "[1/5] Packaging zero-leakage dataset + training code..."
COPYFILE_DISABLE=1 tar -czf "${BUNDLE_TAR}" -C "${ROOT_DIR}" \
  training/modeling_nullspan.py \
  training/generate_dataset.py \
  training/train_gcp.py \
  training/export_onnx.py \
  data/train.jsonl \
  data/cal_dev.jsonl \
  data/val.jsonl \
  data/golden_eval.json \
  bench/squad_v2_golden.json \
  bench/contracts_sealed_golden.json
ls -lh "${BUNDLE_TAR}"

echo "[2/5] Provisioning ephemeral GPU VM on Google Cloud (${PROJECT_ID})..."
# Priority 1: NVIDIA L4 in us-central1-a (Ada, pre-baked PyTorch 2.9 + Driver 580)
for z in us-central1-a; do
  echo "  Trying NVIDIA L4 (g2-standard-8) in ${z}..."
  if gcloud compute instances create "${INSTANCE_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${z}" \
      --machine-type="g2-standard-8" \
      --maintenance-policy=TERMINATE \
      --image-project="deeplearning-platform-release" \
      --image-family="pytorch-2-9-cu129-ubuntu-2204-nvidia-580" \
      --boot-disk-size=100GB \
      --boot-disk-type=pd-ssd \
      --quiet 2>/dev/null; then
    ACTIVE_ZONE="${z}"
    GPU_MODE="l4_dlvm"
    break
  fi
done

# Priority 2: NVIDIA Tesla V100 16GB HBM2 (Ubuntu 22.04 LTS + Proprietary NVIDIA 580-server Driver + CUDA 12.6)
if [[ -z "${ACTIVE_ZONE}" ]]; then
  for z in us-central1-a us-central1-b us-central1-c us-central1-f; do
    echo "  Trying NVIDIA Tesla V100 (n1-standard-8) in ${z}..."
    if gcloud compute instances create "${INSTANCE_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${z}" \
        --machine-type="n1-standard-8" \
        --accelerator="type=nvidia-tesla-v100,count=1" \
        --maintenance-policy=TERMINATE \
        --image-project="ubuntu-os-cloud" \
        --image-family="ubuntu-2204-lts" \
        --boot-disk-size=80GB \
        --boot-disk-type=pd-ssd \
        --quiet; then
      ACTIVE_ZONE="${z}"
      GPU_MODE="v100_ubuntu"
      break
    fi
  done
fi

if [[ -z "${ACTIVE_ZONE}" ]]; then
  echo "ERROR: Could not provision GPU VM on Google Cloud." >&2
  exit 1
fi

echo "[GCP] Provisioned ${INSTANCE_NAME} (${GPU_MODE}) in ${ACTIVE_ZONE}. Waiting for SSH..."
wait_for_ssh

if [[ "${GPU_MODE}" == "v100_ubuntu" ]]; then
  echo "[GCP] Installing proprietary NVIDIA 580-server driver on V100 VM..."
  gcloud compute ssh "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ACTIVE_ZONE}" \
    --quiet \
    --command='
      set -euo pipefail
      sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        python3-venv python3-pip \
        linux-modules-nvidia-580-server-gcp \
        nvidia-headless-580-server \
        nvidia-utils-580-server
      echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
    '
  echo "[GCP] Rebooting V100 VM to bind proprietary nvidia.ko cleanly..."
  gcloud compute ssh "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ACTIVE_ZONE}" \
    --quiet \
    --command="sudo reboot" 2>/dev/null || true
  sleep 10
  wait_for_ssh
fi

echo "[3/5] Uploading bundle to ${INSTANCE_NAME}..."
gcloud compute scp "${BUNDLE_TAR}" "${INSTANCE_NAME}:~/nullspan_train_bundle.tar.gz" \
  --project="${PROJECT_ID}" \
  --zone="${ACTIVE_ZONE}" \
  --quiet

echo "[4/5] Running NullspanForExtraction v2.0 GPU training on Google Cloud (${GPU_MODE})..."
if [[ "${GPU_MODE}" == "l4_dlvm" ]]; then
  gcloud compute ssh "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ACTIVE_ZONE}" \
    --quiet \
    --command='
      set -euo pipefail
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
      mkdir -p ~/work && cd ~/work
      tar -xzf ~/nullspan_train_bundle.tar.gz
      PY_BIN="/opt/conda/bin/python"
      [[ -x "${PY_BIN}" ]] || PY_BIN="python3"
      "${PY_BIN}" -m pip install -q --no-warn-script-location "transformers>=4.44.0,<5.0.0" safetensors
      "${PY_BIN}" -c "import torch; assert torch.cuda.is_available(); print(\"GPU Verified:\", torch.cuda.get_device_name(0))"
      "${PY_BIN}" training/train_gcp.py --skip-generate --epochs 8 --batch-size 32 --lr 3.5e-4
      tar -czf ~/nullspan_artifacts.tar.gz artifacts/nullspan
    '
else
  gcloud compute ssh "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ACTIVE_ZONE}" \
    --quiet \
    --command='
      set -euo pipefail
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
      python3 -m venv ~/venv
      ~/venv/bin/pip install -q --upgrade pip
      ~/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126
      ~/venv/bin/pip install -q "transformers>=4.44.0,<5.0.0" safetensors
      ~/venv/bin/python -c "import torch; assert torch.cuda.is_available(); x = torch.ones(4, device=\"cuda\") + 1; print(\"GPU + Kernel Verified:\", torch.cuda.get_device_name(0), x.sum().item())"
      mkdir -p ~/work && cd ~/work
      tar -xzf ~/nullspan_train_bundle.tar.gz
      ~/venv/bin/python training/train_gcp.py --skip-generate --epochs 8 --batch-size 32 --lr 3.5e-4
      tar -czf ~/nullspan_artifacts.tar.gz artifacts/nullspan
    '
fi

echo "[5/5] Downloading trained checkpoint back to ${ROOT_DIR}/artifacts/nullspan..."
gcloud compute scp "${INSTANCE_NAME}:~/nullspan_artifacts.tar.gz" "${RESULTS_TAR}" \
  --project="${PROJECT_ID}" \
  --zone="${ACTIVE_ZONE}" \
  --quiet

mkdir -p "${ROOT_DIR}/artifacts/nullspan"
tar -xzf "${RESULTS_TAR}" -C "${ROOT_DIR}"
echo "[GCP] Training complete! Local artifacts updated at ${ROOT_DIR}/artifacts/nullspan"
