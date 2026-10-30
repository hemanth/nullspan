#!/usr/bin/env bash
# Ephemeral Google Cloud GPU Public Benchmark Runner (JevBench / Benchmark Heaven + HF Typed Decisions + SQuAD v2.0)
set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"

PROJECT_ID="${GCP_PROJECT:-hemanth-hm}"
INSTANCE_NAME="nullspan-gpu-bench-$(date +%s)"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
BUNDLE_TAR="${TMP_DIR}/nullspan_bench_bundle.tar.gz"
RESULTS_TAR="${TMP_DIR}/nullspan_bench_results.tar.gz"
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
  for attempt in $(seq 1 35); do
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

echo "[1/5] Packaging System One + Public Benchmark harness..."
COPYFILE_DISABLE=1 tar -czf "${BUNDLE_TAR}" -C "${ROOT_DIR}" \
  training/modeling_nullspan.py \
  training/systemone_decision.py \
  training/serve_systemone.py \
  bench/eval_public_benchmarks.py \
  data/train.jsonl \
  data/cal_dev.jsonl \
  data/val.jsonl
ls -lh "${BUNDLE_TAR}"

echo "[2/5] Provisioning ephemeral GPU VM on Google Cloud (${PROJECT_ID})..."
for z in us-central1-a us-central1-b us-central1-c us-east4-a us-west1-a; do
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

if [[ -z "${ACTIVE_ZONE}" ]]; then
  for z in us-central1-a us-central1-b us-central1-c us-central1-f us-west1-a us-west1-b; do
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
        --quiet 2>/dev/null; then
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
        git python3-venv python3-pip \
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

echo "[3/5] Uploading benchmark bundle to ${INSTANCE_NAME}..."
gcloud compute scp "${BUNDLE_TAR}" "${INSTANCE_NAME}:~/nullspan_bench_bundle.tar.gz" \
  --project="${PROJECT_ID}" \
  --zone="${ACTIVE_ZONE}" \
  --quiet

echo "[4/5] Running Typed Decisions + Benchmark Heaven (JevBench) + SQuAD v2.0 on GCP GPU (${GPU_MODE})..."
if [[ "${GPU_MODE}" == "l4_dlvm" ]]; then
  gcloud compute ssh "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ACTIVE_ZONE}" \
    --quiet \
    --command='
      set -euo pipefail
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
      rm -rf ~/jevbench && git clone --depth 1 https://github.com/fstandhartinger/jevbench.git ~/jevbench
      mkdir -p ~/work && cd ~/work
      tar -xzf ~/nullspan_bench_bundle.tar.gz
      PY_BIN="/opt/conda/bin/python"
      [[ -x "${PY_BIN}" ]] || PY_BIN="python3"
      "${PY_BIN}" -m pip install -q --no-warn-script-location "transformers>=4.44.0,<5.0.0" safetensors datasets huggingface_hub
      "${PY_BIN}" -c "from huggingface_hub import snapshot_download; snapshot_download(\"gnumanth/nullspan\", local_dir=\"artifacts/nullspan\", ignore_patterns=[\"onnx/*\"])"
      cp training/modeling_nullspan.py artifacts/nullspan/modeling_nullspan.py
      JEVBENCH_DIR="$HOME/jevbench" "${PY_BIN}" -u bench/eval_public_benchmarks.py
      tar -czf ~/nullspan_bench_results.tar.gz artifacts/nullspan/systemone_decision.pt bench/public_benchmarks_results.json
    '
else
  gcloud compute ssh "${INSTANCE_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ACTIVE_ZONE}" \
    --quiet \
    --command='
      set -euo pipefail
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
      rm -rf ~/jevbench && git clone --depth 1 https://github.com/fstandhartinger/jevbench.git ~/jevbench
      python3 -m venv ~/venv
      ~/venv/bin/pip install -q --upgrade pip
      ~/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126
      ~/venv/bin/pip install -q "transformers>=4.44.0,<5.0.0" safetensors datasets huggingface_hub
      mkdir -p ~/work && cd ~/work
      tar -xzf ~/nullspan_bench_bundle.tar.gz
      ~/venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(\"gnumanth/nullspan\", local_dir=\"artifacts/nullspan\", ignore_patterns=[\"onnx/*\"])"
      cp training/modeling_nullspan.py artifacts/nullspan/modeling_nullspan.py
      JEVBENCH_DIR="$HOME/jevbench" ~/venv/bin/python -u bench/eval_public_benchmarks.py
      tar -czf ~/nullspan_bench_results.tar.gz artifacts/nullspan/systemone_decision.pt bench/public_benchmarks_results.json
    '
fi

echo "[5/5] Downloading System One checkpoint + public benchmark results back to ${ROOT_DIR}..."
gcloud compute scp "${INSTANCE_NAME}:~/nullspan_bench_results.tar.gz" "${RESULTS_TAR}" \
  --project="${PROJECT_ID}" \
  --zone="${ACTIVE_ZONE}" \
  --quiet

tar -xzf "${RESULTS_TAR}" -C "${ROOT_DIR}"
echo "[GCP] Public benchmarks complete! Results written to ${ROOT_DIR}/bench/public_benchmarks_results.json"
