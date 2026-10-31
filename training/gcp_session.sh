#!/usr/bin/env bash
# Fast Reusable Google Cloud GPU Session Manager for Nullspan /goal Optimization
set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE="${ROOT_DIR}/.gcp_session.env"
PROJECT_ID="${GCP_PROJECT:-hemanth-hm}"

cmd="${1:-status}"
shift || true

wait_for_ssh() {
  local inst="$1"
  local zn="$2"
  for _ in $(seq 1 35); do
    if gcloud compute ssh "${inst}" --project="${PROJECT_ID}" --zone="${zn}" --command="echo SSH_OK" --quiet 2>/dev/null; then
      return 0
    fi
    sleep 4
  done
  return 1
}

case "${cmd}" in
  up)
    if [[ -f "${STATE_FILE}" ]]; then
      source "${STATE_FILE}"
      if gcloud compute instances describe "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --format="value(status)" 2>/dev/null | grep -q "RUNNING"; then
        echo "[GCP] Reusing active GPU VM ${INSTANCE_NAME} (${GPU_MODE}) in ${ACTIVE_ZONE}"
        exit 0
      fi
    fi

    INSTANCE_NAME="nullspan-goal-gpu-$(date +%s)"
    ACTIVE_ZONE=""
    GPU_MODE=""
    PY_REMOTE="python3"

    # Priority 1: NVIDIA L4 (24GB Ada sm_89, native BF16/FP16, pre-baked PyTorch 2.9 + CUDA 12.9, ~45s boot)
    for mtype in g2-standard-8 g2-standard-4; do
      for z in us-west4-a us-west4-c us-west1-c northamerica-northeast1-b northamerica-northeast1-c northamerica-northeast2-a northamerica-northeast2-b europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west2-a europe-west3-a us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-west1-a us-west1-b; do
        echo "  Trying L4 (${mtype}) in ${z}..."
        if gcloud compute instances create "${INSTANCE_NAME}" \
            --project="${PROJECT_ID}" \
            --zone="${z}" \
            --machine-type="${mtype}" \
            --maintenance-policy=TERMINATE \
            --image-project="deeplearning-platform-release" \
            --image-family="pytorch-2-9-cu129-ubuntu-2204-nvidia-580" \
            --boot-disk-size=120GB \
            --boot-disk-type=pd-ssd \
            --quiet 2>/dev/null; then
          ACTIVE_ZONE="${z}"
          GPU_MODE="l4_dlvm"
          break 2
        fi
      done
    done

    # Priority 2: NVIDIA Tesla T4 (16GB Turing sm_75, native FP16 TensorCores, supported by nvidia-open 580, ~45s boot)
    if [[ -z "${ACTIVE_ZONE}" ]]; then
      for z in us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c us-west1-a us-west1-b; do
        echo "  Trying T4 (n1-standard-8) with DLVM image in ${z}..."
        if gcloud compute instances create "${INSTANCE_NAME}" \
            --project="${PROJECT_ID}" \
            --zone="${z}" \
            --machine-type="n1-standard-8" \
            --accelerator="type=nvidia-tesla-t4,count=1" \
            --maintenance-policy=TERMINATE \
            --image-project="deeplearning-platform-release" \
            --image-family="pytorch-2-9-cu129-ubuntu-2204-nvidia-580" \
            --boot-disk-size=120GB \
            --boot-disk-type=pd-ssd \
            --quiet 2>/dev/null; then
          ACTIVE_ZONE="${z}"
          GPU_MODE="t4_dlvm"
          break
        fi
      done
    fi

    # Priority 3: NVIDIA Tesla V100 (16GB Volta sm_70, Ubuntu 22.04 LTS + proprietary nvidia-580-server driver)
    if [[ -z "${ACTIVE_ZONE}" ]]; then
      for z in us-central1-a us-central1-b us-central1-c us-central1-f us-west1-a us-west1-b; do
        echo "  Trying V100 (n1-standard-8) with Ubuntu 22.04 + 580-server in ${z}..."
        if gcloud compute instances create "${INSTANCE_NAME}" \
            --project="${PROJECT_ID}" \
            --zone="${z}" \
            --machine-type="n1-standard-8" \
            --accelerator="type=nvidia-tesla-v100,count=1" \
            --maintenance-policy=TERMINATE \
            --image-project="ubuntu-os-cloud" \
            --image-family="ubuntu-2204-lts" \
            --boot-disk-size=100GB \
            --boot-disk-type=pd-ssd \
            --quiet 2>/dev/null; then
          ACTIVE_ZONE="${z}"
          GPU_MODE="v100_ubuntu"
          PY_REMOTE="\$HOME/venv/bin/python"
          break
        fi
      done
    fi

    if [[ -z "${ACTIVE_ZONE}" ]]; then
      echo "ERROR: Could not provision GPU VM in any region." >&2
      exit 1
    fi

    cat > "${STATE_FILE}" <<EOF
INSTANCE_NAME=${INSTANCE_NAME}
ACTIVE_ZONE=${ACTIVE_ZONE}
GPU_MODE=${GPU_MODE}
PROJECT_ID=${PROJECT_ID}
PY_REMOTE=${PY_REMOTE}
EOF
    echo "[GCP] Provisioned ${INSTANCE_NAME} (${GPU_MODE}) in ${ACTIVE_ZONE}. Waiting for SSH..."
    wait_for_ssh "${INSTANCE_NAME}" "${ACTIVE_ZONE}"

    if [[ "${GPU_MODE}" == "v100_ubuntu" ]]; then
      echo "[GCP] Installing proprietary NVIDIA 580-server driver on V100..."
      gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet --command='
        set -euo pipefail
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
          git python3-venv python3-pip \
          linux-modules-nvidia-580-server-gcp \
          nvidia-headless-580-server \
          nvidia-utils-580-server
        echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf >/dev/null
      '
      gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet --command="sudo reboot" 2>/dev/null || true
      sleep 10
      wait_for_ssh "${INSTANCE_NAME}" "${ACTIVE_ZONE}"
      gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet --command='
        set -euo pipefail
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        python3 -m venv ~/venv
        ~/venv/bin/pip install -q --upgrade pip
        ~/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126
        ~/venv/bin/pip install -q "transformers>=4.44.0,<5.0.0" safetensors datasets huggingface_hub sentencepiece protobuf tiktoken onnx onnxruntime
        mkdir -p ~/work
        [[ -d ~/jevbench ]] || git clone --depth 1 https://github.com/fstandhartinger/jevbench.git ~/jevbench
        cd ~/work
        ~/venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(\"gnumanth/nullspan\", local_dir=\"artifacts/nullspan\", ignore_patterns=[\"onnx/*\"])"
        echo "REMOTE_ENV_READY"
      '
    else
      gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet --command='
        set -euo pipefail
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        mkdir -p ~/work
        [[ -d ~/jevbench ]] || git clone --depth 1 https://github.com/fstandhartinger/jevbench.git ~/jevbench
        PY_BIN="/opt/conda/bin/python"
        [[ -x "${PY_BIN}" ]] || PY_BIN="python3"
        "${PY_BIN}" -m pip install -q --no-warn-script-location "transformers>=4.44.0,<5.0.0" safetensors datasets huggingface_hub sentencepiece protobuf tiktoken onnx onnxruntime
        cd ~/work
        "${PY_BIN}" -c "from huggingface_hub import snapshot_download; snapshot_download(\"gnumanth/nullspan\", local_dir=\"artifacts/nullspan\", ignore_patterns=[\"onnx/*\"])"
        echo "REMOTE_ENV_READY"
      '
    fi
    ;;

  push)
    source "${STATE_FILE}"
    TMP_TAR="$(mktemp /tmp/nullspan_push.XXXXXX)"
    COPYFILE_DISABLE=1 tar -czf "${TMP_TAR}" -C "${ROOT_DIR}" "$@"
    gcloud compute scp "${TMP_TAR}" "${INSTANCE_NAME}:~/push_bundle.tar.gz" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet
    rm -f "${TMP_TAR}"
    gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet --command="cd ~/work && tar -xzf ~/push_bundle.tar.gz && rm -f ~/push_bundle.tar.gz"
    ;;

  exec)
    source "${STATE_FILE}"
    gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet --command="export PY_BIN=${PY_REMOTE}; cd ~/work && $*"
    ;;

  pull)
    source "${STATE_FILE}"
    TMP_TAR="$(mktemp /tmp/nullspan_pull.XXXXXX)"
    gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet --command="cd ~/work && tar -czf ~/pull_bundle.tar.gz $*"
    gcloud compute scp "${INSTANCE_NAME}:~/pull_bundle.tar.gz" "${TMP_TAR}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet
    tar -xzf "${TMP_TAR}" -C "${ROOT_DIR}"
    rm -f "${TMP_TAR}" ~/pull_bundle.tar.gz 2>/dev/null || true
    ;;

  down)
    if [[ -f "${STATE_FILE}" ]]; then
      source "${STATE_FILE}"
      echo "[GCP] Deleting GPU VM ${INSTANCE_NAME} (${ACTIVE_ZONE})..."
      gcloud compute instances delete "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ACTIVE_ZONE}" --quiet || true
      rm -f "${STATE_FILE}"
      echo "[GCP] Deleted ${INSTANCE_NAME}."
    fi
    ;;

  status)
    if [[ -f "${STATE_FILE}" ]]; then
      cat "${STATE_FILE}"
    else
      echo "No active .gcp_session.env"
    fi
    ;;
esac
