#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_DIR="${WORK_DIR:-${SOURCE}/artifacts}"
BUILD_DIR="${BUILD_DIR:-${WORK_DIR}/build}"
OUT_SUFFIX="${OUT_SUFFIX:-full_e2e}"
OUT="${WORK_DIR}/${OUT_SUFFIX}"

MODEL="${MODEL:?Set MODEL to the Qwen3.5-9B checkpoint directory}"
DATASET="${DATASET:?Set DATASET to an ISL-1024/OSL-1024 benchmark JSON file}"
CONFIG="${CONFIG:?Set CONFIG to the TensorRT-LLM benchmark YAML file}"
IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc21}"

TARGET_INPUT_LEN="${TARGET_INPUT_LEN:-1024}"
TARGET_OUTPUT_LEN="${TARGET_OUTPUT_LEN:-1024}"
RESIDENT_TOKENS="${RESIDENT_TOKENS:-$((TARGET_OUTPUT_LEN - 2))}"
RUN_BASELINE="${RUN_BASELINE:-yes}"
RUN_RESIDENT="${RUN_RESIDENT:-yes}"
EXTENSION="${BUILD_DIR}/persistent_gdn_state_r192/persistent_gdn_state_r192.so"

mkdir -p "${BUILD_DIR}" "${OUT}"
chmod a+rwx "${BUILD_DIR}" "${OUT}"

if [[ ! -f "${EXTENSION}" ]]; then
    docker run --rm \
        --gpus all \
        --ipc host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e TORCH_EXTENSIONS_DIR=/persistent_build \
        -e GDN_MAX_REGS=192 \
        -v "${SOURCE}:/persistent_patch:ro" \
        -v "${BUILD_DIR}:/persistent_build" \
        "${IMAGE}" \
        bash -lc \
        "cd /persistent_patch && python -c 'from test_persistent_state import build; build()'"
fi

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "hostname=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv

bench_args=(
    trtllm-bench
    -m Qwen/Qwen3.5-9B
    --model_path /model
    throughput
    --backend pytorch
    --dataset /inputs/dataset.json
    --num_requests 1
    --warmup 0
    --target_input_len "${TARGET_INPUT_LEN}"
    --target_output_len "${TARGET_OUTPUT_LEN}"
    --tp 1
    --concurrency 1
    --streaming
    --config /inputs/config.yaml
    --max_batch_size 1
    --max_num_tokens 1025
    --max_seq_len 2048
)

common_docker_args=(
    --rm
    --gpus all
    --ipc host
    --cap-add SYS_ADMIN
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${MODEL}:/model:ro"
    -v "${DATASET}:/inputs/dataset.json:ro"
    -v "${CONFIG}:/inputs/config.yaml:ro"
    -v "${WORK_DIR}:/work"
)

run_baseline()
{
    local run_dir="${OUT}/baseline"
    local rc

    mkdir -p "${run_dir}/home"
    chmod a+rwx "${run_dir}" "${run_dir}/home"
    echo "CASE_BEGIN name=baseline" | tee "${run_dir}/run.log"

    set +e
    docker run \
        "${common_docker_args[@]}" \
        --name "qwen35-full-baseline-${SLURM_JOB_ID:-manual}-$$" \
        -e HOME="/work/${OUT_SUFFIX}/baseline/home" \
        "${IMAGE}" \
        "${bench_args[@]}" \
        2>&1 | tee -a "${run_dir}/run.log"
    rc=${PIPESTATUS[0]}
    set -e

    echo "CASE_END name=baseline rc=${rc}" | tee -a "${run_dir}/run.log"
    return "${rc}"
}

run_resident()
{
    local run_dir="${OUT}/resident"
    local activation="${run_dir}/enable_capture"
    local deadline
    local rc
    local run_pid

    mkdir -p "${run_dir}/home"
    chmod a+rwx "${run_dir}" "${run_dir}/home"
    rm -f "${activation}" "${run_dir}/run.log"

    echo "CASE_BEGIN name=resident" | tee "${run_dir}/run.log"
    docker run \
        "${common_docker_args[@]}" \
        --name "qwen35-full-resident-${SLURM_JOB_ID:-manual}-$$" \
        -e HOME="/work/${OUT_SUFFIX}/resident/home" \
        -e PYTHONPATH=/persistent_patch \
        -e TRTLLM_PERSISTENT_GDN_EXTENSION=/persistent_build/persistent_gdn_state_r192/persistent_gdn_state_r192.so \
        -e TRTLLM_PERSISTENT_GDN_ACTIVATION_FILE="/work/${OUT_SUFFIX}/resident/enable_capture" \
        -e TRTLLM_PERSISTENT_GDN_RESIDENT_TOKENS="${RESIDENT_TOKENS}" \
        -e TRTLLM_PERSISTENT_GDN_TOKEN_LOG_INTERVAL=100 \
        -v "${SOURCE}:/persistent_patch:ro" \
        -v "${BUILD_DIR}:/persistent_build:ro" \
        "${IMAGE}" \
        "${bench_args[@]}" \
        2>&1 | tee -a "${run_dir}/run.log" &
    run_pid=$!

    deadline=$((SECONDS + 900))
    while ! grep -q "Setting PyTorch memory fraction" "${run_dir}/run.log"; do
        if ! kill -0 "${run_pid}" 2>/dev/null; then
            wait "${run_pid}"
            return $?
        fi
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for initialized benchmark worker" >&2
            kill "${run_pid}" || true
            wait "${run_pid}" || true
            return 1
        fi
        sleep 0.01
    done

    touch "${activation}"
    echo "PERSISTENT_GDN_HOST_ACTIVATED: model/JIT initialization complete" \
        | tee -a "${run_dir}/run.log"

    set +e
    wait "${run_pid}"
    rc=$?
    set -e

    echo "CASE_END name=resident rc=${rc}" | tee -a "${run_dir}/run.log"
    return "${rc}"
}

cases=()
if [[ "${RUN_BASELINE}" == "yes" ]]; then
    run_baseline
    cases+=(baseline)
fi
if [[ "${RUN_RESIDENT}" == "yes" ]]; then
    run_resident
    cases+=(resident)
fi

{
    echo "case,total_latency_ms,ttft_ms,tpot_ms,output_tps"
    for name in "${cases[@]}"; do
        log="${OUT}/${name}/run.log"
        total=$(grep "Total Latency (ms):" "${log}" | tail -1 | awk '{print $NF}')
        ttft=$(grep "Average time-to-first-token" "${log}" | tail -1 | awk '{print $NF}')
        tpot=$(grep "Average time-per-output-token" "${log}" | tail -1 | awk '{print $NF}')
        tps=$(grep "Total Output Throughput" "${log}" | tail -1 | awk '{print $NF}')
        echo "${name},${total},${ttft},${tpot},${tps}"
    done
} >"${OUT}/metrics.csv"

cat "${OUT}/metrics.csv"
