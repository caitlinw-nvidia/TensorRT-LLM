#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
printf 'Allocated-GPU compute processes (must be empty):\n'
if nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader | grep -q .; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader
    exit 23
fi

benchmark_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${benchmark_root}"
python3 examples/green_context/benchmark_executor_green_context.py "$@"
