/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <utility>

#include "register_state_generated.cuh"

namespace
{

constexpr int kLayers = 24;
constexpr int kRows = 4096;
constexpr int kK = 128;
constexpr int kRowsPerCta = 32;
constexpr int kRfRowsPerCta = 12;
constexpr int kTmemRowsPerCta = 20;
constexpr int kThreads = 256;
constexpr int kBlocks = (kRows + kRowsPerCta - 1) / kRowsPerCta; // 128
constexpr int kTmemColumns0 = 256;
constexpr int kTmemColumns1 = 256;

#ifndef GDN_MAX_REGS
#define GDN_MAX_REGS 160
#endif

enum Opcode : uint32_t
{
    kIdle = 0,
    kNoop = 1,
    kAddRfLayer = 2,
    kAddTmemLayer = 3,
    kAddLayer = 4,
    kStopAndEvict = 5,
    kGdnDecode = 6,
    kStopWithoutEvict = 7,
    kGdnDecodeAndWrite = 8,
    kGdnDecodeRoundTrip = 9,
};

struct alignas(64) ServiceControl
{
    uint32_t epoch;
    uint32_t done_epoch;
    uint32_t ready_blocks;
    uint32_t arrived_blocks;
    uint32_t opcode;
    uint32_t layer;
    float delta;
    uint32_t reserved;
    uint64_t final_state;
    uint64_t q;
    uint64_t k;
    uint64_t v;
    uint64_t a;
    uint64_t b;
    uint64_t a_log;
    uint64_t dt_bias;
    uint64_t output;
    uint64_t state;
};

__device__ __forceinline__ int64_t rf_global_offset(int layer, int tid, int slot, int row_base)
{
    int const local = tid + slot * kThreads;
    int const local_row = local / kK;
    int const k = local % kK;
    int const row = row_base + local_row;
    // The 128 CTAs partition all 4096 rows exactly.
    return (static_cast<int64_t>(layer) * kRows + row) * kK + k;
}

__device__ __forceinline__ int64_t rf_layer_offset(int tid, int slot, int row_base)
{
    int const local = tid + slot * kThreads;
    int const local_row = local / kK;
    int const k = local % kK;
    int const row = row_base + local_row;
    return static_cast<int64_t>(row) * kK + k;
}

__device__ __forceinline__ bool rf_row_valid(int tid, int slot, int row_base)
{
    int const local = tid + slot * kThreads;
    return row_base + local / kK < kRows;
}

__device__ __forceinline__ uint32_t to_smem_addr(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void tmem_alloc(uint32_t* destination, uint32_t columns)
{
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030
    const uint32_t smem_addr = to_smem_addr(destination);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(smem_addr), "r"(columns)
                 : "memory");
#else
    (void) destination;
    (void) columns;
#endif
}

__device__ __forceinline__ void tmem_dealloc(uint32_t base, uint32_t columns)
{
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" : : "r"(base), "r"(columns) : "memory");
#else
    (void) base;
    (void) columns;
#endif
}

__device__ __forceinline__ void tmem_relinquish()
{
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" : : : "memory");
#endif
}

__device__ __forceinline__ float tmem_load_one(uint32_t address)
{
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030
    uint32_t bits;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(bits) : "r"(address) : "memory");
    return __uint_as_float(bits);
#else
    (void) address;
    return 0.0f;
#endif
}

__device__ __forceinline__ void tmem_store_one(uint32_t address, float value)
{
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030
    const uint32_t bits = __float_as_uint(value);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" : : "r"(address), "r"(bits) : "memory");
#else
    (void) address;
    (void) value;
#endif
}

__device__ __forceinline__ void tmem_wait_load()
{
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030
    asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
#endif
}

__device__ __forceinline__ void tmem_wait_store()
{
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1030
    asm volatile("tcgen05.wait::st.sync.aligned;" : : : "memory");
#endif
}

__device__ __forceinline__ uint32_t tmem_address(
    uint32_t base0, uint32_t base1, int layer, int local_row, int k_partition)
{
    bool const in_first_allocation = layer < 12;
    const uint32_t base = in_first_allocation ? base0 : base1;
    int const allocation_layer = in_first_allocation ? layer : layer - 12;
    int const column = allocation_layer * kTmemRowsPerCta + local_row;
    return base + static_cast<uint32_t>(column) + (static_cast<uint32_t>(k_partition * 32) << 16);
}

__device__ __forceinline__ int64_t tmem_global_offset(
    int layer, int local_tmem_row, int lane, int k_partition, int row_base)
{
    int const row = row_base + kRfRowsPerCta + local_tmem_row;
    int const k = k_partition * 32 + lane;
    return (static_cast<int64_t>(layer) * kRows + row) * kK + k;
}

__device__ __forceinline__ uint32_t load_acquire_u32(uint32_t const* ptr)
{
    uint32_t value;
    asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ void store_release_u32(uint32_t* ptr, uint32_t value)
{
    asm volatile("st.global.release.sys.u32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ float warp_sum(float value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ __maxnreg__(GDN_MAX_REGS) void persistent_state_kernel(ServiceControl* control, float const* init_state)
{
    __shared__ uint32_t tmem_base_shared[2];
    __shared__ uint32_t command_epoch;
    __shared__ uint32_t command_opcode;
    __shared__ uint32_t command_layer;
    __shared__ float command_delta;
    __shared__ uint64_t command_final_state;
    __shared__ uint64_t command_q;
    __shared__ uint64_t command_k;
    __shared__ uint64_t command_v;
    __shared__ uint64_t command_a;
    __shared__ uint64_t command_b;
    __shared__ uint64_t command_a_log;
    __shared__ uint64_t command_dt_bias;
    __shared__ uint64_t command_output;
    __shared__ uint64_t command_state;
    __shared__ float rf_stage[kRfRowsPerCta * kK];
    __shared__ float q_inv_norm[16];
    __shared__ float k_inv_norm[16];
    __shared__ float state_decay[32];
    __shared__ float beta_value[32];
    __shared__ float group_partial[2][4];
    __shared__ float group_scalar[2];

    int const tid = threadIdx.x;
    int const warp = tid / 32;
    int const lane = tid % 32;
    int const row_base = blockIdx.x * kRowsPerCta;

    GDN_DECLARE_RF_STATE();

    // One complete warp participates in the TMEM allocation instruction.
    if (warp == 0)
    {
        tmem_alloc(&tmem_base_shared[0], kTmemColumns0);
        tmem_alloc(&tmem_base_shared[1], kTmemColumns1);
        tmem_relinquish();
    }
    __syncthreads();
    const uint32_t tmem_base0 = tmem_base_shared[0];
    const uint32_t tmem_base1 = tmem_base_shared[1];

    // Initial transfer from global memory into the 144 scalar RF registers.
#pragma unroll 1
    for (int layer = 0; layer < kLayers; ++layer)
    {
        GDN_LOAD_RF_SWITCH(layer, asm volatile(""));
    }

    // Initial transfer of the remaining 20 rows/layer into 480 TMEM columns.
    // A TMEM 32x32b access is tied to the issuing warp's 32-row subpartition.
    // Two four-warp groups split columns; warp%4 owns one K partition.
    {
        int const warp_group = warp / 4;
        int const part = warp % 4;
#pragma unroll 1
        for (int layer = 0; layer < kLayers; ++layer)
        {
#pragma unroll 1
            for (int local_row = 0; local_row < kTmemRowsPerCta; ++local_row)
            {
                int const linear_column = layer * kTmemRowsPerCta + local_row;
                if ((linear_column & 1) != warp_group)
                {
                    continue;
                }
                int const row = row_base + kRfRowsPerCta + local_row;
                float value = 0.0f;
                if (row < kRows)
                {
                    value = init_state[tmem_global_offset(layer, local_row, lane, part, row_base)];
                }
                tmem_store_one(tmem_address(tmem_base0, tmem_base1, layer, local_row, part), value);
            }
        }
        tmem_wait_store();
    }
    __syncthreads();

    if (tid == 0)
    {
        atomicAdd(&control->ready_blocks, 1u);
    }

    uint32_t local_epoch = 0;
    bool running = true;
    while (running)
    {
        if (tid == 0)
        {
            uint32_t observed;
            do
            {
                observed = load_acquire_u32(&control->epoch);
                if (observed == local_epoch)
                {
                    __nanosleep(256);
                }
            } while (observed == local_epoch);
            command_epoch = observed;
            command_opcode = control->opcode;
            command_layer = control->layer;
            command_delta = control->delta;
            command_final_state = control->final_state;
            command_q = control->q;
            command_k = control->k;
            command_v = control->v;
            command_a = control->a;
            command_b = control->b;
            command_a_log = control->a_log;
            command_dt_bias = control->dt_bias;
            command_output = control->output;
            command_state = control->state;
        }
        __syncthreads();

        const uint32_t opcode = command_opcode;
        const uint32_t layer = command_layer;
        float const delta = command_delta;

        if (opcode == kAddRfLayer || opcode == kAddLayer)
        {
            GDN_ADD_RF_SWITCH(layer, asm volatile(""));
        }

        if (opcode == kAddTmemLayer || opcode == kAddLayer)
        {
            // Each four-warp group owns ten rows. Warp%4 accesses one K
            // subpartition for every row handled by the group.
            int const warp_group = warp / 4;
            int const part = warp % 4;
#pragma unroll 1
            for (int local_row = warp_group; local_row < kTmemRowsPerCta; local_row += 2)
            {
                const uint32_t address = tmem_address(tmem_base0, tmem_base1, static_cast<int>(layer), local_row, part);
                float value = tmem_load_one(address);
                tmem_wait_load();
                value += delta;
                tmem_store_one(address, value);
            }
            tmem_wait_store();
        }
        else if (opcode == kGdnDecode || opcode == kGdnDecodeAndWrite || opcode == kGdnDecodeRoundTrip)
        {
            __nv_bfloat16* state = reinterpret_cast<__nv_bfloat16*>(command_state);
            if (opcode == kGdnDecodeRoundTrip)
            {
                // Reload this layer from the same BF16 global-memory layout
                // consumed by the FlashInfer baseline. This deliberately
                // removes the persistence advantage for the matched
                // microbenchmark.
                GDN_LOAD_RF_BF16_SWITCH(layer, asm volatile(""));

                int const warp_group = warp / 4;
                int const part = warp % 4;
#pragma unroll 1
                for (int local_row = 0; local_row < kTmemRowsPerCta; ++local_row)
                {
                    int const linear_column = static_cast<int>(layer) * kTmemRowsPerCta + local_row;
                    if ((linear_column & 1) != warp_group)
                    {
                        continue;
                    }
                    int const row = row_base + kRfRowsPerCta + local_row;
                    int const k_index = part * 32 + lane;
                    float value = 0.0f;
                    if (row < kRows)
                    {
                        value = __bfloat162float(state[static_cast<int64_t>(row) * kK + k_index]);
                    }
                    tmem_store_one(
                        tmem_address(tmem_base0, tmem_base1, static_cast<int>(layer), local_row, part), value);
                }
                tmem_wait_store();
                __syncthreads();
            }

            // Stage this layer's register-resident rows into shared memory so the
            // four-warp K-reduction groups can consume them.
            GDN_STAGE_RF_SWITCH(layer, asm volatile(""));
            __syncthreads();

            {
                __nv_bfloat16 const* q = reinterpret_cast<__nv_bfloat16 const*>(command_q);
                __nv_bfloat16 const* k = reinterpret_cast<__nv_bfloat16 const*>(command_k);
                // One warp computes the L2 normalization factor for two key heads.
#pragma unroll
                for (int head_offset = 0; head_offset < 2; ++head_offset)
                {
                    int const head = warp + head_offset * 8;
                    float squared = 0.0f;
#pragma unroll
                    for (int item = 0; item < 4; ++item)
                    {
                        float const value = __bfloat162float(q[head * kK + item * 32 + lane]);
                        squared = fmaf(value, value, squared);
                    }
                    squared = warp_sum(squared);
                    if (lane == 0)
                    {
                        q_inv_norm[head] = 1.0f / (sqrtf(squared) + 1.0e-6f);
                    }

                    squared = 0.0f;
#pragma unroll
                    for (int item = 0; item < 4; ++item)
                    {
                        float const value = __bfloat162float(k[head * kK + item * 32 + lane]);
                        squared = fmaf(value, value, squared);
                    }
                    squared = warp_sum(squared);
                    if (lane == 0)
                    {
                        k_inv_norm[head] = 1.0f / (sqrtf(squared) + 1.0e-6f);
                    }
                }
            }

            if (tid < 32)
            {
                __nv_bfloat16 const* a = reinterpret_cast<__nv_bfloat16 const*>(command_a);
                __nv_bfloat16 const* b = reinterpret_cast<__nv_bfloat16 const*>(command_b);
                float const* a_log = reinterpret_cast<float const*>(command_a_log);
                float const* dt_bias = reinterpret_cast<float const*>(command_dt_bias);
                float const x = __bfloat162float(a[tid]) + dt_bias[tid];
                float const softplus = x <= 20.0f ? log1pf(expf(x)) : x;
                float const g = -expf(a_log[tid]) * softplus;
                state_decay[tid] = expf(g);
                float const b_float = __bfloat162float(b[tid]);
                beta_value[tid] = 1.0f / (1.0f + expf(-b_float));
            }
            __syncthreads();

            int const warp_group = warp / 4;
            int const part = warp % 4;
            constexpr float kQueryScale = 0.08838834764831845f;
            __nv_bfloat16 const* q = reinterpret_cast<__nv_bfloat16 const*>(command_q);
            __nv_bfloat16 const* k = reinterpret_cast<__nv_bfloat16 const*>(command_k);
            __nv_bfloat16 const* v = reinterpret_cast<__nv_bfloat16 const*>(command_v);
            __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(command_output);
#pragma unroll 1
            for (int iteration = 0; iteration < 16; ++iteration)
            {
                int const local_row = warp_group + iteration * 2;
                int const row = row_base + local_row;
                bool const valid = row < kRows;
                int const safe_row = valid ? row : 0;
                int const value_head = safe_row / 128;
                int const key_head = value_head / 2;
                int const k_index = part * 32 + lane;

                float h;
                if (local_row < kRfRowsPerCta)
                {
                    h = rf_stage[local_row * kK + k_index];
                }
                else
                {
                    h = tmem_load_one(
                        tmem_address(tmem_base0, tmem_base1, static_cast<int>(layer), local_row - kRfRowsPerCta, part));
                    tmem_wait_load();
                }
                h *= valid ? state_decay[value_head] : 0.0f;

                float const k_normalized = __bfloat162float(k[key_head * kK + k_index]) * k_inv_norm[key_head];
                float dot_hk = warp_sum(h * k_normalized);
                if (lane == 0)
                {
                    group_partial[warp_group][part] = dot_hk;
                }
                __syncthreads();

                if (part == 0 && lane == 0)
                {
                    float const full_dot = group_partial[warp_group][0] + group_partial[warp_group][1]
                        + group_partial[warp_group][2] + group_partial[warp_group][3];
                    group_scalar[warp_group]
                        = valid ? (__bfloat162float(v[safe_row]) - full_dot) * beta_value[value_head] : 0.0f;
                }
                __syncthreads();

                float const v_new = group_scalar[warp_group];
                h = fmaf(k_normalized, v_new, h);
                float const q_normalized
                    = __bfloat162float(q[key_head * kK + k_index]) * q_inv_norm[key_head] * kQueryScale;
                float dot_hq = warp_sum(h * q_normalized);
                if (lane == 0)
                {
                    group_partial[warp_group][part] = dot_hq;
                }
                __syncthreads();

                if (part == 0 && lane == 0 && valid)
                {
                    float const full_dot = group_partial[warp_group][0] + group_partial[warp_group][1]
                        + group_partial[warp_group][2] + group_partial[warp_group][3];
                    output[safe_row] = __float2bfloat16_rn(full_dot);
                }

                if (local_row < kRfRowsPerCta)
                {
                    rf_stage[local_row * kK + k_index] = h;
                }
                else
                {
                    tmem_store_one(
                        tmem_address(tmem_base0, tmem_base1, static_cast<int>(layer), local_row - kRfRowsPerCta, part),
                        h);
                    tmem_wait_store();
                }
                __syncthreads();
            }

            // Return the updated 12 staged rows to their owning scalar registers.
            GDN_UNSTAGE_RF_SWITCH(layer, asm volatile(""));

            if (opcode == kGdnDecodeAndWrite || opcode == kGdnDecodeRoundTrip)
            {
                // Round the updated state back to the baseline's BF16
                // global-memory representation. The write-only mode starts
                // from resident RF/TMEM; round-trip mode also reloads above.
                GDN_STORE_RF_BF16_SWITCH(layer, asm volatile(""));

                int const warp_group = warp / 4;
                int const part = warp % 4;
#pragma unroll 1
                for (int local_row = 0; local_row < kTmemRowsPerCta; ++local_row)
                {
                    int const linear_column = static_cast<int>(layer) * kTmemRowsPerCta + local_row;
                    if ((linear_column & 1) != warp_group)
                    {
                        continue;
                    }
                    int const row = row_base + kRfRowsPerCta + local_row;
                    float value
                        = tmem_load_one(tmem_address(tmem_base0, tmem_base1, static_cast<int>(layer), local_row, part));
                    tmem_wait_load();
                    if (row < kRows)
                    {
                        int const k_index = part * 32 + lane;
                        state[static_cast<int64_t>(row) * kK + k_index] = __float2bfloat16_rn(value);
                    }
                }
            }
        }
        else if (opcode == kStopAndEvict)
        {
            float* final_state = reinterpret_cast<float*>(command_final_state);
#pragma unroll 1
            for (int layer_to_store = 0; layer_to_store < kLayers; ++layer_to_store)
            {
                GDN_STORE_RF_SWITCH(layer_to_store, asm volatile(""));
            }

            {
                int const warp_group = warp / 4;
                int const part = warp % 4;
#pragma unroll 1
                for (int layer_to_store = 0; layer_to_store < kLayers; ++layer_to_store)
                {
#pragma unroll 1
                    for (int local_row = 0; local_row < kTmemRowsPerCta; ++local_row)
                    {
                        int const linear_column = layer_to_store * kTmemRowsPerCta + local_row;
                        if ((linear_column & 1) != warp_group)
                        {
                            continue;
                        }
                        int const row = row_base + kRfRowsPerCta + local_row;
                        float value
                            = tmem_load_one(tmem_address(tmem_base0, tmem_base1, layer_to_store, local_row, part));
                        tmem_wait_load();
                        if (row < kRows)
                        {
                            final_state[tmem_global_offset(layer_to_store, local_row, lane, part, row_base)] = value;
                        }
                    }
                }
            }
            running = false;
        }
        else if (opcode == kStopWithoutEvict)
        {
            // Release RF/TMEM ownership without materializing state in global
            // memory. This is the production-run teardown path for a completed
            // request whose recurrent state will not be reused.
            running = false;
        }
        __syncthreads();

        if (tid == 0)
        {
            const uint32_t previous = atomicAdd(&control->arrived_blocks, 1u);
            if (previous == kBlocks - 1)
            {
                control->arrived_blocks = 0;
                __threadfence_system();
                store_release_u32(&control->done_epoch, command_epoch);
            }
        }
        local_epoch = command_epoch;
        __syncthreads();
    }

    if (warp == 0)
    {
        tmem_dealloc(tmem_base1, kTmemColumns1);
        tmem_dealloc(tmem_base0, kTmemColumns0);
    }
}

std::mutex g_mutex;
cudaStream_t g_service_stream = nullptr;
torch::Tensor g_init_state;
torch::Tensor g_control;
uint32_t g_epoch = 0;
bool g_running = false;

void check_cuda(cudaError_t status, char const* operation)
{
    TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

void check_driver(CUresult status, char const* operation)
{
    if (status == CUDA_SUCCESS)
    {
        return;
    }
    char const* name = nullptr;
    char const* message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    TORCH_CHECK(false, operation, " failed: ", name == nullptr ? "unknown" : name, " (",
        message == nullptr ? "no description" : message, ")");
}

CUdeviceptr control_field(torch::Tensor const& control, size_t offset)
{
    return reinterpret_cast<CUdeviceptr>(control.data_ptr<uint8_t>()) + offset;
}

void stream_write32(cudaStream_t stream, torch::Tensor const& control, size_t offset, uint32_t value)
{
    check_driver(cuStreamWriteValue32(reinterpret_cast<CUstream>(stream), control_field(control, offset), value,
                     CU_STREAM_WRITE_VALUE_DEFAULT),
        "cuStreamWriteValue32");
}

void stream_write64(cudaStream_t stream, torch::Tensor const& control, size_t offset, uint64_t value)
{
    check_driver(cuStreamWriteValue64(reinterpret_cast<CUstream>(stream), control_field(control, offset), value,
                     CU_STREAM_WRITE_VALUE_DEFAULT),
        "cuStreamWriteValue64");
}

void stream_wait_epoch(cudaStream_t stream, torch::Tensor const& control, uint32_t epoch)
{
    check_driver(cuStreamWaitValue32(reinterpret_cast<CUstream>(stream),
                     control_field(control, offsetof(ServiceControl, done_epoch)), epoch, CU_STREAM_WAIT_VALUE_GEQ),
        "cuStreamWaitValue32");
}

void enqueue_command(cudaStream_t stream, torch::Tensor const& control, uint32_t epoch, uint32_t opcode, uint32_t layer,
    float delta, uint64_t final_state, bool wait_for_completion = true)
{
    uint32_t delta_bits;
    static_assert(sizeof(delta_bits) == sizeof(delta));
    std::memcpy(&delta_bits, &delta, sizeof(delta_bits));
    stream_write32(stream, control, offsetof(ServiceControl, opcode), opcode);
    stream_write32(stream, control, offsetof(ServiceControl, layer), layer);
    stream_write32(stream, control, offsetof(ServiceControl, delta), delta_bits);
    stream_write64(stream, control, offsetof(ServiceControl, final_state), final_state);
    // The default stream-write semantics fence all earlier descriptor writes
    // before publishing the new epoch to the resident CTAs.
    stream_write32(stream, control, offsetof(ServiceControl, epoch), epoch);
    if (wait_for_completion)
    {
        stream_wait_epoch(stream, control, epoch);
    }
}

void validate_state(torch::Tensor const& state)
{
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be float32");
    TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
    TORCH_CHECK(state.sizes() == torch::IntArrayRef({kLayers, kRows, kK}),
        "state must have shape [24, 4096, 128], got ", state.sizes());
}

void validate_cuda_contiguous(torch::Tensor const& tensor, torch::ScalarType dtype, int64_t elements, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() == elements, name, " must contain ", elements, " elements, got ", tensor.numel());
}

torch::Tensor start(torch::Tensor init_state)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(!g_running, "persistent state service is already running");
    validate_state(init_state);
    c10::cuda::CUDAGuard guard(init_state.device());

    g_init_state = init_state;
    g_control = torch::zeros({static_cast<long>(sizeof(ServiceControl))},
        torch::TensorOptions().device(init_state.device()).dtype(torch::kUInt8));
    g_epoch = 0;

    if (g_service_stream == nullptr)
    {
        check_cuda(cudaStreamCreateWithFlags(&g_service_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }

    const cudaStream_t current = at::cuda::getCurrentCUDAStream();
    cudaEvent_t inputs_ready;
    check_cuda(cudaEventCreateWithFlags(&inputs_ready, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    check_cuda(cudaEventRecord(inputs_ready, current), "cudaEventRecord");
    check_cuda(cudaStreamWaitEvent(g_service_stream, inputs_ready), "cudaStreamWaitEvent");

    persistent_state_kernel<<<kBlocks, kThreads, 0, g_service_stream>>>(
        reinterpret_cast<ServiceControl*>(g_control.data_ptr<uint8_t>()), g_init_state.data_ptr<float>());
    check_cuda(cudaGetLastError(), "persistent_state_kernel launch");

    check_cuda(cudaEventDestroy(inputs_ready), "cudaEventDestroy");
    g_running = true;
    return g_control;
}

void add_layer(torch::Tensor control, int64_t layer, double delta)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(g_running, "persistent state service is not running");
    TORCH_CHECK(control.is_same(g_control), "control tensor does not own active service");
    TORCH_CHECK(layer >= 0 && layer < kLayers, "layer must be in [0, 24)");

    const cudaStream_t current = at::cuda::getCurrentCUDAStream();
    const uint32_t epoch = ++g_epoch;
    enqueue_command(
        current, control, epoch, kAddLayer, static_cast<uint32_t>(layer), static_cast<float>(delta), 0, false);
}

void debug_command(torch::Tensor control, int64_t opcode, int64_t layer, double delta)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(g_running, "persistent state service is not running");
    TORCH_CHECK(control.is_same(g_control), "control tensor does not own active service");
    TORCH_CHECK(opcode == kNoop || opcode == kAddRfLayer || opcode == kAddTmemLayer,
        "debug opcode must be 1 (noop), 2 (RF), or 3 (TMEM)");
    TORCH_CHECK(layer >= 0 && layer < kLayers, "layer must be in [0, 24)");

    const cudaStream_t current = at::cuda::getCurrentCUDAStream();
    const uint32_t epoch = ++g_epoch;
    enqueue_command(current, control, epoch, static_cast<uint32_t>(opcode), static_cast<uint32_t>(layer),
        static_cast<float>(delta), 0);
}

torch::Tensor gdn_decode_impl(torch::Tensor control, int64_t layer, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor state,
    torch::Tensor output, uint32_t opcode)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(g_running, "persistent state service is not running");
    TORCH_CHECK(control.is_same(g_control), "control tensor does not own active service");
    TORCH_CHECK(layer >= 0 && layer < kLayers, "layer must be in [0, 24)");
    TORCH_CHECK(opcode == kGdnDecode || opcode == kGdnDecodeAndWrite || opcode == kGdnDecodeRoundTrip,
        "invalid GDN decode opcode");
    validate_cuda_contiguous(q, torch::kBFloat16, 16 * 128, "q");
    validate_cuda_contiguous(k, torch::kBFloat16, 16 * 128, "k");
    validate_cuda_contiguous(v, torch::kBFloat16, 32 * 128, "v");
    validate_cuda_contiguous(a, torch::kBFloat16, 32, "a");
    validate_cuda_contiguous(b, torch::kBFloat16, 32, "b");
    validate_cuda_contiguous(a_log, torch::kFloat32, 32, "a_log");
    validate_cuda_contiguous(dt_bias, torch::kFloat32, 32, "dt_bias");
    TORCH_CHECK(q.device() == g_init_state.device() && k.device() == q.device() && v.device() == q.device()
            && a.device() == q.device() && b.device() == q.device() && a_log.device() == q.device()
            && dt_bias.device() == q.device(),
        "all GDN tensors must be on the service device");
    if (opcode != kGdnDecode)
    {
        validate_cuda_contiguous(state, torch::kBFloat16, kRows * kK, "state");
        TORCH_CHECK(state.device() == q.device(), "state must be on the service device");
    }
    if (output.defined())
    {
        validate_cuda_contiguous(output, torch::kBFloat16, 32 * 128, "output");
        TORCH_CHECK(output.device() == q.device(), "output must be on the service device");
    }
    else
    {
        output = torch::empty({32, 128}, v.options());
    }

    const cudaStream_t current = at::cuda::getCurrentCUDAStream();
    stream_write64(current, control, offsetof(ServiceControl, q), reinterpret_cast<uint64_t>(q.data_ptr()));
    stream_write64(current, control, offsetof(ServiceControl, k), reinterpret_cast<uint64_t>(k.data_ptr()));
    stream_write64(current, control, offsetof(ServiceControl, v), reinterpret_cast<uint64_t>(v.data_ptr()));
    stream_write64(current, control, offsetof(ServiceControl, a), reinterpret_cast<uint64_t>(a.data_ptr()));
    stream_write64(current, control, offsetof(ServiceControl, b), reinterpret_cast<uint64_t>(b.data_ptr()));
    stream_write64(current, control, offsetof(ServiceControl, a_log), reinterpret_cast<uint64_t>(a_log.data_ptr()));
    stream_write64(current, control, offsetof(ServiceControl, dt_bias), reinterpret_cast<uint64_t>(dt_bias.data_ptr()));
    stream_write64(current, control, offsetof(ServiceControl, output), reinterpret_cast<uint64_t>(output.data_ptr()));
    stream_write64(current, control, offsetof(ServiceControl, state),
        state.defined() ? reinterpret_cast<uint64_t>(state.data_ptr()) : 0);

    const uint32_t epoch = ++g_epoch;
    enqueue_command(current, control, epoch, opcode, static_cast<uint32_t>(layer), 0.0f, 0);
    return output;
}

torch::Tensor gdn_decode(torch::Tensor control, int64_t layer, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias)
{
    return gdn_decode_impl(control, layer, std::move(q), std::move(k), std::move(v), std::move(a), std::move(b),
        std::move(a_log), std::move(dt_bias), torch::Tensor(), torch::Tensor(), kGdnDecode);
}

torch::Tensor gdn_decode_into(torch::Tensor control, int64_t layer, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor output)
{
    return gdn_decode_impl(control, layer, std::move(q), std::move(k), std::move(v), std::move(a), std::move(b),
        std::move(a_log), std::move(dt_bias), torch::Tensor(), std::move(output), kGdnDecode);
}

torch::Tensor gdn_decode_gmem(torch::Tensor control, int64_t layer, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor b, torch::Tensor a_log, torch::Tensor dt_bias, torch::Tensor state, bool reload,
    torch::Tensor output)
{
    return gdn_decode_impl(control, layer, std::move(q), std::move(k), std::move(v), std::move(a), std::move(b),
        std::move(a_log), std::move(dt_bias), std::move(state), std::move(output),
        reload ? kGdnDecodeRoundTrip : kGdnDecodeAndWrite);
}

torch::Tensor stop(torch::Tensor control)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(g_running, "persistent state service is not running");
    TORCH_CHECK(control.is_same(g_control), "control tensor does not own active service");

    auto final_state = torch::empty_like(g_init_state);
    const cudaStream_t current = at::cuda::getCurrentCUDAStream();
    const uint32_t epoch = ++g_epoch;
    enqueue_command(
        current, control, epoch, kStopAndEvict, 0, 0.0f, reinterpret_cast<uint64_t>(final_state.data_ptr<float>()));
    check_cuda(cudaStreamSynchronize(current), "stop current stream synchronize");
    check_cuda(cudaStreamSynchronize(g_service_stream), "stop service stream synchronize");

    g_running = false;
    g_init_state = torch::Tensor();
    g_control = torch::Tensor();
    return final_state;
}

void discard(torch::Tensor control)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(g_running, "persistent state service is not running");
    TORCH_CHECK(control.is_same(g_control), "control tensor does not own active service");

    const cudaStream_t current = at::cuda::getCurrentCUDAStream();
    const uint32_t epoch = ++g_epoch;
    enqueue_command(current, control, epoch, kStopWithoutEvict, 0, 0.0f, 0);
    check_cuda(cudaStreamSynchronize(current), "discard current stream synchronize");
    check_cuda(cudaStreamSynchronize(g_service_stream), "discard service stream synchronize");

    g_running = false;
    g_init_state = torch::Tensor();
    g_control = torch::Tensor();
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def("start", &start, "Start resident RF/TMEM state service");
    module.def("debug_command", &debug_command, "Run a staged debug command");
    module.def("add_layer", &add_layer, "Update one resident layer");
    module.def("gdn_decode", &gdn_decode, "Run one resident GDN decode layer");
    module.def("gdn_decode_into", &gdn_decode_into, "Run one resident GDN decode layer into a provided output");
    module.def("gdn_decode_gmem", &gdn_decode_gmem, "Run GDN decode with BF16 global-memory state writeback");
    module.def("stop", &stop, "Evict resident state once and stop");
    module.def("discard", &discard, "Discard resident state without global writeback");
}
