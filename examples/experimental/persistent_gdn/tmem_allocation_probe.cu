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

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

__global__ void allocation_probe(int mode, uint32_t* result)
{
    __shared__ uint32_t bases[2];
    int const warp = threadIdx.x / 32;
    if (warp == 0)
    {
        if (mode == 0)
        {
            tmem_alloc(&bases[0], 256);
        }
        else if (mode == 1)
        {
            tmem_alloc(&bases[0], 256);
            tmem_alloc(&bases[1], 128);
        }
        else if (mode == 2)
        {
            tmem_alloc(&bases[0], 128);
            tmem_alloc(&bases[1], 256);
        }
        tmem_relinquish();
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        result[0] = bases[0];
        result[1] = bases[1];
    }
    __syncthreads();
    if (warp == 0)
    {
        if (mode == 0)
        {
            tmem_dealloc(bases[0], 256);
        }
        else if (mode == 1)
        {
            tmem_dealloc(bases[1], 128);
            tmem_dealloc(bases[0], 256);
        }
        else if (mode == 2)
        {
            tmem_dealloc(bases[1], 256);
            tmem_dealloc(bases[0], 128);
        }
    }
}

int main(int argc, char** argv)
{
    int const mode = argc > 1 ? std::atoi(argv[1]) : 0;
    uint32_t* result;
    cudaMallocManaged(&result, 2 * sizeof(uint32_t));
    result[0] = result[1] = 0xdeadbeef;
    allocation_probe<<<1, 256>>>(mode, result);
    cudaError_t status = cudaDeviceSynchronize();
    std::printf("mode=%d status=%s base0=%u base1=%u\n", mode, cudaGetErrorString(status), result[0], result[1]);
    return status == cudaSuccess ? 0 : 1;
}
