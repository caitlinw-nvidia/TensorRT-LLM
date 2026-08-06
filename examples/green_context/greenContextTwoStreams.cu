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

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define DRIVER_CHECK(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult status_ = (call);                                                                                     \
        if (status_ != CUDA_SUCCESS)                                                                                   \
        {                                                                                                              \
            const char* name_ = nullptr;                                                                               \
            const char* message_ = nullptr;                                                                            \
            cuGetErrorName(status_, &name_);                                                                           \
            cuGetErrorString(status_, &message_);                                                                      \
            std::cerr << #call << " failed: " << (name_ ? name_ : "unknown") << " ("                                   \
                      << (message_ ? message_ : "unknown") << ")\n";                                                   \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

#define RUNTIME_CHECK(call)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status_ = (call);                                                                                  \
        if (status_ != cudaSuccess)                                                                                    \
        {                                                                                                              \
            std::cerr << #call << " failed: " << cudaGetErrorName(status_) << " (" << cudaGetErrorString(status_)      \
                      << ")\n";                                                                                        \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

__global__ void fillAndSpin(int* output, int count, int value, unsigned long long cycles)
{
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long const start = clock64();
    while (clock64() - start < cycles)
    {
    }
    if (index < count)
    {
        output[index] = value;
    }
}

int main()
{
    constexpr int kDeviceOrdinal = 0;
    constexpr int kElements = 1 << 20;
    constexpr int kThreads = 256;
    constexpr unsigned long long kSpinCycles = 100000;

    // cudaSetDevice initializes and makes the device primary context current.
    // Green contexts are derived from device resources; cuCtxCreate is not used.
    RUNTIME_CHECK(cudaSetDevice(kDeviceOrdinal));
    RUNTIME_CHECK(cudaFree(nullptr));
    DRIVER_CHECK(cuInit(0));

    CUdevice device{};
    DRIVER_CHECK(cuDeviceGet(&device, kDeviceOrdinal));

    char deviceName[256]{};
    DRIVER_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), device));

    CUdevResource allSms{};
    DRIVER_CHECK(cuDeviceGetDevResource(device, &allSms, CU_DEV_RESOURCE_TYPE_SM));
    if (allSms.type != CU_DEV_RESOURCE_TYPE_SM)
    {
        std::cerr << "Device returned a non-SM resource\n";
        return EXIT_FAILURE;
    }

    unsigned int const alignment = allSms.sm.smCoscheduledAlignment;
    if (alignment == 0)
    {
        std::cerr << "Invalid zero SM co-scheduling alignment\n";
        return EXIT_FAILURE;
    }
    unsigned int const requestedPerGroup = (allSms.sm.smCount / 2 / alignment) * alignment;

    CUdevResource groups[2]{};
    CUdevResource remainder{};
    unsigned int groupCount = 2;
    DRIVER_CHECK(cuDevSmResourceSplitByCount(groups, &groupCount, &allSms, &remainder, 0, requestedPerGroup));
    if (groupCount != 2)
    {
        std::cerr << "Requested two SM groups, driver returned " << groupCount << "\n";
        return EXIT_FAILURE;
    }

    CUdevResourceDesc descriptors[2]{};
    CUgreenCtx greenContexts[2]{};
    CUstream streams[2]{};
    for (int i = 0; i < 2; ++i)
    {
        DRIVER_CHECK(cuDevResourceGenerateDesc(&descriptors[i], &groups[i], 1));
        // CUDA 13.2 requires CU_GREEN_CTX_DEFAULT_STREAM here. Newer toolkits also
        // expose CU_GREEN_CTX_NONE, but using DEFAULT_STREAM is compatible with
        // the deployment tested by this prototype.
        DRIVER_CHECK(cuGreenCtxCreate(&greenContexts[i], descriptors[i], device, CU_GREEN_CTX_DEFAULT_STREAM));
        DRIVER_CHECK(cuGreenCtxStreamCreate(&streams[i], greenContexts[i], CU_STREAM_NON_BLOCKING, 0));
    }

    std::cout << "device=" << deviceName << "\n"
              << "all_sms=" << allSms.sm.smCount << "\n"
              << "min_partition=" << allSms.sm.minSmPartitionSize << "\n"
              << "coscheduled_alignment=" << alignment << "\n"
              << "group0_sms=" << groups[0].sm.smCount << "\n"
              << "group1_sms=" << groups[1].sm.smCount << "\n"
              << "remainder_type=" << static_cast<int>(remainder.type) << "\n"
              << "remainder_sms=" << (remainder.type == CU_DEV_RESOURCE_TYPE_SM ? remainder.sm.smCount : 0) << "\n";

    for (int i = 0; i < 2; ++i)
    {
        CUdevResource streamSms{};
        CUgreenCtx queriedContext{};
        DRIVER_CHECK(cuStreamGetDevResource(streams[i], &streamSms, CU_DEV_RESOURCE_TYPE_SM));
        DRIVER_CHECK(cuStreamGetGreenCtx(streams[i], &queriedContext));
        if (queriedContext != greenContexts[i] || streamSms.sm.smCount != groups[i].sm.smCount)
        {
            std::cerr << "Stream " << i << " is not associated with the expected green context/resource\n";
            return EXIT_FAILURE;
        }
        std::cout << "stream" << i << "_sms=" << streamSms.sm.smCount << "\n";
    }

    int* deviceOutputs[2]{};
    for (auto& output : deviceOutputs)
    {
        RUNTIME_CHECK(cudaMalloc(&output, kElements * sizeof(int)));
    }

    int const blocks = (kElements + kThreads - 1) / kThreads;
    fillAndSpin<<<blocks, kThreads, 0, reinterpret_cast<cudaStream_t>(streams[0])>>>(
        deviceOutputs[0], kElements, 17, kSpinCycles);
    RUNTIME_CHECK(cudaGetLastError());
    fillAndSpin<<<blocks, kThreads, 0, reinterpret_cast<cudaStream_t>(streams[1])>>>(
        deviceOutputs[1], kElements, 29, kSpinCycles);
    RUNTIME_CHECK(cudaGetLastError());

    // This is the host-side join for the minimal prototype.
    DRIVER_CHECK(cuStreamSynchronize(streams[0]));
    DRIVER_CHECK(cuStreamSynchronize(streams[1]));

    std::vector<int> hostOutput(kElements);
    RUNTIME_CHECK(cudaMemcpy(hostOutput.data(), deviceOutputs[0], kElements * sizeof(int), cudaMemcpyDeviceToHost));
    for (int value : hostOutput)
    {
        if (value != 17)
        {
            std::cerr << "Validation failed for stream 0\n";
            return EXIT_FAILURE;
        }
    }
    RUNTIME_CHECK(cudaMemcpy(hostOutput.data(), deviceOutputs[1], kElements * sizeof(int), cudaMemcpyDeviceToHost));
    for (int value : hostOutput)
    {
        if (value != 29)
        {
            std::cerr << "Validation failed for stream 1\n";
            return EXIT_FAILURE;
        }
    }

    for (auto* output : deviceOutputs)
    {
        RUNTIME_CHECK(cudaFree(output));
    }
    for (int i = 0; i < 2; ++i)
    {
        DRIVER_CHECK(cuStreamDestroy(streams[i]));
        DRIVER_CHECK(cuGreenCtxDestroy(greenContexts[i]));
    }

    std::cout << "validation=PASS\n";
    return EXIT_SUCCESS;
}
