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

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{

CUgreenCtx gContexts[2]{};
CUstream gStreams[2]{};
bool gInitialized = false;
char gLastError[512]{};

int fail(CUresult status, char const* operation)
{
    char const* name = nullptr;
    char const* message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    std::snprintf(gLastError, sizeof(gLastError), "%s failed: %s (%s)", operation, name ? name : "unknown",
        message ? message : "unknown");
    return static_cast<int>(status);
}

#define GC_CHECK(call)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        const CUresult status = (call);                                                                                \
        if (status != CUDA_SUCCESS)                                                                                    \
        {                                                                                                              \
            return fail(status, #call);                                                                                \
        }                                                                                                              \
    } while (0)

void destroyPartial()
{
    for (int i = 0; i < 2; ++i)
    {
        if (gStreams[i] != nullptr)
        {
            cuStreamDestroy(gStreams[i]);
            gStreams[i] = nullptr;
        }
        if (gContexts[i] != nullptr)
        {
            cuGreenCtxDestroy(gContexts[i]);
            gContexts[i] = nullptr;
        }
    }
    gInitialized = false;
}

} // namespace

extern "C" char const* gcLastError()
{
    return gLastError;
}

extern "C" int gcCreatePair(
    int deviceOrdinal, std::uint64_t* streamHandles, unsigned int* smCounts, std::uint64_t* contextIds)
{
    if (streamHandles == nullptr || smCounts == nullptr || contextIds == nullptr)
    {
        std::snprintf(gLastError, sizeof(gLastError), "output pointer was null");
        return -1;
    }
    if (gInitialized)
    {
        std::snprintf(gLastError, sizeof(gLastError), "green-context pair already initialized");
        return -2;
    }

    std::memset(gLastError, 0, sizeof(gLastError));
    GC_CHECK(cuInit(0));

    CUdevice device{};
    GC_CHECK(cuDeviceGet(&device, deviceOrdinal));

    CUdevResource allSms{};
    GC_CHECK(cuDeviceGetDevResource(device, &allSms, CU_DEV_RESOURCE_TYPE_SM));
    if (allSms.type != CU_DEV_RESOURCE_TYPE_SM || allSms.sm.smCoscheduledAlignment == 0)
    {
        std::snprintf(gLastError, sizeof(gLastError), "device returned an invalid SM resource");
        return -3;
    }

    unsigned int const alignment = allSms.sm.smCoscheduledAlignment;
    unsigned int const requestedPerGroup = (allSms.sm.smCount / 2 / alignment) * alignment;
    CUdevResource groups[2]{};
    CUdevResource remainder{};
    unsigned int groupCount = 2;
    const CUresult splitStatus
        = cuDevSmResourceSplitByCount(groups, &groupCount, &allSms, &remainder, 0, requestedPerGroup);
    if (splitStatus != CUDA_SUCCESS)
    {
        return fail(splitStatus, "cuDevSmResourceSplitByCount");
    }
    if (groupCount != 2)
    {
        std::snprintf(gLastError, sizeof(gLastError), "requested two groups but driver returned %u", groupCount);
        return -4;
    }

    for (int i = 0; i < 2; ++i)
    {
        CUdevResourceDesc descriptor{};
        CUresult status = cuDevResourceGenerateDesc(&descriptor, &groups[i], 1);
        if (status != CUDA_SUCCESS)
        {
            destroyPartial();
            return fail(status, "cuDevResourceGenerateDesc");
        }
        // Required by the CUDA 13.2 driver API used in the test container.
        status = cuGreenCtxCreate(&gContexts[i], descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM);
        if (status != CUDA_SUCCESS)
        {
            destroyPartial();
            return fail(status, "cuGreenCtxCreate");
        }
        status = cuGreenCtxStreamCreate(&gStreams[i], gContexts[i], CU_STREAM_NON_BLOCKING, 0);
        if (status != CUDA_SUCCESS)
        {
            destroyPartial();
            return fail(status, "cuGreenCtxStreamCreate");
        }

        unsigned long long contextId{};
        status = cuGreenCtxGetId(gContexts[i], &contextId);
        if (status != CUDA_SUCCESS)
        {
            destroyPartial();
            return fail(status, "cuGreenCtxGetId");
        }
        streamHandles[i] = reinterpret_cast<std::uint64_t>(gStreams[i]);
        smCounts[i] = groups[i].sm.smCount;
        contextIds[i] = contextId;
    }

    smCounts[2] = allSms.sm.smCount;
    smCounts[3] = allSms.sm.minSmPartitionSize;
    smCounts[4] = alignment;
    smCounts[5] = remainder.type == CU_DEV_RESOURCE_TYPE_SM ? remainder.sm.smCount : 0;

    if (contextIds[0] == contextIds[1])
    {
        destroyPartial();
        std::snprintf(gLastError, sizeof(gLastError), "separately created green contexts have the same id");
        return -5;
    }
    gInitialized = true;
    return 0;
}

extern "C" int gcQueryStream(std::uint64_t streamHandle, unsigned int* smCount, std::uint64_t* contextId)
{
    if (smCount == nullptr || contextId == nullptr)
    {
        std::snprintf(gLastError, sizeof(gLastError), "query output pointer was null");
        return -1;
    }
    CUstream stream = reinterpret_cast<CUstream>(streamHandle);
    CUdevResource streamSms{};
    GC_CHECK(cuStreamGetDevResource(stream, &streamSms, CU_DEV_RESOURCE_TYPE_SM));
    CUgreenCtx context{};
    GC_CHECK(cuStreamGetGreenCtx(stream, &context));
    unsigned long long id{};
    GC_CHECK(cuGreenCtxGetId(context, &id));
    *smCount = streamSms.sm.smCount;
    *contextId = id;
    return 0;
}

extern "C" int gcDestroyPair()
{
    if (!gInitialized)
    {
        return 0;
    }
    for (int i = 0; i < 2; ++i)
    {
        if (gStreams[i] != nullptr)
        {
            const CUresult status = cuStreamDestroy(gStreams[i]);
            if (status != CUDA_SUCCESS)
            {
                return fail(status, "cuStreamDestroy");
            }
            gStreams[i] = nullptr;
        }
        if (gContexts[i] != nullptr)
        {
            const CUresult status = cuGreenCtxDestroy(gContexts[i]);
            if (status != CUDA_SUCCESS)
            {
                return fail(status, "cuGreenCtxDestroy");
            }
            gContexts[i] = nullptr;
        }
    }
    gInitialized = false;
    return 0;
}
