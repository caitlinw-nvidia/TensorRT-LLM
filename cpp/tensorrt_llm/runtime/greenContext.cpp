/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/greenContext.h"

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/logger.h"

#include <stdexcept>
#include <string>

namespace tensorrt_llm::runtime
{
namespace
{

void logCleanupFailure(CUresult status, char const* operation) noexcept
{
    if (status == CUDA_SUCCESS || status == CUDA_ERROR_DEINITIALIZED)
    {
        return;
    }
    TLLM_LOG_WARNING("CUDA driver cleanup failed in %s with status %d", operation, static_cast<int>(status));
}

} // namespace

GreenContext::GreenContext(int device, unsigned int smCount)
    : mRequestedSmCount{smCount}
{
    if (smCount == 0)
    {
        throw std::invalid_argument("GreenContext smCount must be positive");
    }

    TLLM_CU_CHECK(cuInit(0));
    TLLM_CU_CHECK(cuDeviceGet(&mDevice, device));

    CUdevResource deviceSmResource{};
    TLLM_CU_CHECK(cuDeviceGetDevResource(mDevice, &deviceSmResource, CU_DEV_RESOURCE_TYPE_SM));
    if (smCount > deviceSmResource.sm.smCount)
    {
        throw std::invalid_argument("GreenContext requested " + std::to_string(smCount) + " SMs, but device "
            + std::to_string(device) + " exposes only " + std::to_string(deviceSmResource.sm.smCount));
    }

    CUdevResource selectedSmResource{};
    unsigned int numGroups{1};
    TLLM_CU_CHECK(cuDevSmResourceSplitByCount(
        &selectedSmResource, &numGroups, &deviceSmResource, nullptr, 0, smCount));
    if (numGroups != 1 || selectedSmResource.type != CU_DEV_RESOURCE_TYPE_SM)
    {
        throw std::runtime_error("CUDA did not produce the requested green-context SM partition");
    }
    mAllocatedSmCount = selectedSmResource.sm.smCount;

    CUdevResourceDesc resourceDesc{};
    TLLM_CU_CHECK(cuDevResourceGenerateDesc(&resourceDesc, &selectedSmResource, 1));
    TLLM_CU_CHECK(cuGreenCtxCreate(&mContext, resourceDesc, mDevice, CU_GREEN_CTX_DEFAULT_STREAM));

    try
    {
        TLLM_CU_CHECK(cuGreenCtxStreamCreate(&mStream, mContext, CU_STREAM_NON_BLOCKING, 0));
    }
    catch (...)
    {
        cleanup();
        throw;
    }

    TLLM_LOG_INFO("Created CUDA green context on device %d with %u requested SMs, %u allocated SMs, and stream %p",
        device, mRequestedSmCount, mAllocatedSmCount, static_cast<void*>(mStream));
}

GreenContext::~GreenContext() noexcept
{
    cleanup();
}

std::uintptr_t GreenContext::getStreamPtr() const noexcept
{
    return reinterpret_cast<std::uintptr_t>(mStream);
}

int GreenContext::getDevice() const noexcept
{
    return static_cast<int>(mDevice);
}

unsigned int GreenContext::getRequestedSmCount() const noexcept
{
    return mRequestedSmCount;
}

unsigned int GreenContext::getAllocatedSmCount() const noexcept
{
    return mAllocatedSmCount;
}

void GreenContext::synchronize() const
{
    TLLM_CU_CHECK(cuStreamSynchronize(mStream));
}

void GreenContext::cleanup() noexcept
{
    if (mStream != nullptr)
    {
        logCleanupFailure(cuStreamSynchronize(mStream), "cuStreamSynchronize");
        logCleanupFailure(cuStreamDestroy(mStream), "cuStreamDestroy");
        mStream = nullptr;
    }
    if (mContext != nullptr)
    {
        logCleanupFailure(cuGreenCtxDestroy(mContext), "cuGreenCtxDestroy");
        mContext = nullptr;
    }
}

} // namespace tensorrt_llm::runtime
