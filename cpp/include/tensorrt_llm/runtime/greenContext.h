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

#pragma once

#include <cuda.h>

#include <cstdint>

namespace tensorrt_llm::runtime
{

//! Owns a CUDA green context and one non-blocking stream associated with it.
//!
//! The stream pointer can be wrapped by torch.cuda.ExternalStream. The owner
//! must outlive every external wrapper and all work submitted to the stream.
class GreenContext
{
public:
    GreenContext(int device, unsigned int smCount);
    ~GreenContext() noexcept;

    GreenContext(GreenContext const&) = delete;
    GreenContext& operator=(GreenContext const&) = delete;
    GreenContext(GreenContext&&) = delete;
    GreenContext& operator=(GreenContext&&) = delete;

    [[nodiscard]] std::uintptr_t getStreamPtr() const noexcept;
    [[nodiscard]] int getDevice() const noexcept;
    [[nodiscard]] unsigned int getRequestedSmCount() const noexcept;
    [[nodiscard]] unsigned int getAllocatedSmCount() const noexcept;

    void synchronize() const;

private:
    void cleanup() noexcept;

    CUdevice mDevice{};
    CUgreenCtx mContext{};
    CUstream mStream{};
    unsigned int mRequestedSmCount{};
    unsigned int mAllocatedSmCount{};
};

} // namespace tensorrt_llm::runtime
