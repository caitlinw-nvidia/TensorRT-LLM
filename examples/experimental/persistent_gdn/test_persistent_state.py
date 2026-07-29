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

import os
import struct
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent


def build():
    max_regs = int(os.environ.get("GDN_MAX_REGS", "160"))
    return load(
        name=f"persistent_gdn_state_r{max_regs}",
        sources=[str(ROOT / "persistent_state.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-arch=sm_103a",
            f"-maxrregcount={max_regs}",
            "--ptxas-options=-v",
            f"-DGDN_MAX_REGS={max_regs}",
        ],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )


def main():
    torch.cuda.set_device(0)
    extension = build()
    initial = torch.arange(24 * 4096 * 128, device="cuda", dtype=torch.float32).reshape(
        24, 4096, 128
    )
    expected = initial.clone()
    for row_base in range(0, 4096, 32):
        expected[0, row_base : min(row_base + 12, 4096)].add_(1.0 / 1024.0)
        expected[0, row_base + 12 : min(row_base + 32, 4096)].add_(1.0 / 1024.0)
    for iteration in range(3):
        for layer in range(24):
            expected[layer].add_((iteration + 1) * (layer + 1) / 1024.0)

    control = extension.start(initial)
    deadline = time.monotonic() + 30
    while True:
        # The service stream is non-blocking, so this tiny D2H copy observes
        # readiness without waiting for the intentionally long-lived kernel.
        snapshot = bytes(control.cpu().tolist())
        ready_blocks = struct.unpack_from("<I", snapshot, 8)[0]
        if ready_blocks == 128:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"resident service initialized only {ready_blocks}/128 CTAs")
        time.sleep(0.01)
    print("READY: 128 resident CTAs initialized RF/TMEM", flush=True)

    def wait_debug_epoch(label):
        deadline = time.monotonic() + 5
        last = None
        while True:
            snapshot = bytes(control.cpu().tolist())
            epoch, done, ready, arrived = struct.unpack_from("<IIII", snapshot)
            last = (epoch, done, ready, arrived)
            if done >= epoch:
                print(
                    f"COMMAND: {label} complete (epoch={epoch}, arrived={arrived})",
                    flush=True,
                )
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{label} stalled: epoch/done/ready/arrived={last}")
            time.sleep(0.01)

    extension.debug_command(control, 1, 0, 0.0)
    wait_debug_epoch("no-op")
    extension.debug_command(control, 2, 0, 1.0 / 1024.0)
    wait_debug_epoch("RF-only update")
    extension.debug_command(control, 3, 0, 1.0 / 1024.0)
    wait_debug_epoch("TMEM-only update")

    for iteration in range(3):
        for layer in range(24):
            delta = (iteration + 1) * (layer + 1) / 1024.0
            extension.add_layer(control, layer, delta)
    final = extension.stop(control)

    torch.testing.assert_close(final, expected, rtol=0, atol=0)
    print("PASS: 72 commands retained [24,4096,128] FP32 state in RF/TMEM")
    print(f"max_abs={torch.max(torch.abs(final - expected)).item():.9g}")


if __name__ == "__main__":
    main()
