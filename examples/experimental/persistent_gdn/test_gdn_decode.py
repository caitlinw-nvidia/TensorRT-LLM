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

import struct
import time

import torch
from test_persistent_state import build

LAYERS = 24
HV = 32
H = 16
V = 128
K = 128


def reference_step(state, q, k, v, a, b, a_log, dt_bias):
    qf = q.float()
    kf = k.float()
    qf = qf / (torch.linalg.vector_norm(qf, dim=-1, keepdim=True) + 1.0e-6)
    kf = kf / (torch.linalg.vector_norm(kf, dim=-1, keepdim=True) + 1.0e-6)
    qf = qf.repeat_interleave(2, dim=0)
    kf = kf.repeat_interleave(2, dim=0)

    x = a.float() + dt_bias
    softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    decay = torch.exp(-torch.exp(a_log) * softplus)
    beta = torch.sigmoid(b.float())

    state.mul_(decay[:, None, None])
    prediction = torch.sum(state * kf[:, None, :], dim=-1)
    v_new = (v.float() - prediction) * beta[:, None]
    state.add_(kf[:, None, :] * v_new[:, :, None])
    output = torch.sum(state * qf[:, None, :], dim=-1) * (K**-0.5)
    return output.to(torch.bfloat16)


def wait_debug_epoch(control):
    deadline = time.monotonic() + 5
    while True:
        snapshot = bytes(control.cpu().tolist())
        epoch, done, ready, arrived = struct.unpack_from("<IIII", snapshot)
        if done >= epoch:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"resident warmup stalled: epoch/done/ready/arrived={(epoch, done, ready, arrived)}"
            )
        time.sleep(0.01)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(19)
    extension = build()

    initial = torch.randn(LAYERS, HV, V, K, device="cuda", dtype=torch.float32) * 0.02
    reference_state = initial.clone()
    commands = []
    expected_outputs = []

    a_logs = [
        torch.empty(HV, device="cuda", dtype=torch.float32).uniform_(-4.0, -2.0)
        for _ in range(LAYERS)
    ]
    dt_biases = [
        torch.empty(HV, device="cuda", dtype=torch.float32).uniform_(-0.5, 0.5)
        for _ in range(LAYERS)
    ]

    for token in range(2):
        for layer in range(LAYERS):
            q = torch.randn(H, K, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(H, K, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(HV, V, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(HV, device="cuda", dtype=torch.bfloat16)
            command = (layer, q, k, v, a, b, a_logs[layer], dt_biases[layer])
            commands.append(command)
            expected_outputs.append(reference_step(reference_state[layer], *command[1:]))

    control = extension.start(initial.view(LAYERS, HV * V, K))
    deadline = time.monotonic() + 30
    while True:
        snapshot = bytes(control.cpu().tolist())
        ready = struct.unpack_from("<I", snapshot, 8)[0]
        if ready == 128:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only {ready}/128 resident CTAs initialized")
        time.sleep(0.01)

    # Prime the command/status path without consuming an SM-side wait CTA.
    extension.debug_command(control, 1, 0, 0.0)
    wait_debug_epoch(control)

    actual_outputs = []
    for command in commands:
        actual_outputs.append(extension.gdn_decode(control, *command))
    final_state = extension.stop(control).view(LAYERS, HV, V, K)

    actual = torch.stack(actual_outputs).float()
    expected = torch.stack(expected_outputs).float()
    output_error = torch.abs(actual - expected)
    state_error = torch.abs(final_state - reference_state)
    print(
        f"output error: max={output_error.max().item():.7g} mean={output_error.mean().item():.7g}"
    )
    print(f"state error: max={state_error.max().item():.7g} mean={state_error.mean().item():.7g}")
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(final_state, reference_state, atol=2.0e-5, rtol=2.0e-5)
    print("PASS: 48 row-sharded resident GDN decode commands match reference")


if __name__ == "__main__":
    main()
