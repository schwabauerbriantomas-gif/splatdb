#!/usr/bin/env python3
"""Techo GPU para búsqueda exacta 26k×768: torch matmul+topk en la 3090."""
import json
import struct
import time

import numpy as np
import torch

HUB = "/mnt/d/splatdb/semantic-hub"
with open(f"{HUB}/data/vectors/default.bin", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    d = struct.unpack("<Q", f.read(8))[0]
    X = torch.from_numpy(
        np.frombuffer(f.read(n * d * 4), dtype="float32").reshape(n, d).copy())
Q = torch.tensor([q["vec"] for q in json.load(open(f"{HUB}/bench-latent/queries.json"))])
print(f"base {tuple(X.shape)} queries {tuple(Q.shape)} | GPU: {torch.cuda.get_device_name(0)}")

t0 = time.monotonic()
_ = X @ Q.T
cpu = (time.monotonic() - t0) * 1000

Xg, Qg = X.cuda(), Q.cuda()
torch.cuda.synchronize()
for _ in range(3):
    _ = Xg @ Qg.T
torch.cuda.synchronize()
t0 = time.monotonic()
D = Xg @ Qg.T
torch.cuda.synchronize()
gpu = (time.monotonic() - t0) * 1000
t0 = time.monotonic()
vals, idx = torch.topk(D, 10, dim=0, largest=False)
torch.cuda.synchronize()
gpu_topk = (time.monotonic() - t0) * 1000

gb = X.numel() * 4 / 1e9
print(f"dist 30 queries | CPU torch: {cpu:.1f} ms ({cpu/30:.2f} ms/q)")
print(f"GPU matmul: {gpu:.1f} ms total ({gpu/30*1000:.0f} us/q) | +topk10: {gpu_topk:.1f} ms ({gpu_topk/30*1000:.0f} us/q exacto)")
print(f"ancho de banda efectivo: {gb:.2f} GB en {gpu/30:.3f} ms -> {gb/(gpu/30/1000):.0f} GB/s")
