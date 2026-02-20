"""Simple GPU benchmark to check RTX 5090 performance."""
import torch
import time

device = torch.device('cuda')
print(f'Device: {torch.cuda.get_device_name(0)}')

# Simple matmul benchmark
x = torch.randn(64, 384, 384, device=device)
y = torch.randn(64, 384, 384, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    z = torch.bmm(x, y)
torch.cuda.synchronize()
print(f'100x bmm (64,384,384): {time.time()-t0:.3f}s')

# Conv2d benchmark
conv = torch.nn.Conv2d(3, 64, 16, stride=16).cuda()
img = torch.randn(128, 3, 256, 256, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    out = conv(img)
torch.cuda.synchronize()
print(f'100x Conv2d 128x3x256x256: {time.time()-t0:.3f}s, out={out.shape}')

# Multi-head attention
mha = torch.nn.MultiheadAttention(384, 8, batch_first=True).cuda()
x = torch.randn(128, 273, 384, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    out, _ = mha(x, x, x)
torch.cuda.synchronize()
print(f'10x MHA(384,8) on (128,273,384): {time.time()-t0:.3f}s')

print('BENCHMARK_DONE')
