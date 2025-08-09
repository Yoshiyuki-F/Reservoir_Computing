#!/usr/bin/env python3
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/targets/x86_64-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import jax
import jax.numpy as jnp

print('JAX Version:', jax.__version__)
print('Available devices:', jax.devices())

devices = jax.devices()
if len(devices) > 1 or (len(devices) > 0 and devices[0].device_kind != 'cpu'):
    print(' GPU detected!')
    gpu_available = True
else:
    print(' GPU not available, using CPU')
    gpu_available = False

# 基本計算テスト
x = jnp.array([1.0, 2.0, 3.0])
result = jnp.sum(x**2) 
print('計算結果:', float(result))

if gpu_available:
    print('GPU上で計算実行中...')
else:
    print('CPU上で計算実行中...')