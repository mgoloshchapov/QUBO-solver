import torch
from torch.utils.cpp_extension import load_inline

cpp_source = r'''
#include <torch/extension.h>

extern "C" void launch_metropolis_step(torch::Tensor states, torch::Tensor Q, torch::Tensor beta,
                                        int W, int L, int K, int N, unsigned long long seed);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_metropolis_step", &launch_metropolis_step, "Launch metropolis CUDA kernel");
}
'''

cuda_source = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

extern "C" __global__ void metropolis_step_kernel(
    float* states, const float* Q, const float* beta,
    int W, int L, int K, int N, unsigned long long seed)
{
    int w = blockIdx.x, l = blockIdx.y, k = threadIdx.x;
    int idx = ((w * L + l) * K + k);
    curandState localState;
    curand_init(seed, idx, 0, &localState);
    
    float* state = states + idx * N;
    const float* Q_w = Q + w * N * N;
    float beta_l = beta[l];
    
    for (int i = 0; i < N; i++) {
         float s = state[i];
         float x = 1.0f - 2.0f * s;
         float h = 0.0f;
         
         for (int j = 0; j < N; j++) {
              h = state[j] * Q_w[i * N + j] + h;
         }
         
         float dE = 2.0f * (h * x) + Q_w[i * N + i];
         dE = (dE < 0.0f) ? 0.0f : dE;
         
         float prob = expf(-beta_l * dE);
         if (curand_uniform(&localState) < prob)
             state[i] = s + x;
    }
}

extern "C" void launch_metropolis_step(torch::Tensor states, torch::Tensor Q, torch::Tensor beta,
                                        int W, int L, int K, int N, unsigned long long seed) {
    float* states_ptr = reinterpret_cast<float*>(states.data_ptr());
    const float* Q_ptr = reinterpret_cast<const float*>(Q.data_ptr());
    const float* beta_ptr = reinterpret_cast<const float*>(beta.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(W, L, 1), block(K, 1, 1);
    metropolis_step_kernel<<<grid, block, 0, stream>>>(states_ptr, Q_ptr, beta_ptr, W, L, K, N, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
         throw std::runtime_error(cudaGetErrorString(err));
}
'''

module = load_inline(name='metropolis_cuda',
                     cpp_sources=cpp_source,
                     cuda_sources=cuda_source,
                     verbose=False)

# Экспорт функции под именем metropolis_step
metropolis_step = module.launch_metropolis_step