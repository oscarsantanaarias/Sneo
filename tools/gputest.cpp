#include "../src/cuda.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

const char* testSource = R"(
extern "C" __global__ void rowDot(const float4* weights, const float4* input, float* output, int rows, int quads) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if(row >= rows) return;
    const float4* line = weights + (size_t)row * quads;
    float sum = 0.0f;
    for(int i = threadIdx.x; i < quads; i += 32){
        float4 w = line[i];
        float4 x = input[i];
        sum += w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w;
    }
    for(int step = 16; step > 0; step >>= 1) sum += __shfl_down_sync(0xffffffff, sum, step);
    if(threadIdx.x == 0) output[row] = sum;
}
)";

int main() {
    auto started = std::chrono::steady_clock::now();
    KernelModule module;
    if(!buildModule(testSource, module)){
        std::cout << "build failed: " << module.log << std::endl;
        return 1;
    }
    double buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    CudaLink& cu = cudaLink();
    std::cout << "kernel built for sm_" << cu.major << cu.minor << " in " << buildMs << " ms" << std::endl;

    int rows = 8192, cols = 8192;
    std::vector<float> weights((size_t)rows * cols), input(cols), output(rows), expected(rows);
    std::mt19937 random(7);
    std::uniform_real_distribution<float> spread(-1.0f, 1.0f);
    for(float& v : weights) v = spread(random);
    for(float& v : input) v = spread(random);

    uint64_t deviceWeights = 0, deviceInput = 0, deviceOutput = 0;
    if(cu.alloc(&deviceWeights, weights.size() * 4) != 0 || cu.alloc(&deviceInput, input.size() * 4) != 0 || cu.alloc(&deviceOutput, output.size() * 4) != 0){
        std::cout << "alloc failed" << std::endl;
        return 1;
    }
    cu.copyUp(deviceWeights, weights.data(), weights.size() * 4);
    cu.copyUp(deviceInput, input.data(), input.size() * 4);

    void* fn = kernelFunction(module, "rowDot");
    int quads = cols / 4;
    void* args[] = {&deviceWeights, &deviceInput, &deviceOutput, &rows, &quads};

    launchKernel(fn, (rows + 7) / 8, 1, 32, 8, 0, args);
    cu.sync();

    int runs = 50;
    auto t0 = std::chrono::steady_clock::now();
    for(int i = 0; i < runs; i++) launchKernel(fn, (rows + 7) / 8, 1, 32, 8, 0, args);
    cu.sync();
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / runs;

    cu.copyDown(output.data(), deviceOutput, output.size() * 4);

    double worst = 0;
    for(int r = 0; r < rows; r += 97){
        double sum = 0;
        for(int c = 0; c < cols; c++) sum += (double)weights[(size_t)r * cols + c] * input[c];
        worst = std::max(worst, std::fabs(sum - output[r]));
    }

    std::cout << "matvec " << rows << "x" << cols << ": " << seconds * 1000.0 << " ms, " << (double)weights.size() * 4 / seconds / 1e9 << " GB/s, max error " << worst << std::endl;
    cu.release(deviceWeights);
    cu.release(deviceInput);
    cu.release(deviceOutput);
    return worst < 1e-2 ? 0 : 2;
}
