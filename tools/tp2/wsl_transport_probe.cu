// WSL/no-P2P transport qualification and timing, independent of model weights.
// Build: nvcc -O2 -arch=sm_70 tools/tp2/wsl_transport_probe.cu -o wsl_transport_probe
// Run without other GPU work. Every route verifies changing payloads on both
// devices, eagerly and over repeated cross-device graph replays. Timings are
// collective microbenchmarks, not model throughput predictions.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(call) do { auto e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

__device__ unsigned pattern(int i, unsigned salt, int rank) {
    return ((i * 2654435761U) ^ (salt * 1664525U) ^ (rank ? 0x12483aebU : 0x790bfe12U)) & 0x7fffffffU;
}
__global__ void produce(unsigned* data, int count, unsigned salt, int rank) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        data[i] = pattern(i, salt, rank);
}
__global__ void publish(const unsigned* data, volatile unsigned* host, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        host[i] = data[i];
    __threadfence_system();
}
__global__ void verify(const volatile unsigned* data, unsigned* errors,
                       int count, unsigned salt, int rank) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        if (data[i] != pattern(i, salt, 1 - rank)) atomicAdd(errors, 1U);
}
__global__ void packet_verify(const unsigned* source, volatile unsigned* local,
    volatile unsigned* peer, unsigned* epoch_state, unsigned* errors, int count, unsigned salt, int rank) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned epoch = epoch_state[tid] & 1U;
    for (int i = tid; i < count; i += gridDim.x * blockDim.x) {
        const int offset = epoch * count + i;
        peer[offset] = source[i];
        __threadfence_system();
        const auto start = clock64();
        unsigned value;
        while ((value = local[offset]) == 0xffffffffU) {
            if (clock64() - start > 1000000000ULL) { atomicAdd(errors, 1U); return; }
        }
        if (value != pattern(i, salt, 1-rank)) atomicAdd(errors, 1U);
        local[offset] = 0xffffffffU;
        __threadfence_system();
    }
    epoch_state[tid] = epoch ^ 1U;
}

struct Pair {
    cudaStream_t stream[2]{};
    cudaEvent_t ready[2]{}, done[2]{}, fork{}, join{};
    unsigned *source[2]{}, *staging[2]{}, *errors[2]{}, *epochs[2]{};
    void* host[2]{};
    void* mapped[2][2]{}; // consuming device, publishing rank
    int count;
    explicit Pair(int n) : count(n) {
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(r));
            CK(cudaStreamCreateWithFlags(&stream[r], cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&ready[r], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&done[r], cudaEventDisableTiming));
            CK(cudaMalloc(&source[r], n * sizeof(unsigned)));
            CK(cudaMalloc(&staging[r], n * sizeof(unsigned)));
            CK(cudaMalloc(&errors[r], sizeof(unsigned)));
            CK(cudaMemset(errors[r], 0, sizeof(unsigned)));
            CK(cudaMalloc(&epochs[r], 2048 * sizeof(unsigned)));
            CK(cudaMemset(epochs[r], 0, 2048 * sizeof(unsigned)));
            CK(cudaMallocHost(&host[r], 2 * n * sizeof(unsigned)));
            std::memset(host[r], 0xff, 2 * n * sizeof(unsigned));
        }
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(r));
            for (int p = 0; p < 2; ++p) CK(cudaHostGetDevicePointer(&mapped[r][p], host[p], 0));
        }
        CK(cudaSetDevice(0)); CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        CK(cudaSetDevice(1)); CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    }
    ~Pair() {
        for (int r = 0; r < 2; ++r) {
            cudaSetDevice(r); cudaStreamSynchronize(stream[r]);
            cudaFree(source[r]); cudaFree(staging[r]); cudaFree(errors[r]);
            cudaFree(epochs[r]);
            cudaFreeHost(host[r]); cudaEventDestroy(ready[r]); cudaEventDestroy(done[r]);
            cudaStreamDestroy(stream[r]);
        }
        cudaEventDestroy(fork); cudaEventDestroy(join);
    }
    void issue(int mode, unsigned salt) {
        const size_t bytes = count * sizeof(unsigned);
        if (mode == 3) {
            for (int r = 0; r < 2; ++r) {
                CK(cudaSetDevice(r));
                produce<<<16,128,0,stream[r]>>>(source[r], count, salt, r);
                packet_verify<<<16,128,0,stream[r]>>>(source[r], static_cast<unsigned*>(mapped[r][r]),
                    static_cast<unsigned*>(mapped[r][1-r]), epochs[r], errors[r], count, salt, r);
            }
            return;
        }
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(r));
            produce<<<16,128,0,stream[r]>>>(source[r], count, salt, r);
            if (mode == 1) CK(cudaMemcpyAsync(host[r], source[r], bytes, cudaMemcpyDeviceToHost, stream[r]));
            if (mode == 2) publish<<<16,128,0,stream[r]>>>(source[r], static_cast<unsigned*>(mapped[r][r]), count);
            CK(cudaEventRecord(ready[r], stream[r]));
        }
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(r)); CK(cudaStreamWaitEvent(stream[r], ready[1-r], 0));
            if (mode == 0) CK(cudaMemcpyAsync(staging[r], source[1-r], bytes, cudaMemcpyDeviceToDevice, stream[r]));
            if (mode == 1) CK(cudaMemcpyAsync(staging[r], host[1-r], bytes, cudaMemcpyHostToDevice, stream[r]));
            verify<<<16,128,0,stream[r]>>>(mode == 2 ? static_cast<unsigned*>(mapped[r][1-r]) : staging[r],
                                          errors[r], count, salt, r);
            CK(cudaEventRecord(done[r], stream[r]));
        }
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(r)); CK(cudaStreamWaitEvent(stream[r], done[1-r], 0));
        }
    }
    void sync() { for (int r = 0; r < 2; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(stream[r])); } }
    void check() {
        sync();
        for (int r = 0; r < 2; ++r) {
            CK(cudaSetDevice(r)); unsigned n = 0;
            CK(cudaMemcpy(&n, errors[r], sizeof(n), cudaMemcpyDeviceToHost));
            if (n) { std::fprintf(stderr, "rank %d: %u incorrect words\n", r, n); std::exit(2); }
        }
    }
    void run(int mode) {
        constexpr unsigned rounds = 128, repetitions = 32;
        sync();
        for (int r = 0; r < 2; ++r) std::memset(host[r], 0xff, 2 * count * sizeof(unsigned));
        auto start = std::chrono::steady_clock::now();
        for (unsigned s = 0; s < rounds; ++s) issue(mode, s);
        check();
        const double eager = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        CK(cudaSetDevice(0)); CK(cudaStreamBeginCapture(stream[0], cudaStreamCaptureModeThreadLocal));
        CK(cudaEventRecord(fork, stream[0]));
        CK(cudaSetDevice(1)); CK(cudaStreamWaitEvent(stream[1], fork, 0));
        for (unsigned s = 0; s < rounds; ++s) issue(mode, s);
        CK(cudaSetDevice(1)); CK(cudaEventRecord(join, stream[1]));
        CK(cudaSetDevice(0)); CK(cudaStreamWaitEvent(stream[0], join, 0));
        cudaGraph_t graph{}; CK(cudaStreamEndCapture(stream[0], &graph));
        cudaGraphExec_t exec{}; CK(cudaGraphInstantiate(&exec, graph, 0));
        CK(cudaGraphLaunch(exec, stream[0])); check();
        start = std::chrono::steady_clock::now();
        CK(cudaSetDevice(0));
        for (unsigned j = 0; j < repetitions; ++j) CK(cudaGraphLaunch(exec, stream[0]));
        check();
        const double captured = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("mode=%s bytes=%zu eager_us=%.3f graph_us=%.3f exact=PASS\n",
                    mode == 0 ? "uva-d2d" : mode == 1 ? "pinned-dma" : mode == 2 ? "mapped-kernel" : "mapped-packet",
                    count * sizeof(unsigned), eager * 1e6 / rounds, captured * 1e6 / (rounds * repetitions));
        std::fflush(stdout); CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
    }
};

int main() {
    int n = 0; CK(cudaGetDeviceCount(&n)); if (n < 2) return 77;
    for (int r = 0; r < 2; ++r) {
        cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, r));
        int peer = 0; CK(cudaDeviceCanAccessPeer(&peer,r,1-r));
        std::printf("rank=%d name=%s sm=%d%d mapped=%d host_native_atomics=%d peer=%d\n",
                    r, p.name, p.major, p.minor, p.canMapHostMemory, p.hostNativeAtomicSupported, peer);
    }
    for (int bytes : {10240, 51200, 81920, 1048576}) {
        Pair p(bytes / sizeof(unsigned));
        for (int mode = 0; mode < (bytes <= 81920 ? 4 : 3); ++mode) p.run(mode);
    }
}
