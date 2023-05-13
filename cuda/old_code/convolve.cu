#include <iostream>
#include <math.h>
#include <chrono>

#define Sy 1
#define Sx 1

// convolution1 parameters
// TODO change back to 224
#define Nx 224 // image feature map width
#define Ny 224 // image feature map height
#define Kx 3   // kernel width
#define Ky 3   // kernel height
// TODO change back to 64
#define Ni 64 // feature map input channels
#define Nn 64 // feature map output channels

#define VTYPE int

#define NYPAD (Ny + Ky - 1)
#define NXPAD (Nx + Kx - 1)

#define NYSCL (Ny / Sy)
#define NXSCL (Nx / Sx)

#define SYNAPSE_SIZE (1L * Ky * Kx * Nn * Ni)

// classifier params
#define NiSIZE 25088
#define NnSIZE 4096

// basic cpu timer
struct Timer
{
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::duration<double> elapsed_time;

    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        std::cout << "Elapsed time: " << get_elapsed_time() << " seconds\n";
    }

    double get_elapsed_time() const
    {
        return elapsed_time.count();
    }
};

void cpuConv(VTYPE (&layer_in)[NXPAD][NYPAD][Ni], VTYPE (&layer_out)[NXSCL][NYSCL][Nn], VTYPE (&synapse)[Kx][Ky][Ni][Nn])
{

    for (int nx = 0; nx < NXSCL; nx++)
    {
        for (int ny = 0; ny < NYSCL; ny++)
        {
            // for every layer, fill out the output depth
            for (int nn = 0; nn < Nn; nn++)
            {
                VTYPE sum = 0;
                for (int ni = 0; ni < Ni; ni++)
                {
                    // loop through synapse and gather partial sum
                    for (int kx = 0; kx < Kx; kx++)
                    {
                        for (int ky = 0; ky < Ky; ky++)
                        {
                            sum += synapse[kx][ky][ni][nn] * layer_in[nx + kx][ny + ky][ni];
                        }
                    }
                }
                // printf("xx: %d, yy: %d, nn: %d, sum: %d\n", xx, yy, nn, sum);
                layer_out[nx][ny][nn] = sum;
            }
        }
    }

    return;
}

__global__ void cudaConv(VTYPE (&layer_in)[NXPAD][NYPAD][Ni], VTYPE (&layer_out)[NXSCL][NYSCL][Nn], VTYPE (&synapse)[Kx][Ky][Ni][Nn])
{
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;

    // printf("blockDim.x: %d, blockDim.y: %d, blockDim.z: %d, gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

    // int output_x_index = blockIdx.x;
    // printf("theadidx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);

    int x_idx = threadIdx.x;
    int x_stride = blockDim.x;
    // printf("threadidx.x: %d\n", threadIdx.x);

    // printf("x_idx: %d, y_idx: %d, x_stride: %d, y_stride: %d\n", x_idx, y_idx, x_stride, y_stride);

    // input channel
    // printf("threadIdx.x: %d, threadIdx.y: %d, blockDim.x: %d, blockDim.y: %d, blockIdx.x: %d, blockIdx.y: %d\n", threadIdx.x, threadIdx.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y);
    int input_channel_index = blockIdx.x;
    int input_channel_stride = gridDim.x;

    // int output_channel_index = blockIdx.y;
    // int output_channel_stride = gridDim.y;

    for (int nx = 0; nx < NXSCL; nx += 1)
    {
        for (int ny = 0; ny < NYSCL; ny += 1)
        {
            // iterate of channel depth
            for (int nn = input_channel_index; nn < Nn; nn += input_channel_stride)
            {
                VTYPE sum = 0;
                for (int ni = 0; ni < Ni; ni += 1)
                {
                    // loop through synapse and gather partial sum
                    for (int kx = 0; kx < Kx; kx++)
                    {
                        for (int ky = 0; ky < Ky; ky++)
                        {
                            // printf("synapse[%d][%d][%d][%d]: %f\n", kx, ky, ni, nn, synapse[kx][ky][ni][nn]);
                            sum += synapse[kx][ky][ni][nn] * layer_in[nx + kx][ny + ky][ni];
                        }
                    }
                }
                __syncthreads();
                layer_out[nx][ny][nn] += sum;
            }
        }
    }
    return;
}

int main(void)
{

    Timer timer;
    srand(time(NULL));

    // cuda timer variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // CPU convolution parameters
    VTYPE(*synapse)
    [Kx][Ky][Ni][Nn] = (VTYPE(*)[Kx][Ky][Ni][Nn])malloc(sizeof(VTYPE) * SYNAPSE_SIZE);
    VTYPE(*neuron_i)
    [NXPAD][NYPAD][Ni] = (VTYPE(*)[NXPAD][NYPAD][Ni])malloc(sizeof(VTYPE) * NXPAD * NYPAD * Ni);
    VTYPE(*neuron_n)
    [NXSCL][NYSCL][Nn] = (VTYPE(*)[NXSCL][NYSCL][Nn])malloc(sizeof(VTYPE) * NXSCL * NYSCL * Nn);

    // GPU versions
    VTYPE(*synapse_gpu)
    [Kx][Ky][Ni][Nn];
    VTYPE(*neuron_i_gpu)
    [NXPAD][NYPAD][Ni];
    VTYPE(*neuron_n_gpu)
    [NXSCL][NYSCL][Nn];

    // classifier parameters
    VTYPE(*class_layer1)
    [NiSIZE] = (VTYPE(*)[NiSIZE])malloc(sizeof(VTYPE) * NiSIZE);
    VTYPE(*class_layer2)
    [NnSIZE] = (VTYPE(*)[NnSIZE])malloc(sizeof(VTYPE) * NnSIZE);
    VTYPE(*class_weights)
    [NnSIZE][NiSIZE] = (VTYPE(*)[NnSIZE][NiSIZE])malloc(sizeof(VTYPE) * NnSIZE * NiSIZE);

    // GPU versions
    VTYPE(*class_layer1_gpu)
    [NiSIZE];
    VTYPE(*class_layer2_gpu)
    [NnSIZE];
    VTYPE(*class_weights_gpu)
    [NnSIZE][NiSIZE];

    // allocate memory for GPU
    cudaMallocManaged(&class_layer1_gpu, sizeof(VTYPE) * NiSIZE);
    cudaMallocManaged(&class_layer2_gpu, sizeof(VTYPE) * NnSIZE);
    cudaMallocManaged(&class_weights_gpu, sizeof(VTYPE) * NnSIZE * NiSIZE);

    cudaMallocManaged(&synapse_gpu, sizeof(VTYPE) * SYNAPSE_SIZE);
    cudaMallocManaged(&neuron_i_gpu, sizeof(VTYPE) * NXPAD * NYPAD * Ni);
    cudaMallocManaged(&neuron_n_gpu, sizeof(VTYPE) * NXSCL * NYSCL * Nn);

    // populate values for neuron_i on CPU
    std::cout << "populating neurons on cpu - ";
    timer.start();
    for (int nx = 0; nx < NXPAD; nx++)
    {
        for (int ny = 0; ny < NYPAD; ny++)
        {
            for (int ni = 0; ni < Ni; ni++)
            {
                // generate a random number between 1 and 10
                // neuron_i[0][nx][ny][ni] = static_cast<VTYPE>(rand()) / static_cast<VTYPE>(RAND_MAX) - 0.5f;
                int randomNumber = rand() % 10 + 1;
                neuron_i[0][nx][ny][ni] = static_cast<VTYPE>(randomNumber);
                // neuron_i[0][nx][ny][ni] = 3.0f;
                neuron_i_gpu[0][nx][ny][ni] = neuron_i[0][nx][ny][ni];
            }
        }
    }
    for (int nx = 0; nx < NXSCL; nx++)
    {
        for (int ny = 0; ny < NYSCL; ny++)
        {
            for (int nn = 0; nn < Nn; nn++)
            {
                neuron_n[0][nx][ny][nn] = 0;
                neuron_n_gpu[0][nx][ny][nn] = 0;
            }
        }
    }
    for (int yy = 0; yy < Ky; ++yy)
    {
        for (int xx = 0; xx < Kx; ++xx)
        {
            for (int nn = 0; nn < Nn; ++nn)
            {
                for (int ni = 0; ni < Ni; ++ni)
                {
                    int randomNumber = rand() % 10 + 1;
                    // neuron_i[0][nx][ny][ni] = static_cast<VTYPE>(randomNumber);
                    // synapse[0][yy][xx][nn][ni] = static_cast<VTYPE>(rand()) / static_cast<VTYPE>(RAND_MAX) - 0.5f;
                    synapse[0][yy][xx][nn][ni] = static_cast<VTYPE>(randomNumber);
                    // synapse[0][yy][xx][nn][ni] = 1.0f;
                    synapse_gpu[0][yy][xx][nn][ni] = synapse[0][yy][xx][nn][ni];
                }
            }
        }
    }
    timer.stop();

    std::cout << "CPU Convolution - ";
    timer.start();
    cpuConv(*neuron_i, *neuron_n, *synapse);
    timer.stop();

    int xdim = NXPAD / Kx;
    int ydim = NYPAD / Ky;
    dim3 block_dim(1, 1, 1);
    dim3 grid_dim(Nn, 1, 1);
    std::cout << "GPU Convolution - "; // << std::endl;
    // timer.start();
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    // cudaConv<<<grid_dim, block_dim>>>(*neuron_i_gpu, *neuron_n_gpu, *synapse_gpu);
    cudaConv<<<grid_dim, block_dim>>>(*neuron_i_gpu, *neuron_n_gpu, *synapse_gpu);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaDeviceSynchronize();
    printf("%f ms\n", milliseconds);

    // input neurons differences just as a sanity check
    float error = 0;
    for (int nx = 0; nx < NXPAD; nx++)
    {
        for (int ny = 0; ny < NYPAD; ny++)
        {
            for (int nn = 0; nn < Nn; nn++)
            {
                error += abs(neuron_i[0][nx][ny][nn] - neuron_i_gpu[0][nx][ny][nn]);
            }
        }
    }
    printf("input error: %f\n", error);

    // accumulate the error
    error = 0;
    for (int nx = 0; nx < NXSCL; nx++)
    {
        for (int ny = 0; ny < NYSCL; ny++)
        {
            for (int nn = 0; nn < Nn; nn++)
            {
                error += abs(neuron_n[0][nx][ny][nn] - neuron_n_gpu[0][nx][ny][nn]);
            }
        }
    }
    printf("output error: %f\n", error);
    return 0;
}