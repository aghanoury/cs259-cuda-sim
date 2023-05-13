#include <iostream>
#include "util.cuh"

#ifndef Nn
  #define Nn 128
  #define Ni 224
#endif

#ifndef Tii
  #define Tnn 32  
  #define Tii 32
  #define Tn 16
  #define Ti 16
#endif

__device__ VTYPE d_transfer(VTYPE i) {
  return (i > 0) ? i : i / 4;
}

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
    VTYPE (&neuron_n)[Nn],   VTYPE (&neuron_n2)[Nn]) {
  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      synapse[n][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    }
  }
  for(int i = 0; i < Ni; ++i) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for(int n = 0; n < Nn; ++n) {
    neuron_n[n] = 0;
    neuron_n2[n] = 0;
  }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
  for (int n = 0; n < Nn; n++) {
    VTYPE temp=0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

__global__ void d_classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  VTYPE temp=0;
  for (int i = 0; i < Ni; i++) {
    temp += synapse[n][i] * neuron_i[i];
  }
  neuron_n[n] = d_transfer(temp);
}

int main(int argc, char** argv) {
  using namespace std;

  VTYPE (*synapse)[Nn][Ni];
  VTYPE (*neuron_i)[Ni];
  VTYPE (*neuron_n)[Nn];
  VTYPE (*neuron_n2)[Nn];

  VTYPE (*d_synapse)[Nn][Ni];
  VTYPE (*d_neuron_i)[Ni];
  VTYPE (*d_neuron_n)[Nn];

  synapse = (VTYPE (*)[Nn][Ni]) malloc(Nn * Ni * sizeof(VTYPE));
  neuron_i = (VTYPE (*)[Ni]) malloc(Ni * sizeof(VTYPE));
  neuron_n = (VTYPE (*)[Nn]) malloc(Nn * sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[Nn]) malloc(Nn * sizeof(VTYPE));

  fill_classifier(*synapse, *neuron_i, *neuron_n, *neuron_n2);

  cudaMalloc(&d_synapse, Nn * Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_i, Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_n, Nn * sizeof(VTYPE));

  cudaMemcpy(d_synapse, synapse, Nn * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron_i, neuron_i, Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);

  int threads = 256;

  begin_roi();
  d_classifier_layer<<<(Nn + threads - 1) / threads, threads>>>(*d_synapse, *d_neuron_i, *d_neuron_n);
  cudaDeviceSynchronize();
  end_roi();

  cudaMemcpy(neuron_n, d_neuron_n, Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);

  begin_roi();
  classifier_layer(*synapse, *neuron_i, *neuron_n2);  
  end_roi();

  compare(*neuron_n, *neuron_n2, Nn);
}