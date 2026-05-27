#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>

namespace py = pybind11;

__global__
void init_weights(float* weights, size_t weights_size){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < weights_size){
        int temp = i % 7;
        weights[i] = (float)temp / (7.0 * sqrtf(weights_size));
    }
}

__global__
void init_biases(float* biases, size_t biases_size){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < biases_size){
        int temp = i % 7;
        biases[i] = (float)temp / 7.0;
    }
}

// Implements Dot Product of Matrix with a vector.
// It also provides implementaion of ReLU activation, if specified.
// Not suited for batching. Batching will be considered later.
__global__
void dot_prod(
    float* mat, size_t rows, float* bias,
    float* vec, size_t cols, float* out, bool relu
    ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float temp_vec[];

    for(int k=i; k<cols && i<rows; k+=rows)
        temp_vec[k] = vec[k];
    __syncthreads();

    float val = 0.0f;
    for(int idx=0; idx<cols && i<rows; idx++)
        val += temp_vec[idx] * mat[i*cols + idx];

    if(i<rows){
        float res = val + bias[i];
        if(relu)
            out[i] = res > 0? res : 0;
        else
            out[i] = res;
    }
            return;
}



class NeuralNet{
private:
    float* weights = nullptr;       // contains the weights of the Neural Net
    float* biases = nullptr;        // contains the biases of the Neurons
    int*   layer_sizes = nullptr;   // contains the architecture of the Neural Net
    int    num_layers;              // stores the number of entries in architecture array 

public:
    // Avoids creation of same class or something,
    // not sure about its working but it is necessary.
    NeuralNet(const NeuralNet&) = delete;
    NeuralNet& operator=(const NeuralNet&) = delete;

    // constructor to initialize the MLP
    NeuralNet(py::array_t<int> input_array){
        py::buffer_info buf = input_array.request();
        int* h_ptr = static_cast<int*>(buf.ptr);
        num_layers = buf.size;
        weights_init(h_ptr);
    }

    // initialize weights and biases in the GPU.
    // store the architecture of Neural Network. 
    void weights_init(int* h_ptr){
        size_t weights_size = 0;
        size_t biases_size = 0;
        cudaError_t err_w;
        cudaError_t err_b;

        for(int i=1; i<num_layers; i++){
            weights_size += h_ptr[i] * h_ptr[i-1];
            biases_size += h_ptr[i];
        }
        err_w = cudaMalloc((void**)&weights, weights_size * sizeof(float));
        err_b = cudaMalloc((void**)&biases, biases_size * sizeof(float));

        if(err_w != cudaSuccess)
            std::cout << cudaGetErrorString(err_w);
        if(err_b != cudaSuccess)
            std::cout << cudaGetErrorString(err_b);

        cudaMallocHost((void**)&layer_sizes, num_layers * sizeof(int));
        memcpy(layer_sizes, h_ptr, num_layers * sizeof(int));

        init_weights<<<(weights_size + 127) / 128, 128>>>(weights, weights_size);
        init_biases<<<(biases_size + 127) / 128, 128>>>(biases, biases_size);
    }

    // calculates the 
    int feedforward(py::array_t<float> arr){
        py::buffer_info buf = arr.request();
        float*  x = static_cast<float*>(buf.ptr);
        size_t  len = buf.size;

        size_t  offset_w = 0;
        size_t  offset_b = 0;
        int     max_net = 0;
        float*  out1;
        float*  out2;
        float*  ans = new float[ layer_sizes[ num_layers - 1 ] ];

        for(int idx=0; idx<num_layers; idx++)
            max_net = std::max(max_net, layer_sizes[idx]);

        cudaMalloc((void**)&out1, max_net * sizeof(float));
        cudaMalloc((void**)&out2, max_net * sizeof(float));
        cudaMemcpy(out1, x, len*sizeof(float), cudaMemcpyHostToDevice);

        // ping pong implementation for dot product calculation
        for(int i=0; i<num_layers-2; i++){
            dot_prod<<<(layer_sizes[i+1] + 63)/64, 64, layer_sizes[i]*sizeof(float)>>>(
                &weights[offset_w], layer_sizes[i+1], &biases[offset_b],
                out1, layer_sizes[i], out2, true
            );
            // cudaDeviceSynchronize();
            offset_w += layer_sizes[i] * layer_sizes[i+1];
            offset_b += layer_sizes[i+1];
            
            std::swap(out1, out2);
        }

        dot_prod<<<(layer_sizes[num_layers-1] + 63)/64, 64, layer_sizes[num_layers-2]*sizeof(float)>>>(
            &weights[offset_w], layer_sizes[num_layers-1], &biases[offset_b],
            out1, layer_sizes[num_layers-2], out2, false
        );
        cudaMemcpy(ans, out2, layer_sizes[num_layers-1] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(out1);
        cudaFree(out2);

        int best=0;
        for(int idx=0; idx<layer_sizes[num_layers-1]; idx++)
            best = ans[idx] > ans[best] ? idx : best;

        delete[] ans;
        return best;
    }

    // Deconstructor to avoid memory leaks after 
    // instance of Neural Net teminates
    ~NeuralNet(){
        if(weights)
            cudaFree(weights);
        if(biases)
            cudaFree(biases);
        if(layer_sizes)
            cudaFreeHost(layer_sizes);
    }
};


PYBIND11_MODULE(cuda_mlp, m){
    py::class_<NeuralNet>(m, "NeuralNet")
        .def(py::init<py::array_t<int>>())
        .def("feedforward", &NeuralNet::feedforward);
}


// command to compile the file:
// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example.so
// nvcc -O3 -shared -Xcompiler -fPIC $(python3 -m pybind11 --includes) cuda_mlp.cu -o cuda_extension.so