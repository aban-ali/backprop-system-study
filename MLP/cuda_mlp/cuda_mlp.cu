#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "kernel.cu"

namespace py = pybind11;


class NeuralNet{
private:
    float* weights = nullptr;       // contains the weights of the Neural Net
    float* biases = nullptr;        // contains the biases of the Neurons
    int*   layer_sizes = nullptr;   // contains the architecture of the Neural Net
    float*  grads_w = nullptr;
    float*  grads_b = nullptr;
    float*  zs = nullptr;
    float*  activations = nullptr;
    float*  ans = nullptr;
    int    num_layers;              // stores the number of entries in architecture array 


    // Implements the forward pass for backpropogation,
    // specifically stores all the z and activation values.
    void forward(float* x, int cols){
        float* vec;
        float* out;
        size_t max_len = 0;
        size_t offset_w = 0;
        size_t offset_b = 0;

        for(int i=0; i<num_layers; i++)
            max_len = std::max((int)max_len, layer_sizes[i]);

        cudaMalloc((void**)&vec, max_len * sizeof(float));
        cudaMalloc((void**)&out, max_len * sizeof(float));
        cudaMemcpy(vec, x, cols * sizeof(float), cudaMemcpyHostToDevice);

        for(int i=0; i<num_layers-1; i++){
            backprop_dotprod<<<(layer_sizes[i+1] + 63)/64, 64, layer_sizes[i]*sizeof(float)>>>(
                &weights[offset_w], layer_sizes[i+1],
                &biases[offset_b], vec, layer_sizes[i],
                &zs[offset_b], &activations[offset_b], out
            );

            offset_w += layer_sizes[i] * layer_sizes[i+1];
            offset_b += layer_sizes[i+1];
            cudaDeviceSynchronize();

            std::swap(vec, out);
        }

        offset_b -= layer_sizes[num_layers-1];
        softmax<<<1, 1>>>(&zs[offset_b], layer_sizes[num_layers-1], &activations[offset_b]);

        float* temp = (float*)malloc(10*sizeof(float));
        cudaMemcpy(temp, &activations[offset_b], 10*sizeof(float), cudaMemcpyDeviceToHost);
        float s=0;
        for(int i=0; i<10; i++){
            s+=temp[i];
        }
        std::cout << "Total Sum of activations: \t" << s << "\n";

        cudaFree(vec);
        cudaFree(out);
        return;
    }

    void backwards(float* x, int cols, size_t weights_size, size_t biases_size){
        biases_size -= layer_sizes[num_layers-1];
        // last layer has only 10 neurons, so 1 block of 64 threads suffices in this case.
        last_layer_backward<<<1, 64>>>(
            &grads_b[biases_size], &activations[biases_size],
            x[cols-1], layer_sizes[num_layers-1]
        );

        for(int i=num_layers-1; i>1; i--){
            weights_size -= layer_sizes[i] * layer_sizes[i-1];
            biases_size -= layer_sizes[i-1];

            cudaDeviceSynchronize();
            update_grads<<<(layer_sizes[i-1] + 63) / 64, 64>>>(
                &weights[weights_size], &grads_w[weights_size], 
                &grads_b[biases_size], layer_sizes[i-1], layer_sizes[i],
                &activations[biases_size], &zs[biases_size]
            );
        }
        cudaDeviceSynchronize();
        update_weights<<<(layer_sizes[0] + 63) / 64 ,64>>>(grads_w, grads_b, x, layer_sizes[0], layer_sizes[1]);
        cudaDeviceSynchronize();
    }



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

        ans = (float*)malloc(h_ptr[num_layers-1]*sizeof(float));
        err_w = cudaMalloc((void**)&weights, weights_size * sizeof(float));
        err_b = cudaMalloc((void**)&biases, biases_size * sizeof(float));

        backprop_vars_init(weights_size, biases_size);      // allocate memory for gradients and all to avoid memory allocation overhead later.

        if(err_w != cudaSuccess)
            std::cout << cudaGetErrorString(err_w);
        if(err_b != cudaSuccess)
            std::cout << cudaGetErrorString(err_b);

        cudaMallocHost((void**)&layer_sizes, num_layers * sizeof(int));
        memcpy(layer_sizes, h_ptr, num_layers * sizeof(int));

        init_weights<<<(weights_size + 127) / 128, 128>>>(weights, weights_size);
        init_biases<<<(biases_size + 127) / 128, 128>>>(biases, biases_size);
    }

    // allocate memory in GPU for gradients, activation and z values
    // This avoids malloc overhead in later part of program
    void backprop_vars_init(size_t weights_size, size_t biases_size){
        cudaError_t e1, e2, e3, e4;

        e1 = cudaMalloc((void**)&grads_w, weights_size * sizeof(float));
        if(e1 != cudaSuccess)
            std::cout << cudaGetErrorString(e1);
        else
            std::cout<<"Space for grads_w allocated.\n";

        e2 = cudaMalloc((void**)&grads_b, biases_size * sizeof(float));
        if(e2 != cudaSuccess)
            std::cout << cudaGetErrorString(e2);
        else
            std::cout<<"Space for grads_b allocated.\n";

        e3 = cudaMalloc((void**)&zs, biases_size * sizeof(float));
        if(e3 != cudaSuccess)
            std::cout<<cudaGetErrorString(e3);
        else
            std::cout<<"Space for zs allocated.\n";

        e4 = cudaMalloc((void**)&activations, biases_size * sizeof(float));
        if(e4 != cudaSuccess)
            std::cout << cudaGetErrorString(e4);
        else
            std::cout<<"Space for activations allocated.\n";
    }

    // calculates the target value of input variable x
    // i.e. implemented forward pass for the network
    py::array_t<float> feedforward(py::array_t<float> arr){
        py::buffer_info buf = arr.request();
        float*  x = static_cast<float*>(buf.ptr);
        size_t  len = buf.size;

        size_t  offset_w = 0;
        size_t  offset_b = 0;
        int     max_net = 0;
        float*  out1;
        float*  out2;

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

        return py::array_t<float>(layer_sizes[ num_layers - 1 ], ans);
    }

    void update_miniBatch(
        py::array_t<float, py::array::c_style | py::array::forcecast> arr,
        float lr
    ){
        py::buffer_info buf = arr.request();
        float* data = static_cast<float*>(buf.ptr);
        int rows = buf.shape[0];
        int cols = buf.shape[1];

        size_t  weights_size = 0;
        size_t  biases_size = 0;

        for(int i=1; i<num_layers; i++){
            weights_size += layer_sizes[i] * layer_sizes[i-1];
            biases_size += layer_sizes[i];
        }

        init_grads<<<(weights_size + 63) / 64, 64>>>(grads_w, weights_size);
        init_grads<<<(biases_size + 63) / 64, 64>>>(grads_b, biases_size);

        for(int batch=0; batch<rows; batch++){
            
            forward(&data[batch * cols], cols-1);
            backwards(&data[batch * cols], cols, weights_size, biases_size);
            
        }

        update_parameters<<<(weights_size + 63) / 64, 64>>>(
            weights, grads_w, weights_size, rows, lr
        );
        update_parameters<<<(biases_size + 63) / 64, 64>>>(
            biases, grads_b, biases_size, rows, lr
        );
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
        if(ans)
            free(ans);

        cudaFree(grads_w);
        cudaFree(grads_b);
        cudaFree(activations);
        cudaFree(zs);
    }
};


PYBIND11_MODULE(cuda_mlp, m){
    py::class_<NeuralNet>(m, "NeuralNet")
        .def(py::init<py::array_t<int>>())
        .def("feedforward", &NeuralNet::feedforward)
        .def("update_miniBatch", &NeuralNet::update_miniBatch);
}


// command to compile the file:
// nvcc -O3 -shared -Xcompiler -fPIC $(python3 -m pybind11 --includes) cuda_mlp.cu -o cuda_extension.so