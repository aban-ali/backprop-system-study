#include <stdio.h>
#include <stdlib.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class NeuralNet{
private:
    float* weights;
    int layers;
public:
    NeuralNet(py::array_t<float> input_array){
        py::buffer_info buf = input_array.request();
        float* h_ptr = static_cast<float*>(buf.ptr);
        layers = buf.size;
    }


    ~NeuralNet(){
        delete[] weights;
    }
}



PYBIND11_MODULE(cuda_mlp, m){
    // m.def("add", &add, "Function to add two numbers.");
}


// command to compile the file:
//  g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
//   g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example.so












void double_numpy_array() {
    // Request a buffer descriptor from NumPy (checks memory layout)
    

    // Allocate GPU memory
    float* d_ptr;
    cudaMalloc(&d_ptr, N * sizeof(float));
}