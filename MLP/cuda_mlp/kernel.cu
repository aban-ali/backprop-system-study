// initializes weights of the network.
// I guess I have taken the sqrt very aggressively
// to avoid making kernel complex
__global__
void init_weights(float* weights, size_t weights_size){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < weights_size){
        float r = ((float)((i * 17 + 13) % 1000) / 500.0f) - 1.0f;
        weights[i] = r * sqrtf(2.0f/weights_size);
    }
}

// initializes biases of the network parallely
__global__
void init_biases(float* biases, size_t biases_size){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < biases_size){
        float r = ((float)((i * 13 + 17) % 1000) / 500.0f) - 1.0f;
        biases[i] = r * sqrtf(2.0f/biases_size);
    }
}

// initialize all the gradient with 0. kind of torch.zero_grad()
__global__
void init_grads(float* grads, size_t grad_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<grad_size)
        grads[i] = 0;
    return;
}

// Implements Dot Product of Matrix with a vector.
// It also provides implementaion of ReLU activation, if specified.
// Not suited for batching. Batching will be considered later.
__global__
void dot_prod(
    float* mat, size_t rows, float* bias,
    float* vec, size_t cols, float* out, bool isrelu
    ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float temp_vec[];

    for(int k=threadIdx.x; k<cols; k+=blockDim.x)
        temp_vec[k] = vec[k];
    __syncthreads();

    float val = 0.0f;
    for(int idx=0; idx<cols && i<rows; idx++)
        val += temp_vec[idx] * mat[i*cols + idx];

    if(i<rows){
        float res = val + bias[i];
        if(isrelu)
            out[i] = res > 0? res : 0;
        else
            out[i] = res;
    }
    return;
}

// implement dot product used during forward pass
// for backpropogation.
__global__
void backprop_dotprod(
    float* mat, size_t rows, float* bias, float* vec, 
    size_t cols, float* zs, float* activations
    ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float temp_vec[];

    for(int k=threadIdx.x; k<cols; k+=blockDim.x)
        temp_vec[k] = vec[k];
    __syncthreads();

    float val = 0.0f;
    for(int idx=0; idx<cols && i<rows; idx++)
        val += temp_vec[idx] * mat[i*cols + idx];

    if(i<rows){
        zs[i] = val + bias[i];
        activations[i] = zs[i] > 0? zs[i] : 0;
    }
    return;
}


// Implements ReLU activation
__global__
void relu(float* in, float* out, size_t len){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<len){
        float val = in[i];
        out[i] = val > 0 ? val : 0;
    }
}


__global__
void last_layer_backward(float* grads_b, float* a, int y, int len){
    int i = threadIdx.x;
    if(i < len)
        grads_b[i] += i==y ? a[i] - 1 : a[i];
}

__device__
float relu_prime(float val){
    return val>0;
}

__global__  
void update_grads(
        float* weights, float* grads_w, float* grads_b,
        size_t l1, size_t l2, 
        float* activations, float* zs
    ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<l1){
        float b=0;
        for(int k=0; k<l2; k++){
            float gradb = grads_b[l1 + k];
            grads_w[k*l1 + i] += gradb * activations[i];
            b += gradb * weights[k*l1 + i];
        }
        grads_b[i] += b * relu_prime(zs[i]);
    }
}


__global__
void update_weights(
        float* grads_w, float* grads_b, 
        float* x, size_t l1, size_t l2
    ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < l1){
        for(int k=0; k<l2; k++)
            grads_w[k*l1 + i] += grads_b[k] * x[i];
    }
}


__global__
void update_parameters(float* parameters, float* grads, int len, int batch_size, float lr){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<len)
        parameters[i] -= lr * grads[i] / (float)batch_size;
}



__global__
void softmax(float* logits, int N, float* probs){

    if(threadIdx.x < 1){
        float max_val = logits[0];
        for(int i=0; i<N; i++)
            max_val = fmaxf(max_val, logits[i]);

        float total = 0.0f;
        for(int i=0; i<N; i++){
            probs[i] = expf(logits[i]-max_val);
            total += probs[i];
        }

        for(int i=0; i<N; i++)
            probs[i] /= total;
    }
}