#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int REPEATS = 10;
const int N = 150000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void timing(real *h_x, real *d_x, const int method);

int main(void)
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("\nUsing global memory only:\n");
    timing(h_x, d_x, 0);
    // The "0" goes first into "timing" and then into "reduce".
    printf("\nUsing static shared memory:\n");
    timing(h_x, d_x, 1);
    printf("\nUsing static shared memory and atomicAdd:\n");
    timing(h_x, d_x, 2);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    real *x = d_x + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}

void __global__ reduce_shared(real *d_x, real *d_y){

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if(tid < offset){
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads(); //make sure that a round have been completed
    }
    
    //after all rounds are completed, pick it if it is the first item
    if(tid == 0){
        d_y[bid] = s_y[0];
    }
}

void __global__ reduce_atom(real *d_x, real *d_y){
 
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if(tid < offset){
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads(); //make sure that a round have been completed
    }
    
    //after all rounds are completed, pick it if it is the first item
    if(tid == 0){
        atomicAdd(d_y, s_y[0]);
    }
}

real reduce(real *d_x, const int method)
{
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ymem = sizeof(real) * grid_size;
    //const int smem = sizeof(real) * BLOCK_SIZE;
    real *d_y, *h_y;

    switch(method){
        case(0):
        case(1):
            CHECK(cudaMalloc(&d_y, ymem));
            h_y = (real *) malloc(ymem);
            break;
        case (2):
            CHECK(cudaMalloc(&d_y, sizeof(real)));
            h_y = new real;
            *h_y = 0.0;
            CHECK(cudaMemcpy(d_y, h_y, sizeof(real), cudaMemcpyHostToDevice));
            //to initialize d_y[0]
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }
    

    switch (method)
    {
        case 0:
            reduce_global<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 1:
            reduce_shared<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        case 2:
            reduce_atom<<<grid_size, BLOCK_SIZE>>>(d_x, d_y);
            break;
        default:
            printf("Error: wrong method\n");
            exit(1);
            break;
    }

    real result = 0.0;
    if(method != 2){
        CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));
        //complete the sum in CPU
        for (int n = 0; n < grid_size; ++n)
        {
            result += h_y[n];
        }
    }
    else{
        CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));
        result = *h_y;
    }

    free(h_y);
    CHECK(cudaFree(d_y));
    return result;
}

void timing(real *h_x, real *d_x, const int method){

    real sum = 0;

    float t_sum = 0;
    float t2_sum = 0;
  
    for(int rep = 0; rep <= REPEATS; rep ++){

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
        sum = reduce(d_x, method);  
        //Inv-cpy is included, which differs according to the method.

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed;
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    
        if(rep > 0){
            t_sum += elapsed;
            t2_sum += elapsed * elapsed;
        }
    }

    const float t_ave = t_sum / REPEATS;
    const float t_err = sqrt(t2_sum/REPEATS - t_ave*t_ave);
    printf("Time = %g +- %g ms\n",t_ave, t_err);  //and \n !
    printf("sum = %f.\n", sum);
    printf("\n");
}