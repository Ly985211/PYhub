#include "../error.cuh"
#include <stdio.h>
#include <cooperative_groups.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

using namespace cooperative_groups;

const int REPEATS = 10;
const int N = 150000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;
const int GRID_SIZE_1 = 10240;
//must be less than (N + BLOCK_SIZE - 1) / BLOCK_SIZE to make an opt.

void timing(real *h_x, real *d_x);

int main()
{
    real *h_x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        h_x[n] = 1.23;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));

    printf("Using static shared memory, the cooperative group with optimization:\n");
    timing(h_x, d_x);

    free(h_x);
    CHECK(cudaFree(d_x));
    return 0;
}

void __global__ reduce_opt(real *d_x, real *d_y,const int N){
 
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ real s_y[BLOCK_SIZE]; 
    //Logically this should be blockDim.x, which is grammarly forbiddened.
    //Use dynamic memory instead to solve this.

    real y = 0.0;//no conflict in the first loop.
    const int stride = gridDim.x * blockDim.x;
    for(int n = bid * blockDim.x + tid; n < N; n += stride){
        y += d_x[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for(int offset = blockDim.x >> 1; offset >= 32; offset >>= 1){
        if(tid < offset){
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads(); //make sure that a round have been completed
    }
    
    y = s_y[tid];
    thread_block_tile<32>g = tiled_partition<32>(this_thread_block());
    for(int ofs = 16; ofs > 0; ofs >>= 1){  
        // To continue reducing in [0,32), instead of blockDim.x >> 1, which turns out with
        // a result 3 times larger.
        y += g.shfl_down(y, ofs);
    }

    //after all rounds are completed, pick it if it is the first item
    if(tid == 0){
        d_y[bid] = y;
    }
}

real reduce(real *d_x){

    real *d_y;
    const int ymem = sizeof(real) * GRID_SIZE_1;
    CHECK(cudaMalloc(&d_y, ymem));
    reduce_opt<<<GRID_SIZE_1, BLOCK_SIZE>>>(d_x, d_y, N);

    real *d_z = new real;
    CHECK(cudaMalloc(&d_z, sizeof(real)));
    reduce_opt<<<1, BLOCK_SIZE>>>(d_y, d_z, GRID_SIZE_1);

    real *h_z = new real;
    CHECK(cudaMemcpy(h_z, d_z, sizeof(real), cudaMemcpyDeviceToHost));
    real result = *h_z;

    CHECK(cudaFree(d_y));
    free(h_z);
    CHECK(cudaFree(d_z));
    return result;
}

void timing(real *h_x, real *d_x){

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
        sum = reduce(d_x);  
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