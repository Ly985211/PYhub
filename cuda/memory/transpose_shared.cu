#include "error.cuh"
#include<stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int REPEATS = 10;
const int TILE_DIM = 32;

void timing(const real* d_A, real* d_B, const int N, const int task);
__global__ void transpose1(const real* A, real* B, const int N);
__global__ void transpose2(const real* A, real* B, const int N);
__global__ void transpose3(const real* A, real* B, const int N);
bool check_equal(const real* A, const real* B, const int N);
void print_mat(const int N, const real* A);

int main(int argc, char** argv){

    if(argc != 2){
        printf("wrong input!\n");
        exit(1);
    }

    const int N = atoi(argv[1]);
    const int N2 = N * N;
    const int M = sizeof(real) * N2;
    real* h_A = new real[M];
    real* h_B = new real[M];
    for(int n = 0; n<N2; n++){
        h_A[n] = n;
    }

    real *d_A, *d_B;  // not "real* d_A, d_B" !
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    printf("\ntranspose with coalesced read:\n");
    timing(d_A, d_B, N, 1);
    printf("\ntranspose with coalesced write(automatic __ldg read):\n");
    timing(d_A, d_B, N, 2);
    printf("\ntranspose with share memory:\n");
    timing(d_A, d_B, N, 3);

    CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));

    printf(check_equal(h_A, h_B, N)? "A = B^T\n" : "A != B^T\n");
    if(N <= 20){
        printf("A = \n");
        print_mat(N, h_A);
        printf("B = \n");
        print_mat(N, h_B);
    }

    free(h_A);
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;

}

void timing(const real* d_A, real* d_B, const int N, const int task){

    const int grid_x = (N+TILE_DIM-1) / TILE_DIM;
    const int grid_y = grid_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_x, grid_y);

    float t_sum = 0;
    float t2_sum = 0;
  
    for(int rep = 0; rep <= REPEATS; rep ++){
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        switch (task)
        {
            case 1:
                transpose1<<<grid_size,block_size>>>(d_A, d_B, N);
                break;
            case 2:
                transpose2<<<grid_size,block_size>>>(d_A, d_B, N);
                break;
            case 3:
                transpose3<<<grid_size,block_size>>>(d_A, d_B, N);
                break;

            default:
                printf("Error!\n");
                exit(1);
                break;
        }

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
}

__global__ void transpose3(const real* A, real* B, const int N){
    //share memory

    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if(nx1 < N && ny1 < N){
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if(nx2 < N && ny2 < N){
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose1(const real* A, real* B, const int N){
    //coalesced read

    int nx = blockIdx.x*TILE_DIM + threadIdx.x;
    int ny = blockIdx.y*TILE_DIM + threadIdx.y;

    if(nx < N && ny < N){
        B[nx*N + ny] = A[ny*N + nx];
    }

}
__global__ void transpose2(const real* A, real* B, const int N){
    //coalesced write(automatic __ldg read)

    int nx = blockIdx.x*TILE_DIM + threadIdx.x;
    int ny = blockIdx.y*TILE_DIM + threadIdx.y;

    if(nx < N && ny < N){
        B[ny*N + nx] = __ldg(&A[nx*N + ny]);
    }
}

bool check_equal(const real* A, const real* B, const int N){
    bool flag = true;
    for(int ny = 0; ny < N; ny++){
        for(int nx = 0; nx < N; nx++){
            if(A[ny * N + nx] != B[nx * N + ny]){
                flag = false;
                break;
            }
        }
    }
    return flag;
}

void print_mat(const int N, const real* A){
    for(int ny = 0; ny < N; ny++){
        for(int nx = 0; nx < N; nx++){
            printf("%g\t", A[ny * N + nx]);
        }
        printf("\n");
    }
}