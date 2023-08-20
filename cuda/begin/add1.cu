#include<stdio.h>
#include<math.h>

const double EPS = 1.0e-15;
const double a = 1.2;
const double b = 2.3;
const double c = 3.5;

void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(){
    const int n = 10000000;  //10^7
    const int M = sizeof(double) * n;
    double *h_x = new double[n];
    double *h_y = new double[n];
    double *h_z = new double[n];

    for(int i=0;i<n;i++){
        h_x[i] = a;
        h_y[i] = b;
    }

    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
    
    const int block_size = 128;
    const int grid_size = n / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);//cudaMemcpyDeviceToHost
    check(h_z, n);

    delete[] h_x;
    delete[] h_y;
    delete[] h_z;
    cudaFree(d_x);//cudaFree
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}

void __global__ add(const double *x, const double *y, double *z){
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    z[i] = x[i] + y[i];
}

void check(const double *z, const int n){
    bool has_err = false;
    for(int i=0; i < n; i++){
        if(fabs(z[i]-c) > EPS){
            has_err = true;
        }
    }
    printf("%s\n", has_err ? "Has_errors" : "No error");
}