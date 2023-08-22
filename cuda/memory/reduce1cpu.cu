#include "error.cuh"
#include <stdio.h>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

const int REPEATS = 10;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

int main(void)
{
    const int N = 150000000;
    const int M = sizeof(real) * N;
    real *x = (real *) malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.23;
    }

    timing(x, N);

    free(x);
    return 0;
}

void timing(const real *x, const int N)
{
    real sum = 0;

    float t_sum = 0;
    float t2_sum = 0;

    for (int rep = 0; rep <= REPEATS; ++rep)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed;
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        printf("Time = %g ms.\n", elapsed);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));

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

real reduce(const real *x, const int N)
{
    real sum = 0.0;
    for (int n = 0; n < N; ++n)
    {
        sum += x[n];
    }
    return sum;
}


