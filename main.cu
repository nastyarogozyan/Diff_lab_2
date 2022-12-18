#include<cuda.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cinttypes>
#include <fstream>
#define PI 3.14159265358979323846
 
const double delta = 0.005;
const double delta_r = 0.01;
const double eps = 1e-13;
 
const uint8_t m = 100; 
const uint16_t p = 2000;
const double g = 9.8;
 
double Ax = -0.353, Bx = 0.353, Ay = 0.3, By = Ay, C = 3 * PI / 8;
 
struct Diff_questions {
  double Ax;
  double Bx;
  double Ay;
  double By;
  double C;
  double Pi;
  double Delta;
  double Eps;
} typedef Diff ;
 
__device__ double Cal_dist(double* x0, double* x1, size_t n) {
  double sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += std::pow(x0[i] - x1[i], 2);
  }
  return std::sqrt(sum);
}
 
__device__ void Cal_F(double* result, double* input, Diff* diff) {
   __shared__ double Arr[5];
  if (threadIdx.x == 0) {
    Arr[0] = input[0] + input[2] * std::cos(3 * diff->Pi / 2 - input[3]) - diff->Ax;
  } else if (threadIdx.x == 1) {
    Arr[1] = input[1] + input[2] * std::cos(3 * diff->Pi / 2 + input[4]) - diff->Bx;
  } else if (threadIdx.x == 2) {
    Arr[2] = input[2] + input[2] * std::sin(3 * diff->Pi / 2 - input[3]) - diff->Ay;
  } else if (threadIdx.x == 3) {
    Arr[3] = (input[3] + input[4]) * input[2] + (input[1] - input[0]) - diff->C;
  } else if (threadIdx.x == 4) {
    Arr[4] = input[2] + input[2] * std::sin(3 * diff->Pi / 2 + input[4]) - diff->By;
  }
 
  __syncthreads();
  if (threadIdx.x == 0) {
    memcpy(result, Arr, sizeof(Arr));
  }
  __syncthreads();
}
 
__global__ void Cal_val(double* x0, double* x1, Diff* diff, size_t n) {
  __shared__ unsigned count;
  double* NewV = new double[5];
  while(true) {
    Cal_F(NewV, x0, diff);
    __syncthreads();
    if (threadIdx.x == 0) {
      for (size_t i = 0; i < n; ++i) {
        x1[i] = x0[i] - NewV[i] * diff->Delta;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(&count, 1);
    __syncthreads();
    if (Cal_dist(x0, x1, n) < diff->Eps) break;
    if (threadIdx.x == 0) {
      for (size_t i = 0; i < n; ++i) {
        x0[i] = x1[i];
      }
    }
    __syncthreads();
  }
  delete[] NewV ;
}
 
__host__ void print_result(double* x) {
    printf("x1 : %lf\n", x[0]);
    printf("x2 : %lf\n", x[1]);
    printf("y : %lf\n", x[2]);
    printf("phi1 : %lf\n", x[3]);
    printf("phi2 : %lf\n", x[4]);
    printf("F(x) = {%.10e, %.10e, %.10e, %.10e, %.10e}\n",
      x[0] + x[2] * std::cos(1.5 * PI - x[3]) - Ax,
      x[1] + x[2] * std::cos(1.5 * PI + x[4]) - Bx,
      x[2] + x[2] * std::sin(1.5 * PI - x[3]) - Ay,
      (x[3] + x[4]) * x[2] + (x[1] - x[0]) - C,
      x[2] + x[2] * std::sin(1.5 * PI + x[4]) - By
    );
}
 
int main() {
  Diff* diff;
  const int Questions = 5;
  double *x0, *x1;
  cudaMallocManaged(&x0, sizeof(double) * Questions );
  cudaMallocManaged(&x1, sizeof(double) * Questions );
  x0[0] = -0.1; x0[1] = 0.1; x0[2] = 0.0; x0[3] = 2.0; x0[4] = 2.0;
  x1[0] = 0.0; x1[1] = 0.0; x1[2] = 0.0; x1[3] = 0.0; x1[4] = 0.0;
  cudaMallocManaged(&diff, sizeof(Diff));
  diff->Ax = Ax, diff->Ay = Ay, 
  diff->Bx = Bx, diff->By = By, 
  diff->C = C, diff->Pi = PI, 
  diff->Delta = delta, diff->Eps = eps;
 
  double Vs = 0;
  int Blocks_ = 1;
  int Threads_Blocks = 5;
  for (double t = 0; t <= 2.5; t += delta_r) {
    Cal_val<<<Blocks_ , Threads_Blocks >>>(x0, x1, diff, Questions);
    cudaDeviceSynchronize();
    print_result(x0);
    Vs += (p * (x1[1] - x1[0]) - m * g) * delta_r/ m;
    diff->Ay += Vs * delta_r;
    diff->By = diff->Ay;
  }
  cudaFree(&x0);
  cudaFree(&x1);
  return 0;
}