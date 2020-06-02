#ifndef NUM_KERNELS
#define NUM_KERNELS 1
#endif

 __kernel void vector_triad1(
    __global float* restrict A,
    __global float* restrict B,
    __global float* restrict C,
    __global float* restrict D,
    unsigned long            size)
{
  for (unsigned long i = 0; i < size; i++) {
    A[i] = B[i] * C[i] + D[i];
  }
}


#if NUM_KERNELS > 1
 __kernel void vector_triad2(
    __global float* restrict A,
    __global float* restrict B,
    __global float* restrict C,
    __global float* restrict D,
    unsigned long            size)
{
  for (unsigned long i = 0; i < size; i++) {
    A[i] = B[i] * C[i] + D[i];
  }
}
#endif

#if NUM_KERNELS > 2
 __kernel void vector_triad3(
    __global float* restrict A,
    __global float* restrict B,
    __global float* restrict C,
    __global float* restrict D,
    unsigned long            size)
{
  for (unsigned long i = 0; i < size; i++) {
    A[i] = B[i] * C[i] + D[i];
  }
}
#endif

#if NUM_KERNELS > 3
 __kernel void vector_triad4(
    __global float* restrict A,
    __global float* restrict B,
    __global float* restrict C,
    __global float* restrict D,
    unsigned long            size)
{
  for (unsigned long i = 0; i < size; i++) {
    A[i] = B[i] * C[i] + D[i];
  }
}
#endif
