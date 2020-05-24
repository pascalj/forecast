__attribute__((num_compute_units(2))) __kernel void vector_triad(
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
