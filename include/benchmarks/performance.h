#pragma once

#include <benchmark/benchmark.h>
#include <benchmarks/fixtures.h>
#include <log.h>
#include <util.h>

// Benchmarks for copying data from/to the FPGA
BENCHMARK_DEFINE_F(BasicKernelFixture, VectorTriad)(benchmark::State& state)
{
  using value_t   = float;
  size_t buf_size = state.range(0);

  std::vector<value_t> A(buf_size), B(buf_size, value_t{1}),
      C(buf_size, value_t{2}), D(buf_size, value_t{3});

  cl::Buffer A_buf(ctx, CL_MEM_WRITE_ONLY, buf_size * sizeof(value_t));
  cl::Buffer B_buf(ctx, CL_MEM_READ_ONLY, buf_size * sizeof(value_t));
  cl::Buffer C_buf(ctx, CL_MEM_READ_ONLY, buf_size * sizeof(value_t));
  cl::Buffer D_buf(ctx, CL_MEM_READ_ONLY, buf_size * sizeof(value_t));

  queue.enqueueWriteBuffer(
      B_buf, CL_TRUE, 0, buf_size * sizeof(value_t), B.data(), NULL, NULL);
  queue.enqueueWriteBuffer(
      C_buf, CL_TRUE, 0, buf_size * sizeof(value_t), C.data(), NULL, NULL);
  queue.enqueueWriteBuffer(
      D_buf, CL_TRUE, 0, buf_size * sizeof(value_t), D.data(), NULL, NULL);

  auto vector_triad = kernel("vector_add_single", "vector_triad");
  unsigned long size = buf_size;

  for (auto _ : state) {
    cl::Event kernel_done;
    vector_triad.setArg(0, A_buf);
    vector_triad.setArg(1, B_buf);
    vector_triad.setArg(2, C_buf);
    vector_triad.setArg(3, D_buf);
    vector_triad.setArg(4, size);
    queue.enqueueTask(vector_triad, NULL, &kernel_done);
    kernel_done.wait();
  }

  cl::copy(queue, A_buf, A.begin(), A.end());
  std::for_each(A.begin(), A.end(), [&B, &C, &D] (auto &val) {
      assert(val == B[0] * C[0] + D[0]);
  });
}

// Benchmarks for copying data from/to the FPGA
BENCHMARK_DEFINE_F(BasicKernelFixture, VectorTriadN2)(benchmark::State& state)
{
  using value_t   = float;
  size_t buf_size = state.range(0);

  std::vector<value_t> A(buf_size), B(buf_size, value_t{1}),
      C(buf_size, value_t{2}), D(buf_size, value_t{3});

  cl::Buffer A_buf(ctx, CL_MEM_WRITE_ONLY, buf_size * sizeof(value_t));
  cl::Buffer B_buf(ctx, CL_MEM_READ_ONLY, buf_size * sizeof(value_t));
  cl::Buffer C_buf(ctx, CL_MEM_READ_ONLY, buf_size * sizeof(value_t));
  cl::Buffer D_buf(ctx, CL_MEM_READ_ONLY, buf_size * sizeof(value_t));

  queue.enqueueWriteBuffer(
      B_buf, CL_TRUE, 0, buf_size * sizeof(value_t), B.data(), NULL, NULL);
  queue.enqueueWriteBuffer(
      C_buf, CL_TRUE, 0, buf_size * sizeof(value_t), C.data(), NULL, NULL);
  queue.enqueueWriteBuffer(
      D_buf, CL_TRUE, 0, buf_size * sizeof(value_t), D.data(), NULL, NULL);

  auto vector_triad = kernel("vector_triage_n2", "vector_triad");
  unsigned long size = buf_size;

  for (auto _ : state) {
    cl::Event kernel_done;
    vector_triad.setArg(0, A_buf);
    vector_triad.setArg(1, B_buf);
    vector_triad.setArg(2, C_buf);
    vector_triad.setArg(3, D_buf);
    vector_triad.setArg(4, size);
    queue.enqueueTask(vector_triad, NULL, &kernel_done);
    kernel_done.wait();
  }

  cl::copy(queue, A_buf, A.begin(), A.end());
  std::for_each(A.begin(), A.end(), [&B, &C, &D] (auto &val) {
      assert(val == B[0] * C[0] + D[0]);
  });
}

BENCHMARK_REGISTER_F(BasicKernelFixture, VectorTriad)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 22)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BasicKernelFixture, VectorTriadN2)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 22)
    ->Unit(benchmark::kMillisecond);
