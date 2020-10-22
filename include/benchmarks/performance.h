#pragma once

#include <benchmark/benchmark.h>
#include <benchmarks/fixtures.h>
#include <log.h>
#include <util.h>
#include <forecast/queue.h>

// Benchmarks for copying data from/to the FPGA
BENCHMARK_DEFINE_F(BasicKernelFixture, VectorTriad)(benchmark::State& state)
{
  using value_t   = float;
  size_t buf_size = state.range(0);

  Buffers<4, value_t> buffers(ctx, buf_size);
  buffers.fill_all(queue, {0, 2, 3, 4});

  auto vector_triad = kernel("vector_triad_n1", "vector_triad1");
	queue.finish();

  for (auto _ : state) {
    cl::Event kernel_done;
    set_bufs_as_args(vector_triad, buffers);
    vector_triad.setArg(4, static_cast<unsigned long>(buf_size));
    queue.enqueueTask(vector_triad, NULL, &kernel_done);
    kernel_done.wait();
  }

  const bool valid = buffers[0].validate(
      queue, [](const auto& val) { return val == 2 * 3 + 4; });

  if(!valid) {
    state.SkipWithError("Validation failed.");
  }
}

// Benchmarks for copying data from/to the FPGA
BENCHMARK_DEFINE_F(BasicKernelFixture, VectorTriadParallel)(benchmark::State& state)
{
  using value_t            = float;
  const size_t buf_size    = state.range(0);
  const size_t parallelism = state.range(1);

  std::vector<Buffers<4, value_t>> buffers;
  buffers.reserve(parallelism);
  for (size_t i = 0; i < parallelism; i++) {
    buffers.emplace_back(ctx, buf_size);
  }

  for(auto& buffer : buffers) {
    buffer.fill_all(queue, {0, 1, 2, 3});
  }

  std::vector<cl::Kernel> kernels;
  std::vector<cl::CommandQueue> queues;
  kernels.reserve(parallelism);
  queues.reserve(parallelism);
  std::stringstream file_stream;
  file_stream << "vector_triad_n" << parallelism;
  for (size_t kern = 1; kern <= parallelism; kern++) {
    std::stringstream ss;
    ss << "vector_triad" << kern;
    kernels.push_back(kernel(file_stream.str(), ss.str()));
    queues.emplace_back(ctx);
  }
  unsigned long size = buf_size;

  buffers.reserve(parallelism);
  for (auto _ : state) {
    std::vector<cl::Event> kernels_done;
    kernels_done.reserve(parallelism);
    for (size_t kern = 0; kern < parallelism; kern++) {
      auto& kernel = kernels[kern];
      auto& buf = buffers[kern];
      set_bufs_as_args(kernel, buf);
      kernel.setArg(4, size);
      kernels_done.emplace_back();
      queues[kern].enqueueTask(
          kernel, NULL, std::addressof(kernels_done.back()));
    }
    cl::Event::waitForEvents(kernels_done);
  }

  state.SetBytesProcessed(
      size_t(4 * state.iterations()) * size_t(state.range(0)) *
      size_t(state.range(1)) * sizeof(value_t));

  for(auto& buffer : buffers) {
    const bool valid = buffer[0].validate(queue, [](const auto& val){
        return val == 1 * 2 + 3;
    });
    if (!valid) {
      state.SkipWithError("Validation failed.");
    }
  }
}

/**
 * Simple matrix multiplication
 *
 * Note that block_size must match the one in the cl file.
 */
BENCHMARK_DEFINE_F(BasicKernelFixture, MatrixMult)(benchmark::State& state)
{
  using value_t            = float;
  const size_t  N          = state.range(0);
  constexpr int block_size = 64;  // must match .cl file

  assert(N % block_size == 0);

  Buffers<3, value_t> buffers(ctx, N * N);
  buffers.fill_all(queue, {0, 2, 3});

  auto kern = kernel("mmult_f_d2", "matrixMult");

  forecast::Model model("mmult_f_d2");

  const cl::NDRange global_work_size(N, N);
  const cl::NDRange local_work_size(block_size, block_size);
  forecast::Task    task{
      "matrixMult",
      forecast::KernelGen{},
      forecast::TaskDims{global_work_size, local_work_size}};

  warn("Cost: {} s", model.cost(task));

  for (auto _ : state) {
    cl::Event kernel_done;
    set_bufs_as_args(kern, buffers);
    kern.setArg(3, static_cast<int>(N));
    kern.setArg(4, static_cast<int>(N));
    queue.enqueueNDRangeKernel(
        kern,
        cl::NullRange,
        global_work_size,
        local_work_size,
        NULL,
        std::addressof(kernel_done));
    kernel_done.wait();
  }

  const unsigned long long flops = N * N * N * 2 * state.iterations();
  state.counters["FLOPs"] =
      benchmark::Counter(flops, benchmark::Counter::kIsRate);

  bool valid = buffers[0].validate(queue, [N] (const auto& val) {
    return val == 6 * N;
  });
  if (!valid) {
    state.SkipWithError("Validation failed.");
  }
}

/**
 * Combined, parallel 
 *
 * Note that block_size must match the one in the cl file.
 */
BENCHMARK_DEFINE_F(BasicKernelFixture, MatrixMultTriad)(benchmark::State& state)
{
  using value_t            = float;
  const size_t  N          = state.range(0);
  const size_t  triad_size = state.range(1);
  constexpr int block_size = 64;  // must match .cl file

  assert(N % block_size == 0);

  Buffers<3, value_t> mbuf(ctx, N * N);
  Buffers<4, value_t> tbuf(ctx, triad_size);
  mbuf.fill_all(queue, {0, 2, 3});
  tbuf.fill_all(queue, {0, 2, 3, 4});

  auto mmult = kernel("matrix_mult_triad", "matrixMult");
  auto triad = kernel("matrix_mult_triad", "vector_triad");

  const cl::NDRange global_work_size(N, N);
  const cl::NDRange local_work_size(block_size, block_size);
  cl::CommandQueue other_queue(ctx);

  std::chrono::duration<double, std::milli> triad_duration{0};

  for (auto _ : state) {
    cl::Event mmult_done, triad_done;
    set_bufs_as_args(mmult, mbuf);
    set_bufs_as_args(triad, tbuf);
    mmult.setArg(3, static_cast<int>(N));
    mmult.setArg(4, static_cast<int>(N));
    triad.setArg(4, triad_size);
    queue.enqueueNDRangeKernel(
        mmult,
        cl::NullRange,
        global_work_size,
        local_work_size,
        NULL,
        std::addressof(mmult_done));
    other_queue.enqueueTask(triad, NULL, std::addressof(triad_done));
    auto start = std::chrono::high_resolution_clock::now();
    triad_done.wait();
    auto end = std::chrono::high_resolution_clock::now();
    triad_duration += end - start;
    mmult_done.wait();
  }

  const unsigned long long flops = N * N * N * 2 * state.iterations();
  state.counters["FLOPs"] =
      benchmark::Counter(flops, benchmark::Counter::kIsRate);
  state.counters["triad_Time"] =
      benchmark::Counter(triad_duration.count() / state.iterations());

  const auto valid = mbuf[0].validate(
      queue, [N](const auto& val) { return val == 2 * 3 * N; });
  if(!valid) {
    state.SkipWithError("Validation failed.");
  }
}

static void ParallelismRange(benchmark::internal::Benchmark* b)
{
  const int from_size = 1 << 5;
  const int to_size = 1 << 22;
  for (int j = 1; j <= 4; j *= 2)
    for (int i = from_size; i <= to_size; i *= 2) b->Args({i, j});
}

static void MatrixTriadRanges(benchmark::internal::Benchmark* b)
{
  const int matrix_from = 1024; // ~10ms
  const int matrix_to = 8192; // 5000ms
  const int triad_from = 1 << 21; // ~8ms
  const int triad_to = 1 << 26; // ~600ms @ 64MB
  for (int j = matrix_from; j <= matrix_to; j *= 2) b->Args({j, 1});
  for (int i = triad_from; i <= triad_to; i *= 2) b->Args({64, i});
  for (int i = triad_from; i <= triad_to; i *= 2)
    for (int j = matrix_from; j <= matrix_to; j *= 2) b->Args({j, i});
}

/**
 * Combined, parallel, queued
 *
 * Note that block_size must match the one in the cl file.
 */
BENCHMARK_DEFINE_F(BasicKernelFixture, MatrixMultTriadQueue)(benchmark::State& state)
{
  using value_t            = float;
  const size_t  N          = state.range(0);
  const size_t  triad_size = state.range(1);
  constexpr int block_size = 64;  // must match .cl file

  assert(N % block_size == 0);

  Buffers<3, value_t> mbuf(ctx, N * N);
  Buffers<4, value_t> tbuf(ctx, triad_size);
  mbuf.fill_all(queue, {0, 2, 3});
  tbuf.fill_all(queue, {0, 2, 3, 4});

  auto mmult = kernel("matrix_mult_triad", "matrixMult");
  auto triad = kernel("matrix_mult_triad", "vector_triad");

  const cl::NDRange global_work_size(N, N);
  const cl::NDRange local_work_size(block_size, block_size);
  cl::CommandQueue other_queue(ctx);

  std::chrono::duration<double, std::milli> triad_duration{0};

  for (auto _ : state) {
    cl::Event mmult_done, triad_done;
    set_bufs_as_args(mmult, mbuf);
    set_bufs_as_args(triad, tbuf);
    mmult.setArg(3, static_cast<int>(N));
    mmult.setArg(4, static_cast<int>(N));
    triad.setArg(4, triad_size);
    queue.enqueueNDRangeKernel(
        mmult,
        cl::NullRange,
        global_work_size,
        local_work_size,
        NULL,
        std::addressof(mmult_done));
    other_queue.enqueueTask(triad, NULL, std::addressof(triad_done));
    auto start = std::chrono::high_resolution_clock::now();
    triad_done.wait();
    auto end = std::chrono::high_resolution_clock::now();
    triad_duration += end - start;
    mmult_done.wait();
  }

  const unsigned long long flops = N * N * N * 2 * state.iterations();
  state.counters["FLOPs"] =
      benchmark::Counter(flops, benchmark::Counter::kIsRate);
  state.counters["triad_Time"] =
      benchmark::Counter(triad_duration.count() / state.iterations());

  const auto valid = mbuf[0].validate(
      queue, [N](const auto& val) { return val == 2 * 3 * N; });
  if(!valid) {
    state.SkipWithError("Validation failed.");
  }
}

BENCHMARK_REGISTER_F(BasicKernelFixture, VectorTriad)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 22)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BasicKernelFixture, VectorTriadParallel)
    ->Apply(ParallelismRange)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BasicKernelFixture, MatrixMult)
    ->RangeMultiplier(2)
    ->Range(64, 64 << 7)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_REGISTER_F(BasicKernelFixture, MatrixMultTriad)
    ->Apply(MatrixTriadRanges)
    ->Unit(benchmark::kMillisecond);
