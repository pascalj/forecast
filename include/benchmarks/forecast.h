#pragma once

#include <benchmark/benchmark.h>
#include <benchmarks/fixtures.h>
#include <log.h>
#include <util.h>
#include <forecast/scheduler.h>
#include <benchmarks/fft.h>

// Benchmark overhead of forecast
BENCHMARK_DEFINE_F(ForecastFixture, Triad)(benchmark::State& state)
{
  using value_t   = float;
  size_t buf_size = state.range(0);
  auto& queue = clstate.queue;
  auto& ctx = clstate.ctx;

  Buffers<4, value_t> buffers(ctx, buf_size);
  buffers.fill_all(queue, {0, 2, 3, 4});

  scheduler.add_config("vector_triad_n2");

  forecast::KernelGen create_kernel = [&buffers,buf_size](const cl::Program &prg, const std::string &kernel_name) {
    int err = 0;
    cl::Kernel kernel(prg, kernel_name.c_str(), &err);
    cl_ok(err);
    kernel.setArg(4, static_cast<unsigned long>(buf_size));
    set_bufs_as_args(kernel, buffers);
    return kernel;
  };


  for (auto _ : state) {
    for(int i = 0; i < 10; i++) {
      scheduler.add_task(forecast::Task("vector_triad1", create_kernel));
      scheduler.add_task(forecast::Task("vector_triad2", create_kernel));
    }
    scheduler.wait();
    warn("done");
  }


  const bool valid = buffers[0].validate(
      queue, [](const auto& val) { return val == 2 * 3 + 4; });

  if(!valid) {
    state.SkipWithError("Validation failed.");
  }
}

BENCHMARK_DEFINE_F(ForecastFixture, Mmult)(benchmark::State& state)
{
  using value_t            = float;
  const size_t  N          = state.range(0);
  constexpr int block_size = 64;  // must match .cl file
  auto& queue = clstate.queue;
  auto& ctx = clstate.ctx;

  assert(N % block_size == 0);

  Buffers<3, value_t> buffers(ctx, N * N);
  buffers.fill_all(queue, {0, 2, 3});

  scheduler.add_config("mmult_f_d2");
  scheduler.add_config("mmult_f_d");


  forecast::KernelGen create_mmult =
      [&buffers, N](const cl::Program& prg, const std::string& kernel_name) {
        int        err = 0;
        cl::Kernel kernel(prg, kernel_name.c_str(), &err);
        set_bufs_as_args(kernel, buffers);
        kernel.setArg(3, static_cast<int>(N));
        kernel.setArg(4, static_cast<int>(N));
        return kernel;
      };

  const cl::NDRange local_work_size(block_size, block_size);
  for (auto _ : state) {
    for (int a = 0; a < 10; a++) {
      const cl::NDRange global_work_size(N, N);
      scheduler.add_task(forecast::Task{
          "matrixMult",
          create_mmult,
          forecast::TaskDims{global_work_size, local_work_size}});
    }
    scheduler.wait();
  }

  const unsigned long long flops = N * N * N * 2 * state.iterations() * 10;
  state.counters["FLOPs"] =
      benchmark::Counter(flops, benchmark::Counter::kIsRate);

  bool valid = buffers[0].validate(
      queue, [N](const auto& val) { return val == 6 * N; });
  if (!valid) {
    state.SkipWithError("Validation failed.");
  }
}

BENCHMARK_DEFINE_F(ForecastFixture, MmultRandom)(benchmark::State& state)
{
  const size_t  N          = 4096;
  constexpr int block_size = 64;  // must match .cl file
  auto&         queue      = clstate.queue;
  auto&         ctx        = clstate.ctx;

  assert(N % block_size == 0);

  Buffers<3, float> f_buffers(ctx, N * N);
  Buffers<3, double> d_buffers(ctx, N * N);
  f_buffers.fill_all(queue, {0, 2, 3});
  d_buffers.fill_all(queue, {0, 2, 3});

  scheduler.add_config("mmult_f_d2");
  scheduler.add_config("mmult_f_d");


  forecast::KernelGen create_mmult_f =
      [&f_buffers, N](const cl::Program& prg, const std::string& kernel_name) {
        int        err = 0;
        cl::Kernel kernel(prg, kernel_name.c_str(), &err);
        set_bufs_as_args(kernel, f_buffers);
        kernel.setArg(3, static_cast<int>(N));
        kernel.setArg(4, static_cast<int>(N));
        return kernel;
      };

  forecast::KernelGen create_mmult_d =
      [&d_buffers, N](const cl::Program& prg, const std::string& kernel_name) {
        int        err = 0;
        cl::Kernel kernel(prg, kernel_name.c_str(), &err);
        set_bufs_as_args(kernel, d_buffers);
        kernel.setArg(3, static_cast<int>(N));
        kernel.setArg(4, static_cast<int>(N));
        return kernel;
      };


  RandomTasks random_tasks;
  const cl::NDRange global_work_size(N, N);
  const cl::NDRange local_work_size(block_size, block_size);
  auto size_gen = [global_work_size, local_work_size]() {
    return forecast::TaskDims{global_work_size, local_work_size};
  };
  random_tasks.add_kernel(0.6f, forecast::Task("matrixMult", create_mmult_f), size_gen);
  random_tasks.add_kernel(0.4f, forecast::Task("matrixMultD", create_mmult_d), size_gen);

  for (auto _ : state) {
    for (int a = 0; a < 10; a++) {
      scheduler.add_task(random_tasks.next_task());
    }
    scheduler.wait();
  }

  const unsigned long long flops = N * N * N * 2 * state.iterations() * 10;
  state.counters["FLOPs"] =
      benchmark::Counter(flops, benchmark::Counter::kIsRate);

  bool valid = f_buffers[0].validate(
      queue, [N](const auto& val) { return val == 6 * N; });
  if (!valid) {
    state.SkipWithError("Validation failed.");
  }
}

BENCHMARK_DEFINE_F(ForecastFixture, FFT1D)(benchmark::State& state)
{
  const size_t  fft_iterations = state.range(0);
  constexpr bool inverse        = false;
  auto& queue = clstate.queue;
  auto& ctx = clstate.ctx;

  float2 *h_inData, *h_outData;
  double2 *h_verify;
  h_inData = (float2 *)aligned_alloc(64, sizeof(float2) * N * fft_iterations);
  h_outData = (float2 *)aligned_alloc(64, sizeof(float2) * N * fft_iterations);
  h_verify = (double2 *)aligned_alloc(64, sizeof(double2) * N * fft_iterations);
  if (!(h_inData && h_outData && h_verify)) {
    state.SkipWithError( "ERROR: Couldn't create host buffers\n");
    return;
  }

  assert(fft_iterations <= std::numeric_limits<int>().max());
  for (int i = 0; i < static_cast<int>(fft_iterations); i++) {
    for (int j = 0; j < N; j++) {
      h_verify[coord(i, j)].x = h_inData[coord(i, j)].x = (float)((double)rand() / (double)RAND_MAX);
      h_verify[coord(i, j)].y = h_inData[coord(i, j)].y = (float)((double)rand() / (double)RAND_MAX);
    }
  }

  cl_int status;
  auto d_inData = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(float2) * N * fft_iterations, NULL, &status);
  cl_ok(status);
  auto d_outData = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(float2) * N * fft_iterations, NULL, &status);
  cl_ok(status);
  cl_ok(queue.enqueueWriteBuffer(d_inData, CL_TRUE, 0, sizeof(float2) * N * fft_iterations, h_inData));
  int inverse_int = inverse;

  scheduler.add_config("fft1d");

  forecast::KernelGen create_fetch =
      [&d_inData](const cl::Program &prg, const std::string &kernel_name) {
        int        err = 0;
        cl::Kernel kernel(prg, kernel_name.c_str(), &err);
        cl_ok(err);
        kernel.setArg(0, d_inData);
        return kernel;
      };

  forecast::KernelGen create_fft1d = [&d_outData, &fft_iterations, &inverse_int](
                                         const cl::Program &prg,
                                         const std::string &kernel_name) {
    int err = 0;
    cl::Kernel kernel(prg, kernel_name.c_str(), &err);
    cl_ok(err);
    cl_ok(kernel.setArg(0, d_outData));
    cl_ok(kernel.setArg(1, fft_iterations));
    cl_ok(kernel.setArg(2, inverse_int));
    return kernel;
  };

  for(auto _ : state) {
    // Launch the kernel - we launch a single work item hence enqueue a task
    auto ls = cl::NDRange{N/8};
    auto gs = cl::NDRange{fft_iterations * ls[0]};
    scheduler.add_task(forecast::Task{"fetch", create_fetch, forecast::TaskDims{}});
    scheduler.add_task(forecast::Task{"fft1d", create_fft1d, forecast::TaskDims{gs, ls}});

    scheduler.finish();
  }
  
  // Copy results from device to host
  cl_ok(queue.enqueueReadBuffer(d_outData, CL_TRUE, 0, sizeof(float2) * N * fft_iterations, h_outData));

  // TODO: check
  const double gflop = 5 * N * (log((float)N) / log((float)2)) *
                       fft_iterations * state.iterations();

  // Pick randomly a few iterations and check SNR
  double fpga_snr = 200;
  for (size_t i = 0; i < fft_iterations; i+= rand() % 20 + 1) {
    fourier_transform_gold(inverse, 12, h_verify + coord(i, 0));
    double mag_sum = 0;
    double noise_sum = 0;
    for (int j = 0; j < N; j++) {
      double magnitude =
          (double)h_verify[coord(i, j)].x * (double)h_verify[coord(i, j)].x +
          (double)h_verify[coord(i, j)].y * (double)h_verify[coord(i, j)].y;
      double noise =
          (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) *
              (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) +
          (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y) *
              (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y);

      mag_sum += magnitude;
      noise_sum += noise;
    }
    double db = 10 * log(mag_sum / noise_sum) / log(10.0);
    // find minimum SNR across all iterations
    if (db < fpga_snr) fpga_snr = db;
  }
  if(fpga_snr <= 125) {
    state.SkipWithError("Validation failed, SNR too high.");
  } else {
    debug("Signal to noise ratio on output sample: {} --> {}", fpga_snr, fpga_snr > 125 ? "PASSED" : "FAILED");
  }
  state.counters["FLOPs"] =
      benchmark::Counter(gflop, benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(ForecastFixture, Triad)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 22)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(ForecastFixture, Mmult)
    ->RangeMultiplier(2)
    ->Range(64, 64 << 7)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_REGISTER_F(ForecastFixture, MmultRandom)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
BENCHMARK_REGISTER_F(ForecastFixture, FFT1D)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 22)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
