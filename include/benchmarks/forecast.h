#pragma once

#include <benchmark/benchmark.h>
#include <benchmarks/fixtures.h>
#include <log.h>
#include <util.h>
#include <forecast/scheduler.h>

// Benchmark overhead of forecast
BENCHMARK_DEFINE_F(ForecastFixture, Triad)(benchmark::State& state)
{
  using value_t   = float;
  size_t buf_size = state.range(0);

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
    for (int i = 0; i < static_cast<int>(N) / 64; i++) {
      const cl::NDRange global_work_size(
          N - 64 * i, N - 64 * i);
      scheduler.add_task(forecast::Task{
          "matrixMult",
          create_mmult,
          forecast::TaskDims{global_work_size, local_work_size}});
    }
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

BENCHMARK_REGISTER_F(ForecastFixture, Triad)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 22)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(ForecastFixture, Mmult)
    ->RangeMultiplier(2)
    ->Range(64, 64 << 7)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
