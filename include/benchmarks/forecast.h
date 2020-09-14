#pragma once

#include <benchmark/benchmark.h>
#include <benchmarks/fixtures.h>
#include <log.h>
#include <util.h>
#include <forecast/scheduler.h>

// Benchmark overhead of forecast
BENCHMARK_DEFINE_F(BasicKernelFixture, ForecastTriad)(benchmark::State& state)
{
  using value_t   = float;
  size_t buf_size = state.range(0);

  Buffers<4, value_t> buffers(ctx, buf_size);
  buffers.fill_all(queue, {0, 2, 3, 4});

  scheduler.add_config("vector_triad_n1");

  forecast::KernelGen create_kernel = [&buffers,buf_size](const cl::Program &prg) {
    int err = 0;
    cl::Kernel kernel(prg, "vector_triad1", &err);
    cl_ok(err);
    kernel.setArg(4, static_cast<unsigned long>(buf_size));
    set_bufs_as_args(kernel, buffers);
    return kernel;
  };


  for (auto _ : state) {
    for(int i = 0; i < 20; i++) {
      scheduler.add_task("vector_triad1", create_kernel);
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

BENCHMARK_REGISTER_F(BasicKernelFixture, ForecastTriad)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 22)
    ->Unit(benchmark::kMillisecond);
