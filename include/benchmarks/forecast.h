#pragma once

#include <benchmark/benchmark.h>
#include <benchmarks/fixtures.h>
#include <log.h>
#include <util.h>
#include <forecast/queue.h>

// Benchmark overhead of forecast
BENCHMARK_DEFINE_F(BasicKernelFixture, ForecastTriad)(benchmark::State& state)
{
  using value_t   = float;
  size_t buf_size = state.range(0);
  forecast::Queue fqueue(queue);

  Buffers<4, value_t> buffers(ctx, buf_size);
  buffers.fill_all(queue, {0, 2, 3, 4});

  auto vector_triad = kernel("vector_triad_n1", "vector_triad1");
	queue.finish();

  for (auto _ : state) {
    cl::Event kernel_done;
    set_bufs_as_args(vector_triad, buffers);
    vector_triad.setArg(4, static_cast<unsigned long>(buf_size));
    fqueue.enqueue(vector_triad, true);
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
