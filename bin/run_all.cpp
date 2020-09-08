#include <log.h>
#include <util.h>
#include <benchmark/benchmark.h>

#include <benchmarks/copy.h>
#include <benchmarks/performance.h>
#include <benchmarks/reconfigure.h>
#include <benchmarks/forecast.h>

int main(int argc, char **argv) {
  spdlog::set_pattern("[%H:%M:%S] [%^%L%$] %v");
  spdlog::set_level(spdlog::level::warn);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
