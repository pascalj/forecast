#include <log.h>
#include <util.h>
#include <benchmark/benchmark.h>

#include <benchmarks/test.h>

int main(int argc, char **argv) {
  spdlog::set_pattern("[%H:%M:%S] [%^%L%$] %v");
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
