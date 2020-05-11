#pragma once

#include <benchmark/benchmark.h>
#include <log.h>
#include <util.h>

static void TestFunc(benchmark::State& state)
{
  auto                    device = get_device();
  std::vector<cl::Device> devices{device};
  auto                    ctx         = cl::Context(device);
  auto                    queue       = cl::CommandQueue(ctx);
  auto                    binary = read_file("../kernels/hello_world.aocx");
  // Mhhh if only there was something that can hold a pointer and a size
  // already in the stdlib...
  auto binary_pair = std::make_pair(binary.data(), binary.size());
  cl::Program::Binaries binaries{binary_pair};

  debug("Finished initialization");


  auto program = cl::Program(ctx, devices, binaries);

  debug("Created program from binary");

  cl_int thread_id = 1;

  program.build();
  auto kernel = cl::Kernel(program, "hello_world");
  kernel.setArg(0, thread_id);

  std::vector <cl::Event> kernel_events;

  for (auto _ : state) {
    queue.enqueueTask(kernel, &kernel_events);
    queue.finish();
  }
}

BENCHMARK(TestFunc);

