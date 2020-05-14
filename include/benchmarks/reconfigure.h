#pragma once

// Benchmarks for the reconfiguration process

#include <benchmark/benchmark.h>
#include <log.h>
#include <util.h>


static void FlashKernelBinary(benchmark::State& state)
{
  auto devices = get_devices();
  auto ctx     = cl::Context(devices.front());
  auto queue   = cl::CommandQueue(ctx);
  auto binary  = Binary("../kernels/hello_world.aocx");
  auto program = cl::Program(ctx, devices, binary.cl_binaries());
  program.build(); // no op

  cl_int thread_id = 1;
  auto kernel = cl::Kernel(program, "hello_world");
  kernel.setArg(0, thread_id);

  for(auto _ : state) {
    cl::Event kernel_event;
    queue.enqueueTask(kernel, NULL, &kernel_event);
    kernel_event.wait();
  }

  queue.finish();
}

BENCHMARK(FlashKernelBinary);
