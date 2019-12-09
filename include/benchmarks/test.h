#pragma once

#include <benchmark/benchmark.h>
#include <log.h>
#include <util.h>

static void TestFunc(benchmark::State& state)
{
  auto                 ctx         = get_device_context();
  auto                 queue       = get_command_queue(ctx);
  auto                 device      = get_device(ctx);
  auto                 binary      = read_file("../kernels/hello_world.aocx");
  const unsigned char* binary_data = binary.data();
  auto                 binary_size = binary.size();
  cl_int               err;

  auto program = clCreateProgramWithBinary(
      ctx, 1, &device, &binary_size, &binary_data, NULL, &err);
  cl_ok(err);

  cl_int thread_id = 1;

  cl_assert(clBuildProgram(program, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel = clCreateKernel(program, "hello_world", &err);
  cl_ok(err);
  cl_assert(clSetKernelArg(kernel, 0, sizeof(cl_int), &thread_id));

  cl_event kernel_done;
  for (auto _ : state) {
    cl_assert(clEnqueueTask(queue, kernel, 0, NULL, &kernel_done));
    clFinish(queue);
  }
}

BENCHMARK(TestFunc);

