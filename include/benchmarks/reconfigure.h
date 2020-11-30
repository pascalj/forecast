#pragma once

#include <benchmark/benchmark.h>
#include <benchmarks/fixtures.h>
#include <log.h>
#include <util.h>

// Benchmarks for the reconfiguration process


BENCHMARK_DEFINE_F(BasicKernelFixture, SameKernel)(benchmark::State& state)
{
  auto hello_world_kernel = kernel("hello_world", "hello_world");
  auto& queue = clstate.queue;

  cl_int thread_id = 1;

  for(auto _ : state) {
    state.PauseTiming();
    {
      cl::Event kernel_event;
      hello_world_kernel.setArg(0, thread_id);
      queue.enqueueTask(hello_world_kernel, NULL, &kernel_event);
      kernel_event.wait();
    }
    state.ResumeTiming();
    {
      cl::Event kernel_event;
      hello_world_kernel.setArg(0, thread_id);
      queue.enqueueTask(hello_world_kernel, NULL, &kernel_event);
      kernel_event.wait();
    }
  }
}


BENCHMARK_DEFINE_F(BasicKernelFixture, SameProgram)(benchmark::State& state)
{

  auto empty1_kernel = kernel("multi_empty", "empty1");
  auto empty2_kernel = kernel("multi_empty", "empty2");
  auto& queue = clstate.queue;

  for(auto _ : state) {
    state.PauseTiming();
    {
      cl::Event kernel_event;
      queue.enqueueTask(empty1_kernel, NULL, &kernel_event);
      kernel_event.wait();
    }
    state.ResumeTiming();
    {
      cl::Event kernel_event;
      queue.enqueueTask(empty2_kernel, NULL, &kernel_event);
      kernel_event.wait();
    }
  }
}

BENCHMARK_DEFINE_F(BasicKernelFixture, ReconfigureEmpty)(benchmark::State& state)
{

  auto empty_kernel = kernel("empty", "empty");
  auto hello_world_kernel = kernel("hello_world", "hello_world");
  auto& queue = clstate.queue;

  cl_int thread_id = 1;
  hello_world_kernel.setArg(0, thread_id);

  for(auto _ : state) {
    state.PauseTiming();
    {
      cl::Event kernel_event;
      queue.enqueueTask(hello_world_kernel, NULL, &kernel_event);
      kernel_event.wait();
    }
    state.ResumeTiming();
    {
      cl::Event kernel_event;
      queue.enqueueTask(empty_kernel, NULL, &kernel_event);
      kernel_event.wait();
    }
  }
}

BENCHMARK_DEFINE_F(BasicKernelFixture, RunSerial)(benchmark::State& state)
{

  auto empty_kernel = kernel("empty", "empty");
  auto hello_world_kernel = kernel("hello_world", "hello_world");
  auto& queue = clstate.queue;

  cl_int thread_id = 1;
  hello_world_kernel.setArg(0, thread_id);
  for(auto _ : state) {
    cl::Event kernel_event1;
    cl::Event kernel_event2;
    queue.enqueueTask(hello_world_kernel, NULL, &kernel_event1);
    std::vector<cl::Event> wait_list{kernel_event1};
    queue.enqueueTask(empty_kernel, NULL, &kernel_event2);
    kernel_event2.wait();
  }
}

BENCHMARK_DEFINE_F(BasicKernelFixture, RunParallel)(benchmark::State& state)
{

  auto empty_kernel = kernel("empty", "empty");
  auto hello_world_kernel = kernel("hello_world", "hello_world");
  auto& queue = clstate.queue;
  auto& ctx = clstate.ctx;

  cl_int thread_id = 1;
  hello_world_kernel.setArg(0, thread_id);
  cl::CommandQueue second_queue(ctx);

  for(auto _ : state) {
    cl::Event kernel_event1;
    cl::Event kernel_event2;
    queue.enqueueTask(hello_world_kernel, NULL, &kernel_event1);
    second_queue.enqueueTask(empty_kernel, NULL, &kernel_event2);
    kernel_event1.wait();
    kernel_event2.wait();
  }
}

BENCHMARK_REGISTER_F(BasicKernelFixture, SameKernel)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BasicKernelFixture, SameProgram)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BasicKernelFixture, ReconfigureEmpty)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BasicKernelFixture, RunSerial)->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(BasicKernelFixture, RunParallel)->Unit(benchmark::kMillisecond);
