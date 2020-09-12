#pragma once

#include <CL/cl.hpp>
#include <chrono>
#include <deque>
#include <functional>

namespace forecast {

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using KernelGen = std::function<cl::Kernel(const cl::Program&)>;
class Scheduler;

class Task {
public:
  Task(uint64_t id, const std::string& function_name, KernelGen kernel_gen)
    : _id(id)
    , _function_name(function_name)
    , _kernel_gen(kernel_gen)
  {
  }

  uint64_t id() const
  {
    return _id;
  }

  cl::Kernel& kernel()
  {
    return _kernel;
  }

  cl::Kernel& generate_kernel(const cl::Program& prg)
  {
    _kernel = _kernel_gen(prg);
    return _kernel;
  }

  cl::Event& kernel_done()
  {
    return _kernel_done;
  }

  void enqueued_now()
  {
    _enqueued_at = Clock::now();
  }

  void finished_now()
  {
    _finished_at = Clock::now();
  }

  std::string function_name() const {
    return _function_name;
  }
  Scheduler* scheduler; 

private:
  uint64_t    _id;
  TimePoint   _created_at;
  TimePoint   _enqueued_at;
  TimePoint   _finished_at;
  cl::Kernel  _kernel;
  cl::Event   _kernel_done;
  std::string _function_name;
  KernelGen   _kernel_gen;
};

using Tasks = std::deque<Task>;

}  // namespace forecast
