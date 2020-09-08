#pragma once

#include <CL/cl.hpp>
#include <chrono>

namespace forecast {

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

class Task {
public:
  Task(uint64_t id, const std::string& function_name)
    : _id(id)
    , _function_name(function_name)
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

private:
  uint64_t    _id;
  TimePoint   _created_at;
  TimePoint   _enqueued_at;
  TimePoint   _finished_at;
  cl::Kernel  _kernel;
  cl::Event   _kernel_done;
  std::string _function_name;
};

}  // namespace forecast
