#pragma once

#include <CL/cl.hpp>
#include <chrono>
#include <deque>
#include <functional>
#include "spdlog/fmt/ostr.h"

namespace forecast {

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using KernelGen = std::function<cl::Kernel(const cl::Program&, const std::string&)>;
class Scheduler;

struct TaskDims {
  TaskDims()
    : global(1)
    , local(1)
  {
  }

  TaskDims(cl::NDRange g, cl::NDRange l)
    : global(g)
    , local(l)
  {
  }

  template <typename OStream>
  friend OStream& operator<<(OStream& os, const TaskDims& c)
  {
    return os << "<" << c.global[0] << "," << c.global[1] << ","
              << c.global[2] << "><" << c.local[0] << "," << c.local[1]
              << "," << c.local[2] << ">";
  }

  cl::NDRange global;
  cl::NDRange local;
  cl::NDRange offset = cl::NullRange;
};

class Task {
public:
  Task(
      const std::string& function_name,
      KernelGen          kernel_gen,
      TaskDims           dims = TaskDims())
    : _created_at(Clock::now()) // we construct the task in-place
    , _function_name(function_name)
    , _kernel_gen(kernel_gen)
    , _dims(dims)
  {
  }

  uint64_t id() const
  {
    return _id;
  }

  void set_id(uint64_t new_id)
  {
    _id = new_id;
  }

  cl::Kernel& kernel()
  {
    return _kernel;
  }

  cl::Kernel& generate_kernel(const cl::Program& prg)
  {
    _kernel = _kernel_gen(prg, _function_name);
    return _kernel;
  }

  std::string kernel_name() const {
    return _kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
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

  cl::NDRange offset() const {
    return _dims.offset;
  }

  void set_dims(TaskDims dims) {
    _dims = dims;
  }

  cl::NDRange global() const {
    return _dims.global;
  }

  cl::NDRange local() const {
    return _dims.local;
  }

  std::chrono::duration<double> duration() const {
    return _finished_at - _enqueued_at;
  }

  template <typename OStream>
  friend OStream& operator<<(OStream& os, const Task& t)
  {
    return os << t.function_name() << t._dims;
  }

private:
  uint64_t    _id;
  TimePoint   _created_at;
  TimePoint   _enqueued_at;
  TimePoint   _finished_at;
  cl::Kernel  _kernel;
  cl::Event   _kernel_done;
  std::string _function_name;
  KernelGen   _kernel_gen;
  TaskDims    _dims;
};

using Tasks = std::deque<Task>;

}  // namespace forecast
