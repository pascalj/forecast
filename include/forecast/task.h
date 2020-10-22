#pragma once

#include <CL/cl.hpp>
#include <chrono>
#include <deque>
#include <functional>

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
    : _function_name(function_name)
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

  const Scheduler& scheduler() {
    return *_scheduler;
  }

  void set_scheduler(Scheduler *new_sched) {
    _scheduler = new_sched;
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

  cl::NDRange global() const {
    return _dims.global;
  }

  cl::NDRange local() const {
    return _dims.local;
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
  Scheduler*  _scheduler;
};

using Tasks = std::deque<Task>;

}  // namespace forecast
