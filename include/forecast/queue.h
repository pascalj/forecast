#pragma once

#include "model.h"

#include <CL/cl.hpp>
#include <map>
#include <set>
#include <util.h>

namespace forecast {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

class Queue {
  class Task {
  public:
    Task(uint64_t id, cl::Kernel&& kernel)
      : _id(id)
      , _kernel(std::move(kernel))
    {}

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

    void enqueued_now() {
      _enqueued_at = Clock::now();
    }

  private:
    uint64_t   _id;
    TimePoint  _created_at;
    TimePoint  _enqueued_at;
    TimePoint  _finished_at;
    cl::Kernel _kernel;
    cl::Event  _kernel_done;
  };

public:
  Queue(const cl::CommandQueue& command_queue) : _command_queue(command_queue) {}
  Queue(cl::CommandQueue&& command_queue) = delete;

  Task& enqueue(cl::Kernel&& kernel) {
    const auto task_id = _current_id++;
    auto       it      = _tasks.emplace(
        std::piecewise_construct,
        std::make_tuple(task_id),
        std::make_tuple(task_id, std::move(kernel)));
    return (it.first)->second;
  }

private:
  void pass_to_cl(Task& task) {
    auto& kernel = task.kernel();
    auto& kernel_done = task.kernel_done();
    task.enqueued_now();
    _command_queue.enqueueTask(kernel, NULL, std::addressof(kernel_done));
    kernel_done.setCallback(
        CL_COMPLETE, &kernel_done_clb, std::addressof(task));
  }

  static void kernel_done_clb(cl_event, cl_int status, void* user_data)
  {
    assert(status == CL_SUCCESS);
    auto task = static_cast<Task*>(user_data);
    info("Task {} completed.", task->id());
  }

private:
  std::map<uint64_t, Task> _tasks;
  const cl::CommandQueue _command_queue;
  uint64_t _current_id;
};
}
