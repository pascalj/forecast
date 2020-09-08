#pragma once

#include "task.h"
#include "model.h"

#include <CL/cl.hpp>
#include <map>
#include <set>
#include <util.h>

namespace forecast {


class Queue {
public:
  Queue(const cl::CommandQueue& command_queue) : _command_queue(command_queue) {}
  Queue(cl::CommandQueue&& command_queue) = delete;

  // We take the kernel by value here to allow the re-use of a kernel object
  // in the main program.
  Task& enqueue(cl::Kernel kernel, bool force = false) {
    const auto task_id = _current_id++;
    auto       it      = _tasks.emplace(
        std::piecewise_construct,
        std::make_tuple(task_id),
        std::make_tuple(task_id, std::move(kernel)));
    auto& task = (it.first)->second;
    if(force) {
      pass_to_cl(task);
    }
    return task;
  }

  auto& tasks() const {
    return _tasks;
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
    task->finished_now();
    warn("Task {} completed.", task->id());
  }

private:
  std::map<uint64_t, Task> _tasks;
  const cl::CommandQueue _command_queue;
  uint64_t _current_id;
};
}
