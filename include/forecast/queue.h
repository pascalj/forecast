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

  static void kernel_done_clb(cl_event, cl_int status, void* user_data)
  {
    assert(status == CL_SUCCESS);
    auto task = static_cast<Task*>(user_data);
    task->finished_now();
    warn("Task {} completed.", task->id());
  }

private:
  const cl::CommandQueue _command_queue;
};
}
