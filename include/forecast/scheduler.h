#pragma once

#include <CL/cl.hpp>
#include <condition_variable>
#include <string>
#include <thread>
#include <vector>

#include "configuration.h"
#include "task.h"

namespace forecast {
class Scheduler {
#if 0
  enum class Event {
    enqueue_task,
    start_task,
    finish_task
  };

  enum class QueueState {
    empty,
    working,
    not_working
  };

  class States {
    QueueState next(Event e) {
      switch (state) {
        case QueueState::empty:
          switch(e) {
            case Event::enqueue_task:
              state = QueueState::not_working;
              break;
            case Event::start_task:
            case Event::finish_task:
              assert(false);
              break;
          }
          break;
        case QueueState::working:
          switch(e) {
            case Event::enqueue_task:
              break;
            case Event::start_task:
            case Event::finish_task:
              assert(false);
              break;
          }
          break;


        case QueueState::not_working:
      }
    }

    QueueState state;
  }
#endif
public:
  Scheduler(cl::Context* ctx)
    : _ctx(ctx)
    , _current_config(0)
    , _current_id(0)
    , _next_executor(std::thread(std::bind(&Scheduler::execute_next, this)))
  {
  }

  void reset() {
    _tasks.clear();
    _current_config = 0;
    _current_id = 0;
  }

  void add_config(const std::string &bitstream)
  {
    if (_configs.size() == 0) {
      _configs.emplace_back(bitstream, _ctx);
    }
  }

  void add_task(const std::string kernel, KernelGen creator)
  {
    const auto task_id = _current_id++;
    info("Task {} ({}) in", kernel, task_id);
    _tasks.emplace_back(task_id, kernel, creator);
    _tasks.back().scheduler = this;
    {
      std::lock_guard<std::mutex> lg(_m);
      if(_tasks.size() == 1) {
        _cv.notify_one();
      }
    }
  }

  Configuration& current_config() {
    return _configs[_current_config];
  }

  void finish() {
    while(true) {
      if(not _working && _tasks.size() == 0) {
        return;
      }
    }
  }

  std::size_t size() const {
    return _tasks.size();
  }

  ~Scheduler() {
    {
      std::lock_guard<std::mutex> lg(_m);
      _finished = true;
    }
    _cv.notify_all();
    _next_executor.join();
  }
private:
  cl::Event& pass_to_cl(Task& task) {
    info("Task {} next", task.id());
    auto& kernel = task.generate_kernel(current_config().program());
    auto& kernel_done = task.kernel_done();
    task.enqueued_now();
    cl_ok(current_config().queue(task).enqueueTask(
        kernel, NULL, std::addressof(kernel_done)));
    current_config().queue(task).flush();
    info("Task {} next done", task.id());
    return kernel_done;
  }

  void execute_next() {
    while(not _finished) {
      std::unique_lock<std::mutex> lk(_m);
      _cv.wait(lk, [this]() { return not _working && _tasks.size() > 0; });
      auto &task = _tasks.front();
      auto &event = pass_to_cl(task);
      _working = true;
      lk.unlock();

      event.wait();
      info("Task {} finished.", task.id());
      lk.lock();
      _working = false;
      _tasks.pop_front();
      lk.unlock();
      _cv.notify_all();
    }
  }



private:
  cl::Context*               _ctx;
  std::vector<Configuration> _configs;
  std::size_t                _current_config;
  Tasks                      _tasks;
  uint64_t                   _current_id;
  std::mutex                 _m;
  std::condition_variable    _cv;
  std::thread                _next_executor;
  bool                       _working  = false;
  bool                       _finished = false;
};
}
