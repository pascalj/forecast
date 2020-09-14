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
public:
  Scheduler(cl::Context* ctx)
    : _ctx(ctx)
    , _current_config(0)
    , _current_id(0)
  {
  }

  void reset() {
    _queues.clear();
    _current_config = 0;
    _current_id     = 0;
  }

  void add_config(const std::string &bitstream)
  {
    if (std::none_of(
            _configs.begin(),
            _configs.end(),
            [&bitstream](const auto& config) {
              return config.bitstream() == bitstream;
            })) {
      _configs.emplace_back(bitstream, _ctx);
    }
  }

  void add_task(const std::string kernel, KernelGen creator)
  {
    const auto task_id = _current_id++;
    Task task(task_id, kernel, creator);
    task.scheduler = this;
    auto queue     = _queues.try_emplace(
        kernel,
        cl::CommandQueue(*_ctx),
        std::addressof(current_config().program()));
    queue.first->second.enqueue(std::move(task));
  }

  Configuration& current_config() {
    return _configs[_current_config];
  }

  void wait() {
    for(auto &queue : _queues) {
      queue.second.wait();
    }
  }

  void finish() {
    wait();
    reset();
  }

  std::size_t size() const {
    return std::accumulate(
        _queues.begin(), _queues.end(), 0, [](std::size_t sum, const auto& queue) {
          return sum + queue.second.size();
        });
  }

private:
  cl::Context*                 _ctx;
  std::vector<Configuration>   _configs;
  std::size_t                  _current_config;
  uint64_t                     _current_id;
  std::map<std::string, Queue> _queues;
};
}
