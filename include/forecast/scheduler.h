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
    _configs.try_emplace(bitstream, bitstream, _ctx);
    if(_configs.size() == 1) {
      _current_config = std::addressof(_configs.begin()->second);
    }
  }

  void add_task(Task &&task)
  {
    const auto task_id = _current_id++;
    task.set_id(task_id);
    task.set_scheduler(this);
    auto& queue     = _queues
                      .try_emplace(
                          task.function_name(),
                          *_ctx,
                          std::addressof(current_config().program()))
                      .first->second;
    queue.enqueue(std::move(task));
    print_costs();
  }

  Configuration& current_config()
  {
    return *_current_config;
  }

  void set_config(const std::string &name) {
    _current_config = std::addressof(_configs.at(name));
    for (auto &queue : _queues) {
      queue.second.set_program(std::addressof(_current_config->program()));
    }
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

  void print_costs() const {
    for(auto& config : _configs) {
      Model model{config.first};
      float cost = 0;
      for(auto& queue : _queues) {
        cost += model.cost(queue.second.tasks());
      }
      info("Cost {}: {}", config.first, cost);
    }
  }

private:
  cl::Context*                         _ctx;
  std::map<std::string, Configuration> _configs;
  Configuration*                       _current_config;
  uint64_t                             _current_id;
  std::map<std::string, Queue>         _queues;
};
}
