#pragma once

#include <CL/cl.hpp>
#include <condition_variable>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#include "configuration.h"
#include "task.h"

namespace forecast { class Scheduler { public: Scheduler(cl::Context* ctx) :
  _ctx(ctx) , _current_config(nullptr) , _current_id(0) ,
    _logger(std::make_unique<spdlog::logger>("file_logger",
          std::make_unique<spdlog::sinks::basic_file_sink_st>("logs/scheduler.csv", true)))
  {
    _logger->set_pattern("%v");
    _logger->info(
        "id, config, kernel, flops, online, offline, log_online, actual");
  }

  void reset() {
    _queues.clear();
    _models.clear();
    _current_config = nullptr;
    _current_id     = 0;
  }

  void add_config(const std::string &bitstream)
  {
    _configs.try_emplace(bitstream, bitstream, _ctx);
    _models.try_emplace(bitstream, bitstream);
    if(_configs.size() == 1) {
      _current_config = std::addressof(_configs.begin()->second);
    }
  }

  void add_task(Task &&task)
  {
    const auto task_id = _current_id++;
    task.set_id(task_id);
    TaskCallback clb =
        std::bind(&Scheduler::task_done, this, std::placeholders::_1);
    auto& queue = _queues
                      .try_emplace(
                          task.function_name(),
                          *_ctx,
                          std::addressof(current_config().program()),
                          std::move(clb))
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
    float cost = 0;
    for(auto& queue : _queues) {
      cost +=
          _models.at(_current_config->bitstream()).cost(queue.second.tasks());
    }
    info("Cost: {}", cost);
  }

  void task_done(Task t) {
    auto params =
        kernel_params(_current_config->bitstream(), t.function_name());
    const auto  total      = t.global()[0] * t.global()[1];
    auto        total_flop = params.flop(total);
    auto& model = _models.at(_current_config->bitstream());
    Measurement measurement{t.duration().count(), total_flop};
    model.add_measurement(t, measurement);
    auto linreg  = model.linreg(t);
    auto online  = linreg.alpha + linreg.beta * total_flop;
    info("alpha: {}, beta: {}", linreg.alpha, linreg.beta);
    auto offline = model.cost(t);
    auto simple_linreg = model.simple_linreg(t);
    auto hybrid = model.offline_alpha + simple_linreg.beta * total_flop;

    _logger->info(
        "{}, {}, {}, {}, {}, {}, {}, {}",
        t.id(),
        _current_config->bitstream(),
        t.function_name(),
        total_flop,
        online,
        offline,
        hybrid,
        t.duration().count());
  }

private:
  cl::Context*                         _ctx;
  std::map<std::string, Configuration> _configs;
  std::map<std::string, Model>         _models;

  Configuration*                       _current_config;
  uint64_t                             _current_id;
  std::map<std::string, Queue>         _queues;
  std::unique_ptr<spdlog::logger>              _logger;
};
}
