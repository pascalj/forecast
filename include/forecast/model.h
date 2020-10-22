#pragma once

#include "task.h"
#include "parameters.h"

#include <numeric>
#include <iostream>

namespace forecast {
class Queue;

class Model {
public:
  Model(const std::string &config)
    : _config(config)
  {
  }

  float cost(const Task &task) const {
    auto       params = kernel_params(_config, task.function_name());
    const auto total = task.global()[0] * task.global()[1];
    float      cost   = 0.0001 + (params.flop(total) / params.max_flops());
    return cost;
  }

  template<typename Tasks>
  float cost(const Tasks &tasks) const {
    return std::accumulate(
        std::begin(tasks),
        std::end(tasks),
        0.0f,
        [this](float accum, auto &t) { return accum + cost(t); });
  }

private:
  std::string _config;
};
}
