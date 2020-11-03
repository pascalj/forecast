#pragma once

#include "task.h"
#include "parameters.h"

#include <numeric>
#include <iostream>

namespace forecast {
class Queue;

struct Measurement {
  double y = 0;
  double x = 0;
};

struct Parameters {
  double alpha = 0;
  double beta = 1.0;
};

class Model {
public:
  Model(const std::string &config)
    : _config(config)
  {
  }

  static constexpr double offline_alpha = 0.0001;

  float cost(const Task &task) const {
    auto       params = kernel_params(_config, task.function_name());
    const auto total = task.global()[0] * task.global()[1];
    float      cost   = offline_alpha + (params.flop(total) / params.max_flops());
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

  void add_measurement(const Task &task, Measurement &m) {
    _measurements.insert({task.function_name(), m});
  }

  Parameters linreg(
      const Task &                  task,
      std::function<double(double)> map = [](double v) { return v; }) const
  {
    auto   matches = _measurements.equal_range(task.function_name());
    auto   begin   = matches.first;
    auto   end     = matches.second;
    double m       = std::distance(begin, end);

    auto sum_x   = std::accumulate(begin, end, 0.0f, [&map](double sum, auto val) {
      return sum + map(val.second.x);
    });
    auto sum_y   = std::accumulate(begin, end, 0.0f, [&map](double sum, auto val) {
      return sum + map(val.second.y);
    });
    double avg_x = sum_x / m;
    double avg_y = sum_y / m;

    double  t = std::accumulate(begin, end, 0.0f, [avg_x, avg_y, &map](double sum, auto val) {
      return sum + (map(val.second.x) - avg_x) * (map(val.second.y) - avg_y);
    });
    double b   = std::accumulate(begin, end, 0.0f, [avg_x, &map](double sum, auto val) {
      return sum + (map(val.second.x) - avg_x) * (map(val.second.x) - avg_x);
    });

    auto   beta    = t / b;
    auto   alpha   = avg_y - beta * avg_x;
    return Parameters{alpha, beta};
  }

  Parameters simple_linreg(const Task &task) {
    auto   matches = _measurements.equal_range(task.function_name());
    auto   begin   = matches.first;
    auto   end     = matches.second;

    auto sum_xy   = std::accumulate(begin, end, 0.0f, [](double sum, auto val) {
      return sum + val.second.x * val.second.y;
    });
    auto sum_x2   = std::accumulate(begin, end, 0.0f, [](double sum, auto val) {
      return sum + val.second.x * val.second.x;
    });

    auto   beta    = sum_xy / sum_x2;
    return Parameters{0, beta};
  }


private:
  std::string _config;
  std::multimap<std::string, Measurement> _measurements;
};
}

