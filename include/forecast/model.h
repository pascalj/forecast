#pragma once

#include "task.h"
#include "parameters.h"

#include <numeric>

namespace forecast {
class Queue;

class Model {
public:
  Model(const std::string &name)
    : _name(name)
  {
  }

  float cost(const Task &) const {
    return 1.337f;
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
  std::string _name;
};
}
