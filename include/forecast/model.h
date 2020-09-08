#pragma once

#include "task.h"
#include "parameters.h"

#include <numeric>

namespace forecast {
class Queue;

class Model {
public:
  Model(Parameters params)
    : _params(params)
  {
  }

  float cost(const Task &) const {
    return _params.dummy;
  }

  template<typename Tasks>
  float cost(const Tasks &tasks) const {
    return std::accumulate(
        std::begin(tasks),
        std::end(tasks),
        0.0f,
        [this](float accum, auto &t) { return accum + cost(t); });
  }

  Parameters _params;
};
}
