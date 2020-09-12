#pragma once

#include "model.h"
#include "queue.h"

#include <string>
#include <sstream>
#include <unordered_map>
#include <util.h>

namespace forecast {

class Configuration {
public:
  Configuration() = delete;
  Configuration(const std::string& bitstream, cl::Context* ctx)
    : _bitstream(bitstream)
    , _model(bitstream)
    , _ctx(ctx)
    , devices(get_devices())
  {
    std::stringstream file;
    file << "../kernels/" << _bitstream << ".aocx";
    _binary = Binary(file.str().c_str());
    _program = cl::Program(*ctx, devices, _binary.cl_binaries());
    cl_ok(_program.build());
  }

  template <typename Tasks>
  float cost(const Tasks& tasks) const
  {
    return _model.cost(tasks);
  }

  cl::Program& program() {
    return _program;
  }

  cl::CommandQueue& queue(Task& task)
  {
    const auto name = task.function_name();
    if(_queues.count(name) == 0) {
      _queues.emplace(name, *_ctx);
    }
    return _queues[task.function_name()];
  }

  auto& queues()
  {
    return _queues;
  }

private:

  Binary& binary() {
    return _binary;
  }

  std::string                                       _bitstream;
  Model                                             _model;
  Binary                                            _binary;
  cl::Program                                       _program;
  cl::Context*                                       _ctx;
  std::unordered_map<std::string, cl::CommandQueue> _queues;
  std::vector<cl::Device>                           devices;
};

}  // namespace forecast
