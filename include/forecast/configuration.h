#pragma once

#include "model.h"

#include <string>
#include <sstream>
#include <util.h>

namespace forecast {

class Configuration {
public:
  Configuration() = delete;
  Configuration(const std::string& bitstream, cl::Context* ctx)
    : _bitstream(bitstream)
    , _ctx(ctx)
    , _model(Parameters())
  {
  }

  template <typename Tasks>
  float cost(const Tasks& tasks) const
  {
    return _model.cost(tasks);
  }

  cl::Kernel& kernel(const std::string& prog, const std::string& kern_name) {
    if(kernels.count(kern_name)) {
      return kernels[kern_name];
    }
    auto prg = program(prog);
    cl::Kernel kernel(prg, kern_name.c_str());
    kernels.insert(std::make_pair(kern_name, kernel));
    return kernels[kern_name];
  }
private:

  cl::Program& program(const std::string& prog_name) {
    if(programs.count(prog_name)) {
      return programs[prog_name];
    }
    cl::Program prg(*_ctx, devices, binary(prog_name).cl_binaries());
    cl_ok(prg.build()); //no op
    programs.insert(std::make_pair(prog_name, prg));
    return programs[prog_name];
  }

  Binary& binary(const std::string& bin_name) {
    if(binaries.count(bin_name)) {
      return binaries[bin_name];
    }
    std::stringstream file;
    file << "../kernels/" << bin_name << ".aocx";
    Binary binary(file.str().c_str());
    binaries.insert(std::make_pair(bin_name, binary));
    return binaries[bin_name];
  }

  std::string                                  _bitstream;
  cl::Context*                                 _ctx;
  Model _model;
  std::unordered_map<std::string, Binary>      binaries;
  std::unordered_map<std::string, cl::Program> programs;
  std::unordered_map<std::string, cl::Kernel>  kernels;
  std::vector<cl::Device> devices;
};

}  // namespace forecast
