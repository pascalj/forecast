#pragma once

#include <benchmark/benchmark.h>
#include <log.h>
#include <util.h>
#include <unordered_map>
#include <vector>
#include <sstream>

class BasicKernelFixture : public benchmark::Fixture {
public:
  BasicKernelFixture() : devices(get_devices()), ctx(devices.front()), queue(ctx) {}

  void SetUp(const ::benchmark::State&) {
  }

  void TearDown(const ::benchmark::State&) {
    queue.finish();
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

  cl::Program& program(const std::string& prog_name) {
    if(programs.count(prog_name)) {
      return programs[prog_name];
    }
    cl::Program prg(ctx, devices, binary(prog_name).cl_binaries());
    prg.build(); //no op
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

  std::vector<cl::Device> devices;
  cl::Context ctx;
  cl::CommandQueue queue;
  std::unordered_map<std::string, Binary> binaries;
  std::unordered_map<std::string, cl::Program> programs;
  std::unordered_map<std::string, cl::Kernel> kernels;
};

