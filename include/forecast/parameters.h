#pragma once

#include <cmath>
#include <cstddef>
#include <cassert>
#include <functional>
#include <map>

#define GFLOPS * static_cast<std::size_t>(1000000000)

namespace forecast {

struct KernelParams {
  KernelParams() = default;
  KernelParams(std::size_t max_flops, std::function<float(std::size_t)> flops)
    : _max_flops(max_flops)
    , _flops_fun(flops)
  {
  }
  float flop(std::size_t n) const
  {
    return _flops_fun(n);
  }

  float max_flops() const {
    return _max_flops;
  }


  private:
  // FLOPS
  float _max_flops = 1;
  std::function<float(std::size_t)> _flops_fun;
};

auto matrix_mult = [](std::size_t n) {
  auto       len         = std::sqrt(n);
  const auto total_flops = 2 * len * len * len;
  return static_cast<float>(total_flops);
};

KernelParams& kernel_params(const std::string& config, const std::string& kernel) {
  static std::map<std::string, std::map<std::string, KernelParams>> params;

  if(params.empty()) {
    params["mmult_f_d"]["matrixMult"] =
        KernelParams{120 GFLOPS, matrix_mult};
    params["mmult_f_d"]["matrixMultD"] =
        KernelParams{63 GFLOPS, matrix_mult};
    params["mmult_f_d2"]["matrixMult"] =
        KernelParams{35 GFLOPS, matrix_mult};
    params["mmult_f_d2"]["matrixMultD"] =
        KernelParams{72 GFLOPS, matrix_mult};
  }

  assert(params.size() == 2);

  return params.at(config).at(kernel);
}

}
