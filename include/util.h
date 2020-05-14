#pragma once

#include <assert.h>
#include <fstream>

#include <CL/cl.hpp>
#include <cl_error.h>
#include <log.h>

struct Binary {
  Binary(const char *path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<unsigned char> bytes_vec(std::istreambuf_iterator<char>(in), {});
    bytes = bytes_vec;
  }

  cl::Program::Binaries cl_binaries() {
    // Mhhh if only there was something that can hold a pointer and a size
    // already in the stdlib...
    auto binary_pair = std::make_pair(bytes.data(), bytes.size());
    return cl::Program::Binaries{binary_pair};
  }

  std::vector<unsigned char> bytes;
};

auto read_file(const char *path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<unsigned char> binary(std::istreambuf_iterator<char>(in), {});
  return binary;
}

auto get_binaries(const char *path) {
  Binary binary(path);
}

auto get_devices(size_t req_platform = 0, size_t req_device = 0)
{
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if(platforms.empty()) {
    critical("No platforms detected");
  }

  for (auto &p : platforms) {
    std::vector<cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    info("Platform: {} ({} devices)", p.getInfo<CL_PLATFORM_NAME>(), devices.size());
    for(auto& device : devices) {
      info("  Device: {}", device.getInfo<CL_DEVICE_NAME>());
    }
  }

  assert(platforms.size() > req_platform);
  auto platform = platforms[req_platform];
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  assert(devices.size() > req_device);
  return std::vector<cl::Device>{devices[req_device]};
}

