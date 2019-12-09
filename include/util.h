#pragma once

#include <assert.h>
#include <fstream>

#include <CL/cl.h>
#include <cl_error.h>
#include <log.h>

constexpr size_t max_name_len   = 128;
constexpr size_t max_device_len = 4;

std::vector<unsigned char> read_file(const char *path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<unsigned char>(std::istreambuf_iterator<char>(in), {});
}

auto get_device_context(size_t req_platform = 0, size_t req_device = 0)
{
  cl_uint        num_platforms;
  cl_platform_id platform;
  cl_device_id   selected_device_id;

  cl_assert(clGetPlatformIDs(1, &platform, &num_platforms));
  if(num_platforms == 0) {
    warn("No platforms detected");
  }

  for (size_t i = 0; i < num_platforms; i++) {
    char         platform_name[max_name_len];
    size_t       platform_name_size;
    cl_uint      num_devices;
    cl_device_id devices[max_device_len];

    cl_assert(clGetPlatformInfo(
        platform,
        CL_PLATFORM_NAME,
        max_name_len,
        platform_name,
        &platform_name_size));
    info("Platform {} name: {}", i, platform_name);

    cl_assert(clGetDeviceIDs(
        platform, CL_DEVICE_TYPE_ALL, 1, devices, &num_devices));
    for (size_t j = 0; j < num_devices; j++) {
      char   device_name[max_name_len];
      size_t device_name_len;
      clGetDeviceInfo(
          devices[i],
          CL_DEVICE_NAME,
          max_name_len,
          device_name,
          &device_name_len);
      info("  Device {} name: {}", j, device_name);

      if(i == req_platform && j == req_device) {
        info("Selected device: {}", device_name);
        selected_device_id = devices[i];
      }
    }
  }

  cl_int err;
  cl_context context = clCreateContext(NULL, 1, &selected_device_id, NULL, NULL, &err);
  cl_ok(err);
  return context;
}

auto get_device(cl_context ctx, size_t device_idx = 0) {
  cl_device_id devices[max_device_len];
  size_t       devices_size;

  cl_assert(clGetContextInfo(
      ctx, CL_CONTEXT_DEVICES, sizeof(cl_device_id) * max_device_len, devices, &devices_size));

  assert(devices_size > device_idx);

  return devices[device_idx];
}

auto get_command_queue(cl_context ctx, size_t device_idx = 0) {
  cl_int err;

  cl_device_id device = get_device(ctx, device_idx);
  cl_command_queue command_queue =
      clCreateCommandQueue(ctx, device, 0, &err);
  cl_ok(err);

  return command_queue;
}

