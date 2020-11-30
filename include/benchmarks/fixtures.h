#pragma once

#include <benchmark/benchmark.h>
#include <forecast/configuration.h>
#include <forecast/scheduler.h>
#include <log.h>
#include <util.h>
#include <unordered_map>
#include <vector>
#include <sstream>

class ClState {
public:
  ClState() : devices(get_devices()), ctx(devices.front()), queue(ctx) {}
  ~ClState() { clear_up(); }

  void clear_up() {
    queue.finish();
    kernels.clear();
    programs.clear();
    binaries.clear();
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

  std::vector<cl::Device> devices;
  cl::Context ctx;
  cl::CommandQueue queue;
  std::unordered_map<std::string, Binary>      binaries;
  std::unordered_map<std::string, cl::Program> programs;
  std::unordered_map<std::string, cl::Kernel>  kernels;
};

class BasicKernelFixture : public benchmark::Fixture {
public:

  void SetUp(const ::benchmark::State&) {
  }

  void TearDown(const ::benchmark::State&) {
    clstate.clear_up();
  }
  cl::Kernel& kernel(const std::string& prog, const std::string& kern_name) {
    return clstate.kernel(prog, kern_name);
  }

  cl::Program& program(const std::string& prog_name) {
    return clstate.program(prog_name);
  }

  Binary& binary(const std::string& bin_name) {
    return clstate.binary(bin_name);
  }


  ClState clstate;
};


class BitstreamFixture : public BasicKernelFixture {
public:
  BitstreamFixture(const std::string& bitstream)
    : BasicKernelFixture(), _bitstream(bitstream)
  {
  }

  cl::Kernel& kernel(const std::string& kern_name) {
    return BasicKernelFixture::kernel(_bitstream, kern_name);
  }
  std::string _bitstream;
};

struct MmultFD : public BitstreamFixture {
  static constexpr char bitstream[] = "mmult_f_d";
};

struct MmultFD2 {
  static constexpr char bitstream[] = "mmult_f_d2";
};

class ForecastFixture : public BasicKernelFixture {
public:
  ForecastFixture()
    : BasicKernelFixture()
    , scheduler(&clstate.ctx)
  {
  }

  forecast::Scheduler scheduler;
};

class RandomTasks {
public:
  void add_kernel(float prob, forecast::Task&& task, std::function<forecast::TaskDims()> size_gen)
  {
    _tasks.emplace_back(std::move(task));
    float init = _offsets.size() > 0 ? _offsets.back() : 0;
    _offsets.emplace_back(init + prob);
    _size_gens.emplace_back(size_gen);
  }

  forecast::Task next_task() {
    assert(_offsets.size() > 0);
    assert(
        std::abs(_offsets.back() - 1.0f) <=
        std::numeric_limits<float>::epsilon());

    float random =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    auto  selected_it =
        std::find_if(_offsets.begin(), _offsets.end(), [random](auto offset) {
          return offset > random;
        });
    auto task_index = std::distance(_offsets.begin(), selected_it);
    auto task = forecast::Task{_tasks[task_index]};
    task.set_dims(_size_gens[task_index]());
    debug("Generated task: {}", task);
    return task;
  }

private:
  std::vector<forecast::Task>           _tasks;
  std::vector<float>                    _offsets;
  std::vector<std::function<forecast::TaskDims()>>  _size_gens;
};

template<typename T, cl_mem_flags mode = CL_MEM_READ_WRITE>
struct Buffer {
  Buffer(const cl::Context& ctx, size_t size)
    : size(size), buf(ctx, mode, size * sizeof(T))
  {
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) & = delete;
  Buffer(Buffer&&) noexcept          = default;
  Buffer& operator=(Buffer&&) noexcept = default;

  void fill(const cl::CommandQueue& queue, T init) const
  {
    std::vector<T> host_buf(size, init);
    // Note: we should be able to use enqueueFillBuffer here, but the
    // implementation segfauls. See Intel KBD article about enqueueFillBuffer.
    cl_ok(queue.enqueueWriteBuffer(
        buf, CL_TRUE, 0, size * sizeof(T), host_buf.data()));
  }

  template <typename Func>
  bool validate(const cl::CommandQueue& queue, Func&& func) const
  {
    std::vector<T> host_copy(size);
    cl_ok(cl::copy(queue, buf, host_copy.begin(), host_copy.end()));
    return std::all_of(host_copy.begin(), host_copy.end(), func);
  }

  size_t     size;
  cl::Buffer buf;
};

template<size_t n_bufs, typename T>
struct Buffers {
  Buffers(const cl::Context& ctx, size_t size)
  {
    bufs.reserve(n_bufs);
    for(size_t i = 0; i < n_bufs; i++) {
      bufs.emplace_back(ctx, size);
    }
  }

  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) & = delete;
  Buffers(Buffers&&) noexcept          = default;
  Buffers& operator=(Buffers&&) noexcept = default;

  void fill_all(
      const cl::CommandQueue& queue, const std::array<T, n_bufs>&& vals)
  {
    for (size_t i = 0; i < n_bufs; i++) {
      bufs[i].fill(queue, vals[i]);
    }
  }

  auto begin() const {
    return bufs.begin();
  }

  auto end() const {
    return bufs.end();
  }

  Buffer<T>& operator[](size_t n) {
    return bufs[n];
  }

  std::vector<Buffer<T>> bufs;
};

template <size_t N, typename T>
void set_bufs_as_args(cl::Kernel& kernel, const Buffers<N, T>& bufs)
{
  int i = 0;
  std::for_each(bufs.begin(), bufs.end(), [&kernel, &i](const auto& buffer) {
    cl_ok(kernel.setArg(i++, buffer.buf));
  });
}
