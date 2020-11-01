#pragma once

#include <benchmark/benchmark.h>
#include <benchmarks/fixtures.h>
#include <log.h>
#include <util.h>
#include <cl_error.h>
#include <forecast/queue.h>

struct double2 {
  double x;
  double y;
};

struct float2 {
  float x;
  float y;
};

static int coord(int iteration, int i);
void fourier_transform_gold(bool inverse, const int lognr_points, double2 *data);
void fourier_stage(int lognr_points, double2 *data);

constexpr int N              = (1 << 12);  // must match .cl file

BENCHMARK_DEFINE_F(BasicKernelFixture, FFT1D)(benchmark::State& state)
{

  const size_t  fft_iterations = state.range(0);
  constexpr bool inverse        = false;

  float2 *h_inData, *h_outData;
  double2 *h_verify;
  h_inData = (float2 *)aligned_alloc(64, sizeof(float2) * N * fft_iterations);
  h_outData = (float2 *)aligned_alloc(64, sizeof(float2) * N * fft_iterations);
  h_verify = (double2 *)aligned_alloc(64, sizeof(double2) * N * fft_iterations);
  if (!(h_inData && h_outData && h_verify)) {
    state.SkipWithError( "ERROR: Couldn't create host buffers\n");
    return;
  }

  for (int i = 0; i < static_cast<int>(fft_iterations); i++) {
    for (int j = 0; j < N; j++) {
      h_verify[coord(i, j)].x = h_inData[coord(i, j)].x = (float)((double)rand() / (double)RAND_MAX);
      h_verify[coord(i, j)].y = h_inData[coord(i, j)].y = (float)((double)rand() / (double)RAND_MAX);
    }
  }

  cl_int status;
  auto d_inData = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(float2) * N * fft_iterations, NULL, &status);
  cl_ok(status);
  /* d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * iterations, NULL, &status); */
  auto d_outData = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(float2) * N * fft_iterations, NULL, &status);
  cl_ok(status);

  cl::CommandQueue queue1(ctx);
  // Copy data from host to device
  status = queue1.enqueueWriteBuffer(d_inData, CL_TRUE, 0, sizeof(float2) * N * fft_iterations, h_inData);
  cl_ok(status);

  // Can't pass bool to device, so convert it to int
  int inverse_int = inverse;

  // Set the kernel arguments
  auto kernel0 = kernel("fft1d", "fft1d");
  auto kernel1 = kernel("fft1d", "fetch");
  cl_ok(kernel1.setArg(0, sizeof(cl_mem), (void *)&d_inData));
  cl_ok(kernel0.setArg(0, sizeof(cl_mem), (void *)&d_outData));
  cl_ok(kernel0.setArg(1, sizeof(cl_int), (void *)&fft_iterations));
  cl_ok(kernel0.setArg(2, sizeof(cl_int), (void *)&inverse_int));

  for(auto _ : state) {
    // Launch the kernel - we launch a single work item hence enqueue a task
    cl_ok(queue.enqueueTask(kernel0));

    auto ls = cl::NDRange{N/8};
    auto gs = cl::NDRange{fft_iterations * ls[0]};
    cl_ok(queue1.enqueueNDRangeKernel(kernel1, cl::NullRange, gs, ls));
    
    // Wait for command queue to complete pending events
    cl_ok(queue.finish());
    cl_ok(queue1.finish());
  }
  
  // Copy results from device to host
  cl_ok(queue.enqueueReadBuffer(d_outData, CL_TRUE, 0, sizeof(float2) * N * fft_iterations, h_outData));

  // TODO: check
  const double gflop = 5 * N * (log((float)N) / log((float)2)) *
                       fft_iterations * state.iterations();

  // Pick randomly a few iterations and check SNR
  double fpga_snr = 200;
  for (size_t i = 0; i < fft_iterations; i+= rand() % 20 + 1) {
    fourier_transform_gold(inverse, 12, h_verify + coord(i, 0));
    double mag_sum = 0;
    double noise_sum = 0;
    for (int j = 0; j < N; j++) {
      double magnitude =
          (double)h_verify[coord(i, j)].x * (double)h_verify[coord(i, j)].x +
          (double)h_verify[coord(i, j)].y * (double)h_verify[coord(i, j)].y;
      double noise =
          (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) *
              (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) +
          (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y) *
              (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y);

      mag_sum += magnitude;
      noise_sum += noise;
    }
    double db = 10 * log(mag_sum / noise_sum) / log(10.0);
    // find minimum SNR across all iterations
    if (db < fpga_snr) fpga_snr = db;
  }
  if(fpga_snr <= 125) {
    state.SkipWithError("Validation failed, SNR too high.");
  } else {
    debug("Signal to noise ratio on output sample: {} --> {}", fpga_snr, fpga_snr > 125 ? "PASSED" : "FAILED");
  }
  state.counters["FLOPs"] =
      benchmark::Counter(gflop, benchmark::Counter::kIsRate);
}

void fourier_transform_gold(bool inverse, const int lognr_points, double2 *data) {
   const int nr_points = 1 << lognr_points;

   // The inverse requires swapping the real and imaginary component
   
   if (inverse) {
      for (int i = 0; i < nr_points; i++) {
         double tmp = data[i].x;
         data[i].x = data[i].y;
         data[i].y = tmp;;
      }
   }
   // Do a FT recursively
   fourier_stage(lognr_points, data);

   // The inverse requires swapping the real and imaginary component
   if (inverse) {
      for (int i = 0; i < nr_points; i++) {
         double tmp = data[i].x;
         data[i].x = data[i].y;
         data[i].y = tmp;;
      }
   }

   // Do the bit reversal

   double2 *temp = (double2 *)alloca(sizeof(double2) * nr_points);
   for (int i = 0; i < nr_points; i++) temp[i] = data[i];
   for (int i = 0; i < nr_points; i++) {
      int fwd = i;
      int bit_rev = 0;
      for (int j = 0; j < lognr_points; j++) {
         bit_rev <<= 1;
         bit_rev |= fwd & 1;
         fwd >>= 1;
      }
      data[i] = temp[bit_rev];
   }
}

void fourier_stage(int lognr_points, double2 *data) {
   int nr_points = 1 << lognr_points;
   if (nr_points == 1) return;
   double2 *half1 = (double2 *)alloca(sizeof(double2) * nr_points / 2);
   double2 *half2 = (double2 *)alloca(sizeof(double2) * nr_points / 2);
   for (int i = 0; i < nr_points / 2; i++) {
      half1[i] = data[2 * i];
      half2[i] = data[2 * i + 1];
   }
   fourier_stage(lognr_points - 1, half1);
   fourier_stage(lognr_points - 1, half2);
   for (int i = 0; i < nr_points / 2; i++) {
      data[i].x = half1[i].x + cos (2 * M_PI * i / nr_points) * half2[i].x + sin (2 * M_PI * i / nr_points) * half2[i].y;
      data[i].y = half1[i].y - sin (2 * M_PI * i / nr_points) * half2[i].x + cos (2 * M_PI * i / nr_points) * half2[i].y;
      data[i + nr_points / 2].x = half1[i].x - cos (2 * M_PI * i / nr_points) * half2[i].x - sin (2 * M_PI * i / nr_points) * half2[i].y;
      data[i + nr_points / 2].y = half1[i].y + sin (2 * M_PI * i / nr_points) * half2[i].x - cos (2 * M_PI * i / nr_points) * half2[i].y;
   }
}

int coord(int iteration, int i) {
  return iteration * N + i;
}

BENCHMARK_REGISTER_F(BasicKernelFixture, FFT1D)
    ->RangeMultiplier(2)
    ->Range(1 << 2, 1 << 15)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
