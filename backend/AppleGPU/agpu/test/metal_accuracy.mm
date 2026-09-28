// Measure a transcendental's default namespace against metal::precise::.
// Not a unit test: it prints numbers a human reads once and writes into the
// table in Builtins.h.
//
// Build:
//   clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
//           -o metal_accuracy test/metal_accuracy.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <vector>

namespace {

struct Probe {
  const char *name;
  float lo, hi;
};

const Probe kProbes[] = {
    {"acos", -1.0f, 1.0f},       {"asin", -1.0f, 1.0f},
    {"atan", -100.0f, 100.0f},   {"tan", -1.5f, 1.5f}, // away from the poles
    {"sinh", -10.0f, 10.0f},     {"cosh", -10.0f, 10.0f},
    {"cbrt", -1000.0f, 1000.0f}, {"exp", -80.0f, 80.0f},
    {"exp2", -80.0f, 80.0f},     {"sqrt", 0.0f, 10000.0f},
};

constexpr int kN = 1 << 16;

std::string sourceFor(const Probe &p) {
  std::string f = p.name;
  return std::string("#include <metal_stdlib>\n"
                     "using namespace metal;\n"
                     "kernel void probe(device float *out [[buffer(0)]],\n"
                     "                  uint t [[thread_position_in_grid]]) {\n"
                     "  float x = ") +
         std::to_string(p.lo) + "f + (" + std::to_string(p.hi - p.lo) +
         "f) * (float)t / " + std::to_string(kN - 1) +
         ".0f;\n"
         "  float a = " +
         f +
         "(x);\n"
         "  float b = precise::" +
         f +
         "(x);\n"
         "  float d = fabs(a - b);\n"
         "  out[t] = fabs(b) > 1e-30f ? d / fabs(b) : d;\n"
         "}\n";
}

} // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
      std::fprintf(stderr, "no Metal device\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    std::printf("%-8s  %-14s  %s\n", "fn", "peak rel diff", "verdict");
    std::printf("%-8s  %-14s  %s\n", "--", "-------------", "-------");

    for (const Probe &p : kProbes) {
      NSError *err = nil;
      NSString *src = [NSString stringWithUTF8String:sourceFor(p).c_str()];
      id<MTLLibrary> lib = [dev newLibraryWithSource:src
                                             options:nil
                                               error:&err];
      if (!lib) {
        std::printf("%-8s  %-14s  %s\n", p.name, "-",
                    "no precise:: form, or would not compile");
        continue;
      }
      id<MTLFunction> fn = [lib newFunctionWithName:@"probe"];
      id<MTLComputePipelineState> pso =
          [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pso) {
        std::printf("%-8s  %-14s  %s\n", p.name, "-", "pipeline failed");
        continue;
      }

      id<MTLBuffer> out =
          [dev newBufferWithLength:kN * sizeof(float)
                           options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setBuffer:out offset:0 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(kN, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];

      const float *v = (const float *)out.contents;
      float peak = 0.0f;
      for (int i = 0; i < kN; ++i)
        if (v[i] == v[i] && v[i] > peak) // NaN-safe
          peak = v[i];

      const char *verdict =
          peak <= 2e-7f ? "default:: is fine" : "use precise::";
      std::printf("%-8s  %-14.3e  %s\n", p.name, (double)peak, verdict);
    }
  }
  return 0;
}
