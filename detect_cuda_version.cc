#include <cuda.h>
#include <cstdio>

int main() {
  constexpr int major = CUDA_VERSION / 1000;
  constexpr int minor = (CUDA_VERSION / 10) % 100;

  std::printf("%d.%d\n", major, minor);
  return 0;
}
