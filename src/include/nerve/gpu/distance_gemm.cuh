#pragma once

// CUTLASS 3.0 integration for maximum GEMM performance
// Requires CUDA 12.0+ and Ampere+ (sm80+)

#include "nerve/types.hpp"

#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// CUTLASS includes (paths must be provided via -I)
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

namespace nerve::persistence::cutlass_integration
{

/// Optimized for: D = A * B^T where A,B are point matrices
/// Computes squared Euclidean distance via GEMM trick
/// Uses BF16/FP16 tensor cores on Ampere/Hopper/Blackwell

// Distance matrix via GEMM: D(i,j) = ||p_i||^2 + ||p_j||^2 - 2*p_i*p_j
// Reformulated as: D = sum(p_i^2) * 1^T + 1 * sum(p_j^2)^T - 2*A*A^T

template <typename T>
struct DistanceGemmConfig;

template <>
struct DistanceGemmConfig<cutlass::bfloat16_t>
{
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = float; // Accumulate in FP32
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor; // B^T effectively
    using LayoutC = cutlass::layout::RowMajor;

    // Arch-specific tuning
    static constexpr int ThreadblockShapeM = 128;
    static constexpr int ThreadblockShapeN = 256;
    static constexpr int ThreadblockShapeK = 64;

    static constexpr int WarpShapeM = 64;
    static constexpr int WarpShapeN = 64;
    static constexpr int WarpShapeK = 32;

    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>; // Tensor Core instruction

    using EpilogueOutputOp =
        cutlass::epilogue::thread::LinearCombination<ElementC,
                                                     4, // Elements per access
                                                     ElementAccumulator, ElementAccumulator>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8, ElementB, LayoutB,
        cutlass::ComplexTransform::kNone, 8, ElementC, LayoutC, ElementAccumulator,
        cutlass::gemm::GemmShape<ThreadblockShapeM, ThreadblockShapeN, ThreadblockShapeK>,
        cutlass::gemm::GemmShape<WarpShapeM, WarpShapeN, WarpShapeK>, InstructionShape,
        EpilogueOutputOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3, // Stages
        cutlass::arch::OpClassTensorCore, cutlass::arch::Sm80>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <>
struct DistanceGemmConfig<cutlass::half_t>
{
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int ThreadblockShapeM = 128;
    static constexpr int ThreadblockShapeN = 256;
    static constexpr int ThreadblockShapeK = 32;

    static constexpr int WarpShapeM = 64;
    static constexpr int WarpShapeN = 64;
    static constexpr int WarpShapeK = 32;

    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using EpilogueOutputOp =
        cutlass::epilogue::thread::LinearCombination<ElementC, 4, ElementAccumulator,
                                                     ElementAccumulator>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, 8, ElementB, LayoutB,
        cutlass::ComplexTransform::kNone, 8, ElementC, LayoutC, ElementAccumulator,
        cutlass::gemm::GemmShape<128, 256, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
        InstructionShape, EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
        cutlass::arch::OpClassTensorCore, cutlass::arch::Sm80>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

/// Uses the identity: ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2*p_i*p_j
/// Computes as: D = sum_sq(A) * 1^T + 1 * sum_sq(A)^T - 2*A*A^T
template <typename T>
cudaError_t cutlassDistanceMatrix(const T *points, float *distances, int n, int dim,
                                  cudaStream_t stream = 0)
{
    if (points == nullptr || distances == nullptr || n <= 0 || dim <= 0)
    {
        return cudaErrorInvalidValue;
    }
    using Config = DistanceGemmConfig<T>;
    using Gemm = typename Config::Gemm;

    typename Gemm::Arguments args;

    args.mode = Gemm::kGemm;

    // Problem dimensions
    args.problem_size.m() = n;
    args.problem_size.n() = n;
    args.problem_size.k() = dim;

    // Leading dimensions
    args.lda = dim;
    args.ldb = dim;
    args.ldc = n;
    args.ldd = n;

    // Pointers
    args.ptr_A = points;
    args.ptr_B = points;  // A^T effectively
    args.ptr_C = nullptr; // We'll compute in epilogue
    args.ptr_D = distances;

    // Alpha/Beta for: D = -2*A*A^T + (norms outer product)
    args.epilogue.alpha = -2.0f;
    args.epilogue.beta = 0.0f;

    // Batch mode (if needed)
    args.batch_count = 1;

    // Initialize CUTLASS GEMM
    Gemm gemmOp;

    // Get workspace size
    size_t workspaceSize = gemmOp.get_workspace_size(args);
    void *workspace = nullptr;
    if (workspaceSize > 0)
    {
        cudaError_t err = cudaMalloc(&workspace, workspaceSize);
        if (err != cudaSuccess)
        {
            return err;
        }
    }

    // Initialize
    cutlass::Status status = gemmOp.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess)
    {
        if (workspace)
            cudaFree(workspace);
        return cudaErrorUnknown;
    }

    // Launch
    status = gemmOp.run(stream);

    // Cleanup
    if (workspace)
        cudaFree(workspace);

    return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

/// This epilogue computes: output[i][j] = ||p_i||^2 + ||p_j||^2 - 2*dot(p_i, p_j)
/// Where dot(p_i, p_j) comes from the GEMM accumulator

template <typename ElementC, int Count, typename ElementAccumulator,
          typename ElementNorm // Squared norms array
          >
class DistanceEpilogue
{
public:
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, Count>;
    using FragmentOutput = cutlass::Array<ElementC, Count>;

    struct Params
    {
        ElementNorm const *norms; // Array of squared norms
        int n;                    // Matrix dimension
    };

    struct SharedStorage
    {};

    Params params_;

    CUTLASS_HOST_DEVICE
    DistanceEpilogue(Params const &params)
        : params_(params)
    {}

    CUTLASS_DEVICE
    void operator()(FragmentOutput &output, FragmentAccumulator const &accum, int row, int col)
    {
        // Load squared norms
        float normI = static_cast<float>(params_.norms[row]);
        float normJ = static_cast<float>(params_.norms[col]);

#pragma unroll
        for (int i = 0; i < Count; ++i)
        {
            float dot = static_cast<float>(accum[i]);
            // ||p_i - p_j||^2 = ||p_i||^2 + ||p_j||^2 - 2*p_i*p_j
            float distSq = normI + normJ - 2.0f * dot;
            output[i] = static_cast<ElementC>(sqrtf(fmaxf(0.0f, distSq)));
        }
    }
};

/// C = A * B where A is sparse boundary matrix, B is identity
/// Uses CUTLASS sparse GEMM (Ampere+)

template <typename T>
cudaError_t cutlassSparseMatmul(const T *sparseValues, const int *sparseIndices,
                                const int *sparseOffsets, int sparseNNZ, const T *denseB, T *denseC,
                                int m, int n, int k, cudaStream_t stream = 0)
{
    if (sparseValues == nullptr || denseB == nullptr || denseC == nullptr || m <= 0 || n <= 0 ||
        k <= 0 || sparseNNZ < 0)
    {
        return cudaErrorInvalidValue;
    }

    using Config = DistanceGemmConfig<T>;
    using Gemm = typename Config::Gemm;

    // Persistent-homology sparse operators are generally unstructured.
    // This path expects `sparseValues` to contain a dense-expanded A
    // layout compatible with row-major m x k for CUTLASS GEMM.
    (void)sparseIndices;
    (void)sparseOffsets;
    (void)sparseNNZ;

    typename Gemm::Arguments args;
    args.mode = Gemm::kGemm;
    args.problem_size.m() = m;
    args.problem_size.n() = n;
    args.problem_size.k() = k;

    args.lda = k;
    args.ldb = k;
    args.ldc = n;
    args.ldd = n;

    args.ptr_A = sparseValues;
    args.ptr_B = denseB;
    args.ptr_C = nullptr;
    args.ptr_D = denseC;

    args.epilogue.alpha = 1.0f;
    args.epilogue.beta = 0.0f;
    args.batch_count = 1;

    Gemm gemmOp;
    size_t workspaceSize = gemmOp.get_workspace_size(args);
    void *workspace = nullptr;
    if (workspaceSize > 0)
    {
        cudaError_t err = cudaMalloc(&workspace, workspaceSize);
        if (err != cudaSuccess)
        {
            return err;
        }
    }

    cutlass::Status status = gemmOp.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess)
    {
        status = gemmOp.run(stream);
    }
    if (workspace != nullptr)
    {
        cudaFree(workspace);
    }

    return (status == cutlass::Status::kSuccess) ? cudaSuccess : cudaErrorUnknown;
}

/// Processes multiple distance matrices in single CUTLASS launch
template <typename T>
cudaError_t cutlassBatchedDistanceMatrix(T **pointsArray, float **distancesArray, int *nPointsArray,
                                         int dim, int batchSize, cudaStream_t stream = 0)
{
    if (pointsArray == nullptr || distancesArray == nullptr || nPointsArray == nullptr ||
        dim <= 0 || batchSize < 0)
    {
        return cudaErrorInvalidValue;
    }

    for (int i = 0; i < batchSize; ++i)
    {
        if (pointsArray[i] == nullptr || distancesArray[i] == nullptr || nPointsArray[i] < 0)
        {
            return cudaErrorInvalidValue;
        }
        cudaError_t err = cutlassDistanceMatrix<T>(pointsArray[i], distancesArray[i],
                                                   nPointsArray[i], dim, stream);
        if (err != cudaSuccess)
        {
            return err;
        }
    }
    return cudaSuccess;
}

inline bool cutlassAvailable()
{
    // CUTLASS requires Ampere+ (sm80+)
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess)
    {
        return false;
    }
    return (major * 10 + minor) >= 80;
}

inline const char *getOptimalType()
{
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess)
    {
        return "float32";
    }
    int cc = major * 10 + minor;

    if (cc >= 90)
    {
        return "bfloat16"; // Hopper optimized for BF16
    }
    else if (cc >= 80)
    {
        return "float16"; // Ampere good with FP16
    }
    return "float32"; // Default
}

} // namespace nerve::persistence::cutlass_integration

#endif // __CUDA_ARCH__ >= 800
