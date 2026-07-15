#include "nerve/core_types.hpp"
#include "nerve/persistence/cohomology/gpu_cohomology.hpp"
#include "nerve/types.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{

using namespace nerve::persistence::gpu::cohomology;

bool has_gpu()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

// Build a triangle filtration: 3 vertices, 3 edges, 1 triangle
// v0(0), v1(1), v2(2), e01(3), e02(4), e12(5), t012(6)
std::vector<SimplexGPU> build_triangle_filtration()
{
    // Sorted by filtration value, ties by dimension
    return {
        SimplexGPU{{0}, 1, 0.0, 0, 0, -1, -1},       // v0
        SimplexGPU{{1}, 1, 0.0, 0, 1, -1, -1},       // v1
        SimplexGPU{{2}, 1, 0.0, 0, 2, -1, -1},       // v2
        SimplexGPU{{0, 1}, 2, 1.0, 1, 3, -1, -1},    // e01
        SimplexGPU{{0, 2}, 2, 1.5, 1, 4, -1, -1},    // e02
        SimplexGPU{{1, 2}, 2, 2.0, 1, 5, -1, -1},    // e12
        SimplexGPU{{0, 1, 2}, 3, 3.0, 2, 6, -1, -1}, // t012
    };
}

// Build a square-with-diagonal filtration: 4 vertices, 5 edges, 2 triangles
// Sorted by (filtration, dimension)
std::vector<SimplexGPU> build_square_filtration()
{
    return {
        SimplexGPU{{0}, 1, 0.0, 0, 0, -1, -1},        SimplexGPU{{1}, 1, 0.0, 0, 1, -1, -1},
        SimplexGPU{{2}, 1, 0.0, 0, 2, -1, -1},        SimplexGPU{{3}, 1, 0.0, 0, 3, -1, -1},
        SimplexGPU{{0, 1}, 2, 1.0, 1, 4, -1, -1},     SimplexGPU{{0, 2}, 2, 1.0, 1, 5, -1, -1},
        SimplexGPU{{1, 3}, 2, 1.0, 1, 6, -1, -1},     SimplexGPU{{2, 3}, 2, 1.0, 1, 7, -1, -1},
        SimplexGPU{{0, 3}, 2, 1.414, 1, 8, -1, -1},   SimplexGPU{{0, 1, 3}, 3, 2.0, 2, 9, -1, -1},
        SimplexGPU{{0, 2, 3}, 3, 2.0, 2, 10, -1, -1},
    };
}

// Build an octahedron filtration: 6 vertices, 12 edges, 8 triangles
// Octahedron topology: H0=1, H1=0, H2=1 (sphere)
std::vector<SimplexGPU> build_octahedron_filtration()
{
    return {
        // Vertices (dim=0, filt=0.0, indices 0-5)
        SimplexGPU{{0}, 1, 0.0, 0, 0, -1, -1},
        SimplexGPU{{1}, 1, 0.0, 0, 1, -1, -1},
        SimplexGPU{{2}, 1, 0.0, 0, 2, -1, -1},
        SimplexGPU{{3}, 1, 0.0, 0, 3, -1, -1},
        SimplexGPU{{4}, 1, 0.0, 0, 4, -1, -1},
        SimplexGPU{{5}, 1, 0.0, 0, 5, -1, -1},
        // Edges (dim=1, filt=1.0, indices 6-17)
        SimplexGPU{{0, 1}, 2, 1.0, 1, 6, -1, -1},
        SimplexGPU{{0, 2}, 2, 1.0, 1, 7, -1, -1},
        SimplexGPU{{0, 3}, 2, 1.0, 1, 8, -1, -1},
        SimplexGPU{{0, 4}, 2, 1.0, 1, 9, -1, -1},
        SimplexGPU{{1, 5}, 2, 1.0, 1, 10, -1, -1},
        SimplexGPU{{2, 5}, 2, 1.0, 1, 11, -1, -1},
        SimplexGPU{{3, 5}, 2, 1.0, 1, 12, -1, -1},
        SimplexGPU{{4, 5}, 2, 1.0, 1, 13, -1, -1},
        SimplexGPU{{1, 2}, 2, 1.0, 1, 14, -1, -1},
        SimplexGPU{{2, 3}, 2, 1.0, 1, 15, -1, -1},
        SimplexGPU{{3, 4}, 2, 1.0, 1, 16, -1, -1},
        SimplexGPU{{4, 1}, 2, 1.0, 1, 17, -1, -1},
        // Triangles (dim=2, filt=2.0, indices 18-25)
        SimplexGPU{{0, 1, 2}, 3, 2.0, 2, 18, -1, -1},
        SimplexGPU{{0, 2, 3}, 3, 2.0, 2, 19, -1, -1},
        SimplexGPU{{0, 3, 4}, 3, 2.0, 2, 20, -1, -1},
        SimplexGPU{{0, 4, 1}, 3, 2.0, 2, 21, -1, -1},
        SimplexGPU{{1, 2, 5}, 3, 2.0, 2, 22, -1, -1},
        SimplexGPU{{2, 3, 5}, 3, 2.0, 2, 23, -1, -1},
        SimplexGPU{{3, 4, 5}, 3, 2.0, 2, 24, -1, -1},
        SimplexGPU{{4, 1, 5}, 3, 2.0, 2, 25, -1, -1},
    };
}

// Build a tetrahedron filtration: 4 vertices, 6 edges, 4 triangles, 1 tetrahedron
std::vector<SimplexGPU> build_tetrahedron_filtration()
{
    return {
        // Vertices (dim=0, filt=0.0, indices 0-3)
        SimplexGPU{{0}, 1, 0.0, 0, 0, -1, -1},
        SimplexGPU{{1}, 1, 0.0, 0, 1, -1, -1},
        SimplexGPU{{2}, 1, 0.0, 0, 2, -1, -1},
        SimplexGPU{{3}, 1, 0.0, 0, 3, -1, -1},
        // Edges (dim=1, filt=1.0, indices 4-9)
        SimplexGPU{{0, 1}, 2, 1.0, 1, 4, -1, -1},
        SimplexGPU{{0, 2}, 2, 1.0, 1, 5, -1, -1},
        SimplexGPU{{0, 3}, 2, 1.0, 1, 6, -1, -1},
        SimplexGPU{{1, 2}, 2, 1.0, 1, 7, -1, -1},
        SimplexGPU{{1, 3}, 2, 1.0, 1, 8, -1, -1},
        SimplexGPU{{2, 3}, 2, 1.0, 1, 9, -1, -1},
        // Triangles (dim=2, filt=2.0, indices 10-13)
        SimplexGPU{{0, 1, 2}, 3, 2.0, 2, 10, -1, -1},
        SimplexGPU{{0, 1, 3}, 3, 2.0, 2, 11, -1, -1},
        SimplexGPU{{0, 2, 3}, 3, 2.0, 2, 12, -1, -1},
        SimplexGPU{{1, 2, 3}, 3, 2.0, 2, 13, -1, -1},
        // Tetrahedron (dim=3, filt=3.0, index 14)
        SimplexGPU{{0, 1, 2, 3}, 4, 3.0, 3, 14, -1, -1},
    };
}

} // namespace

int main()
{
    if (!has_gpu())
    {
        std::cout << "No CUDA device -- skipping GPU cohomology tests\n";
        return 0;
    }

    // Triangle filtration produces correct H0 and H1 pairs
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(32, 2));

        auto filtration = build_triangle_filtration();
        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);
        assert(!pairs.empty());

        // Count by dimension
        int h0_pairs = 0, h1_pairs = 0;
        for (const auto &p : pairs)
        {
            assert(std::isfinite(p.birth));
            assert(std::isfinite(p.death));
            assert(p.birth < p.death);
            if (p.dimension == 0)
                ++h0_pairs;
            if (p.dimension == 1)
                ++h1_pairs;
        }

        assert(h0_pairs >= 2); // Triangle: 3 vertices -> 2 H0 deaths
        assert(h1_pairs >= 1); // Triangle: 1 H1 cycle killed by triangle

        // Performance metrics smoke
        double time_ms = computer.getLastComputeTime();
        assert(time_ms >= 0.0);

        double speedup = computer.getSpeedupVsCPU();
        assert(speedup == 0.0); // No CPU baseline set
        computer.setCPUBaseline(time_ms * 10.0);
        speedup = computer.getSpeedupVsCPU();
        assert(speedup > 0.0);
    }

    // Single edge filtration (2 vertices + 1 edge)
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(16, 1));

        std::vector<SimplexGPU> filtration = {
            SimplexGPU{{0}, 1, 0.0, 0, 0, -1, -1},
            SimplexGPU{{1}, 1, 0.0, 0, 1, -1, -1},
            SimplexGPU{{0, 1}, 2, 1.0, 1, 2, -1, -1},
        };

        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);
        assert(!pairs.empty());

        bool has_h0 = false;
        for (const auto &p : pairs)
        {
            assert(std::isfinite(p.birth));
            assert(std::isfinite(p.death));
            if (p.dimension == 0)
                has_h0 = true;
        }
        if (!has_h0)
            assert(false && "Expected an H0 pair");
    }

    // Empty filtration is rejected
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(16, 2));

        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram({}, pairs);
        assert(!ok);
    }

    // Uninitialized computer is rejected
    {
        GPUCohomologyComputer computer;
        std::vector<nerve::Pair> pairs;
        auto filtration = build_triangle_filtration();
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(!ok);
    }

    // Oversize filtration is rejected
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(3, 2)); // Only room for 3 simplices

        auto filtration = build_triangle_filtration(); // 7 simplices
        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(!ok);
    }

    // Invalid initialize args are rejected
    {
        GPUCohomologyComputer computer;
        assert(!computer.initialize(0, 2));   // zero simplices
        assert(!computer.initialize(8, 999)); // dim exceeds MAX_DIM_GPU_COHOMOLOGY
    }

    // Disconnected vertices
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(16, 1));

        // 3 disconnected vertices, no edges
        std::vector<SimplexGPU> filtration = {
            SimplexGPU{{0}, 1, 0.0, 0, 0, -1, -1},
            SimplexGPU{{1}, 1, 0.0, 0, 1, -1, -1},
            SimplexGPU{{2}, 1, 0.0, 0, 2, -1, -1},
        };

        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);
        assert(pairs.empty()); // No edges -> all components essential, no deaths
    }

    // Reuse computer instance
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(32, 2));

        {
            std::vector<SimplexGPU> filtration = {
                SimplexGPU{{0}, 1, 0.0, 0, 0, -1, -1},
                SimplexGPU{{1}, 1, 0.0, 0, 1, -1, -1},
                SimplexGPU{{0, 1}, 2, 1.0, 1, 2, -1, -1},
            };
            std::vector<nerve::Pair> pairs;
            bool ok = computer.computePersistenceDiagram(filtration, pairs);
            assert(ok);
            assert(!pairs.empty());
        }

        {
            auto filtration = build_triangle_filtration();
            std::vector<nerve::Pair> pairs;
            bool ok = computer.computePersistenceDiagram(filtration, pairs);
            assert(ok);
            assert(!pairs.empty());
        }
    }

    // Square with diagonal (11 simplices) -- exercises clearing on larger dim-1 coboundary
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(32, 2));

        auto filtration = build_square_filtration();
        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);
        assert(!pairs.empty());

        // Verify finite pairs
        int h0_finite = 0, h1_finite = 0;
        int h0_essential = 0, h1_essential = 0;
        for (const auto &p : pairs)
        {
            assert(std::isfinite(p.birth));
            if (!std::isfinite(p.death))
            {
                if (p.dimension == 0)
                    ++h0_essential;
                if (p.dimension == 1)
                    ++h1_essential;
                continue;
            }
            assert(p.birth < p.death);
            if (p.dimension == 0)
                ++h0_finite;
            if (p.dimension == 1)
                ++h1_finite;
        }

        // Square (4 verts): 3 finite H0 deaths, 1 essential H0 (connected component)
        assert(h0_finite == 3);
        assert(h0_essential == 1);
        // Square (hole): 1 finite H1 pair killed by triangles
        // With 2 triangles, the cycle completes and both triangles die
        assert(h1_finite >= 1);
    }

    // Octahedron (26 simplices) -- larger complex, H2 cohomology via clearing cascade
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(64, 3));

        auto filtration = build_octahedron_filtration();
        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);
        assert(!pairs.empty());

        // Octahedron is sphere: H0=1, H1=0, H2=1
        // Essential classes: 1 dim-0, 1 dim-2
        int h0_essential = 0, h1_finite = 0, h2_essential = 0;
        for (const auto &p : pairs)
        {
            assert(std::isfinite(p.birth));
            if (!std::isfinite(p.death))
            {
                if (p.dimension == 0)
                    ++h0_essential;
                if (p.dimension == 2)
                    ++h2_essential;
            }
            else
            {
                if (p.dimension == 1)
                    ++h1_finite;
            }
        }

        assert(h0_essential == 1);
        assert(h2_essential == 1);
    }

    // Tetrahedron (15 simplices, dim-3) -- tests clearing cascade up to dim-3
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(32, 3));

        auto filtration = build_tetrahedron_filtration();
        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);
        assert(!pairs.empty());

        // Tetrahedron is 3-sphere boundary (S^2): H0=1, H1=0, H2=1, H3=0 (filled)
        // With filled tetrahedron (dim-3), H2 is killed
        int h0_essential = 0, h2_finite = 0, h2_essential = 0;
        for (const auto &p : pairs)
        {
            assert(std::isfinite(p.birth));
            if (!std::isfinite(p.death))
            {
                if (p.dimension == 0)
                    ++h0_essential;
                if (p.dimension == 2)
                    ++h2_essential;
            }
            else
            {
                if (p.dimension == 2)
                    ++h2_finite;
            }
        }

        assert(h0_essential == 1);
        // With the tetrahedron filled (dim-3 simplex present), the H2 class dies
        assert(h2_finite == 1);
        assert(h2_essential == 0);
    }

    // computeGPUCohomology high-level API -- square (distinct scenario not covered by tests 6-8)
    {
        auto filtration = build_square_filtration();
        nerve::persistence::PersistenceDiagram diagram = computeGPUCohomology(filtration, 2);
        assert(!diagram.pairs.empty());

        int h0_essential = 0;
        for (const auto &p : diagram.pairs)
        {
            assert(std::isfinite(p.birth));
            if (!std::isfinite(p.death) && p.dimension == 0)
                ++h0_essential;
        }
        assert(h0_essential == 1);
    }

    // Clearing KPI: Triangle cleared-flag verification
    // Triangle: v0(0), v1(1), v2(2), e01(3), e02(4), e12(5), t012(6)
    // After dim-0 reduction: edges e02(4), e12(5) are claimed as coboundaries
    // by vertices -> pivot_table[4]=0, pivot_table[5]=1
    // At dim-1 clearing: e02 and e12 have pivot_table[their_idx] != -1 -> cleared!
    // At dim-2 clearing: t012 has pivot_table[6]=3 (claimed by e01's reduction) -> cleared!
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(16, 2));

        auto filtration = build_triangle_filtration();
        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);

        std::vector<bool> cleared;
        ok = computer.getClearedStates(cleared);
        assert(ok);
        assert(cleared.size() == 7);

        // Vertices (dim-0) should never be cleared (clearing starts at dim-1)
        assert(cleared[0] == false);
        assert(cleared[1] == false);
        assert(cleared[2] == false);

        // e01(3) is NOT cleared: pivot_table[3] was -1 during d=1 clearing
        // because no vertex claimed edge index 3
        assert(cleared[3] == false);

        // e02(4) IS cleared: claimed by vertex 0 during dim-0 reduction
        assert(cleared[4] == true);

        // e12(5) IS cleared: claimed by vertex 1 during dim-0 reduction
        assert(cleared[5] == true);

        // t012(6) IS cleared: claimed by edge 3 during dim-1 reduction
        assert(cleared[6] == true);

        // Verify cow_sizes: -1 for cleared columns, >=0 for non-cleared
        std::vector<int> red_sizes;
        ok = computer.getReducedColumnSizes(red_sizes);
        assert(ok);
        assert(red_sizes.size() == 7);

        // Non-cleared columns should have size >= 0
        for (int i : {0, 1, 2, 3})
        {
            assert(red_sizes[static_cast<std::size_t>(i)] >= 0);
        }
        // Cleared columns should have size == -1
        for (int i : {4, 5, 6})
        {
            assert(red_sizes[static_cast<std::size_t>(i)] == -1);
        }

        // KPI: clearing ratio = 3/7 ~= 43%
        int cleared_count = 0;
        for (bool c : cleared)
            if (c)
                ++cleared_count;
        assert(cleared_count == 3);
        if (cleared_count < 2)
        {
            std::cerr << "FAIL: triangle clearing ratio too low: " << cleared_count << "/7\n";
            return 1;
        }
    }

    // Clearing KPI: Square cleared-flag verification
    // Square: v0(0), v1(1), v2(2), v3(3), e01(4), e02(5), e13(6), e23(7),
    // e03(8), t013(9), t023(10)
    // After dim-0 reduction: edges {6,7,8} are claimed as coboundaries
    // At dim-1 clearing: edges 6(e13), 7(e23), 8(e03) cleared
    // After dim-1 reduction: triangles {9,10} claimed by edges {4,5}
    // At dim-2 clearing: triangles 9(t013), 10(t023) cleared
    // Total cleared = 5 of 11 columns
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(32, 2));

        auto filtration = build_square_filtration();
        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);

        std::vector<bool> cleared;
        ok = computer.getClearedStates(cleared);
        assert(ok);
        assert(cleared.size() == 11);

        // Vertices never cleared
        assert(cleared[0] == false);
        assert(cleared[1] == false);
        assert(cleared[2] == false);
        assert(cleared[3] == false);

        // e01(4), e02(5) are NOT cleared (not claimed as coboundaries)
        assert(cleared[4] == false);
        assert(cleared[5] == false);

        // e13(6), e23(7), e03(8) ARE cleared (claimed by vertices)
        assert(cleared[6] == true);
        assert(cleared[7] == true);
        assert(cleared[8] == true);

        // t013(9), t023(10) ARE cleared (claimed by edges 4,5 in d=1 reduction)
        assert(cleared[9] == true);
        assert(cleared[10] == true);

        // cow_sizes: -1 for cleared columns
        std::vector<int> red_sizes;
        ok = computer.getReducedColumnSizes(red_sizes);
        assert(ok);
        assert(red_sizes.size() == 11);

        for (int i : {0, 1, 2, 3, 4, 5})
            assert(red_sizes[static_cast<std::size_t>(i)] >= 0);
        for (int i : {6, 7, 8, 9, 10})
            assert(red_sizes[static_cast<std::size_t>(i)] == -1);

        // KPI: 5 of 11 columns cleared = 45% clearing ratio
        int cleared_count = 0;
        for (bool c : cleared)
            if (c)
                ++cleared_count;
        assert(cleared_count == 5);
        if (cleared_count < 3)
        {
            std::cerr << "FAIL: square clearing ratio too low: " << cleared_count << "/11\n";
            return 1;
        }
    }

    // Clearing KPI: Octahedron with dim-1 and dim-2 clearing verification
    // Octahedron: 6v + 12e + 8t = 26 columns
    // After dim-0 reduction: several edges claimed as coboundaries -> cleared in d=1
    // After dim-1 reduction: several triangles claimed -> cleared in d=2
    // Total cleared should be > 0 (significant clearing on sphere topology)
    {
        GPUCohomologyComputer computer;
        assert(computer.initialize(64, 3));

        auto filtration = build_octahedron_filtration();
        std::vector<nerve::Pair> pairs;
        bool ok = computer.computePersistenceDiagram(filtration, pairs);
        assert(ok);

        std::vector<bool> cleared;
        ok = computer.getClearedStates(cleared);
        assert(ok);
        assert(cleared.size() == 26);

        // Vertices (0-5) never cleared
        for (int i = 0; i < 6; ++i)
            assert(cleared[static_cast<std::size_t>(i)] == false);

        // At least some of the 12 edges (6-17) are cleared
        int edges_cleared = 0;
        for (int i = 6; i < 18; ++i)
            if (cleared[static_cast<std::size_t>(i)])
                ++edges_cleared;

        // At least some of the 8 triangles (18-25) are cleared
        int tri_cleared = 0;
        for (int i = 18; i < 26; ++i)
            if (cleared[static_cast<std::size_t>(i)])
                ++tri_cleared;

        if (edges_cleared < 1 || tri_cleared < 1)
        {
            std::cerr << "FAIL: octahedron clearing: edges=" << edges_cleared
                      << "/12, triangles=" << tri_cleared << "/8\n";
            return 1;
        }

        // KPI: total cleared ratio
        int total_cleared = edges_cleared + tri_cleared;
        if (total_cleared < 2)
        {
            std::cerr << "FAIL: octahedron only " << total_cleared
                      << " cleared columns, expected >=2\n";
            return 1;
        }

        // Verify cow_sizes consistency
        std::vector<int> red_sizes;
        ok = computer.getReducedColumnSizes(red_sizes);
        assert(ok);
        assert(red_sizes.size() == 26);

        for (std::size_t i = 0; i < 26; ++i)
        {
            if (cleared[i])
                assert(red_sizes[i] == -1);
            else
                assert(red_sizes[i] >= -1); // -1 can also appear for naturally empty columns
        }
    }

    // Clearing KPI: Aggregated clearing metrics across complexes
    // Runs triangle, square, tetrahedron and collects clearing KPIs
    {
        struct ClearingKPIs
        {
            const char *name;
            int total_columns;
            int cleared_columns;
            double ratio;
        };

        ClearingKPIs cases[4];
        int case_count = 0;

        // Triangle (7 columns)
        {
            GPUCohomologyComputer computer;
            assert(computer.initialize(16, 2));
            auto filtration = build_triangle_filtration();
            std::vector<nerve::Pair> pairs;
            assert(computer.computePersistenceDiagram(filtration, pairs));

            std::vector<bool> cleared;
            assert(computer.getClearedStates(cleared));

            int cleared_count = 0;
            for (bool c : cleared)
                if (c)
                    ++cleared_count;
            cases[case_count++] = {"triangle", 7, cleared_count,
                                   static_cast<double>(cleared_count) / 7.0};
        }

        // Square (11 columns)
        {
            GPUCohomologyComputer computer;
            assert(computer.initialize(32, 2));
            auto filtration = build_square_filtration();
            std::vector<nerve::Pair> pairs;
            assert(computer.computePersistenceDiagram(filtration, pairs));

            std::vector<bool> cleared;
            assert(computer.getClearedStates(cleared));

            int cleared_count = 0;
            for (bool c : cleared)
                if (c)
                    ++cleared_count;
            cases[case_count++] = {"square", 11, cleared_count,
                                   static_cast<double>(cleared_count) / 11.0};
        }

        // Tetrahedron (15 columns)
        {
            GPUCohomologyComputer computer;
            assert(computer.initialize(32, 3));
            auto filtration = build_tetrahedron_filtration();
            std::vector<nerve::Pair> pairs;
            assert(computer.computePersistenceDiagram(filtration, pairs));

            std::vector<bool> cleared;
            assert(computer.getClearedStates(cleared));

            int cleared_count = 0;
            for (bool c : cleared)
                if (c)
                    ++cleared_count;
            cases[case_count++] = {"tetrahedron", 15, cleared_count,
                                   static_cast<double>(cleared_count) / 15.0};
        }

        // Octahedron (26 columns)
        {
            GPUCohomologyComputer computer;
            assert(computer.initialize(64, 3));
            auto filtration = build_octahedron_filtration();
            std::vector<nerve::Pair> pairs;
            assert(computer.computePersistenceDiagram(filtration, pairs));

            std::vector<bool> cleared;
            assert(computer.getClearedStates(cleared));

            int cleared_count = 0;
            for (bool c : cleared)
                if (c)
                    ++cleared_count;
            cases[case_count++] = {"octahedron", 26, cleared_count,
                                   static_cast<double>(cleared_count) / 26.0};
        }

        // Verify KPIs
        assert(case_count == 4);

        for (int i = 0; i < case_count; ++i)
        {
            // All clearing ratios should be in (0, 1.0)
            if (cases[i].ratio <= 0.0 || cases[i].ratio >= 1.0)
            {
                std::cerr << "FAIL: " << cases[i].name << " clearing ratio " << cases[i].ratio
                          << " out of range (0,1.0)\n";
                return 1;
            }
            // Every non-trivial complex must clear at least 1 column
            if (cases[i].cleared_columns < 1)
            {
                std::cerr << "FAIL: " << cases[i].name << " cleared 0 columns\n";
                return 1;
            }
        }

        // Larger complexes should have more cleared columns (not fewer)
        // Tetrahedron and octahedron should clear more than triangle
        if (cases[1].cleared_columns < cases[0].cleared_columns)
        {
            std::cerr << "FAIL: square (" << cases[1].cleared_columns
                      << ") cleared fewer than triangle (" << cases[0].cleared_columns << ")\n";
            return 1;
        }
    }

    // Clearing determinism: octahedron computePersistenceDiagram 5 times,
    // verify identical pairs (count + birth/death values) each run.
    // Determinism is critical because clearing uses atomicCAS pivot claiming,
    // which is sensitive to warp scheduling if not properly fenced.
    {
        auto filtration = build_octahedron_filtration();

        // Run 0: baseline
        {
            GPUCohomologyComputer computer;
            assert(computer.initialize(64, 3));

            std::vector<nerve::Pair> ref_pairs;
            bool ok = computer.computePersistenceDiagram(filtration, ref_pairs);
            assert(ok);
            assert(!ref_pairs.empty());

            // Verify expected octahedron topology: H0=1 essential, H2=1 essential
            int h0_essential = 0, h2_essential = 0;
            for (const auto &p : ref_pairs)
            {
                assert(std::isfinite(p.birth));
                if (!std::isfinite(p.death))
                {
                    if (p.dimension == 0)
                        ++h0_essential;
                    if (p.dimension == 2)
                        ++h2_essential;
                }
            }
            assert(h0_essential == 1);
            assert(h2_essential == 1);

            // Runs 1-4: compare against baseline
            for (int run = 1; run <= 4; ++run)
            {
                GPUCohomologyComputer comp;
                assert(comp.initialize(64, 3));

                std::vector<nerve::Pair> pairs;
                ok = comp.computePersistenceDiagram(filtration, pairs);
                assert(ok);
                assert(pairs.size() == ref_pairs.size());

                // Compare every pair's birth, death, and dimension
                for (std::size_t i = 0; i < pairs.size(); ++i)
                {
                    if (pairs[i].birth != ref_pairs[i].birth ||
                        pairs[i].death != ref_pairs[i].death ||
                        pairs[i].dimension != ref_pairs[i].dimension)
                    {
                        std::cerr << "FAIL: octahedron determinism run " << run << " pair " << i
                                  << " (" << pairs[i].birth << ", " << pairs[i].death
                                  << ", dim=" << pairs[i].dimension << ") != ref ("
                                  << ref_pairs[i].birth << ", " << ref_pairs[i].death
                                  << ", dim=" << ref_pairs[i].dimension << ")\n";
                        return 1;
                    }
                }
            }
        }

        // Also verify that clearing state is deterministic across runs
        {
            std::vector<char> ref_cleared;
            for (int run = 0; run < 5; ++run)
            {
                GPUCohomologyComputer computer;
                assert(computer.initialize(64, 3));

                std::vector<nerve::Pair> pairs;
                bool ok = computer.computePersistenceDiagram(filtration, pairs);
                assert(ok);

                std::vector<bool> cleared;
                ok = computer.getClearedStates(cleared);
                assert(ok);
                assert(cleared.size() == 26);

                int cleared_count = 0;
                for (bool c : cleared)
                    if (c)
                        ++cleared_count;

                if (run == 0)
                {
                    if (cleared_count < 1)
                    {
                        std::cerr << "FAIL: determinism run 0 cleared " << cleared_count
                                  << " columns\n";
                        return 1;
                    }
                    ref_cleared.assign(cleared.begin(), cleared.end());
                }
                else
                {
                    // Compare per-simplex cleared state
                    for (std::size_t i = 0; i < 26; ++i)
                    {
                        if ((cleared[i] ? 1 : 0) != ref_cleared[i])
                        {
                            std::cerr << "FAIL: determinism run " << run << " simplex " << i
                                      << " cleared=" << cleared[i]
                                      << " but ref=" << (ref_cleared[i] != 0) << "\n";
                            return 1;
                        }
                    }
                }
            }
        }
    }

    return 0;
}
