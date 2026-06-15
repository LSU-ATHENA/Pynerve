
#pragma once
#include "nerve/algebra/complex.hpp"
#include "nerve/algebra/simplex.hpp"
#include "nerve/core_types.hpp"
#include "nerve/persistence/core/core_types.hpp"
#include "nerve/persistence/streaming/streaming_reducer.hpp"
#include "nerve/persistence/utils/exact_engine.hpp"

#include <deque>
#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

namespace nerve::streaming
{
using algebra::Simplex;
using algebra::SimplicialComplex;
using persistence::Diagram;
using Pair = persistence::Pair; // Use persistence Pair type

class IncrementalPH
{
public:
    IncrementalPH();
    explicit IncrementalPH(Size max_dimension);
    void addSimplex(const Simplex &simplex);
    void removeSimplex(const Simplex &simplex);
    void addSimplices(const std::vector<Simplex> &simplices);
    void removeSimplices(const std::vector<Simplex> &simplices);
    void addComplex(const SimplicialComplex &complex);
    void addFiltrationStep(const std::vector<std::pair<Simplex, double>> &filtration_step);
    Diagram getPersistenceDiagram() const;
    std::vector<Pair> getPersistencePairs() const;
    std::vector<Size> getBettiNumbers() const;
    void add_simplex(const Simplex &simplex) { addSimplex(simplex); }
    void remove_simplex(const Simplex &simplex) { removeSimplex(simplex); }
    std::vector<Pair> get_persistence_pairs() const { return getPersistencePairs(); }
    Size numSimplices() const;
    Size getMaxDimension() const;
    void reset();
    void clear();
    void setMaxDimension(Size max_dim);
    void setCoefficientField(int p);
    void setAlgorithm(const std::string &algorithm);
    double getComputationTime() const;
    Size getNumUpdates() const;

private:
    Size max_dimension_;
    int coefficient_field_;
    std::string algorithm_;
    SimplicialComplex current_complex_;
    std::vector<Pair> current_pairs_;
    std::vector<Size> current_betti_;
    std::map<Simplex, Size> simplex_to_index_;
    std::vector<Simplex> index_to_simplex_;
    std::vector<bool> is_active_;
    std::vector<double> filtration_values_;
    double total_computation_time_;
    Size num_updates_;
    double next_filtration_value_;
    persistence::IncrementalExactZ2 incremental_engine_;
    bool incremental_engine_ready_;
    void initializeIncrementalStructures();
    void updatePersistenceAfterAddition(const Simplex &simplex);
    void updatePersistenceAfterRemoval(const Simplex &simplex);
    void recomputePersistence();
    Size getSimplexIndex(const Simplex &simplex) const;
    bool isSimplexPresent(const Simplex &simplex) const;
    void updateBettiNumbers();
};

class ZigzagPH
{
public:
    ZigzagPH();
    explicit ZigzagPH(Size max_dimension);
    void addSimplex(const Simplex &simplex);
    void removeSimplex(const Simplex &simplex);
    void addFiltrationStep(const std::vector<std::pair<Simplex, double>> &step);
    void removeFiltrationStep(const std::vector<std::pair<Simplex, double>> &step);
    std::vector<std::vector<Pair>> getZigzagPersistence() const;
    Diagram getCurrentPersistence() const;
    void add_simplex(const Simplex &simplex) { addSimplex(simplex); }
    void remove_simplex(const Simplex &simplex) { removeSimplex(simplex); }
    Diagram get_current_persistence() const { return getCurrentPersistence(); }
    void set_witness_mode(bool witness_mode) { setWitnessMode(witness_mode); }
    void set_sparse_mode(bool sparse_mode) { setSparseMode(sparse_mode); }
    void set_deterministic_seed(uint32_t seed) { setDeterministicSeed(seed); }
    Size currentStep() const;
    void reset();
    void setMaxDimension(Size max_dim);
    void setTrackIntermediate(bool track);
    void setDeterministicOrdering(bool deterministic);
    void setMemoryLimit(size_t memory_limit_bytes);
    void setWitnessMode(bool witness_mode);
    void setSparseMode(bool sparse_mode);
    size_t getMemoryUsage() const;
    bool isMemoryLimitExceeded() const;
    void setDeterministicSeed(uint32_t seed);
    uint32_t getDeterministicSeed() const;

private:
    struct Checkpoint
    {
        Size step = 0;
        SimplicialComplex complex;
        std::vector<Pair> pairs;
    };

    Size max_dimension_;
    bool track_intermediate_;
    Size current_step_;
    bool deterministic_ordering_;
    uint32_t deterministic_seed_;
    size_t memory_limit_bytes_;
    bool witness_mode_;
    bool sparse_mode_;
    SimplicialComplex current_complex_;
    std::vector<std::vector<Pair>> zigzag_history_;
    std::vector<Pair> current_pairs_;
    std::vector<Simplex> addition_stack_;
    std::vector<Simplex> removal_stack_;
    std::deque<Checkpoint> checkpoints_;
    Size checkpoint_interval_;
    Size max_checkpoints_;
    persistence::IncrementalExactZ2 incremental_engine_;
    bool incremental_engine_ready_;
    mutable size_t current_memory_usage_;
    void processAddition(const Simplex &simplex);
    void processAdditionWithFiltration(const Simplex &simplex, double filtration);
    void processRemoval(const Simplex &simplex);
    void updateZigzagHistory();
    std::vector<Pair> computeZigzagPairs() const;
    void updateCurrentPairs(const std::vector<Pair> &pairs);
    void applyDeterministicOrdering(std::vector<Simplex> &simplices);
    uint32_t hashSimplex(const Simplex &simplex) const;
    void updateMemoryUsage();
    void saveCheckpointIfNeeded();
    void rebuildIncrementalEngine();
    void recomputeCurrentPairs();
    std::vector<Pair> computeWitnessPairs() const;
    std::vector<Pair> computeSparsePairs() const;
};

class WindowedPH
{
public:
    WindowedPH();
    explicit WindowedPH(Size window_size, Size max_dimension);
    void addDataPoint(const std::vector<double> &point);
    void addSimplexToWindow(const Simplex &simplex);
    void slideWindow();
    Diagram getWindowPersistence() const;
    std::vector<Pair> getWindowPairs() const;
    double getWindowStability() const;
    void setWindowSize(Size size);
    void setOverlapSize(Size overlap);
    void setUpdateThreshold(double threshold);

private:
    Size window_size_;
    Size overlap_size_;
    double update_threshold_;
    Size max_dimension_;
    std::queue<Simplex> simplex_window_;
    IncrementalPH incremental_ph_;
    Diagram current_diagram_;
    double last_stability_;
    void removeOldestSimplex();
    void updateWindowPersistence();
    double computeStability(const Diagram &diagram) const;
};

class DistributedPH
{
public:
    DistributedPH();
    explicit DistributedPH(Size num_workers, Size max_dimension);
    void distributeComplex(const SimplicialComplex &complex);
    void distributeFiltration(const std::vector<std::pair<Simplex, double>> &filtration);
    void collectResults();
    Diagram getGlobalPersistence() const;
    std::vector<Diagram> getLocalPersistence() const;
    void setNumWorkers(Size num_workers);
    void setCommunicationPattern(const std::string &pattern);
    void setMergeStrategy(const std::string &strategy);
    void distribute_complex(const SimplicialComplex &complex) { distributeComplex(complex); }
    void distribute_filtration(const std::vector<std::pair<Simplex, double>> &filtration)
    {
        distributeFiltration(filtration);
    }
    void collect_results() { collectResults(); }
    Diagram get_global_persistence() const { return getGlobalPersistence(); }
    void set_num_workers(Size num_workers) { setNumWorkers(num_workers); }

private:
    Size num_workers_;
    Size max_dimension_;
    std::string communication_pattern_;
    std::string merge_strategy_;
    std::vector<IncrementalPH> local_ph_;
    std::unordered_map<Simplex, double, Simplex::Hash> global_filtration_;
    Diagram global_diagram_;
    bool results_collected_;
    void initializeWorkers();
    void rebuildWorkersFromGlobalFiltration();
    void mergeLocalResults();
    std::vector<Pair> computeZigzagPairs() const;
    void updateCurrentPairs(const std::vector<Pair> &pairs);
};

class RealtimePH
{
public:
    RealtimePH();
    explicit RealtimePH(double update_frequency, Size max_dimension);
    void startMonitoring();
    void stopMonitoring();
    void addStreamData(const std::vector<Simplex> &simplices);
    void processStreamBatch();
    Diagram getLatestPersistence() const;
    std::vector<double> getPersistenceTimeline() const;
    std::vector<Size> getBettiTimeline() const;
    void start_monitoring() { startMonitoring(); }
    void stop_monitoring() { stopMonitoring(); }
    void add_stream_data(const std::vector<Simplex> &simplices) { addStreamData(simplices); }
    void process_stream_batch() { processStreamBatch(); }
    Diagram get_latest_persistence() const { return getLatestPersistence(); }
    std::vector<double> get_persistence_timeline() const { return getPersistenceTimeline(); }
    std::vector<Size> get_betti_timeline() const { return getBettiTimeline(); }
    void setUpdateFrequency(double frequency);
    void setChangeThreshold(double threshold);
    void setBufferSize(Size size);

private:
    double update_frequency_;
    Size max_dimension_;
    double change_threshold_;
    Size buffer_size_;
    bool monitoring_active_;
    IncrementalPH incremental_ph_;
    std::vector<Simplex> stream_buffer_;
    std::vector<Diagram> persistence_history_;
    std::vector<double> timestamp_history_;
    std::vector<Size> betti_timeline_;
    void processBuffer();
    bool detectSignificantChange(const Diagram &old_diagram, const Diagram &new_diagram) const;
    void updateHistory(const Diagram &diagram);
};

std::vector<Pair>
computeIncrementalPersistence(const std::vector<std::vector<Simplex>> &simplex_sequence);

std::vector<std::vector<Pair>> computeZigzagPersistence(
    const std::vector<std::pair<std::vector<Simplex>, std::vector<Simplex>>> &zigzag_sequence);

Diagram computeWindowedPersistence(const std::vector<std::vector<Simplex>> &data_windows,
                                   Size window_size);

std::vector<Diagram> computeDistributedPersistence(const std::vector<SimplicialComplex> &complexes,
                                                   Size num_workers);

} // namespace nerve::streaming
