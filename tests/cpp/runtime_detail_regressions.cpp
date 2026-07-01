#include "nerve/core_types.hpp"
#include "nerve/runtime/calibration_model.hpp"
#include "nerve/runtime/hardware_probe.hpp"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace
{


constexpr double kTol = 1e-10;

bool check_hardware_snapshot_collection()
{
    auto snap = nerve::runtime::collectHardwareSnapshot();
    if (snap.collected_unix_ms == 0)
    {
        std::cerr << "hardware snapshot timestamp is 0\n";
        return false;
    }
    return true;
}

bool check_cpu_topology_detected()
{
    auto snap = nerve::runtime::collectHardwareSnapshot();
    if (!snap.cpu.ok())
    {
        std::cerr << "CPU probe not ok: " << snap.cpu.diagnostics << "\n";
        return false;
    }
    if (snap.cpu.value.logical_cores == 0)
    {
        std::cerr << "CPU logical cores is 0\n";
        return false;
    }
    return true;
}

bool check_memory_detected()
{
    auto snap = nerve::runtime::collectHardwareSnapshot();
    if (snap.total_memory_bytes.ok())
    {
        if (snap.total_memory_bytes.value == 0)
        {
            std::cerr << "total memory is 0\n";
            return false;
        }
    }
    return true;
}

bool check_hardware_fingerprint()
{
    auto snap = nerve::runtime::collectHardwareSnapshot();
    auto fp = nerve::runtime::getHardwareFingerprint(snap);
    if (fp.empty())
    {
        std::cerr << "hardware fingerprint empty\n";
        return false;
    }
    return true;
}

bool check_has_cuda_gpu()
{
    auto snap = nerve::runtime::collectHardwareSnapshot();
    bool has = nerve::runtime::has_cuda_gpu(snap);
    (void)has;
    return true;
}

bool check_probe_status_to_string()
{
    std::string ok_str = nerve::runtime::probe_status_to_string(nerve::runtime::ProbeStatus::kOk);
    if (ok_str.empty())
    {
        std::cerr << "probe status string empty\n";
        return false;
    }
    std::string missing_str =
        nerve::runtime::probe_status_to_string(nerve::runtime::ProbeStatus::kMissing);
    if (missing_str.empty())
    {
        std::cerr << "probe status missing string empty\n";
        return false;
    }
    std::string err_str =
        nerve::runtime::probe_status_to_string(nerve::runtime::ProbeStatus::kError);
    if (err_str.empty())
    {
        std::cerr << "probe status error string empty\n";
        return false;
    }
    return true;
}

bool check_calibration_default_path()
{
    std::string path = nerve::runtime::defaultCalibrationStorePath();
    if (path.empty())
    {
        std::cerr << "default calibration store path empty\n";
        return false;
    }
    return true;
}

bool check_calibration_sample_defaults()
{
    nerve::runtime::CalibrationSample sample;
    if (sample.schema_version != "v1")
    {
        std::cerr << "default schema version wrong\n";
        return false;
    }
    if (sample.confidence != 0.0)
    {
        std::cerr << "default confidence wrong\n";
        return false;
    }
    return true;
}

bool check_prediction_key_defaults()
{
    nerve::runtime::PredictionKey key;
    if (!key.hardware_fingerprint.empty())
    {
        std::cerr << "default prediction key hardware fingerprint should be empty\n";
        return false;
    }
    return true;
}

bool check_prediction_with_bounds_defaults()
{
    nerve::runtime::PredictionWithBounds pwb;
    if (pwb.available)
    {
        std::cerr << "default prediction should not be available\n";
        return false;
    }
    return true;
}

bool check_selection_gate_defaults()
{
    nerve::runtime::SelectionGateDiagnostics sg;
    if (sg.gate_passed)
    {
        std::cerr << "default gate should not be passed\n";
        return false;
    }
    return true;
}

bool check_set_calibration_path_for_testing()
{
    nerve::runtime::setCalibrationStorePathForTesting("/tmp/pynerve_cal_test");
    return true;
}

bool check_calibration_sample_fields()
{
    nerve::runtime::CalibrationSample sample;
    sample.hardware_fingerprint = "test_hw";
    sample.problem_bucket = "test_bucket";
    sample.algorithm = "test_alg";
    sample.predicted_time_ms = 100.0;
    sample.predicted_memory_mb = 256.0;
    sample.observed_time_ms = 95.0;
    sample.observed_memory_mb = 260.0;
    sample.confidence = 0.95;
    sample.relative_error = 0.05;
    if (sample.hardware_fingerprint != "test_hw")
    {
        std::cerr << "hardware fingerprint not stored\n";
        return false;
    }
    return true;
}

} // namespace

int main()
{
    if (!check_hardware_snapshot_collection())
    {
        std::cerr << "FAIL: hardware snapshot\n";
        return 1;
    }
    if (!check_cpu_topology_detected())
    {
        std::cerr << "FAIL: CPU topology\n";
        return 1;
    }
    if (!check_memory_detected())
    {
        std::cerr << "FAIL: memory detection\n";
        return 1;
    }
    if (!check_hardware_fingerprint())
    {
        std::cerr << "FAIL: hardware fingerprint\n";
        return 1;
    }
    if (!check_has_cuda_gpu())
    {
        std::cerr << "FAIL: has cuda gpu\n";
        return 1;
    }
    if (!check_probe_status_to_string())
    {
        std::cerr << "FAIL: probe status string\n";
        return 1;
    }
    if (!check_calibration_default_path())
    {
        std::cerr << "FAIL: calibration default path\n";
        return 1;
    }
    if (!check_calibration_sample_defaults())
    {
        std::cerr << "FAIL: calibration sample defaults\n";
        return 1;
    }
    if (!check_prediction_key_defaults())
    {
        std::cerr << "FAIL: prediction key defaults\n";
        return 1;
    }
    if (!check_prediction_with_bounds_defaults())
    {
        std::cerr << "FAIL: prediction bounds defaults\n";
        return 1;
    }
    if (!check_selection_gate_defaults())
    {
        std::cerr << "FAIL: selection gate defaults\n";
        return 1;
    }
    if (!check_set_calibration_path_for_testing())
    {
        std::cerr << "FAIL: set calibration path\n";
        return 1;
    }
    if (!check_calibration_sample_fields())
    {
        std::cerr << "FAIL: calibration sample fields\n";
        return 1;
    }
    return 0;
}
