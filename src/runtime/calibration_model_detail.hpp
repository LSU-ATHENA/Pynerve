#pragma once

#include "nerve/runtime/calibration_model.hpp"

#include <string>

namespace nerve::runtime::detail
{

std::string trimCopy(const std::string &value);

std::string serializeSampleJson(const CalibrationSample &sample);

bool parseJsonSample(const std::string &line, CalibrationSample *sample);

bool isValidCalibrationSample(const CalibrationSample &sample);

} // namespace nerve::runtime::detail
