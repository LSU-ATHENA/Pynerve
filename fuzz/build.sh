#!/bin/bash -eu
# OSS-Fuzz build script for pynerve.
# This script is invoked by OSS-Fuzz infrastructure to build fuzz targets.

cd "$SRC/pynerve"

# Build fuzz targets
for fuzzer in fuzz/*.cpp; do
    name=$(basename "$fuzzer" .cpp)
    $CXX $CXXFLAGS -std=c++20 \
        -I "$SRC/pynerve/src/include" \
        "$fuzzer" \
        -o "$OUT/$name" \
        $LIB_FUZZING_ENGINE
done
