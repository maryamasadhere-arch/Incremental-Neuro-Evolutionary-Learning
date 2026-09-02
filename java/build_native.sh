#!/usr/bin/env bash
# Compiles the JNI native fitness kernel (src/main/c/fitness_native.c) into
# target/native/libfitness.<ext>. Optional: NativeFitness.java falls back to
# a pure-Java implementation if this hasn't been built, or fails to load.
set -euo pipefail
cd "$(dirname "$0")"

: "${JAVA_HOME:?JAVA_HOME must be set (e.g. export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))) }"

mkdir -p target/native

UNAME="$(uname -s)"
case "$UNAME" in
    Linux)
        CC=${CC:-gcc}
        "$CC" -shared -fPIC -O2 \
            -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux" \
            -o target/native/libfitness.so \
            src/main/c/fitness_native.c -lm
        echo "Built target/native/libfitness.so"
        ;;
    Darwin)
        CC=${CC:-clang}
        "$CC" -shared -fPIC -O2 \
            -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/darwin" \
            -o target/native/libfitness.dylib \
            src/main/c/fitness_native.c
        echo "Built target/native/libfitness.dylib"
        ;;
    *)
        echo "Unsupported platform '$UNAME' for the native build; the pure-Java fallback will be used." >&2
        ;;
esac
