#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build TensorRT engine from vit_encoder_fixed.onnx.
#
# After baking grid_thw as a constant, TRT constant-folded the entire shape
# chain, making the model fully static (all shapes fixed at build time).
# A static model must NOT use --minShapes/optShapes/maxShapes — TRT rejects
# them with "Static model does not take explicit shapes".
# We just pass --fp16 and let TRT determine everything from the ONNX.
# ─────────────────────────────────────────────────────────────────────────────
set -eo pipefail

ONNX=/home/acf-thor/SHIVAM/alpamayo/vit_encoder_fixed.onnx
ENGINE=/home/acf-thor/SHIVAM/alpamayo/vit_encoder.engine
LOG=/home/acf-thor/SHIVAM/alpamayo/vit_build.log

/usr/src/tensorrt/bin/trtexec \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --builderOptimizationLevel=5 \
  2>&1 | tee "${LOG}"

echo ""
echo "Engine: ${ENGINE}"
echo "Log:    ${LOG}"
echo ""
echo "Benchmark the built engine:"
echo "  /usr/src/tensorrt/bin/trtexec --loadEngine=${ENGINE}"