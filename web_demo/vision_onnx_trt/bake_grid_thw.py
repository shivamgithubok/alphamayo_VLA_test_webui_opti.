# SPDX-License-Identifier: Apache-2.0
#
# Bake grid_thw as a constant into the ONNX graph.
#
# WHY: TRT 10 classifies int64 inputs that influence tensor shapes as
# "shape tensors" and enforces different validation rules (it expects
# element counts, not Nx3 dimensions). grid_thw is int64 and drives
# internal reshapes, so TRT rejects the dynamic profile.
#
# Since your demo always uses the same grid ([[1,20,36]] x 4 rows),
# baking it as a constant removes the input entirely and avoids the
# shape-tensor classification. The engine then has only ONE input:
# hidden_states (the patch embeddings), which is a clean data tensor.
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto

ONNX_IN  = "/home/acf-thor/SHIVAM/alpamayo/vit_encoder.onnx"
ONNX_OUT = "/home/acf-thor/SHIVAM/alpamayo/vit_encoder_fixed.onnx"

# Your runtime grid — 4 rows of [1, 20, 36] (4 frames at 448px→20×36 patches)
GRID_TWH = np.array([[1, 20, 36],
                     [1, 20, 36],
                     [1, 20, 36],
                     [1, 20, 36]], dtype=np.int64)

print(f"Loading {ONNX_IN}...")
model = onnx.load(ONNX_IN)
graph = model.graph

# Remove grid_thw from graph inputs
new_inputs = [i for i in graph.input if i.name != "grid_thw"]
del graph.input[:]
graph.input.extend(new_inputs)

# Add grid_thw as an initializer (constant)
grid_tensor = numpy_helper.from_array(GRID_TWH, name="grid_thw")
graph.initializer.append(grid_tensor)

print(f"Baked grid_thw = {GRID_TWH.tolist()} as constant initializer")
print(f"Remaining inputs: {[i.name for i in graph.input]}")

onnx.checker.check_model(model)
onnx.save(model, ONNX_OUT)
print(f"Saved fixed ONNX to {ONNX_OUT}")