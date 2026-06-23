# SPDX-License-Identifier: Apache-2.0
#
# vit_trt_runner.py — TensorRT 10.x vision encoder integration.
#
# Fixed for TRT 10 API:
#   - engine['name'] removed → use engine.get_tensor_name(i) to enumerate
#   - engine.get_tensor_shape(name) used directly for output shapes
#   - context.set_tensor_address(name, ptr) used for I/O binding
# ─────────────────────────────────────────────────────────────────────────────
import torch
import tensorrt as trt

ENGINE_PATH = "/home/acf-thor/SHIVAM/alpamayo/vit_encoder.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _load_engine(path: str) -> trt.ICudaEngine:
    with open(path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


class TRTViTRunner:
    """
    Wraps the TensorRT ViT engine (TRT 10.x API).
    Accepts hidden_states (N_patches, 1536) fp16/bf16 on CUDA.
    Returns (main_embeds, [deepstack_0, deepstack_1, deepstack_2]) on CUDA bf16.
    grid_thw is baked into the engine as a constant — ignored at runtime.
    """

    def __init__(self, engine_path: str = ENGINE_PATH):
        print(f"Loading TRT ViT engine from {engine_path}...")
        self.engine = _load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self._stream = torch.cuda.Stream()

        # Enumerate tensors using TRT 10 API
        n = self.engine.num_io_tensors
        self._inputs = []
        self._outputs = []
        self._out_shapes = {}

        for i in range(n):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                self._inputs.append(name)
            else:
                self._outputs.append(name)
                self._out_shapes[name] = shape

        print(f"  Inputs:  {self._inputs}")
        print(f"  Outputs: {list(self._out_shapes.keys())}")
        print(f"  Output shapes: {self._out_shapes}")
        print("TRT ViT engine ready.")

    def __call__(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor = None,
        **kwargs,
    ):
        # Input must be fp16, contiguous, on CUDA
        hs = hidden_states.to(dtype=torch.float16, device="cuda").contiguous()

        with torch.cuda.stream(self._stream):
            # Bind input
            self.context.set_tensor_address("hidden_states", hs.data_ptr())

            # Allocate and bind outputs
            out_tensors = {}
            for name, shape in self._out_shapes.items():
                t = torch.empty(shape, dtype=torch.float16, device="cuda")
                out_tensors[name] = t
                self.context.set_tensor_address(name, t.data_ptr())

            # Execute
            ok = self.context.execute_async_v3(
                stream_handle=self._stream.cuda_stream
            )
            if not ok:
                raise RuntimeError("TRT ViT engine execution failed")

            self._stream.synchronize()

        # Convert to bf16 to match what the LLM expects
        main = out_tensors["main_embeds"].to(torch.bfloat16)
        deepstack = [
            out_tensors["deepstack_0"].to(torch.bfloat16),
            out_tensors["deepstack_1"].to(torch.bfloat16),
            out_tensors["deepstack_2"].to(torch.bfloat16),
        ]
        return main, deepstack


def patch_vit_with_trt(model, engine_path: str = ENGINE_PATH):
    """
    Replaces model.vlm.model.visual.forward with the TRT engine.
    Call once after model loads, before first inference.
    """
    runner = TRTViTRunner(engine_path)
    vis = model.vlm.model.visual

    def trt_forward(hidden_states, grid_thw=None, **kwargs):
        return runner(hidden_states, grid_thw, **kwargs)

    vis.forward = trt_forward
    print("ViT forward patched → TRT engine active.")
    return runner