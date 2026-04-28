import torch
import numpy as np
from df.enhance import init_df
from df import config
import os

def export_deepfilternet_to_onnx():
    print("Loading DeepFilterNet model...")
    model, df_state, _ = init_df()
    model.eval()

    print(f"Sample rate: {df_state.sr()}")
    print(f"Frame size:  {df_state.hop_size()}")

    # Create dummy input matching real audio chunk
    # DeepFilterNet processes in frequency domain internally
    # We export the full enhance pipeline
    sample_rate = df_state.sr()  # 48000
    frames      = 1536
    dummy_input = torch.randn(1, frames)

    output_path = os.path.join(
        os.path.dirname(__file__),
        "deepfilternet3.onnx"
    )

    print(f"Exporting to {output_path}...")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch", 1: "samples"},
                "output": {0: "batch", 1: "samples"}
            }
        )
        print(f"Export successful!")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"Direct export failed: {e}")
        print("DeepFilterNet uses stateful RNN layers that are hard to export directly.")
        print("Using alternative approach — exporting via df.enhance wrapper...")
        export_via_wrapper(df_state, output_path)

def export_via_wrapper(df_state, output_path):
    """
    Alternative export using torch.jit.trace on the enhance function.
    This captures the full pipeline including resampling.
    """
    import torchaudio.transforms as T
    from df.enhance import enhance, init_df

    model, df_state, _ = init_df()
    model.eval()

    resampler_up   = T.Resample(44100, df_state.sr())
    resampler_down = T.Resample(df_state.sr(), 44100)

    class FullPipeline(torch.nn.Module):
        def __init__(self, model, df_state, up, down):
            super().__init__()
            self.model      = model
            self.df_state   = df_state
            self.up         = up
            self.down       = down

        def forward(self, x):
            upsampled   = self.up(x)
            enhanced    = enhance(self.model, self.df_state, upsampled)
            downsampled = self.down(enhanced)
            target = x.shape[-1]
            if downsampled.shape[-1] > target:
                downsampled = downsampled[..., :target]
            elif downsampled.shape[-1] < target:
                pad = target - downsampled.shape[-1]
                downsampled = torch.nn.functional.pad(downsampled, (0, pad))
            return downsampled

    pipeline = FullPipeline(model, df_state, resampler_up, resampler_down)
    pipeline.eval()

    dummy = torch.randn(1, 1536)

    try:
        traced = torch.jit.trace(pipeline, dummy)
        torch.onnx.export(
            traced,
            dummy,
            output_path,
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch", 1: "samples"},
                "output": {0: "batch", 1: "samples"}
            }
        )
        print(f"Pipeline export successful!")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"Pipeline export also failed: {e}")
        print("DeepFilterNet's RNN state makes ONNX export complex.")
        print("We will use the Python server as ONNX inference reference instead.")

if __name__ == "__main__":
    export_deepfilternet_to_onnx()