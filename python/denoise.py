import torch
from df.enhance import enhance, init_df, load_audio, save_audio

def denoise_file(input_path: str, output_path: str):
    """
    Takes a noisy .wav file and produces a clean one.
    This is the core of what Phase 3 will do in real-time.
    """

    print(f"Loading model...")

    # 1. Initialize DeepFilterNet
    model, df_state, _ = init_df()

    print(f"Model loaded!")
    print(f"Processing: {input_path}")

    # 2. Load the noisy audio file
    # df_state.sr() gives us the correct sample rate (48kHz)
    audio, _ = load_audio(input_path, sr=df_state.sr())

    sample_rate = df_state.sr()  # use df_state directly — always correct

    print(f"  Sample rate:  {sample_rate} Hz")
    print(f"  Duration:     {audio.shape[-1] / sample_rate:.2f} seconds")
    print(f"  Shape:        {audio.shape}")

    # 3. Run the AI model — this is the noise cancellation
    enhanced_audio = enhance(model, df_state, audio)

    # 4. Save the cleaned audio
    save_audio(output_path, enhanced_audio, df_state.sr())

    print(f"Done! Clean audio saved to: {output_path}")


if __name__ == "__main__":
    denoise_file(
        input_path="noisy_input.wav",
        output_path="clean_output.wav"
    )