#include <iostream>
#include <portaudio.h>

// -----------------------------------------------
// Configuration — matches Python test exactly
// device=(1, 4), samplerate=44100,
// blocksize=512, channels=1
// -----------------------------------------------
constexpr int SAMPLE_RATE = 44100;
constexpr int FRAMES      = 512;
constexpr int CHANNELS    = 1; // mono for BOTH input AND output

// -----------------------------------------------
// PortAudio Callback
// Pure passthrough — exactly like Python test
// -----------------------------------------------
static int audioCallback(
    const void* inputBuffer,
    void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData)
{
    const float* in  = static_cast<const float*>(inputBuffer);
    float*       out = static_cast<float*>(outputBuffer);

    if (inputBuffer != nullptr) {
        for (unsigned long i = 0; i < framesPerBuffer; ++i)
            out[i] = in[i];
    } else {
        for (unsigned long i = 0; i < framesPerBuffer; ++i)
            out[i] = 0.0f;
    }

    return paContinue;
}

// -----------------------------------------------
// Main
// -----------------------------------------------
int main() {
    std::cout << "=== C++ Passthrough Test ===\n";
    std::cout << "Mirroring Python test exactly:\n";
    std::cout << "  device=(1, 4)\n";
    std::cout << "  samplerate=44100\n";
    std::cout << "  blocksize=512\n";
    std::cout << "  channels=1\n\n";

    // 1. Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init failed: "
                  << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    // 2. Print default devices for verification
    PaDeviceIndex inputDevice  = 1; // Microfoon (Realtek Audio) MME
    PaDeviceIndex outputDevice = 4; // Headphones (WH-CH720N Stereo) MME

    std::cout << "Input  device: [" << inputDevice  << "] "
              << Pa_GetDeviceInfo(inputDevice)->name  << "\n";
    std::cout << "Output device: [" << outputDevice << "] "
              << Pa_GetDeviceInfo(outputDevice)->name << "\n\n";

    // 3. Configure input — mono, MME
    PaStreamParameters inputParams;
    inputParams.device                    = inputDevice;
    inputParams.channelCount              = CHANNELS;
    inputParams.sampleFormat              = paFloat32;
    inputParams.suggestedLatency          = Pa_GetDeviceInfo(inputDevice)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // 4. Configure output — mono, MME
    PaStreamParameters outputParams;
    outputParams.device                    = outputDevice;
    outputParams.channelCount              = CHANNELS;
    outputParams.sampleFormat              = paFloat32;
    outputParams.suggestedLatency          = Pa_GetDeviceInfo(outputDevice)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    // 5. Verify format
    PaError supported = Pa_IsFormatSupported(
        &inputParams, &outputParams, SAMPLE_RATE);
    if (supported != paFormatIsSupported) {
        std::cerr << "Format not supported: "
                  << Pa_GetErrorText(supported) << "\n";
        Pa_Terminate();
        return 1;
    }
    std::cout << "Format supported!\n\n";

    // 6. Open stream
    PaStream* stream;
    err = Pa_OpenStream(
        &stream,
        &inputParams,
        &outputParams,
        SAMPLE_RATE,
        FRAMES,
        paClipOff,
        audioCallback,
        nullptr
    );

    if (err != paNoError) {
        std::cerr << "Failed to open stream: "
                  << Pa_GetErrorText(err) << "\n";
        Pa_Terminate();
        return 1;
    }

    // 7. Start stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Failed to start stream: "
                  << Pa_GetErrorText(err) << "\n";
        Pa_Terminate();
        return 1;
    }

    std::cout << "Stream started!\n";
    std::cout << "  Sample Rate: " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Buffer Size: " << FRAMES << " frames\n\n";
    std::cout << "Speak into your mic.\n";
    std::cout << "You should hear yourself AND YouTube should keep playing!\n";
    std::cout << "Press ENTER to stop...\n\n";

    std::cin.get();

    // 8. Clean up
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "Stopped.\n";
    return 0;
}