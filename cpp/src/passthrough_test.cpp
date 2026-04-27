#include <iostream>
#include <vector>
#include <string>
#include <portaudio.h>

// -----------------------------------------------
// Configuration
// -----------------------------------------------
constexpr int SAMPLE_RATE     = 44100;
constexpr int FRAMES          = 1536;
constexpr int INPUT_CHANNELS  = 1; // mic is mono
constexpr int OUTPUT_CHANNELS = 2; // output is stereo

// -----------------------------------------------
// Device Selection
// Using Microsoft Sound Mapper which always
// routes to Windows default devices
// and is guaranteed to be MME compatible
// -----------------------------------------------
const std::string INPUT_DEVICE_NAME  = "Microsoft Sound Mapper - Input";
const std::string OUTPUT_DEVICE_NAME = "Microsoft Sound Mapper - Output";

// -----------------------------------------------
// Find device by partial name match
// -----------------------------------------------
int findDevice(const std::string& name, bool needsInput) {
    int deviceCount = Pa_GetDeviceCount();
    for (int i = 0; i < deviceCount; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        std::string deviceName(info->name);
        if (deviceName.find(name) != std::string::npos) {
            if (needsInput  && info->maxInputChannels  > 0) return i;
            if (!needsInput && info->maxOutputChannels > 0) return i;
        }
    }
    return -1;
}

// -----------------------------------------------
// PortAudio Callback
// Mono input → Stereo output
// Duplicates mono mic signal to both ears
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
        for (unsigned long i = 0; i < framesPerBuffer; ++i) {
            out[i * 2]     = in[i]; // left channel
            out[i * 2 + 1] = in[i]; // right channel
        }
    } else {
        for (unsigned long i = 0; i < framesPerBuffer * 2; ++i)
            out[i] = 0.0f;
    }

    return paContinue;
}

// -----------------------------------------------
// List all available audio devices
// -----------------------------------------------
void listAudioDevices() {
    int deviceCount = Pa_GetDeviceCount();
    std::cout << "\n=== Available Audio Devices ===\n";
    for (int i = 0; i < deviceCount; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        const PaHostApiInfo* api = Pa_GetHostApiInfo(info->hostApi);
        std::cout << "[" << i << "] " << info->name
                  << " | IN: "  << info->maxInputChannels
                  << " | OUT: " << info->maxOutputChannels
                  << " | SR: "  << info->defaultSampleRate
                  << " | API: " << api->name
                  << "\n";
    }
    std::cout << "================================\n\n";
}

// -----------------------------------------------
// Main
// -----------------------------------------------
int main() {
    std::cout << "=== Passthrough Test — Mono to Stereo ===\n";
    std::cout << "Purpose: verify audio output works\n";
    std::cout << "         mono mic input to stereo output\n\n";

    // 1. Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init failed: "
                  << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    listAudioDevices();

    // 2. Find devices by name
    int inputDevice  = findDevice(INPUT_DEVICE_NAME,  true);
    int outputDevice = findDevice(OUTPUT_DEVICE_NAME, false);

    if (inputDevice < 0) {
        std::cerr << "Input device not found: "
                  << INPUT_DEVICE_NAME << "\n";
        Pa_Terminate();
        return 1;
    }
    if (outputDevice < 0) {
        std::cerr << "Output device not found: "
                  << OUTPUT_DEVICE_NAME << "\n";
        Pa_Terminate();
        return 1;
    }

    const PaHostApiInfo* inApi  = Pa_GetHostApiInfo(
        Pa_GetDeviceInfo(inputDevice)->hostApi);
    const PaHostApiInfo* outApi = Pa_GetHostApiInfo(
        Pa_GetDeviceInfo(outputDevice)->hostApi);

    std::cout << "Using devices:\n";
    std::cout << "  Input:  [" << inputDevice  << "] "
              << Pa_GetDeviceInfo(inputDevice)->name
              << " | API: " << inApi->name  << "\n";
    std::cout << "  Output: [" << outputDevice << "] "
              << Pa_GetDeviceInfo(outputDevice)->name
              << " | API: " << outApi->name << "\n";
    std::cout << "  Input channels:  " << INPUT_CHANNELS  << " (mono)\n";
    std::cout << "  Output channels: " << OUTPUT_CHANNELS << " (stereo)\n\n";

    // 3. Configure input (mono mic)
    PaStreamParameters inputParams;
    inputParams.device                    = inputDevice;
    inputParams.channelCount              = INPUT_CHANNELS;
    inputParams.sampleFormat              = paFloat32;
    inputParams.suggestedLatency          = Pa_GetDeviceInfo(inputDevice)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // 4. Configure output (stereo)
    PaStreamParameters outputParams;
    outputParams.device                    = outputDevice;
    outputParams.channelCount              = OUTPUT_CHANNELS;
    outputParams.sampleFormat              = paFloat32;
    outputParams.suggestedLatency          = Pa_GetDeviceInfo(outputDevice)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    // 5. Verify format is supported
    PaError supported = Pa_IsFormatSupported(
        &inputParams, &outputParams, SAMPLE_RATE);
    if (supported != paFormatIsSupported) {
        std::cerr << "Format not supported: "
                  << Pa_GetErrorText(supported) << "\n";
        Pa_Terminate();
        return 1;
    }
    std::cout << "Format supported at " << SAMPLE_RATE << " Hz!\n\n";

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
    std::cout << "  Sample Rate:    " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Buffer Size:    " << FRAMES << " frames\n";
    std::cout << "  Latency:        "
              << (FRAMES * 1000.0 / SAMPLE_RATE) << " ms\n\n";
    std::cout << "Speak into your mic.\n";
    std::cout << "You should hear yourself in BOTH ears.\n";
    std::cout << "Press ENTER to stop...\n\n";

    std::cin.get();

    // 8. Clean up
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "Passthrough test stopped.\n";
    return 0;
}