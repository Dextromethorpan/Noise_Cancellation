#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <portaudio.h>

// -----------------------------------------------
// Configuration
// -----------------------------------------------
constexpr int SAMPLE_RATE     = 44100;
constexpr int FRAMES          = 512;
constexpr int INPUT_CHANNELS  = 1; // mic is mono
constexpr int OUTPUT_CHANNELS = 2; // headphones are stereo

// -----------------------------------------------
// Device Selection
// Input:  Realtek mic — keeps Bluetooth in stereo mode
// Output: Bluetooth headphones in stereo mode
// Two separate streams to avoid HFP conflict
// -----------------------------------------------
const std::string INPUT_DEVICE_NAME  = "Microphone (Realtek HD Audio Mic input)";
const std::string OUTPUT_DEVICE_NAME = "Headphones (WH-CH720N Stereo)";

// -----------------------------------------------
// Shared audio buffer between streams
// -----------------------------------------------
struct SharedBuffer {
    std::vector<float> data;
    std::atomic<bool>  ready{false};

    SharedBuffer() {
        data.resize(FRAMES, 0.0f);
    }
};

// -----------------------------------------------
// Find device by partial name and channel type
// No host API restriction — separate streams
// don't need to match APIs
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
// Input Callback — captures mic audio
// Stores samples in shared buffer
// -----------------------------------------------
static int inputCallback(
    const void* inputBuffer,
    void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData)
{
    auto* shared     = static_cast<SharedBuffer*>(userData);
    const float* in  = static_cast<const float*>(inputBuffer);

    if (inputBuffer != nullptr) {
        for (unsigned long i = 0; i < framesPerBuffer; ++i)
            shared->data[i] = in[i];
        shared->ready = true;
    }

    return paContinue;
}

// -----------------------------------------------
// Output Callback — plays audio to headphones
// Reads from shared buffer
// Duplicates mono to both stereo channels
// -----------------------------------------------
static int outputCallback(
    const void* inputBuffer,
    void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData)
{
    auto*  shared = static_cast<SharedBuffer*>(userData);
    float* out    = static_cast<float*>(outputBuffer);

    if (shared->ready) {
        for (unsigned long i = 0; i < framesPerBuffer; ++i) {
            out[i * 2]     = shared->data[i]; // left
            out[i * 2 + 1] = shared->data[i]; // right
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
    std::cout << "=== Passthrough Test — Two Separate Streams ===\n";
    std::cout << "Input:  Realtek mic (keeps Bluetooth in stereo)\n";
    std::cout << "Output: Bluetooth WH-CH720N Stereo\n\n";

    // 1. Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init failed: "
                  << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    listAudioDevices();

    // 2. Find devices
    int inputDevice  = findDevice(INPUT_DEVICE_NAME,  true);
    int outputDevice = findDevice(OUTPUT_DEVICE_NAME, false);

    if (inputDevice < 0) {
        std::cerr << "Input device not found: "
                  << INPUT_DEVICE_NAME << "\n";
        std::cerr << "Check device list above\n";
        Pa_Terminate();
        return 1;
    }
    if (outputDevice < 0) {
        std::cerr << "Output device not found: "
                  << OUTPUT_DEVICE_NAME << "\n";
        std::cerr << "Check device list above\n";
        Pa_Terminate();
        return 1;
    }

    const PaHostApiInfo* inApi  = Pa_GetHostApiInfo(
        Pa_GetDeviceInfo(inputDevice)->hostApi);
    const PaHostApiInfo* outApi = Pa_GetHostApiInfo(
        Pa_GetDeviceInfo(outputDevice)->hostApi);

    std::cout << "Input  device: [" << inputDevice  << "] "
              << Pa_GetDeviceInfo(inputDevice)->name
              << " | API: " << inApi->name  << "\n";
    std::cout << "Output device: [" << outputDevice << "] "
              << Pa_GetDeviceInfo(outputDevice)->name
              << " | API: " << outApi->name << "\n\n";

    // 3. Shared buffer between input and output streams
    SharedBuffer shared;

    // 4. Configure input stream (Realtek mic — mono)
    PaStreamParameters inputParams;
    inputParams.device                    = inputDevice;
    inputParams.channelCount              = INPUT_CHANNELS;
    inputParams.sampleFormat              = paFloat32;
    inputParams.suggestedLatency          = Pa_GetDeviceInfo(inputDevice)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // 5. Configure output stream (Bluetooth — stereo)
    PaStreamParameters outputParams;
    outputParams.device                    = outputDevice;
    outputParams.channelCount              = OUTPUT_CHANNELS;
    outputParams.sampleFormat              = paFloat32;
    outputParams.suggestedLatency          = Pa_GetDeviceInfo(outputDevice)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    // 6. Open input stream
    PaStream* inputStream;
    err = Pa_OpenStream(
        &inputStream,
        &inputParams,
        nullptr,        // no output on this stream
        SAMPLE_RATE,
        FRAMES,
        paClipOff,
        inputCallback,
        &shared
    );

    if (err != paNoError) {
        std::cerr << "Failed to open input stream: "
                  << Pa_GetErrorText(err) << "\n";
        Pa_Terminate();
        return 1;
    }
    std::cout << "Input stream opened!\n";

    // 7. Open output stream
    PaStream* outputStream;
    err = Pa_OpenStream(
        &outputStream,
        nullptr,        // no input on this stream
        &outputParams,
        SAMPLE_RATE,
        FRAMES,
        paClipOff,
        outputCallback,
        &shared
    );

    if (err != paNoError) {
        std::cerr << "Failed to open output stream: "
                  << Pa_GetErrorText(err) << "\n";
        Pa_StopStream(inputStream);
        Pa_CloseStream(inputStream);
        Pa_Terminate();
        return 1;
    }
    std::cout << "Output stream opened!\n\n";

    // 8. Start both streams
    Pa_StartStream(inputStream);
    Pa_StartStream(outputStream);

    std::cout << "Both streams started!\n";
    std::cout << "  Sample Rate:     " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Buffer Size:     " << FRAMES << " frames\n";
    std::cout << "  Input channels:  " << INPUT_CHANNELS  << " (mono)\n";
    std::cout << "  Output channels: " << OUTPUT_CHANNELS << " (stereo)\n\n";
    std::cout << "Speak into your laptop mic.\n";
    std::cout << "You should hear yourself through the headphones.\n";
    std::cout << "Press ENTER to stop...\n\n";

    std::cin.get();

    // 9. Clean up both streams
    Pa_StopStream(inputStream);
    Pa_StopStream(outputStream);
    Pa_CloseStream(inputStream);
    Pa_CloseStream(outputStream);
    Pa_Terminate();

    std::cout << "Both streams stopped.\n";
    return 0;
}