#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <portaudio.h>
#include <zmq.hpp>

// -----------------------------------------------
// Configuration
// -----------------------------------------------
constexpr int SAMPLE_RATE  = 44100; // MME default sample rate
constexpr int FRAMES       = 1536;  // Buffer size per chunk
constexpr int NUM_CHANNELS = 1;     // Mono

// -----------------------------------------------
// Shared state between callback and main thread
// -----------------------------------------------
struct AudioState {
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
    bool cleanReady = false;

    AudioState() {
        inputBuffer.resize(FRAMES, 0.0f);
        outputBuffer.resize(FRAMES, 0.0f);
    }
};

// -----------------------------------------------
// PortAudio Callback
// Runs on high-priority audio thread
// IMPORTANT: No memory allocation or I/O here!
// -----------------------------------------------
static int audioCallback(
    const void* inputBuffer,
    void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData)
{
    auto* state = static_cast<AudioState*>(userData);
    const float* in  = static_cast<const float*>(inputBuffer);
    float*       out = static_cast<float*>(outputBuffer);

    // Copy mic input into shared state
    if (inputBuffer != nullptr) {
        for (unsigned long i = 0; i < framesPerBuffer; ++i)
            state->inputBuffer[i] = in[i];
    }

    // Play clean audio if ready, otherwise silence
    if (state->cleanReady) {
        for (unsigned long i = 0; i < framesPerBuffer; ++i)
            out[i] = state->outputBuffer[i];
    } else {
        for (unsigned long i = 0; i < framesPerBuffer; ++i)
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
        std::cout << "[" << i << "] " << info->name
                  << " | IN: "  << info->maxInputChannels
                  << " | OUT: " << info->maxOutputChannels
                  << " | SR: "  << info->defaultSampleRate
                  << "\n";
    }
    std::cout << "Default Input:  " << Pa_GetDefaultInputDevice() << "\n";
    std::cout << "Default Output: " << Pa_GetDefaultOutputDevice() << "\n";
    std::cout << "================================\n\n";
}

// -----------------------------------------------
// Main
// -----------------------------------------------
int main() {
    std::cout << "=== Noise Cancellation - Passthrough Test ===\n\n";
    std::cout << "NOTE: Python server NOT needed for this test\n\n";

    // 1. Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init failed: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    listAudioDevices();

    // 2. Use MME default devices
    PaDeviceIndex inputDevice  = Pa_GetDefaultInputDevice();
    PaDeviceIndex outputDevice = Pa_GetDefaultOutputDevice();

    // 3. Configure input (mic)
    PaStreamParameters inputParams;
    inputParams.device                    = inputDevice;
    inputParams.channelCount              = NUM_CHANNELS;
    inputParams.sampleFormat              = paFloat32;
    inputParams.suggestedLatency          = Pa_GetDeviceInfo(inputDevice)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // 4. Configure output (speakers)
    PaStreamParameters outputParams;
    outputParams.device                    = outputDevice;
    outputParams.channelCount              = NUM_CHANNELS;
    outputParams.sampleFormat              = paFloat32;
    outputParams.suggestedLatency          = Pa_GetDeviceInfo(outputDevice)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    std::cout << "Using MME default devices:\n";
    std::cout << "  Input:  [" << inputDevice  << "] "
              << Pa_GetDeviceInfo(inputDevice)->name  << "\n";
    std::cout << "  Output: [" << outputDevice << "] "
              << Pa_GetDeviceInfo(outputDevice)->name << "\n";
    std::cout << "  Sample Rate: " << SAMPLE_RATE << " Hz\n\n";

    // 5. Verify format is supported
    PaError supported = Pa_IsFormatSupported(
        &inputParams, &outputParams, SAMPLE_RATE);
    if (supported != paFormatIsSupported) {
        std::cerr << "Format not supported: "
                  << Pa_GetErrorText(supported) << "\n";
        Pa_Terminate();
        return 1;
    }
    std::cout << "Format supported!\n\n";

    // 6. Open the stream
    AudioState state;
    PaStream* stream;

    err = Pa_OpenStream(
        &stream,
        &inputParams,
        &outputParams,
        SAMPLE_RATE,
        FRAMES,
        paClipOff,
        audioCallback,
        &state
    );

    if (err != paNoError) {
        std::cerr << "Failed to open stream: " << Pa_GetErrorText(err) << "\n";
        Pa_Terminate();
        return 1;
    }

    // 7. Start the stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Failed to start stream: " << Pa_GetErrorText(err) << "\n";
        Pa_Terminate();
        return 1;
    }

    std::cout << "Audio stream started!\n";
    std::cout << "  Sample Rate:  " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Buffer Size:  " << FRAMES << " frames\n";
    std::cout << "  Latency:      "
              << (FRAMES * 1000.0 / SAMPLE_RATE) << " ms\n\n";
    std::cout << "PASSTHROUGH TEST — speak into your mic\n";
    std::cout << "You should hear yourself in the output.\n";
    std::cout << "Press ENTER to stop...\n\n";

    // 8. Passthrough test — bypass Python completely
    //    Copies mic input directly to output
    //    If you hear yourself = audio pipeline works
    //    If you hear nothing  = output device problem
    std::atomic<bool> running(true);

    std::thread processingThread([&]() {
        while (running) {
            // Direct passthrough — no AI processing
            state.outputBuffer = state.inputBuffer;
            state.cleanReady   = true;

            // Small sleep to avoid hammering the CPU
            std::this_thread::sleep_for(
                std::chrono::milliseconds(10));
        }
    });

    // 9. Wait for ENTER
    std::cin.get();
    running = false;
    processingThread.join();

    // 10. Clean up
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "\nPassthrough test stopped.\n";
    return 0;
}