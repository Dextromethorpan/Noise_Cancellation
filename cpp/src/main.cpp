#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <portaudio.h>
#include <zmq.hpp>

// -----------------------------------------------
// Configuration
// -----------------------------------------------
constexpr int SAMPLE_RATE  = 44100;
constexpr int FRAMES       = 1536;
constexpr int NUM_CHANNELS = 1;

// -----------------------------------------------
// Device Selection — search by name
// Change these strings to switch devices
// without worrying about index numbers
// -----------------------------------------------
const std::string INPUT_DEVICE_NAME  = "Microphone (Realtek HD Audio Mic input)";
const std::string OUTPUT_DEVICE_NAME = "Headphones";

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
// Find device index by partial name match
// needsInput = true  → look for input device
// needsInput = false → look for output device
// Returns -1 if not found
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

    // Play clean audio if ready
    // Fall back to passthrough until first clean chunk arrives
    if (state->cleanReady) {
        for (unsigned long i = 0; i < framesPerBuffer; ++i)
            out[i] = state->outputBuffer[i];
    } else {
        if (inputBuffer != nullptr) {
            for (unsigned long i = 0; i < framesPerBuffer; ++i)
                out[i] = in[i];
        } else {
            for (unsigned long i = 0; i < framesPerBuffer; ++i)
                out[i] = 0.0f;
        }
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
    std::cout << "=== Noise Cancellation - Phase 4: Async Pipeline ===\n\n";

    // 1. Initialize ZeroMQ PUSH/PULL sockets
    std::cout << "Connecting to Python server...\n";
    zmq::context_t zmqContext(1);

    // Push noisy audio TO Python
    zmq::socket_t pushSocket(zmqContext, zmq::socket_type::push);
    pushSocket.connect("tcp://127.0.0.1:5555");

    // Pull clean audio FROM Python
    zmq::socket_t pullSocket(zmqContext, zmq::socket_type::pull);
    pullSocket.connect("tcp://127.0.0.1:5556");

    std::cout << "Connected!\n\n";

    // 2. Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init failed: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    listAudioDevices();

    // 3. Find devices by name — immune to index reshuffling
    int inputDevice  = findDevice(INPUT_DEVICE_NAME,  true);
    int outputDevice = findDevice(OUTPUT_DEVICE_NAME, false);

    if (inputDevice < 0) {
        std::cerr << "Input device not found: " << INPUT_DEVICE_NAME << "\n";
        std::cerr << "Check the device list above and update INPUT_DEVICE_NAME\n";
        Pa_Terminate();
        return 1;
    }
    if (outputDevice < 0) {
        std::cerr << "Output device not found: " << OUTPUT_DEVICE_NAME << "\n";
        std::cerr << "Check the device list above and update OUTPUT_DEVICE_NAME\n";
        Pa_Terminate();
        return 1;
    }

    // 4. Configure input (Realtek mic)
    PaStreamParameters inputParams;
    inputParams.device                    = inputDevice;
    inputParams.channelCount              = NUM_CHANNELS;
    inputParams.sampleFormat              = paFloat32;
    inputParams.suggestedLatency          = Pa_GetDeviceInfo(inputDevice)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // 5. Configure output (headphones)
    PaStreamParameters outputParams;
    outputParams.device                    = outputDevice;
    outputParams.channelCount              = NUM_CHANNELS;
    outputParams.sampleFormat              = paFloat32;
    outputParams.suggestedLatency          = Pa_GetDeviceInfo(outputDevice)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    std::cout << "Using devices:\n";
    std::cout << "  Input:  [" << inputDevice  << "] "
              << Pa_GetDeviceInfo(inputDevice)->name  << "\n";
    std::cout << "  Output: [" << outputDevice << "] "
              << Pa_GetDeviceInfo(outputDevice)->name << "\n";
    std::cout << "  Sample Rate: " << SAMPLE_RATE << " Hz\n\n";

    // 6. Verify format is supported
    PaError supported = Pa_IsFormatSupported(
        &inputParams, &outputParams, SAMPLE_RATE);
    if (supported != paFormatIsSupported) {
        std::cerr << "Format not supported: "
                  << Pa_GetErrorText(supported) << "\n";
        std::cerr << "Try changing INPUT_DEVICE_NAME or OUTPUT_DEVICE_NAME\n";
        Pa_Terminate();
        return 1;
    }
    std::cout << "Format supported!\n\n";

    // 7. Open the stream
    AudioState state;
    PaStream*  stream;

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

    // 8. Start the stream
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
    std::cout << "Speak into your laptop mic - noise will be cancelled!\n";
    std::cout << "Press ENTER to stop...\n\n";

    // 9. Sender thread — pushes mic audio to Python continuously
    //    Does not wait for reply — fully async
    std::atomic<bool> running(true);
    int sentCount = 0;

    std::thread senderThread([&]() {
        while (running) {
            zmq::message_t msg(
                state.inputBuffer.data(),
                state.inputBuffer.size() * sizeof(float));
            pushSocket.send(msg, zmq::send_flags::none);
            sentCount++;

            // Sleep to match audio buffer period
            std::this_thread::sleep_for(
                std::chrono::milliseconds(
                    FRAMES * 1000 / SAMPLE_RATE));
        }
    });

    // 10. Receiver thread — pulls clean audio from Python
    //     Updates output buffer whenever clean audio is available
    //     Never blocks the audio callback
    int receivedCount = 0;

    std::thread receiverThread([&]() {
        while (running) {
            zmq::message_t reply;

            // Non-blocking receive
            auto result = pullSocket.recv(
                reply, zmq::recv_flags::dontwait);

            if (result) {
                memcpy(state.outputBuffer.data(),
                       reply.data(),
                       FRAMES * sizeof(float));
                state.cleanReady = true;
                receivedCount++;

                if (receivedCount % 50 == 0) {
                    std::cout << "Received " << receivedCount
                              << " clean chunks ("
                              << (receivedCount * FRAMES / (float)SAMPLE_RATE)
                              << " seconds)\n";
                }
            }

            // Small sleep to avoid hammering the socket
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // 11. Wait for ENTER
    std::cin.get();
    running = false;
    senderThread.join();
    receiverThread.join();

    // 12. Clean up
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "\nSent:     " << sentCount    << " chunks\n";
    std::cout << "Received: " << receivedCount << " chunks\n";
    std::cout << "Stream stopped. Phase 4 complete!\n";
    return 0;
}