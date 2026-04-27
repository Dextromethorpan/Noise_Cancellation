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
// Mirrors the working Python sounddevice test:
// device=(1, 4), samplerate=44100,
// blocksize=512, channels=1, MME shared mode
// -----------------------------------------------
constexpr int SAMPLE_RATE    = 44100;
constexpr int FRAMES         = 512;
constexpr int CHANNELS       = 1;    // mono for BOTH input AND output
constexpr int INPUT_DEVICE   = 1;    // Microfoon (Realtek Audio) MME
constexpr int OUTPUT_DEVICE  = 4;    // Headphones (WH-CH720N Stereo) MME

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
        const PaHostApiInfo* api = Pa_GetHostApiInfo(info->hostApi);
        std::cout << "[" << i << "] " << info->name
                  << " | IN: "  << info->maxInputChannels
                  << " | OUT: " << info->maxOutputChannels
                  << " | SR: "  << info->defaultSampleRate
                  << " | API: " << api->name
                  << "\n";
    }
    std::cout << "Default Input:  " << Pa_GetDefaultInputDevice()  << "\n";
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
        std::cerr << "PortAudio init failed: "
                  << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    listAudioDevices();

    // 3. Verify selected devices exist
    int deviceCount = Pa_GetDeviceCount();
    if (INPUT_DEVICE >= deviceCount || OUTPUT_DEVICE >= deviceCount) {
        std::cerr << "Invalid device index!\n";
        std::cerr << "Check device list above\n";
        Pa_Terminate();
        return 1;
    }

    std::cout << "Using devices:\n";
    std::cout << "  Input:  [" << INPUT_DEVICE  << "] "
              << Pa_GetDeviceInfo(INPUT_DEVICE)->name  << "\n";
    std::cout << "  Output: [" << OUTPUT_DEVICE << "] "
              << Pa_GetDeviceInfo(OUTPUT_DEVICE)->name << "\n";
    std::cout << "  Channels:    " << CHANNELS    << " (mono)\n";
    std::cout << "  Sample Rate: " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Buffer Size: " << FRAMES      << " frames\n\n";

    // 4. Configure input (Realtek mic — MME)
    PaStreamParameters inputParams;
    inputParams.device                    = INPUT_DEVICE;
    inputParams.channelCount              = CHANNELS;
    inputParams.sampleFormat              = paFloat32;
    inputParams.suggestedLatency          = Pa_GetDeviceInfo(INPUT_DEVICE)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // 5. Configure output (WH-CH720N Stereo — MME)
    PaStreamParameters outputParams;
    outputParams.device                    = OUTPUT_DEVICE;
    outputParams.channelCount              = CHANNELS;
    outputParams.sampleFormat              = paFloat32;
    outputParams.suggestedLatency          = Pa_GetDeviceInfo(OUTPUT_DEVICE)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    // 6. Verify format is supported
    PaError supported = Pa_IsFormatSupported(
        &inputParams, &outputParams, SAMPLE_RATE);
    if (supported != paFormatIsSupported) {
        std::cerr << "Format not supported: "
                  << Pa_GetErrorText(supported) << "\n";
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
        std::cerr << "Failed to open stream: "
                  << Pa_GetErrorText(err) << "\n";
        Pa_Terminate();
        return 1;
    }

    // 8. Start the stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Failed to start stream: "
                  << Pa_GetErrorText(err) << "\n";
        Pa_Terminate();
        return 1;
    }

    std::cout << "Audio stream started!\n";
    std::cout << "  Latency: "
              << (FRAMES * 1000.0 / SAMPLE_RATE) << " ms\n\n";
    std::cout << "Speak into your mic - noise will be cancelled!\n";
    std::cout << "Press ENTER to stop...\n\n";

    // 9. Sender thread — pushes mic audio to Python continuously
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
    //     Non-blocking — never stalls the audio callback
    int receivedCount = 0;

    std::thread receiverThread([&]() {
        while (running) {
            zmq::message_t reply;

            auto result = pullSocket.recv(
                reply, zmq::recv_flags::dontwait);

            if (result) {
                memcpy(state.outputBuffer.data(),
                       reply.data(),
                       FRAMES * sizeof(float));
                state.cleanReady = true;
                receivedCount++;

                if (receivedCount % 100 == 0) {
                    std::cout << "Processed " << receivedCount
                              << " chunks ("
                              << (receivedCount * FRAMES / (float)SAMPLE_RATE)
                              << " seconds)\n";
                }
            }

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