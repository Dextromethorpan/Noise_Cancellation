#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstring>
#include <numeric>
#include <portaudio.h>
#include <zmq.hpp>

// -----------------------------------------------
// Configuration
// Fix: use microseconds for precise sleep
// avoids integer truncation drift
// -----------------------------------------------
constexpr int SAMPLE_RATE   = 44100;
constexpr int FRAMES        = 1536;
constexpr int CHANNELS      = 1;
constexpr int INPUT_DEVICE  = 1;
constexpr int OUTPUT_DEVICE = 4;

// Precise sleep duration in microseconds
// 1536 * 1000000 / 44100 = 34829 microseconds
constexpr int SLEEP_MICROS  = FRAMES * 1000000 / SAMPLE_RATE;

// -----------------------------------------------
// Shared state
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

    if (inputBuffer != nullptr) {
        for (unsigned long i = 0; i < framesPerBuffer; ++i)
            state->inputBuffer[i] = in[i];
    }

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
// List devices
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
    std::cout << "=== Experiment: Sleep Fix (microsecond precision) ===\n";
    std::cout << "Sleep duration: " << SLEEP_MICROS << " microseconds\n";
    std::cout << "              = " << SLEEP_MICROS / 1000.0 << " ms\n\n";

    // 1. ZeroMQ
    std::cout << "Connecting to Python server...\n";
    zmq::context_t zmqContext(1);
    zmq::socket_t pushSocket(zmqContext, zmq::socket_type::push);
    pushSocket.connect("tcp://127.0.0.1:5555");
    zmq::socket_t pullSocket(zmqContext, zmq::socket_type::pull);
    pullSocket.connect("tcp://127.0.0.1:5556");
    std::cout << "Connected!\n\n";

    // 2. PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init failed: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    listAudioDevices();

    // 3. Stream parameters
    PaStreamParameters inputParams;
    inputParams.device                    = INPUT_DEVICE;
    inputParams.channelCount              = CHANNELS;
    inputParams.sampleFormat              = paFloat32;
    inputParams.suggestedLatency          = Pa_GetDeviceInfo(INPUT_DEVICE)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    PaStreamParameters outputParams;
    outputParams.device                    = OUTPUT_DEVICE;
    outputParams.channelCount              = CHANNELS;
    outputParams.sampleFormat              = paFloat32;
    outputParams.suggestedLatency          = Pa_GetDeviceInfo(OUTPUT_DEVICE)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    std::cout << "Using devices:\n";
    std::cout << "  Input:  [" << INPUT_DEVICE  << "] "
              << Pa_GetDeviceInfo(INPUT_DEVICE)->name  << "\n";
    std::cout << "  Output: [" << OUTPUT_DEVICE << "] "
              << Pa_GetDeviceInfo(OUTPUT_DEVICE)->name << "\n\n";

    PaError supported = Pa_IsFormatSupported(
        &inputParams, &outputParams, SAMPLE_RATE);
    if (supported != paFormatIsSupported) {
        std::cerr << "Format not supported: "
                  << Pa_GetErrorText(supported) << "\n";
        Pa_Terminate();
        return 1;
    }
    std::cout << "Format supported!\n\n";

    // 4. Open stream
    AudioState state;
    PaStream*  stream;

    err = Pa_OpenStream(&stream, &inputParams, &outputParams,
                        SAMPLE_RATE, FRAMES, paClipOff,
                        audioCallback, &state);
    if (err != paNoError) {
        std::cerr << "Failed to open stream: " << Pa_GetErrorText(err) << "\n";
        Pa_Terminate();
        return 1;
    }

    Pa_StartStream(stream);

    std::cout << "Audio stream started!\n";
    std::cout << "  Sample Rate:    " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Buffer Size:    " << FRAMES << " frames\n";
    std::cout << "  Chunk time:     " << SLEEP_MICROS / 1000.0 << " ms\n\n";
    std::cout << "Speak into your mic — noise will be cancelled!\n";
    std::cout << "Press ENTER to stop...\n\n";

    // 5. Metrics tracking
    std::atomic<bool> running(true);
    int sentCount     = 0;
    int receivedCount = 0;
    std::vector<double> receiveTimes;

    // 6. Sender thread — microsecond precision sleep
    std::thread senderThread([&]() {
        while (running) {
            zmq::message_t msg(
                state.inputBuffer.data(),
                state.inputBuffer.size() * sizeof(float));
            pushSocket.send(msg, zmq::send_flags::none);
            sentCount++;

            // KEY FIX: microseconds avoids integer truncation
            std::this_thread::sleep_for(
                std::chrono::microseconds(SLEEP_MICROS));
        }
    });

    // 7. Receiver thread
    std::thread receiverThread([&]() {
        while (running) {
            zmq::message_t reply;
            auto result = pullSocket.recv(
                reply, zmq::recv_flags::dontwait);

            if (result) {
                auto now = std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count();
                receiveTimes.push_back(now / 1e6);

                memcpy(state.outputBuffer.data(),
                       reply.data(),
                       FRAMES * sizeof(float));
                state.cleanReady = true;
                receivedCount++;

                if (receivedCount % 50 == 0) {
                    int dropped   = sentCount - receivedCount;
                    double dropPct = 100.0 * dropped / sentCount;
                    std::cout << "Chunks: sent=" << sentCount
                              << " received=" << receivedCount
                              << " dropped=" << dropped
                              << " (" << dropPct << "%)\n";
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    std::cin.get();
    running = false;
    senderThread.join();
    receiverThread.join();

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    // 8. Final metrics
    int dropped    = sentCount - receivedCount;
    double dropPct = sentCount > 0 ?
        100.0 * dropped / sentCount : 0.0;

    // Calculate jitter (variance between receive times)
    double jitter = 0.0;
    if (receiveTimes.size() > 1) {
        std::vector<double> intervals;
        for (size_t i = 1; i < receiveTimes.size(); ++i)
            intervals.push_back(receiveTimes[i] - receiveTimes[i-1]);
        double mean = std::accumulate(
            intervals.begin(), intervals.end(), 0.0) / intervals.size();
        double sq_sum = std::inner_product(
            intervals.begin(), intervals.end(),
            intervals.begin(), 0.0);
        jitter = std::sqrt(sq_sum / intervals.size() - mean * mean);
    }

    std::cout << "\n=== Experiment Results ===\n";
    std::cout << "Experiment:    Sleep Fix (microsecond precision)\n";
    std::cout << "Sleep micros:  " << SLEEP_MICROS << " us\n";
    std::cout << "Sent:          " << sentCount     << " chunks\n";
    std::cout << "Received:      " << receivedCount << " chunks\n";
    std::cout << "Dropped:       " << dropped       << " chunks\n";
    std::cout << "Drop rate:     " << dropPct       << "%\n";
    std::cout << "Jitter:        " << jitter        << " ms\n";
    std::cout << "==========================\n";

    return 0;
}