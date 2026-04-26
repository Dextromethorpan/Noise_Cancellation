cmake_minimum_required(VERSION 3.20)
project(NoiseCancellation)

set(CMAKE_CXX_STANDARD 17)

# PortAudio paths
set(PORTAUDIO_ROOT "C:/dev/portaudio")

# ZeroMQ paths
set(ZEROMQ_ROOT "C:/dev/zeromq")

add_executable(NoiseCancellation src/main.cpp)

target_include_directories(NoiseCancellation PRIVATE
    ${PORTAUDIO_ROOT}/include
    ${ZEROMQ_ROOT}/include
)

target_link_libraries(NoiseCancellation PRIVATE
    ${PORTAUDIO_ROOT}/lib/portaudio.lib
    ${ZEROMQ_ROOT}/lib/libzmq-v143-mt-4_3_5.lib
)

# Copy DLLs next to the executable after build
add_custom_command(TARGET NoiseCancellation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PORTAUDIO_ROOT}/portaudio.dll"
        $<TARGET_FILE_DIR:NoiseCancellation>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ZEROMQ_ROOT}/bin/libzmq-v143-mt-4_3_5.dll"
        $<TARGET_FILE_DIR:NoiseCancellation>
)