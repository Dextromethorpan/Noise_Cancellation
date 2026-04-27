cmake_minimum_required(VERSION 3.20)
project(NoiseCancellation)

set(CMAKE_CXX_STANDARD 17)

# PortAudio paths
set(PORTAUDIO_ROOT "C:/dev/portaudio")

# ZeroMQ paths
set(ZEROMQ_ROOT "C:/dev/zeromq")

# -----------------------------------------------
# Main executable — full async pipeline
# -----------------------------------------------
add_executable(NoiseCancellation src/main.cpp)

target_include_directories(NoiseCancellation PRIVATE
    ${PORTAUDIO_ROOT}/include
    ${ZEROMQ_ROOT}/include
)

target_link_libraries(NoiseCancellation PRIVATE
    ${PORTAUDIO_ROOT}/lib/portaudio.lib
    ${ZEROMQ_ROOT}/lib/libzmq-v143-mt-4_3_5.lib
)

add_custom_command(TARGET NoiseCancellation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PORTAUDIO_ROOT}/portaudio.dll"
        $<TARGET_FILE_DIR:NoiseCancellation>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ZEROMQ_ROOT}/bin/libzmq-v143-mt-4_3_5.dll"
        $<TARGET_FILE_DIR:NoiseCancellation>
)

# -----------------------------------------------
# Passthrough test — no ZeroMQ
# Verifies audio devices work correctly
# -----------------------------------------------
add_executable(PassthroughTest src/passthrough_test.cpp)

target_include_directories(PassthroughTest PRIVATE
    ${PORTAUDIO_ROOT}/include
)

target_link_libraries(PassthroughTest PRIVATE
    ${PORTAUDIO_ROOT}/lib/portaudio.lib
)

add_custom_command(TARGET PassthroughTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PORTAUDIO_ROOT}/portaudio.dll"
        $<TARGET_FILE_DIR:PassthroughTest>
)

# -----------------------------------------------
# Passthrough 1536 test — with ZeroMQ
# Tests correct buffer size for Python AI
# -----------------------------------------------
add_executable(Passthrough1536 src/passthrough_1536.cpp)

target_include_directories(Passthrough1536 PRIVATE
    ${PORTAUDIO_ROOT}/include
    ${ZEROMQ_ROOT}/include
)

target_link_libraries(Passthrough1536 PRIVATE
    ${PORTAUDIO_ROOT}/lib/portaudio.lib
    ${ZEROMQ_ROOT}/lib/libzmq-v143-mt-4_3_5.lib
)

add_custom_command(TARGET Passthrough1536 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PORTAUDIO_ROOT}/portaudio.dll"
        $<TARGET_FILE_DIR:Passthrough1536>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ZEROMQ_ROOT}/bin/libzmq-v143-mt-4_3_5.dll"
        $<TARGET_FILE_DIR:Passthrough1536>
)