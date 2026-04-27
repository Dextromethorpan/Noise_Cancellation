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