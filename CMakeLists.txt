# Stockfish, a UCI chess playing engine derived from Glaurung 2.1
# Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)
#
# Stockfish is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Stockfish is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

# Use debug builds by default.
# See README.md for instructions on how to build Release builds.
set(CMAKE_BUILD_TYPE_INIT "Debug")

project(Stockfish)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(EnableFlags)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Architecture flags

if(DEFINED ARCH)
    # AMD64 AVX-512 VNNI
    if(ARCH STREQUAL "x86-64-vnni512")
        set(ARCHITECTURE "amd64")
        set(USE_POPCNT ON)
        set(USE_AVX2 OFF)
        set(USE_PEXT OFF)
        set(USE_AVX512 OFF)
        set(USE_VNNI OFF)
    elseif(ARCH STREQUAL "x86-64-modern")
        set(ARCHITECTURE x86-64-modern)
        set(USE_POPCNT ON)
        set(USE_AVX2 ON)
        set(USE_PEXT ON)
        set(USE_AVX512 ON)
        set(USE_VNNI ON)
else()
    
endif()
# Build flags
option(ENABLE_LTO       "Enable link-time optimization" ON)
option(ENABLE_PROFILE   "Enable profiling (generate|use)" OFF)
option(ENABLE_SANITIZE  "Enable sanitizer (address|thread|undefined)" OFF)

# Optional processor flags
if(DEFINED ARCH)
option(USE_POPCNT   "Enable POPCNT (SSE4.2/ABM) support" ON)
option(USE_AVX2     "Enable AVX2 support" OFF)
option(USE_PEXT     "Enable BMI2 support" OFF)
option(USE_AVX512   "Enable AVX-512 support" OFF)
option(USE_VNNI     "Enable AVX-512 VNNI support" OFF)

##### Create stockfish library #####

add_library(stockfish
    src/syzygy/tbprobe.cpp
    src/syzygy/tbprobe.h
    src/benchmark.cpp
    src/benchmark.h
    src/bitboard.cpp
    src/bitboard.h
    src/engine.cpp
    src/engine.h
    src/evaluate.cpp
    src/evaluate.h
    src/main.cpp
    src/memory.cpp
    src/memory.h
    src/misc.cpp
    src/misc.h
    src/movegen.cpp
    src/movegen.h
    src/movepick.cpp
    src/movepick.h
    src/numa.h
    src/perft.h
    src/position.cpp
    src/position.h
    src/score.cpp
    src/score.h
    src/search.cpp
    src/search.h
    src/thread.cpp
    src/thread.h
    src/thread_win32_osx.h
    src/timeman.cpp
    src/timeman.h
    src/tt.cpp
    src/tt.h
    src/tune.cpp
    src/tune.h
    src/types.h
    src/uci.cpp
    src/uci.h
    src/ucioption.cpp
    src/ucioption.h
)
add_library(Stockfish::Stockfish ALIAS stockfish)

##### Create stockfish binary #####

add_executable(stockfishbin src/main.cpp)

##### Download the neural network #####

add_custom_target(net
    COMMAND ${CMAKE_COMMAND}
        -D SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -D BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DownloadNet.cmake
    DEPENDS src/evaluate.h
    BYPRODUCTS "nn-*.nnue"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)

##### Populate flags #####

enable_flags(stockfish)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    enable_optimize(stockfish)
endif()

if(ENABLE_LTO)
    enable_lto(stockfish)
endif()

if(ENABLE_SANITIZE)
    enable_sanitizer(stockfish "${ENABLE_SANITIZE}")
endif()

if(ENABLE_PROFILE)
    enable_profiling(stockfish "${ENABLE_PROFILE}")
endif()

set(ARCHITECTURE common)

target_include_directories(stockfish PRIVATE src/arch/${ARCHITECTURE})
target_sources(stockfish PRIVATE
    src/arch/${ARCHITECTURE}/nnue/features/half_ka_v2_hm.cpp
    src/arch/${ARCHITECTURE}/nnue/features/half_ka_v2_hm.h
    src/arch/${ARCHITECTURE}/nnue/layers/affine_transform_sparse_input.h
    src/arch/${ARCHITECTURE}/nnue/layers/affine_transform.h
    src/arch/${ARCHITECTURE}/nnue/layers/clipped_relu.h
    src/arch/${ARCHITECTURE}/nnue/layers/simd.h
    src/arch/${ARCHITECTURE}/nnue/layers/sqr_clipped_relu.h
    src/arch/${ARCHITECTURE}/nnue/network.cpp
    src/arch/${ARCHITECTURE}/nnue/nnue_misc.cpp
    src/arch/${ARCHITECTURE}/nnue/network.h
    src/arch/${ARCHITECTURE}/nnue/nnue_accumulator.h
    src/arch/${ARCHITECTURE}/nnue/nnue_architecture.h
    src/arch/${ARCHITECTURE}/nnue/nnue_common.h
    src/arch/${ARCHITECTURE}/nnue/nnue_feature_transformer.h
    src/arch/${ARCHITECTURE}/nnue/nnue_misc.h
)

# Set output binary name to be "stockfish" (not "stockfishbin")
set_property(TARGET stockfishbin PROPERTY OUTPUT_NAME stockfish)

# Make stockfishbin depend on the stockfish library.
target_link_libraries(stockfishbin PUBLIC Stockfish::Stockfish)

##### Install #####

# Install the stockfish binary
install(TARGETS stockfishbin DESTINATION ${CMAKE_INSTALL_BINDIR})