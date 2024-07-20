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

# CMake script for downloading the NNUE network file.
#
# NOTE: Run this in CMake script mode (`cmake -P`)!!

file(STRINGS "${SOURCE_DIR}/src/evaluate.h" EVALFILE_BIG REGEX "[^\n]+EvalFileDefaultNameBig[^\n]+")
file(STRINGS "${SOURCE_DIR}/src/evaluate.h" EVALFILE_SMALL REGEX "[^\n]+EvalFileDefaultNameSmall[^\n]+")
string(REGEX MATCH "(nn-[a-z0-9]+.nnue)" NNUE_BIG "${EVALFILE_BIG}")
string(REGEX MATCH "(nn-[a-z0-9]+.nnue)" NNUE_SMALL "${EVALFILE_SMALL}")

if(NOT NNUE_BIG OR NNUE_SMALL)
    message(FATAL_ERROR "Could not extract NNUE filename")
endif()

message("Default network: ${NNUE_BIG}, ${NNUE_SMALL}")
set(NNUE_BIG_FILE "${BUILD_DIR}/${NNUE_BIG}")
set(NNUE_BIG_URL "https://tests.stockfishchess.org/api/nn/${NNUE_BIG}")
set(NNUE_SMALL_FILE "${BUILD_DIR}/${NNUE_SMALL}")
set(NNUE_SMALL_URL "https://tests.stockfishchess.org/api/nn/${NNUE_SMALL}")

function(fetch_network nnue)
    set(NNUE_FILE "${BUILD_DIR}/${nnue}")

    if(NOT EXISTS "${NNUE_FILE}")
        set(NNUE_URL "https://tests.stockfishchess.org/api/nn/${nnue}")
        message("Downloading from ${NNUE_URL}")
        file(DOWNLOAD "${NNUE_URL}" "${NNUE_FILE}")
    endif()

    file(SHA256 ${NNUE_FILE} SHASUM)
    string(SUBSTRING ${SHASUM} 0 12 ACTUAL)
    string(SUBSTRING ${NNUE} 3 12 EXPECTED)

    if(NOT ACTUAL STREQUAL EXPECTED)
        message(FATAL_ERROR "Download failed or corrupted - please delete ${NNUE_FILE}")
    endif()

    message("Network {nnue} validated")
endfunction()
