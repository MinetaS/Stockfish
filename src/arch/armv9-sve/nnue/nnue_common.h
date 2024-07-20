/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "misc.h"

#include <arm_sve.h>

namespace Stockfish::Eval::NNUE {

// Version of the evaluation file
constexpr std::uint32_t Version = 0x7AF32F20u;

// Constant used in evaluation value calculation
constexpr int OutputScale     = 16;
constexpr int WeightScaleBits = 6;

// Size of cache line (in bytes)
constexpr std::size_t CacheLineSize = 64;

constexpr const char        Leb128MagicString[]   = "COMPRESSED_LEB128";
constexpr const std::size_t Leb128MagicStringSize = sizeof(Leb128MagicString) - 1;

// SIMD width (in bytes)
/// TODO: __ARM_FEATURE_SVE_BITS
constexpr std::size_t SimdWidth    = 16;
constexpr std::size_t MaxSimdWidth = 32;

// Type of input feature after conversion
using TransformedFeatureType = std::uint8_t;
using IndexType              = std::uint32_t;

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_COMMON_H_INCLUDED
