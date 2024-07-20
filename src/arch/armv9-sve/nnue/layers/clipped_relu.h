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

#ifndef NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "nnue_common.h"
#include "nnue_misc.h"
#include "simd.h"

namespace Stockfish::Eval::NNUE::Layers {

template<IndexType InDims>
class ClippedReLU {
public:
    // Input/output type
    using InputType  = std::int32_t;
    using OutputType = std::uint8_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = InputDimensions;
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
        std::uint32_t hashValue = 0x538D24C7u;
        hashValue += prevHash;
        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream&) { return true; }

    // Write network parameters
    bool write_parameters(std::ostream&) const { return true; }

    // Forward propagation
    void propagate(const InputType* input, OutputType* output) const {
        const pred_t all = svptrue_b32();

        constexpr IndexType NumChunks = InputDimensions / (SimdWidth / 2);
        const int8x8_t      Zero      = {0};
        const auto          in        = reinterpret_cast<const vec_s32_t*>(input);
        const auto          out       = reinterpret_cast<vec_s8_t*>(output);

        for (IndexType i = 0; i < NumChunks; ++i)
        {
            int16x8_t  shifted;
            const auto pack = reinterpret_cast<int16x4_t*>(&shifted);

            svasr_n_s32_z(all, in[i * 2 + 0], WeightScaleBits);
            svasr_n_s32_z(all, in[i * 2 + 1], WeightScaleBits);
            pack[0]         = vqshrn_n_s32(in[i * 2 + 0], WeightScaleBits);
            pack[1]         = vqshrn_n_s32(in[i * 2 + 1], WeightScaleBits);
            out[i]          = vmax_s8(vqmovn_s16(shifted), Zero);
        }
        constexpr IndexType Start = NumChunks * (SimdWidth / 2);

        // Calculate remaining bytes
        for (IndexType i = Start; i < InputDimensions; ++i)
            output[i] = static_cast<OutputType>(std::clamp(input[i] >> WeightScaleBits, 0, 127));
    }

private:
    static constexpr RegisterSize = sizeof(vec_s32_t);
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
