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

#ifndef I386_NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define I386_NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#error "Never use architecture specific header files directly."
#endif

// Check x86 SIMD extensions.
// If none is defined, fall back to the generic implementation.
#ifndef __SSE2__

#include "arch/generic/nnue/nnue_feature_transformer.h"

#else

#include "../arch.h"

#include <algorithm>
#include <cstring>

#include "position.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/nnue_common.h"
#include "types.h"

namespace Stockfish::Eval::NNUE {

template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
struct FeatureTransformer<TransformedFeatureDimensions, accPtr>::Details {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    // The size of the current PSQT weights array is too small for AVX-512.
    using vec_t      = __m512i;
    using psqt_vec_t = __m256i;
#elif defined(__AVX2__)
    using vec_t      = __m256i;
    using psqt_vec_t = __m256i;
#else
    using vec_t      = __m128i;
    using psqt_vec_t = __m128i;
#endif

   private:
#if defined(__AVX512F__)
    static constexpr int NumXMM = 32;  // EVEX encoding scheme
#else
    static constexpr int NumXMM = Is64Bit ? 16 : 8;
#endif

    template<std::size_t RegisterSize, std::size_t LaneSize, int NumLanes>
    static constexpr int optimal_register_count() {
        static_assert(RegisterSize > 0 && LaneSize > 0 && NumLanes > 0);
        static_assert(RegisterSize >= LaneSize && RegisterSize % LaneSize == 0);
        static_assert((NumLanes * LaneSize) % RegisterSize == 0);

        // The exact number of registers that can fit in the whole input
        // vectors.
        constexpr int Ideal = (NumLanes * LaneSize) / RegisterSize;

        if constexpr (Ideal <= NumXMM)
            return Ideal;

        // Look for the largest divisor of the ideal register count that is
        // smaller than NumXMM.
        for (int divisor = NumXMM; divisor > 1; --divisor)
            if (Ideal % divisor == 0)
                return divisor;

        return 1;
    }

   public:
    static constexpr std::size_t AccRegisterSize  = sizeof(vec_t);
    static constexpr std::size_t PSQTRegisterSize = sizeof(psqt_vec_t);

    static constexpr int OptimalAccRegisterCount =
      optimal_register_count<AccRegisterSize, sizeof(WeightType), TransformedFeatureDimensions>();
    static constexpr int OptimalPSQTRegisterCount =
      optimal_register_count<PSQTRegisterSize, sizeof(PSQTWeightType), PSQTBuckets>();

    static constexpr IndexType TileHeight     = OptimalAccRegisterCount * AccRegisterSize / 2;
    static constexpr IndexType PsqtTileHeight = OptimalPSQTRegisterCount * PSQTRegisterSize / 4;

    static_assert(HalfDimensions % TileHeight == 0,
                  "HalfDimensions must be multiple of TileHeight");
    static_assert(PSQTBuckets % PsqtTileHeight == 0,
                  "PSQTBuckets must be multiple of PsqtTileHeight");
};

template<std::size_t RegisterSize, bool Write>
static inline constexpr void permute_pack(std::uint64_t* v) {
    if constexpr (RegisterSize == 64)
        if constexpr (Write)
        {
            std::uint64_t tmp0 = v[2], tmp1 = v[3];
            v[2] = v[8], v[3] = v[9];
            v[8] = v[4], v[9] = v[5];
            v[4] = tmp0, v[5] = tmp1;
            tmp0 = v[6], tmp1 = v[7];
            v[6] = v[10], v[7] = v[11];
            v[10] = v[12], v[11] = v[13];
            v[12] = tmp0, v[13] = tmp1;
        }
        else
        {
            std::uint64_t tmp0 = v[2], tmp1 = v[3];
            v[2] = v[4], v[3] = v[5];
            v[4] = v[8], v[5] = v[9];
            v[8] = tmp0, v[9] = tmp1;
            tmp0 = v[6], tmp1 = v[7];
            v[6] = v[12], v[7] = v[13];
            v[12] = v[10], v[13] = v[11];
            v[10] = tmp0, v[11] = tmp1;
        }
    else if constexpr (RegisterSize == 32)
    {
        std::swap(v[2], v[4]);
        std::swap(v[3], v[5]);
    }
}

template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
template<bool Write>
void FeatureTransformer<TransformedFeatureDimensions, accPtr>::permute_weights() {
    // The weight numbers are permuted preliminarily, due to the use of
    // AVX2/AVX-512 pack intrinsics.
    if constexpr (Details::AccRegisterSize >= 32)
    {
        constexpr IndexType Width = Details::AccRegisterSize == 64 ? 16 : 8;

        for (IndexType i = 0; i < HalfDimensions * sizeof(BiasType) / 8; i += Width)
            permute_pack<Details::AccRegisterSize, Write>(
              &reinterpret_cast<std::uint64_t*>(biases)[i]);

        for (IndexType j = 0; j < InputDimensions; ++j)
            for (IndexType i = 0; i < HalfDimensions * sizeof(WeightType) / 8; i += Width)
                permute_pack<Details::AccRegisterSize, Write>(
                  &reinterpret_cast<std::uint64_t*>(&weights[j * HalfDimensions])[i]);
    }
}

template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
template<Color Perspective, size_t N>
void FeatureTransformer<TransformedFeatureDimensions, accPtr>::
  apply_accumulator_updates_incremental(StateInfo*            computed_st,
                                        StateInfo*            states_to_update[N],
                                        FeatureSet::IndexList removed[N],
                                        FeatureSet::IndexList added[N]) const {
    using vec_t      = typename Details::vec_t;
    using psqt_vec_t = typename Details::psqt_vec_t;

    StateInfo* st = computed_st;

    // The most common case when updating the accumulator incrementally.
    // Calculates feature differences directly without using tiling mechanism.
    if (N == 1 && (removed[0].size() == 1 || removed[0].size() == 2) && added[0].size() == 1)
    {
        const auto accIn =
          reinterpret_cast<const vec_t*>(&(st->*accPtr).accumulation[Perspective][0]);
        const auto accOut =
          reinterpret_cast<vec_t*>(&(states_to_update[0]->*accPtr).accumulation[Perspective][0]);

        const IndexType offsetR0 = HalfDimensions * removed[0][0];
        const auto      columnR0 = reinterpret_cast<const vec_t*>(&weights[offsetR0]);
        const IndexType offsetA  = HalfDimensions * added[0][0];
        const auto      columnA  = reinterpret_cast<const vec_t*>(&weights[offsetA]);

        if (removed[0].size() == 1)
        {
            for (IndexType k = 0; k < HalfDimensions * sizeof(WeightType) / sizeof(vec_t); ++k)
                accOut[k] = vadd_16(vsub_16(accIn[k], columnR0[k]), columnA[k]);
        }
        else
        {
            const IndexType offsetR1 = HalfDimensions * removed[0][1];
            auto            columnR1 = reinterpret_cast<const vec_t*>(&weights[offsetR1]);

            for (IndexType k = 0; k < HalfDimensions * sizeof(WeightType) / sizeof(vec_t); ++k)
                accOut[k] =
                  vsub_16(vadd_16(accIn[k], columnA[k]), vadd_16(columnR0[k], columnR1[k]));
        }

        const auto accPsqtIn =
          reinterpret_cast<const psqt_vec_t*>(&(st->*accPtr).psqtAccumulation[Perspective][0]);
        const auto accPsqtOut = reinterpret_cast<psqt_vec_t*>(
          &(states_to_update[0]->*accPtr).psqtAccumulation[Perspective][0]);

        const IndexType offsetPsqtR0 = PSQTBuckets * removed[0][0];
        auto columnPsqtR0 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtR0]);
        const IndexType offsetPsqtA = PSQTBuckets * added[0][0];
        auto columnPsqtA = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtA]);

        if (removed[0].size() == 1)
        {
            for (IndexType k = 0; k < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t);
                 ++k)
                accPsqtOut[k] = vadd_32(vsub_32(accPsqtIn[k], columnPsqtR0[k]), columnPsqtA[k]);
        }
        else
        {
            const IndexType offsetPsqtR1 = PSQTBuckets * removed[0][1];
            auto columnPsqtR1 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtR1]);

            for (IndexType k = 0; k < PSQTBuckets * sizeof(PSQTWeightType) / sizeof(psqt_vec_t);
                 ++k)
                accPsqtOut[k] = vadd_32(vadd_32(accPsqtIn[k], columnPsqtA[k]),
                                        vsub_32(columnPsqtR0[k], columnPsqtR1[k]));
        }
    }
    else
    {
        // Update accumulator
        vec_t acc[Details::OptimalAccRegisterCount];

        for (IndexType j = 0; j < HalfDimensions / Details::TileHeight; ++j)
        {
            const IndexType offsetRow = j * Details::TileHeight;

            const auto accTileIn =
              reinterpret_cast<const vec_t*>(&(st->*accPtr).accumulation[Perspective][offsetRow]);
            for (std::size_t k = 0; k < array_size(acc); ++k)
                acc[k] = accTileIn[k];

            for (IndexType i = 0; i < N; ++i)
            {
                for (const auto index : removed[i])
                {
                    const IndexType offset = HalfDimensions * index + offsetRow;
                    const auto      column = reinterpret_cast<const vec_t*>(&weights[offset]);
                    for (std::size_t k = 0; k < array_size(acc); ++k)
                        acc[k] = vsub_16(acc[k], column[k]);
                }

                for (const auto index : added[i])
                {
                    const IndexType offset = HalfDimensions * index + offsetRow;
                    const auto      column = reinterpret_cast<const vec_t*>(&weights[offset]);
                    for (std::size_t k = 0; k < array_size(acc); ++k)
                        acc[k] = vadd_16(acc[k], column[k]);
                }

                auto accTileOut = reinterpret_cast<vec_t*>(
                  &(states_to_update[i]->*accPtr).accumulation[Perspective][offsetRow]);
                for (std::size_t k = 0; k < array_size(acc); ++k)
                    accTileOut[k] = acc[k];
            }
        }

        // Update PSQT
        psqt_vec_t psqt[Details::OptimalPSQTRegisterCount];

        for (IndexType j = 0; j < PSQTBuckets / Details::PsqtTileHeight; ++j)
        {
            const IndexType offsetRow = j * Details::PsqtTileHeight;

            auto accTilePsqtIn = reinterpret_cast<const psqt_vec_t*>(
              &(st->*accPtr).psqtAccumulation[Perspective][offsetRow]);
            for (std::size_t k = 0; k < array_size(psqt); ++k)
                psqt[k] = accTilePsqtIn[k];

            for (IndexType i = 0; i < N; ++i)
            {
                for (const auto index : removed[i])
                {
                    const IndexType offset = PSQTBuckets * index + offsetRow;
                    auto columnPsqt = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);
                    for (std::size_t k = 0; k < array_size(psqt); ++k)
                        psqt[k] = vsub_32(psqt[k], columnPsqt[k]);
                }

                for (const auto index : added[i])
                {
                    const IndexType offset = PSQTBuckets * index + offsetRow;
                    auto columnPsqt = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);
                    for (std::size_t k = 0; k < array_size(psqt); ++k)
                        psqt[k] = vsub_32(psqt[k], columnPsqt[k]);
                }

                auto accTilePsqtOut = reinterpret_cast<psqt_vec_t*>(
                  &(states_to_update[i]->*accPtr).psqtAccumulation[Perspective][offsetRow]);
                for (std::size_t k = 0; k < array_size(psqt); ++k)
                    accTilePsqtOut[k] = psqt[k];
            }
        }
    }
}

template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
template<Color Perspective>
void FeatureTransformer<TransformedFeatureDimensions, accPtr>::
  apply_accumulator_updates_refresh_cache(
    Accumulator<TransformedFeatureDimensions>&                accumulator,
    typename AccumulatorCaches::Cache<HalfDimensions>::Entry& entry,
    FeatureSet::IndexList                                     removed,
    FeatureSet::IndexList                                     added) const {
    using vec_t      = typename Details::vec_t;
    using psqt_vec_t = typename Details::psqt_vec_t;

    vec_t acc[Details::OptimalAccRegisterCount];

    for (IndexType j = 0; j < HalfDimensions / Details::TileHeight; ++j)
    {
        const IndexType offsetRow = j * Details::TileHeight;

        const auto accTile =
          reinterpret_cast<vec_t*>(&accumulator.accumulation[Perspective][offsetRow]);
        const auto entryTile = reinterpret_cast<vec_t*>(&entry.accumulation[offsetRow]);

        for (IndexType k = 0; k < array_size(acc); ++k)
            acc[k] = entryTile[k];

        std::size_t i = 0;
        for (; i < std::min(removed.size(), added.size()); ++i)
        {
            const IndexType offsetR = HalfDimensions * removed[i] + offsetRow;
            const auto      columnR = reinterpret_cast<const vec_t*>(&weights[offsetR]);
            const IndexType offsetA = HalfDimensions * added[i] + offsetRow;
            const auto      columnA = reinterpret_cast<const vec_t*>(&weights[offsetA]);

            for (std::size_t k = 0; k < array_size(acc); ++k)
                acc[k] = vadd_16(acc[k], vsub_16(columnA[k], columnR[k]));
        }
        for (; i < removed.size(); ++i)
        {
            const IndexType offset = HalfDimensions * removed[i] + offsetRow;
            const auto      column = reinterpret_cast<const vec_t*>(&weights[offset]);

            for (std::size_t k = 0; k < array_size(acc); ++k)
                acc[k] = vsub_16(acc[k], column[k]);
        }
        for (; i < added.size(); ++i)
        {
            const IndexType offset = HalfDimensions * added[i] + offsetRow;
            const auto      column = reinterpret_cast<const vec_t*>(&weights[offset]);

            for (std::size_t k = 0; k < array_size(acc); ++k)
                acc[k] = vadd_16(acc[k], column[k]);
        }

        for (IndexType k = 0; k < array_size(acc); k++)
            entryTile[k] = acc[k];
        for (IndexType k = 0; k < array_size(acc); k++)
            accTile[k] = acc[k];
    }

    psqt_vec_t psqt[Details::OptimalPSQTRegisterCount];

    for (IndexType j = 0; j < PSQTBuckets / Details::PsqtTileHeight; ++j)
    {
        const IndexType offsetRow = j * Details::PsqtTileHeight;

        const auto accTilePsqt =
          reinterpret_cast<psqt_vec_t*>(&accumulator.psqtAccumulation[Perspective][offsetRow]);
        const auto entryTilePsqt =
          reinterpret_cast<psqt_vec_t*>(&entry.psqtAccumulation[offsetRow]);

        for (std::size_t k = 0; k < array_size(psqt); ++k)
            psqt[k] = entryTilePsqt[k];

        for (std::size_t i = 0; i < removed.size(); ++i)
        {
            const IndexType offset     = PSQTBuckets * removed[i] + offsetRow;
            const auto      columnPsqt = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);

            for (std::size_t k = 0; k < array_size(psqt); ++k)
                psqt[k] = vsub_32(psqt[k], columnPsqt[k]);
        }
        for (std::size_t i = 0; i < added.size(); ++i)
        {
            const IndexType offset     = PSQTBuckets * added[i] + offsetRow;
            const auto      columnPsqt = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);

            for (std::size_t k = 0; k < array_size(psqt); ++k)
                psqt[k] = vadd_32(psqt[k], columnPsqt[k]);
        }

        for (std::size_t k = 0; k < array_size(psqt); ++k)
            entryTilePsqt[k] = psqt[k];
        for (std::size_t k = 0; k < array_size(psqt); ++k)
            accTilePsqt[k] = psqt[k];
    }
}

template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
void FeatureTransformer<TransformedFeatureDimensions, accPtr>::convert_accumulators(
  const Position& pos, OutputType* output) const {
    using vec_t = typename Details::vec_t;

    static constexpr IndexType OutputChunkSize = Details::AccRegisterSize / sizeof(OutputType);
    static_assert((HalfDimensions / 2) % OutputChunkSize == 0);

    static constexpr IndexType NumOutputChunks = HalfDimensions / 2 / OutputChunkSize;

    const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
    const auto& accumulation    = (pos.state()->*accPtr).accumulation;

    for (IndexType p = 0; p < 2; ++p)
    {
        const auto in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0]));
        const auto in1 =
          reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][HalfDimensions / 2]));
        const auto out = reinterpret_cast<vec_t*>(&output[(HalfDimensions / 2) * p]);

        for (IndexType j = 0; j < NumOutputChunks; ++j)
        {
            // What we want to do is multiply inputs in a pairwise manner
            // (after clipping), and then shift right by 9. Instead, we
            // shift left by 7, and use mulhi, stripping the bottom 16 bits,
            // effectively shifting right by 16, resulting in a net shift
            // of 9 bits. We use mulhi because it maintains the sign of
            // the multiplication (unlike mullo), allowing us to make use
            // of packus to clip 2 of the inputs, resulting in a save of 2
            // "vmax_s16" calls.

            static const vec_t Zeroes = vzero<vec_t>();
            static const vec_t Ones   = vset_16<vec_t>(127 * 2);

            const vec_t sum0a = vsll_16(vmax_s16(vmin_s16(in0[j * 2 + 0], Ones), Zeroes), 7);
            const vec_t sum0b = vsll_16(vmax_s16(vmin_s16(in0[j * 2 + 1], Ones), Zeroes), 7);
            const vec_t sum1a = vmin_s16(in1[j * 2 + 0], Ones);
            const vec_t sum1b = vmin_s16(in1[j * 2 + 1], Ones);

            const vec_t pa = vmulhi_s16(sum0a, sum1a);
            const vec_t pb = vmulhi_s16(sum0b, sum1b);

            out[j] = vpackus_s16(pa, pb);
        }
    }
}

}  // namespace Stockfish::Eval::NNUE

#endif  // !__SSE2__

#endif  // I386_NNUE_FEATURE_TRANSFORMER_H_INCLUDED
