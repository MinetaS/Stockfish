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

#ifndef I386_ARCH_H_INCLUDED
#define I386_ARCH_H_INCLUDED

#if !defined(__i386__) && !defined(__amd64__)
#error "Not supported in the current architecture."
#endif

#include <cassert>
#include <cstdint>

#if defined(__AVX__)

#include <immintrin.h>

#elif defined(__SSE4_1__)

#include <smmintrin.h>

#elif defined(__SSSE3__)

#include <tmmintrin.h>

#elif defined(__SSE2__)

#include <emmintrin.h>

#endif

namespace Stockfish {

inline int popcnt_64(std::uint64_t n) {
#if defined(__POPCNT__)
    return _mm_popcnt_u64(n);
#elif defined(__ABM__)
    return __popcnt64(n);
#else
    return __builtin_popcountll(n);
#endif
}

inline std::uint64_t blsr_64(std::uint64_t n) {
#ifdef __BMI__
    return _blsr_u64(n);
#else
    return n & (n - 1);
#endif
}

inline std::uint64_t tzcnt_64(std::uint64_t n) {
#ifdef __BMI__
    return _tzcnt_u64(n);
#else
    assert(n != 0);
    return __builtin_ctzll(n);
#endif
}

template<typename T>
inline T vzero() {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
        return _mm512_setzero_si512();
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
        return _mm256_setzero_si256();
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_setzero_si128();
#endif
}

template<typename T>
inline T vset_16(std::uint16_t n) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
        return _mm512_set1_epi16(n);
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
        return _mm256_set1_epi16(n);
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_set1_epi16(n);
#endif
}

template<typename T>
inline T vset_32(std::uint32_t n) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
        return _mm512_set1_epi32(n);
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
        return _mm256_set1_epi32(n);
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_set1_epi32(n);
#endif
}

template<typename T>
inline T vpackus_s16(T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
#ifdef __AVX512BW__
        return _mm512_packus_epi16(a, b);
#else
        static_assert(false, "vpackus_s16<__m512i> is not supported without AVX-512 BW.");
#endif
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_packus_epi16(a, b);
#else
        static_assert(false, "vpackus_s16<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_packus_epi16(a, b);
#endif
}

template<typename T>
inline T vadd_16(T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
#ifdef __AVX512BW__
        return _mm512_add_epi16(a, b);
#else
        static_assert(false, "vadd_16<__m512i> is not supportd without AVX-512 BW.");
#endif
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_add_epi16(a, b);
#else
        static_assert(false, "vadd_16<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_add_epi16(a, b);
#endif
}

template<typename T>
inline T vadd_32(T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
        return _mm512_add_epi32(a, b);
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_add_epi32(a, b);
#else
        static_assert(false, "vadd_32<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_add_epi32(a, b);
#endif
}

template<typename T>
inline T vsub_16(T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
#ifdef __AVX512BW__
        return _mm512_sub_epi16(a, b);
#else
        static_assert(false, "vsub_16<__m512i> is not supportd without AVX-512 BW.");
#endif
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_sub_epi16(a, b);
#else
        static_assert(false, "vsub_16<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_sub_epi16(a, b);
#endif
}

template<typename T>
inline T vsub_32(T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
        return _mm512_sub_epi32(a, b);
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_sub_epi32(a, b);
#else
        static_assert(false, "vsub_32<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_sub_epi32(a, b);
#endif
}

template<typename T>
inline T vmulhi_s16(T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
#ifdef __AVX512BW__
        return _mm512_mulhi_epi16(a, b);
#else
        static_assert(false, "vmulhi_16<__m512i> is not supportd without AVX-512 BW.");
#endif
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_mulhi_epi16(a, b);
#else
        static_assert(false, "vmulhi_16<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_mulhi_epi16(a, b);
#endif
}

template<typename T>
inline T vsll_16(T a, int n) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
#ifdef __AVX512BW__
        return _mm512_slli_epi16(a, n);
#else
        static_assert(false, "vsll_16<__m512i> is not supportd without AVX-512 BW.");
#endif
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_slli_epi16(a, n);
#else
        static_assert(false, "vsll_16<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_slli_epi16(a, n);
#endif
}

template<typename T>
inline T vmax_s16(T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
#ifdef __AVX512BW__
        return _mm512_max_epi16(a, b);
#else
        static_assert(false, "vmax_s16<__m512i> is not supportd without AVX-512 BW.");
#endif
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_max_epi16(a, b);
#else
        static_assert(false, "vmax_s16<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_max_epi16(a, b);
#endif
}

template<typename T>
inline T vmin_s16(T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
#ifdef __AVX512BW__
        return _mm512_min_epi16(a, b);
#else
        static_assert(false, "vmin_16<__m512i> is not supported without AVX-512 BW.");
#endif
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
#ifdef __AVX2__
        return _mm256_min_epi16(a, b);
#else
        static_assert(false, "vmin_16<__m256i> is not supported without AVX2.");
#endif
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
        return _mm_min_epi16(a, b);
#endif
}

template<typename T>
inline std::int32_t vaddv_s32(T a) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
        return _mm512_reduce_add_epi32(a);
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
    {
        __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        sum         = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_PERM_BADC));
        sum         = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_PERM_CDAB));
        return _mm_cvtsi128_si32(sum);
    }
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
    {
        a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4E));  // _MM_PERM_BADC
        a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xB1));  // _MM_PERM_CDAB
        return _mm_cvtsi128_si32(a);
    }
#endif
}

// Non-VNNI implementation of dpbusd works even with type saturation, only
// because output values are clamped in ReLU layers immediately after
// AffineTransform layer. Do not use this without VNNI for general purpose.
template<typename T>
inline void vdpbusd_s32(T& acc, T a, T b) {
#ifdef __AVX512F__
    if constexpr (sizeof(T) == 64)
    {
#if defined(__AVX512VNNI__)

        acc = _mm512_dpbusd_epi32(acc, a, b);

#elif defined(__AVX512BW__)

        __m512i product = _mm512_maddubs_epi16(a, b);
        product         = _mm512_madd_epi16(product, _mm512_set1_epi16(1));
        acc             = _mm512_add_epi32(acc, product);

#else
        static_assert(false, "vdpbusd_s32<__m512i> is not supportd without AVX-512 BW.");
#endif
    }
#endif

#ifdef __AVX__
    if constexpr (sizeof(T) == 32)
    {
#if (defined(__AVX512VL__) && defined(__AVX512VNNI__)) || defined(__AVXVNNI__)

        acc = _mm256_dpbusd_epi32(acc, a, b);

#elif defined(__AVX2__)

        __m256i product = _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), _mm256_set1_epi16(1));
        acc             = _mm256_add_epi32(acc, product);

#else
        static_assert(false, "vdpbusd_s32<__m256i> is not supported without AVX2.");
#endif
    }
#endif

#ifdef __SSE2__
    if constexpr (sizeof(T) == 16)
    {
#if (defined(__AVX512VL__) && defined(__AVX512VNNI__)) || defined(__AVXVNNI__)

        acc = _mm_dpbusd_epi32(acc, a, b);

#elif defined(__SSSE3__)

        __m128i product = _mm_madd_epi16(_mm_maddubs_epi16(a, b), _mm_set1_epi16(1));
        acc             = _mm_add_epi32(acc, product);

#else

        __m128i a0       = _mm_unpacklo_epi8(a, _mm_setzero_si128());
        __m128i a1       = _mm_unpackhi_epi8(a, _mm_setzero_si128());
        __m128i sgn      = _mm_cmplt_epi8(b, _mm_setzero_si128());
        __m128i b0       = _mm_unpacklo_epi8(b, sgn);
        __m128i b1       = _mm_unpackhi_epi8(b, sgn);
        __m128i product0 = _mm_madd_epi16(a0, b0);
        __m128i product1 = _mm_madd_epi16(a1, b1);
        __m128i product  = _mm_madd_epi16(_mm_packs_epi32(product0, product1), _mm_set1_epi16(1));
        acc              = _mm_add_epi32(acc, product);

#endif
    }
#endif
}

}  // namespace Stockfish

#endif  // I386_ARCH_H_INCLUDED
