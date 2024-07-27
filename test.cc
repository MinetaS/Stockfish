#include <algorithm>
#include <iostream>

#include <immintrin.h>
int main() {
    __m256i v1 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    __m256i v2 = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);

    __mmask8 mask = _mm256_cmpgt_epi32_mask(v1, v2);
    __m256i r = _mm256_cmpgt_epi32(v1, v2);
    _mm256_castsi256_ps;
    _mm256_movemask_ps;
    return 0;
}