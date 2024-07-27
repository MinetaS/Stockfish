#include <cstdint>
#include <iostream>

#include <arm_sve.h>
#include <arm_neon.h>

static constexpr size_t SVERegisterSize = __ARM_FEATURE_SVE_BITS;
static_assert(SVERegisterSize % 128 == 0);

#define __sve_vlst__ __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)))

using vec_s8_t  = svint8_t   __sve_vlst__;
using vec_u8_t  = svuint8_t  __sve_vlst__;
using vec_s16_t = svint16_t  __sve_vlst__;
using vec_u16_t = svuint16_t __sve_vlst__;
using vec_s32_t = svint32_t  __sve_vlst__;
using vec_u32_t = svuint32_t __sve_vlst__;
using vec_s64_t = svint64_t  __sve_vlst__;
using vec_u64_t = svuint64_t __sve_vlst__;
using pred_t    = svbool_t   __sve_vlst__;

void print32(vec_s32_t v) {
    int32_t out[4];
    svst1_s32(svptrue_b32(), out, v);

    for (int i = 0; i < 4; i++) {
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;
}

void test_sve(const int32_t *data) {
    vec_s32_t v = svld1_s32(svptrue_b32(), data);
    print32(v);
    v = svld1_vnum_s32(svptrue_b32(), data, 1);
    print32(v);
}

int main() {
    int32_t data[2][4] = {
        {-50000, -4000, 4000, 50000},
        {-10000, -1000, 1000, 10000}
    };

    test_sve(reinterpret_cast<int32_t *>(data));

    return 0;
}