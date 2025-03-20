#include "RandomGenIntegerLinear.h"
#include <stdint.h>

namespace burt
{
    RandomGenIntegerLinear& RandomGenIntegerLinear::global()
    {
        static RandomGenIntegerLinear global;
        return global;
    }

    RandomGenIntegerLinear::RandomGenIntegerLinear(uint32_t A, uint32_t B, uint32_t initSeed)
    : xCurrent(0), sA(A), sB(B), a(214013), c(2531011)
    {
        // Numerical recipies in C. The art of scientific computing. 2-nd edition, Cambridge University Press, 1992., 925 pp.
        // https://en.wikipedia.org/wiki/Linear_congruential_generator

        xCurrent = (uint32_t) (initSeed);
        allUint32AreaIsUsed = (A == 0U && B == 4294967295U) ? true : false;

#if (SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD) && SUPPORT_VECTORIZATION_FOR_RND_NUMBER_GEN
        {
            // xCurrent = (a * xCurrent + c);
            // xCurrent = (a * (a * xCurrent + c) + c) = a^2 * xCurrent + a * c + c
            // xCurrent = (a * (a * (a * xCurrent + c); + c) + c) = a^3 * xCurrent + a^2 * c + a * c + c

            constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecTypeUI32>();
            
            alignas(kVecBatchSize * sizeof(uint32_t)) uint32_t aVec[kVecBatchSize];
            alignas(kVecBatchSize * sizeof(uint32_t)) uint32_t cVec[kVecBatchSize];
            
            aVec[0] = a;
            cVec[0] = c;
            for (size_t i = 1; i < kVecBatchSize; ++i)
            {
                aVec[i] = aVec[i - 1] * a;
                cVec[i] = a * cVec[i - 1] + c;
            }
            aVecSimd.load_a(aVec);
            cVecSimd.load_a(cVec);
        }
#endif
        
    }

    uint32_t RandomGenIntegerLinear::last() const
    {
        constexpr uint32_t m_uint32 = 4294967295;
        constexpr double m_inv_double = 1.0 / double(m_uint32);

        if (allUint32AreaIsUsed)
            return xCurrent;
        else
            return sA + roundToNearestInt<decltype(sA)>(xCurrent * m_inv_double * (sB - sA));
    }

    uint32_t RandomGenIntegerLinear::getSeed() const
    {
        return xCurrent;
    }

    void RandomGenIntegerLinear::setSeed(uint32_t seed)
    {
        xCurrent = seed;
    }

    uint32_t RandomGenIntegerLinear::getA() const
    {
        return sA;
    }

    uint32_t RandomGenIntegerLinear::getB() const
    {
        return sB;
    }
}
