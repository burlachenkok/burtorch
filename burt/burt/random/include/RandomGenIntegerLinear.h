/** @file
* Emulate discrete integer r.v via random congruent linear generator
*/

#pragma once

#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include <stdint.h>
#include <stddef.h>

// Configuration Variable for Support Special Batched Vectorization mode during generating random numbers
#define SUPPORT_VECTORIZATION_FOR_RND_NUMBER_GEN 0

namespace burt
{
    /**Emulate discrete integer r.v via random congruent linear generator
    * @remark https://en.wikipedia.org/wiki/Linear_congruential_generator
    */
    class RandomGenIntegerLinear
    {
    public:
        /** Get reference to some global instance of generator
        * @return global instance of this generator
        */
        static RandomGenIntegerLinear& global();

        /** Ctor. Initialize generator to generate random values in [A,B]
        * @param A lower bound of interval to generate numbers
        * @param B upper bound of interval to generate numbers
        * @param initSeed seed used to initialized random generator
        */
        RandomGenIntegerLinear(uint32_t A = 0U, uint32_t B = 4294967295U, uint32_t initSeed = 1234);
        
        /** Get lower bound
        * @return lower bound
        */
        uint32_t getA() const;

        /** Get upper bound of generator
        * @return lower bound of generator
        */
        uint32_t getB() const;

        /** Generate pseudo random number in [0,1]
        * @return generated
        */
        template <class TFloatType = double>
        TFloatType generateRealInUnitInterval()
        {
            xCurrent = (a * xCurrent + c);

            constexpr uint32_t m_uint32 = 4294967295;
            constexpr TFloatType m_inv_double = 1.0 / TFloatType(m_uint32);
            TFloatType result = xCurrent * m_inv_double;
            return result;            
        }

        /** Generate batch of pseudo random number in [0,1]
        * @param placeHolder result array
        * @param kNumbers numbers to generate
        * @return nothing
        */
        template <class TFloatType>
        void generateBatchOfRealsInUnitInterval(TFloatType* placeHolder, size_t kNumbers)
        {
#if (SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD) && SUPPORT_VECTORIZATION_FOR_RND_NUMBER_GEN
            constexpr uint32_t m_uint32 = 4294967295;
            constexpr TFloatType m_inv_double = 1.0 / TFloatType(m_uint32);
            
            constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecTypeUI32>();
            size_t items = burt::roundToNearestMultipleDown<kVecBatchSize>(kNumbers);
            
            alignas(kVecBatchSize * sizeof(uint32_t)) uint32_t resultVec[kVecBatchSize];
            
            size_t i = 0;
            for (; i < items; i += kVecBatchSize)
            {
                VecTypeUI32 xCurrentSimd(xCurrent);
                xCurrentSimd = aVecSimd * xCurrentSimd + cVecSimd;
                xCurrentSimd.store_a(resultVec);
                
                xCurrent = static_cast<uint32_t>(xCurrentSimd[kVecBatchSize - 1]);
                
                for (size_t kk = 0; kk < kVecBatchSize; ++kk)
                {
                    placeHolder[i + kk] = resultVec[kk] * m_inv_double;
                }
            }
            
            for (; i < kNumbers; ++i)
            {
                placeHolder[i] = generateRealInUnitInterval<TFloatType>();
            } 
#else
            for (size_t i = 0; i < kNumbers; ++i)
                placeHolder[i] = generateRealInUnitInterval<TFloatType>();
#endif
            return;

        }

        /** Generate pseudo random number
        * @return generated
        */
        uint32_t generateInteger()
        {
            constexpr uint32_t m_uint32 = 4294967295;
            constexpr double m_inv_double = 1.0 / double(m_uint32);

            xCurrent = (a * xCurrent + c);

            if (allUint32AreaIsUsed)
            {
                return xCurrent;
            }
            else
            {
                uint32_t result = sA + roundToNearestInt<decltype(sA)>(xCurrent * m_inv_double * (sB - sA));
                return result;
            }
        }
        
        /** Obtain information about last generate pseudo random number
        * @return generated
        */
        uint32_t last() const;

        /** Internal seed for random generator
        * @return seed value
        */
        uint32_t getSeed() const;
        
        /** Set seed as static function because CRT did not give many functions to operate on it
        * @param seed used seed to initialize random generator
        */
        void setSeed(uint32_t seed);

    private:
        bool allUint32AreaIsUsed; ///< flag that [sA, sB] = [0U, 4294967295U]
        uint32_t xCurrent;   ///< current seed
        uint32_t sA;         ///< start of interval for generate random integers [A,B]
        uint32_t sB;         ///< end of interval for generate random integers [A,B]
        uint32_t a;          ///< Linear param near X for recurrent generate formula
        uint32_t c;          ///< Free param near X for recurrent generate formula

#if (SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD) && SUPPORT_VECTORIZATION_FOR_RND_NUMBER_GEN
        typedef burt::VectorSimdTraits<uint32_t, cpu_extension>::VecType VecTypeUI32;
        VecTypeUI32 aVecSimd;
        VecTypeUI32 cVecSimd;
#endif
        
    };
}
