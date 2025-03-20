/** @file
* Emulate discrete integer r.v via xorshift* algorithm
*/

#pragma once

#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include <stdint.h>
#include <stddef.h>

namespace burt
{
    /** Emulate discrete integer r.v via xorshift* algorithm
    * @remark https://en.wikipedia.org/wiki/Xorshift
    */
    class RandomGenXorShiftInteger
    {
    public:
        /** Get reference to some global instance of generator
        * @return global instance of this generator
        */
        static RandomGenXorShiftInteger& global();

        /** Ctor. Initialize generator to generate random values in [A,B]
        * @param A lower bound of interval to generate numbers
        * @param B upper bound of interval to generate numbers
        * @param seed seed used to initialized random generator
        */
        RandomGenXorShiftInteger(uint32_t A = 0U, uint32_t B = 4294967295U, uint64_t seed = 6712);
        
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
            // https://en.wikipedia.org/wiki/Xorshift (xorshift*)
            xCurrent ^= xCurrent >> 12; // a
            xCurrent ^= xCurrent << 25; // b
            xCurrent ^= xCurrent >> 27; // c

            xCurrent = xCurrent * (2685821657736338717ULL);

            constexpr uint32_t m_uint32 = 4294967295;
            constexpr TFloatType m_inv_double = 1.0 / TFloatType(m_uint32);
            TFloatType res = (uint32_t)xCurrent * m_inv_double;
            return res;
        }

        /** Generate batch of pseudo random number in [0,1]
        * @param placeHolder result array
        * @param kNumbers numbers to generate
        * @return nothing
        */
        template <class TFloatType>
        void generateBatchOfRealsInUnitInterval(TFloatType* placeHolder, size_t kNumbers)
        {
            for (size_t i = 0; i < kNumbers; ++i)
                placeHolder[i] = generateRealInUnitInterval<TFloatType>();
            
            return;
        }
        
        /** Generate pseudo random number
        * @return generated
        */
        uint32_t generateInteger()
        {
            // https://en.wikipedia.org/wiki/Xorshift (xorshift*)
            xCurrent ^= xCurrent >> 12; // a
            xCurrent ^= xCurrent << 25; // b
            xCurrent ^= xCurrent >> 27; // c
            
            xCurrent = xCurrent * (2685821657736338717ULL);
            
            if (allUint32AreaIsUsed)
            {
                return (uint32_t)xCurrent;
            }
            else
            {
                constexpr uint32_t m_uint32 = 4294967295;
                constexpr double m_inv_double = 1.0 / double(m_uint32);
                return sA + roundToNearestInt<decltype(sA)>( ( (uint32_t)xCurrent ) * m_inv_double * (sB - sA));
            }
        }
        
        /** Obtain information about last generate pseudo random number
        * @return generated
        */
        uint32_t last() const;

        /** Internal seed for random generator
        * @return seed value
        */
        uint64_t getSeed() const;
        
        /** Set seed as static function because CRT did not give many functions to operate on it
        * @param seed used seed to initialize random generator
        */
        void setSeed(uint64_t seed);

    private:
        bool allUint32AreaIsUsed; ///< flag that [sA, sB] = [0U, 4294967295U]
        uint64_t xCurrent;   ///< current seed
        uint32_t sA;         ///< start of interval for generate random integers
        uint32_t sB;         ///< end of interval for generate random integers
        uint32_t a;          ///< linear param near X for recurrent generate formula
        uint32_t c;          ///< free param near X for recurrent generate formula
    };
}
