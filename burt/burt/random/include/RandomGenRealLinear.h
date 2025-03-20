/** @file
* Emulate i.r.v via random congruent linear generator
*/
#pragma once

#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include <stdint.h>
#include <stddef.h>

namespace burt
{
    /** Emulate real i.r.v via random congruent linear generator
    */
    class RandomGenRealLinear
    {
    public:
        /** Get reference to some global instance of generator
        * @return global instance of this generator
        */
        static RandomGenRealLinear& global();

        /** Ctor. Initialize generator to generate random values in [A,B]
        * @param A lower bound of interval to generate numbers
        * @param B upper bound of interval to generate numbers
        * @param initSeed seed used to initialized random generator
        */
        RandomGenRealLinear(double A = 0.0, double B = 1.0, uint32_t initSeed = 1234)
        : xCurrent(0), 
          sA(A), 
          sB(B)
        {
            xCurrent = (uint32_t)(initSeed);
        }
        
        /** Get lower bound
        * @return lower bound
        */
        double getA() const;

        /** Get upper bound of generator
        * @return lower bound of generator
        */
        double getB() const;
        
        /** Generate pseudo random number in [0,1]
        * @return generated
        */
        template <class TFloatType = double>
        TFloatType generateRealInUnitInterval()
        {
            constexpr TFloatType my_inv_m = TFloatType(1.0) / TFloatType(4294967295.0);
            xCurrent = (a * xCurrent + c) % m;
            return (xCurrent * my_inv_m);
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
        double generateReal()
        {
            xCurrent = (a * xCurrent + c) % m;
            return sA + (xCurrent * inv_m) * (sB - sA);
        }
        
        /** Obtain information about last generate pseudo random number
        * @return generated
        */
        double last() const;

        /** Internal seed for random generator
        * @return seed value
        */
        uint32_t getSeed() const;
        
        /** Set seed as static function because CRT did not give many functions to operate on it.
        * @param seed used seed to initialize random generator
        */
        void setSeed(uint32_t seed);

    private:
        
        const double sA;
        const double sB;

        static double inv_m;

        uint32_t xCurrent;
        
        static uint32_t a;
        static uint32_t c;
        static uint32_t m;
    };
}
