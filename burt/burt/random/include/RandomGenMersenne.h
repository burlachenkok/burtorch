/** @file
* Emulate integer r.v via random Mersenne Twister Generator
*/

#pragma once

#include <stdint.h>
#include <stddef.h>

namespace burt
{
    /** Emulate integer r.v via random Mersenne Twister Generator. 
    * @remark The Mersenne Twister was developed in 1997. It was designed specifically to rectify most of the flaws found in older PRNGs.
    */
    class RandomGenMersenne
    {
    public:
        /** Get reference to some global instance of generator
        * @return global instance of this generator
        */
        static RandomGenMersenne& global();

        /** Ctor. Initialize generator to generate random values in [A,B]
        * @param A lower bound of interval to generate numbers
        * @param B upper bound of interval to generate numbers
        * @param initSeed seed used to initialized random generator
        */
        RandomGenMersenne(int initSeed = 1234);

        /** Get lower bound of generator for generateInteger()
        * @return lower bound
        */
        uint32_t getA() const;

        /** Get upper bound of generator for generateInteger()
        * @return lower bound of generator
        */
        uint32_t getB() const;
        
        /** Generate integer pseudo random number  in range [a,b]
        * @return generated number
        */
        uint32_t generateInteger();
        
        /** Generate integer pseudo random number  in range [a,b]
        * @param a start of interval
        * @param b end of interval
        * @return generated number
        */
        uint32_t generateInteger(uint32_t a, uint32_t b);

        /** Generate real pseudo random number in range [0,1]
        * @return generated number
        */
        template <class TFloatType = double>
        TFloatType generateRealInUnitInterval()
        {
            const double res = Mersenne_Random();
            return (TFloatType)res;
        }


        /** Generate batch of pseudo random number in [0,1]
        * @param placeHolder result array
        * @param kNumbers numbers to generate
        * @retur nothing
        */
        template <class TFloatType>
        void generateBatchOfRealsInUnitInterval(TFloatType* placeHolder, size_t kNumbers)
        {
            for (size_t i = 0; i < kNumbers; ++i)
                placeHolder[i] = generateRealInUnitInterval<TFloatType>();
            
            return;
        }

        /** Generate real pseudo random number in range [a,b]
        * @param a lower bound of random value
        * @param b upper possible bound of random value
        * @return generated number
        */
        double generateReal(double a = 0.0, double b = 1.0);
        
        /** Setup internal seed for random generator
        * @param seed used seed to be setup in generator
        */
        void setSeed(uint32_t seed);

        /** Internal seed for random generator
        * @return seed value
        */
        void setSeeds(uint32_t * seeds, int count);

    private:
        void Mersenne_RandomInit(uint32_t seed);

        void Mersenne_RandomInitByArray(uint32_t seeds[], uint32_t length);
        
        uint32_t Mersenne_BRandom();
        
        double Mersenne_Random();
        
        uint32_t Mersenne_IRandom(uint32_t min_v, uint32_t max_v);

        // or constants for MT19937:
        static const uint32_t MERS_N = 624;
        static const uint32_t MERS_M = 397;
        static const uint32_t MERS_R = 31;
        static const uint32_t MERS_U = 11;
        static const uint32_t MERS_S = 7;
        static const uint32_t MERS_T = 15;
        static const uint32_t MERS_L = 18;
        static const uint32_t MERS_A = 0x9908B0DF;
        static const uint32_t MERS_B = 0x9D2C5680;
        static const uint32_t MERS_C = 0xEFC60000;

        static const uint32_t MERS_LITTLEENDIAN = 1;
        static const uint32_t MERS_BIGENDIAN = 2;
        static const uint32_t MERS_NONIEEE = 3;

        uint32_t mt[MERS_N];             ///< State vector
        uint32_t mti;                    ///< Index into mt
        char Architecture;               ///< Conversion to float depends on computer architecture
    };
}
