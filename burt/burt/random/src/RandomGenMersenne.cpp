#include "RandomGenMersenne.h"
#include <stdint.h>

namespace burt
{
    RandomGenMersenne& RandomGenMersenne::global()
    {
        static RandomGenMersenne global;
        return global;
    }
    
    /************************** MERSENNE.CPP ******************** AgF 2001-10-18 *
    *  Random Number generator 'Mersenne Twister'                                *
    *                                                                            *
    *  This random number generator is described in the article by               *
    *  M. Matsumoto & T. Nishimura, in:                                          *
    *  ACM Transactions on Modeling and Computer Simulation,                     *
    *  vol. 8, no. 1, 1998, pp. 3-30.                                            *
    *  Details on the initialization scheme can be found at                      *
    *  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html                  *
    *                                                                            *
    *  Experts consider this an excellent random number generator.               *
    *                                                                            *
    *  2001 - 2004 A. Fog.                                                       *
    *****************************************************************************/

    /*****************************************************************************
    * Our project do not really rely on the the Mersenne Twister  pseudorandom   *
    * number generator. This implementation can be removed                       * 
    ******************************************************************************/

    void RandomGenMersenne::Mersenne_RandomInit(uint32_t seed)
    {
        // re-seed generator
        mt[0] = seed;
        for (mti = 1; mti < MERS_N; ++mti)
        {
            mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
        }

        // detect computer architecture
        union { double f; uint32_t i[2]; } convert;
        convert.f = 1.0;

        // Note: Old versions of the Gnu g++ compiler may make an error here
        // compile with the option  -fenum-int-equiv  to fix the problem
        if (convert.i[1] == 0x3FF00000)
            Architecture = MERS_LITTLEENDIAN;
        else if (convert.i[0] == 0x3FF00000)
            Architecture = MERS_BIGENDIAN;
        else
            Architecture = MERS_NONIEEE;
    }


    void RandomGenMersenne::Mersenne_RandomInitByArray(uint32_t seeds[], uint32_t length)
    {
        // seed by more than 32 bits
        uint32_t i, j, k;
        Mersenne_RandomInit(19650218UL);
        if (length <= 0)
            return;

        i = 1;  j = 0;
        k = (MERS_N > length ? MERS_N : length);

        for (; k; k--)
        {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL)) + seeds[j] + j;
            i++; j++;
            if (i >= MERS_N)
            {
                mt[0] = mt[MERS_N - 1]; i = 1;
            }

            if (j >= length)
                j = 0;
        }

        for (k = MERS_N - 1; k; k--)
        {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941UL)) - i;
            if (++i >= MERS_N)
            {
                mt[0] = mt[MERS_N - 1]; i = 1;
            }
        }
        mt[0] = 0x80000000UL;
    } // MSB is 1; assuring non-zero initial array


    uint32_t RandomGenMersenne::Mersenne_BRandom()
    {
        // generate 32 random bits
        uint32_t y;
        if (mti >= MERS_N)
        {
            // generate MERS_N words at one time
            const uint32_t LOWER_MASK = (1LU << MERS_R) - 1;   // lower MERS_R bits
            const uint32_t UPPER_MASK = 0xFFFFFFFF << MERS_R; // upper (32 - MERS_R) bits
            static const uint32_t mag01[2] = { 0, MERS_A };

            int kk;
            for (kk = 0; kk < MERS_N - MERS_M; kk++)
            {
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + MERS_M] ^ (y >> 1) ^ mag01[y & 1];
            }

            for (; kk < MERS_N - 1; kk++)
            {
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + (MERS_M - MERS_N)] ^ (y >> 1) ^ mag01[y & 1];
            }

            y = (mt[MERS_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[MERS_N - 1] = mt[MERS_M - 1] ^ (y >> 1) ^ mag01[y & 1];
            mti = 0;
        }

        y = mt[mti++];

        // Tempering (May be omitted):
        y ^= y >> MERS_U;
        y ^= (y << MERS_S) & MERS_B;
        y ^= (y << MERS_T) & MERS_C;
        y ^= y >> MERS_L;

        return y;
    }

    double RandomGenMersenne::Mersenne_Random()
    {
        // output random float number in the interval 0 <= x < 1
        union { double f; uint32_t i[2]; } convert;
        uint32_t r = Mersenne_BRandom(); // get 32 random bits
        // The fastest way to convert random bits to floating point is as follows:
        // Set the binary exponent of a floating point number to 1+bias and set
        // the mantissa to random bits. This will give a random number in the
        // interval [1,2). Then subtract 1.0 to get a random number in the interval
        // [0,1). This procedure requires that we know how floating point numbers
        // are stored. The storing method is tested in function RandomInit and saved
        // in the variable Architecture. The following switch statement can be
        // omitted if the architecture is known. (A PC running Windows or Linux uses
        // LITTLEENDIAN architecture):
        switch (Architecture)
        {
        case MERS_LITTLEENDIAN:
            convert.i[0] = r << 20;
            convert.i[1] = (r >> 12) | 0x3FF00000;
            return convert.f - 1.0;
        case MERS_BIGENDIAN:
            convert.i[1] = r << 20;
            convert.i[0] = (r >> 12) | 0x3FF00000;
            return convert.f - 1.0;
        case MERS_NONIEEE:
        default:
            ;
        }

        // This somewhat slower method works for all architectures, including
        // non-IEEE floating point representation:
        return (double)r * (1. / ((double)(uint32_t)(-1L) + 1.));
    }


    uint32_t RandomGenMersenne::Mersenne_IRandom(uint32_t min_v, uint32_t max_v)
    {
        // output random integer in the interval min <= x <= max
        uint32_t r;
        r = (uint32_t)((max_v - min_v + 1) * Mersenne_Random()) + min_v; // multiply interval with random and truncate
        if (r > max_v)
            r = max_v;
        if (max_v < min_v)
            return 0x80000000;

        return r;
    }

    RandomGenMersenne::RandomGenMersenne(int initSeed)
    {
        Mersenne_RandomInit(initSeed);
        mti = 0;
        Architecture = 0;
    }
    
    double RandomGenMersenne::generateReal(double a, double b)
    {
        const double res = Mersenne_Random() * (b - a) + a;
        return res;
    }
    
    uint32_t RandomGenMersenne::getA() const
    {
        constexpr uint32_t a = 0U;
        return a;        
    }

    uint32_t RandomGenMersenne::getB() const
    {
        constexpr uint32_t b = 4294967294U;
        return b;
    }

    uint32_t RandomGenMersenne::generateInteger()
    {
        constexpr uint32_t a = 0U;
        constexpr uint32_t b = 4294967294U;
        
        return Mersenne_IRandom(a, b);
    }

    uint32_t RandomGenMersenne::generateInteger(uint32_t a, uint32_t b)
    {
        return Mersenne_IRandom(a, b);
    }

    void RandomGenMersenne::setSeed(uint32_t seed)
    {
        Mersenne_RandomInit(seed);
    }
    void RandomGenMersenne::setSeeds(uint32_t * seeds, int count)
    {
        Mersenne_RandomInitByArray(seeds, count);
    }
}
