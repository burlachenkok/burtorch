#include "RandomGenRealLinear.h"

namespace burt
{
    double RandomGenRealLinear::inv_m = 1.0 / 4294967295.0;
    uint32_t RandomGenRealLinear::a = 214013;
    uint32_t RandomGenRealLinear::c = 2531011;
    uint32_t RandomGenRealLinear::m = 4294967295;   
        
    RandomGenRealLinear& RandomGenRealLinear::global()
    {
        static RandomGenRealLinear global;
        return global;
    }

    double RandomGenRealLinear::last() const
    {
        return sA + (xCurrent * inv_m) * (sB - sA);
    }

    uint32_t RandomGenRealLinear::getSeed() const
    {
        return xCurrent;
    }

    void RandomGenRealLinear::setSeed(uint32_t seed)
    {
        xCurrent = seed;
    }

    double RandomGenRealLinear::getA() const
    {
        return sA;
    }

    double RandomGenRealLinear::getB() const
    {
        return sB;
    }
}
