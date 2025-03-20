#include "RandomGenXorShiftInteger.h"

namespace burt
{
    RandomGenXorShiftInteger& RandomGenXorShiftInteger::global()
    {
        static RandomGenXorShiftInteger global;
        return global;
    }

    RandomGenXorShiftInteger::RandomGenXorShiftInteger(uint32_t A, uint32_t B, uint64_t initSeed)
        : xCurrent(0), sA(A), sB(B), a(214013), c(2531011)
    {
        xCurrent = initSeed;
        allUint32AreaIsUsed = (A == 0U && B == 4294967295U) ? true : false;
    }

    uint32_t RandomGenXorShiftInteger::last() const
    {
        if (allUint32AreaIsUsed)
        {
            return (uint32_t)xCurrent;
        }
        else
        {
            constexpr uint32_t m_uint32 = 4294967295;
            constexpr double m_inv_double = 1.0 / double(m_uint32);
            return sA + roundToNearestInt<decltype(sA)>(((uint32_t)xCurrent) * m_inv_double * (sB - sA));
        }
    }

    uint64_t RandomGenXorShiftInteger::getSeed() const
    {
        return xCurrent;
    }

    void RandomGenXorShiftInteger::setSeed(uint64_t seed)
    {
        xCurrent = seed;
    }

    uint32_t RandomGenXorShiftInteger::getA() const
    {
        return sA;
    }

    uint32_t RandomGenXorShiftInteger::getB() const
    {
        return sB;
    }
}
