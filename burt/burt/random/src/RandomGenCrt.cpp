#include "RandomGenCrt.h"

namespace burt
{
    uint64_t RandomGenCrt::seed = 0;

    RandomGenCrt& RandomGenCrt::global() {
        static RandomGenCrt global;
        return global;
    }

    RandomGenCrt::RandomGenCrt(double A, double B)
    : genlast(0.0), sA(A), sB(B)
    {
        generateReal();
    }

    double RandomGenCrt::generateReal() 
    {
        const double val = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        genlast = sA + val * (sB - sA);

        return genlast;
    }

    double RandomGenCrt::last() const
    {
        return genlast;
    }

    uint64_t RandomGenCrt::getSeed() 
    {
        return seed;
    }

    void RandomGenCrt::setSeed(uint64_t theSeed)
    {
        seed = theSeed;
        srand(static_cast<unsigned int>(seed));
    }

    double RandomGenCrt::getA() const {
        return sA;
    }

    double RandomGenCrt::getB() const {
        return sB;
    }
}
