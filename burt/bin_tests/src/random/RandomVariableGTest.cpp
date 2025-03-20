#include "burt/random/include/RandomVariable.h"
#include "burt/random/include/MathStatistics.h"

#include "gtest/gtest.h"

#include <math.h>
#include <vector>

namespace
{
    constexpr double kPi = 3.14159265359;
}

TEST(burt, RandomVariableGTest)
{
    using namespace burt;

    // Uniform distribution
    {
        RandomVariable rv;
        std::vector<double> tempVector;
        for (size_t i = 0; i < 10000; ++i)
            tempVector.push_back(rv.generateUniform(2,10));
        double mx_theoretical = (2 + 10)/2.0;
        double dx_theoretical = (10-2)*(10-2)/12.0;
        EXPECT_TRUE(fabs(mx_theoretical -mathstats::mx(&tempVector[0], tempVector.size())) < 0.01);
        EXPECT_TRUE(fabs(dx_theoretical -mathstats::dx(&tempVector[0], tempVector.size())) < 0.1);
    }
    // Exp distribution
    {
        RandomVariable rv;
        std::vector<double> tempVector;
        for (size_t i = 0; i < 10000; ++i)
            tempVector.push_back(rv.generateExp(3));
        double mx_theoretical = 1.0/3.0;
        double dx_theoretical = (1.0/3.0)*(1.0/3.0);
        EXPECT_TRUE(fabs(mx_theoretical - mathstats::mx(&tempVector[0], tempVector.size())) < 0.01);
        EXPECT_TRUE(fabs(dx_theoretical - mathstats::dx(&tempVector[0], tempVector.size())) < 0.1);
    }
    // Relei distribution
    {
        double sigma = 4;
        RandomVariable rv;
        std::vector<double> tempVector;
        for (size_t i = 0; i < 10000; ++i)
            tempVector.push_back(rv.generateRayleigh(sigma));

        double mx_theoretical = sqrt(kPi/2.)*sigma;
        double dx_theoretical = (2 -kPi/2)*sigma*sigma;

        EXPECT_TRUE(fabs(mx_theoretical - mathstats::mx(&tempVector[0], tempVector.size())) < 0.01);
        EXPECT_TRUE(fabs(dx_theoretical - mathstats::dx(&tempVector[0], tempVector.size())) < 0.1);
    }
    // Norm distribution
    {
        RandomVariable rv;
        std::vector<double> tempVector;
        for (size_t i = 0; i < 10000; ++i)
            tempVector.push_back(rv.generateNorm(3,1.5));

        EXPECT_TRUE(fabs(3 - mathstats::mx(&tempVector[0], tempVector.size())) < 0.01);
        EXPECT_TRUE(fabs(1.5*1.5 - mathstats::dx(&tempVector[0], tempVector.size())) < 0.1);
    }

    // HiSquare distribution
    {
        RandomVariable rv;
        std::vector<double> tempVector;
        for (size_t i = 0; i < 20000; ++i)
            tempVector.push_back(rv.generateHiSquare(10));

        EXPECT_TRUE(fabs(10 - mathstats::mx(&tempVector[0], tempVector.size())) < 0.1);
        EXPECT_TRUE(fabs(2*10 - mathstats::dx(&tempVector[0], tempVector.size())) < 0.5);
    }
}
