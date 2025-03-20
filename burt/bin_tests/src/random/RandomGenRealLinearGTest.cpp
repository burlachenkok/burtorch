#include "burt/random/include/RandomGenRealLinear.h"
#include "burt/random/include/MathStatistics.h"

#include "gtest/gtest.h"

#include <math.h>
#include <vector>

TEST(burt, RandomRealLinearGTest)
{
    using namespace burt;

    RandomGenRealLinear gen1, gen2;
    EXPECT_TRUE(gen1.last() == gen2.last());
    EXPECT_TRUE(gen1.generateReal() == gen2.generateReal());
    gen1.generateReal();
    EXPECT_TRUE(gen1.generateReal() != gen2.generateReal()) << "Can not be 100% sure. But it should be so.\n";
    EXPECT_TRUE(gen1.getSeed() != gen2.getSeed()) << "Can not be 100% sure. But it should be so.\n";

    gen1.setSeed(gen2.getSeed());
    EXPECT_TRUE(gen1.generateReal() == gen2.generateReal());

    {
        RandomGenRealLinear gen3(2, 100);
        std::vector<double> numbers;
        for (int i = 0; i < 500; ++i)
        {
            numbers.push_back(gen3.generateReal());
            EXPECT_GE(gen3.last(), 2.);
            EXPECT_LE(gen3.last(), 100.);
            EXPECT_TRUE(gen3.last() == gen3.last());
        }
        EXPECT_LE(mathstats::minElement<double>(&(numbers[0]), numbers.size()), 2.2);
        EXPECT_GE(mathstats::maxElement<double>(&(numbers[0]), numbers.size()), 98.1);
        EXPECT_TRUE(fabs(51 -mathstats::mx(&(numbers[0]), numbers.size())) < 5);
    }

    {
        RandomGenRealLinear gen4(1, 1);
        EXPECT_DOUBLE_EQ(1, gen4.generateReal());
        EXPECT_DOUBLE_EQ(1, gen4.generateReal());
        EXPECT_DOUBLE_EQ(1, gen4.generateReal());
    }

    {
        RandomGenRealLinear gen5(2, 3);
        std::vector<double> numbers;
        for (int i = 0; i < 15; ++i)
        {
            numbers.push_back(gen5.generateReal());
            EXPECT_GE(gen5.last(), 2);
            EXPECT_LE(gen5.last(), 3);
            EXPECT_TRUE(gen5.last() == gen5.last());
        }
        EXPECT_GE(2.1, mathstats::minElement<double>(&(numbers[0]), numbers.size()));
        EXPECT_LE(2.7, mathstats::maxElement<double>(&(numbers[0]), numbers.size()));
    }

    {
        RandomGenRealLinear rv(0.0, 20.0);
        std::vector<double> tempVector;
        for (size_t i = 0; i < 100000; ++i)
            tempVector.push_back(rv.generateReal());

        double mx_theoretical = (0.0 + 20.0)/2.0;
        EXPECT_TRUE(fabs(mx_theoretical -mathstats::mx(&tempVector[0], tempVector.size()))/mx_theoretical  < 0.05); // allow 5% mistake

        double dx_theoretical = (20.0-0.0)*(20.0-0.0)/12.0;
        EXPECT_TRUE(fabs(dx_theoretical -mathstats::dx(&tempVector[0], tempVector.size()))/dx_theoretical < 0.15); // allow 15% mistake
    }

    {
        RandomGenRealLinear rv;
        double ones = 0;
        double zeros = 0;
        for (size_t i = 0; i < 100000; ++i)
        {
            double next = rv.generateReal();
            EXPECT_TRUE(next >= 0.0);
            EXPECT_TRUE(next <= 1.0);

            if (next > 0.5)
                ones++;
            else
                zeros++;
        }
        EXPECT_TRUE(fabs(double(ones - zeros))/ones < 0.01);
    }
    {
        RandomGenRealLinear rv2;
        double ones = 0;
        double zeros = 0;

        for (size_t i = 0; i < 100000; ++i)
        {
            double next1 = rv2.generateReal();
            double next2 = rv2.generateReal();
            if (next1 > 0.5 && next2 > 0.5)
                ones++;
            else if (next1 <= 0.5 && next2 <= 0.5)
                zeros++;
        }
        ones /= 100000.0;
        zeros /= 100000.0;
        EXPECT_TRUE(fabs(ones - 1/4.) < 0.01);
        EXPECT_TRUE(fabs(zeros - 1/4.) < 0.01);
    }

    {
        RandomGenRealLinear gen1;

        bool middle_hit = false;
        for (size_t i = 0; i < 25; ++i)
        {
            double r = gen1.generateRealInUnitInterval();
            if (r > 0.25 && r < 0.75)
                middle_hit = true;
        }
        EXPECT_TRUE(middle_hit);
    }
}

TEST(burt, RandomRealLinearBatchTest)
{
    using namespace burt;
    {
        RandomGenRealLinear gen1;

        double placeholderA[1000] = {};
        double placeholderB[1000] = {};

        for (size_t i = 1; i < sizeof(placeholderA) / sizeof(placeholderA[0]); ++i)
        {
            gen1.setSeed(123 + i);

            for (size_t j = 0; j < i; ++j)
            {
                placeholderA[j] = gen1.generateRealInUnitInterval();
                placeholderB[j] = -1.0;
            }
            gen1.setSeed(123 + i);
            gen1.generateBatchOfRealsInUnitInterval(placeholderB, i);

            for (size_t j = 0; j < i; ++j)
            {
                EXPECT_TRUE(fabs(placeholderA[j] - placeholderB[j]) < 1e-10);
            }
        }
    }
}
