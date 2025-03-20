#include "burt/random/include/RandomGenXorShiftInteger.h"
#include "burt/random/include/MathStatistics.h"

#include "gtest/gtest.h"

#include <math.h>
#include <vector>

TEST(burt, RandomGenXorShiftGTest)
{
    using namespace burt;

    RandomGenXorShiftInteger gen1, gen2;
    EXPECT_TRUE(gen1.last() == gen2.last());
    EXPECT_TRUE(gen1.generateInteger() == gen2.generateInteger());
    gen1.generateInteger();
    EXPECT_TRUE(gen1.generateInteger() != gen2.generateInteger()) << "Can not be 100% sure. But it should be so.\n";
    EXPECT_TRUE(gen1.getSeed() != gen2.getSeed()) << "Can not be 100% sure. But it should be so.\n";

    gen1.setSeed(gen2.getSeed());
    EXPECT_TRUE(gen1.generateInteger() == gen2.generateInteger());

    {
        RandomGenXorShiftInteger gen3(2, 10);
        std::vector<double> numbers;
        for (int i = 0; i < 50; ++i)
        {
            numbers.push_back(gen3.generateInteger());
            EXPECT_GE(gen3.last(), 2U);
            EXPECT_LE(gen3.last(), 10U);
            EXPECT_TRUE(gen3.last() == gen3.last());
        }
        EXPECT_DOUBLE_EQ(2, mathstats::minElement<double>(&(numbers[0]), numbers.size()));
        EXPECT_DOUBLE_EQ(10, mathstats::maxElement<double>(&(numbers[0]), numbers.size()));
        EXPECT_LE(3, int(mathstats::mx(&(numbers[0]), numbers.size())));
    }

    {
        RandomGenXorShiftInteger gen4(1, 1);
        EXPECT_EQ(1, gen4.generateInteger());
        EXPECT_EQ(1, gen4.generateInteger());
        EXPECT_EQ(1, gen4.generateInteger());
    }

    {
        RandomGenXorShiftInteger gen5(2, 3);
        std::vector<int> numbers;
        for (int i = 0; i < 15; ++i)
        {
            numbers.push_back(gen5.generateInteger());
            EXPECT_GE(gen5.last(), 2U);
            EXPECT_LE(gen5.last(), 3U);
            EXPECT_TRUE(gen5.last() == gen5.last());
        }
        EXPECT_EQ(2, mathstats::minElement<int>(&(numbers[0]), numbers.size()));
        EXPECT_EQ(3, mathstats::maxElement<int>(&(numbers[0]), numbers.size()));
    }

    {
        RandomGenXorShiftInteger gen3(0, 21);
        std::vector<int> numberIsExist;
        numberIsExist.resize(gen3.getB() + size_t(1));

        for (int i = 0; i < 1000; ++i)
        {
            numberIsExist[gen3.generateInteger()]++;
        }

        for (size_t j = 0; j < numberIsExist.size(); ++j)
        {
            EXPECT_GE(numberIsExist[j], 1);
        }
    }

    {
        RandomGenXorShiftInteger rv(0, 20);
        std::vector<double> tempVector;
        for (size_t i = 0; i < 100000; ++i)
            tempVector.push_back(rv.generateInteger());

        double mx_theoretical = (0.0 + 20.0)/2.0;
        EXPECT_TRUE(fabs(mx_theoretical -mathstats::mx(&tempVector[0], tempVector.size()))/mx_theoretical  < 0.05); // allow 5% mistake       

        double dx_theoretical = (20.0-0.0)*(20.0-0.0)/12.0;
        EXPECT_TRUE(fabs(dx_theoretical -mathstats::dx(&tempVector[0], tempVector.size()))/dx_theoretical < 0.15); // allow 15% mistake
    }
    {
        RandomGenXorShiftInteger rv(0, 1);
        double ones = 0;
        double zeros = 0;
        for (size_t i = 0; i < 100000; ++i)
        {
            uint32_t next = rv.generateInteger();
            EXPECT_TRUE(next == 0 || next == 1);
            if (next == 1)
                ones++;
            else
                zeros++;
        }
        EXPECT_TRUE(fabs(ones - zeros)/ones < 0.01);
    }
    {
        RandomGenXorShiftInteger rv1(0,1);
        double ones = 0;
        double zeros = 0;

        for (size_t i = 0; i < 100000; ++i)
        {
            uint32_t next1 = (rv1.generateInteger()); // > 0xFFFFFFFF/2);
            uint32_t next2 = (rv1.generateInteger()); // > 0xFFFFFFFF/2);

            if (next1 == 1 && next2 == 1)
                ones++;
            else if (next1 == 0 && next2 == 0)
                zeros++;
        }
        ones /= 100000.0;
        zeros /= 100000.0;
        EXPECT_TRUE(fabs(double(ones - 1/4.)) < 0.01);
        EXPECT_TRUE(fabs(double(zeros - 1/4.)) < 0.01);
    }
}


TEST(burt, RandomGenXorShiftIntegerCoverageGTest)
{
    using namespace burt;

    std::vector<int> hit_mask;
    constexpr size_t kSize = 100 * 1000;
    hit_mask.resize(kSize, 0);

    constexpr size_t kDraws = 100 * kSize;

    {
        RandomGenXorShiftInteger gen1(0, kSize - 1);
        gen1.setSeed(123);

        for (size_t i = 0; i < kSize; ++i)
            hit_mask[i] = 0;

        for (size_t i = 0; i < kDraws; ++i)
        {
            const uint32_t index = gen1.generateInteger();
            hit_mask[index] += 1;
        }

        for (size_t i = 0; i < kSize; ++i)
        {
            EXPECT_TRUE(hit_mask[i] > 0);
        }
    }
}


TEST(burt, RandomIntegerXorShiftCornerCasesGTest)
{
    using namespace burt;
    {
        RandomGenXorShiftInteger gen1(12, 28);
        int counters[30] = { 0 };

        for (size_t i = 0; i < 1000; ++i)
        {
            auto var = gen1.generateInteger();

            ASSERT_TRUE(var >= 12);
            ASSERT_TRUE(var <= 28);

            counters[var]++;
        }

        for (size_t j = 0; j < sizeof(counters) / sizeof(counters[0]); ++j)
        {
            if (j <= 11 || j >= 29)
                continue;
            EXPECT_TRUE(counters[j] > 0);
        }

        {
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
}


TEST(burt, RandomXorShiftBatchTest)
{
    using namespace burt;
    {
        RandomGenXorShiftInteger gen1;

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
