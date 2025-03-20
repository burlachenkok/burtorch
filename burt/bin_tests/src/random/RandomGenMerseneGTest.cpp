#include "burt/random/include/RandomGenMersenne.h"
#include "burt/random/include/MathStatistics.h"

#include "gtest/gtest.h"

#include <vector>

#include <math.h>
#include <stdint.h>
#include <memory>

TEST(burt, RandomMerseneGTest)
{
    using namespace burt;

    std::unique_ptr<RandomGenMersenne> gen1 = std::make_unique<RandomGenMersenne>();
    std::unique_ptr<RandomGenMersenne> gen2 = std::make_unique<RandomGenMersenne>();

    EXPECT_TRUE(gen1->generateInteger() == gen2->generateInteger());
    gen1->generateInteger();
    EXPECT_TRUE(gen1->generateInteger() != gen2->generateInteger()) << "Can not be 100% sure. But it should be so.\n";    

    gen1->setSeed(111);
	gen2->setSeed(111);
    EXPECT_TRUE(gen1->generateInteger() == gen2->generateInteger());

    {
        std::unique_ptr<RandomGenMersenne> gen3 = std::make_unique<RandomGenMersenne>();

        std::vector<double> numbers;
        for (int i = 0; i < 50; ++i)
        {
            numbers.push_back(gen3->generateReal(2, 10));
            EXPECT_GE(numbers.back(), 2U);
            EXPECT_LE(numbers.back(), 10U);
        }
        EXPECT_LE(mathstats::minElement<double>(&(numbers[0]), numbers.size()), 2.2);
        EXPECT_GE(mathstats::maxElement<double>(&(numbers[0]), numbers.size()), 9.8);
        EXPECT_LE(3, int(mathstats::mx(&(numbers[0]), numbers.size())));
    }

    {
        std::unique_ptr<RandomGenMersenne> gen4 = std::make_unique<RandomGenMersenne>();

        EXPECT_EQ(1, gen4->generateInteger(1,1));
        EXPECT_EQ(1, gen4->generateInteger(1,1));
        EXPECT_EQ(1, gen4->generateInteger(1,1));
    }

    {
        std::unique_ptr<RandomGenMersenne> gen5 = std::make_unique<RandomGenMersenne>();
        std::vector<int> numbers;
        for (int i = 0; i < 15; ++i)
        {
            numbers.push_back(gen5->generateInteger(2,3));
            EXPECT_GE(numbers.back(), 2);
            EXPECT_LE(numbers.back(), 3);
        }
        EXPECT_EQ(2, mathstats::minElement<int>(&(numbers[0]), numbers.size()));
        EXPECT_EQ(3, mathstats::maxElement<int>(&(numbers[0]), numbers.size()));
    }

    {
        std::unique_ptr<RandomGenMersenne> gen3 = std::make_unique<RandomGenMersenne>();

        std::vector<int> numberIsExist;
        numberIsExist.resize(22);

        for (int i = 0; i < 100; ++i)
        {
            numberIsExist[gen3->generateInteger(0, 21)]++;
        }

        for (size_t j = 0; j < numberIsExist.size(); ++j)
        {
            EXPECT_GE(numberIsExist[j], 1);
        }
    }
    {
        RandomGenMersenne rv;
        std::vector<double> tempVector;
        for (size_t i = 0; i < 100000; ++i)
            tempVector.push_back(rv.generateReal(0.0, 20.0));

        EXPECT_TRUE(fabs(mathstats::maxElement<double>(tempVector.begin(), tempVector.size()) - 20) < 0.001);
        EXPECT_TRUE(fabs(mathstats::minElement<double>(tempVector.begin(), tempVector.size()) - 0) < 0.001);

        // Mersene is amazing
        double mx_theoretical = (0 + 20)/2.0;
        EXPECT_TRUE(fabs(mx_theoretical -mathstats::mx(&tempVector[0], tempVector.size()))/mx_theoretical  < 0.005); // allow 0.5% mistake

        double dx_theoretical = (20.0-0.0)*(20.0-0.0)/12.0;
        EXPECT_TRUE(fabs(dx_theoretical -mathstats::dx(&tempVector[0], tempVector.size()))/dx_theoretical < 0.005); // allow 0.5% mistake
    }

    {
        RandomGenMersenne rv;
        std::vector<double> tempVector;
        for (size_t i = 0; i < 100000; ++i)
            tempVector.push_back(rv.generateInteger(0, 20));

        EXPECT_TRUE(fabs(mathstats::maxElement<double>(tempVector.begin(), tempVector.size()) - 20) < 0.001);
        EXPECT_TRUE(fabs(mathstats::minElement<double>(tempVector.begin(), tempVector.size()) - 0) < 0.001);

        // Mersene is amazing
        double mx_theoretical = (0 + 20)/2.0;
        EXPECT_TRUE(fabs(mx_theoretical - mathstats::mx(&tempVector[0], tempVector.size()))/mx_theoretical  < 0.005); // allow 0.5% mistake

        double dx_theoretical = (20.0-0.0)*(20.0-0.0)/12.0;
        EXPECT_TRUE(fabs(dx_theoretical - mathstats::dx(&tempVector[0], tempVector.size()))/dx_theoretical < 0.2); // allow 20% mistake
    }

    {
        RandomGenMersenne rv;
        double ones = 0;
        double zeros = 0;
        for (size_t i = 0; i < 100000; ++i)
        {
            uint32_t next = rv.generateInteger(0, 1);
            EXPECT_TRUE(next == 0 || next == 1);
            if (next == 1)
                ones++;
            else
                zeros++;
        }
        EXPECT_TRUE(fabs(ones - zeros)/ones < 0.01);
    }
    {
        RandomGenMersenne rv1;
        double ones = 0;
        double zeros = 0;

        for (size_t i = 0; i < 100000; ++i)
        {
            uint32_t next1 = rv1.generateInteger(0,1);
            uint32_t next2 = rv1.generateInteger(0,1);

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


TEST(burt, RandomGenMersenneCoverageGTest)
{
    using namespace burt;

    std::vector<int> hit_mask;
    constexpr size_t kSize = size_t(100) * size_t(1000);
    hit_mask.resize(kSize, 0);

    constexpr size_t kDraws = 100 * kSize;

    {
        RandomGenMersenne gen1;
        gen1.setSeed(123);

        for (size_t i = 0; i < kSize; ++i)
            hit_mask[i] = 0;

        for (size_t i = 0; i < kDraws; ++i)
        {
            const uint32_t index = gen1.generateInteger((uint32_t)0, (uint32_t)(kSize - 1));
            hit_mask[index] += 1;
        }

        for (size_t i = 0; i < kSize; ++i)
        {
            EXPECT_TRUE(hit_mask[i] > 0);
        }
    }
}


TEST(burt, RandomIntegerMerseneCornerCasesGTest)
{
    using namespace burt;
    {
        std::unique_ptr<RandomGenMersenne> gen1 = std::make_unique<RandomGenMersenne>();

        int counters[30] = { 0 };

        for (size_t i = 0; i < 1000; ++i)
        {
            auto var = gen1->generateInteger(12, 28);

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
                double r = gen1->generateRealInUnitInterval();
                if (r > 0.25 && r < 0.75)
                    middle_hit = true;
            }
            EXPECT_TRUE(middle_hit);
        }

    }
}


TEST(burt, RandomMerseneBatchTest)
{
    using namespace burt;
    {
        std::unique_ptr<RandomGenMersenne> gen1 = std::make_unique<RandomGenMersenne>();
        
        constexpr size_t kSize = 1000;
        
        std::vector<double> placeholderA;
        std::vector<double> placeholderB;

        placeholderA.resize(kSize);
        placeholderB.resize(kSize);


        for (size_t i = 1; i < sizeof(placeholderA) / sizeof(placeholderA[0]); ++i)
        {
            gen1->setSeed(123 + i);

            for (size_t j = 0; j < i; ++j)
            {
                placeholderA[j] = gen1->generateRealInUnitInterval();
                placeholderB[j] = -1.0;
            }
            gen1->setSeed(123 + i);
            gen1->generateBatchOfRealsInUnitInterval(placeholderB.data(), i);

            for (size_t j = 0; j < i; ++j)
            {
                EXPECT_TRUE(fabs(placeholderA[j] - placeholderB[j]) < 1e-10);
            }
        }
    }
}
