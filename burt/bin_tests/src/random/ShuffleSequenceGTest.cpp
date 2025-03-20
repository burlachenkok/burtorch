#include "burt/random/include/Shuffle.h"
#include "burt/random/include/MathStatistics.h"
#include "burt/random/include/RandomGenIntegerLinear.h"
#include "burt/random/include/RandomGenXorShiftInteger.h"
#include "burt/random/include/RandomGenMersenne.h"

#include "gtest/gtest.h"

#include <vector>
#include <set>

TEST(burt, ShuffleSequenceGTest)
{
    {
        std::vector<int> elements = { 4, 1, 3, 2, 16, 9, 10, 14, 8, 7 };
        size_t kRawElements = elements.size();

        std::vector<int> elementsCopy0 = elements;
        std::vector<int> elementsCopy1 = elements;
        std::vector<int> elementsCopy2 = elements;
        std::vector<int> elementsCopy3 = elements;
        std::vector<int> elementsCopy4 = elements;

        EXPECT_TRUE(elements == elementsCopy1);
        burt::RandomGenIntegerLinear gen;
        gen.setSeed(12345);

        burt::shuffle(elementsCopy0, 0, gen);
        EXPECT_TRUE(elements == elementsCopy0);
        burt::shuffle(elementsCopy0, 2, gen);
        {
            size_t misplaced = 0;
            for (size_t i = 2; i < kRawElements; ++i)
            {
                if (elementsCopy0[i] != elements[i])
                    misplaced++;
            }
            EXPECT_TRUE(misplaced == 2) << "More likely.But for fixed seed we know that this is a case.";
        }

        gen.setSeed(1234);
        burt::shuffle(elementsCopy1, elementsCopy1.size(), gen);
        burt::shuffle(elementsCopy2, elementsCopy2.size(), gen);
        EXPECT_TRUE(elementsCopy2[0] != 4);
        EXPECT_TRUE(elementsCopy2[elementsCopy2.size() - 1] != 7);

        EXPECT_TRUE(elements != elementsCopy1);
        EXPECT_TRUE(elementsCopy1 != elementsCopy2);
        gen.setSeed(1234);
        burt::shuffle(elementsCopy3, elementsCopy3.size(), gen);
        EXPECT_TRUE(elementsCopy1 == elementsCopy3);

        burt::shuffle(elementsCopy4, gen);
        
        EXPECT_EQ(burt::mathstats::sum(elements.begin(), elements.size()), 
                  burt::mathstats::sum(elementsCopy1.begin(), elementsCopy1.size())
                 );

        EXPECT_EQ(burt::mathstats::sum(elementsCopy3.begin(), elementsCopy3.size()),
                  burt::mathstats::sum(elementsCopy4.begin(), elementsCopy4.size())
                 );
    }
}

TEST(burt, ShuffleSequenceForSubsetGenerationGTest)
{
    std::set<std::pair<int, int>> hitSubsets;    
    std::vector<int> elements = { 0, 1, 2, 3, 4, 5, 6, 7};

    burt::RandomGenXorShiftInteger gen;
    gen.setSeed(874);
    
    std::vector<int> elementsCopy = elements;

    for (size_t i = 0; i < 1500; ++i)
    {
        elementsCopy = elements; // TODO: Figure out for some reasons linear congruet generator does not work well

        burt::shuffle(elementsCopy, 2, gen);

        if (elementsCopy[1] > elementsCopy[0])
        {
            hitSubsets.insert(std::pair<int, int>(elementsCopy[0], elementsCopy[1]));
        }
        else
        {
            hitSubsets.insert(std::pair<int, int>(elementsCopy[1], elementsCopy[0]));
        }
    }

    for (size_t i = 0; i < elements.size(); ++i)
    {
        for (size_t j = i + 1; j < elements.size(); ++j)
        {
            EXPECT_TRUE(hitSubsets.contains(std::pair<int, int>(i, j)));
        }
    }
}

TEST(burt, ShuffleSequenceFullPartialAsShuffleAllGTest)
{
    std::vector<int> elementsA = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::vector<int> elementsB = elementsA;

    {
        burt::RandomGenXorShiftInteger gen;
        gen.setSeed(874);
        burt::shuffle(elementsA, elementsA.size(), gen);

        gen.setSeed(874);
        burt::shuffle(elementsB, gen);
    }

    for (size_t i = 0; i < elementsA.size(); ++i)
    {
        EXPECT_TRUE(elementsA[i] == elementsB[i]);
    }
}

TEST(burt, ShuffleSequenceRandomGenIntegerLinearGPerf)
{
    std::vector<int> elements;
    for (size_t i = 0; i < 50 * 1000; ++i)
        elements.push_back(i);

    {
        burt::RandomGenIntegerLinear gen;
        gen.setSeed(123);
        for (size_t i = 0; i < 10*100; ++i)
        {
            burt::shuffle(elements, elements.size(), gen);
        }
        EXPECT_TRUE(elements[10] != 10);
    }
}

TEST(burt, ShuffleSequenceRandomGenXorShiftIntegerGPerf)
{
    std::vector<int> elements;
    for (size_t i = 0; i < 50 * 1000; ++i)
        elements.push_back(i);

    {
        burt::RandomGenXorShiftInteger gen;
        gen.setSeed(123);

        for (size_t i = 0; i < 100; ++i)
        {
            burt::shuffle(elements, elements.size(), gen);
        }
        EXPECT_TRUE(elements[10] != 10);
    }
}

TEST(burt, ShuffleSequenceRandomGenMersenneGPerf)
{
    std::vector<int> elements;
    for (size_t i = 0; i < 50 * 1000; ++i)
        elements.push_back(i);

    {
        burt::RandomGenMersenne gen;
        gen.setSeed(123);
        for (size_t i = 0; i < 100; ++i)
            burt::shuffle(elements, elements.size(), gen);
        EXPECT_TRUE(elements[10] != 10);
    }
    
}
