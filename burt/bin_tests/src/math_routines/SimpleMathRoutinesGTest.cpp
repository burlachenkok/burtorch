#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include "burt/random/include/RandomGenRealLinear.h"

#include "burt/linalg_vectors/include/VectorND_Raw.h"

#include "burt/random/include/RandomGenIntegerLinear.h"

#include "gtest/gtest.h"

#include <stddef.h>
#include <vector>
#include <algorithm>
#include <set>



struct Pair2TestStable
{
    Pair2TestStable(int theX = 0, int theY = 0)
    : x(theX)
    , y(theY)
    {}

    bool operator < (const Pair2TestStable& rhs)
    {
        return x < rhs.x;
    }

    operator int() const
    {
        return x;
    }

    int x;
    int y;
};

static_assert(std::is_integral<uint8_t>::value && std::is_integral<uint8_t>::value && std::is_integral<uint8_t>::value);
static_assert(std::is_integral<int8_t>::value && std::is_integral<int8_t>::value && std::is_integral<int8_t>::value);
static_assert(std::is_integral<size_t>::value);
static_assert(!std::is_integral<float>::value && !std::is_integral<double>::value );


TEST(burt, SanityCheckMinimumGTest)
{
    {
        uint32_t a = 1, b = 33;
        EXPECT_TRUE(burt::minimum(a, b) == 1);
        EXPECT_TRUE(burt::maximum(a, b) == 33);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }

    {
        int32_t a = 1, b = -33;
        EXPECT_TRUE(burt::minimum(a, b) == -33);
        EXPECT_TRUE(burt::maximum(a, b) == 1);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }
    
    {
        uint16_t a = 1, b = 133;
        EXPECT_TRUE(burt::minimum(a, b) == 1);
        EXPECT_TRUE(burt::maximum(a, b) == 133);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }

    {
        int16_t a = -100, b = 313;
        EXPECT_TRUE(burt::minimum(a, b) == -100);
        EXPECT_TRUE(burt::maximum(a, b) == 313);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }

    {
        int64_t a = 1, b = 33;
        EXPECT_TRUE(burt::minimum(a, b) == 1);
        EXPECT_TRUE(burt::maximum(a, b) == 33);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimumForIntegrals(b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximumForIntegrals(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }

    {
        float a = 1.0f, b = -33.0f;
        EXPECT_TRUE(burt::minimum(a, b) == -33.0f);
        EXPECT_TRUE(burt::maximum(a, b) == 1.0f);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimum(b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }

    {
        double a = -1.0, b = 33.0;
        EXPECT_TRUE(burt::minimum(a, b) == -1.0);
        EXPECT_TRUE(burt::maximum(a, b) == 33.0);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimum(b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }
    
    {
        float a = 1.0f, b = 33.0f;
        EXPECT_TRUE(burt::minimum(a, b) == 1.0f);
        EXPECT_TRUE(burt::maximum(a, b) == 33.0f);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimum < /*promise that number are non-neg*/true> (b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximum < /*promise that number are non-neg*/true> (a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }

    {
        double a = 1.0, b = 33.0;
        EXPECT_TRUE(burt::minimum(a, b) == 1.0);
        EXPECT_TRUE(burt::maximum(a, b) == 33.0);

        EXPECT_TRUE(burt::minimum(b, a) == burt::minimum(a, b));
        EXPECT_TRUE(burt::minimum(a, b) == burt::noBracnhMinimum < /*promise that number are non-neg*/true> (b, a));
        EXPECT_TRUE(burt::minimum(a, b) == burt::bracnhMinimum(b, a));

        EXPECT_TRUE(burt::maximum(b, a) == burt::maximum(a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::noBracnhMaximum < /*promise that number are non-neg*/true> (a, b));
        EXPECT_TRUE(burt::maximum(b, a) == burt::bracnhMaximum(a, b));
    }
    
    {
        uint32_t a = 11, b = 3;
        EXPECT_TRUE(burt::add_two_numbers_modN(a, b, uint32_t(13)) == (11 + 3) % 13);
        EXPECT_TRUE(burt::add_two_numbers_modN(a, b, uint32_t(12)) == (11 + 3) % 12);
        EXPECT_TRUE(burt::add_two_numbers_modN(a, b, uint32_t(14)) == 0);
        EXPECT_TRUE(burt::add_two_numbers_modN(a, b, uint32_t(15)) == 14);

        a = 0, b = 0;
        EXPECT_TRUE(burt::add_two_numbers_modN(a, b, uint32_t(12)) == 0);
        a = 0, b = 1;
        EXPECT_TRUE(burt::add_two_numbers_modN(a, b, uint32_t(12)) == 1);
    }
}

TEST(burt, SimpleMathRoutinesRoundingTest)
{

    {
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(128), 7);
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(120), 6);
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(119), 6);
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(64), 6);
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(32), 5);
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(16), 4);
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(8),  3);
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(4),  2);
        EXPECT_EQ(burt::log2AtCompileTimeLowerBound(2),  1);


    }
    {
        size_t r1 = burt::roundToNearestMultipleUp<4>(10);
        EXPECT_EQ(r1, 12);

        size_t r1d = burt::roundToNearestMultipleDown<4>(10);
        EXPECT_EQ(r1d, 8);

        size_t r2 = burt::roundToNearestMultipleUp<6>(12);
        EXPECT_EQ(r2, 12);

        size_t r2d = burt::roundToNearestMultipleDown<6>(12);
        EXPECT_EQ(r2d, 12);

        size_t r3 = burt::roundToNearestMultipleUp<4>(4);
        EXPECT_EQ(r3, 4);

        size_t r3d = burt::roundToNearestMultipleDown<4>(4);
        EXPECT_EQ(r3d, 4);

        for (size_t i = 0; i < 200; ++i)
        {
            size_t rTest = burt::roundToNearestMultipleUp<7>(i);
            EXPECT_TRUE(rTest % 7 == 0);

            if (i % 7 == 0)
            {
                EXPECT_TRUE(rTest == i);
            }

            EXPECT_TRUE(rTest >= i);
        }

        for (size_t i = 0; i < 200; ++i)
        {
            size_t rTest = burt::roundToNearestMultipleDown<7>(i);
            EXPECT_TRUE(rTest % 7 == 0);

            if (i % 7 == 0)
            {
                EXPECT_TRUE(rTest == i);
            }

            EXPECT_TRUE(rTest <= i);
        }
    }
}

TEST(burt, SimpleMathComparisions)
{
    EXPECT_TRUE(burt::isFirstHigherThenSecondIgnoreSign(14, 12));
    EXPECT_FALSE(burt::isFirstHigherThenSecondIgnoreSign(12, 14));

    EXPECT_TRUE(burt::isFirstHigherThenSecondIgnoreSign(14, -12));
    EXPECT_TRUE(burt::isFirstHigherThenSecondIgnoreSign(-14, 12));

    EXPECT_TRUE(burt::isFirstHigherThenSecondIgnoreSign(-14.0f, -12.0f));
    EXPECT_TRUE(burt::isFirstHigherThenSecondIgnoreSign(-14.0, -12.0));

    constexpr double step = 0.1;
    for (double a = -100.0; a < -10.0; a += step)
    {
        for (double b = a + 1.0; b < -10.0; b += step)
        {
            EXPECT_TRUE(burt::isFirstHigherThenSecondIgnoreSign(a, b));
        }
    }
    
}
