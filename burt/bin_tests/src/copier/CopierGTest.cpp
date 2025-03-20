#include "burt/copylocal/include/Copier.h"
#include "gtest/gtest.h"

#include <vector>
#include <string>
#include <iterator>
#include <stddef.h>

namespace
{
    template <class T>
    struct Scalar
    {
        virtual ~Scalar() = default;

        Scalar(const T& x = 0)
            : value(x)
        {}

        virtual void f() {}

        bool operator == (const Scalar& rhs) const
        {
            return value == rhs.value;
        }

        T value;
    };
}

TEST(burt, CopierGTest)
{
    using burt::CopyHelpers;

    {
        int a = 1;
        int b = 2;
        EXPECT_EQ(1, a);
        EXPECT_EQ(2, b);
        burt::CopyHelpers::swap<int>(&a, &b);
        EXPECT_EQ(1, b);
        EXPECT_EQ(2, a);
    }
    {
        bool a = true;
        bool b = false;
        CopyHelpers::swap(&a, &b);
        EXPECT_EQ(true, b);
        EXPECT_EQ(false, a);

        int arr[30] = {};
        EXPECT_EQ(30, sizeof(arr) / sizeof(arr[0]));
        EXPECT_EQ(30, CopyHelpers::arrayLength(arr, arr + 30));
    }
    {
        int a[] = { 1,2,3 };
        int b[] = { 0,0,0 };
        CopyHelpers::copy(b, a, 3);
        EXPECT_EQ(1, b[0]);
        EXPECT_EQ(2, b[1]);
        EXPECT_EQ(3, b[2]);

        CopyHelpers::move(b, a, 3);
        EXPECT_EQ(1, b[0]);
        EXPECT_EQ(2, b[1]);
        EXPECT_EQ(3, b[2]);
    }
    {
        int a[] = { 1, 2, 3, -4 };
        int b[] = { 0, 0, 0,  0 };
        int c[] = { 0, 0, 0,  0 };

        CopyHelpers::copy(b, a, 4);
        CopyHelpers::copy(c, a, 4);

        for (size_t i = 0; i < 4; ++i)
            EXPECT_EQ(a[i], b[i]);

        for (size_t i = 0; i < 4; ++i)
            EXPECT_EQ(a[i], c[i]);

        CopyHelpers::move(b, b, 4);
        for (size_t i = 0; i < 4; ++i)
            EXPECT_EQ(a[i], b[i]);

        // overwrite case
        CopyHelpers::move(b + 1, b, 4 - 2);
        EXPECT_EQ(1, b[0]);
        EXPECT_EQ(1, b[1]);
        EXPECT_EQ(2, b[2]);
        EXPECT_EQ(-4, b[3]);
    }
    {
        std::vector<std::string> a;
        std::vector<std::string> b;
        b.push_back("123");
        b.push_back("456");

        std::back_insert_iterator< std::vector<std::string> > backPusher(a);
        CopyHelpers::copyWithIterators(backPusher, b.begin(), b.end());
        EXPECT_EQ(2, a.size());
        EXPECT_STREQ(a[0].c_str(), "123");
    }
    {
        typedef  Scalar<int> ScalarInt;
        ScalarInt a[4] = { ScalarInt(1), ScalarInt(2), ScalarInt(3), ScalarInt(4) };
        CopyHelpers::move(a, a + 1, 2);
        EXPECT_EQ(ScalarInt(2), a[0]);
        EXPECT_EQ(ScalarInt(3), a[1]);
        EXPECT_EQ(ScalarInt(3), a[2]);
        EXPECT_EQ(ScalarInt(4), a[3]);
    }
    {
        typedef  Scalar<int> ScalarInt;
        ScalarInt a[] = { ScalarInt(1), ScalarInt(2), ScalarInt(3), ScalarInt(4), ScalarInt(5) };
        CopyHelpers::move(a + 1, a + 0, 2);
        ScalarInt expected[] = { ScalarInt(1), ScalarInt(1), ScalarInt(2), ScalarInt(4), ScalarInt(5) };
        for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i)
            EXPECT_EQ(expected[i], a[i]);
    }
    {
        typedef  Scalar<int> ScalarInt;
        ScalarInt a[] = { ScalarInt(1), ScalarInt(2), ScalarInt(3), ScalarInt(4), ScalarInt(5), ScalarInt(6) };
        CopyHelpers::move(a + 2, a + 0, 3);
        ScalarInt expected[] = { ScalarInt(1), ScalarInt(2), ScalarInt(1), ScalarInt(2), ScalarInt(3), ScalarInt(6) };
        for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i)
            EXPECT_EQ(expected[i], a[i]);
    }
}
