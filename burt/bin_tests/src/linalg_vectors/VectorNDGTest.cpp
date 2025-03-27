#include "burt/linalg_vectors/include/VectorND_Raw.h"
#include "burt/linalg_vectors/include/LightVectorND.h"
#include "burt/timers/include/HighPrecisionTimer.h"

#include "gtest/gtest.h"

#include <math.h>
#include <vector>

// Explicit installation for debug that template code is buildable
template class burt::VectorNDRaw<double>;
template class burt::VectorNDRaw<float>;
template class burt::VectorNDRaw<int>;

template< template<class> class Vec>
void TestVector()
{
    {
        Vec<int> vec(20000);
        vec[1] = 12;
        vec[100] = 12;

        Vec<int> other(20000);
        other[122] = 19;
        other[100] = 12;

        EXPECT_TRUE(other == (-1 * -other));
        EXPECT_TRUE(other == +other * 1 / 1);

        EXPECT_EQ(12 * 12, vec & other);

        EXPECT_EQ(12, other[100]);
        EXPECT_EQ(19, other.get(122));
        EXPECT_EQ(0, other.get(111));

        other[10000] = 0;
        EXPECT_EQ(0, other.compress());
        EXPECT_TRUE(1 * vec == vec * 1);
    }

    {
        Vec<int> a(5);
        a[0] = 12;   a[1] = 13; a[2] = 13;

        Vec<int> b(5);
        b[0] = 12;   b[1] = 13; b[2] = 13;

        EXPECT_TRUE(a - b + b == a);
        EXPECT_TRUE(a + b == b + a);
        EXPECT_TRUE((a + b) + b == a + (b + b));
        EXPECT_TRUE(a * 2 == a + a);
        EXPECT_TRUE(a + Vec<int>(a.size()) == a);
    }

    {
        Vec<int> a(3);
        a[0] = 10;   a[1] = 13; a[2] = -2;
        Vec<int> aLower = {0, 0,  0};
        Vec<int> aUpper = {9, 12, 20};

        a.clamp(aLower, aUpper);
        EXPECT_TRUE(a.get(0) == 9);
        EXPECT_TRUE(a.get(1) == 12);
        EXPECT_TRUE(a.get(2) == 0);
        a.clean();
        EXPECT_TRUE(a.maybeNoEmptyIndicies().size() == 0);
        EXPECT_TRUE(a.isNull());
        a[0] = 10;
        a *= 2;
        a = a + a - a;
        a = -a;
        a = +(1 * a * 1);
        EXPECT_TRUE(a[0] == -20);
        EXPECT_TRUE(a.maybeNoEmptyIndicies().size() == 1);
    }

    {
        Vec<int> a = {1,2,3};
        Vec<int> b = {4,5,6,7};
        Vec<int> abConcat = { 1, 2, 3, 4, 5, 6, 7 };

        EXPECT_TRUE(Vec<int>::concat(a, b) == abConcat);
        EXPECT_TRUE(Vec<int>::concat(a, b).get(0, 3) == a);
        EXPECT_TRUE(Vec<int>::concat(a, b).get(3, 4) == b);
    }

    for (size_t sz = 1; sz < 18; ++sz)
    {
        Vec<double> a(sz);
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = 2.0;

        Vec<double> b(sz);
        for (size_t i = 0; i < b.size(); ++i)
            b[i] = 3.0;

        Vec<double> c(sz);
        for (size_t i = 0; i < b.size(); ++i)
            c[i] = a[i] + b[i];

        EXPECT_TRUE(a + b == b + a);
        EXPECT_TRUE(a + b == c);
        EXPECT_TRUE(1.0*a + 1.0*b == 1.0*c);
        b.setAll(0.0);
        EXPECT_TRUE(a + b == a);
        EXPECT_FALSE(b == a);

        Vec<double> d(sz);
        double accumSquares = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            a[i] = i*i + 1.0;
            accumSquares += (i*i + 1.0)*(i*i + 1.0);
        }

        EXPECT_TRUE((d - d.inv().inv()).vectorL2NormSquare() <= 1e-6);
        EXPECT_TRUE((d - d.invSquare().inv().sqrt()).vectorL2NormSquare() <= 1e-6);

        EXPECT_DOUBLE_EQ(a.vectorL2NormSquare(), accumSquares);

        {
            Vec<double> b = a;
            EXPECT_TRUE(b == a);
            Vec<double> mia = -a;
            EXPECT_TRUE(mia == -a);
            EXPECT_TRUE(mia != a);

            EXPECT_TRUE(a == (a - a + a));
            EXPECT_TRUE(a == (2 * a) / 2);
            EXPECT_TRUE(b == a);
            EXPECT_TRUE(b == (2*a - a));
            EXPECT_TRUE(b == (a - a + 2 * a - a));


            EXPECT_TRUE(b == (a-a+2*a-a));

            b += a;
            b -= a;
            EXPECT_TRUE(b == a);
            b += b;
            EXPECT_TRUE(b == 2*a);
        }
        {
            Vec<double> b(sz);
            for (size_t i = 0; i < b.size(); ++i)
            {
                b[i] = 7.0;
            }
            Vec<double> bOrig(b);

            b /= 7.0;
            double dotProductOfEye =  b & b;
            EXPECT_DOUBLE_EQ(dotProductOfEye, double(a.size()));
            b *= 7;
            EXPECT_TRUE(bOrig == b);

            b = -b;
            EXPECT_FALSE(bOrig == b);
            b = +b;
            b = -b;
            EXPECT_TRUE(bOrig == b);
        }

        {
            Vec<double> b(sz);
            Vec<double> c(sz);

            for (size_t i = 0; i < b.size(); ++i)
            {
                b[i] = 7.0*i + 0.5;
                c[i] = 2.0*i + 4;
            }
            Vec<double> expVec = b.exp();
            Vec<double> logVec = b.log();
            for (size_t i = 0; i < b.size(); ++i)
            {
                EXPECT_DOUBLE_EQ(::exp(b[i]), expVec[i]);
                EXPECT_DOUBLE_EQ(::log(b[i]), logVec[i]);
            }

            Vec<double> empty;

            EXPECT_TRUE(Vec<double>::concat(b, b).size() == b.size() * 2);
            EXPECT_TRUE(Vec<double>::concat(b, empty) == Vec<double>::concat(empty, b));
            EXPECT_TRUE(Vec<double>::concat(b, empty) == b);
            EXPECT_TRUE(Vec<double>::concat(b, c) != Vec<double>::concat(c, b));

            for (size_t i = 0; i < b.size(); ++i)
                EXPECT_DOUBLE_EQ(Vec<double>::concat(b, c)[i], b[i]);
            for (size_t i = 0; i < c.size(); ++i)
                EXPECT_DOUBLE_EQ(Vec<double>::concat(b, c)[b.size() + i], c[i]);
        }

        {
            Vec<double> b(sz);
            Vec<double> abs_vec(sz);
            Vec<double> exp_vec(sz);
            Vec<double> neg_vec(sz);
            Vec<double> three_b(sz);

            for (size_t i = 0; i < b.size(); ++i)
            {
                abs_vec[i] = (7.0*i + 0.5);
                if (i % 2 == 0)
                    b[i] = abs_vec[i];
                else
                    b[i] = -abs_vec[i];

                exp_vec[i] = exp(b[i]);
                neg_vec[i] = -(b[i]);
                three_b[i] = 3.0 * b[i];
            }

            EXPECT_TRUE(abs_vec == b.abs());
            EXPECT_TRUE(exp_vec == b.exp());
            EXPECT_TRUE(neg_vec == -b);
            EXPECT_TRUE(three_b == 3.0 * b);
        }

        {
            Vec<double> a = {3.0, 2.0, 3.0};
            Vec<double> b = {1.0, 8.0, 2.0, 3.0};

            EXPECT_DOUBLE_EQ(Vec<double>::reducedDotProduct(a, b, 0, 1), 3.0);
            EXPECT_DOUBLE_EQ(Vec<double>::reducedDotProduct(a, b, 0, 2), 3.0 + 16.0);
            EXPECT_DOUBLE_EQ(Vec<double>::reducedDotProduct(a, b, 0, 3), 3.0 + 16.0 + 6.0);
        }

        {
            Vec<float> a = { 3.0f, 2.0f, 3.0f };
            Vec<float> b = { 1.0f, 8.0f, 2.0f, 3.0f };

            EXPECT_FLOAT_EQ(Vec<float>::reducedDotProduct(a, b, 0, 1), 3.0f);
            EXPECT_FLOAT_EQ(Vec<float>::reducedDotProduct(a, b, 0, 2), 3.0f + 16.0f);
            EXPECT_FLOAT_EQ(Vec<float>::reducedDotProduct(a, b, 0, 3), 3.0f + 16.0f + 6.0f);
        }
        {
            Vec<float> a = { 3.0f, 2.0f, 3.0f };
            Vec<float> b = { 3.0f, 2.0f, 3.0f };
            a.zeroOutItems(1e-1f);
            EXPECT_TRUE(a == b);
            a.zeroOutItems(2.5f);

            Vec<float> c = { 3.0f, 0.0f, 3.0f };
            EXPECT_TRUE(a == c);
            a.zeroOutItems(3.5f);
            EXPECT_FLOAT_EQ(a.vectorL2NormSquare(), 0.0f);
        }

        {
            Vec<float> a = { 3.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            Vec<float> b = { 1.0f, 8.0f, 2.0f, 3.0f, 15.0f };
            auto c1 = a - b;
            auto c2 = a;
            float l2NormOfDifference = 0.0;
            c2.computeDiffAndComputeL2Norm(b, l2NormOfDifference);

            float diff_1 = fabs(l2NormOfDifference - c1.vectorL2Norm());
            EXPECT_TRUE(diff_1 < 1e-9);

            float diff_2 = (c2 - c1).vectorL2Norm();
            EXPECT_TRUE(diff_2 < 1e-9);
        }

        {
            Vec<double> a = { 3.0, 2.0, 3.0, 4.0, 5.0 };
            Vec<double> b = { 1.0, 8.0, 2.0, 3.0, 15.0 };
            auto c1 = a - b;
            auto c2 = a;
            double l2NormOfDifference = 0.0;
            c2.computeDiffAndComputeL2Norm(b, l2NormOfDifference);

            double diff_1 = fabs(l2NormOfDifference - c1.vectorL2Norm());
            EXPECT_TRUE(diff_1 < 1e-9);

            double diff_2 = (c2 - c1).vectorL2Norm();
            EXPECT_TRUE(diff_2 < 1e-9);
        }

        {
            Vec<float> a = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
            Vec<float> b = { 1.0f, 8.0f, 2.0f, 3.0f, 15.0f };
            EXPECT_TRUE(((((2*a) - b) * 0.5) - Vec<float>::scaledDifferenceWithEye(2.0f, b, 0.5f)).vectorL2Norm() < 1e-6f);
            EXPECT_TRUE(((((-3 * a) - b) * 0.5) - Vec<float>::scaledDifferenceWithEye(-3.0f, b, 0.5f)).vectorL2Norm() < 1e-6f);

        }

        {
            Vec<double> a = { 1.0, 1.0, 1.0, 1.0, 1.0 };
            Vec<double> b = { 1.0, 8.0, 2.0, 3.0, 15.0 };
            EXPECT_TRUE(((((-3* a) - b) * 0.5) - Vec<double>::scaledDifferenceWithEye(-3.0, b, 0.5)).vectorL2Norm() < 1e-6);
        }
    }
}

TEST(burt, VectorNDGTest)
{
    TestVector<burt::VectorNDRaw>();
}

template <class TVec>
void TestVectorMinMaxOps()
{
    {
        constexpr size_t kTestVecSize = 1000;
        TVec myVec = TVec::template sequence<1>(kTestVecSize);
        EXPECT_TRUE(myVec.size() == kTestVecSize);
        EXPECT_TRUE(myVec.minItem() == myVec[0]);
        EXPECT_TRUE(myVec.maxItem() == myVec[kTestVecSize - 1]);
    }
    {
        constexpr size_t kTestVecSize = 1000;
        TVec myVec = TVec::template sequence<1>(kTestVecSize);
        myVec.set(kTestVecSize / 2, myVec[kTestVecSize / 2] * typename TVec::TElementType(1000));
        myVec.set(kTestVecSize / 2 - 1, myVec[kTestVecSize / 2 - 1] * typename TVec::TElementType(-1000));
        
        EXPECT_TRUE(myVec.size() == kTestVecSize);
        EXPECT_TRUE(myVec.minItem() == myVec[kTestVecSize / 2 - 1]);
        EXPECT_TRUE(myVec.maxItem() == myVec[kTestVecSize / 2]);
    }
}

TEST(burt, VectorNDMinMaxGTest)
{
    TestVectorMinMaxOps<burt::VectorNDRaw<double>>();
    TestVectorMinMaxOps<burt::VectorNDRaw<float>>();
    TestVectorMinMaxOps<burt::VectorNDRaw<int>>();
}

template<class Vec>
void TestVectorIndexing()
{
    auto v1 = Vec::template sequence<0>(5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(v1[i] == i);
    }
    EXPECT_TRUE(v1.size() == 5);

    auto v2 = Vec::template sequence<0>(13);
    for (int i = 0; i < 13; ++i) {
        EXPECT_TRUE(v2[i] == i);
    }
    EXPECT_TRUE(v2.size() == 13);

    auto v3 = Vec::template sequence<3>(1025);
    for (int i = 0; i < 1025; ++i) {
        EXPECT_TRUE(v3[i] == 3 + i);
    }
    EXPECT_TRUE(v3.size() == 1025);

    auto v4 = Vec::template sequence<12>(128);
    for (int i = 0; i < 128; ++i) {
        EXPECT_TRUE(v4[i] == i + 12);
    }
    EXPECT_TRUE(v4.size() == 128);
}


template<class Vec>
void TestVectorIndexingSmall()
{
    auto v1 = Vec::template sequence<0>(5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(v1[i] == i);
    }
    EXPECT_TRUE(v1.size() == 5);

    auto v2 = Vec::template sequence<0>(13);
    for (int i = 0; i < 13; ++i) {
        EXPECT_TRUE(v2[i] == i);
    }
    EXPECT_TRUE(v2.size() == 13);

    auto v4 = Vec::template sequence<12>(64);
    for (int i = 0; i < 64; ++i) {
        EXPECT_TRUE(v4[i] == i + 12);
    }
    EXPECT_TRUE(v4.size() == 64);
}

TEST(burt, VectorNDIndexingOpsGTest)
{
    TestVectorIndexingSmall< burt::VectorNDRaw<int8_t>>();
    TestVectorIndexingSmall< burt::VectorNDRaw<uint8_t>>();
    TestVectorIndexing< burt::VectorNDRaw<int16_t>>();
    TestVectorIndexing< burt::VectorNDRaw<uint16_t>>();
    TestVectorIndexing< burt::VectorNDRaw<int32_t>>();
    TestVectorIndexing< burt::VectorNDRaw<uint32_t>>();
    TestVectorIndexing< burt::VectorNDRaw<int64_t>>();
    TestVectorIndexing< burt::VectorNDRaw<uint64_t>>();
}
    
template <class VecType>
void executeVectorSigmoidTest()
{
    double min = -50.0;
    double max = +50.0;
    double dh = 0.0001;
    
    double tol = 0.05;
    
    size_t dim = size_t((max - min) / dh);
    VecType input(dim);
    std::vector<typename VecType::TElementType> scratch_buffer(dim);    
    for (size_t i = 0; i < dim; ++i)
    {
        scratch_buffer[i] = min + i * dh;
        //input.set(i, min + i * dh);
    }
    EXPECT_TRUE(scratch_buffer.size() == dim);
    input.load(scratch_buffer.data(), scratch_buffer.size());

    
    burt::HighPrecisionTimer timer;
    timer.reset();
    
    VecType out_sigmoid_a = input.elementwiseSigmoid();
    std::cout << "  Construct original sigmoid: " << timer.getTimeMs() << " ms\n";
    
    out_sigmoid_a.store(scratch_buffer.data(), scratch_buffer.size());
    
    for (size_t j = 0; j < dim; ++j)
    {
        double x = min + j * dh;
        double y = 1.0 / (1.0 + std::exp(-x));
        //EXPECT_TRUE(std::abs(out_sigmoid_a[j] - y) < tol * y);
        EXPECT_TRUE(std::abs(scratch_buffer[j] - y) < tol * y);
    }

    timer.reset();
    VecType out_sigmoid_approx = input.elementwiseSigmoidApproximate();
    std::cout << "  Construct approximate sigmoid: " << timer.getTimeMs() << " ms\n";

    double maxDescr = -10.0;
    double xwithMaxDescr = 0.0;
    out_sigmoid_approx.store(scratch_buffer.data(), scratch_buffer.size());
    
    for (size_t j = 0; j < dim; ++j)
    {
        double x = min + j * dh;
        double y = 1.0 / (1.0 + std::exp(-x));
        //double descr = (std::abs(out_sigmoid_approx[j] - y));
        double descr = (std::abs(scratch_buffer[j] - y));

        if (descr > maxDescr)
        {
            maxDescr = descr;
            xwithMaxDescr = x;
        }
    }
    
    std::cout << " Maximum discrepancy with real sigmoid at [" << min << ", " << max << "] = " << maxDescr << " at x = " << xwithMaxDescr << '\n';
    EXPECT_TRUE(maxDescr < 0.25);
}

TEST(burt, VectorSigmoidTest)
{
    executeVectorSigmoidTest<burt::VectorNDRaw<double>>();
    executeVectorSigmoidTest<burt::VectorNDRaw<float>>();
}

