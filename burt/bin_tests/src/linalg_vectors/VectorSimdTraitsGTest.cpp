#include "gtest/gtest.h"

#include <math.h>

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD

#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"

template<class TElementType, bool isSigned>
void executeVectorSimdTraitsGTest()
{
    typedef typename burt::VectorSimdTraits<TElementType, burt::cpu_extension>::VecType VecType;
    constexpr size_t kVecBatchSize = burt::template getVecBatchSize<VecType>();
    constexpr size_t kUnrollFactor = burt::template getUnrollFactor<VecType>();
    size_t kItemSizeInBytes = sizeof(TElementType);

    EXPECT_TRUE(kVecBatchSize >= 1);
    EXPECT_TRUE(kUnrollFactor >= 1);

    VecType a = {};
    VecType b = {};
    std::vector<TElementType> ai;
    std::vector<TElementType> bi;

    ai.resize(kVecBatchSize);
    bi.resize(kVecBatchSize);
    
    for (size_t i = 0; i < kVecBatchSize; ++i)
    {
        ai[i] = TElementType(i);
        bi[i] = TElementType(2 * i + 1);
    }
    
    a.load(ai.data());
    b.load(bi.data());

    {
        VecType c = a + b;
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (ai[i] + bi[i]);
            EXPECT_TRUE( fabs( double(res - c[i]) ) < 1e-6 );
        }
    }

    {
        VecType c = a - b;
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (ai[i] - bi[i]);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    {
        VecType c = a * b;
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (ai[i] * bi[i]);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    //=======================================================================//

    {
        VecType c = a + TElementType(5);
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (ai[i] + 5);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    {
        VecType c = a - TElementType(4);
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (ai[i] - 4);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    {
        VecType c = a * TElementType(7);
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (ai[i] * 7);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    //=======================================================================//

    {
        VecType c = TElementType(5) + b;
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (5 + bi[i]);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    {
        VecType c = TElementType(4) - b;
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (TElementType(4) - bi[i]);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    {
        VecType c = TElementType(7) * b;
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (TElementType(7) * bi[i]);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }
    
    //=======================================================================//

    {
        VecType c = b;
        c += TElementType(10);
        
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (10 + bi[i]);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    {
        VecType c = b;
        c -= TElementType(14);
        
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (bi[i] - 14);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    {
        VecType c = b;
        c *= TElementType(11);
        
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (11 * bi[i]);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    if (isSigned)
    {
        VecType c;
        c = -b;

        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            TElementType res = (-bi[i]);
            EXPECT_TRUE(fabs(double(res - c[i])) < 1e-6);
        }
    }

    {
        TElementType res_sum = TElementType();
        
        for (size_t i = 0; i < kVecBatchSize; ++i)
            res_sum += bi[i];

        TElementType res_sum_simd = horizontal_add(b);
        EXPECT_TRUE(fabs(double(res_sum - res_sum_simd)) < 1e-6);
    }

    // Load-Store test unaligned memory
    {
        VecType a = {};
        std::vector<TElementType> ai;
        std::vector<TElementType> scratch;

        ai.resize(kVecBatchSize);
        scratch.resize(kVecBatchSize);

        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            ai[i] = TElementType(2 * i + 1);
        }
        a.load(ai.data());
        a.store(scratch.data());

        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            EXPECT_TRUE(fabs(double(scratch[i] - ai[i])) < 1e-6);
            EXPECT_TRUE(fabs(double(scratch[i] - a[i])) < 1e-6);
        }
    }

    // Load-Store test aligned memory
    {
        VecType a = VecType();
        constexpr size_t kBufSizeInItems = 128;
        alignas(sizeof(TElementType)* kBufSizeInItems) TElementType ai[kBufSizeInItems] = {};      // 128 elements is enough E.g. AVX-512 with 8bits/item can contain 64 items.
        alignas(sizeof(TElementType)* kBufSizeInItems) TElementType scratch[kBufSizeInItems] = {}; // 128 elements is enough E.g. AVX-512 with 8bits/item can contain 64 items.

        ASSERT_TRUE(sizeof(ai) >= a.size() * sizeof(TElementType));
        ASSERT_TRUE(sizeof(scratch) >= a.size() * sizeof(TElementType));

        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            ai[i] = TElementType(2 * i + 1);
        }
        a.load_a(ai);
        a.store_a(scratch);

        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            EXPECT_TRUE(fabs(double(scratch[i] - ai[i])) < 1e-6);
            EXPECT_TRUE(fabs(double(scratch[i] - a[i])) < 1e-6);
        }
    }
}

template<class TElementType>
void executeVectorSimdTraits4FloatGTest()
{
    typedef typename burt::VectorSimdTraits<TElementType, burt::cpu_extension>::VecType VecType;
    constexpr size_t kVecBatchSize = burt::template getVecBatchSize<VecType>();
    constexpr size_t kUnrollFactor = burt::template getUnrollFactor<VecType>();
    size_t kItemSizeInBytes = sizeof(TElementType);

    EXPECT_TRUE(kVecBatchSize >= 1);
    EXPECT_TRUE(kUnrollFactor >= 1);

    VecType a = {};
    std::vector<TElementType> ai;
 
    ai.resize(kVecBatchSize);
    
    for (size_t i = 0; i < kVecBatchSize; ++i)
        ai[i] = TElementType(2*i+1);

    a.load(ai.data());
    
    VecType aexp = ::exp(a);
    VecType alog = ::log(a);
    VecType aabs = ::abs(a);

    for (size_t i = 0; i < kVecBatchSize; ++i)
    {
        EXPECT_TRUE(fabs(double(aexp[i] - ::exp(a[i]))) < 1e-0);
        EXPECT_TRUE(fabs(double(alog[i] - ::log(a[i]))) < 1e-6);
        EXPECT_TRUE(fabs(double(aabs[i] - ::abs(a[i]))) < 1e-6);
    }

    VecType a_div_3 = a / TElementType(3);
    for (size_t i = 0; i < kVecBatchSize; ++i)
    {
        EXPECT_TRUE(fabs(double(a_div_3[i] - ai[i]/TElementType(3))) < 1e-6);
    }

    VecType a_div_5 = a;
    a_div_5 /= TElementType(5);

    for (size_t i = 0; i < kVecBatchSize; ++i)
    {
        EXPECT_TRUE(fabs(double(a_div_5[i] - ai[i]/TElementType(5))) < 1e-6);
    }

    VecType ten_div_a = TElementType(10) / a;

    for (size_t i = 0; i < kVecBatchSize; ++i)
    {
        EXPECT_TRUE(fabs(double(ten_div_a[i] - TElementType(10)/ai[i])) < 1e-6);
    }
 }

TEST(burt, VectorSimdTraitsGTest)
{
    executeVectorSimdTraitsGTest<double,   true/*isSigned*/>();
    executeVectorSimdTraitsGTest<float,    true/*isSigned*/>();

    executeVectorSimdTraitsGTest<uint64_t,!true/*isSigned*/>();
    executeVectorSimdTraitsGTest<int64_t,  true/*isSigned*/>();

    executeVectorSimdTraitsGTest<uint32_t,!true/*isSigned*/>();
    executeVectorSimdTraitsGTest<int32_t,  true/*isSigned*/>();

    executeVectorSimdTraitsGTest<uint16_t,!true/*isSigned*/>();
    executeVectorSimdTraitsGTest<int16_t,  true/*isSigned*/>();

    executeVectorSimdTraitsGTest<uint8_t,!true/*isSigned*/>();
    executeVectorSimdTraitsGTest<int8_t,   true/*isSigned*/>();

    executeVectorSimdTraits4FloatGTest<double>();
    executeVectorSimdTraits4FloatGTest<float>();
}

TEST(burt, VectorSimdTraitsAndMaskGTest)
{

    typedef burt::VectorSimdTraits<double, burt::cpu_extension>::VecType VecTypeFP64;
    typedef burt::VectorSimdTraits<float, burt::cpu_extension>::VecType VecTypeFP32;
    typedef burt::VectorSimdTraits<uint64_t, burt::cpu_extension>::VecType VecTypeUI64;
    typedef burt::VectorSimdTraits<uint32_t, burt::cpu_extension>::VecType VecTypeUI32;

    if (1)
    {
        typedef double TElementType;
        typedef uint64_t TUIntType;

        VecTypeFP64 a = VecTypeFP64();
        VecTypeFP64 b = VecTypeFP64();
        VecTypeFP64 c = VecTypeFP64();

        std::vector<TElementType> ai;
        std::vector<TElementType> bi;

        ai.resize(a.size());
        bi.resize(b.size());

        for (size_t i = 0; i < a.size(); ++i)
        {
            ai[i] = TElementType(3 * i + 11);
            bi[i] = TElementType(7 * i + 3);
        }

        a.load(ai.data());
        b.load(bi.data());

        c = a & b;

        union ValueIntView
        {
            TElementType value;
            TUIntType intRepr;

            ValueIntView(TElementType theValue) :value(theValue) {}
        };

        for (size_t i = 0; i < a.size(); ++i)
        {
            ValueIntView aCopy(a[i]), bCopy(b[i]), cCopy(c[i]);
            EXPECT_TRUE(cCopy.intRepr == (aCopy.intRepr & bCopy.intRepr));
        }
    }

    if (1)
    {
        typedef float TElementType;
        typedef uint32_t TUIntType;
        VecTypeFP32 a = VecTypeFP32();
        VecTypeFP32 b = VecTypeFP32();
        VecTypeFP32 c = VecTypeFP32();

        std::vector<TElementType> ai;
        std::vector<TElementType> bi;

        ai.resize(a.size());
        bi.resize(b.size());

        for (size_t i = 0; i < a.size(); ++i)
        {
            ai[i] = TElementType(3 * i + 11);
            bi[i] = TElementType(7 * i + 3);
        }

        a.load(ai.data());
        b.load(bi.data());

        c = a & b;

        union ValueIntView
        {
            TElementType value;
            TUIntType intRepr;

            ValueIntView(TElementType theValue) :value(theValue) {}
        };

        for (size_t i = 0; i < a.size(); ++i)
        {
            ValueIntView aCopy(a[i]), bCopy(b[i]), cCopy(c[i]);
            EXPECT_TRUE(cCopy.intRepr == (aCopy.intRepr & bCopy.intRepr));
        }
    }

    if (1)
    {
        typedef uint32_t TElementType;
        typedef uint32_t TUIntType;
        
        VecTypeUI32 a = VecTypeUI32();
        VecTypeUI32 b = VecTypeUI32();
        VecTypeUI32 c = VecTypeUI32();

        std::vector<TElementType> ai;
        std::vector<TElementType> bi;

        ai.resize(a.size());
        bi.resize(b.size());

        for (size_t i = 0; i < a.size(); ++i)
        {
            ai[i] = TElementType(3 * i + 11);
            bi[i] = TElementType(7 * i + 3);
        }

        a.load(ai.data());
        b.load(bi.data());

        c = a & b;

        union ValueIntView
        {
            TElementType value;
            TUIntType intRepr;

            ValueIntView(TElementType theValue) :value(theValue) {}
        };

        for (size_t i = 0; i < a.size(); ++i)
        {
            ValueIntView aCopy(a[i]), bCopy(b[i]), cCopy(c[i]);
            EXPECT_TRUE(cCopy.intRepr == (aCopy.intRepr & bCopy.intRepr));
        }
    }

    if (1)
    {
        typedef uint64_t TElementType;
        typedef uint64_t TUIntType;
        VecTypeUI64 a = VecTypeUI64();
        VecTypeUI64 b = VecTypeUI64();
        VecTypeUI64 c = VecTypeUI64();

        std::vector<TElementType> ai;
        std::vector<TElementType> bi;

        ai.resize(a.size());
        bi.resize(b.size());

        for (size_t i = 0; i < a.size(); ++i)
        {
            ai[i] = TElementType(3 * i + 11);
            bi[i] = TElementType(7 * i + 3);
        }

        a.load(ai.data());
        b.load(bi.data());

        c = a & b;

        union ValueIntView
        {
            TElementType value;
            TUIntType intRepr;

            ValueIntView(TElementType theValue) :value(theValue) {}
        };

        for (size_t i = 0; i < a.size(); ++i)
        {
            ValueIntView aCopy(a[i]), bCopy(b[i]), cCopy(c[i]);
            EXPECT_TRUE(cCopy.intRepr == (aCopy.intRepr & bCopy.intRepr));
        }
    }
}

TEST(burt, VectorSimdTraitsSizeGTest)
{
    typedef burt::VectorSimdTraits<double, burt::cpu_extension>::VecType VecTypeFP64;
    typedef burt::VectorSimdTraits<float, burt::cpu_extension>::VecType VecTypeFP32;
    typedef burt::VectorSimdTraits<uint64_t, burt::cpu_extension>::VecType VecTypeUI64;
    typedef burt::VectorSimdTraits<uint32_t, burt::cpu_extension>::VecType VecTypeUI32;
    
    // Needed for the casting
    static_assert(sizeof(VecTypeFP64) == sizeof(VecTypeUI64));
    static_assert(sizeof(VecTypeFP32) == sizeof(VecTypeUI32));

    size_t kVecSizeFP64 = burt::getVecBatchSize<VecTypeFP64>();
    size_t kVecSizeFP32 = burt::getVecBatchSize<VecTypeFP32>();
    size_t kVecSizeUI64 = burt::getVecBatchSize<VecTypeUI64>();
    size_t kVecSizeUI32 = burt::getVecBatchSize<VecTypeUI32>();
    
    EXPECT_EQ(kVecSizeFP64, kVecSizeUI64);
    EXPECT_EQ(kVecSizeFP32, kVecSizeUI32);

    EXPECT_TRUE(kVecSizeFP32 >= 1);
    EXPECT_TRUE(kVecSizeFP64 >= 1);
    EXPECT_TRUE(kVecSizeUI64 >= 1);
    EXPECT_TRUE(kVecSizeUI32 >= 1);
}

#endif
