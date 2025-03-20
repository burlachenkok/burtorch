#include "burt/linalg_vectors/include/LightVectorND.h"
#include "burt/linalg_vectors/include/VectorND_Raw.h"
#include "burt/system/include/threads/Thread.h"

#include "gtest/gtest.h"

// Explicit installation for debug that template code is buildable
template class burt::LightVectorND<burt::VectorNDRaw_d>;
template class burt::LightVectorND<burt::VectorNDRaw_f>;

template <class TVec, 
          template<class> class TLightVec>
void testLightVectorND4StdGTest()
{
    {
        TVec vec({ -1.0, -2.0, 3.0, -4.0 });
        TLightVec<TVec> lightVec1 = TLightVec<TVec>(vec, 1, 2);
        TLightVec<TVec> lightVec2 = TLightVec<TVec>(vec, 1, 2);
        TLightVec<TVec> lightVec3 = TLightVec<TVec>(vec, 2, 1);
        TLightVec<TVec> lightVec4 = TLightVec<TVec>(vec, 2, 1);

        EXPECT_TRUE(lightVec1 == lightVec2);
        EXPECT_FALSE(lightVec1 != lightVec2);
        EXPECT_TRUE(lightVec1 != lightVec3);

        EXPECT_EQ(2, lightVec1.size());
        EXPECT_EQ(2, lightVec2.size());
        EXPECT_EQ(1, lightVec3.size());

        EXPECT_TRUE(lightVec1 != TVec({ -2.0, 4.0 }));
        EXPECT_TRUE(lightVec1 == TVec({ -2.0, 3.0 }));

        EXPECT_DOUBLE_EQ(lightVec1[0], vec[1]);
        EXPECT_DOUBLE_EQ(lightVec1[1], vec[2]);
        EXPECT_DOUBLE_EQ(vec[2] * vec[2], lightVec3 & lightVec4);

        TLightVec<TVec> lightVec5 = TLightVec<TVec>(vec, 1);
        EXPECT_TRUE(lightVec5.size() == vec.size() - 1);
        EXPECT_DOUBLE_EQ(lightVec5.template sum<typename TVec::TElementType>(), vec.template sum<typename TVec::TElementType>() - vec[0]);
        EXPECT_FALSE(lightVec5.isNull());
        EXPECT_TRUE(&lightVec5.getRaw(2) == &vec.getRaw(3));
        EXPECT_TRUE(&lightVec5.getRaw(2) == &vec.getRaw(3));

        TLightVec<TVec> lightVec6(lightVec5);
        EXPECT_TRUE(lightVec6 == lightVec5);
        EXPECT_TRUE(lightVec6 == TLightVec<TVec>(lightVec6, 0));
        EXPECT_TRUE(lightVec6 != TLightVec<TVec>(lightVec6, 2));
    }
    {
        //         0     1    2     3
        TVec vec({ 1.0, 2.0, 3.0, 4.0 });
        
        TLightVec<TVec> v1_2 = TLightVec<TVec>(vec, 1, 2);
        EXPECT_DOUBLE_EQ(5.0, v1_2.template sum<typename TVec::TElementType>());
        EXPECT_DOUBLE_EQ( (2.0 + 3.0), v1_2.template sum<typename TVec::TElementType>());

        TLightVec<TVec> v1_3a = TLightVec<TVec>(vec, 1);
        TLightVec<TVec> v1_3b = TLightVec<TVec>(vec, 1, 3);
        EXPECT_TRUE(v1_3a == v1_3b);
        EXPECT_DOUBLE_EQ(2.0, v1_3a[0]);
        EXPECT_DOUBLE_EQ(3.0, v1_3a[1]);

        // 3.0 4.0
        TLightVec<TVec> v2_2a = TLightVec<TVec>(vec, 2, 2);
        TLightVec<TVec> v2_2b = TLightVec<TVec>(v1_3a, 1, 2);
        EXPECT_TRUE(v2_2a == v2_2b);

        EXPECT_DOUBLE_EQ(3.0*3.0 + 4.0*4.0, v2_2a & v2_2b);

        EXPECT_DOUBLE_EQ(3.0*3.0 + 0 * 4.0*4.0, TLightVec<TVec>::reducedDotProduct( v2_2a, v2_2b, 0, 1));
        EXPECT_DOUBLE_EQ(3.0*3.0 +  4.0*4.0, TLightVec<TVec>::reducedDotProduct(v2_2a, v2_2b, 0, 2));
        EXPECT_DOUBLE_EQ(0.0, TLightVec<TVec>::reducedDotProduct(v2_2a, v2_2b, 0, 0));

        EXPECT_DOUBLE_EQ(25.0, v2_2b.vectorL2NormSquare());
        EXPECT_TRUE(v2_2b == +v2_2b);
    }

    {
        //         0     1    2     3
        TVec vec({ 1.0, 2.0, 3.0, 4.0 });
        EXPECT_TRUE(vec == TLightVec<TVec>::concat(TLightVec<TVec>(vec, 0, 2),
                                                             TLightVec<TVec>(vec, 2, 2))
                   );
    }

    {
        TVec vec_a({ -1.0, -2.0, 3.0, -4.0 });
        TLightVec<TVec> lightVecA = TLightVec<TVec>(vec_a, 0);

        TVec vec_b({ -2.0, 2.0, -30.0, 5.0 });
        TLightVec<TVec> lightVecB = TLightVec<TVec>(vec_b, 0);

        TVec vec_c = vec_a + vec_b;

        lightVecA.addAnotherVectrorNonBlockingMulthithread(lightVecB);
        
        for (size_t i = 0; i < vec_c.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(vec_c[i], lightVecA[i]);
        }
        //vec_c.dbgPrintInMatlabStyle(std::cout, "vec_c");
        //lightVecA.dbgPrintInMatlabStyle(std::cout, "lightVecA");
    }

    {
        TVec vec_a({ -1.0, -2.0, 3.0, -4.0, 12.0, 13.0, 14.0});
        TVec vec_b(vec_a.size());

        TLightVec<TVec> lightVecA = TLightVec<TVec>(vec_a, 0);
        TLightVec<TVec> lightVecB = TLightVec<TVec>(vec_b, 0);

        using TItem = typename TVec::TElementType;
        
        TVec vec_r = vec_a * TItem(3.0);

        lightVecA.multiplyByScalarNonBlockingMulthithread(TItem(3.0));

        for (size_t i = 0; i < vec_r.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(vec_r[i], lightVecA[i]);
        }
        
        lightVecB.assignWithVectorMultiple(lightVecA, TItem(1.0));
        for (size_t i = 0; i < vec_b.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(vec_b[i], lightVecA[i]);
        }

        lightVecB.assignWithVectorMultiple(lightVecA, TItem(-1.0));
        for (size_t i = 0; i < vec_b.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(vec_b[i], -lightVecA[i]);
        }

        lightVecB.assignWithVectorMultiple(lightVecA, TItem(13.0));
        for (size_t i = 0; i < vec_b.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(vec_b[i], TItem(13.0) * lightVecA[i]);
        }


    }
}

template <class TVec, 
          template<class> class TLightVec>
void testAssignSequence()
{
    {
        TVec vec({ 11, 12, 13, 22, 33, 89 });
        TLightVec<TVec> lightVec1 = TLightVec<TVec>(vec, 0, 4);
        lightVec1.assignIncreasingSequence(100);
        EXPECT_TRUE(vec == TVec({ 100, 101, 102, 103, 33, 89 }));
    }

    {
        TVec vec({ 11, 12, 13, 22, 33, 89 });
        TLightVec<TVec> lightVec1 = TLightVec<TVec>(vec, 1, 4);
        lightVec1.assignIncreasingSequence(100);
        EXPECT_TRUE(vec == TVec({ 11, 100, 101, 102, 103, 89 }));
    }

    {
        TVec vec({ 11, 12, 13, 22, 33, 89, 90, 91, 93, 94, 67, 1, 2, 3, 15, 6, 17});
        TLightVec<TVec> lightVec1 = TLightVec<TVec>(vec, 2, 10);
        lightVec1.assignIncreasingSequence(200);
        EXPECT_TRUE(vec == TVec({ 11, 12, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 2, 3, 15, 6, 17 }));
    }
}

template <class TVec,
    template<class> class TLightVec>
void testMultithreadNonBlockingOps()
{
    size_t th = 48;
    size_t s = 4*1024;

    TVec vec(s);
    for (size_t i = 0; i < s; ++i)
        vec.set(i, i);
    
    TVec vec_desired(s);
    for (size_t i = 0; i < s; ++i)
        vec_desired.set(i, vec[i] + th);
    
    // create 10 threads: each thread goal is to add "1" to vec;
    std::vector<burt::DefaultThread*> threads;
    auto add_one = [](void* arg1, void* arg2) -> int32_t
    {
#if 1
        TVec& vec_destination = *(TVec*)arg1;
        TLightVec<TVec> vec_destination_light = TLightVec<TVec>(vec_destination, 0);

        TVec increaser = TVec(vec_destination.size());
        increaser.setAll(1);
        TLightVec<TVec> increaser_view = TLightVec<TVec>(increaser, 0);

        vec_destination_light.addAnotherVectrorNonBlockingMulthithread(increaser_view);
#endif

        return 0;
    };
        
    for (size_t i = 0; i < th; ++i)
        threads.push_back(new burt::DefaultThread(add_one, &vec));

    for (size_t i = 0; i < th; ++i)
        threads[i]->join();

    for (size_t i = 0; i < th; ++i)
        delete threads[i];

    for (size_t i = 0; i < s; ++i)
    {
        EXPECT_EQ(vec[i], vec_desired[i]);
    }   
}

TEST(burt, LightVectorGTest)
{
    testLightVectorND4StdGTest<burt::VectorNDRaw_d, burt::LightVectorND>();
    testLightVectorND4StdGTest<burt::VectorNDRaw_f, burt::LightVectorND>();

    testAssignSequence<burt::VectorNDRaw_ui, burt::LightVectorND>();
    testAssignSequence<burt::VectorNDRaw_i, burt::LightVectorND>();
    testAssignSequence<burt::VectorNDRaw_ui64, burt::LightVectorND>();
}

TEST(burt, LightVectorGTestMT)
{
    testMultithreadNonBlockingOps<burt::VectorNDRaw_d, burt::LightVectorND>();
    testMultithreadNonBlockingOps<burt::VectorNDRaw_f, burt::LightVectorND>();
}

