#pragma once

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD

#include "burt/linalg_vectors/include/LightVectorND.h"
#include "burt/linalg_vectors/include/VectorND_Raw.h"

#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include <limits>

#include <assert.h>
#include <math.h>
#include <stddef.h>

namespace burt
{
    template <>
    inline LightVectorND<VectorNDRaw_f>& LightVectorND<VectorNDRaw_f>::addInPlaceVectorWithMultiple(float multiple, const LightVectorND& v)
    {
        assert(size() == v.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        
        VecType multiple_vec(multiple);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // 1st * 2nd + 3rd
                avec[k] = ::mul_add(bvec[k], multiple_vec, avec[k]);
#else
                avec[k] += (bvec[k] * multiple_vec);
#endif
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec += (bvec * multiple_vec);
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec += (bvec * multiple_vec);
            avec.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
        {
            resRaw[i] += (vData[i] * multiple);
        }

#endif

        return *this;
    }

    template <>
    inline LightVectorND<VectorNDRaw_f>& LightVectorND<VectorNDRaw_f>::subInPlaceVectorWithMultiple(float multiple, const LightVectorND& v)
    {
        assert(size() == v.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType multiple_vec(multiple);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));
                
#if SUPPORT_CPU_FMA_EXT
                // -(1st * 2nd) + 3rd
                avec[k] = ::nmul_add(bvec[k], multiple_vec, avec[k]);
#else
                avec[k] -= (bvec[k] * multiple_vec);
#endif
                
                
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec -= (bvec * multiple_vec);
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec -= (bvec * multiple_vec);
            avec.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
        {
            resRaw[i] -= (vData[i] * multiple);
        }

#endif

        return *this;
    }


    template <>
    inline float LightVectorND<VectorNDRaw_f>::subInPlaceVectorWithMultipleAndReportL2NormSqr(float multiple, const LightVectorND& v)
    {
        assert(size() == v.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {};        // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {};        // default ctor -- value is created, but not initialized

        VecType res_v_l2sqr[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            res_v_l2sqr[k] = VecType(0);

        VecType multiple_vec(multiple);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // -(1st * 2nd) + 3rd
                avec[k] = ::nmul_add(bvec[k], multiple_vec, avec[k]);
#else
                avec[k] -= (bvec[k] * multiple_vec);
                res_v_l2sqr[k] += ::square(bvec[k]);
#endif


                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec -= (bvec * multiple_vec);
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);

            avec -= (bvec * multiple_vec);
            res_v_l2sqr[0] += ::square(bvec);

            avec.store_partial(int(resLen), resRaw + items);

        }
#else

        TElementType res_v_l2sqr_final = TElementType();

        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res_v_l2sqr_final += ::horizontal_add(res_v_l2sqr[k]);
        }

        for (; i < sz; ++i)
        {
            res_v_l2sqr_final += vData[i];
            resRaw[i] -= (vData[i] * multiple);
        }

#endif

        return res_v_l2sqr_final;
    }
    
    template <>
    inline LightVectorND<VectorNDRaw_f>& LightVectorND<VectorNDRaw_f>::operator += (const LightVectorND<VectorNDRaw_f>& v)
    {
        assert(size() == v.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = v.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k].load(vData + (i + k * kVecBatchSize));
                avec[k] += bvec[k];
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(resRaw + items);
                bvec.load(vData + items);
                avec += bvec;
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec += bvec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else

        for (; i < sz; ++i)
        {
            resRaw[i] += vData[i];
        }
#endif

        return *this;
    }

    template <>
    inline float LightVectorND<VectorNDRaw_f>::operator & (const LightVectorND<VectorNDRaw_f>& rhs) const
    {
        assert(size() == rhs.size());
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        
        const TElementType* thisData = this->dataConst();
        const TElementType* rhsData = rhs.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized        
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                bvec[k].load(rhsData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                cvec[k] = ::mul_add(avec[k], bvec[k], cvec[k]);
#else
                cvec[k] += avec[k] * bvec[k];
#endif
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;

            while (resLen > kVecBatchSize)
            {
                VecType avec, bvec;
                avec.load(thisData + items);
                bvec.load(rhsData + items);
                resRest += ::horizontal_add(avec * bvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            VecType avec, bvec;
            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            resRest += ::horizontal_add(avec * bvec);    
        }
#else

        for (; i < sz; ++i)
        {
            resRest += get(i) * rhs.get(i);
        }
#endif
        
        return resFinal + resRest;
    }

    template <>
    inline float LightVectorND<VectorNDRaw_f>::dotProductForAlignedMemory(const LightVectorND<VectorNDRaw_f>& rhs) const
    {
        assert(size() == rhs.size());
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        const TElementType* thisData = this->dataConst();
        const TElementType* rhsData = rhs.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k].load_a(rhsData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                cvec[k] = ::mul_add(avec[k], bvec[k], cvec[k]);
#else
                cvec[k] += avec[k] * bvec[k];
#endif
            }
        }
        
        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;

            while (resLen > kVecBatchSize)
            {
                VecType avec, bvec;
                avec.load_a(thisData + items);
                bvec.load_a(rhsData + items);
                resRest += ::horizontal_add(avec * bvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            VecType avec, bvec;
            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            resRest += ::horizontal_add(avec * bvec);
        }
#else
        for (; i < sz; ++i)
        {
            resRest += get(i) * rhs.get(i);
        }
#endif
        
        return resFinal + resRest;
    }

    template <>
    inline float LightVectorND<VectorNDRaw_f>::vectorL2NormSquare() const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                cvec[k] += ::square(avec[k]);
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();

        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;
            
            while (resLen > kVecBatchSize)
            {
                avec.load(thisData + items);
                resRest += ::horizontal_add(::square(avec));

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), thisData + items);
            resRest += ::horizontal_add(::square(avec));
        }
#else
        for (; i < sz; ++i)
        {
            const TElementType& value = get(i);
            resRest += value * value;
        }
#endif

        return resFinal + resRest;
    }

    template <>
    inline float LightVectorND<VectorNDRaw_f>::vectorL2NormSquareForAlignedMemory() const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                cvec[k] += ::square(avec[k]);
            }
        }

        TElementType res[kUnrollFactor] = {};
        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            res[k] = ::horizontal_add(cvec[k]);
        }

        TElementType resFinal = TElementType();

        for (size_t k = 0; k < kUnrollFactor; ++k)
        {
            resFinal += res[k];
        }

        TElementType resRest = TElementType();
        
#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                resRest += ::horizontal_add(::square(avec));

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), thisData + items);
            resRest += ::horizontal_add(::square(avec));
        }
#else
        for (; i < sz; ++i)
        {
            const TElementType& value = get(i);
            resRest += value * value;
        }
#endif

        return resFinal + resRest;
    }

    template <>
    inline LightVectorND<VectorNDRaw_f>& LightVectorND<VectorNDRaw_f>::addWithVectorMultiple(const LightVectorND<VectorNDRaw_f>& v,
                                                                                             const TElementType multiplier)
    {
        assert(size() == v.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        const TElementType* restrict_ext inputData = v.dataConst();

        size_t sz = v.size();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType multiple_vec(multiplier);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(inputData + (i + k * kVecBatchSize));

#if SUPPORT_CPU_FMA_EXT
                bvec[k].load(resRaw + (i + k * kVecBatchSize));

                // (1st * 2nd) + 3rd
                bvec[k] = ::mul_add(avec[k], multiple_vec, bvec[k]);
#else
                avec[k] *= multiple_vec;
                bvec[k].load(resRaw + (i + k * kVecBatchSize));
                bvec[k] += avec[k];
#endif
                
                bvec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(inputData + items);
                avec *= multiple_vec;
                
                bvec.load(resRaw + items);
                bvec += avec;
                
                bvec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            
            avec.load_partial(int(resLen), inputData + items);
            avec *= multiple_vec;
            
            bvec.load_partial(int(resLen), resRaw + items);
            bvec += avec;

            bvec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] += inputData[i] * multiplier;
        }
#endif
        
        return *this;
    }

    template<>
    // template <float multiplier>
    inline LightVectorND<VectorNDRaw_f>& LightVectorND<VectorNDRaw_f>::assignWithVectorMultiple(const LightVectorND<VectorNDRaw_f>& v, float multiplier)
    {
        assert(size() == v.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        const TElementType* restrict_ext inputData = v.dataConst();

        size_t sz = v.size();
        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        
        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        if (multiplier == -1.0f)
        {
            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load(inputData + (i + k * kVecBatchSize));
                    avec[k] = -avec[k];
                    avec[k].store(resRaw + (i + k * kVecBatchSize));
                }
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load(inputData + items);
                    avec = -avec;
                    avec.store(resRaw + items);

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }
                
                avec.load_partial(int(resLen), inputData + items);
                avec = -avec;
                avec.store_partial(int(resLen), resRaw + items);
            }
#else
            for (; i < sz; ++i)
            {
                resRaw[i] = -inputData[i];
            }
#endif
            
        }
        else
        {
            VecType multiple_vec(multiplier);

            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load(inputData + (i + k * kVecBatchSize));
                    avec[k] *= multiple_vec;
                    avec[k].store(resRaw + (i + k * kVecBatchSize));
                }
            }

#if SUPPORT_CPU_LOAD_STORE_PART
            {
                size_t resLen = sz - items;
                VecType avec;

                while (resLen > kVecBatchSize)
                {
                    avec.load(inputData + items);
                    avec *= multiple_vec;
                    avec.store(resRaw + items);

                    items += kVecBatchSize;
                    resLen -= kVecBatchSize;
                }

                avec.load_partial(int(resLen), inputData + items);
                avec *= multiple_vec;
                avec.store_partial(int(resLen), resRaw + items);
            }
#else
            for (; i < sz; ++i)
            {
                resRaw[i] = inputData[i] * multiplier;
            }
#endif
            
        }
        
        return *this;
    }

    template<>
    inline LightVectorND<VectorNDRaw_f>& LightVectorND<VectorNDRaw_f>::assignWithVectorDifference(const LightVectorND<VectorNDRaw_f>& a, const LightVectorND<VectorNDRaw_f>& b)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        const TElementType* restrict_ext aInputData = a.dataConst();
        const TElementType* restrict_ext bInputData = b.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(aInputData + (i + k * kVecBatchSize));
                bvec[k].load(bInputData + (i + k * kVecBatchSize));

                avec[k] -= bvec[k];
                avec[k].store(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(aInputData + items);
                bvec.load(bInputData + items);
                avec -= bvec;
                avec.store(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), aInputData + items);
            bvec.load_partial(int(resLen), bInputData + items);
            avec -= bvec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] = aInputData[i] - bInputData[i];
        }
#endif

        return *this;
    }

    template<>
    inline LightVectorND<VectorNDRaw_f>& LightVectorND<VectorNDRaw_f>::assignWithVectorDifferenceAligned(const LightVectorND<VectorNDRaw_f>& a, const LightVectorND<VectorNDRaw_f>& b)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        const TElementType* restrict_ext aInputData = a.dataConst();
        const TElementType* restrict_ext bInputData = b.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(aInputData + (i + k * kVecBatchSize));
                bvec[k].load_a(bInputData + (i + k * kVecBatchSize));

                avec[k] -= bvec[k];
                avec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(aInputData + items);
                bvec.load_a(bInputData + items);
                avec -= bvec;
                avec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }
            

            avec.load_partial(int(resLen), aInputData + items);
            bvec.load_partial(int(resLen), bInputData + items);
            avec -= bvec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] = aInputData[i] - bInputData[i];
        }
#endif

        return *this;
    }
    //========================================================================================================//

    template <>
    template <class TAccumulator>
    inline TAccumulator LightVectorND<VectorNDRaw_f>::sum() const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                cvec[k] += avec[k];
            }
        }

        TAccumulator res[kUnrollFactor] = {};
        TAccumulator resFinal = TAccumulator();

        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                res[k] = ::horizontal_add(cvec[k]);
                resFinal += res[k];
            }
        }

        TElementType resRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load(thisData + items);
                resRest += ::horizontal_add(avec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            resRest += ::horizontal_add(avec);
        }
#else
        for (; i < sz; ++i)
        {
            const TElementType& value = thisData[i];
            resRest += value;
        }
#endif

        return resFinal + resRest;
    }

    template <>
    inline float LightVectorND<VectorNDRaw_f>::maxItem() const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();
        constexpr TElementType defValue4Reduction = -(std::numeric_limits<TElementType>::max());

        size_t sz = size();

        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(defValue4Reduction);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                cvec[k] = ::maximum(cvec[k], avec[k]);
            }
        }

        TElementType res[kUnrollFactor] = {};
        TElementType resFinal = defValue4Reduction;

        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                res[k] = ::horizontal_max(cvec[k]);
                resFinal = maximum(resFinal, res[k]);
            }
        }

        TElementType resRest = defValue4Reduction;

        for (; i < sz; ++i)
        {
            const TElementType& value = thisData[i];
            resRest = maximum(resRest, value);
        }

        return maximum(resFinal, resRest);
    }

    template <>
    inline float LightVectorND<VectorNDRaw_f>::minItem() const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();
        constexpr TElementType defValue4Reduction = +(std::numeric_limits<TElementType>::max());

        size_t sz = size();

        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(defValue4Reduction);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                cvec[k] = ::minimum(cvec[k], avec[k]);
            }
        }

        TElementType res[kUnrollFactor] = {};
        TElementType resFinal = defValue4Reduction;

        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                res[k] = ::horizontal_min(cvec[k]);
                resFinal = minimum(resFinal, res[k]);
            }
        }

        TElementType resRest = defValue4Reduction;

        for (; i < sz; ++i)
        {
            const TElementType& value = thisData[i];
            resRest = minimum(resRest, value);
        }

        return minimum(resFinal, resRest);
    }


    template<>
    inline void LightVectorND<VectorNDRaw_f>::exp_inplace()
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = 1;// burt::getUnrollFactor<VecType>();

        size_t sz = size();
        VectorNDRaw<TElementType> resVector(sz);
        TElementType* restrict_ext thisData = this->data();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load(thisData + (i + k * kVecBatchSize));
                bvec[k] = ::exp(avec[k]);
                bvec[k].store(thisData + (i + k * kVecBatchSize));
            }
        }
        for (; i < sz; ++i)
            thisData[i] = ::exp(thisData[i]);
    }
}

#endif
