#pragma once

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD

#include "burt/linalg_vectors/include/LightVectorND.h"
#include "burt/linalg_vectors/include/VectorND_Raw.h"

#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include "burt/mathroutines/include/SimpleMathRoutines.h"

#include <limits>

#include <assert.h>
#include <math.h>
#include <stddef.h>

namespace burt
{    
    template <>
    inline LightVectorND<VectorNDRaw_i>& LightVectorND<VectorNDRaw_i>::assignIncreasingSequence(int32_t initialValue)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecRegType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecRegType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};
        
        for (size_t i = 0; i < kVecBatchSize; ++i) 
        {
            indicies[i] = ( initialValue + static_cast<TElementType>(i) );
        }
        
        VecRegType add_value(kVecBatchSize);
        VecRegType indicies_reg = VecRegType();
        indicies_reg.load_a(indicies);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize>(sz);
        size_t i = 0;
        
        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i) 
        {
            int32_t add_i = int32_t(i);
            resRaw[i] = (initialValue + add_i);
        }
#endif

        return *this;
    }
    
    template <>
    inline LightVectorND<VectorNDRaw_ui>& LightVectorND<VectorNDRaw_ui>::assignIncreasingSequence(uint32_t initialValue)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecRegType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecRegType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            indicies[i] = (initialValue + static_cast<TElementType>(i));
        }

        VecRegType add_value(kVecBatchSize);
        VecRegType indicies_reg = VecRegType();
        indicies_reg.load_a(indicies);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize>(sz);
        size_t i = 0;

        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i) 
        {
            int32_t add_i = int32_t(i);
            resRaw[i] = (initialValue + add_i);
        }
#endif

        return *this;
    }

    template <>
    inline LightVectorND<VectorNDRaw_i64>& LightVectorND<VectorNDRaw_i64>::assignIncreasingSequence(int64_t initialValue)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecRegType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecRegType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            indicies[i] = (initialValue + static_cast<TElementType>(i));
        }

        VecRegType add_value(kVecBatchSize);
        VecRegType indicies_reg = VecRegType();
        indicies_reg.load_a(indicies);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize>(sz);
        size_t i = 0;

        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store(resRaw + i);
            indicies_reg += add_value;
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i) {
            resRaw[i] = (initialValue + i);
        }
#endif

        return *this;
    }

    template <>
    inline LightVectorND<VectorNDRaw_ui64>& LightVectorND<VectorNDRaw_ui64>::assignIncreasingSequence(uint64_t initialValue)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecRegType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecRegType>();

        TElementType* restrict_ext resRaw = data();
        size_t sz = size();

        alignas(kVecBatchSize * sizeof(TElementType)) TElementType indicies[kVecBatchSize] = {};
        for (size_t i = 0; i < kVecBatchSize; ++i)
        {
            indicies[i] = (initialValue + static_cast<TElementType>(i));
        }

        VecRegType add_value(kVecBatchSize);
        VecRegType indicies_reg = VecRegType();
        indicies_reg.load_a(indicies);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize>(sz);
        size_t i = 0;

        for (; i < items; i += kVecBatchSize)
        {
            indicies_reg.store(resRaw + i);
            indicies_reg += add_value;
        }


#if SUPPORT_CPU_LOAD_STORE_PART
        {
            const size_t resLen = sz - items;
            indicies_reg.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i) {
            resRaw[i] = (initialValue + i);
        }
#endif

        return *this;
    }
    
    //=========================================================================================================================================//
        
    template<>
    inline LightVectorND<VectorNDRaw_i>& LightVectorND<VectorNDRaw_i>::assignWithVectorDifference(const LightVectorND<VectorNDRaw_i>& a, const LightVectorND<VectorNDRaw_i>& b)
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
    inline LightVectorND<VectorNDRaw_i>& LightVectorND<VectorNDRaw_i>::assignWithVectorDifferenceAligned(const LightVectorND<VectorNDRaw_i>& a, const LightVectorND<VectorNDRaw_i>& b)
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
}

#endif
