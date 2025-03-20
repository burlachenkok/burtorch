/** @file
* C++ cross-platform extension of math vector, elements of which are stored in std::vector.
* If there are some problem with compilation for Visual Studio specify Properties->Project->Code Generation->Enable Enhancement Instruction Set
*/

#pragma once

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits || SUPPORT_CPU_CPP_TS_V2_SIMD

#include "burt/linalg_vectors/include/VectorND_Raw.h"
#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include "burt/mathroutines/include/SimpleMathRoutines.h"

#include <limits>

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <limits>

namespace burt
{
    template <>
    inline VectorNDRaw<double>  VectorNDRaw<double>::operator + (const VectorNDRaw<double>& rhs) const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        VectorNDRaw<TElementType> res(sz);

        TElementType* restrict_ext resRaw = res.data();
        const TElementType* thisData = this->dataConst();
        const TElementType* rhsData  = rhs.dataConst();

        VecType avec[kUnrollFactor] = {};
        VecType bvec[kUnrollFactor] = {};
        VecType cvec[kUnrollFactor] = {};

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + i + k * kVecBatchSize);
                bvec[k].load_a(rhsData + i + k * kVecBatchSize);
                cvec[k] = avec[k] + bvec[k];
                cvec[k].store_a(resRaw + i + k * kVecBatchSize);
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec, cvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec.load_a(rhsData + items);
                cvec = avec + bvec;
                cvec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            cvec = avec + bvec;
            cvec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
            resRaw[i] = thisData[i] + rhsData[i];
#endif

        return res;
    }

    template <>
    inline VectorNDRaw<double> VectorNDRaw<double>::operator - (const VectorNDRaw<double>& rhs) const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        VectorNDRaw<TElementType> res(sz);

        TElementType* restrict_ext resRaw = res.data();
        const TElementType* thisData = this->dataConst();
        const TElementType* rhsData = rhs.dataConst();

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

        size_t i = 0;

        {
            VecType avec[kUnrollFactor] = {};
            VecType bvec[kUnrollFactor] = {};
            VecType cvec[kUnrollFactor] = {};

            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load_a(thisData + i + k * kVecBatchSize);
                    bvec[k].load_a(rhsData + i + k * kVecBatchSize);
                    cvec[k] = avec[k] - bvec[k];
                    cvec[k].store_a(resRaw + i + k * kVecBatchSize);
                }
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec, cvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec.load_a(rhsData + items);
                
                cvec = avec - bvec;
                cvec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            cvec = avec - bvec;
            cvec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
            resRaw[i] = thisData[i] - rhsData[i];
#endif

        return res;
    }

    template <>
    inline VectorNDRaw<double>  VectorNDRaw<double>::scaledDifferenceWithEye(double a, const VectorNDRaw<double>& rhs, double multiplier)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = rhs.size();
        VectorNDRaw<TElementType> res(sz);

        TElementType* restrict_ext resRaw = res.data();
        const TElementType* restrict_ext rhsData = rhs.dataConst();

        VecType bvec[kUnrollFactor] = {};
        VecType cvec[kUnrollFactor] = {};
        VecType a_vec(a);
        VecType multiplier_vec(multiplier);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                bvec[k].load_a(rhsData + i + k * kVecBatchSize);
                cvec[k] = (a_vec - bvec[k]) * multiplier_vec;
                cvec[k].store_a(resRaw + i + k * kVecBatchSize);
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType bvec, cvec;

            while (resLen > kVecBatchSize)
            {
                bvec.load_a(rhsData + items);
                cvec = (a_vec - bvec) * multiplier_vec;
                cvec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            bvec.load_partial(int(resLen), rhsData + items);
            cvec = (a_vec - bvec) * multiplier_vec;
            cvec.store_partial(int(resLen), resRaw + items);

        }
#else
        for (; i < sz; ++i)
            resRaw[i] = (a - rhsData[i]) * multiplier;
#endif

        return res;
    }

    template <>
    inline VectorNDRaw<double>&  VectorNDRaw<double>::setAll(double value)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        // constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize>(sz);
        TElementType* restrict_ext resRaw = data();

        size_t i = 0;
        VecType svec(value);

        for (; i < items; i += kVecBatchSize)
            svec.store_a(resRaw + i);

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            svec.store_partial(int(resLen), resRaw + i);
        }
#else
        for (; i < sz; ++i)
            set(i, value);
#endif

        return  *this;
    }

    template <>
    inline VectorNDRaw<double>& VectorNDRaw<double>::setAllToDefault()
    {
        size_t lenInBytes = sizeInBytes();
        double* restrict_ext rawItems = data();

        if (lenInBytes > 0) {
            ::memset(rawItems, 0, lenInBytes);
        }
        return *this;
    }

    template <>
    inline double VectorNDRaw<double>::operator & (const VectorNDRaw<double>&rhs) const
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
        TElementType resFinal = TElementType();

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
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec.load_a(rhsData + items);
                resRest += ::horizontal_add(avec * bvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            resRest += ::horizontal_add(avec * bvec);
        }
#else
        for (; i < sz; ++i)
            resRest += thisData[i] * rhsData[i];
#endif
        
        return resFinal + resRest;
    }

    template <>
    inline double VectorNDRaw<double>::reducedDotProduct(const VectorNDRaw<double>& a,
                                                         const VectorNDRaw<double>& b,
                                                         size_t start,
                                                         size_t sz)
    {
        assert(start < a.size());
        assert(start < b.size());
        assert(start + sz <= a.size());
        assert(start + sz <= b.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        const TElementType* aData = a.dataConst();
        const TElementType* bData = b.dataConst();

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
                avec[k].load(aData + (i + k * kVecBatchSize + start));
                bvec[k].load(bData + (i + k * kVecBatchSize + start));

#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                cvec[k] = ::mul_add(avec[k], bvec[k], cvec[k]);
#else
                cvec[k] += avec[k] * bvec[k];
#endif                
            }
        }

        TElementType res[kUnrollFactor] = {};
        TElementType resFinal = TElementType();

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
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load(aData + items + start);
                bvec.load(bData + items + start);

                resRest += ::horizontal_add(avec * bvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), aData + items + start);
            bvec.load_partial(int(resLen), bData + items + start);

            resRest += ::horizontal_add(avec * bvec);
        }
#else
        for (; i < sz; ++i)
            resRest += aData[i + start] * bData[i + start];
#endif

        return resFinal + resRest;
    }

    template <>
    inline VectorNDRaw<double> VectorNDRaw<double>::operator - () const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        VectorNDRaw<TElementType> res(sz);

        TElementType* restrict_ext resRaw = res.data();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        
        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k] = -avec[k];
                bvec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                avec = -avec;
                avec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            avec = -avec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
            resRaw[i] = -thisData[i];
#endif
        return res;
    }

    template <>
    inline VectorNDRaw<double>& VectorNDRaw<double>::operator += (const VectorNDRaw<double>& v)
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
                avec[k].load_a(resRaw + (i + k * kVecBatchSize));
                bvec[k].load_a(vData + (i + k * kVecBatchSize));
                avec[k] += bvec[k];
                avec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }        

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;


            while (resLen > kVecBatchSize)
            {
                avec.load_a(resRaw + items);
                bvec.load_a(vData + items);
                avec += bvec;
                avec.store_a(resRaw + items);

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
    inline void VectorNDRaw<double>::addInPlaceVectorWithMultiple(double multiple, const VectorNDRaw<double>& other)
    {
        assert(size() == other.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = other.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType multiplier_vec(multiple);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(resRaw + (i + k * kVecBatchSize));
                bvec[k].load_a(vData + (i + k * kVecBatchSize));
                
#if SUPPORT_CPU_FMA_EXT
                // (1st * 2nd) + 3rd
                avec[k] = ::mul_add(bvec[k], multiplier_vec, avec[k]);
#else
                avec[k] += (bvec[k] * multiplier_vec);
#endif                
                avec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(resRaw + items);
                bvec.load_a(vData + items);

                avec += (bvec * multiplier_vec);

                avec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec += (bvec * multiplier_vec);
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] += (vData[i] * multiple);
        }
#endif

        return;
    }

    template <>
    inline void VectorNDRaw<double>::subInPlaceVectorWithMultiple(double multiple, const VectorNDRaw<double>& other)
    {
        assert(size() == other.size());

        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* resRaw = this->data();
        const TElementType* vData = other.dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType multiplier_vec(multiple);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(resRaw + (i + k * kVecBatchSize));
                bvec[k].load_a(vData + (i + k * kVecBatchSize));
                
#if SUPPORT_CPU_FMA_EXT
                // -(1st * 2nd) + 3rd
                avec[k] = ::nmul_add(bvec[k], multiplier_vec, avec[k]);
#else
                avec[k] -= (bvec[k] * multiplier_vec);
#endif     

                avec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;

            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(resRaw + items);
                bvec.load_a(vData + items);

                avec -= (bvec * multiplier_vec);

                avec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec -= (bvec * multiplier_vec);
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] -= (vData[i] * multiple);
        }
#endif

        return;
    }

    template <>
    inline VectorNDRaw<double>& VectorNDRaw<double>::operator -= (const VectorNDRaw<double>& v)
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
                avec[k].load_a(resRaw + (i + k * kVecBatchSize));
                bvec[k].load_a(vData + (i + k * kVecBatchSize));
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
                avec.load_a(resRaw + items);
                bvec.load_a(vData + items);
                avec -= bvec;
                avec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), resRaw + items);
            bvec.load_partial(int(resLen), vData + items);
            avec -= bvec;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] -= vData[i];
        }
#endif

        return *this;
    }

    template <>
    inline double VectorNDRaw<double>::vectorL2NormSquare() const
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
        TElementType resFinal = TElementType();

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
            const TElementType& value = thisData[i];
            resRest += value * value;
        }
#endif

        return resFinal + resRest;
    }

    template<>
    inline VectorNDRaw<double>& VectorNDRaw<double>::operator *= (double factor)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* restrict_ext thisData = this->data();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType vecFactor(factor);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;
        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                avec[k] *= vecFactor;
                avec[k].store_a(thisData + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                avec *= vecFactor;
                avec.store_a(thisData + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            avec *= vecFactor;
            avec.store_partial(int(resLen), thisData + items);
        }
#else
        for (; i < sz; ++i)
            thisData[i] *= factor;
#endif

        return *this;
    }

    template<>
    inline VectorNDRaw<double>& VectorNDRaw<double>::operator /= (double factor)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        TElementType* restrict_ext thisData = this->data();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType vecFactor(TElementType(1) / factor);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);
        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                avec[k] *= vecFactor;
                avec[k].store_a(thisData + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                avec *= vecFactor;
                avec.store_a(thisData + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            avec *= vecFactor;
            avec.store_partial(int(resLen), thisData + items);
        }
#else
        for (; i < sz; ++i)
            thisData[i] /= factor;
#endif

        return *this;
    }

    template<>
    inline VectorNDRaw<double> VectorNDRaw<double>::operator * (double factor) const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        VectorNDRaw<TElementType> resVector(sz);

        TElementType* restrict_ext resRaw = resVector.data();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType vecFactor(factor);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);
        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                avec[k] *= vecFactor;
                avec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                avec *= vecFactor;
                avec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            avec *= vecFactor;
            avec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
            resRaw[i] = thisData[i] * factor;
#endif

        return resVector;
    }

    template<>
    inline VectorNDRaw<double> VectorNDRaw<double>::exp() const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        VectorNDRaw<TElementType> resVector(sz);
        TElementType* restrict_ext resRaw = resVector.data();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k] = ::exp(avec[k]);
                bvec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec = ::exp(avec);
                bvec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec = ::exp(avec);
            bvec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
            resRaw[i] = ::exp(thisData[i]);
#endif
        

        return resVector;
    }

    template<>
    inline VectorNDRaw<double> VectorNDRaw<double>::square()
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        VectorNDRaw<TElementType> resVector(sz);

        TElementType* restrict_ext resRaw = resVector.data();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k] = ::square(avec[k]);
                bvec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec = ::square(avec);
                bvec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec = ::square(avec);
            bvec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
            resRaw[i] = thisData[i] * thisData[i];
#endif

        return resVector;
    }

    template<>
    inline VectorNDRaw<double> VectorNDRaw<double>::log()
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        VectorNDRaw<TElementType> resVector(sz);

        TElementType* restrict_ext resRaw = resVector.data();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k] = ::log(avec[k]);
                bvec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

        {
            VecType avec = {};
            VecType bvec = {};

            while (i + kVecBatchSize < sz)
            {
                avec.load_a(thisData + items);
                bvec = ::log(avec);
                bvec.store_a(resRaw + items);

                i += kVecBatchSize;
            }
        }

        for (; i < sz; ++i)
            resRaw[i] = ::log(thisData[i]);

        return resVector;
    }

    template <>
    inline VectorNDRaw<double> VectorNDRaw<double>::concat(const VectorNDRaw<double>& a, const VectorNDRaw<double>& b)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();
        //---------------------------------------------------------------//
        const size_t asize = a.size();
        const size_t bsize = b.size();
        const size_t ab_size = asize + bsize;

        VectorNDRaw<TElementType> res(ab_size);

        TElementType* restrict_ext resRaw = res.data();
        const TElementType* aData = a.dataConst();
        const TElementType* bData = b.dataConst();

        // copy a part
        {
            size_t items = burt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(asize);

            size_t i = 0;
            VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load_a(aData + (i + k * kVecBatchSize));
                    avec[k].store_a(resRaw + (i + k * kVecBatchSize));
                }
            }

            for (; i < asize; ++i)
                resRaw[i] = aData[i];
        }

        // copy b part
        {
            size_t items = burt::roundToNearestMultipleDown<kVecBatchSize* kUnrollFactor>(bsize);
            const size_t i_offset = asize;
            size_t i = 0;
            VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

            for (; i < items; i += kVecBatchSize * kUnrollFactor)
            {
                for (size_t k = 0; k < kUnrollFactor; ++k)
                {
                    avec[k].load_a(bData + (i + k * kVecBatchSize));

                    // "i_offset" -- may lead to misalignment
                    avec[k].store(resRaw + (i_offset + i + k * kVecBatchSize));
                }
            }

            for (; i < bsize; ++i)
                resRaw[i_offset + i] = bData[i];

            assert(asize + bsize == i_offset + i);
        }

        return res;
    }


    template <>
    inline double VectorNDRaw<double>::maxItem() const
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
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
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
    inline double VectorNDRaw<double>::minItem() const
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
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
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

    template <>
    template <class TAccumulator>
    inline TAccumulator VectorNDRaw<double>::sum() const
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
                avec.load_a(thisData + items);
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
    template <class TAccumulator>
    inline TAccumulator VectorNDRaw<double>::logisticUnweightedLossFromMargin(const VectorNDRaw& margin)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = margin.size();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);
        size_t i = 0;

        const TElementType* restrict_ext thisData = margin.dataConst();

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                cvec[k] += ::log(1.0 + ::exp(-avec[k]));
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

        TAccumulator resRest = TAccumulator();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, cvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                cvec = ::log(1.0 + ::exp(-avec));
                resRest += ::horizontal_add(cvec);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            cvec = ::log(1.0 + ::exp(-avec));
            cvec.cutoff(int(resLen));
            resRest += ::horizontal_add(cvec);
        }
#else
        for (; i < sz; ++i) {
            resRest += ::log(1.0 + ::exp(-(thisData[i])));
        }
#endif

        return (resFinal + resRest) / TAccumulator(sz);
    }

    template <>
    template <class TAccumulator>
    inline TAccumulator VectorNDRaw<double>::logisticUnweightedLossFromMarginSigmoid(const VectorNDRaw& classificationMarginSigmoid)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = classificationMarginSigmoid.size();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType cvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        for (size_t k = 0; k < kUnrollFactor; ++k)
            cvec[k] = VecType(0);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize*kUnrollFactor>(sz);
        size_t i = 0;

        const TElementType* restrict_ext classificationMarginSigmoidData = classificationMarginSigmoid.dataConst();

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(classificationMarginSigmoidData + i + k * kVecBatchSize);
                cvec[k] -= ::log(avec[k]);
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

        TAccumulator resRest = TAccumulator();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, cvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(classificationMarginSigmoidData + items);            
                cvec = ::log(avec);

                resRest += -(::horizontal_add(cvec));

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }


            avec.load_partial(int(resLen), classificationMarginSigmoidData + items);            
            cvec = ::log(avec);

            resRest += -(::horizontal_add(cvec));
        }
#else
        for (; i < sz; ++i) {
            resRest -= ::log(classificationMarginSigmoidData[i]);
        }
#endif

        return (resFinal + resRest) / TAccumulator(sz);
    }

    template <>
    inline VectorNDRaw<double> VectorNDRaw<double>::abs() const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();
        VectorNDRaw<TElementType> resVector(sz);

        TElementType* restrict_ext resRaw = resVector.data();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k] = ::abs(avec[k]);
                bvec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec = ::abs(avec);
                bvec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec = ::abs(avec);
            bvec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] = fabs(thisData[i]);
        }
#endif
        
        return resVector;
    }

    template <>
    inline VectorNDRaw<double> VectorNDRaw<double>::elementwiseSigmoid() const
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        VectorNDRaw<TElementType> resVector(sz);

        TElementType* restrict_ext resRaw = resVector.data();
        const TElementType* restrict_ext thisData = this->dataConst();

        VecType avec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized
        VecType bvec[kUnrollFactor] = {}; // default ctor -- value is created, but not initialized

        VecType one_vec(1);

        size_t items = burt::roundToNearestMultipleDown<kVecBatchSize * kUnrollFactor>(sz);

        size_t i = 0;

        for (; i < items; i += kVecBatchSize * kUnrollFactor)
        {
            for (size_t k = 0; k < kUnrollFactor; ++k)
            {
                avec[k].load_a(thisData + (i + k * kVecBatchSize));
                bvec[k] = one_vec / (one_vec + ::exp(-avec[k]));
                bvec[k].store_a(resRaw + (i + k * kVecBatchSize));
            }
        }

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec = one_vec / (one_vec + ::exp(-avec));
                bvec.store_a(resRaw + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec = one_vec / (one_vec + ::exp(-avec));
            bvec.store_partial(int(resLen), resRaw + items);
        }
#else
        for (; i < sz; ++i)
        {
            resRaw[i] = TElementType(1) / (TElementType(1) + ::exp(-thisData[i]));
        }
#endif

        return resVector;
    }

    template <>
    inline void VectorNDRaw<double>::computeDiffAndComputeL2Norm(const VectorNDRaw<double>& rhs, double& restrict_ext l2NormOfDifference)
    {
        typedef burt::VectorSimdTraits<TElementType, cpu_extension>::VecType VecType;
        constexpr size_t kVecBatchSize = burt::getVecBatchSize<VecType>();
        constexpr size_t kUnrollFactor = burt::getUnrollFactor<VecType>();

        size_t sz = size();

        assert(sz == rhs.size());

        TElementType* thisData = this->data();
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
                avec[k].load_a(thisData + i + k * kVecBatchSize);
                bvec[k].load_a(rhsData + i + k * kVecBatchSize);
                avec[k] -= bvec[k]; // avec[k]  = avec[k] - bvec[k]

                cvec[k] += ::square(avec[k]);
                avec[k].store_a(thisData + i + k * kVecBatchSize);
            }
        }

        TElementType l2NormOfDifferenceForPacked = TElementType();
        TElementType res[kUnrollFactor] = {};

        {
            for (size_t k = 0; k < kUnrollFactor; ++k) {
                res[k] = ::horizontal_add(cvec[k]);
            }

            for (size_t k = 0; k < kUnrollFactor; ++k) {
                l2NormOfDifferenceForPacked += res[k];
            }
        }

        TElementType l2NormOfDifferenceForRest = TElementType();

#if SUPPORT_CPU_LOAD_STORE_PART
        {
            size_t resLen = sz - items;
            VecType avec, bvec;

            while (resLen > kVecBatchSize)
            {
                avec.load_a(thisData + items);
                bvec.load_a(rhsData + items);
                avec -= bvec;

                l2NormOfDifferenceForRest += ::horizontal_add(::square(avec));
                avec.store_a(thisData + items);

                items += kVecBatchSize;
                resLen -= kVecBatchSize;
            }

            avec.load_partial(int(resLen), thisData + items);
            bvec.load_partial(int(resLen), rhsData + items);
            avec -= bvec;
            l2NormOfDifferenceForRest += ::horizontal_add(::square(avec));
            avec.store_partial(int(resLen), thisData + items);
        }
#else
        for (; i < sz; ++i)
        {
            thisData[i] -= rhsData[i];
            l2NormOfDifferenceForRest += (thisData[i]) * (thisData[i]);
        }
#endif
        l2NormOfDifference = ::sqrt(l2NormOfDifferenceForRest + l2NormOfDifferenceForPacked);
    }
}

#endif
