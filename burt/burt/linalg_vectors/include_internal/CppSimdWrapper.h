#pragma once

#if SUPPORT_CPU_CPP_TS_V2_SIMD

#include <experimental/simd>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <memory.h>

namespace burt
{
    template <class T>
    struct CppSimdWrapper
    {
        using TUsedSimd = ::std::experimental::native_simd<T>;

        TUsedSimd data;

        /** Ctor. Data potentially not initialized
        */
        CppSimdWrapper()
        {
        }

        /** Ctor. All items in data will be initialized with the same value
        * @param value value to broadcast for initialiazation
        */
        CppSimdWrapper(T value) : data(value)
        {
        }

        /** Copy Ctor.
        * @param rhs another vector register wrapper to copy data from
        */
        CppSimdWrapper(const CppSimdWrapper& rhs) : data(rhs.data)
        {}

        /** Move Ctor.
        * @param rhs another vector register wrapper to move data from
        */
        CppSimdWrapper(CppSimdWrapper&& rhs) : data(std::move(rhs.data))
        {}

        /** Assigment copy operator.
        * @param rhs another vector register wrapper to copy from
        */
        CppSimdWrapper& operator = (const CppSimdWrapper& rhs)
        {
            data = rhs.data;
            return *this;
        }

        /** Assigment move operator.
        * @param rhs another vector register wrapper to move from
        */
        CppSimdWrapper& operator = (CppSimdWrapper&& rhs)
        {
            data = std::move(rhs.data);
            return *this;
        }

        /** Dtor
        */
        ~CppSimdWrapper()
        {}

        /** Mostly internal method to create a new wrapper from a simd vector with moving from underying data
        * @param theData data to copy
        */
        static CppSimdWrapper createSimdFromVec(TUsedSimd&& theData)
        {
            CppSimdWrapper res;
            res.data = std::move(theData);
            return res;
        }

        /** Return the i-th element of the simd vector
        * @param i index of element to return
        * @return i-th element of the simd vector
        * @remark A simd is not a container of individual objects and therefore cannot return an lvalue.
        */            
        typename TUsedSimd::value_type operator[] (size_t i) const {
            return data[i];
        }
        
        /* Returns the width (number of elements) in the simd vector
        * @return number of elements
        * @remark We have used consteval modifier. If consteval function cannot be run during compile time, it leads to compile time error.
        */
        static consteval size_t size()
        {
            return TUsedSimd::size();
        }

        /* Replaces all elements of a simd vector with items
        * @param items2read array of items to copy from and memory is aligned
        */
        void load_a(const T* items2read)
        {
            data.copy_from(items2read, ::std::experimental::vector_aligned);
        }

        /* Replaces all elements of a simd vector with items
        * @param items2read array of items to copy from and memory is not aligned
        */
        void load(const T* items2read)
        {
            constexpr size_t sz = size();

#if 0
            // backup in case if compilers support https://en.cppreference.com/w/cpp/experimental/simd/simd/operator_at
            //  in incorrect/restricted way

            alignas(sz * sizeof(T)) TElementType items_aligned[sz] = {};
            for (size_t i = 0; i < sz; ++i)
                items_aligned[i] = items2read[i];
            data.copy_from(items_aligned, ::std::experimental::vector_aligned);
#else
            for (size_t i = 0; i < sz; ++i)
                data[i] = items2read[i];
#endif
        }

        /* Store all elements of a simd vector into items buffer
        * @param items2write array of items to copy from and memory is not aligned
        */
        void store_a(T* items2write) const
        {
            data.copy_to(items2write, ::std::experimental::vector_aligned);
        }

        /* Store all elements of a simd vector into items buffer
        * @param items2write array of items to copy from and memory is not aligned
        */
        void store(T* items2write) const
        {
            constexpr size_t sz = size();

            for (size_t i = 0; i < sz; ++i)
            {
                items2write[i] = data[i];
            }
        }
    };
    //=========================================================================================//

    template <class T>
    inline CppSimdWrapper<T>& operator &= (CppSimdWrapper<T>& a, const CppSimdWrapper<T>& rhs)
    {
        a.data &= rhs.data;
        return a;
    }

    template <>
    inline CppSimdWrapper<double>& operator &= (CppSimdWrapper<double>& a, const CppSimdWrapper<double>& rhs)
    {
        using TUsedSimdUint64 = ::std::experimental::native_simd<uint64_t>;

        static_assert(sizeof(uint64_t) == sizeof(double));
        static_assert(sizeof(TUsedSimdUint64) == sizeof(a.data));

        // Type punning via memcpy
        TUsedSimdUint64 src_a_buffer, src_rhs_buffer;
        memcpy(&src_a_buffer, &a.data, sizeof(a.data));
        memcpy(&src_rhs_buffer, &rhs.data, sizeof(rhs.data));

        src_a_buffer &= src_rhs_buffer;

        memcpy(&a.data, &src_a_buffer, sizeof(a.data));

        return a;
    }

    template <>
    inline CppSimdWrapper<float>& operator &= (CppSimdWrapper<float>& a, const CppSimdWrapper<float>& rhs)
    {
        using TUsedSimdUint32 = ::std::experimental::native_simd<uint32_t>;
        static_assert(sizeof(uint32_t) == sizeof(float));
        static_assert(sizeof(TUsedSimdUint32) == sizeof(a.data));

        // Type punning via memcpy
        TUsedSimdUint32 src_a_buffer, src_rhs_buffer;
        memcpy(&src_a_buffer, &a.data, sizeof(a.data));
        memcpy(&src_rhs_buffer, &rhs.data, sizeof(rhs.data));

        src_a_buffer &= src_rhs_buffer;

        memcpy(&a.data, &src_a_buffer, sizeof(a.data));

        return a;
    }

    template <class T>
    inline CppSimdWrapper<T>& operator += (CppSimdWrapper<T>& a, const CppSimdWrapper<T>& rhs)
    {
        a.data += rhs.data;
        return a;
    }

    template <class T>
    inline CppSimdWrapper<T>& operator -= (CppSimdWrapper<T>& a, const CppSimdWrapper<T>& rhs)
    {
        a.data -= rhs.data;
        return a;
    }

    template <class T>
    inline CppSimdWrapper<T>& operator *= (CppSimdWrapper<T>& a, const CppSimdWrapper<T>& rhs)
    {
        a.data *= rhs.data;
        return a;
    }

    template <class T>
    inline CppSimdWrapper<T>& operator /= (CppSimdWrapper<T>& a, const CppSimdWrapper<T>& rhs)
    {
        a.data /= rhs.data;
        return a;
    }

    template <class T>
    inline CppSimdWrapper<T>& operator += (CppSimdWrapper<T>& a, T rhs)
    {
        a.data += rhs;
        return a;
    }

    template <class T>
    inline CppSimdWrapper<T>& operator -= (CppSimdWrapper<T>& a, T rhs)
    {
        a.data -= rhs;
        return a;
    }

    template <class T>
    inline CppSimdWrapper<T>& operator *= (CppSimdWrapper<T>& a, T rhs)
    {
        a.data *= rhs;
        return a;
    }
    
    template <class T>
    inline CppSimdWrapper<T>& operator /= (CppSimdWrapper<T>& a, T rhs)
    {
        a.data /= rhs;
        return a;
    }

    template <class T>
    inline CppSimdWrapper<T> operator - (const CppSimdWrapper<T>& a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(a.data - b.data);
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator + (const CppSimdWrapper<T>& a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(a.data + b.data);
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator * (const CppSimdWrapper<T>& a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(a.data * b.data);
        return res;
    }
    
    template <class T>
    inline CppSimdWrapper<T> operator / (const CppSimdWrapper<T>& a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(a.data / b.data);
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator & (const CppSimdWrapper<T>& a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>(a);
        res &= b;
        return res;
    }

    // Unary minus
    template <class T>
    inline CppSimdWrapper<T> operator - (const CppSimdWrapper<T>& a)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(-a.data);
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator - (T a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(typename CppSimdWrapper<T>::TUsedSimd(a) - b.data);
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator + (T a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(typename CppSimdWrapper<T>::TUsedSimd(a) + b.data);
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator * (T a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(typename CppSimdWrapper<T>::TUsedSimd(a) * b.data);
        return res;
    }
    
    template <class T>
    inline CppSimdWrapper<T> operator / (T a, const CppSimdWrapper<T>& b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(typename CppSimdWrapper<T>::TUsedSimd(a) / b.data);
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator - (const CppSimdWrapper<T>& a, T b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(a.data - typename CppSimdWrapper<T>::TUsedSimd(b));
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator + (const CppSimdWrapper<T>& a, T b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(a.data + typename CppSimdWrapper<T>::TUsedSimd(b));
        return res;
    }

    template <class T>
    inline CppSimdWrapper<T> operator * (const CppSimdWrapper<T>& a, T b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(a.data * typename CppSimdWrapper<T>::TUsedSimd(b));
        return res;
    }
    
    template <class T>
    inline CppSimdWrapper<T> operator / (const CppSimdWrapper<T>& a, T b)
    {
        CppSimdWrapper<T> res = CppSimdWrapper<T>::createSimdFromVec(a.data / typename CppSimdWrapper<T>::TUsedSimd(b));
        return res;
    }
}

// horizontal reductions
template <class T>
inline T horizontal_min(const burt::CppSimdWrapper<T>& a)
{
    return ::std::experimental::hmin(a.data);
}

template <class T>
inline T horizontal_max(const burt::CppSimdWrapper<T>& a)
{
    return ::std::experimental::hmax(a.data);
}

template <class T>
inline T horizontal_add(const burt::CppSimdWrapper<T>& a)
{
    return ::std::experimental::reduce(a.data);
}

// minimum/maximum elementwise

template <class T>
inline burt::CppSimdWrapper<T> minimum(const burt::CppSimdWrapper<T>& a, const burt::CppSimdWrapper<T>& b)
{
    burt::CppSimdWrapper<T> res = burt::CppSimdWrapper<T>::createSimdFromVec(::std::experimental::min(a.data, b.data));
    return res;
}

template <class T>
inline burt::CppSimdWrapper<T> min(const burt::CppSimdWrapper<T>& a, const burt::CppSimdWrapper<T>& b)
{
    burt::CppSimdWrapper<T> res = burt::CppSimdWrapper<T>::createSimdFromVec(::std::experimental::min(a.data, b.data));
    return res;
}

template <class T>
inline burt::CppSimdWrapper<T> maximum(const burt::CppSimdWrapper<T>& a, const burt::CppSimdWrapper<T>& b)
{
    burt::CppSimdWrapper<T> res = burt::CppSimdWrapper<T>::createSimdFromVec(::std::experimental::max(a.data, b.data));
    return res;
}

template <class T>
inline burt::CppSimdWrapper<T> max(const burt::CppSimdWrapper<T>& a, const burt::CppSimdWrapper<T>& b)
{
    burt::CppSimdWrapper<T> res = burt::CppSimdWrapper<T>::createSimdFromVec(::std::experimental::max(a.data, b.data));
    return res;
}

// Unary math functions
template <class T>
inline burt::CppSimdWrapper<T> square(const burt::CppSimdWrapper<T>& a)
{
    burt::CppSimdWrapper<T> res = burt::CppSimdWrapper<T>::createSimdFromVec(a.data * a.data);
    return res;
}

template <class T>
inline burt::CppSimdWrapper<T> exp(const burt::CppSimdWrapper<T>& a)
{
    burt::CppSimdWrapper<T> res = burt::CppSimdWrapper<T>::createSimdFromVec(::std::experimental::exp(a.data));
    return res;
}

template <class T>
inline burt::CppSimdWrapper<T> log(const burt::CppSimdWrapper<T>& a)
{
    burt::CppSimdWrapper<T> res = burt::CppSimdWrapper<T>::createSimdFromVec(::std::experimental::log(a.data));
    return res;
}

template <class T>
inline burt::CppSimdWrapper<T> abs(const burt::CppSimdWrapper<T>& a)
{
    burt::CppSimdWrapper<T> res = burt::CppSimdWrapper<T>::createSimdFromVec(::std::experimental::abs(a.data));
    return res;
}

#endif
