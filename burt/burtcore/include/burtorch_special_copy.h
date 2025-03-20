#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"

#include <type_traits>
#include <initializer_list>
#include <stddef.h>
#include <memory.h>

/**
 * @brief Copies a fixed number of elements from one array to another at compile time.
 *
 * This function performs a compile-time copy of `kItems` elements from the source array to the destination array.
 * It uses a simple loop to copy the elements. The function is marked as `inline` and `static`, and is intended to
 * be used for small, fixed-size arrays that can be handled entirely at compile time.
 *
 * @tparam kItems The number of elements to copy from the source array to the destination array.
 * @tparam Type The type of the elements to be copied.
 * @param dst A pointer to the destination array where the elements will be copied.
 * @param src A pointer to the source array from which the elements will be copied.
 *
 * @note This function assumes that both `dst` and `src` are large enough to hold `kItems` elements.
 *
 * @see memcpy
 */
template<size_t kItems, class Type>
inline static void memcpyAtCompileTime(Type* restrict_ext dst, const Type* restrict_ext src)
{
    for (size_t i = 0; i < kItems; ++i)
    {
        dst[i] = src[i];
    }
}

/**
 * @brief Copies a fixed number of elements from one array to two destination arrays at compile time.
 *
 * This function copies `kItems` elements from the source array to two destination arrays simultaneously
 * at compile time. The function uses a loop to copy the elements to both `dstA` and `dstB`. It is intended for
 * use when you need to duplicate data into two separate arrays at compile time.
 *
 * @tparam kItems The number of elements to copy from the source array to both destination arrays.
 * @tparam Type The type of the elements to be copied.
 * @param dstA A pointer to the first destination array where the elements will be copied.
 * @param dstB A pointer to the second destination array where the elements will be copied.
 * @param src A pointer to the source array from which the elements will be copied.
 *
 * @note This function assumes that all destination arrays (`dstA`, `dstB`, and `src`) are large enough to
 *       hold `kItems` elements.
 *
 * @see memcpy
 */
template<size_t kItems, class Type>
inline static void memcpyAtCompileTime(Type* restrict_ext dstA, Type* restrict_ext dstB, const Type* restrict_ext src)
{
    for (size_t i = 0; i < kItems; ++i)
    {
        dstA[i] = src[i];
        dstB[i] = src[i];
    }
}
