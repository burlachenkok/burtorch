#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"

#include <stdint.h>
#include <assert.h>
#include <stdlib.h> 
#include <stddef.h>
#include <type_traits>

#include <cmath>

namespace burt
{
    /**
     * Create a byte from individual bits.
     *
     * @param b7 Most significant bit.
     * @param b6 Sixth bit.
     * @param b5 Fifth bit.
     * @param b4 Fourth bit.
     * @param b3 Third bit.
     * @param b2 Second bit.
     * @param b1 First bit.
     * @param b0 Least significant bit.
     * @return A byte constructed from the provided bits.
     */
    inline uint8_t createByte(uint8_t b7, uint8_t b6, uint8_t b5, uint8_t b4, uint8_t b3, uint8_t b2, uint8_t b1, uint8_t b0)
    {
        return (b7 & 0x1) << 7 | (b6 & 0x1) << 6 | (b5 & 0x1) << 5 | (b4 & 0x1) << 4 | (b3 & 0x1) << 3 | (b2 & 0x1) << 2 | (b1 & 0x1) << 1 | (b0 & 0x1) << 0;
    }

    /**
     * Create a byte from a list of least significant bits.
     *
     * @param lsbBitsList An array of 8 uint8_t values representing the bits,
     *                    where lsbBitsList[0] is the least significant bit and
     *                    lsbBitsList[7] is the most significant bit.
     * @return A byte constructed from the provided bits.
     */
    inline uint8_t createByteFromLsbList(uint8_t* lsbBitsList)
    {
        return createByte(lsbBitsList[7], lsbBitsList[6], lsbBitsList[5], lsbBitsList[4], lsbBitsList[3], lsbBitsList[2], lsbBitsList[1], lsbBitsList[0]);
    }

    /**
     * Create a byte from a list of most significant bits.
     *
     * @param msbBitsList Pointer to an array containing the most significant bits.
     *        The array must have at least 8 elements.
     * @return A byte constructed from the provided bits.
     */
    inline uint8_t createByteFromMsbList(uint8_t* msbBitsList)
    {
        return createByte(msbBitsList[0], msbBitsList[1], msbBitsList[2], msbBitsList[3], msbBitsList[4], msbBitsList[5], msbBitsList[6], msbBitsList[7]);
    }

    /** Convert from Little Endian(LSB) to Big Ending(MSB) or from  Big Ending(MSB) to Little Endian(LSB).
    * @param value input value which is represented in 64 bits
    * @return converted representation
    */
    inline uint64_t lsbToMsb64(uint64_t value)
    {
        return ((value >> 56) & 0x00000000000000FFULL) |  // Move byte 7 to byte 0
               ((value >> 40) & 0x000000000000FF00ULL) |  // Move byte 6 to byte 1
               ((value >> 24) & 0x0000000000FF0000ULL) |  // Move byte 5 to byte 2
               ((value >> 8)  & 0x00000000FF000000ULL) |  // Move byte 4 to byte 3
               ((value << 8)  & 0x000000FF00000000ULL) |  // Move byte 3 to byte 4
               ((value << 24) & 0x0000FF0000000000ULL) |  // Move byte 2 to byte 5
               ((value << 40) & 0x00FF000000000000ULL) |  // Move byte 1 to byte 6
               ((value << 56) & 0xFF00000000000000ULL);   // Move byte 0 to byte 7
    }

    /** Convert from Little Endian(LSB) to Big Ending(MSB) or from  Big Ending(MSB) to Little Endian(LSB).
    * @param value input value which is represented in 32 bits
    * @return converted representation
    */
    inline uint32_t lsbToMsb32(uint32_t value)
    {
        return ((value >> 24) & 0xFF) |        // Move byte 3 to byte 0
               ((value >> 8) & 0xFF00) |       // Move byte 2 to byte 1
               ((value << 8) & 0xFF0000) |     // Move byte 1 to byte 2
               ((value << 24) & 0xFF000000);   // Move byte 0 to byte 3
    }

    /** Convert from Little Endian(LSB) to Big Ending(MSB) or from  Big Ending(MSB) to Little Endian(LSB).
    * @param value input value which is represented in 16 bits
    * @return converted representation
    */
    inline uint16_t lsbToMsb16(uint16_t value)
    {
        return ((value >> 8) & 0x00FF) |   // Move byte 1 to byte 0
                ((value << 8) & 0xFF00);   // Move byte 0 to byte 1
    }

    /**
     * Extract the least significant bits (LSBs) of a given object and store them in an array.
     *
     * @param bits Pointer to an array where the extracted bits will be stored.
     * @param b The object from which LSBs will be extracted.
     * @return The total number of bits extracted.
     */
    template <class T>
    inline size_t getLsbBits(uint8_t* bits, const T& b)
    {
        static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD types are allowed");

        const uint8_t* b_bytes = reinterpret_cast<const uint8_t*>(&b);

        for (size_t i = 0; i < sizeof(b); ++i)
        {
            for (size_t j = 0; j < 8; ++j)
            {
                if (b_bytes[i] & (0x1 << j))
                {
                    bits[i * 8 + j] = 1;
                }
                else
                {
                    bits[i * 8 + j] = 0;
                }
            }
        }
        
        return sizeof(b) * 8;
    }

    /**
     * Extract the most significant bits from a provided variable and store them in an array.
     *
     * @param bits Pointer to an array where the extracted bits will be stored.
     * @param b The variable from which the most significant bits will be extracted.
     * @return The total number of bits extracted.
     */
    template <class T>
    inline size_t getMsbBits(uint8_t* bits, const T& b)
    {
        static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD types are allowed");

        const uint8_t* b_bytes = reinterpret_cast<const uint8_t*>(&b);

        for (size_t i = 0; i < sizeof(b); ++i)
        {
            for (size_t j = 0; j < 8; ++j)
            {
                if (b_bytes[i] & (0x1 << (7 - j)))
                {
                    bits[i * 8 + j] = 1;
                }
                else
                {
                    bits[i * 8 + j] = 0;
                }
            }
        }

        return sizeof(b) * 8;
    }

    /**
     * Calculate the square of a number.
     *
     * @param x The number to be squared.
     * @return The square of the given number.
     */
    template <class T>
    forceinline_ext T sqr(const T x) {
        return x * x;
    }

    /**
     * Calculate the power of a number or some math object raised to a natural exponent.
     *
     * @param a The base value to be raised.
     * @param naturalPower The non-negative exponent.
     * @return The result of raising the base value to the given exponent.
     */
    template <class T>
    T powerNatural(const T a, uint32_t naturalPower)
    {
        if (naturalPower == 0)
        {
            return T(1);
        }
        else if (naturalPower % 2 == 0)
        {
            T tmp = powerNatural(a, naturalPower / 2);
            return tmp * tmp;
        }
        else
        {
            T tmp = powerNatural(a, (naturalPower - 1) / 2);
            return tmp * tmp * a;
        }
    }

    /**
     * Compute the power of a base raised to a natural number.
     *
     * @param a The base value which can be of any numeric type.
     * @param naturalPower The exponent, a non-negative integer.
     * @return The result of raising the base to the power of the specified exponent.
     */
    template <class T>
    T powerNatural(const T a, uint64_t naturalPower)
    {
        if (naturalPower == 0)
        {
            return T(1);
        }
        else if (naturalPower % 2 == 0)
        {
            T tmp = powerNatural(a, naturalPower / 2);
            return tmp * tmp;
        }
        else
        {
            T tmp = powerNatural(a, (naturalPower - 1) / 2);
            return tmp * tmp * a;
        }
    }

    /**
     * Compute the power of a given positive number raised to a real (double) exponent.
     *
     * @param a The base number, which must be greater than 0.
     * @param realPower The exponent to which the base number is raised.
     * @return The result of raising the base number to the given exponent.
     * @warning number a should be postive
     */
    template <class T>
    forceinline_ext T powerReal(const T a, double realPower)
    {
        assert(double(a) > 0.0);
        return (T)(exp(realPower * log(double(a))));
    }

    /**
     * Compute the floor value of the logarithm of `a` with the given `base`.
     *
     * @param a The value for which to compute the logarithm. Must be greater than zero.
     * @param base The base of the logarithm. Must be greater than 1.
     * @return The floor value of the logarithm of `a` in the specified base.
     */
    template <class T>
    forceinline_ext uint32_t logBaseFloor(T a, uint32_t base)
    {
        assert(a > T());
        uint32_t r = 0;
        while (a > 1)
        {
            a /= base;
            r++;
        }
        return r;
    }

    /**
     * Compute the floor of the base-2 logarithm of a given value.
     *
     * @param a The value for which the base-2 logarithm floor is to be computed.
     * @return The largest integer less than or equal to the base-2 logarithm of the input value.
     */
    template <class T>
    forceinline_ext uint32_t log2IntFloor(const T a)
    {
        assert(a > T());
        return logBaseFloor(a, 2);
    }

    /**
     * Compute the floor value of the base-10 logarithm of a given number.
     *
     * @param a The number for which to compute the base-10 logarithm. Must be greater than zero.
     * @return The floor value of the base-10 logarithm of the given number.
    */
    template <class T>
    forceinline_ext uint32_t log10IntFloor(const T a)
    {
        assert(a > T());
        return logBaseFloor(a, 10);
    }

    /**
     * Compute the base-2 logarithm of a number.
     *
     * @param x The number for which the base-2 logarithm is to be calculated.
     * @return The base-2 logarithm of the specified number.
     */
    forceinline_ext double log2(double x)
    {
        return log(x) / log(2.0);
    }

    /**
     * Check if a number is prime.
     *
     * @param x The number to check for primality.
     * @return True if the number is prime, false otherwise.
    */
    template <class T>
    forceinline_ext bool isPrime(const T x)
    {
        if (x <= 1)
            return false;

        T last = static_cast<T>(sqrt(double(x)));

        for (T i = 2; i <= last; ++i)
        {
            if (x % i == 0)
                return false;
        }
        return true;
    }

    /**
     * Compute a^naturalPower mod p using an efficient modular exponentiation method.
     *
     * @param a The base.
     * @param naturalPower The exponent (it should be a natural number).
     * @param p The modulus.
     * @return The result of a raised to the power naturalPower modulo p.
     */
    template <class T>
    forceinline_ext T powerModP(const T a, const T naturalPower, const T p)
    {
        if (naturalPower == 0)
        {
            return T(1);
        }
        else if (naturalPower % 2 == 0)
        {
            T tmp = powerModP(a, naturalPower / 2, p);
            return (tmp * tmp) % p;
        }
        else
        {
            T tmp = powerModP(a, (naturalPower - 1) / 2, p);
            return (tmp * tmp * a) % p;
        }
    }

    /**
     * Compute the absolute value of a number.
     *
     * @param x A number of any type that supports comparison and negation operations.
     * @return The absolute value of the number.
    */
    template <class T>
    forceinline_ext T abs(const T x) {
        return x > T() ? x : -x;
    }

    /**
     * Return the absolute value of an 8-bit unsigned integer.
     *
     * @param x An 8-bit unsigned integer.
     * @return The absolute value of the input integer (which is the input itself since it is unsigned).
     */
    template <>
    forceinline_ext uint8_t abs(const uint8_t x) {
        return x;
    }

    /**
     * Compute the absolute value of an unsigned 16-bit integer.
     *
     * @param x The unsigned 16-bit integer whose absolute value is to be computed.
     * @return The absolute value of the provided unsigned 16-bit integer.
     */
    template <>
    forceinline_ext uint16_t abs(const uint16_t x) {
        return x;
    }

    /**
     * Compute the absolute value of an unsigned 32-bit integer. Since the input is unsigned,
     * the value is already non-negative, and the function simply returns the input.
     *
     * @param x The unsigned 32-bit integer whose absolute value is to be computed.
     * @return The absolute value of the input, which is the input itself.
     */
    template <>
    forceinline_ext uint32_t abs(const uint32_t x) {
        return x;
    }

    /**
     * Calculate the absolute value of an unsigned 64-bit integer.
     *
     * @param x The unsigned 64-bit integer whose absolute value is to be calculated.
     * @return The absolute value of the given unsigned 64-bit integer.
     */
    template <>
    forceinline_ext uint64_t abs(const uint64_t x) {
        return x;
    }

    /**
    * Rounds a given value to the nearest integer.
    *
    * @param v The value to be rounded.
    * @return The value rounded to the nearest integer.
    * @todo potentially this "simple" instruction takes plenty of clocks like "50". fl=>int conversion is not free
    */
    template <class TInteger = int, class T>
    forceinline_ext TInteger roundToNearestInt(const T v)
    {
        if (v - (TInteger)v > 0.5)
            return (TInteger)v + 1;
        else
            return (TInteger)v;
    }

    /** Get sign of the item
    * @param input vector with positive/negative elements
    * @param posSignValue value which will be used to encode positive item
    * @param negSignValue value which will be used to encode negative item
    * @return posSignValue or negSignValue depends on the sign of input
    */
    template <class T>
    forceinline_ext int sign(const T input, const T posSignValue = T(+1), const T negSignValue = T(-1))
    {
        return input < T() ? negSignValue : posSignValue;
    }

    /** Clamp value into segment [lower, upper]
    * @param lower lower bound of interval
    * @param upper upper bound of interval
    * @return point inside [lower, upper]
    */
    template <class T>
    forceinline_ext T clamp(const T val, const T lower, const T upper)
    {
        if (val < lower)
        {
            return lower;
        }
        else if (val > upper)
        {
            return upper;
        }
        else
        {
            return val;
        }
    }

    /** linearly interpolated vector a + t * (b - a)
    * @param t parameter for linear interpolation
    * @param a start point
    * @param b end point point
    * @return point inside [a,b] if t inside [0,1].
    */
    template <class TElementType, class TPoint>
    forceinline_ext TPoint lerp(const TElementType t, const TPoint& a, const TPoint& b)
    {
        return a + t * (b - a);
    }

    /** linearly interpolated vector a + t * (b - a)
    * @param t parameter for linear interpolation
    * @param a start point
    * @param dir direction
    * @return point inside [a,b] if t inside [0,1].
    */
    template <class TElementType, class TPoint, class TVec>
    forceinline_ext TVec lerp(const TElementType t, const TPoint& a, const TVec& dir)
    {
        return a + t * dir;
    }

    /**
     * Determine the minimum of two values with usual branching construction.
     *
     * @param x The first value to be compared.
     * @param y The second value to be compared.
     * @return The smaller of the two values.
     */
    template <class T>
    forceinline_ext T bracnhMinimum(T x, T y)
    {
        if (x < y)
            return x;
        else
            return y;
    }

    /**
     * Determine the maximum of two values with usual branching construction.
     *
     * @param x The first value to be compared.
     * @param y The second value to be compared.
     * @return The maximum of the two values.
    */
    template <class T>
    forceinline_ext T bracnhMaximum(T x, T y)
    {
        if (x > y)
            return x;
        else
            return y;
    }
    
    //======================================== No branch functions minimum start====================================//

    template<class T>
    concept isIntegral = std::is_integral<T>::value;

    /**
     * Computes the minimum of two integral values without branching.
     *
     * @param x The first integral value.
     * @param y The second integral value.
     * @return The minimum of the two provided integral values.
     */
    template <class T>
    requires isIntegral<T>
    forceinline_ext T noBracnhMinimumForIntegrals(T x, T y)
    {
        static_assert(~(-T(1)) == T(0), "Check that -1 consist of 0b1111");
        T result = y ^ ((x ^ y) & -(x < y));
        return result;
    }

    template<bool promiseThatNumbersAreNonNegative = false>
    forceinline_ext double noBracnhMinimum(double x, double y) 
    {
        static_assert(sizeof(double) == sizeof(int64_t));
        
        const int64_t& restrict_ext x_ref = reinterpret_cast<uint64_t&>(x);
        const int64_t& restrict_ext y_ref = reinterpret_cast<uint64_t&>(y);

        if constexpr (promiseThatNumbersAreNonNegative)
        {
            // uint comparison -- fp64 is not compatible to compare fp64 and uint64 in this case. comprare fp32 as ui64
            int64_t result = y_ref ^ ((x_ref ^ y_ref) & -(x_ref < y_ref));
            return reinterpret_cast<double&>(result);
        }
        else
        {
            // uint comparison -- fp64 is not compatible to compare fp64 and uint64 in this case. comprare fp64 as fp64
            int64_t result = y_ref ^ ((x_ref ^ y_ref) & -(x < y));
            return reinterpret_cast<double&>(result);
        }
    }

    template<bool promiseThatNumbersAreNonNegative = false>
    forceinline_ext float noBracnhMinimum(float x, float y) 
    {
        static_assert(sizeof(float) == sizeof(int32_t));

        const int32_t& restrict_ext x_ref = reinterpret_cast<int32_t&>(x);
        const int32_t& restrict_ext y_ref = reinterpret_cast<int32_t&>(y);

        if constexpr (promiseThatNumbersAreNonNegative)
        {
            // uint comparison -- fp32 is fully compatible to compare fp32 and uint32 in this case
            int32_t result = y_ref ^ ((x_ref ^ y_ref) & -(x_ref < y_ref));
            return reinterpret_cast<float&>(result);
        }
        else
        {
            // uint comparison -- fp32 is not compatible to compare fp32 and uint32 in this case. comprare fp32 as fp32
            int32_t result = y_ref ^ ((x_ref ^ y_ref) & -(x < y));
            return reinterpret_cast<float&>(result);
        }
    }

    /**
     * Compute the maximum of two integral values without using branching.
     *
     * @param x First integral value.
     * @param y Second integral value.
     * @return The maximum of the two provided integral values.
    */
    template <class T>
    requires isIntegral<T>
    forceinline_ext T noBracnhMaximumForIntegrals(T x, T y) {
        static_assert(~(-T(1)) == T(0), "Check that -1 consist of 0b1111");
        T result = y ^ ((x ^ y) & -(x > y));
        return result;
    }

    /**
     * Compute the maximum of two double values without using branching.
     *
     * @param x First double value.
     * @param y Second double value.
     * @return The maximum of x and y.
     */
    template<bool promiseThatNumbersAreNonNegative = false>
    forceinline_ext double noBracnhMaximum(double x, double y)
    {
        static_assert(sizeof(double) == sizeof(int64_t));

        const int64_t& restrict_ext x_ref = reinterpret_cast<uint64_t&>(x);
        const int64_t& restrict_ext y_ref = reinterpret_cast<uint64_t&>(y);
        
        if constexpr (promiseThatNumbersAreNonNegative)
        {
            int64_t result = y_ref ^ ((x_ref ^ y_ref) & -(x_ref > y_ref));
            return reinterpret_cast<double&>(result);
        }
        else
        { 
            int64_t result = y_ref ^ ((x_ref ^ y_ref) & -(x > y));
            return reinterpret_cast<double&>(result);
        }        
    }

    /**
     * Compute the maximum of two floating-point numbers without using a branching instruction.
     *
     * @param x The first floating-point number.
     * @param y The second floating-point number.
     * @return The maximum of x and y.
    */
    template<bool promiseThatNumbersAreNonNegative = false>
    forceinline_ext float noBracnhMaximum(float x, float y)
    {
        static_assert(sizeof(float) == sizeof(int32_t));

        const int32_t& restrict_ext x_ref = reinterpret_cast<int32_t&>(x);
        const int32_t& restrict_ext y_ref = reinterpret_cast<int32_t&>(y);

        if constexpr (promiseThatNumbersAreNonNegative)
        {
            int32_t result = y_ref ^ ((x_ref ^ y_ref) & -(x_ref > y_ref));
            return reinterpret_cast<float&>(result);
        }
        else
        {
            int32_t result = y_ref ^ ((x_ref ^ y_ref) & -(x > y));
            return reinterpret_cast<float&>(result);
        }
    }
    //======================================== No branch functions end======================================//

    /**
     * Determine the minimum of two values.
     *
     * @param a The first value.
     * @param b The second value.
     * @return The minimum of the two values.
     * @note  for intergral type not-branch version is executed
    */
    template <class T>
    forceinline_ext T minimum(T a, T b)
    {
        if constexpr (std::is_integral<T>::value)
        {
            return noBracnhMinimumForIntegrals(a, b);
        }
        else
        {
            return bracnhMinimum(a, b);
        }
    }

    /**
     * Computes the minimum of two floating-point numbers.
     *
     * @param a The first floating-point number.
     * @param b The second floating-point number.
     * @return The minimum of the two provided numbers.
     */
    template<bool promiseThatNumbersAreNonNegative = false>
    forceinline_ext float minimum(float a, float b)
    {
        if constexpr (true)
            return noBracnhMinimum<promiseThatNumbersAreNonNegative> (a, b);
        else
            return bracnhMinimum(a, b);
    }
    
    /**
     * Determine the minimum of two double values.
     *
     * @param a The first double value.
     * @param b The second double value.
     * @return The minimum of the two provided double values.
     */
    template<bool promiseThatNumbersAreNonNegative = false>
    forceinline_ext double minimum(double a, double b)
    {
        if constexpr (true)
            return noBracnhMinimum<promiseThatNumbersAreNonNegative> (a, b);
        else
            return bracnhMinimum(a, b);
    }

    /**
     * Compute the maximum of two values.
     *
     * @param a First value.
     * @param b Second value.
     * @return The maximum of the two values.
    */
    template <class T>
    forceinline_ext T maximum(T a, T b)
    {
        if constexpr (std::is_integral<T>::value)
        {
            return noBracnhMaximumForIntegrals(a, b);
        }
        else
        {
            return bracnhMaximum(a, b);
        }
    }

    /**
     * Returns the maximum of two floating-point numbers.
     *
     * @param a First floating-point number.
     * @param b Second floating-point number.
     * @return The maximum of the two provided floating-point numbers.
     */
    forceinline_ext float maximum(float a, float b)
    {
        return bracnhMaximum(a, b);
    }

    /**
     * Compute the maximum of two double values.
     *
     * @param a The first double value to compare.
     * @param b The second double value to compare.
     * @return The greater of the two double values.
     */
    forceinline_ext double maximum(double a, double b)
    {
        return bracnhMaximum(a, b);
    }

    /** Compute (x+y)%n with strong assumption: a, b \in [0,n-1]
    * @param a integer value in [0,n-1]
    * @param b integer value in [0,n-1]
    * @param (a + b)%n
    */
    template <class T>
    forceinline_ext T add_two_numbers_modN(T a, T b, T n) {
        T z = a + b;
        return minimum(z, z - n);
    }

    /**
     * Compute the lower bound of the base-2 logarithm of a number at compile time.
     *
     * @param n The number for which the base-2 logarithm lower bound is to be computed.
     * @return The lower bound of the base-2 logarithm of the input number.
    */
    inline constexpr size_t log2AtCompileTimeLowerBound(size_t n)
    {
        return ( (n < 2) ? 0 : 1 + log2AtCompileTimeLowerBound(n/2) );
    }

    /** Compute the smallest unsigned integer which is greater or equal to number, such that it is dividable by kMultiple
    @param number requested input number
    @tparam kMultiple mulitiplicative factor
    @return smallest unsigned value greater or equal to number dividable by kMultiple
    @note implementation use essentially bit tricks
    */
    template<size_t kMultiple>
    forceinline_ext size_t roundToNearestMultipleUp(size_t number)
    {
        if constexpr (kMultiple == 1)
        {
            return number;
        }
        
        else if constexpr (kMultiple == 65536 || kMultiple == 32768 || kMultiple == 16384 || kMultiple == 8192 || 
                           kMultiple == 4096  || kMultiple == 2048  || kMultiple == 1024  || kMultiple == 512  ||
                           kMultiple == 256   || kMultiple == 128   || kMultiple == 64    || kMultiple == 32   || 
                           kMultiple == 16    || kMultiple == 8     || kMultiple == 4     || kMultiple == 2)
        {
            constexpr size_t log2value = log2AtCompileTimeLowerBound(kMultiple);
            constexpr size_t mask = (kMultiple - 1);
            if ( (number & mask) == 0)
            {
                // Residual is already has a need power of "2" because all bits lower to 2^{log2value}
                return number;
            }
            else
            {
                size_t numberMasked = number >> log2value;       // make residual (via shift right)
                numberMasked++;                                  // round up by 1
                size_t numberResult = numberMasked << log2value; // shift back

                // This is in fact equivalent to:
                //  1. take number which unfortunately is not dividable by kMultiple.
                //  2. the number will be dividable by kMultiple once all lowest log2(kMultiple) bits will be equal to zero.
                //  3. now to achieve this we increment "number" by "1"
                //  4. once all bits will filled with "1" the next addition drops all residual bits to "0"
                //  5. and one bit with value "1" will be bring into number as CF=1
                return numberResult;
            }
        }
        else
        {
            size_t residual = number % kMultiple;

            if (residual == 0)
            {
                return number;
            }
            else
            {
                size_t augment = (kMultiple - residual);
                size_t numberAsMultipleForLineSize = number + augment;
                return numberAsMultipleForLineSize;
            }
        }
    }

    /** Compute the smallest unsigned integer which is less or equal to number, such that it is dividable by kMultiple
    @param number requested input number
    @tparam kMultiple mulitiplicative factor
    @return smallest unsigned value greater or equal to number dividable by kMultiple
    */
    template<size_t kMultiple>
    forceinline_ext size_t roundToNearestMultipleDown(size_t number)
    {
        if constexpr (kMultiple == 1)
        {
            return number;
        }
        else if constexpr (kMultiple == 65536 || kMultiple == 32768 || kMultiple == 16384 || kMultiple == 8192 ||
            kMultiple == 4096 || kMultiple == 2048 || kMultiple == 1024 || kMultiple == 512 ||
            kMultiple == 256 || kMultiple == 128 || kMultiple == 64 || kMultiple == 32 ||
            kMultiple == 16 || kMultiple == 8 || kMultiple == 4 || kMultiple == 2)
        {
            constexpr size_t mask = (kMultiple - 1);
            size_t result = number & (~mask);            
            return result;
        }
        else
        {
            size_t result = number - (number % kMultiple);
            return result;
        }
    }

    /** Is the address properly aligned to kAligment
    @param address poitner to adresse, specifically to some byte.
    @return true if address is aligned
    @tparam kAligment alignment of address in bytes
    */
    template<size_t kAligment>
    forceinline_ext bool isAddressAligned(const void * address)
    {
        if constexpr (kAligment == 1)
        {
            return true;
        }
        else if constexpr (kAligment == 65536 || kAligment == 32768 || kAligment == 16384 || kAligment == 8192 ||
                           kAligment == 4096 || kAligment == 2048 || kAligment == 1024 || kAligment == 512 ||
                           kAligment == 256 || kAligment == 128 || kAligment == 64 || kAligment == 32 ||
                           kAligment == 16 || kAligment == 8 || kAligment == 4 || kAligment == 2)
        {
            constexpr uintptr_t mask = (kAligment - 1);
            uintptr_t result = (reinterpret_cast<uintptr_t>(address)) & (mask);

            return result == 0;
        }
        else
        {
            return (reinterpret_cast<uintptr_t>(address)) % kAligment;
        }
    }

    /**
     * Compares the absolute values of two numbers to determine if the first is greater than the second.
     *
     * @param first The first number to compare.
     * @param second The second number to compare.
     * @return A boolean value indicating whether the absolute value of the first number is greater than the absolute value of the second number.
    */
    template <class T>
    forceinline_ext bool isFirstHigherThenSecondIgnoreSign(T first, T second)
    {
        return burt::abs(first) > burt::abs(second);
    }

    /**
     * Compare two floating point numbers disregarding their sign.
     *
     * @param first The first floating point number.
     * @param second The second floating point number.
     * @return True if the absolute value of the first number is greater than the absolute value of the second number. False otherwise.
     */
    template <>
    forceinline_ext bool isFirstHigherThenSecondIgnoreSign(float first, float second)
    {
        constexpr bool kOptimized = false;

        if (kOptimized)
        {
            uint32_t& firstUInt32 = reinterpret_cast<uint32_t&> (first);
            uint32_t& secondUInt32 = reinterpret_cast<uint32_t&> (second);
            
            firstUInt32 &= ~(uint32_t(0x1) << 31);
            secondUInt32 &= ~(uint32_t(0x1) << 31);
            return firstUInt32 > secondUInt32;
        }
        else
        {
            return burt::abs(first) > burt::abs(second);
        }
    }

    /**
     * Compare two double values by their absolute values.
     *
     * @param first The first double value.
     * @param second The second double value.
     * @return True if the absolute value of the first double is greater than the absolute value of the second double, false otherwise.
     */
    template <>
    forceinline_ext bool isFirstHigherThenSecondIgnoreSign(double first, double second)
    {
        constexpr bool kOptimized = false;
        
        if (kOptimized)
        {
            uint64_t& firstUInt64 = reinterpret_cast<uint64_t&> (first);
            uint64_t& secondUInt64 = reinterpret_cast<uint64_t&> (second);

            firstUInt64 &= ~(uint64_t(0x1) << 63);
            secondUInt64 &= ~(uint64_t(0x1) << 63);

            return firstUInt64 > secondUInt64;
        }
        else
        {
            return burt::abs(first) > burt::abs(second);
        }
    }
}
