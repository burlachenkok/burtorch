/** @file
* Math statistic routines: expectation, variance, correlation, etc.
*/
#pragma once

#include <type_traits>

#include <math.h>
#include <stdint.h>
#include <stddef.h>

namespace burt
{
    namespace mathstats
    {
        /** Calculate the sum of all items.
        * @param x array of values
        * @param size size of the array of values
        * @return sum of all items
        */
        template <class Iterator>
        inline auto sum(Iterator x, size_t size) -> typename std::remove_reference<decltype(*x)>::type
        {
            // Due to C++11 auto never deduces to a reference type, always to a value type.
            auto s = *x;
            ++x;

            for (size_t i = 1; i < size; ++i, ++x)
                s += *x;
            return s;
        }

        /** Calculate the sum of all items.
        * @param x array of values
        * @param size size of the array of values
        * @return sum of all items
        */
        template <class T>
        inline auto sum(const T* x, size_t size) -> T
        {
            auto s = x[0];
            for (size_t i = 1; i < size; ++i)
                s += x[i];
            return s;
        }

        /** Find index of the minimum element
        * @param x iterator with random-access
        * @param size number of elements in the "array" begins from x
        * @return index of first minimum element
        */
        template <class Iterator>
        inline size_t minIndex(Iterator x, size_t size)
        {
            size_t index = 0;
            for (size_t i = 1; i < size; ++i)
            {
                if (x[i] < x[index])
                {
                    index = i;
                }
            }
            return index;
        }

        /** Find minimum element
        * @param x iterator with random-access
        * @param size number of elements in the "array" begins from x
        * @return reference to element with minimum value
        */
        template <class DataType, class Iterator>
        inline const DataType& minElement(Iterator x, size_t size)
        {
            return x[minIndex(x, size)];
        }

        /** Find index of the maximum element
        * @param x iterator with random-access
        * @param size number of elements in the "array" begins from x
        * @return index of first maximum element
        */
        template <class Iterator>
        inline size_t maxIndex(Iterator x, size_t size)
        {
            size_t index = 0;
            for (size_t i = 1; i < size; ++i)
            {
                if (x[i] > x[index])
                {
                    index = i;
                }
            }
            return index;
        }

        /** Find maximum element within a container
        * @param x iterator with random-access
        * @param size number of elements in the "array" begins from x
        * @return reference to element with maximum value
        */
        template <class DataType, class Iterator>
        inline const DataType& maxElement(Iterator x, size_t size)
        {
            return x[maxIndex(x, size)];
        }

        /** Find index of the minimum and maximum element
        * @param minIndex placeholder to store index of minimum item
        * @param minIndex placeholder to store index of maximum item
        * @param x iterator with random-access
        * @param size number of elements in the "array" begins from x
        */
        template <class Iterator>
        inline void maxAndMinIndex(size_t& minIndex, size_t& maxIndex, Iterator x, uint32_t size)
        {
            maxIndex = minIndex = 0;
            size_t i = 0;
            if (size % 2 == 1)
                i = 1;
            else
                i = 0;

            for (; i < size; i += 2)
            {
                if (x[i] > x[i + 1])
                {
                    if (x[i] > x[maxIndex])   maxIndex = i;
                    if (x[i + 1] < x[minIndex]) minIndex = i + 1;
                }
                else
                {
                    if (x[i + 1] > x[maxIndex]) maxIndex = i + 1;
                    if (x[i] < x[minIndex])   minIndex = i;
                }
            }
        }

        /** Find maximum value between two items
        * @param a first item
        * @param a second item
        * @return copy of item with maximum value
        */
        template <class TData>
        TData maxValue(const TData& a, const TData& b)
        {
            return (a) > (b) ? (a) : (b);
        }

        /** Find maximum value between two items
        * @param a first item
        * @param a second item
        * @return copy of item with minimum value
        */
        template <class TData>
        TData minValue(const TData& a, const TData& b)
        {
            return (b) > (a) ? (a) : (b);
        }

        /** Calculate the sample expectation of x
        * @param x array of values
        * @param size size of the array of values
        * @return value of sample expectation
        **/
        template <class Iterator>
        inline double mx(Iterator x, size_t size)
        {
            auto mxEvaluated = sum(x, size);
            return double(mxEvaluated) / double(size);
        }

        /** Calculate the sample uncorrected variance of x
        * @param x array of values
        * @param size size of the array of values
        * @return result value
        */
        template <class Iterator>
        inline double dx(Iterator x, size_t size)
        {
            if (size <= 1)
                return double(0.0);

            double m = mx(x, size);

            double Sum = 0.0;
            for (uint32_t i = 0; i < size; ++i)
                Sum += (x[i] - m)*(x[i] - m);
            double dxCalced = Sum / (size);

            return dxCalced;
        }

        /** Calculate the standard deviation of the variance of x
        * @param dxEvaluated variance (theoretical or estimated)
        * @return result value
        */
        inline double sigma(double dxEvaluated) {
            return sqrt(dxEvaluated);
        }

        /** Calculate the sample covariance of r.v. a, b. It is: M ((a-Ma) (b-Mb)).
        * @param a array of values
        * @param b array of values
        * @param size size of arrays of values
        * @return value of covariance between a and b
        * @remark If you have i=1,...n r.v. then you can create create covariance matrix of the elements (i, j) = cov (xi, xj). This matrix is non-negative definite.
        */
        template <class IteratorA, class IteratorB>
        inline double covariance(IteratorA a, IteratorB b, uint32_t size)
        {
            double ma = mx(a, size);
            double mb = mx(b, size);

            double cov = double();
            for (uint32_t i = 0; i < size; ++i)
                cov += (a[i] - ma)*(b[i] - mb);
            cov /= size;

            return cov;
        }

        /** Calculate the sample correlation of r.v. a, b. It is: M ((a-Ma) (b-Mb)).
        * @param a array of values
        * @param b array of values
        * @param size size of arrays of values
        * @return value of correlation from -1.0 to 1.0 indicating a correlation between a and b. In fact it describes the level of linear depency between r.v.
        */
        template <class DataType>
        inline double correlation(DataType * a, DataType * b, uint32_t size)
        {
            DataType cor = covariance(a, b, size) / (sqrt(dx(a, size) * dx(b, size)));
            return cor;
        }
    }
}
