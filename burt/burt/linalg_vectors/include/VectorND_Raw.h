/** @file
* C++ cross-platform implementation of mathematical vector, elements of which are stored in dynamically allocated memory
*/

#pragma once

#include "burt/copylocal/include/Copier.h"
#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include "burt/system/include/PlatformSpecificMacroses.h"

#include <vector>
#include <initializer_list>
#include <type_traits>
#include <sstream>
#include <iostream>
#include <unordered_map>

#include <cstdlib>

#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

#define BURT_USE_CUSTOM_HEAPS 1 ///< Local Heap Strategy. Good for spead, bad when memory movement across threads is required.

namespace burt
{
    template <typename TVec>
    class LightVectorND;

    /** Allocated memory enough for store specific number of items
    * @param nBytes number of bytes to allocate
    * @return pointer to allocated memory
    */
    forceinline_ext void* memAllocationCRT(size_t nBytes) {
        if (nBytes == 0)
            return nullptr;

        #if __cpp_lib_hardware_interference_size >= 201703L
            constexpr size_t kCacheLizeSizeInBytes = std::hardware_destructive_interference_size;
        #else
            constexpr size_t kCacheLizeSizeInBytes = 64;
        #endif

        size_t usedSize = burt::roundToNearestMultipleUp<kCacheLizeSizeInBytes>(nBytes);

        #if BURT_WINDOWS
            return _aligned_malloc(usedSize, kCacheLizeSizeInBytes);
        #else
            return aligned_alloc(kCacheLizeSizeInBytes, usedSize);
        #endif
    }

    /** Free memory previously allocated via memAllocation
    * @param rawPointer pointer previously allocated with memAllocation
    * @param nBytes number of bytes to allocate memory
    */
    forceinline_ext void memFreeCRT(void* rawPointer, size_t nBytes) {
        if (nBytes == 0 || rawPointer == nullptr)
            return;

        #if BURT_WINDOWS
            _aligned_free(rawPointer);
        #else
            free(rawPointer);
        #endif
    }

    inline size_t roundBytesForPool(size_t nBytes)
    {
        if (nBytes >= 4 * 1024)
        {
            constexpr size_t chunkSize = 1024;
            size_t destNBytesAllocation = burt::roundToNearestMultipleUp<chunkSize>(nBytes);
            return destNBytesAllocation;
        }
        else
        {
            if (nBytes <= 64)
                return 64;
            else if (nBytes <= 128)
                return 128;
            else if (nBytes <= 256)
                return 256;
            else if (nBytes <= 512)
                return 512;
            else if (nBytes <= 1 * 1024)
                return 1 * 1024;
            else if (nBytes <= 2 * 1024)
                return 2 * 1024;
            else // if (nBytes < 4 * 1024)
                return 4 * 1024;
        }
    }


    typedef void* (*TMemoryAllocationCallback)(size_t nBytes);
    typedef void (*TMemoryFreeCallback)(void* rawPointer, size_t nBytes);

    /** Vector with arbitrarily dimension
    * @tparam type of elements inside the vector
    * @tparam memAllocation callback used for allocate memory
    * @tparam memFree callback used for free memory
    * @sa VectorNDStd
    */
    template <typename T>
    class VectorNDRaw
    {
    private:
        const static TMemoryAllocationCallback memAllocation;
        const static TMemoryFreeCallback memFree;

    public:

        /** Typedef for elements types.
        */
        using TElementType = T;

    private:

        /** Helper function to declare number of bytes which is needed to store specified number of components in flat array
        * @param components number of components in the array
        * @return number of bytes in flat array
        */
        static size_t components2bytes(size_t components) {
            return components * sizeof(T);
        }

        size_t componentsCount;                   ///< Components count in items
        T* components;                            ///< Components of the vector, raw pointer

    public:

        /** Debug print with debug representation of the vector
        * @param out output steam for which operator << (const char* str) and  operator << (const T&) is defined.
        * @param variableName used variable name
        * @param delimiter used delimiter during printing values
        */
        template<class text_out_steam>
        void dbgPrintInMatlabStyle(text_out_steam& out, const char* variableName = "x=", const char* delimiter = ",") const
        {
            out << variableName << "[";

            size_t sz = size();
            for (size_t i = 0; i < sz; ++i)
            {
                if (i != 0)
                    out << delimiter;
                out << components[i];
            }
            out << "];\n";
        }

        /** Create empty vector
        */
        VectorNDRaw() noexcept
            : components(0)
            , componentsCount(0)
        {
        }

        /** Copy ctor
        */
        VectorNDRaw(const VectorNDRaw& rhs) noexcept
        : VectorNDRaw(eAllocNotInit, rhs.componentsCount)
        {
            burt::CopyHelpers::copy<T>(components, rhs.components, rhs.size());
        }

        /** Create vector with specified dimension
        * @param dimension specified dimension
        * @remark all component are initialization with default ctor or zero.
        * @remark underlying memory is allocated via memAllocation callback and data is zero-initialized
        */
        VectorNDRaw(size_t dimension) noexcept
        : VectorNDRaw(eAllocAndSetToZero, dimension)
        {
        }

        /** Destructor.
        */
        ~VectorNDRaw() noexcept
        {
            if (!components)
                return;

            size_t sz = size();

            if constexpr (std::is_trivially_copyable<T>::value)
            {
                memFree(components, components2bytes(sz));
            }
            else
            {
                for (size_t i = 0; i < sz; ++i)
                {
                    components[i].~T();
                }

                memFree(components, components2bytes(sz));
            }
            componentsCount = 0;
        }

        /** Assignment move operator
        * @param rhs xvalue expression from which we perform move
        */
        VectorNDRaw& operator = (VectorNDRaw&& rhs) noexcept
        {
            this->~VectorNDRaw();
            componentsCount = rhs.componentsCount;
            components = rhs.components;
            rhs.components = nullptr;
            rhs.componentsCount = 0;
            return *this;
        }

        /** Copy move operator
        */
        VectorNDRaw(VectorNDRaw&& rhs) noexcept
        {
            componentsCount = rhs.componentsCount;
            components = rhs.components;
            rhs.components = nullptr;
            rhs.componentsCount = 0;
        }

        /** Usual assignment operator
        */
        VectorNDRaw& operator = (const VectorNDRaw& rhs) noexcept
        {
            if (this == &rhs)
                return *this;
            
            if (componentsCount != rhs.componentsCount)
            {
                this->~VectorNDRaw();
                components = new(memAllocation(components2bytes(rhs.componentsCount))) T[rhs.componentsCount];
                componentsCount = rhs.componentsCount;
            }

            
            burt::CopyHelpers::copy<T>(components, rhs.components, rhs.size());
            
            return *this;
        }

        /** Create vector in form [start, start + 1, start + 2,...]
        * @param dimension dimension of the vector
        * @tparam start initial value for sequence
        * @return result vector
        */
        template<int start>
        static VectorNDRaw sequence(size_t dimension)
        {
            VectorNDRaw res(eAllocNotInit, dimension);
            T value = (T)start;

            for (size_t i = 0; i < dimension; ++i, value += T(1))
            {
                res.components[i] = start + i;
            }
            return res;
        }

        /** Create vector in form [1,1,1,...,1]
        * @param dimension dimension of the vector
        */
        static VectorNDRaw eye(size_t dimension)
        {
            VectorNDRaw res(eAllocNotInit, dimension);
            res.setAll(T(1));
            return res;
        }

        /** Get uninitialized vector
        * @param dimension dimension of the vector
        * @remark Please call it only if you understand what you're doing.
        */
        static VectorNDRaw getUninitializedVector(size_t dimension)
        {
            VectorNDRaw res(eAllocNotInit, dimension);
            return res;
        }

        /** Create vector in form [value,value,value,...,value]
        * @param dimension dimension of the vector
        * @param value used value to initialize vector
        */
        static VectorNDRaw eye(size_t dimension, T value)
        {
            VectorNDRaw res(eAllocNotInit, dimension);
            res.setAll(value);
            return res;
        }

        /** Create vector via using specified values from initialize list
        * @param objects init. list in form {v1, v2,...vn}
        */
        VectorNDRaw(std::initializer_list<T> objects)
        : VectorNDRaw(eAllocNotInit, objects.size())
        {
            size_t sz = objects.size();
            const T* item = objects.begin();

            burt::CopyHelpers::copy<T>(components, item, sz);
        }

        /** Create vector via using specified values from initialize list
        * @param objects init. list in form {v1, v2,...vn}
        * @remark it's due to lack of poor support of std::initializer list for temporary objects in Microsoft Visual Studio compiler
        */
        static VectorNDRaw init(std::initializer_list<T> objects)
        {
            VectorNDRaw res = objects;
            return res;
        }

        /** Create vector via using specified values from raw array
        * @param srcItems start of the array used to initialize vector
        * @param srcNumberOfItems number of items used from srcItems array
        */
        VectorNDRaw(const T* srcItems, size_t srcNumberOfItems)
        : VectorNDRaw(eAllocNotInit, srcNumberOfItems)
        {
            burt::CopyHelpers::copy<T>(components, srcItems, srcNumberOfItems);
        }

        /** Dump vector (which is almost in all applied math is column matrix with one column) by rows
        * @param out pointer to memory in which content of the vector will be dumped
        * @param startRow first row
        * @param endRow last row
        */
        void dumpByRows(T* out, size_t startRow, size_t endRow) const
        {
            for (size_t i = startRow; i <= endRow; ++i)
                *(out++) = get(i);
        }

        /** Dump all vector (which is almost in all applied math is column matrix with one column) by rows
        * @param out pointer to memory in which content of the vector will be dumped
        */
        void dumpByRows(T* out) const
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                *(out++) = get(i);
        }

        /** Create string representation of the content of the vector
        * @param delimiter symbol used as delimiter for vector elements
        * @return formed string
        */
        ::std::string toString(const char* delimiter = ",") const
        {
            ::std::stringstream s;
            s << "[";

            size_t sz = size();
            for (size_t i = 0; i < sz; ++i)
            {
                if (i != 0)
                    s << delimiter;
                s << get(i);
            }
            s << "]";
            return s.str();
        }

        /** Effective swap content with other vector
        * @param vecA first vector
        * @param vecB second vector
        * @remark vecA and vecB are different objects in memory
        * @tparam swappedVectorsHasTheSameDimension prior knowledge that vectors has the same dimension
        */
        template <bool swappedVectorsHasTheSameDimension = false>
        static void swapDifferentVectors(VectorNDRaw& vecA, VectorNDRaw& vecB) noexcept {
            burt::CopyHelpers::swapDifferentObjects(&vecA.components, &vecB.components);
            
            if constexpr (swappedVectorsHasTheSameDimension)
            {
                // Do nothing
            }
            else
            {
                burt::CopyHelpers::swapDifferentObjects(&vecA.componentsCount, &vecB.componentsCount);
            }
        }

        /** Size of the vector in terms of number of items in it
        * @return size of the vector
        */
        size_t size() const {
            return componentsCount;
        }

        /** Size of the vector in terms of bytes occupied by it
        * @return size of the vector in bytes
        */
        size_t sizeInBytes() const {
            return components2bytes(componentsCount);
        }

        /** Memory reserved for this vector
        * @return real number of elements reserved for the vector content
        */
        size_t capacity() const {
            return componentsCount;
        }

        /** Resize vector for new size
        * @param newSize new size of the vector
        * @return reference to itself
        * @remark this call can invalidate previously obtained pointers via data() and dataConst()
        */
        VectorNDRaw& resize(size_t newSize)
        {
            T* mewComponents = new(memAllocation(components2bytes(newSize))) T[newSize]();
            size_t newComponentsCount = newSize;

            if (!mewComponents) [[unlikely]]
            {
                assert(!"Allocation problems in resize() ");
                return *this;
            }

            burt::CopyHelpers::copy<T>(mewComponents, components, componentsCount);

            if (components) [[likely]]
            {
                size_t sz = componentsCount;

                if constexpr (std::is_trivially_copyable<T>::value)
                {
                    memFree(components, components2bytes(sz));
                }
                else
                {
                    for (size_t i = 0; i < sz; ++i)
                        components[i].~T();
                    memFree(components, components2bytes(sz));
                }
                components = nullptr;
            }
            components = mewComponents;
            componentsCount = newComponentsCount;
            return *this;
        }

        /** Get maximum element in the vector
        * @return maximum element
        */
        T maxItem() const
        {
            size_t sz = size();

            if (sz == 0) [[unlikely]]
                return T();

            T max_item = components[0];
            for (size_t i = 1; i < sz; ++i)
                if (components[i] > max_item)
                    max_item = components[i];

            return max_item;
        }

        /** Get minimum element value in the vector
        * @return minimum element value in the vector
        */
        T minItem() const
        {
            size_t sz = size();

            if (sz == 0) [[unlikely]]
                return T();

            T min_item = components[0];
            for (size_t i = 1; i < sz; ++i)
                if (components[i] < min_item)
                    min_item = components[i];

            return min_item;
        }

        /** Current vector of elements is interpretated as a dense vector with margins for each example
        * @return return unweighted logistic loss value for margins
        * @remark Margin is quantity used in Machine Learning are relative areas to denote dot product of features by datapoints value times sign of example.
        */
        template <class TAccumulator = T>
        static TAccumulator logisticUnweightedLossFromMargin(const VectorNDRaw& margin)
        {
            if (margin.size() == 0) [[unlikely]]
                return T();

            TAccumulator res = TAccumulator();
            size_t sz = margin.size();

            for (size_t i = 0; i < sz; ++i) {
                res += ::log(1.0 + ::exp(-(margin.get(i))));
            }
            res *= 1.0 / TAccumulator(sz);
            return res;
        }

        template <class TAccumulator = T>
        static TAccumulator logisticUnweightedLossFromMarginSigmoid(const VectorNDRaw& classificationMarginSigmoid)
        {
            TAccumulator res = TAccumulator();
            size_t sz = classificationMarginSigmoid.size();

            for (size_t i = 0; i < sz; ++i) {
                res -= ::log(classificationMarginSigmoid[i]);
            }
            res *= 1.0 / TAccumulator(sz);
            return res;
        }

        /** Evaluate sum of all elements
        * @return result sum
        * @remark Implementation contains naive way. More robust way in terms of numerics is firstly sort array.
        */
        template <class TAccumulator = T>
        TAccumulator sum() const
        {
            TAccumulator res = TAccumulator();
            size_t sz = size();
            for (size_t i = 0; i < sz; ++i)
                res += components[i];
            return res;
        }

        /** Create vector which element wise contains absolute values of current (*this) vector
        * @return vector with element wise positive values
        */
        VectorNDRaw abs() const
        {
            T nullItem = T();

            size_t dim = size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i)
            {
                if (get(i) >= nullItem)
                    res.components[i] = get(i);
                else
                    res.components[i] = -get(i);
            }
            return res;
        }

        /** Get new vector which store sign's of vec input
        * @param input vector with positive/negative elements
        * @param posSignValue value which is set for positive items
        * @param negSignValue value which is set for negative items
        * @return result vector which element wise contain -1/+1 depend on sign
        */
        static VectorNDRaw sign(const VectorNDRaw& vec,
            const T posSignValue = T(+1),
            const T negSignValue = T(-1))
        {
            size_t dim = vec.size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i)
            {
                if (vec.get(i) > 0)
                    res.set(i, posSignValue);
                else
                    res.set(i, negSignValue);
            }
            return res;
        }

        /** Get copy of item by index i
        * @param i index of item
        * @return reference to item offseted by "i"
        */
        T get(size_t i) const
        {
            #if BURT_DEBUG_BUILD
            if (i >= componentsCount)
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
            #endif

            const T& result = components[i];
            return result;
        }

        /** Get non-const reference item by index i
        * @param i index of item
        * @return reference to item offseted by "i"
        */
        T& getRaw(size_t i)
        {
            #if BURT_DEBUG_BUILD
            if (i >= componentsCount)
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
            #endif

            T& result = components[i];
            return result;
        }

        /** Get const reference item by index i
        * @param i index of item
        * @return reference to item offseted by "i"
        */
        const T& getRaw(size_t i) const
        {
            #if BURT_DEBUG_BUILD
            if (i >= componentsCount)
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
            #endif

            T& result = components[i];
            return result;
        }
        
        /** Get raw pointer for underlying buffer in which elements of dense vector are stored.
        * @return pointer for underlying buffer which contains at least size() elements
        * @sa size()
        * @warning The pointer can be invalidated by resize()
        */
        T* data() {
            return components;
        }

        /** Get raw pointer for underlying buffer in which elements of dense vector are stored.
        * @return pointer for underlying buffer which contains at least size() elements
        * @sa size()
        * @warning The pointer can be invalidated by resize()
        */
        const T* dataConst() const {
            return components;
        }

        /** Get subvector of vector (*this)
        * @param theStartIndex startIndex of slice
        * @param theCount count number of items in sliced vector                
        * @return constructed vector
        * @remark Please be aware that this routine performs a copy. Don't use it when you need only view of Vector.
        */
        VectorNDRaw get(size_t theStartIndex, size_t theCount) const 
        {
            if (theStartIndex + theCount > size())
            {
                assert(!"Not enough items to get subvector");
                return VectorNDRaw();
            }

            size_t theEndIndex = theStartIndex + theCount;
            VectorNDRaw result(theCount);

            for (size_t j = 0, i = theStartIndex; i < theEndIndex; ++i, ++j)
                result[j] = components[i];

            return result;
        }

        /** Load values from virtual process memory into CPU vector
        * @param valuesCopyFrom values to copy into vector stored in host/cpu/virtual memory
        * @param itemsToCopy number of items to copy
        * @return reference vector
        */
        VectorNDRaw& load(const T* valuesCopyFrom, size_t itemsToCopy)
        {
            assert(componentsCount >= itemsToCopy);
            CopyHelpers::copy(components, valuesCopyFrom, itemsToCopy);
            return *this;
        }

        /** Store(or dump) values from current vector into Virtual memory of current process
        * @param valuesCopyTo storage in which values from vector will be copied into
        * @param itemsToCopy number of items to copy
        * @return this vector
        */
        VectorNDRaw& store(T* valuesCopyTo, size_t itemsToCopy)
        {
            assert(componentsCount >= itemsToCopy);
            CopyHelpers::copy(valuesCopyTo, components, itemsToCopy);
            return *this;
        }
        
        /** Set it-the item to value
        * @param i index of setup
        * @param value for which item should be setuped
        * @return reference to itself
        */
        VectorNDRaw& set(size_t i, T value) {
            components[i] = value;
            return *this;
        }

        /** Set all items of vector to specific value
        * @param value for which all items should be setted
        * @return reference to itself
        */
        VectorNDRaw& setAll(T value) {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] = value;
            return *this;
        }

        /** Set all items of vector to default value
        * @return reference to itself
        */
        VectorNDRaw& setAllToDefault() {
            return setAll(T());
        }

        /** Set all randomly
        * @param generator used pseudo random generator
        * @return reference to itself
        */
        template<class Generator>
        VectorNDRaw& setAllRandomly(Generator& generator)
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                set(i, generator.generateReal());
            return *this;
        }

        /** operator[] to have common syntax access to items of the vector
        * @param index index of item to setup
        * @return reference to item
        */
        T& operator [] (size_t index) {
            return components[index];
        }

        /** operator[] to have common syntax access to items of the vector
        * @param index index of item to setup
        * @return const-reference to item
        */
        const T& operator [] (size_t index) const {
            return components[index];
        }
        
        /** Method provided by API for generality. Idea to compress data storage if possible.                
        * @return 0 when compression have been finished
        */
        size_t compress() {
            return 0;
        }

        /** Check that vector is null vector. Null vector contains all default items
        * @return true if all items have been initialized to default value
        * @remark Pretty slow method with complexity O(d), where d is dimension
        */
        bool isNull() const {
            T nullItem = T();
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                if (get(i) != nullItem)
                    return false;
            }
            return true;
        }

        /** Check that vector item i is null.
        * @param i index of item to check
        * @return true if items i have been initialized to default value
        */
        bool isNull(size_t i) const
        {
            if (i >= size()) [[unlikely]]
                return true;

            T nullItem = T();

            return get(i) == nullItem;
        }
        
        /** Number of non-zero elements in the vector
        * @return number of non-zero elements
        */
        size_t nnz() const {
            size_t result = 0;

            T nullItem = T();
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                if (get(i) != nullItem)
                    result++;
            }
            return result;
        }

        /** Return vector of possibly non-empty indices
        * @return vector of indices
        */
        ::std::vector<size_t> maybeNoEmptyIndicies() const
        {
            ::std::vector<size_t> res;
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
            {
                if (!isNull(i))
                    res.push_back(i);
            }
            return res;
        }

        /** Clean vector via initialize all it's items to default value
        * @return reference to itself
        */
        VectorNDRaw& clean()  
        {
            setAllToDefault();
            return *this;
        }

        /* Unary operator +. Returns hard copy of vector. 
        * @return copy of object
        */
        VectorNDRaw operator + () const
        {
            return *this;
        }

        /* Unary operator -. Returns new vector where all elements has a flipped sign.
        * @return copy of object with all items reverse their sign
        */
        VectorNDRaw operator - () const
        {
            size_t dim = size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i) {
                res.components[i] = -components[i];
            }

            return res;
        }

        /* Append element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        VectorNDRaw& operator += (const VectorNDRaw& v) {
            assert(size() == v.size());
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] += v.components[i];
            return *this;
        }

        /** Add to current vector [this] the muplipler of another vector [other] in a way that:
        *  [this] := this + (multiple) * other
        * @param multiple muplitple of vector to add
        * @param other another vector to add with specific multiplicative factor
        * @remark Compute is in-place
        */
        void addInPlaceVectorWithMultiple(TElementType multiple, const VectorNDRaw& other)
        {
            assert(size() == other.size());
            size_t dim = size();
            
            for (size_t i = 0; i < dim; ++i)
            {
                components[i] += multiple * other.components[i];
            }
        }


        /** Subtract from current vector [this] the muplipler of another vector [other] in a way that:
        *  [this] := this - (multiple) * other
        * @param multiple muplitple of vector subtract
        * @param other another vector to subtract with specific multiplicative factor
        * @remark Compute is in-place
        */
        void subInPlaceVectorWithMultiple(TElementType multiple, const VectorNDRaw& other)
        {
            assert(size() == other.size());
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                components[i] -= multiple * other.components[i];
            }
        }

        /* Remove element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        VectorNDRaw& operator -= (const VectorNDRaw& v) {
            assert(size() == v.size());
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                set(i, get(i) - v.get(i));
            return *this;
        }

        /* Multiply element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        VectorNDRaw& operator *= (const VectorNDRaw& v) {
            assert(size() == v.size());
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] *= v.components[i];
            return *this;
        }

        /* Multiply element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return reference to itself
        */
        template<typename TFactorType>
        VectorNDRaw& operator *= (TFactorType factor) 
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                getRaw(i) *= factor;
            
            return *this;
        }

        /* Multiply element wise elements of the vector to specified factor which is real value in fp64 format
        * @param factor specified factor
        * @return reference to itself
        */
        VectorNDRaw& operator *= (double factor) 
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                getRaw(i) *= factor;
            
            return *this;
        }
        
        /* Divide element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return reference to itself
        */
        template<typename TFactorType>
        VectorNDRaw& operator /= (TFactorType factor) 
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                getRaw(i) /= factor;
            
            return *this;
        }
        
        /* Divide element wise elements of the vector to specified factor which is real value in fp64 format
        * @param factor specified factor
        * @return reference to itself
        */
        VectorNDRaw& operator /= (double factor) 
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                getRaw(i) /= factor;
            
            return *this;
        }
        
        /* Add two vectors element wise and return new vector
        * @param rhs other vector
        * @return vector with result
        */
        VectorNDRaw<T> operator + (const VectorNDRaw<T>& rhs) const
        {
            VectorNDRaw<T> res(*this);
            res += rhs;
            return res;
        }

        /*  Evaluate (a-b) for two vectors
        * @param rhs other vector
        * @return copy of vector with result
        */
        VectorNDRaw<T> operator - (const VectorNDRaw<T>& rhs) const
        {
            VectorNDRaw<T> res(*this);
            res -= rhs;
            return res;
        }

        /*  Evaluate (a[1,...,1] - rhs) * multipler
        * @param a scalar which is used to scale vector consisted elementwise of all "1"
        * @param rhs other vector
        * @param multiplier final multiplier
        */
        static VectorNDRaw scaledDifferenceWithEye(TElementType a, const VectorNDRaw& rhs, TElementType multiplier)
        {
            size_t dim = rhs.size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i) {
                res.components[i] = (a - rhs.components[i]) * multiplier;
            }
            return res;
        }

        /* Divide element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return copy of vector with result
        */
        template <typename TFactorType>
        VectorNDRaw<T> operator / (TFactorType factor) const {
            VectorNDRaw<T> res(*this);
            res /= factor;
            return res;
        }

        /* Multiply element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return copy of vector with result
        */
        template <typename TFactorType>
        VectorNDRaw<T> operator * (TFactorType factor) const {
            VectorNDRaw<T> res(*this);
            res *= factor;
            return res;
        }
        
        /* Multiply element wise elements of the vector to specified factor
        * @param factor specified factor
        * @return copy of vector with result
        */
        VectorNDRaw<T> operator * (double factor) const {
            VectorNDRaw<T> res(*this);
            res *= factor;
            return res;
        }

        /* Evaluate do product on two vectors
        * @param rhs other vector
        * @return result dot product
        */  
        T operator & (const VectorNDRaw& rhs) const {
            assert(size() == rhs.size());
            size_t dim = size();
            T res = T();
            for (size_t i = 0; i < dim; ++i)
            {
                res += get(i) * rhs.get(i);
            }
            return res;
        }

        /* Evaluate do product on slice of two vectors which correspond to indices [0:n)
        * @param a first vector
        * @param b second vector
        * @param start start index for making slice of vector "a" and vector "b"
        * @param end end index for making slice of vector "a" and vector "b"
        * @return result dot product
        */
        static T reducedDotProduct(const VectorNDRaw& a, const VectorNDRaw& b, size_t start, size_t end) {
            assert(start < a.size());
            assert(start < b.size());
            assert(end <= a.size());
            assert(end <= b.size());

            T res = T();
            for (size_t i = start; i < end; ++i)
                res += a.get(i) * b.get(i);
            return res;
        }

        /* Evaluate L2 norm square of vector
        * @return result
        */
        T vectorL2NormSquare() const {
            size_t dim = size();
            T res = T();
            for (size_t i = 0; i < dim; ++i)
                res += get(i) * get(i);
            return res;
        }

        /* Evaluate L2 norm of vector
        * @return result
        */
        T vectorL2Norm() const {
            return ::sqrt(vectorL2NormSquare());
        }
        
        /* Evaluate LP norm to the power of P
        * @return result
        */
        T vectorLPNormPowerP(uint32_t p) const {
            size_t dim = size();
            T res = T();
            for (size_t i = 0; i < dim; ++i)
                res += burt::powerNatural(burt::abs(get(i)), p);
            return res;
        }

        /* Evaluate LP norm to the power of P
        * @return result
        */
        T vectorLPNorm(uint32_t p) const {
            size_t dim = size();
            T res = burt::powerReal( vectorLPNormPowerP(p), 1.0 / double(p) );

            return res;
        }

        /* Evaluate L1 norm of vector
        * @return result
        */
        T vectorL1Norm() const {
            size_t dim = size();
            T res = T();
            for (size_t i = 0; i < dim; ++i)
                res += burt::abs(get(i));
            return res;
        }

        /* Evaluate Linf norm of vector
        * @return result
        */
        T vectorLinfNorm() const 
        {
            size_t dim = size();

            if (dim == 0) [[unlikely]]
                return T();

            T res = burt::abs(get(0));

            for (size_t i = 1; i < dim; ++i)
            {
                T item = burt::abs(get(i));
                if (item > res)
                    res = item;
            }
            return res;
        }

        /* Check if two vectors are equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator == (const VectorNDRaw& v) const {
            assert(size() == v.size());

            if (size() != v.size()) [[unlikely]]
                return false;

            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
                if (get(i) != v.get(i))
                    return false;

            return true;
        }

        /* Check if two vectors are not equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator != (const VectorNDRaw& v) const {
            return !(*this == v);
        }
        
        /* Obtain raw pointer to underlying data
        * @return raw pointer
        */
        T* rawData() {
            if (componentsCount == 0) [[unlikely]]
                return nullptr;
            else
                return &components[0];
        }

        /* Obtain raw const pointer to underlying data
        * @return raw pointer
        */
        const T* rawData() const {
            if (componentsCount == 0) [[unlikely]]
                return nullptr;
            else
                return &components[0];
        }

        /** Clamp each item into segment [lower, upper]
        * @param lower lower bound of interval
        * @param upper upper bound of interval
        * @return vector with results
        */
        VectorNDRaw& clamp(const VectorNDRaw& lower, const VectorNDRaw& upper)
        {
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                const T& ai = get(i);

                if (ai <= lower[i])
                    set(i, lower[i]);
                else if (ai >= upper[i])
                    set(i, upper[i]);
            }
            return *this;
        }

        /** Clamp each item into segment [lower, upper]
        * @param lower lower bound of interval
        * @param upper upper bound of interval
        * @return vector with results
        */
        VectorNDRaw& clamp(const T lower, const T upper)
        {
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                const T& ai = get(i);

                if (ai <= lower)
                    set(i, lower);
                else if (ai >= upper)
                    set(i, upper);
            }

            return *this;
        }

        /** Make items with values [- eps, + eps] make them just "zero"
        * @param eps
        * @return reference to this
        */
        VectorNDRaw& zeroOutItems(T eps)
        {
            size_t dim = size();
            T zero = T();
            T minusEps = -eps;

            for (size_t i = 0; i < dim; ++i)
            {
                const T& ai = get(i);
                if (ai >= minusEps && ai <= eps)
                {
                    set(i, zero);
                }
            }

            return *this;
        }

        /** Apply element wise function exp() to elements of the vector
        * @return vector with results
        */
        VectorNDRaw exp() const
        {
            size_t dim = size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i)
                res.set(i, ::exp(get(i)));
            return res;
        }

        /** Apply element wise function log() which is natural logarithm to elements of the vector
        * @return vector with results
        */
        VectorNDRaw log()
        {
            size_t dim = size();
            VectorNDRaw res(dim);

            for (size_t i = 0; i < dim; ++i)
                res.set(i, ::log(get(i)));
            return res;
        }
        
        /** Apply element wise function inv(x)=1/x
        * @return vector with results
        */
        VectorNDRaw inv()
        {
            size_t dim = size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i)
                res.set(i, T(1) / (get(i)));
            return res;
        }

        /** Apply element wise square function
        * @return vector with results
        */
        VectorNDRaw square()
        {
            size_t dim = size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i)
            {
                const T& item = get(i);
                res.set(i, item * item);
            }
            return res;
        }

        /** Apply element wise square root function
        * @return vector with results
        */
        VectorNDRaw sqrt()
        {
            size_t dim = size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i)
            {
                const T& item = get(i);
                res.set(i, ::sqrt(item));
            }
            return res;
        }

        /** Apply element wise function inv(x)=1/(x*x)
        * @return vector with results
        */
        VectorNDRaw invSquare()
        {
            size_t dim = size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i)
            {
                auto xi = get(i);
                res.set(i, T(1) / (xi*xi));
            }
            return res;
        }

        /** Concatenate two vectors a and b => [a, b]
        * @param a first vector
        * @param b second vector
        * @return result vector
        */
        static VectorNDRaw concat(const VectorNDRaw& a, const VectorNDRaw& b)
        {
            size_t asize = a.size();
            size_t bsize = b.size();

            VectorNDRaw<T> res(asize + bsize);
            for (size_t i = 0; i < asize; ++i)
                res[i] = a[i];

            for (size_t i = 0; i < bsize; ++i)
                res[asize + i] = b[i];

            return res;
        }

        /** Interpretate vector as vector which consist of classification margins. I.e. element wise it contains -b_i <a_i, x>
        * @return value of (unweghted) logistic loss
        */
        T logisticLoss() const {
            return logisticUnweightedLossFromMargin<T>(*this);
        }

        /** Compute element wise sigmoid(x_i), where sigmoid(x)=1.0/(1.0 + e^(-x))
        * @return value of (unweghted) logistic loss
        */
        VectorNDRaw elementwiseSigmoid() const
        {
            size_t dim = size();
            VectorNDRaw res(dim);
            for (size_t i = 0; i < dim; ++i)
                res.set(i, ( T(1) / (T(1) + ::exp(-get(i)))) );
            return res;
        }

        /** Compute approximate version of element wise sigmoid(x_i), where sigmoid(x)=1.0/(1.0 + e^(-x))
        * @return value of (unweghted) logistic loss
        * @remark this method designed via Taylor expansio of exp(-x) near zero and it uses symmetry of original logistic loss
        */
        VectorNDRaw elementwiseSigmoidApproximate() const
        {
            TElementType k_x0 = TElementType(0);
            TElementType k_exp_minus_x0 = ::exp(k_x0);
            TElementType k_minus_exp_minus_x0 = -k_exp_minus_x0;

            size_t dim = size();
            VectorNDRaw res(dim);
            
            size_t kSeries = 10;
            
            for (size_t i = 0; i < dim; ++i)
            {
                TElementType xi = get(i);
                
                if (xi < 0)
                {
                    TElementType dh_step = (xi - k_x0);                    
                    TElementType dh_step_sqr = dh_step * dh_step;

                    TElementType dh_pow_1 = TElementType(1); // div 0!
                    TElementType dh_pow_2 = dh_step;         // div 1!
                    
                    TElementType numerator = TElementType(1);
                    TElementType denomonitor_exp_minux_x_part = (k_exp_minus_x0 * dh_pow_1) + (k_minus_exp_minus_x0) * dh_pow_2;
                    
                    for (size_t k = 2; k < kSeries; k += 2)
                    {
                        dh_pow_1 *= dh_step_sqr / (TElementType(k) * TElementType(k - 1));
                        dh_pow_2 *= dh_step_sqr / (TElementType(k + 1) * TElementType(k));
                        
                        denomonitor_exp_minux_x_part += (k_exp_minus_x0 * dh_pow_1) + (k_minus_exp_minus_x0 * dh_pow_2);
                    }
                    
                    TElementType compute = numerator / (TElementType(1) + denomonitor_exp_minux_x_part);
                    res.set(i, compute);
                }                
                else
                {
                    xi = -xi;

                    TElementType dh_step = (xi - k_x0);
                    TElementType dh_step_sqr = dh_step * dh_step;

                    TElementType dh_pow_1 = TElementType(1); // div 0!
                    TElementType dh_pow_2 = dh_step;         // div 1!

                    TElementType numerator = TElementType(1);
                    TElementType denomonitor_exp_minux_x_part = (k_exp_minus_x0 * dh_pow_1) + (k_minus_exp_minus_x0)*dh_pow_2;

                    for (size_t k = 2; k < kSeries; k += 2)
                    {
                        dh_pow_1 *= dh_step_sqr / (TElementType(k) * TElementType(k - 1));
                        dh_pow_2 *= dh_step_sqr / (TElementType(k + 1) * TElementType(k));

                        denomonitor_exp_minux_x_part += (k_exp_minus_x0 * dh_pow_1) + (k_minus_exp_minus_x0 * dh_pow_2);
                    }

                    TElementType compute = numerator / (TElementType(1) + denomonitor_exp_minux_x_part);

                    res.set(i, TElementType(1) - compute);
                }
            }
            
            return res;
        }

        /** Compute: (a) difference between current vector and rhs; (b) L2 norm of the result
        * @param rhs vector from which difference is computed
        * @param l2NormOfDifference reference to value which will store the result L2 norm of the result
        * @remark Function is presented for compute optimization
        */
        void computeDiffAndComputeL2Norm(const VectorNDRaw& rhs, TElementType& restrict_ext l2NormOfDifference) {
            *this -= rhs;
            l2NormOfDifference = vectorL2Norm();            
        }
        
    protected:

        enum InitPolicyForStorage {
            eNotAllocate = 0,        ///< Warning: Not allocate underlying storage. Use it if you understand what you are doing.
            eAllocNotInit = 1,       ///< Warning: Allocate underlying storage, but not initialized. Use it if you understand what you are doing.
            eAllocAndSetToZero = 2   ///< Allocate underlying storage and initialize to zero
        };
        
        /** Construct vector with using specific initialization policy
        * @param policy initialization policy
        * @remark Use only if you understand what you're doing.
        * @remark Be very careful with uninitilized vectors
        * @remark For std::vector it's tricky to create std::vector with uninitialized values
        */
        VectorNDRaw(InitPolicyForStorage policy, size_t dimension)
        : components(nullptr)
        , componentsCount(dimension)
        {
            switch (policy)
            {
                case eNotAllocate:
                {
                    components = nullptr;
                    break;
                }
                
                case eAllocNotInit:
                {
                    // default initialized values (i.e. nothing happens except allocation) -- https://en.cppreference.com/w/cpp/language/default_initialization
                    // one more time -- default initialization does not perform any initialization
                    components = new (memAllocation(components2bytes(dimension))) T[dimension];
                    break;
                }

                case eAllocAndSetToZero:
                {
                    // value-initialized values
                    //   - If T is a class type with a default constructor is called.
                    //   - Otherwise, the object is zero-initialized. (i.e. set to 0)

                    components = new (memAllocation(components2bytes(dimension))) T[dimension] ();
                    break;
                }
            }
        }
    };

    template <typename T>
    const TMemoryAllocationCallback VectorNDRaw<T>::memAllocation = memAllocationCRT;
    template <typename T>
    const TMemoryFreeCallback VectorNDRaw<T>::memFree = memFreeCRT;

    /** Helper function when factor multiplied by vector from the right
    * @param factor factor for multiply
    * @param v vector which will be multiplied by factor
    * @return result vector
    */
    template <typename T, typename TFactorType>
    inline VectorNDRaw<T> operator * (TFactorType factor, const VectorNDRaw<T>& v) {
        return v * factor;
    }

    using VectorNDRaw_ui64 = VectorNDRaw<uint64_t>;
    using VectorNDRaw_i64 = VectorNDRaw<int64_t>;
    
    using VectorNDRaw_ui = VectorNDRaw<uint32_t>;
    using VectorNDRaw_i = VectorNDRaw<int32_t>;
    
    using VectorNDRaw_f = VectorNDRaw<float>;
    using VectorNDRaw_d = VectorNDRaw<double>;
    using VectorNDRaw_b = VectorNDRaw<bool>;
}

#if BURT_INCLUDE_VECTORIZED_CPU_IMP_VECS
    #include "burt/linalg_vectors/include_internal/VectorND_Raw_SIMD_int.h"
    #include "burt/linalg_vectors/include_internal/VectorND_Raw_SIMD_double.h"
    #include "burt/linalg_vectors/include_internal/VectorND_Raw_SIMD_float.h"
#endif
