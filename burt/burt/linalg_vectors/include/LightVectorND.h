#pragma once

#include "burt/copylocal/include/Copier.h"
#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include "burt/system/include/threads/Thread.h"

#include <vector>
#include <sstream>

#include <math.h>
#include <assert.h>
#include <stddef.h>

namespace burt
{
    /* Append argument to memory refereed by dst. To it in atomic way if size of T is 1,2,4,8 bytes.
    * @param dst destination
    * @param argument argument which we add to dst
    * @remark update happens in atomic way for T sizeof 1,2,4,8 bytes. And backuped in another case with slow mutex.
    */
    template <class T>
    forceinline_ext void appendMT(volatile T& restrict_ext dst, T argument)
    {
        T old_value_expected, new_value_desired;
        do
        {
            old_value_expected = dst;
            new_value_desired = old_value_expected + argument;
        } while (old_value_expected != dst || !burt::internal::myCAS(&dst, old_value_expected, new_value_desired));
    }
    
    /** Light vector view. It's a view into some elements of the linear vector.
    * @tparam T Type of elements inside vector
    * @remark The vector does not copy any elements. Please use it only if you're really understand what you're doing.
    */
    template <typename Vec>
    class LightVectorND
    {
    public:
        using TElementType = typename Vec::TElementType;   ///< Typedef for elements types
        
        size_t componentsCount;                            ///< Components count inside vector view
        TElementType* components;                          ///< Components of the vector

        /** Ctor. Create light view into parentVector[theStartIndex:theStartIndex+theCount]
        * @param parentVector vector some items of which is used to create "view" into the data.
        * @param theStartIndex index from which items are considered
        * @param theCount number of items which should be in LightVectorND
        */
        LightVectorND(Vec& parentVector, size_t theStartIndex, size_t theCount)
        : components(parentVector.data() + theStartIndex)
        , componentsCount(theCount)
        {}

        /** Ctor. Create light view into parentVector[theStartIndex:to the end]
        * @param parentVector vector some items of which is used to create "view" into the data.
        * @param theStartIndex index from which items are considered
        */
        LightVectorND(Vec& parentVector, size_t theStartIndex)
        : components(parentVector.data() + theStartIndex)
        , componentsCount(0)
        {
            size_t parentVectorSize = parentVector.size();

            if (theStartIndex >= parentVectorSize) [[unlikely]]
            {
                components = nullptr;
                componentsCount = 0;
            }
            else
            {
                componentsCount = parentVectorSize - theStartIndex;
            }
        }

        /** Ctor. Create light view into parentVector[theStartIndex:theStartIndex+theCount]
        * @param parentVector vector some items of which is used to create "view" into the data.
        * @param theStartIndex index from which items are considered
        * @param theCount number of items which should be in LightVectorND
        * @reamrk Using the complete(full) template identifier within a template definition scope is not essential.
        */
        LightVectorND(LightVectorND& rhs, size_t theStartIndex, size_t theCount)
        : components(rhs.components + theStartIndex)
        , componentsCount(theCount)
        {}

        /** Ctor. Create light view into parentVector[theStartIndex:to the end]
        * @param parentVector vector some items of which is used to create "view" into the data.
        * @param theStartIndex index from which items are considered
        */
        LightVectorND(LightVectorND& parentVector, size_t theStartIndex)
        : components(parentVector.data() + theStartIndex)
        , componentsCount(0)
        {
            size_t parentVectorSize = parentVector.size();

            if (theStartIndex >= parentVectorSize) [[unlikely]]
            {
                components = nullptr;
                componentsCount = 0;
            }
            else
            {
                componentsCount = parentVectorSize - theStartIndex;
            }
        }

        /** Create vector from some arbitarily buffer in memory
        * @param theComponents beginning of buffer of elements in memory
        * @param theComponentsCount number of components
        */
        LightVectorND(TElementType* theComponents, size_t theComponentsCount)
        : components(theComponents)
        , componentsCount(theComponentsCount)
        {}

        /** Create empty vector
        */
        LightVectorND()
        : components(nullptr)
        , componentsCount(0)
        {}

        /** Debug print with debug representation of the vector
        * @param out output steam for which operator << (const char* str) and  operator << (const T&) is defined.
        * @param variableName used variable name
        * @param delimiter used delimiter during printing values
        */
        template<class text_out_steam>
        void dbgPrintInMatlabStyle(text_out_steam& out, const char* variableName  = "x=", const char* delimiter = ",") const
        {
            out << variableName << "[";
            for (size_t i = 0; i < componentsCount; ++i)
            {
                if (i != 0)
                    out << delimiter;
                out << components[i];
            }
            out << "];\n";
        }

        ~LightVectorND()
        {
            components = nullptr;
            componentsCount = 0;
        }

        /** Get raw pointer for underlying buffer in which elements of dense vector are stored.
        * @return pointer for underlying buffer which contains at least size() elements
        * @sa size()
        */
        TElementType* data() {
            return components;
        }

        /** Get raw pointer for underlying buffer in which elements of dense vector are stored.
        * @return pointer for underlying buffer which contains at least size() elements
        * @sa size()
        */
        const TElementType* dataConst() const {
            return components;
        }

        /** Dump vector (which is almost in all applied math is column matrix with one column) by rows
        * @param outMemory pointer to memory in which content of the vector will be dumped
        * @param startRow first row. The first row is included into output.
        * @param endRow last row. The end row is included into output.
        */
        void dumpByRows(TElementType* outMemory, size_t startRow, size_t endRow) const
        {
            for (size_t i = startRow; i <= endRow; ++i)
                *(outMemory++) = get(i);
        }

        /** Dump all light vector (which is almost in all applied math is column matrix with one column) by rows
        * @param out pointer to memory in which content of the vector will be dumped
        */
        void dumpByRows(TElementType* outMemory) const
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                *(outMemory++) = get(i);
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
        * @param other other vector
        */
        void swap(LightVectorND& other) noexcept {
            if (this != &other)
            {
                burt::CopyHelpers::swapDifferentObjects(&components, &other.components);
                burt::CopyHelpers::swapDifferentObjects(&componentsCount, &other.componentsCount);
            }
        }

        /** Size of the vector in terms of number of items in it
        * @return size of the vector
        */
        size_t size() const {
            return componentsCount;
        }

        /** Memory reserved for this vector
        * @return real number of elements reserved for the vector content
        */
        size_t capacity() const {
            return componentsCount;
        }

        /** Evaluate sum of all elements
        * @return result sum of all elements.
        * @remark Implementation contains naive way. 
        * @reamrk More robust way in terms of numeric is firstly sort array, or use numerical stable Kahan summation algorithm.
        */
        template <class TAccumulator = typename Vec::TElementType>
        TAccumulator sum() const
        {
            TAccumulator res = TAccumulator();
            size_t sz = size();
            for (size_t i = 0; i < sz; ++i)
                res += components[i];
            return res;
        }

        /** Get const reference item by index i
        * @param i index of item
        * @return reference to item offset by "startIndex + i"
        * @remark For DEBUG build in case if index is out of range the provided reference is a reference to dummy variable
        * @remark If argument is out of range behaviour is undefined
        */
        const TElementType get(size_t i) const  {
            #if BURT_DEBUG_BUILD
            if (i >= componentsCount)
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
            #endif

            const TElementType result = components[i];
            return result;
        }

        /** Create hard copy of vector just create another complete new vector from current view
        * @param result destination vector
        * @return number of elements in a new vector
        */
        template <class VecOther>
        size_t hardcopy(VecOther& result) const
        {
            result = Vec(componentsCount);
            for (size_t i = 0; i < componentsCount; ++i)
                result.set(i, get(i));

            return componentsCount;
        }

        /** Get non-const reference item by index i
        * @param i index of item
        * @return reference to item offset by "startIndex + i"
        * @remark For DEBUG build in case if index is out of range the provided reference is a reference to dummy variable
        */
        TElementType& getRaw(size_t i) {
            #if BURT_DEBUG_BUILD
            if (i >= componentsCount)
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
            #endif

            TElementType& result = components[i];
            return result;
        }

        /** Get sub vector of vector (*this)
        * @param theStartIndex startIndex of slice
        * @param theCount count number of items in sliced vector
        * @return constructed vector        
        * @remark For DEBUG build in case if index is out of range the provided reference is a reference to dummy variable
        */
        LightVectorND get(size_t theStartIndex, size_t theCount) const
        {
            #if BURT_DEBUG_BUILD
            if (theStartIndex + theCount > size())
            {
                assert(!"Not enough items to get subvector");
                return LightVectorND();
            }
            #endif

            LightVectorND result;
            result.components = components + theStartIndex;
            result.componentsCount = theCount;

            return result;
        }

        /** Set it-the item to value
        * @param i index of setup
        * @param value for which item should be setup
        * @return reference to itself        
        * @remark For DEBUG build in case if index is out of range assertion is raised and nothing happens
        */
        LightVectorND& set(size_t i, TElementType value) {
            #if BURT_DEBUG_BUILD
            if (i >= componentsCount)
            {
                assert(!"Not correct index for set value!");
                return *this;
            }
            #endif

            components[i] = value;
            return *this;
        }

        /** Set all items of vector to specific value
        * @param value for which all items should be setted
        * @return reference to itself
        */
        LightVectorND& setAll(TElementType value)
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] = value;
            return *this;
        }

        /** Set all items of vector to default value
        * @return reference to itself
        */
        LightVectorND& setAllToDefault() {
            return setAll(TElementType());
        }

        /** Set all randomly
        * @param generator used pseudo random generator
        * @return reference to itself
        */
        template<class Generator>
        LightVectorND& setAllRandomly(Generator& generator)
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                set(i, generator.generateReal());
            return *this;
        }

        /** Make items with values [- eps, + eps] make them just "zero"
        * @param eps
        * @return reference to this
        */
        LightVectorND& zeroOutItems(TElementType eps)
        {
            size_t dim = size();
            TElementType zero = TElementType();
            TElementType minusEps = -eps;

            for (size_t i = 0; i < dim; ++i)
            {
                const TElementType& ai = components[i];
                if (ai >= minusEps && ai <= eps)
                {
                    components[i] = zero;
                }
            }

            return *this;
        }

        /** operator[] to have common syntax access to items of the vector
        * @param index index of item to setup
        * @return reference to item
        * @remark For DEBUG build in case if index is out of range assertion is raised and reference to dummy variable is provided
        */
        TElementType& operator [] (size_t index) {
            #if BURT_DEBUG_BUILD
            if (index >= componentsCount || index < 0)
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
            #endif

            return components[index];
        }

        /** operator[] to have common syntax access to items of the vector
        * @param index index of item to setup
        * @return const-reference to item
        * @remark For DEBUG build in case if index is out of range assertion is raised and reference to dummy variable is provided
        */
        const TElementType& operator [] (size_t index) const {
            #if BURT_DEBUG_BUILD
            if (index >= size())
            {
                assert(!"Not correct index for set value!");
                static TElementType dummy;
                return dummy;
            }
            #endif

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
        */
        bool isNull() const {
            TElementType nullItem = TElementType();
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
            if (i >= size())
                return true;

            TElementType nullItem = TElementType();

            return get(i) == nullItem;            
        }

        /** Number of non-zero elements in the vector
        * @return number of non-zero elements
        */
        size_t nnz() const
        {
            size_t result = 0;

            TElementType nullItem = TElementType();
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                if (get(i) != nullItem)
                    result++;
            }
            return result;
        }

        /** Return vector with indices of item which possibly are non-zero
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
        LightVectorND& clean()
        {
            setAllToDefault();
            return *this;
        }

        /* Unary operator+
        * @return copy of object
        */
        LightVectorND operator + () const
        {
            return *this;
        }

        /* Append element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        LightVectorND& operator += (const LightVectorND& v) {
            assert(size() == v.size());
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] += v.components[i];
            return *this;
        }

        /** Add to current vector [view] the muplipler of another vector [view] in a way that:
        *  [this] := this + (multiple) * other
        * @param multiple muplitple of vector view to add
        * @param other another vector view to add with specific multiplicative factor
        * @remark Compute is in-place
        */
        LightVectorND& addInPlaceVectorWithMultiple(TElementType multiple, const LightVectorND& v) {
            assert(size() == v.size());
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] += v.components[i] * multiple;
            return *this;
        }

        /** Subtract from current vector [this] the muplipler of another vector [other] in a way that:
        *  [this] := this - (multiple) * other
        * @param multiple muplitple of vector subtract
        * @param other another vector view to subtract with specific multiplicative factor
        * @remark Compute is in-place
        */
        LightVectorND& subInPlaceVectorWithMultiple(TElementType multiple, const LightVectorND& v) {
            assert(size() == v.size());
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] -= v.components[i] * multiple;
            return *this;
        }

        TElementType subInPlaceVectorWithMultipleAndReportL2NormSqr(TElementType multiple, const LightVectorND& v)
        {
            assert(size() == v.size());
            TElementType res = TElementType();

            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
            {
                res += v.components[i] * v.components[i];

                components[i] -= v.components[i] * multiple;
            }
            return res;
        }

        /* Append element wise other items of vector v thread safely
        * @param v other vector
        * @return reference to itself
        * @remark non-blocking implementation
        */
        LightVectorND& addAnotherVectrorNonBlockingMulthithread(const LightVectorND& v)
        {
            assert(size() == v.size());
            size_t dim = size();
            
            volatile TElementType* pDst = components;
            const TElementType* pSrc = v.components;

            for (size_t i = 0; i < dim; ++i)
                appendMT(pDst[i], pSrc[i]);            
            
            return *this;
        }

        /* Append element wise other items of vector v thread safely
        * @param v other vector
        * @return reference to itself
        * @remark non-blocking implementation
        * @tparam highContentionIsPossible - please specify this into the true, if you know that a lot of threads are going to make update on the same memory
        */
        template<bool highContentionIsPossible = true>
        LightVectorND& multiplyByScalarNonBlockingMulthithread(const TElementType multiplier)
        {
            size_t dim = size();

            volatile TElementType* pDst = components;

            TElementType old_value_expected, new_value_desired, read_value;

            if constexpr (highContentionIsPossible)
            {
                for (size_t i = 0; i < dim; ++i)
                {
                    read_value = pDst[i];
                    do
                    {
                        old_value_expected = pDst[i];
                        new_value_desired = old_value_expected * multiplier;

                        // Compare and Swap acquire cache line in exclusive mode, invalidating another caches. 
                        // Unofrtunately it leads to high contention if all processors doing CAS at the same place
                        // To reduce this effect it's better to waste time on reading value from memory and do CAS only if it's necessary.

                    } while (old_value_expected != pDst[i] ||
                             !burt::internal::myCAS(&pDst[i], old_value_expected, new_value_desired));
                }
            }
            else
            {
                for (size_t i = 0; i < dim; ++i)
                {
                    read_value = pDst[i];
                    do
                    {
                        old_value_expected = pDst[i];
                        new_value_desired = old_value_expected * multiplier;

                    } while (! burt::internal::myCAS(&pDst[i], old_value_expected, new_value_desired) );
                }
            }
            return *this;
        }

        /* Remove element wise other items of vector v
        * @param v other vector
        * @return reference to itself
        */
        LightVectorND& operator -= (const LightVectorND& v) {
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
        LightVectorND& operator *= (const LightVectorND& v) {
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
        LightVectorND& operator *= (TFactorType factor)
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
        LightVectorND& operator *= (double factor)
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
        LightVectorND& operator /= (TFactorType factor)
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
        LightVectorND& operator /= (double factor)
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                getRaw(i) /= factor;
            return *this;
        }

        /* Evaluate dot product on two vectors
        * @param rhs other vector
        * @return result dot product
        */
        TElementType operator & (const LightVectorND& rhs) const
        {
            assert(size() == rhs.size());
            size_t dim = size();
            TElementType res = TElementType();
            for (size_t i = 0; i < dim; ++i)
            {
                res += get(i) * rhs.get(i);
            }
            return res;
        }

        /* Evaluate dot product on two vectors for which there is a garantee of propery memory alignment of the start
        * @param rhs other vector
        * @return result dot product
        */
        TElementType dotProductForAlignedMemory(const LightVectorND& rhs) const
        {
            assert(size() == rhs.size());
            size_t dim = size();
            TElementType res = TElementType();
            for (size_t i = 0; i < dim; ++i)
            {
                res += get(i) * rhs.get(i);
            }
            return res;
        }

        /* Evaluate dot product on slice of two vectors which correspond to indices [start:end)
        * @param a first vector
        * @param b second vector
        * @param start start index for making slice of vector "a" and vector "b"
        * @param end end index for making slice of vector "a" and vector "b"
        * @return result dot product
        */
        static TElementType reducedDotProduct(const LightVectorND& a, const LightVectorND& b, size_t start, size_t end)
        {
            assert(start < a.size());
            assert(start < b.size());
            assert(end <= a.size());
            assert(end <= b.size());

            TElementType res = TElementType();
            for (size_t i = start; i < end; ++i)
                res += a.get(i) * b.get(i);
            return res;
        }

        /* Evaluate L2 norm square of vector
        * @return result
        */
        TElementType vectorL2NormSquare() const
        {
            size_t dim = size();
            TElementType res = TElementType();
            for (size_t i = 0; i < dim; ++i)
                res += get(i) * get(i);
            return res;
        }

        /* Evaluate L2 norm square of vector
        * @return result
        */
        TElementType vectorL2NormSquareForAlignedMemory() const
        {
            size_t dim = size();
            TElementType res = TElementType();
            for (size_t i = 0; i < dim; ++i)
                res += get(i) * get(i);
            return res;
        }

        /* Evaluate L2 norm of vector
        * @return result
        */
        TElementType vectorL2Norm() const {
            return ::sqrt(vectorL2NormSquare());
        }

        /* Evaluate L1 norm of vector
        * @return result
        */
        TElementType vectorL1Norm() const {
            size_t dim = size();
            TElementType res = TElementType();
            for (size_t i = 0; i < dim; ++i)
                res += burt::abs(get(i));
            return res;
        }

        /* Evaluate Linf norm of vector
        * @return result
        */
        TElementType vectorLinfNorm() const 
        {
            size_t dim = size();

            if (dim == 0)
                return TElementType();

            TElementType res = burt::abs(get(0));

            for (size_t i = 1; i < dim; ++i)
            {
                TElementType item = burt::abs(get(i));
                if (item > res)
                    res = item;
            }
            return res;
        }

        /* Check if two vectors are equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator == (const LightVectorND& v) const
        {
            size_t dim = size();

            if (dim != v.size())
                return false;

            for (size_t i = 0; i < dim; ++i)
                if (get(i) != v.get(i))
                    return false;

            return true;
        }

        /* Check if two vectors are not equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator != (const LightVectorND& v) const {
            return !(*this == v);
        }

        /* Obtain raw pointer to underlying data
        * @return raw pointer
        */
        TElementType* rawData() {
            if (componentsCount == 0)
                return nullptr;
            else
                return components;
        }

        /* Obtain raw const pointer to underlying data
        * @return raw pointer
        */
        const TElementType* rawData() const {
            if (componentsCount == 0)
                return nullptr;
            else
                return components;
        }

        /* Check if two vectors are equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator == (const Vec& v) const
        {
            size_t dim = size();

            if (dim != v.size())
                return false;

            for (size_t i = 0; i < dim; ++i)
                if ( get(i) != v.get(i) )
                    return false;

            return true;
        }

        /** Concatenate two vectors a and b => [a, b]
        * @param a first vector
        * @param b second vector
        * @return result vector
        */
        static Vec concat(const LightVectorND& a, const LightVectorND& b)
        {
            size_t asize = a.size();
            size_t bsize = b.size();

            Vec res(asize + bsize);
            for (size_t i = 0; i < asize; ++i)
                res.getRaw(i) = a[i];

            for (size_t i = 0; i < bsize; ++i)
                res.getRaw(asize + i) = b[i];

            return res;
        }

        /* Check if two vectors are not equal by elements values
        * @param v other vector with which comparison is occurring
        * @return true if vectors are equal
        */
        bool operator != (const Vec& v) const {
            return !(*this == v);
        }

        /** Clamp each item into segment [lower, upper]
        * @param lower lower bound of interval
        * @param upper upper bound of interval
        * @return reference to *this
        */
        LightVectorND& clamp(const LightVectorND& lower, const LightVectorND& upper)
        {
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                const TElementType& ai = get(i);

                if ( ai <= lower[i] )
                    set(i, lower[i]);
                else if ( ai >= upper[i] )
                    set(i, upper[i]);
            }
            return *this;
        }

        /** Clamp each item into segment [lower, upper]
        * @param lower lower bound of interval
        * @param upper upper bound of interval
        * @return reference to *this
        */
        LightVectorND& clamp(const TElementType lower, const TElementType upper)
        {
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
            {
                const TElementType& ai = get(i);

                if (ai <= lower)
                    set(i, lower);
                else if (ai >= upper)
                    set(i, upper);
            }

            return *this;
        }

        /** For current vector view add vector "v" with multiple "multiplier"
        * @param v vector to be added
        * @param multiplier the multiplication factor near vector "v" which should be added
        * @return reference to *this
        */
        LightVectorND& addWithVectorMultiple(const LightVectorND& v, const TElementType multiplier)
        {
            assert(size() == v.size());
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] += v.components[i] * multiplier;
            return *this;
        }
        
        /** For current vector view assign vector "v" with multiple "multiplier"
        * @param v vector to be added
        * @param multiplier the multiplication factor near vector "v" which should be added
        * @return reference to *this
        */
        // template <TElementType multiplier>
        LightVectorND& assignWithVectorMultiple(const LightVectorND& v, TElementType multiplier)
        {
            assert(size() == v.size());
            size_t dim = size();

            for (size_t i = 0; i < dim; ++i)
                components[i] = v.components[i] * multiplier;
            
            return *this;
        }

        /** For current vector view assign value of vector "a" - "b" 
        * @param a first vector
        * @param b second vector
        * @return reference to *this
        */
        LightVectorND& assignWithVectorDifference(const LightVectorND& a, const LightVectorND& b)
        {
            assert(size() == a.size());
            assert(size() == b.size());

            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
            {
                components[i] = a.components[i] - b.components[i];
            }
            return *this;
        }

        /** For current vector view assign value of vector "a" - "b"
        * @param a first vector
        * @param b second vector
        * @return reference to *this
        */
        LightVectorND& assignWithVectorDifferenceAligned(const LightVectorND& a, const LightVectorND& b)
        {
            assert(size() == a.size());
            assert(size() == b.size());

            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
            {
                components[i] = a.components[i] - b.components[i];
            }
            return *this;
        }

        /* Copy values from one light view array "r" to current.
        * @param r other light vector
        * @return reference to itself
        */
        LightVectorND& assignAllValues(const LightVectorND& r)
        {
            assert(size() == r.size());
            assert(this != &r);
            
            burt::CopyHelpers::copy<TElementType>(components, r.components, componentsCount);
            return *this;
        }

        /** Create vector in form [start, start + 1, start + 2,...]
        * @param dimension dimension of the vector
        * @tparam start initial value for sequence
        * @return result vector
        */
        LightVectorND& assignIncreasingSequence(TElementType initialValue)
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] = (initialValue + i);
            return *this;
        }

        TElementType maxItem() const
        {
            size_t sz = size();

            if (sz == 0) [[unlikely]]
                return TElementType();

            TElementType max_item = components[0];
            for (size_t i = 1; i < sz; ++i)
                if (components[i] > max_item)
                    max_item = components[i];
            return max_item;
        }

        TElementType minItem() const
        {
            size_t sz = size();

            if (sz == 0) [[unlikely]]
                return TElementType();

            TElementType min_item = components[0];
            for (size_t i = 1; i < sz; ++i)
                if (components[i] < min_item)
                    min_item = components[i];
            return min_item;
        }

        void exp_inplace()
        {
            size_t dim = size();
            for (size_t i = 0; i < dim; ++i)
                components[i] = ::exp(components[i]);
        }
    };
}

#if BURT_INCLUDE_VECTORIZED_CPU_IMP_VECS
    #include "burt/linalg_vectors/include_internal/LightVector_SIMD_double.h"
    #include "burt/linalg_vectors/include_internal/LightVector_SIMD_float.h"
    #include "burt/linalg_vectors/include_internal/LightVector_SIMD_int.h"
#endif
