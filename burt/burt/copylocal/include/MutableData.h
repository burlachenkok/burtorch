/** @file
* cross-platform memory stream / mutable data / mutation byte stream
*/
#pragma once

#include <stdint.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>

#include <atomic>
#include <string>
#include <memory>
#include <type_traits>
#include <string_view>
#include <stddef.h>

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/mathroutines/include/SimpleMathRoutines.h"

namespace burt
{
    class Data;

    enum class MutableDataRelocationPolicy
    {
        eIncreaseWithChunk,
        eIncreaseByTwoAndRoundUpToChunk
    };

    forceinline_ext void* mdata_crt_aligned_alloc(size_t size)
    {
        ///< Step of increasing buffer size from MutableData
        constexpr size_t kChunkSizeIncrease = 64;
        
#if BURT_WINDOWS
        return _aligned_malloc(size, kChunkSizeIncrease);
#else
        return aligned_alloc(kChunkSizeIncrease, size);
#endif
    }

    forceinline_ext void mdata_crt_aligned_free(void* ptr, size_t size)
    {
#if BURT_WINDOWS
        // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-free?view=msvc-170
        // ..If memblock is a NULL pointer, this function simply performs no actions..
        _aligned_free(ptr);
#else
        // https://en.cppreference.com/w/c/memory/free
        free(ptr);
#endif
    }

    /** A container that represents the ability to add data to a buffer, the size of which is dynamically changing
    */
    //template<class TAllocator, class TDeallocator>
    class MutableData
    {
    public:
        inline static constexpr size_t kChunkSizeIncrease = 64; ///< Step of increasing buffer size

    private:
        size_t length;            ///< length of raw data in bytes
        size_t pos;               ///< current pos where next written will be occur
        uint8_t* bits;            ///< raw data
    
        //TAllocator allocator;
        //TDeallocator deallocator;

    public:
        
        /** Rewind write position of head to the beginning of the write buffer and put value into bytestream
        * @param val value copy of which will be put into bytestream
        * @return true if value has been put fine
        * @sa putValueToStream
        */
        template<class T>
        bool rewindAndPutValueToStream(const T& restrict_ext val)
        {
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD types are allowed");
            size_t oldLength = length;

            if (sizeof(val) > oldLength) [[unlikely]]
            {
                if (!realloc_internal_with_increase(sizeof(val))) [[unlikely]]
                {
                    assert(!"Can not reallocate buffer to extra size'");
                    return false;
                }
            }

            uint8_t* placeToWrite = this->bits;

            *((T*)(placeToWrite)) = val;

            this->pos = sizeof(val);

            return true;
        }
        
        /** Rewind write position of head to the beginning of the write buffer and put value into bytestream
        * @param val value copy of which will be put into bytestream
        * @return true if value has been put fine
        * @sa putValueToStream
        */
        template<class T>
        inline bool rewindAndPutValueToStream(const std::atomic<T>& restrict_ext val)
        {
            T v = val;
            return rewindAndPutValueToStream(v);
        }

        /** Put value into bytestream
        * @param val value copy of which will be put into bytestream
        * @return true if value has been put fine
        * @sa rewindAndPutValueToStream
        */
        template<class T>
        bool putValueToStream(const T& restrict_ext val)
        {
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD types are allowed");

            size_t oldLength = length;
            size_t myPos = pos;
            size_t newPos = myPos + sizeof(val);

            if (newPos > oldLength) [[unlikely]]
            {
                if (!realloc_internal_with_increase(newPos)) [[unlikely]]
                {
                    assert(!"Can not reallocate buffer to extra size'");
                    return false;
                }
            }
            uint8_t* placeToWrite = bits + myPos;
            *((T*)(placeToWrite)) = val;

            // Update true position state
            pos = newPos;

            return true;
        }

        /** Put value into bytestream
        * @param val value copy of which will be put into bytestream
        * @return true if value has been put fine
        * @sa rewindAndPutValueToStream
        */
        template<class T>
        inline bool putValueToStream(const std::atomic<T>& restrict_ext val)
        {
            T v = val;
            return putValueToStream(v);
        }


    protected:

        /**
         * Allocates a block of memory of the specified size, aligned to a cache line size.
         * If the size is zero, the function returns nullptr.
         *
         * @param sz The size of the memory block to allocate in bytes.
         * @return A pointer to the allocated memory block. Returns nullptr if the requested size is zero.
         * @tparam promiseThatSzIsNotZero promize that sz is for sure not-zero
         */
        template <bool promiseThatSzIsNotZero>
        static void* allocateBytes(size_t sz)
        {
            if constexpr (!promiseThatSzIsNotZero)
            {
                if (sz == 0)
                {
                    assert(!"YOU HAVE PROMISED THAT YOU ALLOCATE NON-ZERO ARRAY");
                    return nullptr;
                }
            }

            assert(sz % kChunkSizeIncrease == 0);
            return mdata_crt_aligned_alloc(sz);
        }

        /**
         * Deallocates memory referred to by the given pointer.
         * Depending on the operating system, this function uses the appropriate deallocation method.
         *
         * @param ptr Pointer to the memory to be deallocated. If the pointer is NULL, the function has no effect.
         */
        static void deallocateBytes(void* ptr, size_t sz)
        {
            mdata_crt_aligned_free(ptr, sz);
        }

    public:

        /** Ctor
        */
        explicit MutableData()
        //TAllocator theAlloc = mdata_crt_aligned_alloc, TDeallocator theDealloc = mdata_crt_aligned_free)
        : pos(0)
        , length(0)
        , bits(nullptr)
        //, allocator(theAlloc)
        //, deallocator(theDealloc)
        {
            //realloc_internal_with_increase(1);
        }

        /** Copy Ctor
        * @param rhs the memory big large binary object from which copy is occuring
        */
        MutableData(const MutableData& rhs)
        : pos(rhs.pos)
        , length(rhs.length)
        , bits(nullptr)
        {
            if (length > 0)
            {
                bits = (uint8_t*)allocateBytes<true>(length);
                memcpy(bits, rhs.bits, length);
            }
        }

        /* Copy assignment
        * @param rhs source which will be copied including writing position
        */
        MutableData& operator = (const MutableData& rhs);

        /** Destructor
        */
        ~MutableData()
        {
            //if (bits != NULL)
                deallocateBytes(bits, length);
        }

        /** Pre-reserve memory to eliminate reallocations
        * @param sizeInBytes expected buffer size
        * @return true if memory has been accocated
        */
        bool reserveMemory(size_t sizeInBytes)
        {
            size_t oldLength = length;

            if (sizeInBytes > oldLength) [[likely]]
            {
                if (!realloc_internal_with_increase(sizeInBytes)) [[unlikely]]
                {
                    assert(!"Can not reallocate buffer to extra size'");
                    return false;
                }
            }
            return true;
        }
        
        /** Is container empty. I.e. nobody puh nothing into it.
        * @return true if container is empty.
        */
        bool isEmpty() const
        {
            return pos == 0;
        }

        /** Get current buffer size, allocated from the heap to store data
        * @return number of bytes in heap allocated to store buffer
        */
        size_t getAllocBytesToStoreData() const
        {
            return length;
        }

        /** Put byte into used buffer stream
        * @param val byte, which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putByte(uint8_t val) {
            return putValueToStream(val);
        }

        /** Put single character into used buffer stream
        * @param val character (one byte size) which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putCharacter(char val) {
            return putValueToStream(val);
        }

        /** Put single float into used buffer stream
        * @param single float, which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putFloat(float val) {
            return putValueToStream(val);
        }

        /** Put (x,y) tuple into byte stream
        * @param x item to push into byte stream
        * @param y item to push into byte stream
        * @return true - if both items are pushed successfully. false - no items have been pushed, but there are some problems with pushing
        */
        template <class Tx, class Ty>
        bool putTuple2(const Tx& x, const Ty& y)
        {
            static_assert(std::is_trivially_copyable<Tx>::value, "For such low level manipulation only POD types are allowed as type for x");
            static_assert(std::is_trivially_copyable<Ty>::value, "For such low level manipulation only POD types are allowed as type for y");

            if (!putValueToStream(x))
            {
                return false;
            }
            if (!putValueToStream(y))
            {
                pos -= sizeof(x); // unput x
                return false;
            }
            return true;
        }

        /**
         * Puts the items of the given matrix into a buffer.
         *
         * @param m The matrix whose items are to be added to the buffer.
         * @return true if the matrix items were successfully added, otherwise false.
         * @tparam Mat matrix type
         * @tparam putMatrixSize push information about total number of bytes in matrix representation w/o strides as uint64
         * @tparam putMatrixSizeAsVaryingInteger  push information about total number of bytes in matrix representation w/o strides as varying integer
        */
        template<class Mat,
                 bool putMatrixSize = true,
                 bool putMatrixSizeAsVaryingInteger = true>
        bool putMatrixItems(const Mat& m)
        {
            size_t cols = m.columns();
            size_t rowsInBytes = m.rows() * sizeof(typename Mat::TElementType);
            size_t LDA = m.LDA;

            const typename Mat::TElementType* rawData = m.matrixByCols.dataConst();

            if constexpr (putMatrixSize)
            {
                if constexpr (putMatrixSizeAsVaryingInteger)
                {
                    putUnsignedVaryingInteger(cols * rowsInBytes);
                }
                else
                {
                    putUint64(cols * rowsInBytes);
                }
            }
            
            for (size_t j = 0, offset = 0; j < cols; ++j, offset += LDA)
            {
                putBytes(rawData + offset, rowsInBytes);
            }

            return true;
        }

        /** Put (x,y,z) tuple into byte stream
        * @param x item to push into byte stream
        * @param y item to push into byte stream
        * @param z item to push into byte stream
        * @return true - if both items are pushed successfully. false - no items have been pushed, but there are some problems with pushing
        */
        template <class Tx, class Ty, class Tz>
        bool putTuple3(const Tx& x, const Ty& y, const Tz& z)
        {
            static_assert(std::is_trivially_copyable<Tx>::value, "For such low level manipulation only POD types are allowed as type for x");
            static_assert(std::is_trivially_copyable<Ty>::value, "For such low level manipulation only POD types are allowed as type for y");
            static_assert(std::is_trivially_copyable<Tz>::value, "For such low level manipulation only POD types are allowed as type for z");

            if (!putValueToStream(x)) [[unlikely]]
            {
                return false;
            }
            if (!putValueToStream(y)) [[unlikely]]
            {
                pos -= sizeof(x); // un put x
                return false;
            }
            if (!putValueToStream(z)) [[unlikely]]
            {
                pos -= sizeof(y); // un put y
                pos -= sizeof(x); // un put x
                return false;
            }

            return true;
        }

        /** Put (x,y,z,w) tuple into byte stream
        * @param x item to push into byte stream
        * @param y item to push into byte stream
        * @param z item to push into byte stream
        * @param w item to push into byte stream
        * @return true - if both items are pushed successfully. false - no items have been pushed, but there are some problems with pushing
        */
        template <class Tx, class Ty, class Tz, class Tw>
        bool putTuple4(const Tx& x, const Ty& y, const Tz& z, const Tw& w) {
            if (!putValueToStream(x)) [[unlikely]]
            {
                return false;
            }
            if (!putValueToStream(y)) [[unlikely]]
            {
                pos -= sizeof(x); // unput x
                return false;
            }
            if (!putValueToStream(z)) [[unlikely]]
            {
                pos -= sizeof(y); // un put y
                pos -= sizeof(x); // un put x
                return false;
            }
            if (!putValueToStream(w)) [[unlikely]]
            {
                pos -= sizeof(z); // unput z
                pos -= sizeof(y); // unput y
                pos -= sizeof(x); // unput x
                return false;
            }
            return true;
        }

        /** Put double into used buffer stream
        * @param val double value which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putDouble(double val) {
            return putValueToStream(val);
        }

        /** Rewind write position and put double into used buffer stream
        * @param val double value which will be pushed into buffer
        * @return true - pushing completed successfully
        * @remark Please use if you understand what you're doing. If you don't understand please use putDouble()
        */
        bool rewindAndPutDouble(double val) {
            return rewindAndPutValueToStream(val);
        }

        /** Put int8_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putInt8(int8_t val) {
            return putValueToStream(val);
        }

        /** Put int16_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putInt16(int16_t val) {
            return putValueToStream(val);
        }

        /** Put uint8_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putUint8(uint8_t val) {
            return putValueToStream(val);
        }
        
        /** Put uint16_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putUint16(uint16_t val) {
            return putValueToStream(val);
        }

        /** Put int32_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putInt32(int32_t val) {
            return putValueToStream(val);
        }

        /** Put uint32_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putUint32(uint32_t val) {
            return putValueToStream(val);
        }

        /** Put uint64_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putUint64(uint64_t val) {
            return putValueToStream(val);
        }

        /** Put int64_t value into used buffer stream
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        bool putInt64(int64_t val) {
            return putValueToStream(val);
        }

        /** Put uint64_t value into used buffer stream as varying size integer. Therefor actual size in bytes can be 1-9 bytes long.
        * @param val item which will be pushed into buffer
        * @return true - pushing completed successfully
        */
        template<class TIntType>
        bool putUnsignedVaryingInteger(TIntType val)
        {
            static_assert(sizeof(val) * 8 <= 70);

            // For uint64:
            // 10 bytes (or 80 bits) will allow to store 10 packs of 7bit value + 1bit information/linkage bit
            // 1bit - will be used to indicate that next byte is not the last one. [1 -- this byte is not the last, 0 -- this byte is the last one]
            // 7bit - information bits with paylod.
            // Therefore 10 bytes will allow to store 70 bits of information.
            
            if (pos + 10 > length) [[unlikely]]
            {
                if (!realloc_internal_with_increase(pos + 10))
                {
                    assert(!"Can not reallocate buffer to extra size'");
                    return false;
                }
            }

            {
                uint8_t* restrict_ext placeToWrite = bits + pos;

                for (;; ++placeToWrite)
                {
                    uint8_t value2Write = (uint8_t)(val & 0b0111'1111);
                    
                    val >>= 7;

                    if (val == 0)
                    {
                        *placeToWrite = value2Write;
                        break;
                    }
                    else
                    {
                        value2Write |= 0b1000'0000;
                        *placeToWrite = value2Write;
                    }
                }

                // Update true position next state
                //   It is: placeToWrite + 1               [pointer]
                //          => (placeToWrite + 1) - (bits) [offset]
                this->pos = placeToWrite - bits + 1;
            }

            return true;
        }

        /**
        * Encodes an unsigned varying integer based on its value known at compile time.
        * The encoding depends on whether the value is less than 128.
        *
        * @return True if the encoding was successful, otherwise false.
        * @tparam TIntType integer type
        * @tparam value value itself
        */
        template<class TIntType, TIntType value>
        bool putUnsignedVaryingIntegerKnowAtCompileTime()
        {
            if constexpr (value < 128)
                return putUint8((uint8_t)value);
            else
                return putUnsignedVaryingInteger(value);            
        }
        
        /** Obtain pointer to raw buffer. Pointer will be change during relocation.
        * @return pointer to raw storage
        */
        uint8_t* getPtr() const {
            return bits;
        }

        /** Get filled size
        * @return number of bytes that have filed the buffer
        */
        size_t getFilledSize() const {
            return pos;
        }

        /** Get offset of current write position with respect to buffer start
        * @return number of bytes that have filed the buffer
        */
        size_t getCurWritePos() const {
            return pos;
        }

        /** Setup write position to the start of byte stream
        */
        void rewindToStart() {
            pos = 0;
        }

        /** Setup write position from the start of byte stream
        * @param offset new position for next following byte with respect to start of the internal buffer
        * @return absolute offset from the start of buffer
        */
        size_t seekStart(size_t offs) {
            pos = offs;
            size_t oldLength = length;
            
            if (pos > oldLength)
                realloc_internal_with_increase(pos);

            return pos;
        }


        /** Move write position in byte stream relative to current position
        * @param delta relative offset with respect to current position
        * @return absolute offset from the start of buffer
        */
        size_t seekCur(int32_t delta)
        {
            if (delta < 0 && pos < uint32_t(-delta))
            {
                pos = 0;
            }
            else
            {
                size_t oldLength = length;
                pos += delta;
                
                if (pos > oldLength)
                    realloc_internal_with_increase(pos);
            }

            return pos;
        }

        size_t seekForward(size_t delta)
        {
            {
                size_t oldLength = length;
                pos += delta;
                if (pos > oldLength)
                    realloc_internal_with_increase(pos);
            }

            return pos;
        }

        template<size_t delta, 
                 bool promise_pos_is_ge_delta>
        size_t seekBackwardAtCompileTime() noexcept
        {
            if constexpr (promise_pos_is_ge_delta)
            {
                if (pos < delta)
                {
                    assert(!"YOU PROMISED THAT POS GREATER OR EQUAL DELTA AND IT IS NOT THE CASE");
                }
            }
            pos -= delta;
            return pos;
        }

        size_t seekBackward(size_t delta) noexcept
        {
            if (pos < delta)
            {
                pos = 0;
            }
            else
            {
                pos -= delta;
            }
            return pos;
        }

        /** Move write head to the start and after that put "len" raw bytes into the stream starting from the start position.
        * @param srcBuffer source buffer
        * @param srcBufferLen length of srcBuffer in bytes
        * @return number of real written bytes
        * @tparam moveDataWindow if true then internally write position is actually moving, such that next "put" operation will push bytes sequentially
        */
        template <bool moveDataWindow = true>
        size_t rewindAndPutBytes(const void* restrict_ext srcBuffer, size_t srcBufferLen)
        {
#if 0
            if (srcBuffer == nullptr)
                return 0;
#endif

            size_t oldLength = length;
            
            if (srcBufferLen > oldLength)
            {
                if (!realloc_internal_with_increase(srcBufferLen)) [[unlikely]]
                {
                    assert(!"Can not reallocate buffer to extra size");
                    return 0;
                }
            }

            memcpy(bits, srcBuffer, srcBufferLen);

            if constexpr (moveDataWindow)
            {
                pos = srcBufferLen;
            }

            return srcBufferLen;
        }

        /** Put "len" raw bytes into the stream starting from  current position  MutableData::pos
        * @param srcBuffer source buffer
        * @param srcBufferLen length of srcBuffer in bytes
        * @param moveDataWindow if true then internally write position is actually moving, such that next "put" operation will push bytes sequentially
        * @return number of real written bytes
        */
        template<bool moveDataWindow = true, 
                 MutableDataRelocationPolicy relocation = MutableDataRelocationPolicy::eIncreaseWithChunk>
        size_t putBytes(const void* restrict_ext srcBuffer, size_t srcBufferLen)
        {
#if 0
            if (srcBuffer == nullptr) [[unlikely]]
                return 0;
#endif

            size_t oldLength = length;

            if (pos + srcBufferLen > oldLength)
            {
                if constexpr (relocation == MutableDataRelocationPolicy::eIncreaseWithChunk)
                {
                    if (!realloc_internal_with_increase(pos + srcBufferLen)) [[unlikely]]
                    {
                        assert(!"Can not reallocate buffer to extra size");
                        return 0;
                    }
                }
                else if constexpr (relocation == MutableDataRelocationPolicy::eIncreaseByTwoAndRoundUpToChunk)
                {
                    if (!realloc_internal_with_increase( 2 * (pos + srcBufferLen) ) ) [[unlikely]]
                    {
                        assert(!"Can not reallocate buffer to extra size");
                        return 0;
                    }
                }
                else
                {
                    assert(!"Uknown");
                }

            }

            memcpy(bits + pos, srcBuffer, srcBufferLen);

            if constexpr (moveDataWindow)
            {
                pos += srcBufferLen;
            }

            return srcBufferLen;
        }

        /** Put "len" raw bytes into the stream starting from  current position  MutableData::pos
        * @param srcBytePattern repeating value which will be appended srcBufferLen times
        * @param srcBufferLen length of srcBuffer in bytes
        * @param moveDataWindow if true then internally write position is actually moving, such that next "put" operation will push bytes sequentially
        * @return numer of real written bytes
        */
        template<bool moveDataWindow = true>
        size_t putBytes(char srcBytePattern, size_t srcBufferLen)
        {
            size_t oldLength = length;
            
            if (pos + srcBufferLen > oldLength)
            {
                if (!realloc_internal_with_increase(pos + srcBufferLen)) [[unlikely]]
                {
                    assert(!"Can not reallocate buffer to extra size");
                    return 0;
                }
            }

            memset(bits + pos, srcBytePattern, srcBufferLen);

            if constexpr (moveDataWindow)
            {
                pos += srcBufferLen;
            }

            return srcBufferLen;
        }


        /** Put objectsNumber "POD" or "Standard Layout" objects into byte stream
        * @param srcBuffer source buffer which contains raw array of objects
        * @param objectsNumber number of objects you want to write
        * @param moveDataWindow if true then after push objectsNumber into byte stream the current write position is moving. if false - it not moves.
        * @return number of real written objects which are less or equal to objectsNumber
        */
        template <class T, 
                  bool moveDataWindow = true,
                  MutableDataRelocationPolicy relocation = MutableDataRelocationPolicy::eIncreaseWithChunk>
        size_t putPODs(const T* restrict_ext srcBuffer, size_t objectsNumber = 1)
        {
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD or StandartLayoutType types are allowed");
            return putBytes<moveDataWindow, relocation> (srcBuffer, objectsNumber * sizeof(T)) / sizeof(T);
        }

        template <class T, 
                  bool moveDataWindow = true, 
                  MutableDataRelocationPolicy relocation = MutableDataRelocationPolicy::eIncreaseWithChunk>
        size_t putPOD(const T& restrict_ext srcBuffer)
        {
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD or StandartLayoutType types are allowed");
            constexpr size_t srcBufferLen = sizeof(srcBuffer);
            size_t oldLength = length;

            if (pos + srcBufferLen > oldLength)
            {
                if constexpr (relocation == MutableDataRelocationPolicy::eIncreaseWithChunk)
                {
                    if (!realloc_internal_with_increase(pos + srcBufferLen)) [[unlikely]]
                    {
                        assert(!"Can not reallocate buffer to extra size");
                        return 0;
                    }
                }
                else if constexpr (relocation == MutableDataRelocationPolicy::eIncreaseByTwoAndRoundUpToChunk)
                {
                    if (!realloc_internal_with_increase(2 * (pos + srcBufferLen))) [[unlikely]]
                    {
                        assert(!"Can not reallocate buffer to extra size");
                        return 0;
                    }
                }
                else
                {
                    assert(!"Uknown");
                }
            }
            
            if constexpr (srcBufferLen == 1)
            {
                *((uint8_t*)(bits + pos)) = *((uint8_t*)&srcBuffer);
            }
            else if constexpr (srcBufferLen == 2)
            {
                *((uint16_t*)(bits + pos)) = *((uint16_t*)&srcBuffer);
            }
            else if constexpr (srcBufferLen == 4)
            {
                *((uint32_t*)(bits + pos)) = *((uint32_t*)&srcBuffer);
            }
            else if constexpr (srcBufferLen == 8)
            {
                *((uint64_t*)(bits + pos)) = *((uint64_t*)&srcBuffer);
            }
            else
            {
                memcpy(bits + pos, &srcBuffer, srcBufferLen);
            }
            
            if constexpr (moveDataWindow)
                pos += srcBufferLen;

            return 1;
        }

        template <class T,
                  bool moveDataWindow = true,
                  MutableDataRelocationPolicy relocation = MutableDataRelocationPolicy::eIncreaseWithChunk,
                  typename... Args>
        size_t emplacePOD(Args... args)
        {
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD or StandartLayoutType types are allowed");
            constexpr size_t srcBufferLen = sizeof(T);
            size_t oldLength = length;

            if (pos + srcBufferLen > oldLength)
            {
                if constexpr (relocation == MutableDataRelocationPolicy::eIncreaseWithChunk)
                {
                    if (!realloc_internal_with_increase(pos + srcBufferLen)) [[unlikely]]
                    {
                        assert(!"Can not reallocate buffer to extra size");
                        return 0;
                    }
                }
                else if constexpr (relocation == MutableDataRelocationPolicy::eIncreaseByTwoAndRoundUpToChunk)
                {
                    if (!realloc_internal_with_increase(2 * (pos + srcBufferLen))) [[unlikely]]
                    {
                        assert(!"Can not reallocate buffer to extra size");
                        return 0;
                    }
                }
                else
                {
                    assert(!"Uknown");
                }
            }

            new(bits + pos) T(args...);

            if constexpr (moveDataWindow)
                pos += srcBufferLen;

            return 1;
        }

        /** Put objectsNumber "POD" or "Standard Layout" objects into byte stream
        * @param srcBuffer source buffer which contains raw array of objects
        * @param objectsNumber number of objects you want to write
        * @param moveDataWindow if true then after push objectsNumber into byte stream the current write position is moving. if false - it not moves.
        * @return number of real written objects which are less or equal to objectsNumber
        */
        template <class T, bool moveDataWindow = true>
        size_t rewindAndPutPODs(const T* restrict_ext srcBuffer, size_t objectsNumber = 1)
        {
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD or StandartLayoutType types are allowed");
            return rewindAndPutBytes<moveDataWindow>(srcBuffer, objectsNumber * sizeof(T)) / sizeof(T);
        }

        /** Put signed integer value content into mutable data object, but write it in form of string.
        * @param value unsigned integer value which will be written as a string.
        * @return operation result
        * @remark This functionality eliminates need to convert from integer to string.
        */
        template<class TInt>
        bool putIntegerAsAString(TInt value)
        {
            if (value < 0)
            {
                putCharacter('-');
                value = -value;
            }

            // kSymbols is number of characters to store value
            size_t kSymbols = 0;

            if (value == 0)
            {
                kSymbols = 1;
            }
            else
            {
                for (TInt valueCopy = value; valueCopy > 0; kSymbols++, valueCopy /= 10)
                {
                }
            }

            // Put zero characters
            putBytes(char(), kSymbols);

            char* buffer = (char*)getPtr() + getCurWritePos() - 1;

            for (size_t i = 0; i < kSymbols; i++, buffer--)
            {
                TInt digit = value % 10;
                value /= 10;
                switch (digit)
                {
                case 0:
                    *buffer = '0';
                    break;
                case 1:
                    *buffer = '1';
                    break;
                case 2:
                    *buffer = '2';
                    break;
                case 3:
                    *buffer = '3';
                    break;
                case 4:
                    *buffer = '4';
                    break;
                case 5:
                    *buffer = '5';
                    break;
                case 6:
                    *buffer = '6';
                    break;
                case 7:
                    *buffer = '7';
                    break;
                case 8:
                    *buffer = '8';
                    break;
                case 9:
                    *buffer = '9';
                    break;
                default:
                    assert(!"ERROR IN CONVERSION");
                    break;
                }
            }

            return true;
        }

        /** Put unsigned integer value content into mutable data object, but write it in form of string.
        * @param value unsigned integer value which will be written as a string.
        * @return operation result
        * @remark This functionality eliminates need to convert from unsigned integer to string.
        */
        template<class TUInt>
        bool putUnsignedIntegerAsAString(TUInt value)
        {
            // kSymbols is number of characters to store value
            size_t kSymbols = 0;

            if (value == 0)
            {
                kSymbols = 1;
            }
            else
            {
                for (TUInt valueCopy = value; valueCopy > 0; kSymbols++, valueCopy /= 10)
                {
                }
            }

            // Put zero characters
            putBytes(char(), kSymbols);

            char* buffer = (char*)getPtr() + getCurWritePos() - 1;

            for (size_t i = 0; i < kSymbols; i++, buffer--)
            {
                TUInt digit = value % 10;
                value /= 10;
                switch (digit)
                {
                case 0:
                    *buffer = '0';
                    break;
                case 1:
                    *buffer = '1';
                    break;
                case 2:
                    *buffer = '2';
                    break;
                case 3:
                    *buffer = '3';
                    break;
                case 4:
                    *buffer = '4';
                    break;
                case 5:
                    *buffer = '5';
                    break;
                case 6:
                    *buffer = '6';
                    break;
                case 7:
                    *buffer = '7';
                    break;
                case 8:
                    *buffer = '8';
                    break;
                case 9:
                    *buffer = '9';
                    break;
                default:
                    assert(!"ERROR IN CONVERSION");
                    break;
                }
            }

            return true;
        }

        /** Flags which controls how exactly string is placed into output buffer
        */
        enum class PutStringFlags : std::int8_t
        {
            ePutNoTerminator,
            ePutNewLine,
            ePutZeroTerminator
        };

        /** Put string content into mutable data object.
        * @param str string to put. Only content with any leading termination
        * @param flag information about terminator that you want to put with the conent.
        * @return operation result
        * @sa PutStringFlags
        */
        bool putString(std::string_view str, PutStringFlags flag = PutStringFlags::ePutZeroTerminator)
        {
            putBytes(str.data(), str.size());

            switch (flag)
            {
            case PutStringFlags::ePutNoTerminator:
                break;
            case PutStringFlags::ePutNewLine:
                putByte('\n');
                break;
            case PutStringFlags::ePutZeroTerminator:
                putByte('\0');
                break;
            default:
                assert(!"This case of 'putString' have not been processed by the current implementation");
                break;
            }

            return true;
        }

        /** Construct in heap MutableData object from all residual data from data object d.
        * @param d input data
        */
        static MutableData* getMutableDataFromData(const Data& d);

        /** Construct in heap MutableData object via reading "len" bytes from "d"
        * @param d input data
        * @param bytesLength input data size
        */
        static MutableData* getMutableDataFromData(const Data& d, size_t bytesLength);

    protected:

        /** Internal function to relocate buffer to at least newSize bytes. If buffer is already have need size it do nothing.
        * @param newSize new size of the array
        * @return true if all is fine
        */
        bool realloc_internal_with_increase(size_t newSize);
    };

    typedef std::unique_ptr<MutableData> MutableDataUniquePtr;
}
