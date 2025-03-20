/** @file
* cross-platform byte stream with several raw data access
*/
#pragma once

#include <memory.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <stddef.h>

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/mathroutines/include/SimpleMathRoutines.h"

namespace burt
{
    /** Forward declaration of class used to collect received bytes
    */
    class MutableData;

    /** Raw binary data to read
    */
    class Data {
    public:
        enum class MemInitializedType : std::int8_t
        {
            eGiftWholeMemoryPleaseFree,    ///< Memory was give for object. But this memory in destructor must be deallocate. Do deallocate memory with C runtime.
            eGiftWholeMemoryPleaseNotFree, ///< Memory was give for object. But this memory in destructor must not be deallocate. Don't deallocate memory with C runtime.
            eAllocAndCopy,                 ///< Memory was allocated and copying into memory have been happened
            eAllocAndInitilizedWithZero,   ///< Memory was allocated and initialized with zeros
            eAlloc                         ///< Memory was allocated in heap
        };

    private:
        size_t pos;                       ///< Current reading pos. The next item will be fetched from this position.
        size_t length;                    ///< Length of raw data.
        uint8_t* bits;                    ///< Raw data.
        MemInitializedType memType;       ///< how memory bits was initialized.
        bool lastGetWasSuccessful;        ///< marker that last attempt to get element was successful

    protected:
        /** Get object from stream. For not standard layout types - error in compile time. (https://stackoverflow.com/questions/13648949/layout-for-not-pod-types-because-have-default-constructor)
        * @param returnObject container to store returned object
        * @return false if there is no way to obtain object because there is no enough bytes and true otherwise
        */
        template<class T>
        void getValueFromStream(T& restrict_ext returnObject)
        {
            optimization_assert((void*)&returnObject != (void*)this);

            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD types are allowed");

            size_t myPos = pos;
            size_t myNewPotentialPos = pos + sizeof(T);
            
            if (myNewPotentialPos > length) [[unlikely]] {
                lastGetWasSuccessful = false;
                return;
            }
            else
            {                
                const T* res = (T*)(bits + myPos);
                returnObject = *(res);
                
                this->pos = myNewPotentialPos;
                this->lastGetWasSuccessful = true;
                return;
            }
        }

        /** Get object from stream. For not standart layout types - error in compile time. (https://stackoverflow.com/questions/13648949/layout-for-not-pod-types-because-have-default-constructor)
        * @param returnValue Object container to store returned object
        * @param advance_offset explicitly value for how internal offset in datastream should be shifted in bytes
        * @return false if there is no way to obtain object because there is no enough bytes and true otherwise
        */
        template<class T>
        void getValueFromStream(T& restrict_ext returnValue, size_t advance_offset)
        {
            optimization_assert((void*)&returnObject != (void*)this);
            
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD types are allowed");
            size_t myPos = pos;
            size_t myNewPotentialPos = pos + sizeof(T);

            if (myNewPotentialPos > length) [[unlikely]]
            {
                lastGetWasSuccessful = false;
                return;
            }
            else
            {
                const T* res = (T*)(bits + myPos);
                returnValue = *(res);

                this->lastGetWasSuccessful = true;
                this->pos = myPos + advance_offset;
                
                return;
            }            
        }

        static void* allocateBytes(size_t sz) {
            if (sz == 0)
                return nullptr;

#if 0
            return malloc(sz);
#else
            constexpr size_t kCacheLizeSizeInBytes = 64;
            size_t usedSize = burt::roundToNearestMultipleUp<kCacheLizeSizeInBytes>(sz);
#if BURT_WINDOWS
            return _aligned_malloc(usedSize, kCacheLizeSizeInBytes);
#else
            return aligned_alloc(kCacheLizeSizeInBytes, usedSize);
#endif
#endif
        }

        static void deallocateBytes(void* ptr) {
//  NO NEED ACCORDING TO C++ Standart
//            if (ptr == nullptr)
//                return;
#if 0
            free(ptr);
#else

#if BURT_WINDOWS
            _aligned_free(ptr);
#else
            free(ptr);
#endif

#endif
        }

    public:
        /** Is memory have been allocated for this object
        * @return true if is so
        */
        bool isRawMemoryBeenAllocated() const
        {
            MemInitializedType myMemType = memType;

            return myMemType == MemInitializedType::eAlloc ||
                myMemType == MemInitializedType::eAllocAndCopy ||
                myMemType == MemInitializedType::eAllocAndInitilizedWithZero;
        }

        /** Is only ascii characters insides?
        * @return true if is so
        */
        bool isLastGetSuccess() const {
            return this->lastGetWasSuccessful;
        }

        /** Is only ascii characters insides?
        * @return true if is so
        */
        bool isOnlyAsciiInside() const;


        /**
         * Constructor for the Data class that initializes memory based on the provided semantics.
         *
         * @param ptr Pointer to the initial buffer or memory.
         * @param len Size of the memory or buffer.
         * @param ptrSemantics Specifies how the memory should be initialized or handled.
         *        - eGiftWholeMemoryPleaseFree
         *        - eGiftWholeMemoryPleaseNotFree
         *        - eAllocAndCopy
         *        - eAllocAndInitilizedWithZero
         *        - eAlloc
         *        Default value is MemInitializedType::eAllocAndCopy.
         *
         * @return A Data object initialized according to the specified memory semantics.
         */
        Data(void* ptr, size_t len, MemInitializedType ptrSemantics = MemInitializedType::eAllocAndCopy) :
            pos(0),
            length(len),
            bits(0),
            memType(ptrSemantics),
            lastGetWasSuccessful(false)
        {
            switch (ptrSemantics)
            {
            case MemInitializedType::eGiftWholeMemoryPleaseFree:
            case MemInitializedType::eGiftWholeMemoryPleaseNotFree:
                bits = (uint8_t*)ptr;
                break;
            case MemInitializedType::eAllocAndCopy:
                if (len > 0)
                {
                    bits = static_cast<uint8_t*>(allocateBytes(len));
                    assert(bits != nullptr);
                    
                    if (ptr)
                    {
                        memcpy(bits, ptr, length);
                    }
                    else
                    {
                        assert(!"Please don't use eAllocAndCopy with empty buffer!");
                    }
                }
                break;
            case MemInitializedType::eAllocAndInitilizedWithZero:
                if (len > 0)
                {
                    bits = static_cast<uint8_t*>(allocateBytes(len));
                    assert(bits != nullptr);
                    
                    memset(bits, 0, length);
                }
                break;
            case MemInitializedType::eAlloc:
                if (len > 0)
                {
                    bits = static_cast<uint8_t*>(allocateBytes(len));
                    assert(bits != nullptr);
                }
                break;
            default:
                assert(!"There is no code to process allocation policy that sofware engineer append recently");
                break;

            }
        }

        /** Copy constructor
        */
        Data(const Data& rhs)
            : pos(rhs.pos)
            , length(rhs.length)
            , bits(nullptr)
            , memType(MemInitializedType::eAllocAndCopy)
            , lastGetWasSuccessful(rhs.lastGetWasSuccessful)
        {
            if (rhs.bits)
            {
                bits = static_cast<uint8_t*>(allocateBytes(rhs.length));
                memcpy(bits, rhs.bits, length);
            }
        }


        /**
         * Move constructor for the Data class.
         *
         * This constructor transfers ownership of the resources from the given
         * Data object to the newly created object, leaving the given object in
         * a valid but unspecified state.
         *
         * @param rhs The Data object to be moved from. After the operation,
         *            rhs will be in a valid but unspecified state.
         * @return A new Data object that has taken ownership of rhs's resources.
         */
        Data(Data&& rhs) noexcept
            : pos(rhs.pos)
            , length(rhs.length)
            , bits(rhs.bits)
            , memType(rhs.memType)
            , lastGetWasSuccessful(rhs.lastGetWasSuccessful)
        {
            rhs.bits = nullptr;
            rhs.length = 0;
        }

        /* Copy assignment
        * @param rhs source which will be copied including reading position
        */
        Data& operator = (const Data& rhs)
        {
            if (this == &rhs)
                return *this;

            if (bits)
            {
                if (isRawMemoryBeenAllocated())
                {
                    deallocateBytes(bits);
                }
                bits = nullptr;
            }
            
            pos = rhs.pos;
            length = rhs.length;
            memType = MemInitializedType::eAllocAndCopy;

            if (rhs.bits)
            {
                bits = static_cast<uint8_t*>(allocateBytes(rhs.length));
                memcpy(bits, rhs.bits, length);
            }

            return *this;
        }

        /** Move assignment operator
        */
        Data& operator = (Data&& rhs) noexcept
        {
            if (this == &rhs)
                return *this;

            if (bits)
            {
                if (isRawMemoryBeenAllocated())
                {
                    deallocateBytes(bits);
                }
                bits = nullptr;
            }

            pos = rhs.pos;
            length = rhs.length;
            bits = rhs.bits;
            memType = rhs.memType;

            rhs.bits = nullptr;
            rhs.length = 0;

            return *this;
        }

        /* Destructor
        */
        ~Data() {
            clear();
        }

        /* Explicit call of clear container to get rid of any pointers
        */
        void clear()
        {
            if (bits)
            {
                if (isRawMemoryBeenAllocated() || memType == MemInitializedType::eGiftWholeMemoryPleaseFree)
                {
                    deallocateBytes(bits);
                }
                bits = nullptr;
            }
        }

        /** Is byte stream empty?
        * @return true if byte stream is empty
        */
        bool isEmpty() const {
            return pos >= length;
        }


        /** Total length in bytes of container without take into account seeked offset
        * @return number of bytes
        */
        size_t getTotalLength() const {
            return length;
        }

        /** Total length in bytes in container that still can be read. I.e. it takes into account seeked offset
        * @return number of bytes
        */
        size_t getResidualLength() const {
            if (isEmpty())
                return 0;
            else
                return length - pos;
        }

        /** Get the current offset in the container from where data comes from
        * @return buffer length
        */
        size_t getPos() const {
            return pos;
        }

        /** Get the next "character" from the buffer
        * @return obtained value
        */
        char getCharacter() {
            char returnedSymbol;
            getValueFromStream(returnedSymbol);
            return returnedSymbol;
        }

        /** Get the next "byte" from the buffer
        * @return obtained value
        */
        uint8_t getByte() {
            uint8_t returnedByte;
            getValueFromStream(returnedByte);
            return returnedByte;
        }

        /** Get the next "float" from the buffer
        * @return obtained value
        */
        float getFloat() {
            float returnedValue;
            getValueFromStream(returnedValue);
            return returnedValue;
        }

        /** Get the next "double" from the buffer
        * @return obtained value
        */
        double getDouble()
        {
            double returnedValue;
            getValueFromStream(returnedValue);
            return returnedValue;
        }

        /** Get the next "signed short" from the buffer
        * @return obtained value
        */
        int16_t getInt16()
        {
            int16_t returnedValue;
            getValueFromStream(returnedValue);
            return returnedValue;
        }

        /** Get the next "unsigned short" from the buffer
        * @return obtained value
        */
        uint16_t getUint16()
        {
            uint16_t returnedValue;
            getValueFromStream(returnedValue);
            return returnedValue;
        }

        /** Get the next "signed integer 32 bit" from the buffer
        * @return obtained value
        */
        int32_t getInt32() {
            int32_t returnedValue;
            getValueFromStream(returnedValue);
            return returnedValue;
        }

        /** Get the next "unsigned integer 32 bit" from the buffer
        * @return obtained value
        */
        uint32_t getUint32() {
            uint32_t returnedValue;
            getValueFromStream(returnedValue);
            return returnedValue;
        }

        /** Get the next "signed integer 64 bit" from the buffer
        * @return obtained value
        */
        int64_t getInt64() {
            int64_t returnedValue;
            getValueFromStream(returnedValue);
            return returnedValue;
        }

        /** Get the next "unsigned integer 64 bit" from the buffer
        * @return obtained value
        */
        uint64_t getUint64() {
            uint64_t returnedValue;
            getValueFromStream(returnedValue);
            return returnedValue;
        }

        /** Get the next "unsigned integer 64 bit" from the buffer
        * @return obtained value
        */
        template<class TResultType = uint64_t>
        TResultType getUnsignedVaryingInteger()
        {
            TResultType returnedValue = 0;
            uint8_t obtainedBits = 0;

            const uint8_t* restrict_ext placeToRead = bits + pos;
            size_t myPos = pos;

            for (;;++placeToRead, ++myPos)
            {
                if (myPos >= length) [[unlikely]]
                {
                    lastGetWasSuccessful = false;
                    break;
                }
                else
                {
                    uint8_t res = *placeToRead;

                    if ( (res & 0b1000'0000) == 0)
                    {
                        // last byte for varying integer
                        if (obtainedBits <= 64 - 7) [[likely]]
                        {
                           // the transfered number is actually smaller or equal then 64-7 bits                                
                            returnedValue >>= 7;
                            returnedValue |= (uint64_t(res) << (64 - 7));
                            obtainedBits += 7;
                        }
                        else
                        {
                            assert(obtainedBits <= 64);
                            
                            // the transfered number is actually bigger then 64-7 bits                                
                            //   =>  we do not need all actual bits from the last byte. we need only residualBits.
                            uint8_t residualBits = uint8_t(64) - obtainedBits;                            
                            returnedValue >>= residualBits;
                            returnedValue |= (uint64_t(res) << (64 - residualBits));
                            obtainedBits += residualBits;
                        }

                        break;
                    }
                    else [[likely]]
                    {
                        // not last byte for varying integer
                        res = res & 0b0111'1111;
                        
                        // shift to the right number with obtained bits so far with SHR 7
                        returnedValue >>= 7;
                        
                        // setup the high-order 7-bits
                        returnedValue |= (uint64_t(res) << (64 - 7));
                        
                        // increse total number of bits
                        obtainedBits += 7;
                    }
                }
            }

            // Form the final result.
            returnedValue = (returnedValue >> (64 - obtainedBits));
            
            this->pos = myPos + 1;
            
            return returnedValue;
        }

        /** Get the packed dense matrix from the buffer
        * @param m matrix placeholder
        * @return true if all is ok
        */
        template<class Mat, 
                 bool getMatrixSize = true,
                 bool getMatrixSizeAsVaryingInteger = true>
        bool getMatrixItems(Mat& m)
        {
            size_t cols = m.columns();
            size_t rowsInBytes = m.rows() * sizeof(typename Mat::TElementType);
            size_t LDA = m.LDA;
            typename Mat::TElementType* rawData = m.matrixByCols.data();

            if constexpr (getMatrixSize)
            {
                uint64_t msgSize = 0;
                
                if constexpr (getMatrixSizeAsVaryingInteger)
                {
                    msgSize = getUnsignedVaryingInteger();
                }
                else
                {
                    msgSize = getUint64();
                }
                
                if (msgSize != cols * rowsInBytes)
                {
                    getBytes(0, msgSize);
                    return false;
                }
            }

            for (size_t j = 0, offset = 0; j < cols; ++j, offset += LDA)
            {
                getBytes(rawData + offset, rowsInBytes);
            }
            return true;
        }

        /** Get 2 elements with type which is standard layout type from data stream.
        * @param x reference to container which will obtain 1st value
        * @param y reference to container which will obtain 2nd value
        * @return true if there is a buffer which has enough storage for obtaining elements
        */
        template <class T>
        bool getTuple2(T& x, T& y)
        {
            if (getResidualLength() < sizeof(T) * 2)
                return false;
            getValueFromStream(x);
            getValueFromStream(y);
            return true;
        }

        /** Get 3 elements with type which is standard layout type from data stream.
        * @param x reference to container which will obtain 1st value
        * @param y reference to container which will obtain 2nd value
        * @param z reference to container which will obtain 3rd value
        * @return true if there is a buffer which has enough storage for obtaining elements
        */
        template <class T>
        bool getTuple3(T& x, T& y, T& z)
        {
            if (getResidualLength() < sizeof(T) * 3)
                return false;
            getValueFromStream(x);
            getValueFromStream(y);
            getValueFromStream(z);
            return true;
        }

        /** Get n elements with type which is standard layout type from data stream.
        * @param container pointer to data storage
        * @param n number of elements of type T which you want to obtain
        * @return true if there is a buffer which has enough storage for obtaining elements
        */
        template <class T>
        bool getTupleN(T* restrict_ext container, size_t n)
        {
            if (getResidualLength() < sizeof(T) * n)
                return false;

            for (size_t i = 0; i < n; ++i)
                getValueFromStream(container[i]);

            return true;
        }

        /** Obtain pointer to raw buffer
        * @return pointer to raw storage
        */
        uint8_t* getPtr() const {
            return bits;
        }

        /** Get pointer to residual data in data stream.
        * @return pointer to raw storage which have been offset to number of bytes which have been read
        */
        uint8_t* getPtrToResidual() const {
            return bits + pos;
        }

        /** Setup read position to the start of byte stream
        */
        void rewindToStart() {
            pos = 0;
        }

        /** Setup read position from the start of byte stream.
        * @param offset new position for next following byte with respect to start of the internal buffer
        * @return absolute offset from the start of buffer
        */
        size_t seekStart(size_t offset) {
            if (offset > length)
                pos = length;
            else
                pos = offset;

            return pos;
        }

        /** Move read position in byte stream relative to current position.
        * @param delta relative offset with respect to current position
        * @return absolute offset from the start of buffer.
        */
        size_t seekCur(int32_t delta) {
            if (delta < 0 && pos < uint32_t(-delta))
                pos = 0;
            else if (pos > length + delta)
                pos = length;
            else
                pos += delta;
            return pos;
        }

        /** Obtain len bytes from stream starting from current position relative to current Data::pos
        * @param dstBuffer buffer receiver
        * @param dstBufferLen size of dst buffer in bytes
        * @param moveDataWindow if true then after obtaining dstBufferLen the current read position is moving. if false - it not moves.
        * @return number of real obtained bytes which are less or equal to dstBufferLen
        */
        size_t getBytes(void* restrict_ext dstBuffer, size_t dstBufferLen, bool moveDataWindow = true);

        /** Obtain objectsNumber "POD" or "Standard Layout" objects from byte stream.
        * @param dstBuffer buffer receiver.
        * @param objectsNumber number of objects you want to obtain.
        * @param moveDataWindow if true then after obtaining dstBufferLen the current read position is moving. if false - it not moves.
        * @return number of real obtained objects which are less or equal to objectsNumber
        */
        template <class T>
        size_t getObjects(T* restrict_ext dstBuffer, size_t objectsNumber = 1, bool moveDataWindow = true)
        {
            static_assert(std::is_trivially_copyable<T>::value, "For such low level manipulation only POD or StandartLayoutType types are allowed");
            const size_t bytesNumber = getBytes(dstBuffer, sizeof(T) * objectsNumber, moveDataWindow);

            return bytesNumber / sizeof(T);
        }

        /** Obtain ASCII string from byte stream. String is terminating with '\0' or with '\n' and '\r' characters are skipping.
        * @return obtained string
        */
        std::string getString();

        /** Obtain ASCII string from byte stream. String is terminating with '\0' or with '\n'
        * @return obtained string view
        */
        std::string_view getStringView();

        /** Obtain copy of data from shared/mutable data storage "m"
        * @param m data storage object with shared data
        * @param shareMDataPtr if true then constructed Data will have a similar pointer of "m . It's rather dangerous!!! Due to that "m" in fact can relocate buffer, but it's very effective. Especially for huge buffers. If false then constructed Data will have it's own copy
        * @return raw pointer to constructed Data object in the heap
        */
        static Data* getDataFromMutableData(const MutableData* m, bool shareMDataPtr);
    };

    typedef std::unique_ptr<Data> DataUniquePtr;
}
