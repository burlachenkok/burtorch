#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burtcore/include/burtorch_special_copy.h"

#include <type_traits>
#include <initializer_list>
#include <stddef.h>
#include <memory.h>


enum class UsedArrayType
{
    eFixedSizeOrResizableArray   = 0,
    eArithmeticProgression       = 1
};

/**
 * A template class for a special array with different memory layouts based on the number of items.
 *
 * The `SpecialArray` class supports different types of arrays:
 * - Fixed-size arrays (with a specified number of items).
 * - Arrays with arithmetic progression indices.
 * - Dynamically allocated arrays.
 *
 * The class provides several constructors, data access methods, and memory management features to handle these
 * different array types.
 *
 * @tparam TItemType The type of the items stored in the array.
 * @tparam TSizeTagType The type used for the size tag, defaulting to TItemType.
 * @tparam kITemsInFixedSizeArray The maximum number of items in a fixed-size array. Default is 2.
 */
template <class TItemType,
          class TSizeTagType = TItemType,
          size_t kITemsInFixedSizeArray = 2>
struct SpecialArray
{
    using ItemType = TItemType;                    ///< The type of the items stored in the array.
    using SizeTagType = TSizeTagType;              ///< The type of the size tag.
    using SizeTagTypeWithArrayType = TSizeTagType; ///< SizeTagType with array type.


    static_assert(kITemsInFixedSizeArray > 1, "Fixed size array should contain 1 and more items");
	static_assert(std::is_trivially_copyable<TItemType>::value, "Check that type is trivially copyable");

    struct /*alignas(alignof(TItemType))*/ ArrayDescr
	{
        TItemType* first_pointer;        ///< Typically 8 bytes
	};

    struct /*alignas(alignof(TItemType))*/ ArithmeticProgressionDescr
	{
        TItemType a1;                    ///< First symbol
        TItemType dstep;                 ///< Difference
	};

    struct /*alignas(alignof(TItemType))*/ FixedSizeArrayDescr
    {
        TItemType items[kITemsInFixedSizeArray]; ///< First and next indicies
    };


    union /*alignas(alignof(TItemType))*/ SpecialArrayState
	{
        ArrayDescr arb_array;                            ///< Arbitarily array
        ArithmeticProgressionDescr arithme_progr_array;  ///< Array in which indicies form arithemtic progression (essentially good strides exist)
        FixedSizeArrayDescr fixed_size_array;            ///< Array with 1 items and more
    };

    /** Returns only the size of the array.
     * @return The size of the array as a SizeTagType.
     */
    constexpr SizeTagType size() const noexcept
    {
        return extSize(size_and_tag_together);
	}

    /** Checks if the array is empty.
    * @return True if the array is empty, false otherwise.
    */
    constexpr bool isEmpty() const noexcept
    {
        constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
        constexpr SizeTagType mask = (((SizeTagType)0x1) << (kBitInNode));
        return size_and_tag_together == 0 || size_and_tag_together == mask;
    }

    /** Returns a pointer to the constant data in the array.
     * @return A pointer to the constant data in the array.
     */
    const TItemType* dataConst() const
    {
        if (isTinyArray())
            return state.fixed_size_array.items;
        else if (isLongArray())
            return state.arb_array.first_pointer;
        else //if (isArithmProgressArray())
            return nullptr;
    }

    /** Returns a pointer to the data in the array.
    * @return A pointer to the data in the array.
    */
    TItemType* data()
    {
        if (isTinyArray())
            return state.fixed_size_array.items;
        else if (isLongArray())
            return state.arb_array.first_pointer;
        else //if (isArithmProgressArray())
            return nullptr;
    }

    /** Initializes the array to be empty and sets the size and tag to zero.
    */
	SpecialArray() noexcept
    {
        size_and_tag_together = 0;
        burt_assert(extType(size_and_tag_together) == UsedArrayType::eFixedSizeOrResizableArray && extSize(size_and_tag_together) == 0);
    }

    /**
     * Constructor for a single item array.
    * @param item_one The first item to initialize the array with.
    */
	SpecialArray(TItemType item_one) noexcept
	{
        size_and_tag_together = 1;
        burt_assert(extType(size_and_tag_together) == UsedArrayType::eFixedSizeOrResizableArray && extSize(size_and_tag_together) == 1);
        state.fixed_size_array.items[0] = item_one;
	}

    /**
    * Constructor for two items.
    *
    * @param item_one The first item to initialize the array with.
    * @param item_two The second item to initialize the array with.
    */
	SpecialArray(TItemType item_one, TItemType item_two) noexcept
	{
        size_and_tag_together = 2;
        burt_assert(extType(size_and_tag_together) == UsedArrayType::eFixedSizeOrResizableArray && extSize(size_and_tag_together) == 2);

        if constexpr (2 <= kITemsInFixedSizeArray)
        {
            state.fixed_size_array.items[0] = item_one;
            state.fixed_size_array.items[1] = item_two;
        }
        else
        {
            size_t sz_in_bytes = sizeof(TItemType) * 2;
            state.arb_array.first_pointer = (TItemType*) allocateBytes(sz_in_bytes);
            state.arb_array.first_pointer[0] = item_one;
            state.arb_array.first_pointer[1] = item_two;
        }
	}

    /**
    * Constructor that initializes the array from an initializer list.
    * @param init_list The initializer list to initialize the array with.
    */
	SpecialArray(std::initializer_list<TItemType> init_list) noexcept
	{
        size_t sz = init_list.size();
        size_and_tag_together = sz;
        burt_assert(extType(size_and_tag_together) == UsedArrayType::eFixedSizeOrResizableArray && extSize(size_and_tag_together) == sz);

        size_t sz_in_bytes = sizeof(TItemType) * sz;

        if (sz <= kITemsInFixedSizeArray)
        {
            memcpy(state.fixed_size_array.items, init_list.begin(), sz_in_bytes);
        }
        else
        {
            state.arb_array.first_pointer = (TItemType*) allocateBytes(sz_in_bytes);
            memcpy(state.arb_array.first_pointer, init_list.begin(), sz_in_bytes);
        }
	}

    /*Creates a new array that is a copy of the provided array.
    *
    * @param rhs The array to copy.
    */
	SpecialArray(const SpecialArray& rhs) noexcept
	{
        size_and_tag_together = rhs.size_and_tag_together;
        burt_assert(extType(size_and_tag_together) == extType(rhs.size_and_tag_together) && extSize(size_and_tag_together) == extSize(rhs.size_and_tag_together));

        if (isTinyArray())
        {
            memcpyAtCompileTime <kITemsInFixedSizeArray> (state.fixed_size_array.items, rhs.state.fixed_size_array.items);
        }
        else if (isArithmProgressArray())
        {
            state.arithme_progr_array.a1 = rhs.state.arithme_progr_array.a1;
            state.arithme_progr_array.dstep = rhs.state.arithme_progr_array.dstep;
		}
        else// if (isLongArray())
        {
            burt_assert(extSize(size_and_tag_together) == size_and_tag_together);
            size_t sz_in_bytes = sizeof(TItemType) * size_and_tag_together;
            state.arb_array.first_pointer = (TItemType*)allocateBytes(sz_in_bytes);
            memcpy(state.arb_array.first_pointer, rhs.state.arb_array.first_pointer, sz_in_bytes);
        }
	}

    /** Moves the contents of the provided array to this array.
    */
	SpecialArray(SpecialArray&& rhs) noexcept
	{
        size_and_tag_together = rhs.size_and_tag_together;
        burt_assert(extType(size_and_tag_together) == extType(rhs.size_and_tag_together) && extSize(size_and_tag_together) == extSize(rhs.size_and_tag_together));

        if (isTinyArray())
        {
            memcpyAtCompileTime<kITemsInFixedSizeArray> (state.fixed_size_array.items, rhs.state.fixed_size_array.items);
        }
        else if (isArithmProgressArray())
        {
            state.arithme_progr_array.a1 = rhs.state.arithme_progr_array.a1;
            state.arithme_progr_array.dstep = rhs.state.arithme_progr_array.dstep;
        }
        else// if (isLongArray())
        {
            state.arb_array.first_pointer = rhs.state.arb_array.first_pointer;
            rhs.size_and_tag_together = 0;
        }
	}

    /**
    * Copy assignment operator.
    *
    * Copies the contents of the provided array to this array.
    *
    * @param rhs The array to copy.
    * @return A reference to this array.
    */
	SpecialArray& operator = (const SpecialArray& rhs) noexcept
	{
		if (this == &rhs)
			return *this;

        if (isLongArray())
			deallocateBytes(state.arb_array.first_pointer);

        size_and_tag_together = rhs.size_and_tag_together;
        burt_assert(extType(size_and_tag_together) == extType(rhs.size_and_tag_together) && extSize(size_and_tag_together) == extSize(rhs.size_and_tag_together));

        if (isTinyArray())
        {
            memcpyAtCompileTime<kITemsInFixedSizeArray>(state.fixed_size_array.items, rhs.state.fixed_size_array.items);
        }
        else if (isArithmProgressArray())
        {
            state.arithme_progr_array.a1 = rhs.state.arithme_progr_array.a1;
            state.arithme_progr_array.dstep = rhs.state.arithme_progr_array.dstep;
        }
        else // if (isLongArray())
        {
            burt_assert(extSize(size_and_tag_together) == size_and_tag_together);
            size_t sz_in_bytes = sizeof(TItemType) * size_and_tag_together;
            state.arb_array.first_pointer = (TItemType*) allocateBytes(sz_in_bytes);
            memcpy(state.arb_array.first_pointer, rhs.state.arb_array.first_pointer, sz_in_bytes);
        }

		return *this;
	}

    /**
    * Move assignment operator.
    *
    * Moves the contents of the provided array to this array.
    *
    * @param rhs The array to move.
    * @return A reference to this array.
    */
	SpecialArray& operator = (SpecialArray&& rhs) noexcept
	{
		// Move self to self - usually not checked because this is not normal behavior. 
		// C++11 defines that behavior is undefined in case it moves to itself.
		// if (this == &rhs)
		//     return *this;

        if (isLongArray())
            deallocateBytes(state.arb_array.first_pointer);

        size_and_tag_together = rhs.size_and_tag_together;
        burt_assert(extType(size_and_tag_together) == extType(rhs.size_and_tag_together) && extSize(size_and_tag_together) == extSize(rhs.size_and_tag_together));

        if (isTinyArray())
        {
            memcpyAtCompileTime<kITemsInFixedSizeArray>(state.fixed_size_array.items, rhs.state.fixed_size_array.items);
        }
        else if (isArithmProgressArray())
        {
            state.arithme_progr_array.a1 = rhs.state.arithme_progr_array.a1;
            state.arithme_progr_array.dstep = rhs.state.arithme_progr_array.dstep;
        }
        else // if (isLongArray())
        {
            state.arb_array.first_pointer = rhs.state.arb_array.first_pointer;
            rhs.size_and_tag_together = 0;
        }

        return *this;
	}

    /** Cleans up any dynamically allocated memory used by the array.
    */
	~SpecialArray() noexcept
	{
        if (isLongArray())
		{
			deallocateBytes(state.arb_array.first_pointer);
		}
	}

    /**
    * Resizes the array in a lossy (for content) manner.
    *
    * This function resizes the array without initializing the new elements.
    *
    * @param sz The new size of the array.
    * @return A pointer to the resized array.
    */
    TItemType* sysArrayResizeLossyWithoutAnyInit(SizeTagType sz) noexcept
	{
        if (isLongArray())
        {
            if (size_and_tag_together == sz)
                return state.arb_array.first_pointer;
            deallocateBytes(state.arb_array.first_pointer);
        }

        size_and_tag_together = sz;
        if (sz > kITemsInFixedSizeArray)
        {
            state.arb_array.first_pointer = (TItemType*)allocateBytes(sizeof(TItemType) * sz);
            return state.arb_array.first_pointer;
        }
        else
        {
            return state.fixed_size_array.items;
        }
	}

    /**
    * Resizes the array to fit an arithmetic progression.
    *
    * This function resizes the array and fills it with an arithmetic progression.
    *
    * @param sz The new size of the array.
    * @param first The first item in the arithmetic progression.
    * @tparam dstep The step size of the arithmetic progression.
    */
    template<TItemType dstep = TItemType(1)>
    void sysArrayResizeLossyToArithmeticProgression(SizeTagType sz, TItemType first) noexcept
    {
        if (isLongArray())
            deallocateBytes(state.arb_array.first_pointer);

        size_and_tag_together = createSizeAndTag(sz, UsedArrayType::eArithmeticProgression);

        state.arithme_progr_array.a1 = first;
        state.arithme_progr_array.dstep = dstep;
        return;
    }

    /**
    * @brief Clears the array by erasing memory (without calling destructors, this array in the first place is only for simple type) and set zero length.
    *
    * This function deallocates any dynamically allocated memory and resets the array to its initial state.
    */
    void sysClearWithErase() noexcept
	{
        if (isLongArray())
            deallocateBytes(state.arb_array.first_pointer);

        // long array can not have size_tag equal to zero => it automatically reset array type from long to tiny or make arithmetic progression not touched
        size_and_tag_together = 0;
        burt_assert(!isLongArray());
	}

    /**
    * Initializes the array to its default state.
    *
    * This function resets the array size and type to zero and initializes it with default values.
    * 
    * @warning Only use when you know what you're doing. This function assume that the underlying data did not contain a valid pointer.
    */
    void sysInitToDefault() noexcept
    {
        size_and_tag_together = 0;
        burt_assert(!isLongArray());
    }

    /**
    * Adds a new item to the end of the array.
    *
    * @param item The item to add to the array.
    * @return True if the item was successfully added, false otherwise.
    */
    bool push_back(TItemType item) noexcept
	{
        if (isTinyArray())
        {
            if (size_and_tag_together < kITemsInFixedSizeArray)
            {
                state.fixed_size_array.items[size_and_tag_together] = item;
            }
            else // (size_tag == kITemsInFixedSizeArray)
            {
                constexpr SizeTagType oldSize = kITemsInFixedSizeArray;
                TItemType* oldItems = state.arb_array.first_pointer;
                TItemType* newItems = (TItemType*) allocateBytes(sizeof(TItemType) * (oldSize + 1));
                memcpyAtCompileTime<kITemsInFixedSizeArray>(newItems, oldItems);
                newItems[oldSize] = item;
                state.arb_array.first_pointer = newItems;
            }
            size_and_tag_together += 1;
            return true;
        }
        else if (isArithmProgressArray())
        {
            if (size_and_tag_together == createSizeAndTagAtCompiletTime(0, UsedArrayType::eArithmeticProgression))
            {
                state.arithme_progr_array.a1 = item;
                size_and_tag_together = createSizeAndTagAtCompiletTime(1, UsedArrayType::eArithmeticProgression);
                return true;
            }
            else if (size_and_tag_together == createSizeAndTagAtCompiletTime(1, UsedArrayType::eArithmeticProgression))
            {
                state.arithme_progr_array.dstep = item - state.arithme_progr_array.a1;
                size_and_tag_together = createSizeAndTagAtCompiletTime(2, UsedArrayType::eArithmeticProgression);
                return true;
            }
            else // size_tag >=2
            {
                // a1, a1+d, ...
                // 0    1    ...
                auto cur_size = extSize(size_and_tag_together);
                TItemType item_index_sz = state.arithme_progr_array.a1 + state.arithme_progr_array.dstep * (cur_size);
                if (item_index_sz == item)
                {
                    size_and_tag_together = createSizeAndTag(cur_size + 1, UsedArrayType::eArithmeticProgression);
                    return true;
                }
                else
                {
                    burt_assert(!"ITEM DOES NOT FOLLOW ARITHMETIC PROGRESSION STYLE");
                    return false;
                }
            }
        }
        else // if (isLongArray())
        {
            SizeTagType oldSize = size_and_tag_together;
            TItemType* oldItems = state.arb_array.first_pointer;
            TItemType* newFirst = (TItemType*) allocateBytes(sizeof(TItemType) * (oldSize + 1));
            memcpy(newFirst, oldItems, oldSize * sizeof(TItemType));
            newFirst[oldSize] = item;

            state.arb_array.first_pointer = newFirst;
            deallocateBytes(oldItems);

            size_and_tag_together += 1;
            return true;
        }

	}

    /**
    * Adds two new items to the end of the array.
    *
    * @param itemA The first item to add to the array.
    * @param itemB The second item to add to the array.
    * @return True if the items were successfully added, false otherwise.
    */
    bool push_back_two_items(TItemType itemA, TItemType itemB) noexcept
	{
        if (isTinyArray())
        {
            if (size_and_tag_together + 2 <= kITemsInFixedSizeArray)
            {
                state.fixed_size_array.items[size_and_tag_together] = itemA;
                state.fixed_size_array.items[size_and_tag_together + 1] = itemB;
            }
            else // size_and_tag_together > kITemsInFixedSizeArray - 2 &&  size_and_tag_together <= kITemsInFixedSizeArray;
            {
                if (size_and_tag_together == kITemsInFixedSizeArray)
                {
                    constexpr SizeTagType oldSize = kITemsInFixedSizeArray;

                    TItemType* oldItems = state.fixed_size_array.items;
                    TItemType* newItems = (TItemType*)allocateBytes(sizeof(TItemType) * (oldSize + 2));
                    memcpyAtCompileTime<oldSize> (newItems, oldItems);
                    newItems[oldSize] = itemA;
                    newItems[oldSize + 1] = itemB;
                    state.arb_array.first_pointer = newItems;
                }
                else if (size_and_tag_together == kITemsInFixedSizeArray - 1)
                {
                    constexpr SizeTagType oldSize = kITemsInFixedSizeArray - 1;

                    TItemType* oldItems = state.fixed_size_array.items;
                    TItemType* newItems = (TItemType*)allocateBytes(sizeof(TItemType) * (oldSize + 2));
                    memcpyAtCompileTime<oldSize>(newItems, oldItems);
                    newItems[oldSize] = itemA;
                    newItems[oldSize + 1] = itemB;
                    state.arb_array.first_pointer = newItems;
                }
            }
            size_and_tag_together += 2;
            return true;
        }
        else if (isArithmProgressArray())
        {
            if (size_and_tag_together == createSizeAndTagAtCompiletTime(0, UsedArrayType::eArithmeticProgression))
            {
                state.arithme_progr_array.a1 = itemA;
                state.arithme_progr_array.dstep = itemB - itemA;
                size_and_tag_together = createSizeAndTagAtCompiletTime(2, UsedArrayType::eArithmeticProgression);
                return true;
            }
            else if (size_and_tag_together == createSizeAndTagAtCompiletTime(1, UsedArrayType::eArithmeticProgression))
            {
                auto old_size = extSize(size_and_tag_together);
                TItemType dstep = itemA - state.arithme_progr_array.a1;
                TItemType item_index_sz_add_0 = state.arithme_progr_array.a1 + dstep * (old_size);

                if (item_index_sz_add_0 == itemB)
                {
                    state.arithme_progr_array.dstep = dstep;
                    size_and_tag_together = createSizeAndTagAtCompiletTime(3, UsedArrayType::eArithmeticProgression);
                    return true;
                }
                else
                {
                    burt_assert(!"ONE OR BOTH OF ITEM DOES NOT FOLLOW ARITHMETIC PROGRESSION STYLE");
                    return false;
                }
            }
            else // size_tag >=2
            {
                auto old_size = extSize(size_and_tag_together);
                // a1, a1+d, ...
                // 0    1    ...
                TItemType item_index_sz_add_0 = state.arithme_progr_array.a1 + state.arithme_progr_array.dstep * (old_size);
                TItemType item_index_sz_add_1 = state.arithme_progr_array.a1 + state.arithme_progr_array.dstep * (old_size + 1);

                if (item_index_sz_add_0 == itemA && item_index_sz_add_1 == itemB)
                {
                    size_and_tag_together = createSizeAndTag(old_size + 2, UsedArrayType::eArithmeticProgression);
                    return true;
                }
                else
                {
                    burt_assert(!"ONE OR BOTH OF ITEM DOES NOT FOLLOW ARITHMETIC PROGRESSION STYLE");
                    return false;
                }
            }
        }
        else  // if (isLongArray())
        {
            SizeTagType oldSize = size_and_tag_together;
            TItemType* oldItems = state.arb_array.first_pointer;
            TItemType* newFirst = (TItemType*) allocateBytes(sizeof(TItemType) * (oldSize + 2));
            memcpy(newFirst, oldItems, oldSize * sizeof(TItemType));
            newFirst[oldSize] = itemA;
            newFirst[oldSize + 1] = itemB;

            state.arb_array.first_pointer = newFirst;
            deallocateBytes(oldItems);

            size_and_tag_together += 2;
            return true;
        }
	}

    /* Depending on the array type(fixed - size, arithmetic progression, or long array),
    * this function retrieves the value at the given index.
    *
    * @param index The index of the item to retrieve.
    * @return The value at the specified index.
    */
	constexpr TItemType get(SizeTagType index) const noexcept
    {
        if (isTinyArray())
        {
            return state.fixed_size_array.items[index];
        }
        else if (isArithmProgressArray())
        {
            return state.arithme_progr_array.a1 + state.arithme_progr_array.dstep * index;
        }
        else // if (isLongArray())
        {
            return state.arb_array.first_pointer[index];
        }
	}

    /* Sets the value at the given index, but only if the value is consistent with the array type.
    *
    * @param index The index to set the value at.
    * @param value The value to set at the specified index.
    * @return `true` if the value was successfully set, `false` if it was not.
    */
    bool set(SizeTagType index, TItemType value) noexcept
	{
        if (isTinyArray())
        {
            state.fixed_size_array.items[index] = value;
            return true;
        }
        else if (isArithmProgressArray())
        {
            if (state.arithme_progr_array.a1 + state.arithme_progr_array.dstep * index == value)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else // if (isLongArray())
        {
            state.arb_array.first_pointer[index] = value;
            return true;
        }
	}

    /** This is a shorthand for calling the `get` function.
    *
    * @param index The index of the item to retrieve.
    * @return The value at the specified index.
    */
    constexpr TItemType operator [] (SizeTagType index) const noexcept
    {
        return get(index);
    }

    /*
    * Checks if the array is a tiny array.
    *
    * A tiny array has a size and tag that is smaller than or equal to the maximum size for a fixed-size array.
    *
    * @return `true` if the array is a tiny array, `false` otherwise.
    */
    constexpr bool isTinyArray() const noexcept 
    {
        // with bit-tricks (eArithmeticProgression has always upper bit equal to 1)
        return size_and_tag_together <= kITemsInFixedSizeArray;
    }

    /**
    * Checks if the array is a long array.
    * @return `true` if the array is a long array, `false` otherwise.
    */
    constexpr bool isLongArray() const noexcept
    {
        constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
        constexpr SizeTagType mask = (((TItemType)0x1) << (kBitInNode));
        return size_and_tag_together > kITemsInFixedSizeArray && size_and_tag_together < mask;
    }

    /**
    * Checks if the array is an arithmetic progression array.
    * @return `true` if the array is an arithmetic progression array, `false` otherwise.
    */
    constexpr bool isArithmProgressArray() const noexcept
    {
        constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
        constexpr SizeTagType mask = (((TItemType)0x1) << (kBitInNode));
        return size_and_tag_together >= mask;
    }

    /**
     * @brief Retrieves the common difference (step) of the arithmetic progression.
     *
     * This function returns the step size of the arithmetic progression if the array is of that type.
     *
     * @return The step size of the arithmetic progression. If array is not this the behaviour is undefined.
     */
    constexpr TItemType getArithmProgressStep() const noexcept
    {
        return state.arithme_progr_array.dstep;
    }

    /**
    * @brief Retrieves the first item of the arithmetic progression.
    *
    * This function returns the first item in the arithmetic progression if the array is of that type.
    *
    * @return The first item of the arithmetic progression.  If array is not this the behaviour is undefined.
    */
    constexpr TItemType getArithmProgressFirstItem() const noexcept
    {
        return state.arithme_progr_array.a1;
    }

    /**
    * @brief Retrieves the item from a tiny array at the specified index.
    *
    * @warning This function is only valid for tiny arrays.
    *
    * @param index The index of the item to retrieve.
    * @return The value at the specified index.
    */
    constexpr TItemType fromTinyArray(SizeTagType index) const noexcept {
        burt_assert(isTinyArray());
        return state.fixed_size_array.items[index];
	}

    /**
    * @brief Retrieves the item from a long array at the specified index.
    *
    * @warning This function is only valid for long arrays.
    *
    * @param index The index of the item to retrieve.
    * @return The value at the specified index.
    */
    constexpr TItemType fromLongArray(SizeTagType index) const noexcept
    {
        burt_assert(isLongArray());
        return state.arb_array.first_pointer[index];
    }


private:

    /**
    * @brief Creates a size and tag value at compile-time for the given size and array type.
    *
    * This function uses bit manipulation to construct a size and tag value for the array at compile-time.
    *
    * @param sz The size of the array.
    * @param array_type The type of array (fixed-size or arithmetic progression).
    * @return The size and tag value combined.
    */
    inline static consteval SizeTagTypeWithArrayType createSizeAndTagAtCompiletTime(SizeTagType sz, UsedArrayType array_type) noexcept
    {
        constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
        constexpr SizeTagType mask = (((TItemType)0x1) << (kBitInNode));

        burt_assert((sz & mask) == 0);
        static_assert( (int)UsedArrayType::eFixedSizeOrResizableArray == 0);

        if (array_type == UsedArrayType::eFixedSizeOrResizableArray)
        {
            return sz;
        }
        else
        {
            return sz | mask;
        }
    }

    /**
    * Creates a size and tag value for the given size and array type.
    *
    * This function uses bit manipulation to construct a size and tag value for the array.
    *
    * @param sz The size of the array.
    * @param array_type The type of array (fixed-size or arithmetic progression).
    * @return The size and tag value combined.
    */
    inline static constexpr SizeTagTypeWithArrayType createSizeAndTag(SizeTagType sz, UsedArrayType array_type) noexcept
    {
        constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
        constexpr SizeTagType mask = (((TItemType)0x1) << (kBitInNode));

        burt_assert((sz & mask) == 0);
        static_assert( (int)UsedArrayType::eFixedSizeOrResizableArray == 0);

        if (array_type == UsedArrayType::eFixedSizeOrResizableArray)
        {
            return sz;
        }
        else
        {
            return sz | mask;
        }
    }

    /**
    * Extracts the array type from a combined size and tag value.
    *
    * This function retrieves the array type (fixed-size/resizable or arithmetic progression) from the combined size and tag value.
    *
    * @param sz_and_type The combined size and tag value.
    * @return The array type.
    */
    inline static constexpr UsedArrayType extType(SizeTagTypeWithArrayType sz_and_type) noexcept
    {
        constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
        constexpr SizeTagType mask = (((TItemType)0x1) << (kBitInNode));

        if (sz_and_type & mask)
        {
            return UsedArrayType::eArithmeticProgression;
        }
        else
        {
            return UsedArrayType::eFixedSizeOrResizableArray;
        }
    }

    /**
    * @brief Extracts the size from a combined size and tag value.
    *
    * This function retrieves the size of the array from the combined size and tag value.
    *
    * @param sz_and_type The combined size and tag value.
    * @return The size of the array.
    */

    inline static constexpr SizeTagType extSize(SizeTagTypeWithArrayType sz_and_type) noexcept
    {
        constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
        constexpr SizeTagType mask = (((SizeTagType)0x1) << (kBitInNode));
        return sz_and_type & (~mask);
    }

private:
    SpecialArrayState state;
    SizeTagTypeWithArrayType size_and_tag_together;

    static_assert( (int)UsedArrayType::eFixedSizeOrResizableArray == 0, "this enum should be equal to zero for some bittricks");

private:
    /**
    * Allocates a specified number of bytes.
    *
    * This function allocates memory for the specified number of bytes and returns a pointer to the allocated memory.
    *
    * @param sz The number of bytes to allocate.
    * @return A pointer to the allocated memory.
    * @remark All memory allocation/deallaction for array happens via this entry point
    */
    inline static void* allocateBytes(size_t sz) noexcept
    {
        burt_assert(sz > 0);
        return malloc(sz);
    }

    /**
    * @brief Deallocates previously allocated memory.
    *
    * This function frees the memory previously allocated by `allocateBytes`.
    *
    * @param ptr A pointer to the memory to free.
    * @remark All memory allocation/deallaction for array happens via this entry point
    */
    inline static void deallocateBytes(void* ptr) noexcept
    {
        free(ptr);             // // If ptr is a null pointer, the function does nothing.
    }
};
