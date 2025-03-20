/** @file
* C++ Templates based memcpy, memmove, and swap operation.
*/
#pragma once

#include <memory.h>
#include <string.h>
#include <type_traits>
#include <utility>
#include <assert.h>
#include <stddef.h>
#include <assert.h>

#include "burt/system/include/PlatformSpecificMacroses.h"

namespace burt
{
    struct CopyHelpers
    {
        /* Swap a and b provided by pointers. Call it only when "a" and "b" is definetly different objects.
        * @param a pointer to first item
        * @param b pointer to second item
        */
        template<class Data>
        static void swapDifferentObjects(Data* restrict_ext a, Data* restrict_ext b) noexcept {
            optimization_assert(a != b);

            Data tmp(std::move(*a));
            *a = std::move(*b);
            *b = std::move(tmp);
        }

        /* Swap a and b provided by references
        * @param a pointer to first item
        * @param b pointer to second item
        */
        template<class Data>
        static void swapDifferentObjects(Data& restrict_ext a, Data& restrict_ext b) noexcept {
            optimization_assert(&a != &b);

            Data tmp(std::move(a));
            a = std::move(b);
            b = std::move(tmp);
        }

        /* Swap a and b provided by pointers
        * @param a pointer to first item
        * @param b pointer to second item
        */
        template<class Data>
        static void swap(Data* a, Data* b) noexcept {
            Data tmp(std::move(*a));
            *a = std::move(*b);
            *b = std::move(tmp);
        }

        /* Swap a and b provided by references
        * @param a pointer to first item
        * @param b pointer to second item
        */
        template<class Data>
        static void swap(Data& a, Data& b) noexcept {
            Data tmp(std::move(a));
            a = std::move(b);
            b = std::move(tmp);
        }
        
        /* Copy to "output" elements in [first, end)
        * @param output iterator to output. Should support dereference, and prefix increment
        * @param first iterator to input. Should support dereference, and prefix increment
        * @param end iterator of input where need to stop.       
        */
        template<class Iterator2Write, class Iterator2Read>
        static void copyWithIterators(Iterator2Write output, Iterator2Read first, Iterator2Read end) {
            while (first != end)
            {
                *output = *first;
                ++output;
                ++first;
            }
        }

        /* Copy to "output" elements in [first, end)
        * @remark If the "[output, output + itemsToCopy)" overlap "[firstToRead, firstToRead + itemsToCopy)" the behavior is undefined
        */
        template<class Data>
        static void copy(Data* restrict_ext output, const Data* restrict_ext firstToRead, size_t itemsToCopy) 
        {
            assert(output >= firstToRead + itemsToCopy || output + itemsToCopy <= firstToRead);
            
            if constexpr (std::is_trivially_copyable<Data>::value)
            {
                memcpy(output, firstToRead, itemsToCopy * sizeof(Data));
            }
            else
            {
                while (itemsToCopy > 0)
                {
                    *output = *firstToRead;
                    ++output;
                    ++firstToRead;
                    --itemsToCopy;
                }
            }
        }

        /* Move to "output" elements in [first, end)
        * @remark Copying takes place as if an intermediate buffer were used, allowing the destination and source to overlap.
        */
        template<class Data>
        static void move(Data* output, Data* firstToRead, size_t itemsToMove)
        {
            if constexpr (std::is_trivially_copyable<Data>::value)
            {
                // https://stackoverflow.com/questions/16542291/what-is-the-difference-between-is-trivially-copyable-and-is-trivially-copy-const
                // The former tests for the trivially copyable property, which in few words means that the type is memcpy-safe
                memmove(output, firstToRead, itemsToMove * sizeof(Data));
            }
            else if (itemsToMove == 1)
            {
                *output = *firstToRead;
            }
            else if (output < firstToRead || output >= firstToRead + itemsToMove)
            {
                while (itemsToMove > 0)
                {
                    *output = *firstToRead;
                    ++output;
                    ++firstToRead;
                    --itemsToMove;
                }
            }
            else // output >= firstToRead
            {
                // output lie in [first, end). To make move correctly do it from the end of the list
                size_t offset = output - firstToRead;
                Data* end = firstToRead + itemsToMove;
                
                while (itemsToMove > 0)
                {
                    *(end + offset - 1) = *(end - 1);
                    --end;
                    --itemsToMove;
                }
            }
        }

        /* Get length of the array [first, end)
        * @param first iterator to elements in the range, including first.
        * @param end iterator to elements in the range, excluding end.
        */
        template<class Iterator>
        static size_t arrayLength(Iterator first, Iterator end) {
            return end - first;
        }

        /* Get length of the array [first, end)
        * @param first iterator to elements in the range, including first.
        * @param end iterator to elements in the range, excluding end.
        */
        template<class Data>
        static size_t arrayLength(const Data* first, const Data* end) {
            return end - first;
        }
    };
}
