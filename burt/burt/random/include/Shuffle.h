/** @file
* Math statistic routines: expectation, variance, correlation, etc.
*/
#pragma once

#include "burt/mathroutines/include/SimpleMathRoutines.h"
#include "burt/random/include/Shuffle.h"
#include "burt/copylocal/include/Copier.h"

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#if BURT_WINDOWS
    #include <malloc.h>
#endif

#if BURT_LINUX || BURT_MACOS
    #include <alloca.h>
#endif

namespace burt
{
    /** Perform Knuth Shuffle of [first, end) and receive uniformly random permutation (proved by Fisher-Yates, 1938) in linear time
    * @param container container elements of which you want to shuffle
    * @param earlyStopAfterShuffleFirstK possible hint that you want only to shuffle first K items. If you want to shuffle whole container provide size().
    * @param generator which is used to generate integer uniformly at random in [0, N] or .
    * @remark shuffling is performing in place
    * @remark scenario when earlyStopAfterShuffleFirstK can be used is generate subset of [0,...n-1] number with cardinality k.
    */
    template<class Container, class PseudoRandomGenerator>
    forceinline_ext void shuffle(Container& container,
                                 size_t earlyStopAfterShuffleFirstK,
                                 PseudoRandomGenerator& generator)
    {
        using FloatType = double;

        size_t n = container.size();

        size_t n_minus_i_minus_one = n - 1;
       
        assert(earlyStopAfterShuffleFirstK <= container.size());

#if 0
        constexpr size_t kBatchSize = 16;
        FloatType randomNumbersStorage[kBatchSize];

        size_t items = burt::roundToNearestMultipleDown<kBatchSize>(earlyStopAfterShuffleFirstK);
        size_t i = 0;

        for (; i < items; i += kBatchSize)
        {
            generator.generateBatchOfRealsInUnitInterval(randomNumbersStorage, kBatchSize);
            
            for (size_t ii = 0; ii < kBatchSize; ++ii, n_minus_i_minus_one--)
            {
                size_t r = i + ii + roundToNearestInt<decltype(i)>(randomNumbersStorage[ii] * n_minus_i_minus_one);
                
                if (i + ii != r)
                {
                    burt::CopyHelpers::swapDifferentObjects(container[i + ii], container[r]);
                }
            }
        }

        for (; i < earlyStopAfterShuffleFirstK; ++i, --n_minus_i_minus_one)
        {
            size_t r = i + roundToNearestInt<decltype(i)>(generator.generateRealInUnitInterval() * n_minus_i_minus_one);
            if (r != i)
            {
                burt::CopyHelpers::swapDifferentObjects(container[i], container[r]);
            }
        }

#else
        for (size_t i = 0; i < earlyStopAfterShuffleFirstK; ++i, --n_minus_i_minus_one)
        {
            // Common Bug: choose r from [0, N-1] when N is length of input sequence.
            // And due to R.Sedgewick, Princeton this is not correct.
            //  Correct Variant-1: select r in [i, N-1] 
            //  Correct variant-2: select r in [0, i]

            const size_t r_offset = roundToNearestInt<decltype(i)>(generator.template generateRealInUnitInterval<FloatType> () * n_minus_i_minus_one);

            // r \in [0, n-1] first iteration
            // r \in [1, n-1] second iteration

            if (r_offset != 0)
            {
                const size_t r = i + r_offset;
                burt::CopyHelpers::swapDifferentObjects(container[i], container[r]);
            }
        }
#endif
    }

    /** Perform Knuth Shuffle of [first, end) and receive uniformly random permutation (proved by Fisher-Yates, 1938) in linear time
    * @param container container elements of which you want to shuffle
    * @param generator which is used to generate integer uniformly at random in [0, N] or .
    * @remark shuffling is performing in place
    */
    template<class Container, class PseudoRandomGenerator>
    forceinline_ext void shuffle(Container& container, PseudoRandomGenerator& generator)
    {
        using FloatType = double;
        
        size_t n = container.size();

        size_t n_minus_i_minus_one = n - 1;

#if 0
        constexpr size_t kBatchSize = 16;
        FloatType randomNumbersStorage[kBatchSize];
        
        size_t items = burt::roundToNearestMultipleDown<kBatchSize>(n);
        size_t i = 0;

        for (; i < items; i += kBatchSize)
        {
            generator.generateBatchOfRealsInUnitInterval(randomNumbersStorage, kBatchSize);

            for (size_t ii = 0; ii < kBatchSize; ++ii, n_minus_i_minus_one--)
            {
                size_t r = i + ii + roundToNearestInt<decltype(i)>(randomNumbersStorage[ii] * n_minus_i_minus_one);

                if (i + ii != r)
                {
                    burt::CopyHelpers::swapDifferentObjects(container[i + ii], container[r]);
                }
            }
        }

        for (; i < n; ++i, --n_minus_i_minus_one)
        {
            size_t r = i + roundToNearestInt<decltype(i)>(generator.generateRealInUnitInterval() * n_minus_i_minus_one);
            if (i != r)
            {
                burt::CopyHelpers::swapDifferentObjects(container[i], container[r]);
            }
        }

#else
        for (size_t i = 0; i < n; ++i, --n_minus_i_minus_one)
        {
            // Common Bug: choose r from [0, N-1] when N is length of input sequence.
            // And due to R.Sedgewick, Princeton this is not correct.
            //   Correct Variant - 1: Select "r" in [i, N-1] 
            //   Correct variant - 2: Select "r" in [0, i]           
            const size_t r_offset = roundToNearestInt<decltype(i)>(generator.template generateRealInUnitInterval<FloatType>() * n_minus_i_minus_one);
            
            // r \in [0, n-1] first iteration
            // r \in [1, n-1] second iteration
            if (r_offset != 0)
            {
                const size_t r = i + r_offset;
                burt::CopyHelpers::swapDifferentObjects(container[i], container[r]);
            }
        }
#endif
    }
}
