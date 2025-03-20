#include "burt/copylocal/include/Data.h"

#include "gtest/gtest.h"

#include <stdint.h>

TEST(burt, DataGTest)
{
    {
	    burt::Data container(0, 0);
        EXPECT_EQ(0, container.getTotalLength());
        EXPECT_EQ(0, container.getResidualLength());
        EXPECT_TRUE(container.isEmpty());
        EXPECT_EQ(0, container.getPos());
        EXPECT_EQ(0, container.seekStart(2));
    }

    {
        int32_t values[] = {-1, 20, 30, 40};

        burt::Data container(values, sizeof(values));
        EXPECT_EQ(sizeof(values), container.getTotalLength());
        EXPECT_EQ(sizeof(values), container.getResidualLength());
        EXPECT_FALSE(container.isEmpty());
        EXPECT_EQ(0, container.getPos());
        EXPECT_EQ(container.getInt32(),-1);
        EXPECT_EQ(container.getInt32(),20);
        EXPECT_EQ(container.getInt32(),30);
        EXPECT_EQ(container.getInt32(),40);
        EXPECT_TRUE(container.isEmpty());
        EXPECT_TRUE(container.isRawMemoryBeenAllocated());
        EXPECT_EQ(sizeof(values), container.getTotalLength());
        EXPECT_EQ(0, container.getResidualLength());

        EXPECT_NE(static_cast<void*>(container.getPtr()), static_cast<void*>(values));
        EXPECT_EQ(container.seekStart(0), 0);
        EXPECT_EQ(sizeof(values), container.getResidualLength());
        EXPECT_EQ(container.seekStart(2), 2);
        EXPECT_EQ(sizeof(values) - 2, container.getResidualLength());
        EXPECT_EQ(container.seekCur(-2), 0);
        EXPECT_EQ(sizeof(values), container.getResidualLength());
        EXPECT_EQ(container.getPos(), 0);
        EXPECT_EQ(container.seekCur(-12), 0);
        EXPECT_EQ(container.getPos(), 0);
        EXPECT_EQ(container.seekCur(3), 3);
        EXPECT_EQ(container.getPos(), 3);
        EXPECT_EQ(container.seekCur(1), 4);
        EXPECT_EQ(container.getPos(), 4);
        EXPECT_TRUE(container.isRawMemoryBeenAllocated());
    }

    {
        uint16_t values[] = { uint16_t(- 1), 20, 30, 40, 60000};
        burt::Data container(values, sizeof(values), burt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);

        EXPECT_EQ(sizeof(values), container.getResidualLength());
        EXPECT_EQ(sizeof(values), container.getTotalLength());
        EXPECT_EQ((uint8_t*)values, container.getPtr());

        values[0] = 2;
        EXPECT_EQ(2, container.getUint16());
        EXPECT_EQ(20, container.getUint16());
        EXPECT_EQ(3 * sizeof(uint16_t), container.getResidualLength());

        {
            burt::Data containerWithZeros((void*)1, 4, burt::Data::MemInitializedType::eAllocAndInitilizedWithZero);
            {
                uint8_t values[] = {0xff, 0xff, 0xff, 0xff};
                containerWithZeros.getTuple3(values[0], values[1], values[2]);
                for (int i = 0; i < 3; ++i)
                    EXPECT_EQ(0, values[i]);
                EXPECT_EQ(0xff, values[3]);
            }

			{
				uint8_t values[] = {0xff, 0xff, 0xff, 0xff};
				EXPECT_EQ(0, containerWithZeros.seekCur(-10000));
				EXPECT_TRUE(containerWithZeros.getTupleN(values, 4));
				EXPECT_TRUE(containerWithZeros.isEmpty());
				for (int i = 0; i < 4; ++i)
					EXPECT_EQ(0, values[i]);
			}

            EXPECT_NE(nullptr, containerWithZeros.getPtr());
            EXPECT_TRUE(containerWithZeros.isRawMemoryBeenAllocated());
        }
    }

	{
        uint8_t values[] = {1U, 20U, 30U, 40U, uint8_t(60000U)};
		burt::Data container(values, sizeof(values), burt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
		{
			burt::Data container2(container);
			EXPECT_TRUE(container2.isRawMemoryBeenAllocated());
			EXPECT_NE(container2.getPtr(), values);
		}
		EXPECT_EQ(container.getPtr(), values);
		EXPECT_EQ(1, container.getByte());
		EXPECT_EQ(20, container.getByte());
	}

	{
        uint8_t values[] = {1U, 20U, 30U, 40U, uint8_t(60000U)};
		burt::Data container(values, sizeof(values), burt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
		burt::Data container2(0,0);

		EXPECT_EQ(0, container2.getTotalLength());
		EXPECT_EQ(0, container2.getResidualLength());
		
		burt::Data container3(0, 10, burt::Data::MemInitializedType::eAllocAndInitilizedWithZero);
		EXPECT_EQ(0, container3.getByte());
		container3 = container2; // now both containers should contain zero buffer
		EXPECT_EQ(0, container2.getTotalLength());
		EXPECT_EQ(0, container2.getResidualLength());
		EXPECT_EQ(NULL, container2.getPtr());
	}

    {
        uint64_t values[] = {18446744073709551615ULL, 20, 4294967297ULL};
        burt::Data container(values, sizeof(values), burt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);

        EXPECT_EQ(18446744073709551615ULL, container.getUint64());
        EXPECT_EQ(20ULL, container.getUint64());
        EXPECT_EQ(4294967297ULL, container.getUint64());
    }
}
