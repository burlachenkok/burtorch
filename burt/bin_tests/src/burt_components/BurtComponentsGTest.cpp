#include "gtest/gtest.h"
#include "burtcore/include/burtorch.h"


TEST(burt, BurtComponentsGTest)
{
	// TEST ARRAYS
	{
		SpecialArray<uint32_t> array;
		EXPECT_TRUE(array.size() == 0);
		EXPECT_TRUE(array.push_back(12));
		EXPECT_TRUE(array.isTinyArray());
		EXPECT_TRUE(array.size() == 1);
		EXPECT_TRUE(array[0] == 12);
		EXPECT_TRUE(array.push_back(16));
		EXPECT_TRUE(array[0] == 12);
		EXPECT_TRUE(array[1] == 16);
		EXPECT_TRUE(array.isTinyArray());
		EXPECT_TRUE(array.push_back_two_items(17, 18) == true);
		EXPECT_TRUE(array[0] == 12);
		EXPECT_TRUE(array[1] == 16);
		EXPECT_TRUE(array[2] == 17);
		EXPECT_TRUE(array[3] == 18);
		EXPECT_TRUE(array.isLongArray());
		EXPECT_TRUE(array.size() == 4);
		array.sysClearWithErase();
		EXPECT_TRUE(array.size() == 0);

		array.sysArrayResizeLossyWithoutAnyInit(5);
		EXPECT_TRUE(array.isLongArray());

		EXPECT_TRUE(array.size() == 5);
		EXPECT_TRUE(array.push_back(16));
		EXPECT_TRUE(array.size() == 6);
	}
}
