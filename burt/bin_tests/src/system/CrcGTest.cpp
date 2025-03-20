#include "burt/system/include/digest/Crc.h"

#include "gtest/gtest.h"

#include <vector>
#include <string>

#include <stdint.h>

TEST(burt, CrcGTest)
{
	std::string s;
	s = "hello world!";
	EXPECT_EQ(0xfc4b3d92, burt::crc32(s.c_str(), s.length(), burt::crc32Seed()));
	s.clear();

	EXPECT_EQ(0xFFFFFFFF, burt::crc32(s.c_str(), s.length(), burt::crc32Seed()));
	s = "Wow!";
	EXPECT_EQ(0xe7fd648a, burt::crc32(s.c_str(), s.length(), burt::crc32Seed()));

	const std::string subStrA = "hello";
	const std::string subStrB = " world!";

	uint32_t crcByParts = burt::crc32Seed();
	crcByParts = burt::crc32(subStrA.c_str(), subStrA.length(), crcByParts);
	crcByParts = burt::crc32(subStrB.c_str(), subStrB.length(), crcByParts);
	EXPECT_EQ(0xfc4b3d92, crcByParts);
	crcByParts = burt::crc32(0, 0, crcByParts);
	EXPECT_EQ(0xfc4b3d92, crcByParts);
}
