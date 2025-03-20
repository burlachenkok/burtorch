#include "burt/system/include/digest/Md5.h"

#include "gtest/gtest.h"

#include <vector>
#include <string>

TEST(burt, Md5GTest)
{
	{
		const std::string s = "hello world!";	
		unsigned char md5Digest[16] = {0};
		char md5String[33] = {0};

		EXPECT_TRUE(burt::getMd5Digest(md5Digest, s.c_str(), s.length()));
		burt::fromMd5DigestToMd5StringLowerCase(md5String, md5Digest);
		EXPECT_STREQ("fc3ff98e8c6a0d3087d515c0473f8677", md5String);
		
		burt::fromMd5DigestToMd5StringUpperCase(md5String, md5Digest);
		EXPECT_STRCASEEQ("fc3ff98e8c6a0d3087d515c0473f8677", md5String);

		EXPECT_EQ(0xFC, md5Digest[0]);
		EXPECT_EQ(0x3F, md5Digest[1]);
	}
	{
		const std::string s = "Wow!";	
		unsigned char md5Digest[16] = {12};
		char md5String[33] = {0};
		
		EXPECT_TRUE(burt::getMd5Digest(md5Digest, s.c_str(), 0));
		burt::fromMd5DigestToMd5StringLowerCase(md5String, md5Digest);
		EXPECT_STREQ("00000000000000000000000000000000", md5String);

		EXPECT_TRUE(burt::getMd5Digest(md5Digest, s.c_str(), s.length()));
		burt::fromMd5DigestToMd5StringLowerCase(md5String, md5Digest);
		EXPECT_STRCASEEQ("359878442cf617606802105e2f439dbc", md5String);
	}
}
