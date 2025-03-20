#include "burt/system/include/digest/Md5.h"
#include "burt/system/include/digest/impl/Md5Impl.h"

#include <stddef.h>

namespace
{
    constexpr char hexLettersUpperCase[] = 
    { '0', '1', '2', '3', '4',
      '5', '6', '7', '8', '9',
      'A', 'B', 'C', 'D', 'E', 'F' 
    };

    constexpr char hexLettersLowerCase[] =
    { '0', '1', '2', '3', '4',
      '5', '6', '7', '8', '9',
      'a', 'b', 'c', 'd', 'e', 'f'
    };
}

namespace burt
{
    bool getMd5Digest(unsigned char digest[kMd5Digits], const void* inputData, size_t inputDataLength)
    {
        if (inputDataLength == 0)
        {
            for (size_t i = 0; i < 16; ++i) {
                digest[i] = 0;
            }
            return true;
        }

        static_assert(sizeof(UINT4) == 4);
        static_assert(sizeof(UINT2) == 2);

        MD5_CTX context;
        MD5Init(&context);
        MD5Update(&context, (unsigned char*)inputData, inputDataLength);
        MD5Final(digest, &context);
        return true;
    }

    void fromMd5DigestToMd5StringUpperCase(char digestString[kMd5Chars], unsigned char digest[kMd5Digits])
    {
        digestString[kMd5Chars - 1] = '\0';

        static_assert(kMd5Chars == kMd5Digits * 2 + 1);

        for (size_t i = 0; i < kMd5Digits; ++i)
        {
            const unsigned char lowOct4 = digest[i] & 0xF;
            const unsigned char highOct4 = (digest[i] >> 4) & 0xF;

            digestString[2 * i + 0] = hexLettersUpperCase[highOct4];
            digestString[2 * i + 1] = hexLettersUpperCase[lowOct4];
        }
    }

    void fromMd5DigestToMd5StringLowerCase(char digestString[kMd5Chars], unsigned char digest[kMd5Digits])
    {
        digestString[kMd5Chars - 1] = '\0';

        static_assert(kMd5Chars == kMd5Digits * 2 + 1);

        for (size_t i = 0; i < kMd5Digits; ++i)
        {
            unsigned char lowOct4 = digest[i] & 0xF;
            unsigned char highOct4 = (digest[i] >> 4) & 0xF;

            digestString[2 * i + 0] = hexLettersLowerCase[highOct4];
            digestString[2 * i + 1] = hexLettersLowerCase[lowOct4];
        }
    }
}
