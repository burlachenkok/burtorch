/** @file
 * Cross-platform implementation of md5 digest calculation
 */

#pragma once

#include <stddef.h>

namespace burt
{
    static constexpr size_t kMd5Digits = 16;
    static constexpr size_t kMd5Chars = 32 + 1;

    /** 128 bit (16 bytes) checksum calculation using the MD5 algorithm according to RFC1321
    * @param digest output parameter with MD5 checksum written in 16 bytes
    * @param inputData input data
    * @param inputDataLength size of input data
    * @return true - if all is well
    */
    bool getMd5Digest(unsigned char digest[kMd5Digits],
                      const void* inputData,
                      size_t inputDataLength);

    /** Converting the calculated data checksum according to the MD5 algorithm according to rfc1321 into a readable string with upper case hex numbers
    * @param digestString output string of 32 ascii characters with last 33 characters set to zero
    * @param digest output parameter with MD5 checksum written in 16 bytes
    */
    void fromMd5DigestToMd5StringUpperCase(char digestString[kMd5Chars], unsigned char digest[kMd5Digits]);

    /** Converting the calculated data checksum according to the MD5 algorithm according to rfc1321 into a readable string with lower case hex numbers
    * @param digestString output string of 32 ascii characters with last 33 characters set to zero
    * @param digest output parameter with MD5 checksum written in 16 bytes
    */
    void fromMd5DigestToMd5StringLowerCase(char digestString[kMd5Chars], unsigned char digest[kMd5Digits]);
}
