/** @file
* C++ CRC implementation
*/
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace burt
{
    /** Calculate the CRC checksum for polynomial 0x4C11DB7(CRC-32-IEEE 802.3).
    * @param buf input buffer
    * @param size input buffer size
    * @param crc initial or previous crc value
    * @return checksum value
    * @remark This algorithm is used for counting in: V.42, MPEG-2, PNG, POSIX cksum, ZIP (last determined from experience).
    * @remark There is no XorOut before the function exits. Formally, we can assume that XorOut(0) occurs.
    */
    uint32_t crc32(const void* buf, size_t size, uint32_t crc);

    uint32_t crc32Seed();
}
