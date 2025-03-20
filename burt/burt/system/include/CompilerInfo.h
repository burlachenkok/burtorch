/** @file
 * C++ check compiler routines
 */

#pragma once

#include <string>

namespace burt
{
    /** Used compiler version
    * @return string-with-compiler-info
    */
    std::string compilerCppVersion();

    /** Size of an empty class is not zero due to ensure that the addresses
    * of two different objects will be different. Check it.
    * @return true, if size of empty class is not 0
    */
    bool compilerSizeOfEmptyIsNotNull();

    /** Check that different, but possible identical objects by content, have different addresses in memory
    * @return true, if different object have different addresses
    */
    bool compilerDifferentObjectsAddrDiffer();

    /** Empty base class need not be represented by a separate byte.
    * It allows a programmer to use empty classes to represent very simple concepts without overhead.
    * Some current compilers provide this "empty base class optimization".
    * @return empty base class optimization support
    */
    bool compilerOptimizedEmptyBaseClass();

    /** Is 'char' equal to 'signed char' or to 'unsigned char'
    * @return true is char type us signed
    */
    bool compilerIsCharTypeSigned();

    /** Is compiler support reference collapsing in template functions. T&& => T& collapsing is allowable
    * @return true if compiler support it
    */
    bool compilerSupportReferenceCollapsing();

    /** Get type of nullptr variable from stddef.h
    * @return type of nullptr crt object-like define variable
    */
    const char* getCrtNullptrTypeInfo();

    /** Check that char type consists of 8 bit
    * @return true if it so
    */
    bool isCharConsistOf8Bits();

    /** Check that size of long double equal to size of double
    * @return true if it so
    */
    bool isLongDoubleSameAsDouble();

    /** Check that without "vptr" base class, derived class, and first member of base class -- share identical address
    * @return string-with-compiler-info
    */
    bool memLayoutIsMatryoshka();

    /** Byte order in processor is right 2 left (little endian, Intel)
    * @return true, if byte order is right to left
    */
    bool isByteOrderRight2Left();

    /** Byte order in processor is left 2 right (big endian, Motorola)
    * @return true, if byte order is right to left
    */
    bool isByteOrderLeft2Right();
}
