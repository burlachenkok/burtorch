#include "CompilerInfo.h"
#include "PlatformSpecificMacroses.h"

#include <sstream>
#include <assert.h>
#include <limits.h>

namespace
{
    template <class T>
    void makeZero(T& val)
    {
        val = 0;
    }

    int getType(int obj) {
        return 0;
    }

    int getType(void* obj) {
        return 1;
    }

    class SimpleBase {
    public:
        SimpleBase()
            : base1(0)
            , base2(0)
        {}

        int base1;
        int base2;
    };

    class Simple final: public SimpleBase {
    public:
        Simple()
            : der1(0)
            , der2(0)
        {}
        int der1;
        int der2;
    };

}

namespace burt {

    std::string compilerCppVersion() {
        std::stringstream str;
#ifdef __cplusplus
#if __cplusplus == 199711L
        str << "C++ 1997/2003 (" << __cplusplus << ")";
#elif __cplusplus == 201103L
        str << "C++ 2011 (" << __cplusplus << ")";
#elif __cplusplus == 201402L
        str << "C++ 2014 (" << __cplusplus << ")";
#elif __cplusplus == 201703L
        str << "C++ 2017 (" << __cplusplus << ")";
#elif __cplusplus == 202002L
        str << "C++ 2020 (" << __cplusplus << ")";
#elif __cplusplus == 202302L
        str << "C++ 2023 (" << __cplusplus << ")";
#else
        str << "C++ (" << __cplusplus << ")";
#endif
#elif defined(__STDC__)
#if defined(__STDC_VERERSION__) &&__STDC_VERERSION__ > 199901L
        str << "C99 standard";
#elif defined(__STDC_VERERSION__) &&__STDC_VERERSION__ > 199409L
        str << "C89 with additions 1";
#else
        str << "C89";
#endif
#else
        str << "C not standardized";
#endif
        str << "/";
        str << BURT_COMPILER_NAME << "/" << BURT_COMPILER_VERSION_STRING_LONG;
        return str.str();
    }

    bool compilerSizeOfEmptyIsNotNull() {
        struct Empty {};
        assert(sizeof(Empty) != 0);
        return (sizeof(Empty) != 0);
    }

    bool compilerDifferentObjectsAddrDiffer() {
        struct Empty {};
        Empty a, b;
        assert(&a != &b);
        return &a != &b;
    }

    bool compilerOptimizedEmptyBaseClass() {
        struct Empty {};
        struct X : Empty {
            int a;
        };
        X obj = X();
        void* p1 = &obj;
        void* p2 = &obj.a;
        return (p1 == p2);
    }

    bool compilerIsCharTypeSigned()
    {
        return CHAR_MIN == SCHAR_MIN;
    }

    bool compilerSupportReferenceCollapsing()
    {
        int i = 0;
        makeZero<int&>(i); // compile-time error in reference collapsing is not support
        return true;
    }

    const char* getCrtNullptrType()
    {
        switch (getType(nullptr))
        {
        case 0:
            return "INTEGER";
        case 1:
            return "VOID_PTR";
        }
        return "UNDEFINE";
    }

    bool isCharConsistOf8Bits()
    {
        return CHAR_BIT == 8;
    }

    bool isLongDoubleSameAsDouble()
    {
        return sizeof(long double) == sizeof(double);
    }

    bool memLayoutIsMatryoshka()
    {
        Simple a;
        void* simpleAddress = &a;
        void* simpleBaseAddress = ((SimpleBase*)&a);
        void* simpleBaseAddressMember1 = &a.base1;

        if (simpleBaseAddress == simpleAddress && simpleAddress == simpleBaseAddressMember1)
            return true;
        else
            return false;
    }

    bool isByteOrderRight2Left()
    {
        union {
            long testWord;
            char testWordInBytes[sizeof(long)];
        } u = {};

        u.testWord = 1;
        bool rightToLeft = false;
        if (u.testWordInBytes[0] == 1)
        {
            rightToLeft = true;
        }
        else if (u.testWordInBytes[sizeof(long) - 1] == 1)
        {
            rightToLeft = false;
        }
        else if (u.testWordInBytes[sizeof(long) - 1] == 0)
        {
            assert(!"Addressing is strange! For some reasons it is not right-to-left or left-to-right");
        }
        return rightToLeft;
    }

    bool isByteOrderLeft2Right()
    {
        return !isByteOrderRight2Left();
    }
}
