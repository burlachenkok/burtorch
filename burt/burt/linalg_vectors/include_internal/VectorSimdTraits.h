#pragma once

#if SUPPORT_CPU_SSE2_128_bits || SUPPORT_CPU_AVX_256_bits || SUPPORT_CPU_AVX_512_bits
    //================================================================================================================//
    // SSE2   -- SIMD registers with length 128 bit registers. Tuples of (4 * 32bits), 16 XMM ISA front-end registers.
    // AVX2   -- SIMD registers with length 256 bit registers. Tuples of (8 * 32bits),  16 YMM ISA front-end registers.
    // AVX512 -- SIMD registers with length 512 bit registers. Tuples of (16 * 32bits), 32 ZMM ISA front-end registers.
    //================================================================================================================//
    // We have patched "VLC library" to select MAX_VECTOR_SIZE based on current available ISA.
    //================================================================================================================//
    #include "burt/3rdparty/vectorclass/vectorclass.h"
    #include "burt/3rdparty/vectorclass/vectormath_exp.h"
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    // From ARMv7 there is a support of ARM Neon (known as Advacned SIMD). Support integers and float inside Instructions itself.
    // Neon consist of 32 V0,...,V31 registers in front-end of ARM. Each register is 128 bit load.
    // For such extension and another extension we leveraga (ongoing actually) compiler support.
    #include "burt/linalg_vectors/include_internal/CppSimdWrapper.h"
#endif

#include <stdint.h>
#include <stddef.h>

namespace burt
{
    /** Vector CPU extension
    */
    enum class VectorExtensionForCPU
    {
        NONE,
        SSE2_128_bits = 1,  ///< SSE2 is CPU extension typically available in i386 CPUs. Support of it means at least there are XMM[0-7] 128 bit registers.
        AVX_256_bits  = 2,  ///< AVX-256 or AVX2 or Haswell New Instructions is extension of SSE2 to support 256 bits vector registers.
        AVX_512_bits  = 3,  ///< AVX-512 is 512-bit extensions to the AVX-256.
        CPP_TS_V2_SIMD = 4  ///< Utilize C++ 20 SIMD extension (std::experimental::simd)
    };

#if SUPPORT_CPU_SSE2_128_bits
    /** Inline variable which denote to a supported CPU extention.
    */
    static inline constexpr VectorExtensionForCPU cpu_extension = VectorExtensionForCPU::SSE2_128_bits;
#elif SUPPORT_CPU_AVX_256_bits
    /** Inline variable which denote to a supported CPU extention.
    */
    static inline constexpr VectorExtensionForCPU cpu_extension = VectorExtensionForCPU::AVX_256_bits;
#elif SUPPORT_CPU_AVX_512_bits
    /** Inline variable which denote to a supported CPU extention.
    */
    static inline constexpr VectorExtensionForCPU cpu_extension = VectorExtensionForCPU::AVX_512_bits;
#elif SUPPORT_CPU_CPP_TS_V2_SIMD
    /** Inline variable which denote to a supported CPU extention.
    */
    static inline constexpr VectorExtensionForCPU cpu_extension = VectorExtensionForCPU::CPP_TS_V2_SIMD;

#else
    /** Inline variable which denote to a supported CPU extention.
    */
    static inline constexpr VectorExtensionForCPU cpu_extension = VectorExtensionForCPU::NONE;
#endif
    
    /** Is any SIMD extension are supported at compile time
    * @return number of elements
    */
    inline consteval bool isSimdComputeSupportedAtCompileTime() {
        return cpu_extension != VectorExtensionForCPU::NONE;
    }
    
    /** How many items can be pulled into vector storage(items in vector)
    * @tparam TRegisterVecType type of used vector register 
    * @return number of elements
    */
    template<class TRegisterVecType>
    static consteval size_t getVecBatchSize()
    {
        return TRegisterVecType::size();
    }
    
#if 0
    
    enum class VectorElementType
    {
        bits = 1,
        bool_compact = 2,
        bool_broad = 3,
        int8_t = 4,
        uint8_t = 5,
        int16_t = 6,
        uint16_t = 7,
        int32_t = 8,
        uint32_t = 9,
        int64_t = 10,
        uint64_t = 11,
        fp16_half = 15,
        fp32 = 16,
        fp64 = 17
};
    
    /** Type of item in vector register
    * @tparam TRegisterVecType register type
    * @return Type of element in each lane of vector register
    */
    template<class TRegisterVecType>
    static consteval VectorElementType getElementType()
    {
        return (VectorElementType)TRegisterVecType::elementtype();
    }
#endif
    
    template<class TRegisterVecType>
    static consteval size_t getUnrollFactor()
    {
        // based on experiments with 1,2,3,4,5,6,7,8 the best unroll factor is just 2.
        return 2;
    }
    //========================================================================================================//
    template <class T, VectorExtensionForCPU extension>
    struct VectorSimdTraits
    {};
    //========================================================================================================//

#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<double, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec2d VecType;
    };
#endif

#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<double, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec4d VecType;
    };
#endif

#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<double, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec8d VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<double, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<double> VecType;
    };
#endif
    //========================================================================================================//
#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<float, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec4f VecType;
    };
#endif

#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<float, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec8f VecType;
    };
#endif

#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<float, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec16f VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<float, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<float> VecType;
    };
#endif
    //========================================================================================================//
#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<::int64_t, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec2q VecType;
    };
#endif

#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<::int64_t, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec4q VecType;
    };
#endif

#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<::int64_t, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec8q VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<::int64_t, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<::int64_t> VecType;
    };
#endif

#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<::uint64_t, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec2uq VecType;
    };
#endif

#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<::uint64_t, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec4uq VecType;
    };
#endif

#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<::uint64_t, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec8uq VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<::uint64_t, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<::uint64_t> VecType;
    };
#endif
    //========================================================================================================//
#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<::int32_t, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec4i VecType;
    };
#endif

#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<::int32_t, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec8i VecType;
    };
#endif

#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<::int32_t, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec16i VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<::int32_t, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<::int32_t> VecType;
    };
#endif

#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<::uint32_t, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec4ui VecType;
    };
#endif

#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<::uint32_t, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec8ui VecType;
    };
#endif

#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<::uint32_t, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec16ui VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<::uint32_t, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<::uint32_t> VecType;
    };
#endif
    //========================================================================================================//


    //========================================================================================================//
#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<::int16_t, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec8s VecType;
    };
#endif

#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<::int16_t, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec16s VecType;
    };
#endif

#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<::int16_t, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec32s VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<::int16_t, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<::int16_t> VecType;
    };
#endif

#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<::uint16_t, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec8us VecType;
    };
#endif


#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<::uint16_t, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec16us VecType;
    };
#endif


#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<::uint16_t, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec32us VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<::uint16_t, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<::uint16_t> VecType;
    };
#endif
    //========================================================================================================//

    //========================================================================================================//
#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<::int8_t, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec16c VecType;
    };
#endif


#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<::int8_t, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec32c VecType;
    };
#endif


#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<::int8_t, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec64c VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<::int8_t, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<::int8_t> VecType;
    };
#endif

#if SUPPORT_CPU_SSE2_128_bits
    template <>
    struct VectorSimdTraits<::uint8_t, VectorExtensionForCPU::SSE2_128_bits> {
        typedef Vec16uc VecType;
    };
#endif

#if SUPPORT_CPU_AVX_256_bits
    template <>
    struct VectorSimdTraits<::uint8_t, VectorExtensionForCPU::AVX_256_bits> {
        typedef Vec32uc VecType;
    };
#endif

#if SUPPORT_CPU_AVX_512_bits
    template <>
    struct VectorSimdTraits<::uint8_t, VectorExtensionForCPU::AVX_512_bits> {
        typedef Vec64uc VecType;
    };
#endif

#if SUPPORT_CPU_CPP_TS_V2_SIMD
    template <>
    struct VectorSimdTraits<::uint8_t, VectorExtensionForCPU::CPP_TS_V2_SIMD> {
        typedef CppSimdWrapper<::uint8_t> VecType;
    };
#endif
    //========================================================================================================//
}
