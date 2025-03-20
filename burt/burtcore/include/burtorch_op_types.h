#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"

#include "burtcore/include/burtorch_config.h"

#include <stdint.h>

enum class OpHint : uint8_t
{
	eOpNoHints = 0,
	eOpHintNotEvaluateValue = 1
};

enum class OpType : uint8_t
{
    eLeaf   = 0,         ///< No Operation. Leaf consant node. stop backpropagation.
    eRelu   = 1,
    eTanh   = 2,         ///< Hyperbolic tangent
    eExp    = 3,         ///< Exp
    eNegLog = 4,         ///< -log_e(x)
    eSigmoid = 5,
    eInv = 6,            ///< For x: 1/x
    eSqr = 7,            ///< For x: x*x
    eCub = 8,            ///< For x: x*x*x
    eLog = 9,            ///< log_e(x)
    eSqrt = 10,          ///< For x: sqrt(x)
    eInvSqrt = 11,       ///< For x: 1/sqrt(x)

    eBinaryAdd  = 12,           ///< Add Operation
    eBinarySub  = 13,           ///< Sub Operation
    eBinaryMult = 14,           ///< Multiply Operation
    eBinaryMultByConst = 15,    ///< Multiply Operation

    eBinaryDiv  = 16,           ///< Division Operation
    eBinaryMean = 17,
    eBinaryAddSquares = 18,
    eBinaryMeanSquares = 19,
    eBinaryNegativeMean = 20,

    eAddVarying  = 21,         ///< Add Operation
    eSubVarying  = 22,         ///< Sub Operation
    eMulVarying = 23,          ///< Multiply Operation
    eMeanVarying = 24,
    eSumOfSquaresVarying = 25,
    eMeanSquaresVarying = 26,
    eNegativeMeanVarying = 27,

    eInnerProductNoBias = 28,
    eInnerProductWithBias = 29,
    eOpsCount
};

/**
 * @brief Structure describing an operation in the graph.
 */
struct OperationDescriptor
{
    unsigned int node_gc_counter : 8;              ///< 8 bits: Node index, indicating if node can be deleted
    unsigned int visiting_number_for_backprop : 3; ///< 3 bits: Type of visit for backpropagation
    unsigned int op_type : 5;                      ///< 5 bits: Operation type, sufficient for up to 32 operations
};

/**
 * @brief Creates a valid operation descriptor for a given operation type.
 *
 * @param operation The operation type.
 * @return A valid operation descriptor.
 */
constexpr inline OperationDescriptor createValidOpDescriptor(OpType operation) noexcept
{
    OperationDescriptor descr;
    descr.op_type = (unsigned int)operation;

#if BURTORCH_USE_GC_COUNTERS
    descr.node_gc_counter = 1;  ///< Enable garbage collection counter if defined
#else
    descr.node_gc_counter = 0;  ///< Disable garbage collection counter
#endif

    descr.visiting_number_for_backprop = 0; ///< Initialize visiting number for backpropagation
    return descr;
}

/**
 * @brief Creates a valid operation descriptor for a given operation type at compile-time.
 *
 * @param operation The operation type.
 * @return A valid operation descriptor.
 */
template<OpType operation>
consteval inline OperationDescriptor createValidOpDescriptorCompileTime() noexcept
{
    OperationDescriptor descr;
    descr.op_type = (unsigned int)operation;

#if BURTORCH_USE_GC_COUNTERS
    descr.node_gc_counter = 1;  ///< Enable garbage collection counter if defined
#else
    descr.node_gc_counter = 0;  ///< Disable garbage collection counter
#endif

    descr.visiting_number_for_backprop = 0; ///< Initialize visiting number for backpropagation
    return descr;
}
