#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"

#include "burtcore/include/burtorch_op_types.h"
#include <assert.h>
#include <stdint.h>

enum BackwardDispatchHint : uint32_t
{
    eNoHints = 0x0 << 0,              ///< No hints provided.
    eOutGradIsOne = 0x1 << 1,         ///< Output gradient is one.
    eReplaceGradsInChilds = 0x1 << 2, ///< Replace gradients in child nodes.
};

enum class OpTypeNumArgs
{
    eZero = 0, ///< Operation takes zero arguments.
    eOne = 1,  ///< Operation takes one argument.
    eTwo = 2,  ///< Operation takes two arguments.
    eAny = 3   ///< Operation can take any number of arguments.
};

/**
 * Helper function to get the number of arguments for a given operation type.
 *
 * @param opType The operation type.
 * @return OpTypeNumArgs The number of arguments for the operation.
 */
inline constexpr OpTypeNumArgs getNumArgs(OpType opType) noexcept
{
	switch (opType)
	{
    // zero-argumnet ops
    case OpType::eLeaf:
		return OpTypeNumArgs::eZero;

    // 1-argumnet ops
    case OpType::eRelu:
        [[fallthrough]];
    case OpType::eTanh:
        [[fallthrough]];
    case OpType::eExp:
        [[fallthrough]];
    case OpType::eNegLog:
        [[fallthrough]];
    case OpType::eSigmoid:
        [[fallthrough]];
    case OpType::eInv:
        [[fallthrough]];
    case OpType::eSqr:
        [[fallthrough]];
    case OpType::eCub:
        [[fallthrough]];
    case OpType::eLog:
        [[fallthrough]];
    case OpType::eSqrt:
        [[fallthrough]];
    case OpType::eInvSqrt:
        return OpTypeNumArgs::eOne;

    // 2-argumnet ops
	case OpType::eBinaryAdd:
        [[fallthrough]];
    case OpType::eBinarySub:
        [[fallthrough]];
    case OpType::eBinaryMult:
        [[fallthrough]];
    case OpType::eBinaryMultByConst:
        [[fallthrough]];
    case OpType::eBinaryDiv:
        [[fallthrough]];
    case OpType::eBinaryMean:
        [[fallthrough]];
    case OpType::eBinaryAddSquares:
        [[fallthrough]];
    case OpType::eBinaryMeanSquares:
        [[fallthrough]];
    case OpType::eBinaryNegativeMean:
        return OpTypeNumArgs::eTwo;

    // n-argumnet ops
    case OpType::eAddVarying:
        [[fallthrough]];
    case OpType::eSubVarying:
        [[fallthrough]];
    case OpType::eMulVarying:
        [[fallthrough]];
    case OpType::eMeanVarying:
        [[fallthrough]];
    case OpType::eSumOfSquaresVarying:
        [[fallthrough]];
    case OpType::eMeanSquaresVarying:
        [[fallthrough]];
    case OpType::eNegativeMeanVarying:
        [[fallthrough]];
    case OpType::eInnerProductNoBias:
        [[fallthrough]];
    case OpType::eInnerProductWithBias:
        return OpTypeNumArgs::eAny;

	default:
        {
            burt_unreahable();
            return OpTypeNumArgs::eAny;
        }
	}
}

/**
* Converts an operation type to its corresponding string representation.
*
* @param opType The operation type.
* @return const char* the string representation of the operation type.
* @note this memory in static sectrion of executable binary
*/
inline constexpr const char* opTypeToString(OpType opType) noexcept
{
    constexpr const char* opTypeStrings[(int)OpType::eOpsCount] = {
		"leaf",               // eLeaf 0
        "relu [s]",           // eRelu 1
        "tanh [s]",           // eTanh 2
        "exp [s]",            // eExp 3
        "-log [s]",           // eNegLog 4
        "sigmoid [s]",        // eSigmoid 5
        "1/x [s]",            // eInv 6
        "x^2 [s]",            // eSqr 7
        "x^3 [s]",            // eCub 8
        "logarithm [s]",      // eLog 9
        "sqrt [s]",           // eSqrt 10
        "inverse-sqrt [s]",   // eInvSqrt 11

        "operator+ [bin]",     // eBinaryAdd 12
        "operator- [bin]",     // eBinarySub 13
        "operator* [bin]",           // eBinaryMult 14
        "operator by const * [bin]", // eBinaryMultByConst 15

        "operator/ [bin]",     // eBinaryDiv 16
        "mean [bin]",          // eBinaryMean 17
        "sum-squares [bin]",   // eBinaryAddSquares 18
        "mean-squares [bin]",  // eBinaryMeanSquares 19
        "negative-mean [bin]", // eBinaryNegativeMean 20

        "operator+ [var]",     // eAddVarying 21
        "operator- [var]",     // eSubVarying 22
        "operator* [var]",     // eMulVarying 23
        "mean [var]",          // eMeanVarying 24
        "sum-squares [var]",   // eSumOfSquaresVarying 25
        "mean-squares [var]",  // eMeanSquaresVarying 26
        "negative-mean [var]", // eNegativeMeanVarying 27
        "inner-product-no-bias [v,w]",      // eInnerProductNoBias 28
        "inner-product-with-bias [v,w,b]",  // eInnerProductWithBias 29

    };

    burt_assert (static_cast<unsigned int>(opType) < sizeof(opTypeStrings) / sizeof(opTypeStrings[0]));

    return opTypeStrings[static_cast<unsigned int>(opType)];
}
