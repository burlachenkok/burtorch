#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include "burtcore/include/burtorch_node.h"
#include "burtcore/include/burtorch_op_types.h"
#include "burtcore/include/burtorch_array4node.h"
#include "burtcore/include/burtorch_special_copy.h"

#include "burt/linalg_vectors/include/LightVectorND.h"
#include "burt/linalg_vectors/include/VectorND_Raw.h"

#include <unordered_map>
#include <vector>
#include <initializer_list>
#include <span>

#include <assert.h>
#include <stddef.h>

inline double max(double a, double b) noexcept {
	return a >= b ? a : b;
}

inline float max(float a, float b) noexcept {
	return a >= b ? a : b;
}

// Operators Wrapper for Compute: value
template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> tanh(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints || opHint == OpHint::eOpHintNotEvaluateValue)
	{
		// derivative use input data of node
		Value<TDataTypeResult> res(tanh(first.dataRef()), OpType::eTanh, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> sigmoid(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints || opHint == OpHint::eOpHintNotEvaluateValue)
	{
		// derivative use input data of node
		static const TDataTypeResult one = TDataTypeResult(1);
		Value<TDataTypeResult> res(one / (one + exp(-(first.dataRef()))), OpType::eSigmoid, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> relu(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints || opHint == OpHint::eOpHintNotEvaluateValue)
	{
		// derivative use input data of node
		Value<TDataTypeResult> res(max(TDataTypeResult(0), first.dataRef()), OpType::eRelu, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> exp(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints || opHint == OpHint::eOpHintNotEvaluateValue)
	{
		// derivative use input data of node
		Value<TDataTypeResult> res(exp(first.dataRef()), OpType::eExp, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> exp_shifted(const Value<TDataTypeArg1>& restrict_ext first, const TDataTypeArg1& shift) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints || opHint == OpHint::eOpHintNotEvaluateValue)
	{
		// derivative use input data of node
		Value<TDataTypeResult> res(exp(first.dataRef() - shift), OpType::eExp, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> negativeLog(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(-log(first.dataRef()), OpType::eNegLog, first.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eNegLog, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> logarithm(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(log(first.dataRef()), OpType::eLog, first.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eLog, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> inv(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(TDataTypeArg1(1) / (first.dataRef()), OpType::eInv, first.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eInv, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> sqr(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		const auto& value = first.dataRef();
		Value<TDataTypeResult> res(value * value, OpType::eSqr, first.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eSqr, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> sqrt(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints || opHint == OpHint::eOpHintNotEvaluateValue)
	{
		// derivative use this information
		Value<TDataTypeResult> res(sqrt(first.dataRef()), OpType::eSqrt, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> invSqrt(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints || opHint == OpHint::eOpHintNotEvaluateValue)
	{
		// derivative use this information
		Value<TDataTypeResult> res(TDataTypeArg1(1)/sqrt(first.dataRef()), OpType::eInvSqrt, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> pow3(const Value<TDataTypeArg1>& restrict_ext first) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		const auto& value = first.dataRef();
		Value<TDataTypeResult> res(value * value * value, OpType::eCub, first.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eCub, first.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProductInternal(const Value<TDataTypeArg1>* a,
	  											   const Value<TDataTypeArg1>* b,
												   size_t a_and_b_items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductNoBias>();

		TDataTypeResult resValue = TDataTypeResult();

		const size_t sz = a_and_b_items;
		const size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			memcpy(resChildSetRaw, a, sz * sizeof(resChildSetRaw[0]));
			memcpy(resChildSetRaw + sz, b, sz * sizeof(resChildSetRaw[0]));
		}

		{
			TDataTypeResult resValue = TDataTypeResult();
			const TDataTypeArg1* aref = &(a->dataRef());


			size_t i = 0;
			size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(sz);
			size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(sz);
			size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(sz);
			size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(sz);

			//typedef burt::VectorSimdTraits<TDataTypeResult, burt::cpu_extension>::VecType VecType;
			//VecType areg, breg;

			for (size_t i = 0; i != packed_sz_by_16; i += 16, aref += 16, b += 16)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();
				auto a5 = aref[4], b5 = b[4].dataCopy();
				auto a6 = aref[5], b6 = b[5].dataCopy();
				auto a7 = aref[6], b7 = b[6].dataCopy();
				auto a8 = aref[7], b8 = b[7].dataCopy();
				auto a9 = aref[8], b9 = b[8].dataCopy();
				auto a10 = aref[9], b10 = b[9].dataCopy();
				auto a11 = aref[10], b11 = b[10].dataCopy();
				auto a12 = aref[11], b12 = b[11].dataCopy();
				auto a13 = aref[12], b13 = b[12].dataCopy();
				auto a14 = aref[13], b14 = b[13].dataCopy();
				auto a15 = aref[14], b15 = b[14].dataCopy();
				auto a16 = aref[15], b16 = b[15].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
					         a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
			}

			for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, aref += 8, b += 8)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();
				auto a5 = aref[4], b5 = b[4].dataCopy();
				auto a6 = aref[5], b6 = b[5].dataCopy();
				auto a7 = aref[6], b7 = b[6].dataCopy();
				auto a8 = aref[7], b8 = b[7].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
			}

			for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, aref += 4, b += 4)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
			}

			for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, aref += 2, b += 2)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();

				resValue += (a1 * b1 + a2 * b2);
			}

			for (size_t i = packed_sz_by_2; i != sz; ++i, ++aref, ++b)
			{
				resValue += aref[0] * b->dataCopy();
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductNoBias>();

		const size_t sz = a_and_b_items;
		const size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			memcpy(resChildSetRaw, a, sz * sizeof(resChildSetRaw[0]));
			memcpy(resChildSetRaw + sz, b, sz * sizeof(resChildSetRaw[0]));
		}
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <size_t a_and_b_items, OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProductInternal(const Value<TDataTypeArg1>* a, const Value<TDataTypeArg1>* b) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductNoBias>();

		TDataTypeResult resValue = TDataTypeResult();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));

			//memcpy(resChildSetRaw, a, sz * sizeof(resChildSetRaw[0]));
			//memcpy(resChildSetRaw + sz, b, sz * sizeof(resChildSetRaw[0]));
			memcpyAtCompileTime<sz>(resChildSetRaw, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(a));
			memcpyAtCompileTime<sz>(resChildSetRaw + sz, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(b));
		}

		{
			TDataTypeResult resValue = TDataTypeResult();
			const TDataTypeArg1* aref = &(a->dataRef());


			size_t i = 0;
			size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(sz);
			size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(sz);
			size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(sz);
			size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(sz);

			//typedef burt::VectorSimdTraits<TDataTypeResult, burt::cpu_extension>::VecType VecType;
			//VecType areg, breg;

			for (size_t i = 0; i != packed_sz_by_16; i += 16, aref += 16, b += 16)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();
				auto a5 = aref[4], b5 = b[4].dataCopy();
				auto a6 = aref[5], b6 = b[5].dataCopy();
				auto a7 = aref[6], b7 = b[6].dataCopy();
				auto a8 = aref[7], b8 = b[7].dataCopy();
				auto a9 = aref[8], b9 = b[8].dataCopy();
				auto a10 = aref[9], b10 = b[9].dataCopy();
				auto a11 = aref[10], b11 = b[10].dataCopy();
				auto a12 = aref[11], b12 = b[11].dataCopy();
				auto a13 = aref[12], b13 = b[12].dataCopy();
				auto a14 = aref[13], b14 = b[13].dataCopy();
				auto a15 = aref[14], b15 = b[14].dataCopy();
				auto a16 = aref[15], b16 = b[15].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
					a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
			}

			for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, aref += 8, b += 8)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();
				auto a5 = aref[4], b5 = b[4].dataCopy();
				auto a6 = aref[5], b6 = b[5].dataCopy();
				auto a7 = aref[6], b7 = b[6].dataCopy();
				auto a8 = aref[7], b8 = b[7].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
			}

			for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, aref += 4, b += 4)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
			}

			for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, aref += 2, b += 2)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();

				resValue += (a1 * b1 + a2 * b2);
			}

			for (size_t i = packed_sz_by_2; i != sz; ++i, ++aref, ++b)
			{
				resValue += aref[0] * b->dataCopy();
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductNoBias>();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			//memcpy(resChildSetRaw, a, sz * sizeof(resChildSetRaw[0]));
			//memcpy(resChildSetRaw + sz, b, sz * sizeof(resChildSetRaw[0]));
			memcpyAtCompileTime<sz>(resChildSetRaw, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(a));
			memcpyAtCompileTime<sz>(resChildSetRaw + sz, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(b));
		}
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProduct(const Value<TDataTypeArg1>* a,
										   const Value<TDataTypeArg1>* b,
										   size_t a_and_b_items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductNoBias>();

		TDataTypeResult resValue = TDataTypeResult();

		const size_t sz = a_and_b_items;
		const size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			memcpy(resChildSetRaw,      a, sz * sizeof(resChildSetRaw[0]));
			memcpy(resChildSetRaw + sz, b, sz * sizeof(resChildSetRaw[0]));
		}

		{
			TDataTypeResult resValue = TDataTypeResult();

			size_t i = 0;
			size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(sz);
			size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(sz);
			size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(sz);
			size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(sz);

			for (size_t i = 0; i != packed_sz_by_16; i += 16, a += 16, b += 16)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();
				auto a5 = a[4].dataCopy(), b5 = b[4].dataCopy();
				auto a6 = a[5].dataCopy(), b6 = b[5].dataCopy();
				auto a7 = a[6].dataCopy(), b7 = b[6].dataCopy();
				auto a8 = a[7].dataCopy(), b8 = b[7].dataCopy();
				auto a9 = a[8].dataCopy(), b9 = b[8].dataCopy();
				auto a10 = a[9].dataCopy(), b10 = b[9].dataCopy();
				auto a11 = a[10].dataCopy(), b11 = b[10].dataCopy();
				auto a12 = a[11].dataCopy(), b12 = b[11].dataCopy();
				auto a13 = a[12].dataCopy(), b13 = b[12].dataCopy();
				auto a14 = a[13].dataCopy(), b14 = b[13].dataCopy();
				auto a15 = a[14].dataCopy(), b15 = b[14].dataCopy();
				auto a16 = a[15].dataCopy(), b16 = b[15].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
           					    a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
			}

			for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, a += 8, b += 8)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();
				auto a5 = a[4].dataCopy(), b5 = b[4].dataCopy();
				auto a6 = a[5].dataCopy(), b6 = b[5].dataCopy();
				auto a7 = a[6].dataCopy(), b7 = b[6].dataCopy();
				auto a8 = a[7].dataCopy(), b8 = b[7].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
			}

			for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, a += 4, b += 4)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
			}

			for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, a += 2, b += 2)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();

				resValue += (a1 * b1 + a2 * b2);
			}

			for (size_t i = packed_sz_by_2; i != sz; ++i, ++a, ++b)
			{
				resValue += a->dataCopy() * b->dataCopy();
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductNoBias>();

		const size_t sz = a_and_b_items;
		const size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			memcpy(resChildSetRaw, a, sz * sizeof(resChildSetRaw[0]));
			memcpy(resChildSetRaw + sz, b, sz * sizeof(resChildSetRaw[0]));
		}
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <size_t a_and_b_items, OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProduct(const Value<TDataTypeArg1>* a, const Value<TDataTypeArg1>* b) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductNoBias>();

		TDataTypeResult resValue = TDataTypeResult();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));

			//memcpy(resChildSetRaw, a, sz * sizeof(resChildSetRaw[0]));
			//memcpy(resChildSetRaw + sz, b, sz * sizeof(resChildSetRaw[0]));
			memcpyAtCompileTime<sz>(resChildSetRaw, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(a));
			memcpyAtCompileTime<sz>(resChildSetRaw + sz, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(b));
		}

		{
			TDataTypeResult resValue = TDataTypeResult();

			size_t i = 0;
			size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(sz);
			size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(sz);
			size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(sz);
			size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(sz);

			for (size_t i = 0; i != packed_sz_by_16; i += 16, a += 16, b += 16)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();
				auto a5 = a[4].dataCopy(), b5 = b[4].dataCopy();
				auto a6 = a[5].dataCopy(), b6 = b[5].dataCopy();
				auto a7 = a[6].dataCopy(), b7 = b[6].dataCopy();
				auto a8 = a[7].dataCopy(), b8 = b[7].dataCopy();
				auto a9 = a[8].dataCopy(), b9 = b[8].dataCopy();
				auto a10 = a[9].dataCopy(), b10 = b[9].dataCopy();
				auto a11 = a[10].dataCopy(), b11 = b[10].dataCopy();
				auto a12 = a[11].dataCopy(), b12 = b[11].dataCopy();
				auto a13 = a[12].dataCopy(), b13 = b[12].dataCopy();
				auto a14 = a[13].dataCopy(), b14 = b[13].dataCopy();
				auto a15 = a[14].dataCopy(), b15 = b[14].dataCopy();
				auto a16 = a[15].dataCopy(), b16 = b[15].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
					a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
			}

			for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, a += 8, b += 8)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();
				auto a5 = a[4].dataCopy(), b5 = b[4].dataCopy();
				auto a6 = a[5].dataCopy(), b6 = b[5].dataCopy();
				auto a7 = a[6].dataCopy(), b7 = b[6].dataCopy();
				auto a8 = a[7].dataCopy(), b8 = b[7].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
			}

			for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, a += 4, b += 4)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
			}

			for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, a += 2, b += 2)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();

				resValue += (a1 * b1 + a2 * b2);
			}

			for (size_t i = packed_sz_by_2; i != sz; ++i, ++a, ++b)
			{
				resValue += a->dataCopy() * b->dataCopy();
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductNoBias>();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));

			memcpyAtCompileTime<sz>(resChildSetRaw, reinterpret_cast<typename Value<TDataTypeArg1>::TNodeIndexType*>(a));
			memcpyAtCompileTime<sz>(resChildSetRaw + sz, reinterpret_cast<typename Value<TDataTypeArg1>::TNodeIndexType*>(b));

			//memcpy(resChildSetRaw, a, sz * sizeof(resChildSetRaw[0]));
			//memcpy(resChildSetRaw + sz, b, sz * sizeof(resChildSetRaw[0]));
		}
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProduct(std::initializer_list<Value<TDataTypeArg1>> a, 
	                                       std::initializer_list<Value<TDataTypeArg1>> b) noexcept
{
	burt_assert(a.size() == b.size());
	return innerProduct<opHint, TDataTypeArg1, TDataTypeResult>(a.begin(), b.begin(), a.size());
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProductWithBias(const Value<TDataTypeArg1>* bias,
												   const Value<TDataTypeArg1>* a,
												   const Value<TDataTypeArg1>* b,
												   size_t a_and_b_items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias> ();

		const size_t sz = a_and_b_items;
		const size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
        burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(1 + sz_times_2);
		assert(resChildSetRaw != nullptr);

		resChildSetRaw[0] = bias->sysGetRawNodeIndex();

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			memcpy(resChildSetRaw + 1, a, sz * sizeof(resChildSetRaw[0]));
			memcpy(resChildSetRaw + 1 + sz, b, sz * sizeof(resChildSetRaw[0]));
		}

		{
			TDataTypeResult resValue = TDataTypeResult();
			
			resValue = TDataTypeResult(bias->dataRef());

			size_t i = 0;
			size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(sz);
			size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(sz);
			size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(sz);
			size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(sz);

			for (size_t i = 0; i != packed_sz_by_16; i += 16, a += 16, b += 16)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();
				auto a5 = a[4].dataCopy(), b5 = b[4].dataCopy();
				auto a6 = a[5].dataCopy(), b6 = b[5].dataCopy();
				auto a7 = a[6].dataCopy(), b7 = b[6].dataCopy();
				auto a8 = a[7].dataCopy(), b8 = b[7].dataCopy();
				auto a9 = a[8].dataCopy(), b9 = b[8].dataCopy();
				auto a10 = a[9].dataCopy(), b10 = b[9].dataCopy();
				auto a11 = a[10].dataCopy(), b11 = b[10].dataCopy();
				auto a12 = a[11].dataCopy(), b12 = b[11].dataCopy();
				auto a13 = a[12].dataCopy(), b13 = b[12].dataCopy();
				auto a14 = a[13].dataCopy(), b14 = b[13].dataCopy();
				auto a15 = a[14].dataCopy(), b15 = b[14].dataCopy();
				auto a16 = a[15].dataCopy(), b16 = b[15].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
					         a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
			}

			for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, a += 8, b += 8)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();
				auto a5 = a[4].dataCopy(), b5 = b[4].dataCopy();
				auto a6 = a[5].dataCopy(), b6 = b[5].dataCopy();
				auto a7 = a[6].dataCopy(), b7 = b[6].dataCopy();
				auto a8 = a[7].dataCopy(), b8 = b[7].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
			}

			for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, a += 4, b += 4)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();
				auto a3 = a[2].dataCopy(), b3 = b[2].dataCopy();
				auto a4 = a[3].dataCopy(), b4 = b[3].dataCopy();
				
				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
			}

			for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, a += 2, b += 2)
			{
				auto a1 = a[0].dataCopy(), b1 = b[0].dataCopy();
				auto a2 = a[1].dataCopy(), b2 = b[1].dataCopy();

				resValue += (a1 * b1 + a2 * b2);
			}

			for (size_t i = packed_sz_by_2; i != sz; ++i, ++a, ++b)
			{
				resValue += a->dataCopy() * b->dataCopy();
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		const size_t sz = a_and_b_items;
		const size_t sz_times_2 = a_and_b_items + a_and_b_items;

		auto& resChildSet = res.sysChildrenSet();
        burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(1 + sz_times_2);
		assert(resChildSetRaw != nullptr);

		resChildSetRaw[0] = bias->sysGetRawNodeIndex();
		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			memcpy(resChildSetRaw + 1, a, sz * sizeof(resChildSetRaw[0]));
			memcpy(resChildSetRaw + 1 + sz, b, sz * sizeof(resChildSetRaw[0]));
		}
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}



template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProductWithBiasInternal(const Value<TDataTypeArg1>* bias,
														   const Value<TDataTypeArg1>* a,
														   const Value<TDataTypeArg1>* b, 
														   size_t a_and_b_items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		size_t sz = a_and_b_items;
		size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(1 + sz_times_2);
		assert(resChildSetRaw != nullptr);

		resChildSetRaw[0] = bias->sysGetRawNodeIndex();
		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			memcpy(resChildSetRaw + 1, a, sz * sizeof(resChildSetRaw[0]));
			memcpy(resChildSetRaw + 1 + sz, b, sz * sizeof(resChildSetRaw[0]));
		}


		{
			TDataTypeResult resValue = TDataTypeResult(bias->dataRef());

			const TDataTypeArg1* aref = &(a->dataRef());

			size_t i = 0;
			size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(sz);
			size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(sz);
			size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(sz);
			size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(sz);

			for (size_t i = 0; i != packed_sz_by_16; i += 16, aref += 16, b += 16)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();
				auto a5 = aref[4], b5 = b[4].dataCopy();
				auto a6 = aref[5], b6 = b[5].dataCopy();
				auto a7 = aref[6], b7 = b[6].dataCopy();
				auto a8 = aref[7], b8 = b[7].dataCopy();
				auto a9 = aref[8], b9 = b[8].dataCopy();
				auto a10 = aref[9], b10 = b[9].dataCopy();
				auto a11 = aref[10], b11 = b[10].dataCopy();
				auto a12 = aref[11], b12 = b[11].dataCopy();
				auto a13 = aref[12], b13 = b[12].dataCopy();
				auto a14 = aref[13], b14 = b[13].dataCopy();
				auto a15 = aref[14], b15 = b[14].dataCopy();
				auto a16 = aref[15], b16 = b[15].dataCopy();

                resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
                             a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
			}

			for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, aref += 8, b += 8)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();
				auto a5 = aref[4], b5 = b[4].dataCopy();
				auto a6 = aref[5], b6 = b[5].dataCopy();
				auto a7 = aref[6], b7 = b[6].dataCopy();
				auto a8 = aref[7], b8 = b[7].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
			}

			for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, aref += 4, b += 4)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
			}

			for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, aref += 2, b += 2)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();

				resValue += (a1 * b1 + a2 * b2);
			}

			for (size_t i = packed_sz_by_2; i != sz; ++i, ++aref, ++b)
			{
				resValue += aref[0] * b->dataCopy();
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		size_t sz = a_and_b_items;
		size_t sz_times_2 = a_and_b_items + a_and_b_items;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(1 + sz_times_2);
		assert(resChildSetRaw != nullptr);

		resChildSetRaw[0] = bias->sysGetRawNodeIndex();
		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));
			memcpy(resChildSetRaw + 1, a, sz * sizeof(resChildSetRaw[0]));
			memcpy(resChildSetRaw + 1 + sz, b, sz * sizeof(resChildSetRaw[0]));
		}
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <size_t a_and_b_items, OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProductWithBiasInternal(const Value<TDataTypeArg1>* bias,
														   const Value<TDataTypeArg1>* a,
														   const Value<TDataTypeArg1>* b) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(1 + sz_times_2);
		assert(resChildSetRaw != nullptr);

		resChildSetRaw[0] = bias->sysGetRawNodeIndex();
		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));

			memcpyAtCompileTime<sz>(resChildSetRaw + 1, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(a));
			memcpyAtCompileTime<sz>(resChildSetRaw + 1 + sz, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(b));
		}


		{
			TDataTypeResult resValue = TDataTypeResult(bias->dataRef());

			const TDataTypeArg1* aref = &(a->dataRef());

			size_t i = 0;
			size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(sz);
			size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(sz);
			size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(sz);
			size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(sz);


			for (size_t i = 0; i != packed_sz_by_16; i += 16, aref += 16, b += 16)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();
				auto a5 = aref[4], b5 = b[4].dataCopy();
				auto a6 = aref[5], b6 = b[5].dataCopy();
				auto a7 = aref[6], b7 = b[6].dataCopy();
				auto a8 = aref[7], b8 = b[7].dataCopy();
				auto a9 = aref[8], b9 = b[8].dataCopy();
				auto a10 = aref[9], b10 = b[9].dataCopy();
				auto a11 = aref[10], b11 = b[10].dataCopy();
				auto a12 = aref[11], b12 = b[11].dataCopy();
				auto a13 = aref[12], b13 = b[12].dataCopy();
				auto a14 = aref[13], b14 = b[13].dataCopy();
				auto a15 = aref[14], b15 = b[14].dataCopy();
				auto a16 = aref[15], b16 = b[15].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
				 			a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
			}

			for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, aref += 8, b += 8)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();
				auto a5 = aref[4], b5 = b[4].dataCopy();
				auto a6 = aref[5], b6 = b[5].dataCopy();
				auto a7 = aref[6], b7 = b[6].dataCopy();
				auto a8 = aref[7], b8 = b[7].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
			}

			for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, aref += 4, b += 4)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();
				auto a3 = aref[2], b3 = b[2].dataCopy();
				auto a4 = aref[3], b4 = b[3].dataCopy();

				resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
			}

			for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, aref += 2, b += 2)
			{
				auto a1 = aref[0], b1 = b[0].dataCopy();
				auto a2 = aref[1], b2 = b[1].dataCopy();

				resValue += (a1 * b1 + a2 * b2);
			}

			for (size_t i = packed_sz_by_2; i != sz; ++i, ++aref, ++b)
			{
				resValue += aref[0] * b->dataCopy();
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = a_and_b_items + a_and_b_items;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(1 + sz_times_2);
		assert(resChildSetRaw != nullptr);

		resChildSetRaw[0] = bias->sysGetRawNodeIndex();
		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*b));

			memcpyAtCompileTime<sz>(resChildSetRaw + 1, reinterpret_cast<typename Value<TDataTypeArg1>::TNodeIndexType*>(a));
			memcpyAtCompileTime<sz>(resChildSetRaw + 1 + sz, reinterpret_cast<typename Value<TDataTypeArg1>::TNodeIndexType*>(b));
		}

		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <size_t a_and_b_items, size_t items_per_array_item, OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProductInternalWithXView(const Value<TDataTypeArg1>* a,
															const std::initializer_list<const Value<TDataTypeArg1>*>& b) noexcept
{
	static_assert(a_and_b_items % items_per_array_item == 0);
	constexpr size_t b_view_chunks = a_and_b_items / items_per_array_item;

	assert(b_view_chunks == b.size());

	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(**b.begin()));

			//memcpy(resChildSetRaw + 1, a, sz * sizeof(resChildSetRaw[0]));
			//memcpy(resChildSetRaw + 1 + sz, b, sz * sizeof(resChildSetRaw[0]));
			memcpyAtCompileTime<sz>(resChildSetRaw, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(a));

			auto bIter = b.begin();
			for (size_t chunk = 0; chunk < b_view_chunks; ++chunk, ++bIter)
				memcpyAtCompileTime<items_per_array_item>( (resChildSetRaw + sz) + chunk * items_per_array_item,
															reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(*bIter));
		}


		{
			TDataTypeResult resValue = TDataTypeResult();
			const TDataTypeArg1* aref = &(a->dataRef());
			auto b_part = b.begin();

			for (size_t chunk = 0; chunk < b_view_chunks; ++chunk, ++b_part)
			{
				auto b_ = *b_part;

				size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(items_per_array_item);
				size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(items_per_array_item);
				size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(items_per_array_item);
				size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(items_per_array_item);

				for (size_t i = 0; i != packed_sz_by_16; i += 16, aref += 16, b_ += 16)
				{
					auto a1 = aref[0], b1 = b_[0].dataCopy();
					auto a2 = aref[1], b2 = b_[1].dataCopy();
					auto a3 = aref[2], b3 = b_[2].dataCopy();
					auto a4 = aref[3], b4 = b_[3].dataCopy();
					auto a5 = aref[4], b5 = b_[4].dataCopy();
					auto a6 = aref[5], b6 = b_[5].dataCopy();
					auto a7 = aref[6], b7 = b_[6].dataCopy();
					auto a8 = aref[7], b8 = b_[7].dataCopy();
					auto a9 = aref[8], b9 = b_[8].dataCopy();
					auto a10 = aref[9], b10 = b_[9].dataCopy();
					auto a11 = aref[10], b11 = b_[10].dataCopy();
					auto a12 = aref[11], b12 = b_[11].dataCopy();
					auto a13 = aref[12], b13 = b_[12].dataCopy();
					auto a14 = aref[13], b14 = b_[13].dataCopy();
					auto a15 = aref[14], b15 = b_[14].dataCopy();
					auto a16 = aref[15], b16 = b_[15].dataCopy();

					resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
						a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
				}

				for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, aref += 8, b_ += 8)
				{
					auto a1 = aref[0], b1 = b_[0].dataCopy();
					auto a2 = aref[1], b2 = b_[1].dataCopy();
					auto a3 = aref[2], b3 = b_[2].dataCopy();
					auto a4 = aref[3], b4 = b_[3].dataCopy();
					auto a5 = aref[4], b5 = b_[4].dataCopy();
					auto a6 = aref[5], b6 = b_[5].dataCopy();
					auto a7 = aref[6], b7 = b_[6].dataCopy();
					auto a8 = aref[7], b8 = b_[7].dataCopy();

					resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
				}

				for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, aref += 4, b_ += 4)
				{
					auto a1 = aref[0], b1 = b_[0].dataCopy();
					auto a2 = aref[1], b2 = b_[1].dataCopy();
					auto a3 = aref[2], b3 = b_[2].dataCopy();
					auto a4 = aref[3], b4 = b_[3].dataCopy();

					resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
				}

				for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, aref += 2, b_ += 2)
				{
					auto a1 = aref[0], b1 = b_[0].dataCopy();
					auto a2 = aref[1], b2 = b_[1].dataCopy();

					resValue += (a1 * b1 + a2 * b2);
				}

				for (size_t i = packed_sz_by_2; i != items_per_array_item; ++i, ++aref, ++b_)
				{
					resValue += aref[0] * b_->dataCopy();
				}
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = a_and_b_items + a_and_b_items;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(sz_times_2);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(**b.begin()));

			memcpyAtCompileTime<sz>(resChildSetRaw, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(a));

			auto bIter = b.begin();
			for (size_t chunk = 0; chunk < b_view_chunks; ++chunk, ++bIter)
				memcpyAtCompileTime<items_per_array_item>((resChildSetRaw + sz) + chunk * items_per_array_item,
														  reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(*bIter));
		}

		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <size_t a_and_b_items, size_t items_per_array_item, OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProductWithBiasInternalWithXView(const Value<TDataTypeArg1>* bias,
																	const Value<TDataTypeArg1>* a,
																	const std::initializer_list<const Value<TDataTypeArg1>*>& b) noexcept
{
	static_assert(a_and_b_items % items_per_array_item == 0);
	constexpr size_t b_view_chunks = a_and_b_items / items_per_array_item;

	assert(b_view_chunks == b.size());

	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = sz + sz;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(1 + sz_times_2);
		assert(resChildSetRaw != nullptr);

		resChildSetRaw[0] = bias->sysGetRawNodeIndex();
		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(**b.begin()));

			memcpyAtCompileTime<sz>(resChildSetRaw + 1, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(a));
			
			auto bIter = b.begin();
			for (size_t chunk = 0; chunk < b_view_chunks; ++chunk, ++bIter)
				memcpyAtCompileTime<items_per_array_item>((resChildSetRaw + 1 + sz) + chunk * items_per_array_item, 
														  reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(*bIter));
		}


		{
			TDataTypeResult resValue = TDataTypeResult(bias->dataRef());

			const TDataTypeArg1* aref = &(a->dataRef());
			auto b_part = b.begin();

			for (size_t chunk = 0; chunk < b_view_chunks; ++chunk, ++b_part)
			{
				auto b_ = *b_part;

				size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(items_per_array_item);
				size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(items_per_array_item);
				size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(items_per_array_item);
				size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(items_per_array_item);

				for (size_t i = 0; i != packed_sz_by_16; i += 16, aref += 16, b_ += 16)
				{
					auto a1 = aref[0], b1 = b_[0].dataCopy();
					auto a2 = aref[1], b2 = b_[1].dataCopy();
					auto a3 = aref[2], b3 = b_[2].dataCopy();
					auto a4 = aref[3], b4 = b_[3].dataCopy();
					auto a5 = aref[4], b5 = b_[4].dataCopy();
					auto a6 = aref[5], b6 = b_[5].dataCopy();
					auto a7 = aref[6], b7 = b_[6].dataCopy();
					auto a8 = aref[7], b8 = b_[7].dataCopy();
					auto a9 = aref[8], b9 = b_[8].dataCopy();
					auto a10 = aref[9], b10 = b_[9].dataCopy();
					auto a11 = aref[10], b11 = b_[10].dataCopy();
					auto a12 = aref[11], b12 = b_[11].dataCopy();
					auto a13 = aref[12], b13 = b_[12].dataCopy();
					auto a14 = aref[13], b14 = b_[13].dataCopy();
					auto a15 = aref[14], b15 = b_[14].dataCopy();
					auto a16 = aref[15], b16 = b_[15].dataCopy();

					resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8 +
						         a9 * b9 + a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13 + a14 * b14 + a15 * b15 + a16 * b16);
				}

				for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, aref += 8, b_ += 8)
				{
					auto a1 = aref[0], b1 = b_[0].dataCopy();
					auto a2 = aref[1], b2 = b_[1].dataCopy();
					auto a3 = aref[2], b3 = b_[2].dataCopy();
					auto a4 = aref[3], b4 = b_[3].dataCopy();
					auto a5 = aref[4], b5 = b_[4].dataCopy();
					auto a6 = aref[5], b6 = b_[5].dataCopy();
					auto a7 = aref[6], b7 = b_[6].dataCopy();
					auto a8 = aref[7], b8 = b_[7].dataCopy();

					resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7 + a8 * b8);
				}

				for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, aref += 4, b_ += 4)
				{
					auto a1 = aref[0], b1 = b_[0].dataCopy();
					auto a2 = aref[1], b2 = b_[1].dataCopy();
					auto a3 = aref[2], b3 = b_[2].dataCopy();
					auto a4 = aref[3], b4 = b_[3].dataCopy();

					resValue += (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4);
				}

				for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, aref += 2, b_ += 2)
				{
					auto a1 = aref[0], b1 = b_[0].dataCopy();
					auto a2 = aref[1], b2 = b_[1].dataCopy();

					resValue += (a1 * b1 + a2 * b2);
				}

				for (size_t i = packed_sz_by_2; i != items_per_array_item; ++i, ++aref, ++b_)
				{
					resValue += aref[0] * b_->dataCopy();
				}
			}

			res.dataRef() = resValue;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eInnerProductWithBias>();

		constexpr size_t sz = a_and_b_items;
		constexpr size_t sz_times_2 = a_and_b_items + a_and_b_items;

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg1>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(1 + sz_times_2);
		assert(resChildSetRaw != nullptr);

		resChildSetRaw[0] = bias->sysGetRawNodeIndex();
		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*a));
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(**b.begin()));

			memcpyAtCompileTime<sz>(resChildSetRaw + 1, reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(a));

			auto bIter = b.begin();
			for (size_t chunk = 0; chunk < b_view_chunks; ++chunk, ++bIter)
				memcpyAtCompileTime<items_per_array_item>((resChildSetRaw + 1 + sz) + chunk * items_per_array_item,
														  reinterpret_cast<const typename Value<TDataTypeArg1>::TNodeIndexType*>(*bIter));
		}

		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeResult = TDataTypeArg1>
inline Value<TDataTypeResult> innerProductWithBias(Value<TDataTypeArg1> bias,
												   std::initializer_list<Value<TDataTypeArg1>> a,
												   std::initializer_list<Value<TDataTypeArg1>> b) noexcept
{
	burt_assert(a.size() == b.size());
	return innerProductWithBias<opHint, TDataTypeArg1, TDataTypeResult>(&bias, a.begin(), b.begin(), a.size());
}

// Operators Wrapper for Compute: value, value
template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceSum(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eAddVarying>();
		// Value<TDataTypeResult> res(TDataTypeResult(), OpType::eAddVarying);
		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
			memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));

			TDataTypeResult accum = TDataTypeResult();

			for (size_t i = 0; i < items; ++i)
			{
				accum += firstItemPointer[i].dataCopy();
			}
			res.dataRef() = accum;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eAddVarying>();
		// Value<TDataTypeResult> res(TDataTypeResult(), OpType::eAddVarying);

		auto& resChildSet = res.sysChildrenSet();
        burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


// Operators Wrapper for Compute: value, value
template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceSub(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eSubVarying>();
		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
			memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));

			if (items > 0)
			{
				TDataTypeResult accum = firstItemPointer[0].dataCopy();
				for (size_t i = 1; i < items; ++i)
				{
					accum -= firstItemPointer[i].dataCopy();
				}
				res.dataRef() = accum;
			}
			else
			{
				res.resetData();
			}
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eSubVarying>();
		// Value<TDataTypeResult> res(TDataTypeResult(), OpType::eAddVarying);

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceMul(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMulVarying>();
		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		assert(resChildSetRaw != nullptr);

		{
			static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
			memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));

			TDataTypeResult accum = TDataTypeResult(1);

			for (size_t i = 0; i < items; ++i)
			{
				accum *= firstItemPointer[i].dataCopy();
			}
			res.dataRef() = accum;
		}

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue<ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eAddVarying>();
		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceSumForSequnetialAllocatedNeurons(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	burt_assert(Value<TDataTypeArg>::isSequentialIndicies(std::span(firstItemPointer, items)));

	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eAddVarying);
		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());
		if (items >= 1)
			resChildSet.sysArrayResizeLossyToArithmeticProgression(items, firstItemPointer[0].sysGetRawNodeIndex());

		const TDataTypeArg* in_items = &firstItemPointer[0].dataRef();
		
		TDataTypeResult resValue = TDataTypeResult();

		size_t i = 0;
		size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(items);
		size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(items);
		size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(items);
		size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(items);

		for (size_t i = 0; i != packed_sz_by_16; i += 16, in_items += 16)
		{
			auto a1 = in_items[0];
			auto a2 = in_items[1];
			auto a3 = in_items[2];
			auto a4 = in_items[3];
			auto a5 = in_items[4];
			auto a6 = in_items[5];
			auto a7 = in_items[6];
			auto a8 = in_items[7];
			auto a9 = in_items[8];
			auto a10 = in_items[9];
			auto a11 = in_items[10];
			auto a12 = in_items[11];
			auto a13 = in_items[12];
			auto a14 = in_items[13];
			auto a15 = in_items[14];
			auto a16 = in_items[15];

			resValue += (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 +
				         a9 + a10  + a11 + a12 + a13 + a14 + a15 + a16);
		}

		for (size_t i = packed_sz_by_16; i != packed_sz_by_8; i += 8, in_items += 8)
		{
			auto a1 = in_items[0];
			auto a2 = in_items[1];
			auto a3 = in_items[2];
			auto a4 = in_items[3];
			auto a5 = in_items[4];
			auto a6 = in_items[5];
			auto a7 = in_items[6];
			auto a8 = in_items[7];

			resValue += (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8);
		}

		for (size_t i = packed_sz_by_8; i != packed_sz_by_4; i += 4, in_items += 4)
		{
			auto a1 = in_items[0];
			auto a2 = in_items[1];
			auto a3 = in_items[2];
			auto a4 = in_items[3];

			resValue += (a1 + a2 + a3 + a4);
		}

		for (size_t i = packed_sz_by_4; i != packed_sz_by_2; i += 2, in_items += 2)
		{
			auto a1 = in_items[0];
			auto a2 = in_items[1];

			resValue += (a1 + a2);
		}

		for (size_t i = packed_sz_by_2; i != items; ++i, ++in_items)
		{
			resValue += in_items[i];
		}

		res.dataRef() = (resValue);

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eAddVarying);
		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());
		if (items >= 1)
			resChildSet.sysArrayResizeLossyToArithmeticProgression(items, firstItemPointer[0].sysGetRawNodeIndex());
		return res;
	}
	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceSumOfSquares(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eSumOfSquaresVarying> ();

		auto& resChildSet = res.sysChildrenSet();
        burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		TDataTypeResult accum = TDataTypeResult();

		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));

		for (size_t i = 0; i < items; ++i)
		{
			accum += (firstItemPointer[i].dataCopy()) * (firstItemPointer[i].dataCopy());
		}

		res.dataRef() = accum;

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eSumOfSquaresVarying>();

		auto& resChildSet = res.sysChildrenSet();
        burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceMean(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		TDataTypeResult divider = TDataTypeResult(1.0 / double(items));
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanVarying>();

		auto& resChildSet = res.sysChildrenSet();
        burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));

		TDataTypeResult accum = TDataTypeResult();

		for (size_t i = 0; i < items; ++i)
		{
			accum += firstItemPointer[i].dataCopy();
		}
		res.dataRef() = accum * divider;

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanVarying>();

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <size_t items, OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceMean(Value<TDataTypeArg>* firstItemPointer) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		TDataTypeResult divider = TDataTypeResult(1.0 / double(items));
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanVarying>();

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));

		//memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		memcpyAtCompileTime<items>(resChildSetRaw, firstItemPointer);

		TDataTypeResult accum = TDataTypeResult();

		for (size_t i = 0; i < items; ++i)
		{
			accum += firstItemPointer[i].dataCopy();
		}
		res.dataRef() = accum * divider;

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanVarying>();

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceNegativeMean(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		TDataTypeResult divider = TDataTypeResult(-1.0/double(items));

		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eNegativeMeanVarying>();

		auto& resChildSet = res.sysChildrenSet();
        burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));

		TDataTypeResult accum = TDataTypeResult();
		for (size_t i = 0; i < items; ++i)
		{
			accum += firstItemPointer[i].dataCopy();
		}
		res.dataRef() = accum * divider;

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eNegativeMeanVarying>();

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceMeanSquares(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		TDataTypeResult divider = TDataTypeResult(1.0 / double(items));

		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();

		auto& resChildSet = res.sysChildrenSet();
        burt_assert(&resChildSet == &res.sysChildrenSet());
		
		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));

		TDataTypeResult accum = TDataTypeResult();
		for (size_t i = 0; i < items; ++i)
		{
			accum += (firstItemPointer[i].dataCopy() * firstItemPointer[i].dataCopy());
		}
		res.dataRef() = accum * divider;

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <size_t items, OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> reduceMeanSquares(Value<TDataTypeArg>* firstItemPointer) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		TDataTypeResult divider = TDataTypeResult(1.0 / double(items));

		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		//memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		memcpyAtCompileTime<items>(resChildSetRaw, firstItemPointer);

		TDataTypeResult accum = TDataTypeResult();
		for (size_t i = 0; i < items; ++i)
		{
			accum += (firstItemPointer[i].dataCopy() * firstItemPointer[i].dataCopy());
		}
		res.dataRef() = accum * divider;

		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		ValueResultType res = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();

		auto& resChildSet = res.sysChildrenSet();
		burt_assert(&resChildSet == &res.sysChildrenSet());

		typename Value<TDataTypeArg>::TNodeIndexType* resChildSetRaw = resChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(resChildSetRaw[0]) == sizeof(*firstItemPointer));
		//memcpy(resChildSetRaw, firstItemPointer, items * sizeof(resChildSetRaw[0]));
		memcpyAtCompileTime<items>(resChildSetRaw, firstItemPointer);

		res.resetData();

		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}


template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline void reduceMeanAndMeanSquares(Value<TDataTypeResult>& restrict_ext mean, 
									 Value<TDataTypeResult>& restrict_ext mean_square,
								     Value<TDataTypeArg>* restrict_ext firstItemPointer, 
									 size_t items) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		TDataTypeResult divider = TDataTypeResult(1.0 / double(items));

		mean = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();
		mean_square = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();

		auto& meanResChildSet = mean.sysChildrenSet();
		typename Value<TDataTypeArg>::TNodeIndexType* meanResChildSetRaw = meanResChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(meanResChildSetRaw[0]) == sizeof(*firstItemPointer));

		auto& meanSqrResChildSet = mean_square.sysChildrenSet();
		typename Value<TDataTypeArg>::TNodeIndexType* meanSqrResChildSetRaw = meanSqrResChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(meanSqrResChildSetRaw[0]) == sizeof(*firstItemPointer));

		memcpy(meanResChildSetRaw, firstItemPointer, items * sizeof(meanResChildSetRaw[0]));
		memcpy(meanSqrResChildSetRaw, firstItemPointer, items * sizeof(meanSqrResChildSetRaw[0]));

		TDataTypeResult accum = TDataTypeResult();
		TDataTypeResult accum_sqr = TDataTypeResult();

		for (size_t i = 0; i < items; ++i)
		{
			auto item = firstItemPointer[i].dataCopy();
			accum += (item);
			accum_sqr += (item * item);
		}

		mean.dataRef() = accum * divider;
		mean_square.dataRef() = accum_sqr * divider;

		return;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		mean = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();
		mean_square = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();

		auto& meanResChildSet = mean.sysChildrenSet();
		typename Value<TDataTypeArg>::TNodeIndexType* meanResChildSetRaw = meanResChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(meanResChildSetRaw[0]) == sizeof(*firstItemPointer));

		auto& meanSqrResChildSet = mean_square.sysChildrenSet();
		typename Value<TDataTypeArg>::TNodeIndexType* meanSqrResChildSetRaw = meanSqrResChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(meanSqrResChildSetRaw[0]) == sizeof(*firstItemPointer));

		memcpy(meanResChildSetRaw, firstItemPointer, items * sizeof(meanResChildSetRaw[0]));
		memcpy(meanSqrResChildSetRaw, firstItemPointer, items * sizeof(meanSqrResChildSetRaw[0]));

		mean.resetData();
		mean_square.resetData();
		return;
	}

	burt_unreahable();
	return;
}


template <size_t items, OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline void reduceMeanAndMeanSquares(Value<TDataTypeResult>& restrict_ext mean, 
									 Value<TDataTypeResult>& restrict_ext mean_square,
									 Value<TDataTypeArg>*restrict_ext firstItemPointer) noexcept
{
	using ValueResultType = Value<TDataTypeResult>;

	if constexpr (opHint == OpHint::eOpNoHints)
	{
		TDataTypeResult divider = TDataTypeResult(1.0 / double(items));

		mean = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();
		mean_square = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();

		auto& meanResChildSet = mean.sysChildrenSet();
		typename Value<TDataTypeArg>::TNodeIndexType* meanResChildSetRaw = meanResChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(meanResChildSetRaw[0]) == sizeof(*firstItemPointer));

		auto& meanSqrResChildSet = mean_square.sysChildrenSet();
		typename Value<TDataTypeArg>::TNodeIndexType* meanSqrResChildSetRaw = meanSqrResChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(meanSqrResChildSetRaw[0]) == sizeof(*firstItemPointer));

		//memcpy(meanResChildSetRaw, firstItemPointer, items * sizeof(meanResChildSetRaw[0]));
		//memcpy(meanSqrResChildSetRaw, firstItemPointer, items * sizeof(meanSqrResChildSetRaw[0]));
		memcpyAtCompileTime<items>(meanResChildSetRaw, meanSqrResChildSetRaw, reinterpret_cast<const typename Value<TDataTypeArg>::TNodeIndexType*>(firstItemPointer));

		TDataTypeResult accum = TDataTypeResult();
		TDataTypeResult accum_sqr = TDataTypeResult();

		for (size_t i = 0; i < items; ++i)
		{
			auto item = firstItemPointer[i].dataCopy();
			accum += (item);
			accum_sqr += (item * item);
		}

		mean.dataRef() = accum * divider;
		mean_square.dataRef() = accum_sqr * divider;

		return;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		mean = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();
		mean_square = ValueResultType::template sysCreateRawValue< ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later, OpType::eMeanSquaresVarying>();

		auto& meanResChildSet = mean.sysChildrenSet();
		typename Value<TDataTypeArg>::TNodeIndexType* meanResChildSetRaw = meanResChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(meanResChildSetRaw[0]) == sizeof(*firstItemPointer));

		auto& meanSqrResChildSet = mean_square.sysChildrenSet();
		typename Value<TDataTypeArg>::TNodeIndexType* meanSqrResChildSetRaw = meanSqrResChildSet.sysArrayResizeLossyWithoutAnyInit(items);
		static_assert(sizeof(meanSqrResChildSetRaw[0]) == sizeof(*firstItemPointer));

		//memcpy(meanResChildSetRaw, firstItemPointer, items * sizeof(meanResChildSetRaw[0]));
		//memcpy(meanSqrResChildSetRaw, firstItemPointer, items * sizeof(meanSqrResChildSetRaw[0]));
		memcpyAtCompileTime<items>(meanResChildSetRaw, meanSqrResChildSetRaw, reinterpret_cast<const typename Value<TDataTypeArg>::TNodeIndexType*>(firstItemPointer));

		mean.resetData();
		mean_square.resetData();
		return;
	}

	burt_unreahable();
	return;
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> variance(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	Value<TDataTypeResult> mean_sqr = reduceMeanSquares<opHint, TDataTypeArg, TDataTypeResult>(firstItemPointer, items);
	Value<TDataTypeResult> mean = reduceMean<opHint, TDataTypeArg, TDataTypeResult>(firstItemPointer, items);
	Value<TDataTypeResult> sqr_mean = sqr(mean);
	
	if (items <= 1)
	{
		return mean_sqr - sqr_mean;
	}
	else
	{
		Value<TDataTypeResult> multiplier_correction = TDataTypeArg(items) / TDataTypeArg(items - 1);
		Value<TDataTypeResult> biased_variance = mean_sqr - sqr_mean;
		Value<TDataTypeResult> unbiased_variance = multiplier_correction * biased_variance;
		return unbiased_variance;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg, class TDataTypeResult = TDataTypeArg>
inline Value<TDataTypeResult> varianceBiased(Value<TDataTypeArg>* firstItemPointer, size_t items) noexcept
{
	Value<TDataTypeResult> mean_sqr = reduceMeanSquares<opHint, TDataTypeArg, TDataTypeResult>(firstItemPointer, items);
	Value<TDataTypeResult> mean = reduceMean<opHint, TDataTypeArg, TDataTypeResult>(firstItemPointer, items);
	Value<TDataTypeResult> sqr_mean = sqr(mean);
	
	return mean_sqr - sqr_mean;
}

// Operators Wrapper for Compute: value, value
template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> add (const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(first.dataRef() + second.dataRef(), OpType::eBinaryAdd, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinaryAdd, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> addSquares(const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(first.dataRef() * first.dataRef() + second.dataRef() * second.dataRef(), OpType::eBinaryAddSquares, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinaryAddSquares, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> mean(const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res( (first.dataRef() + second.dataRef()) * TDataTypeResult(1.0/2.0), OpType::eBinaryMean, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinaryMean, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> negativeMean(const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	if constexpr (opHint == OpHint::eOpNoHints)
	{
        Value<TDataTypeResult> res((first.dataRef() + second.dataRef()) * TDataTypeResult(-1.0/2.0), OpType::eBinaryNegativeMean, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
        Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinaryNegativeMean, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> meanSquares(const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res((first.dataRef() * first.dataRef() + second.dataRef() * second.dataRef()) * TDataTypeResult(1.0 / 2.0), OpType::eBinaryMeanSquares, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinaryMean, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> operator + (const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	return add<opHint, TDataTypeArg1, TDataTypeArg2, TDataTypeResult>(first, second);
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> sub(const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(first.dataRef() - second.dataRef(), OpType::eBinarySub, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinarySub, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> operator - (const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	return sub<opHint, TDataTypeArg1, TDataTypeArg2, TDataTypeResult>(first, second);
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> mulByConstant(const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& constantValue) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(first.dataRef() * constantValue.dataRef(), OpType::eBinaryMultByConst, first.sysGetRawNodeIndex(), constantValue.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinaryMultByConst, first.sysGetRawNodeIndex(), constantValue.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> mul (const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(first.dataRef() * second.dataRef(), OpType::eBinaryMult, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinaryMult, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> operator * (const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	return mul<opHint, TDataTypeArg1, TDataTypeArg2, TDataTypeResult>(first, second);
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> div (const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept
{
	if constexpr (opHint == OpHint::eOpNoHints)
	{
		Value<TDataTypeResult> res(first.dataRef() / second.dataRef(), OpType::eBinaryDiv,
			first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex()
		);
		return res;
	}
	else if constexpr (opHint == OpHint::eOpHintNotEvaluateValue)
	{
		Value<TDataTypeResult> res(TDataTypeResult(), OpType::eBinaryDiv, first.sysGetRawNodeIndex(), second.sysGetRawNodeIndex());
		return res;
	}

	burt_unreahable();
	return Value<TDataTypeResult>();
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult> operator / (const Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	return div<opHint, TDataTypeArg1, TDataTypeArg2, TDataTypeResult>(first, second);
}

//===========================================================================================================================================================================================//
// Some in-place operations start
//===========================================================================================================================================================================================//
template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult>& addInplace(Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	//first.sysDecreaseGC();
	first = add<opHint>(first, second);
	return first;
}
template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult>& operator += (Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	return addInplace<opHint>(first, second);
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult>& subInplace(Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	//first.sysDecreaseGC();
	first = sub<opHint>(first, second);
	return first;
}
template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult>& operator -= (Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	return subInplace<opHint>(first, second);
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult>& multInplace(Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	//first.sysDecreaseGC();
	first = mul<opHint>(first, second);
	return first;
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult>& operator *= (Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	return multInplace<opHint>(first, second);
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult>& divInplace(Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	//first.sysDecreaseGC();
	first = div<opHint>(first, second);
	return first;
}

template <OpHint opHint = OpHint::eOpNoHints, class TDataTypeArg1, class TDataTypeArg2, class TDataTypeResult = TDataTypeArg2>
inline Value<TDataTypeResult>& operator /= (Value<TDataTypeArg1>& first, const Value<TDataTypeArg2>& second) noexcept {
	return divInplace<opHint>(first, second);
}
//===========================================================================================================================================================================================//
// Some in-place operations end
//===========================================================================================================================================================================================//

template <class TValueType>
inline std::vector<typename TValueType::TNodeIndexType> collectAllNodes(TValueType& root) noexcept
{
	using TNodeIndexType = typename TValueType::TNodeIndexType;

	// collect all nodes in DFS manners: such not-recursive apprach is fine except it does not handle post-order event w/o serious modifications
	std::vector<TNodeIndexType> res;
	std::vector<TNodeIndexType> queue;
	std::unordered_map<TNodeIndexType, bool> marked_nodes;

	queue.push_back(root.sysGetRawNodeIndex());

	for (; !queue.empty();)
	{
		// Take item from BFS queue
		TNodeIndexType item_index = queue.back();
		const TValueType* item = TValueType::sysViewMemoryAsNode(&item_index);

		queue.pop_back();

		// Is node not yet marked: find node in market set and flag for it is set to true
		if (marked_nodes.find(item_index) != marked_nodes.end() && marked_nodes[item_index])
		{
			// Already processed
			continue;
		}

		// Get children
		const auto& children = item->childrenSet();
		size_t numChidren = children.size();

        for (size_t i = 0; i < numChidren; ++i)
		{
            auto cvalue = children[i];

            if (marked_nodes.find(cvalue) == marked_nodes.end())
			{
                queue.push_back(cvalue);
			}
		}
		marked_nodes[item_index] = true;

		// Process node
		res.push_back(item_index);
	}

	return res;
}

template <class TValueType>
inline std::vector<typename TValueType::TNodeIndexType> collectAllNonLeafNodes(TValueType& root) noexcept
{
	using TNodeIndexType = typename TValueType::TNodeIndexType;

	// collect all nodes in DFS manners: such not-recursive apprach is fine except it does not handle post-order event w/o serious modifications
	std::vector<TNodeIndexType> res;
	std::vector<TNodeIndexType> queue;
	std::unordered_map<TNodeIndexType, bool> marked_nodes;
	
	if (root.childrenSet().size() != 0)
		queue.push_back(root.sysGetRawNodeIndex());

	for (; !queue.empty();)
	{
		// Take item from BFS queue
		TNodeIndexType item_index = queue.back();
		const TValueType* item = TValueType::sysViewMemoryAsNode(&item_index);

		queue.pop_back();

		// Is node not yet marked: find node in market set and flag for it is set to true
		if (marked_nodes.find(item_index) != marked_nodes.end() && marked_nodes[item_index])
		{
			// Already processed
			continue;
		}

		// Get children
		const auto& children = item->childrenSet();
		size_t numChidren = children.size();

        for (size_t i = 0; i < numChidren; ++i)
        {
            auto cvalue = children[i];
            if (marked_nodes.find(cvalue) == marked_nodes.end())
			{
                queue.push_back(cvalue);
			}
		}
		marked_nodes[item_index] = true;

		// Process node
		if (numChidren != 0)
			res.push_back(item_index);
	}

	return res;
}

template <class TValueType>
inline void cleanGrad(TValueType& root) noexcept
{
	using TGradDataType = typename TValueType::TGradDataType;
	using TNodeIndexType = typename TValueType::TNodeIndexType;

	std::vector<TNodeIndexType> nodes = collectAllNodes(root);
	for (size_t i = 0; i < nodes.size(); ++i)
	{
		TNodeIndexType node_index = nodes[i];
		TValueType::sysViewMemoryAsNode(&node_index)->setGradToZero();
	}
}

template <class TValueType>
inline void cleanGradForNonLeafNodes(TValueType& root) noexcept
{
	using TGradDataType = typename TValueType::TGradDataType;
	using TNodeIndexType = typename TValueType::TNodeIndexType;

	std::vector<TNodeIndexType> nodes = collectAllNonLeafNodes(root);
	for (size_t i = 0; i < nodes.size(); ++i)
	{
		TNodeIndexType node_index = nodes[i];
		TValueType::sysViewMemoryAsNode(&node_index)->setGradToZero();
	}
}
