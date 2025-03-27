#include "gtest/gtest.h"

#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"
#include "burtcore/include/burtorch.h"

#if SUPPORT_CPU_SSE2_128_bits | SUPPORT_CPU_AVX_256_bits | SUPPORT_CPU_AVX_512_bits | SUPPORT_CPU_CPP_TS_V2_SIMD

TEST(burt, BurtSimdGTest)
{
	using TElementType = double;
	typedef burt::VectorSimdTraits<TElementType, burt::cpu_extension>::VecType VecType;

	{
		Value<VecType> a = Value(VecType(1.0));
		EXPECT_TRUE(fabs(::horizontal_add(a.dataCopy() - VecType(1.0))) < 1e-6);

		Value<VecType> b = Value(VecType(2.0));
		EXPECT_TRUE(fabs(::horizontal_add(b.dataCopy() - VecType(2.0))) < 1e-6);

		EXPECT_TRUE(Value<VecType>::numActiveNodes() == 2);
	}

	{
		Value<VecType> a_p_one = Value<VecType>(1.0);
		Value<VecType> a_n_one = Value<VecType>(-1.0);

		Value<VecType> s_p_one = sigmoid(a_p_one);
		Value<VecType> s_n_one = sigmoid(a_n_one);
		Value r_p_one = relu(a_p_one);
		Value r_n_one = relu(a_n_one);

		for (size_t i = 0; i < a_p_one.dataRef().size(); ++i)
		{
			EXPECT_TRUE(fabs(s_p_one.dataCopy()[i] - (0.7310585786300049)) < 1e-6);
			EXPECT_TRUE(fabs(s_n_one.dataCopy()[i] - (0.2689414213699951)) < 1e-6);
			EXPECT_TRUE(fabs(r_p_one.dataCopy()[i] - (1.0)) < 1e-6);
			EXPECT_TRUE(fabs(r_n_one.dataCopy()[i] - (0.0)) < 1e-6);
		}
	}

	{
		Value<VecType> a = Value(VecType(3.0));
		Value<VecType> b = Value(VecType(2.0));
		Value<VecType> c = a - b;

		for (size_t i = 0; i < c.dataRef().size(); ++i)
		{
			EXPECT_TRUE(fabs(c.dataRef()[i] - 1.0) < 1e-6);
			EXPECT_TRUE(fabs(a.dataRef()[i] - 3.0) < 1e-6);
			EXPECT_TRUE(fabs(b.dataRef()[i] - 2.0) < 1e-6);

			EXPECT_TRUE(fabs(c.dataCopy()[i] - 1.0) < 1e-6);
			EXPECT_TRUE(fabs(a.dataCopy()[i] - 3.0) < 1e-6);
			EXPECT_TRUE(fabs(b.dataCopy()[i] - 2.0) < 1e-6);
		}
	}

	{
		Value<VecType> a = Value(VecType(3.0));
		Value<VecType> b = Value(VecType(2.0));

		Value<VecType> c_mult = a * b;
		for (size_t i = 0; i < a.dataRef().size(); ++i) {
			EXPECT_TRUE(fabs(c_mult.dataRef()[i] - 3.0 * 2.0) < 1e-6);
		}
		Value<VecType> c_div = a / b;
		for (size_t i = 0; i < a.dataRef().size(); ++i) {
			EXPECT_TRUE(fabs(c_div.dataRef()[i] - 3.0 / 2.0) < 1e-6);
		}
	}

	{
		Value<VecType> b = Value(VecType(3.0));
		backward(b);
		for (size_t i = 0; i < b.dataRef().size(); ++i) {
			EXPECT_TRUE(fabs(b.dataCopy()[i] - (3.0)) < 1e-10);
			EXPECT_TRUE(fabs(b.gradCopy()[i] - (1.0)) < 1e-10);
		}
	}

	{
		Value<VecType> a = Value<VecType>(-41.0);
		Value<VecType> b = Value<VecType>(2.0);
		Value<VecType> c = a + b;
		Value<VecType> ab = a * b;
		Value<VecType> b_cub = pow3(b);
		Value<VecType> d = ab + b_cub;
		Value<VecType> e = c - d;
		Value<VecType> f = sqr(e);
		Value<VecType> const_0_5 = Value<VecType>::getConstant(1 / 2.0);
		Value<VecType> g = f * const_0_5;

		backward(g);
		EXPECT_TRUE(fabs(g.dataCopy()[0] - 612.50) < 1e-10);
		EXPECT_TRUE(fabs(a.gradCopy()[0] - (-35.0)) < 1e-10);
		EXPECT_TRUE(fabs(b.gradCopy()[0] - 1050.0) < 1e-10);
	}

	{
		double dh = 1e-6;

		Value<VecType> a = Value<VecType>(-41.0);
		Value<VecType> b = Value<VecType>(2.0);
		Value<VecType> c = a / b;
		backward(c);
		{
			Value<VecType> a_ = Value<VecType>(-41.0);
			Value<VecType> b_ = Value<VecType>(2.0 + dh);
			Value<VecType> c_ = a_ / b_;
			VecType grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			VecType grad_bdiff = b.gradCopy();
			for (int ii = 0; ii < grad_num.size(); ++ii)
			{
				EXPECT_TRUE(fabs(grad_num[ii] - grad_bdiff[ii]) < 1e-3);
			}
		}
	}

	{
		Value<VecType> a1 = Value<VecType>(-41.0);
		Value<VecType> a2 = Value<VecType>(-42.0);
		Value<VecType> a3 = Value<VecType>(+11.0);
		Value<VecType> a4 = Value<VecType>(-43.0);
		Value<VecType> a5 = Value<VecType>(-44.5);
		Value<VecType> a6 = Value<VecType>(-1.0);
		Value<VecType> a7 = Value<VecType>(+2.0);
		Value<VecType> a8 = Value<VecType>(+3.5);

		Value<VecType> a_arr[8] = { a1, a2, a3, a4, a5, a6, a7, a8 };
		Value b = reduceSum(a_arr, sizeof(a_arr) / sizeof(a_arr[0]));
		backward(b);

		for (size_t i = 0; i < sizeof(a_arr) / sizeof(a_arr[0]); ++i)
		{
			for (int ii = 0; ii < a_arr[i].gradRef().size(); ++ii)
			{
				EXPECT_TRUE(fabs(a_arr[i].gradRef()[ii] - 1.0) <= 1e-6);
				EXPECT_TRUE(fabs(a_arr[i].gradCopy()[ii] - 1.0) <= 1e-6);
			}
		}
	}

	{
		double dh = 1e-6;
		Value<VecType> a = Value<VecType>(11.0);
		Value<VecType> b = negativeLog(a);
		backward(b);
		{
			Value<VecType> a_ = Value<VecType>(11.0 + dh);
			Value<VecType> b_ = negativeLog(a_);
			for (int ii = 0; ii < a.gradRef().size(); ++ii)
			{
				double grad_num = (b_.dataCopy()[ii] - b.dataCopy()[ii]) / dh;
				double grad_adiff = a.gradCopy()[ii];
				EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
			}
		}
	}
}

#endif
