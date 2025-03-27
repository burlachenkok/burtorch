#include "gtest/gtest.h"
#include "burtcore/include/burtorch.h"

TEST(burt, BurtScalarInPlaceGTest)
{
	// TEST EXPRESSIONS INPLACE
	{
		Value a = Value(2.0);
		Value b = Value(3.0);
		Value c = Value(70.0);

		std::vector<decltype(a)> abc_vec = { a, b, c };

		Value abc_mean = reduceMean(abc_vec.data(), abc_vec.size());
		EXPECT_TRUE(fabs(abc_mean.dataCopy() - 25) < 1e-10);

		Value abc_sum = reduceSum(abc_vec.data(), abc_vec.size());
		EXPECT_TRUE(fabs(abc_sum.dataCopy() - 75) < 1e-10);

		Value abc_var = variance(abc_vec.data(), abc_vec.size());
		EXPECT_TRUE(fabs(abc_var.dataCopy() - 1519) < 1e-10);

		Value abc_var_biased = varianceBiased(abc_vec.data(), abc_vec.size());
		EXPECT_TRUE(fabs(abc_var_biased.dataCopy() - (((2. - 25.) * (2. - 25.) + (3. - 25.) * (3. - 25.) + (70. - 25.) * (70. - 25.)) / 3.0)) < 1e-10);
	}

	{
		Value a = Value(2.0);
		Value b = Value(3.0);
		Value c = Value(5.0);

		Value d_mult = Value(10.0);
		d_mult *= a;
		EXPECT_TRUE(fabs(d_mult.dataCopy() - 20.0) < 1e-10);
		d_mult *= b;
		EXPECT_TRUE(fabs(d_mult.dataCopy() - 60.0) < 1e-10);
		d_mult *= c;
		EXPECT_TRUE(fabs(d_mult.dataCopy() - 300.0) < 1e-10);
		d_mult *= c;
		EXPECT_TRUE(fabs(d_mult.dataCopy() - 1500.0) < 1e-10);

		Value ab_avg = mean(a, b);
		EXPECT_TRUE(fabs(ab_avg.dataCopy() - 5.0 / 2) < 1e-10);

		Value<double> arr[3] = { a, b, c };
		Value abc_avg = reduceMean(arr, 3);
		EXPECT_TRUE(fabs(abc_avg.dataCopy() - 10.0 / 3) < 1e-10);

		Value ab_avg_neg = negativeMean(a, b);
		EXPECT_TRUE(fabs(ab_avg_neg.dataCopy() - (-5.0 / 2)) < 1e-10);

		Value<double> arr_neg[3] = { a, b, c };
		Value abc_avg_neg = reduceNegativeMean(arr_neg, 3);
		EXPECT_TRUE(fabs(abc_avg_neg.dataCopy() - (-10.0 / 3)) < 1e-10);
	}

	// TEST EXPRESSIONS INPLACE
	{
		Value a = Value(2.0);
		Value b = Value(3.0);
		Value c = Value(5.0);

		Value d_add = Value(10.0);
		d_add += a;
		EXPECT_TRUE(fabs(d_add.dataCopy() - 12.0) < 1e-10);
		d_add += b;
		EXPECT_TRUE(fabs(d_add.dataCopy() - 15.0) < 1e-10);
		d_add += c;
		EXPECT_TRUE(fabs(d_add.dataCopy() - 20.0) < 1e-10);
		d_add += c;
		EXPECT_TRUE(fabs(d_add.dataCopy() - 25.0) < 1e-10);
		d_add *= a;
		EXPECT_TRUE(fabs(d_add.dataCopy() - 50.0) < 1e-10);
		d_add /= a;
		EXPECT_TRUE(fabs(d_add.dataCopy() - 25.0) < 1e-10);
	}

	// TEST EXPRESSIONS INPLACE
	{
		Value a = Value(2.0);
		Value b = Value(3.0);
		Value c = Value(5.0);

		Value d_sub = Value(10.0);
		d_sub -= a;
		EXPECT_TRUE(fabs(d_sub.dataCopy() - 8.0) < 1e-10);
		d_sub -= b;
		EXPECT_TRUE(fabs(d_sub.dataCopy() - 5.0) < 1e-10);
		d_sub -= c;
		EXPECT_TRUE(fabs(d_sub.dataCopy() - 0.0) < 1e-10);
		d_sub -= c;
		EXPECT_TRUE(fabs(d_sub.dataCopy() - (-5.0)) < 1e-10);
		d_sub *= a;
		EXPECT_TRUE(fabs(d_sub.dataCopy() - (-10.0)) < 1e-10);
	}
}

TEST(burt, BurtScalarGTest)
{
	{
		Value b = Value(3.0);
		backward(b);
		EXPECT_TRUE(fabs(b.dataCopy() - (3.0)) < 1e-10);
		EXPECT_TRUE(fabs(b.gradCopy() - (1.0)) < 1e-10);
	}

	{
		Value b = Value(3.0);

		Value x1 = Value(-41.0);
		Value x2 = Value(2.5);

		Value w1 = Value(5.0);
		Value w2 = Value(4.0);

		Value in = innerProduct({ x1, x2 }, { w1, w2 });
		EXPECT_TRUE(fabs(in.dataCopy() - (-41.0 * 5.0 + 2.5 * 4.0)) < 1e-10);

		Value in_with_bias = innerProductWithBias(b, { x1, x2 }, { w1, w2 });
		EXPECT_TRUE(fabs(in_with_bias.dataCopy() - (3.0 + -41.0 * 5.0 + 2.5 * 4.0)) < 1e-10);
	}
	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a + b;
		Value ab = a * b;
		Value b_cub = pow3(b);
		Value d = ab + b_cub;
		Value e = c - d;
		Value f = sqr(e);
		Value const_0_5 = Value<double>::getConstant(1 / 2.0);
		Value g = f * const_0_5;

		backward(g);
		EXPECT_TRUE(fabs(g.dataCopy() - 612.50) < 1e-10);
		EXPECT_TRUE(fabs(a.gradCopy() - (-35.0)) < 1e-10);
		EXPECT_TRUE(fabs(b.gradCopy() - 1050.0) < 1e-10);
	}
	// no final value
	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a + b;
		Value ab = a * b;
		Value b_cub = pow3(b);
		Value d = ab + b_cub;
		Value e = c - d;
		Value f = sqr(e);
		Value const_0_5 = Value<double>::getConstant(1 / 2.0);

		Value g = mul<OpHint::eOpHintNotEvaluateValue>(f, const_0_5);

		backward(g);
		EXPECT_TRUE(fabs(g.dataCopy() - 0.0) < 1e-10);
		EXPECT_TRUE(fabs(a.gradCopy() - (-35.0)) < 1e-10);
		EXPECT_TRUE(fabs(b.gradCopy() - 1050.0) < 1e-10);
	}

	// TEST VALUES
	{
		auto c1 = Value<double>::numActiveNodes();
		Value a = Value(-41.0);
		auto c2 = Value<double>::numActiveNodes();
		Value b = a;
		auto c3 = Value<double>::numActiveNodes();
		EXPECT_TRUE(c1 + 1 == c2);
		EXPECT_TRUE(c3 == c2);
	}

	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a + b;
		EXPECT_TRUE(fabs(c.dataCopy() - (-41.0 + 2.0)) < 1e-6);
	}

	{
		Value a = Value(16.0);
		Value csqrt = sqrt(a);
		EXPECT_TRUE(fabs(csqrt.dataCopy() - (4.0)) < 1e-6);
		backward(csqrt);

		{
			double dh = 1e-3;
			Value a_ = Value(16.0 + dh);
			Value csqrt_ = sqrt(a_);
			double grad_num = (csqrt_.dataCopy() - csqrt.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(1 / 16.0);
		Value icsqrt = invSqrt(a);
		EXPECT_TRUE(fabs(icsqrt.dataCopy() - (4.0)) < 1e-6);
		backward(icsqrt);

		{
			double dh = 1e-6;
			Value a_ = Value(1 / 16.0 + dh);
			Value icsqrt_ = invSqrt(a_);
			double grad_num = (icsqrt_.dataCopy() - icsqrt.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}


	{
		Value a_p_one = Value(1.0);
		Value a_n_one = Value(-1.0);

		Value s_p_one = sigmoid(a_p_one);
		Value s_n_one = sigmoid(a_n_one);
		EXPECT_TRUE(fabs(s_p_one.dataCopy() - (0.7310585786300049)) < 1e-6);
		EXPECT_TRUE(fabs(s_n_one.dataCopy() - (0.2689414213699951)) < 1e-6);

		Value r_p_one = relu(a_p_one);
		Value r_n_one = relu(a_n_one);
		EXPECT_TRUE(fabs(r_p_one.dataCopy() - (1.0)) < 1e-6);
		EXPECT_TRUE(fabs(r_n_one.dataCopy() - (0.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a / b;
		EXPECT_TRUE(fabs(c.dataCopy() - (-41.0 / 2.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a - b;
		EXPECT_TRUE(fabs(c.dataCopy() - (-41.0 - 2.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a * b;
		EXPECT_TRUE(fabs(c.dataCopy() - (-41.0 * 2.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = exp(a);
		EXPECT_TRUE(fabs(b.dataCopy() - exp(-41.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = inv(a);
		EXPECT_TRUE(fabs(b.dataCopy() - 1.0 / (-41.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = sqr(a);
		EXPECT_TRUE(fabs(b.dataCopy() - (41.0 * 41.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = pow3(a);
		EXPECT_TRUE(fabs(b.dataCopy() - (-41.0 * 41.0 * 41.0)) < 1e-6);
	}

	{
		Value a1 = Value(-41.0);
		Value a2 = Value(-42.0);
		Value a3 = Value(+11.0);
		Value a4 = Value(-43.0);
		Value a5 = Value(-44.5);
		Value a6 = Value(-1.0);
		Value a7 = Value(+2.0);
		Value a8 = Value(+3.5);

		Value<double> a_arr[8] = { a1, a2, a3, a4, a5, a6, a7, a8 };
		double a_sum = (-41.0) + (-42.0) + (+11.0) + (-43.0) + (-44.5) + (-1.0) + (+2.0) + (+3.5);
		Value b = reduceSum(a_arr, sizeof(a_arr) / sizeof(a_arr[0]));
		EXPECT_TRUE(fabs(b.dataCopy() - a_sum) < 1e-6);
	}

	{
		Value a = Value(11.0);
		{
			Value b = negativeLog(a);
			EXPECT_TRUE(fabs(b.dataCopy() - (-log(11.0))) < 1e-6);
		}
		{
			Value b = logarithm(a);
			EXPECT_TRUE(fabs(b.dataCopy() - (log(11.0))) < 1e-6);
		}
	}

	// TEST GRADS
	double dh = 1e-6;
	EXPECT_TRUE(dh > DBL_EPSILON);

	// TEST EXPRESSIONS INPLACE
	{
		Value a = Value(2.0);
		Value c = Value(10.0);
		c *= a + Value(1.0);
		EXPECT_TRUE(fabs(c.dataCopy() - 10.0 * 3.0) < 1e-10);
		backward(c);
		{
			Value a_ = Value(2.0 + dh);
			Value c_ = Value(10.0);
			c_ *= a_ + Value(1.0);
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}
	{

		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a + b;
		backward(c);

		{
			Value a_ = Value(-41.0 + dh);
			Value b_ = Value(2.0);
			Value c_ = a_ + b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
		{
			Value a_ = Value(-41.0);
			Value b_ = Value(2.0 + dh);
			Value c_ = a_ + b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = b.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a - b;
		backward(c);
		{
			Value a_ = Value(-41.0 + dh);
			Value b_ = Value(2.0);
			Value c_ = a_ - b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
		{
			Value a_ = Value(-41.0);
			Value b_ = Value(2.0 + dh);
			Value c_ = a_ - b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = b.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a * b;
		backward(c);
		{
			Value a_ = Value(-41.0 + dh);
			Value b_ = Value(2.0);
			Value c_ = a_ * b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(-1.5);
		Value b = exp(a);
		backward(b);
		{
			Value a_ = Value(-1.5 + dh);
			Value b_ = exp(a_);
			double grad_num = (b_.dataCopy() - b.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(-1.0);
		Value b = tanh(a);
		backward(b);
		{
			Value a_ = Value(-1.0 + dh);
			Value b_ = tanh(a_);
			double grad_num = (b_.dataCopy() - b.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(-41.0);
		Value b = inv(a);
		backward(b);
		{
			Value a_ = Value(-41.0 + dh);
			Value b_ = inv(a_);
			double grad_num = (b_.dataCopy() - b.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(-41.0);
		Value b = sqr(a);
		backward(b);
		{
			Value a_ = Value(-41.0 + dh);
			Value b_ = sqr(a_);

			double grad_num = (b_.dataCopy() - b.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(-41.0);
		Value b = pow3(a);

		backward(b);
		{
			Value a_ = Value(-41.0 + dh);
			Value b_ = pow3(a_);

			double grad_num = (b_.dataCopy() - b.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a / b;
		backward(c);
		{
			Value a_ = Value(-41.0 + dh);
			Value b_ = Value(2.0);
			Value c_ = a_ / b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}


	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a / b;
		backward(c);
		{
			Value a_ = Value(-41.0);
			Value b_ = Value(2.0 + dh);
			Value c_ = a_ / b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_bdiff = b.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_bdiff) < 1e-3);
		}
	}

	{
		Value a1 = Value(-41.0);
		Value a2 = Value(-42.0);
		Value a3 = Value(+11.0);
		Value a4 = Value(-43.0);
		Value a5 = Value(-44.5);
		Value a6 = Value(-1.0);
		Value a7 = Value(+2.0);
		Value a8 = Value(+3.5);
		Value<double> a_arr[8] = { a1, a2, a3, a4, a5, a6, a7, a8 };
		Value b = reduceSum(a_arr, sizeof(a_arr) / sizeof(a_arr[0]));
		backward(b);

		for (size_t i = 0; i < sizeof(a_arr) / sizeof(a_arr[0]); ++i)
		{
			EXPECT_TRUE(fabs(a_arr[i].gradRef() - 1.0) <= 1e-6);
			EXPECT_TRUE(fabs(a_arr[i].gradCopy() - 1.0) <= 1e-6);
		}
	}

	{
		Value a1 = Value(-41.0);
		Value a2 = Value(-42.0);
		Value a3 = Value(+11.0);
		Value a4 = Value(-43.0);
		Value a5 = Value(-44.5);
		Value a6 = Value(-1.0);
		Value a7 = Value(+2.0);
		Value a8 = Value(+3.5);

		double a_raw_data[8] = { -41.0, -42.0, +11.0, -43.0, -44.5, -1.0, +2.0, +3.5 };
		Value<double> a_arr[8] = { a1, a2, a3, a4, a5, a6, a7, a8 };
	}

	{
		Value a = Value(11.0);
		Value b = negativeLog(a);
		backward(b);
		{
			Value a_ = Value(11.0 + dh);
			Value b_ = negativeLog(a_);
			double grad_num = (b_.dataCopy() - b.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value x1 = Value(-41.0);
		Value x2 = Value(2.5);
		Value w1 = Value(5.0);
		Value w2 = Value(4.0);
		Value in = innerProduct({ x1, x2 }, { w1, w2 });
		backward(in);
		{
			Value x1_ = Value(-41.0 + dh);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0);
			Value in_ = innerProduct({ x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
			EXPECT_TRUE(fabs(x1.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5 + dh);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0);
			Value in_ = innerProduct({ x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
			EXPECT_TRUE(fabs(x2.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0 + dh);
			Value w2_ = Value(4.0);
			Value in_ = innerProduct({ x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
			EXPECT_TRUE(fabs(w1.gradCopy() - grad_num) < 1e-3);
		}

		{
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0 + dh);
			Value in_ = innerProduct({ x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
			EXPECT_TRUE(fabs(w2.gradCopy() - grad_num) < 1e-3);
		}
	}



	{
		Value bias = Value(3.5);

		Value x1 = Value(-41.0);
		Value x2 = Value(2.5);
		Value w1 = Value(5.0);
		Value w2 = Value(4.0);
		Value in = innerProductWithBias(bias, { x1, x2 }, { w1, w2 });
		backward(in);

		{
			Value bias_ = Value(3.5 + dh);

			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0);
			Value in_ = innerProductWithBias(bias_, { x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
			EXPECT_TRUE(fabs(bias.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value bias_ = Value(3.5);
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5 + dh);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0);
			Value in_ = innerProductWithBias(bias_, { x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
			EXPECT_TRUE(fabs(x2.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value bias_ = Value(3.5);
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0 + dh);
			Value w2_ = Value(4.0);
			Value in_ = innerProductWithBias(bias_, { x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
			EXPECT_TRUE(fabs(w1.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value bias_ = Value(3.5);
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0 + dh);
			Value in_ = innerProductWithBias(bias_, { x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
			EXPECT_TRUE(fabs(w2.gradCopy() - grad_num) < 1e-3);
		}
	}

	{
		Value c = Value<double>::getConstant(1.0);

		Value x = Value(-4.0);
		Value y = Value(8.0);

		Value y2 = sqr(y);
		Value x2 = sqr(x);

		Value z = x2 + y2;
		Value v = z + c;

		backward(v);
		EXPECT_TRUE(fabs(x.gradCopy() - (2.0 * -4.0)) < 1e-3);
		cleanGrad(v);
		EXPECT_TRUE(fabs(x.gradCopy() - (0.0)) < 1e-3);
		backward(v);
		EXPECT_TRUE(fabs(x.gradCopy() - (2.0 * -4.0)) < 1e-3);
		cleanGradForNonLeafNodes(v);
		EXPECT_TRUE(fabs(x.gradCopy() - (2.0 * -4.0)) < 1e-3);
		EXPECT_TRUE(fabs(v.gradCopy() - (0.0)) < 1e-3);
		EXPECT_TRUE(fabs(z.gradCopy() - (0.0)) < 1e-3);
		EXPECT_TRUE(fabs(x2.gradCopy() - (0.0)) < 1e-3);
		EXPECT_TRUE(fabs(y2.gradCopy() - (0.0)) < 1e-3);

		backward(v);
		EXPECT_TRUE(fabs(x.gradCopy() - 2 * (2.0 * -4.0)) < 1e-3);
	}


	{
		double dh = 1e-6;
		{
			Value a = Value(1.0);
			Value s = sigmoid(a);
			backward(s);

			Value a_ = Value(1.0 + dh);
			Value s_ = sigmoid(a_);
			double grad_num = (s_.dataCopy() - s.dataCopy()) / dh;
			EXPECT_TRUE(fabs(a.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value a = Value(-1.0);
			Value s = sigmoid(a);
			backward(s);

			Value a_ = Value(-1.0 + dh);
			Value s_ = sigmoid(a_);
			double grad_num = (s_.dataCopy() - s.dataCopy()) / dh;
			EXPECT_TRUE(fabs(a.gradCopy() - grad_num) < 1e-3);
		}
	}

	{
		double dh = 1e-6;
		{
			Value a = Value(1.0);
			Value r = relu(a);
			backward(r);

			Value a_ = Value(1.0 + dh);
			Value r_ = relu(a_);
			double grad_num = (r_.dataCopy() - r.dataCopy()) / dh;
			EXPECT_TRUE(fabs(a.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value a = Value(-1.0);
			Value r = relu(a);
			backward(r);

			Value a_ = Value(-1.0 + dh);
			Value r_ = relu(a_);
			double grad_num = (r_.dataCopy() - r.dataCopy()) / dh;
			EXPECT_TRUE(fabs(a.gradCopy() - grad_num) < 1e-3);
		}
	}

	{
		double dh = 1e-6;
		Value a = Value(11.0);
		Value b = Value(23.0);
		Value c = addSquares(a, b);

		Value a_ = Value(11.0 + dh);
		Value b_ = Value(23.0);
		Value c_ = addSquares(a_, b_);

		EXPECT_TRUE(fabs(c.dataCopy() - (11.0 * 11.0 + 23.0 * 23)) < 1e-3);

		backward(c);
		double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
		double grad_adiff = a.gradCopy();
		EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);

		Value<double> in[2] = { a, b };
		Value c_var = reduceSumOfSquares(in, 2);
		EXPECT_TRUE(fabs(c_var.dataCopy() - (11.0 * 11.0 + 23.0 * 23)) < 1e-3);

		cleanGrad(c_var);
		backward(c_var);
		grad_adiff = a.gradCopy();
		EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
	}

	{
		double dh = 1e-6;
		Value a = Value(11.0);
		Value b = Value(23.0);
		Value c = meanSquares(a, b);

		Value a_ = Value(11.0 + dh);
		Value b_ = Value(23.0);
		Value c_ = meanSquares(a_, b_);

		EXPECT_TRUE(fabs(c.dataCopy() - (11.0 * 11.0 / 2.0 + 23.0 * 23 / 2.0)) < 1e-3);

		backward(c);
		double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
		double grad_adiff = a.gradCopy();
		EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
		backward(c);
		grad_adiff = a.gradCopy() / 2.0;
		EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);

		Value<double> in[2] = { a, b };
		Value c_var = reduceMeanSquares(in, 2);
		EXPECT_TRUE(fabs(c_var.dataCopy() - (11.0 * 11.0 / 2.0 + 23.0 * 23 / 2.0)) < 1e-3);

		cleanGrad(c_var);
		backward(c_var);
		grad_adiff = a.gradCopy();
		EXPECT_TRUE(fabs(grad_num - grad_adiff) < 1e-3);
	}
}