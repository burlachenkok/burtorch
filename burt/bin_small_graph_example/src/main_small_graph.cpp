#include "burtcore/include/burtorch.h"

#include "burt/fs/include/FileSystemHelpers.h"
#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <type_traits>

#include <math.h>
#include <assert.h>
#include <float.h>

int main_manual_backprop(int argc, char** argv)
{
	Value a(2.0);
	Value b(-3.0);
	Value c(10.0);
	Value e = a * b;
	Value<double> d = e + c;
	Value f(-2.0);
	Value L = d * f;

	a.setLabel("a");
    b.setLabel("b");
    c.setLabel("c");
    d.setLabel("d");
    e.setLabel("e");
	f.setLabel("f");
	L.setLabel("L");

	// manual gradient setup
	L.setGrad(1.0);
	f.setGrad(4.0);
	d.setGrad(-2.0);
	c.setGrad(-2.0);
	e.setGrad(-2.0);
	a.setGrad(-2.0 * -3.0);
	b.setGrad(-2.0 * 2.0);
	

	std::cout << (std::string)L << '\n';
	std::cout << L.childrenNum() << '\n';
	std::string res = buildDotGraph(L, "manual-autograd");
    burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());

	{
		double dh = 1e-5;
		Value a(2.0 + dh);
		Value b(-3.0);
		Value c(10.0);
		Value e = a * b;
		Value<double> d = e + c;
		Value f(-2.0);
		Value L2 = d * f;

		std::cout << "grad-check: " << (L2.dataCopy() - L.dataCopy()) / dh;
	}

	return 0;
}

#if 1
int main_backprop_almost(int argc, char** argv)
{
	// inputs
	Value x1(2.0, "x1");
	Value x2(0.0, "x2");

	// weights
	Value w1(-3.0, "w1");
	Value w2(1.0, "w2");
	// bias
	Value b(6.8813735870195, "b");

	Value x1w1 = x1 * w1;
	x1w1.setLabel("x1w1");

	Value x2w2 = x2 * w2;
	x2w2.setLabel("x2w2");

	Value x1w1_x2w2 = x1w1 + x2w2;
	x1w1_x2w2.setLabel("x1w1 + x2w2");

	Value n = x1w1_x2w2 + b;
	n.setLabel("n");

	Value o = tanh(n);
	o.setLabel("o");
	// showPlot(-10.0, +10.0, [](double x) {return tanh(x); }, "tanh");

	o.setGrad(1.0);
	n.setGrad(1.0 - o.dataRef() * o.dataRef());
	x1w1_x2w2.setGrad(0.5);
	b.setGrad(0.5);

	x1w1.setGrad(0.5);
	x2w2.setGrad(0.5);

	x2w2.setGrad(0.5);
	x1w1.setGrad(0.5);

	x2.setGrad(w2.dataCopy() * x2w2.gradCopy());
	w2.setGrad(x2.dataCopy() * x2w2.gradCopy());

	x1.setGrad(w1.dataCopy() * x1w1.gradCopy());
	w1.setGrad(x1.dataCopy() * x1w1.gradCopy());

	std::string res = buildDotGraph(o, "manual-autograd");
	burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());

	cleanGrad(o);
	res = buildDotGraph(o, "manual-autograd-clean");
	burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());

	o.setGrad(1.0);
	o.backward();
	n.backward();
	b.backward();
	x1w1_x2w2.backward();
	x2w2.backward();
	x1w1.backward();

	res = buildDotGraph(o, "manual-autograd-new");
	burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());



	cleanGrad(o);
	backward(o);
	res = buildDotGraph(o, "manual-autograd-auto");
	burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());

	return 0;
}
#endif

int main_(int argc, char** argv)
{
	if (1)
	{
		// subtle bug with using variable more then once
		Value a(3.0, "a");
		Value b = a + a;

		cleanGrad(b);
		backward(b);
		std::string res = buildDotGraph(b, "manual-autograd-auto");
		burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());
	}

	if (0)
	{
		// subtle bug with using variable more then once
		Value a(3.0, "a");
		Value c1 = Value<double>::getConstant(1.0);
		Value b = c1 + a;
		b.setLabel("b");

		cleanGrad(b);
		backward(b);
		std::string res = buildDotGraph(b, "manual-autograd-auto");
		burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());
	}

	if (0)
	{
		// subtle bug with using variable more then once
		Value a(2.0, "a");
		Value e = exp(a);

		cleanGrad(e);
		backward(e);
		std::string res = buildDotGraph(e, "manual-autograd-auto");
		burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());
	}

	if (1)
	{
		// subtle bug with using variable more then once
		Value a(10.0, "a");
		Value e = inv(a);

		cleanGrad(e);
		backward(e);
		std::string res = buildDotGraph(e, "manual-autograd-auto");
		burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());
	}


	Neuron<double> neu(3);
	std::vector<Value<double>> x = {  Value(1.0),  Value(2.0),  Value(3.0) };
	Value<double> act = neu.forward(x);

	MLPLayer n = MLPLayer<double>(3, 5);
	n.forward(x);
	std::vector<Value<double>> actL = n.forward(x);
	int hh = 1;

	MLP mlp = MLP<double>(3, { 4, 4, 1});
	std::vector<Value<double>> out = mlp.forward(x);
	std::string res = buildDotGraph(out[0], "manual-autograd-auto");
	burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());

	//
	{
		std::vector<Value<double>> x1 = {  Value(1.0),  Value(2.0),  Value(3.0) };
		std::vector<Value<double>> x2 = {  Value(113.0),  Value(1.0),  Value(2.1) };
		std::vector<Value<double>> x3 = {  Value(5.0),  Value(5.0),  Value(-111.0) };

		auto y1 = Value<double>::getConstant(1.0);
		auto y2 = Value<double>::getConstant(-1.0);
		auto y3 = Value<double>::getConstant(1.0);

		for (size_t e = 0; e < 2000; ++e)
		{
			auto diff_1 = y1 - mlp.forward(x1)[0];
			auto diff_2 = y2 - mlp.forward(x2)[0];
			auto diff_3 = y3 - mlp.forward(x3)[0];

			auto diff_1_sqr = sqr(diff_1);
			auto diff_2_sqr = sqr(diff_2);
			auto diff_3_sqr = sqr(diff_3);

			auto accum1 = diff_1_sqr + diff_2_sqr;
			auto accum2 = accum1 + diff_3_sqr;

			std::cout << " loss: " << accum2.asString() << '\n';

			cleanGrad(accum2);
			backward(accum2);
			auto params = mlp.parameters();
			for (auto node : params)
			{
				node.dataRef() -= 0.0001 * node.gradRef();
			}
		}

		std::cout << " pred-x1: " << mlp.forward(x1)[0].asString() << '\n';
		std::cout << " pred-x2: " << mlp.forward(x2)[0].asString() << '\n';
		std::cout << " pred-x3: " << mlp.forward(x3)[0].asString() << '\n';

	}

	return 0;
}

#include "burt/timers/include/HighPrecisionTimer.h"

template class Value<double>;
template class Value<float>;
template class SpecialArray<int, size_t>;

int main_simple(int argc, char** argv)
{
	int main_test_scalar(int argc, char** argv);
	main_test_scalar(argc, argv);

	int main_test_simd(int argc, char** argv);
	main_test_simd(argc, argv);

	burt::HighPrecisionTimer timer_main;

	for (size_t i = 0; i < 100*1000; ++i)
	{
		Value a = (-41.0);
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
	}
	double deltaSec = timer_main.getTimeSec();
	std::cout << deltaSec << '\n';

	return 0;
}

template<class TElementType>
int main_benchmark(int argc, char** argv, const char* test_name)
{
	std::cout << "================================\n";

	std::cout << test_name << '\n';

	burt::MutableData res_seq_topo;
	burt::MutableData res_set_topo;
	burt::MutableData recursion;

	// reserve 1KByte of memory of draft storage
	res_seq_topo.reserveMemory(128 * 8);
	res_set_topo.reserveMemory(128 * 8);
	recursion.reserveMemory(128 * 8);
	Value<TElementType>::reserveMemoryForNodes(128);

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
    burt_assert(Value<TElementType>::numActiveNodes() == 0);
	auto chk = Value<TElementType>::checkpointForNeurons();

	constexpr size_t kIterations = 100 * 1000;// *0 + 1;
	std::cout << "iterations: " << kIterations << '\n';

	// run without pre-allocated buffers
	{
		burt::HighPrecisionTimer timer_main;
		for (size_t i = 0; i < kIterations; ++i)
		{
			{
				Value a = TElementType(-41.0);
				Value b = Value(TElementType(2.0));
				Value c = a + b;
				Value ab = a * b;
				Value b_cub = pow3(b);
				Value d = ab + b_cub;
				Value e = c - d;
				Value f = sqr(e);
				Value const_0_5 = Value<TElementType>::getConstant(1/2.0);
				Value g = f * const_0_5;
				
				//Value<TElementType>::cleanAllGradFrom(chk);
				backward(g);
			}
			Value<TElementType>::restoreCheckpoint(chk);
//			Value<TElementType>::restoreCheckpoint(chk);
			//Value<TElementType>::deactiveUnusedNodes();
		}
		double deltaSec = timer_main.getTimeSec();
		std::cout << "without pre-allocated buffers: " << deltaSec << '\n';
	}

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
    burt_assert(Value<TElementType>::numActiveNodes() == 0);
	chk = Value<TElementType>::checkpointForNeurons();

	{
		burt::HighPrecisionTimer timer_main;
		for (size_t i = 0; i < kIterations; ++i)
		{
			{
				Value a = TElementType(-41.0);
				Value b = Value(TElementType(2.0));
				Value c = a + b;
				Value ab = a * b;
				Value b_cub = pow3(b);
				Value d = ab + b_cub;
				Value e = c - d;
				Value f = sqr(e);
				Value const_0_5 = Value<TElementType>::getConstant(1 / 2.0);
#if 1
				Value g = f * const_0_5;
#else
				Value g = mul<OpHint::eOpHintNotEvaluateValue>(f, const_0_5);
#endif
				//Value<TElementType>::cleanAllGradFrom(chk);
				backwardWithScratchStorage(g, res_seq_topo, res_set_topo, recursion);
			}
			Value<TElementType>::restoreCheckpoint(chk);
			//Value<TElementType>::deactiveUnusedNodes();
		}
		double deltaSec = timer_main.getTimeSec();
		std::cout << "with pre-allocated buffers: " << deltaSec << '\n';
	}
	// run with using topo sort

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
    burt_assert(Value<TElementType>::numActiveNodes() == 0);
	chk = Value<TElementType>::checkpointForNeurons();

	{
		burt::HighPrecisionTimer timer_main;
		for (size_t i = 0; i < kIterations; ++i)
		{
			{
				Value a = TElementType(-41.0);
				Value b = Value(TElementType(2.0));
				Value c = a + b;
				Value ab = a * b;
				Value b_cub = pow3(b);
				Value d = ab + b_cub;
				Value e = c - d;
				Value f = sqr(e);
				Value const_0_5 = Value<TElementType>::getConstant(1 / 2.0);
				Value g = f * const_0_5;

				//Value<TElementType>::cleanAllGradFrom(chk);
				backwardWithScratchStorage<decltype(a), false, true, false>(g, res_seq_topo, res_set_topo, recursion);
			}
			Value<TElementType>::restoreCheckpoint(chk);
			//Value<TElementType>::deactiveUnusedNodes();
		}
		double deltaSec = timer_main.getTimeSec();
		std::cout << "with precalculated topo-sort: " << deltaSec << '\n';
	}
	std::cout << "================================\n\n";


	{
		Value a = TElementType(-41.0);
		a.setLabel("a");

		Value b = Value(TElementType(2.0));
		b.setLabel("b");

		Value c = a + b;
		c.setLabel("c=a+b");

		Value ab = a * b;
		ab.setLabel("ab = a * b");

		Value b_cub = pow3(b);
		b_cub.setLabel("b^3");

		Value d = ab + b_cub;
		d.setLabel("d = ab + b^3");

		Value e = c - d;
		e.setLabel("e = c - d");

		Value f = sqr(e);
		f.setLabel("f=e^2");

		Value const_0_5 = Value<TElementType>::getConstant(1 / 2.0);
		const_0_5.setLabel("0.5");

		Value g = f * const_0_5;
		g.setLabel("g = f * 0.5");

		//Value<TElementType>::cleanAllGradFrom(chk);
		backward(g);

		std::string res = buildDotGraph(g, "manual-autograd");
		burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());
	}


	return 0;
}

template<class TElementType, bool save_graph_to_dot_file>
int main_full_benchmark(int argc, char** argv, const char* test_name)
{
	//getchar();
	std::cout << "================================\n";

	std::cout << test_name << '\n';

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
    burt_assert(Value<TElementType>::numActiveNodes() == 0);
	auto chk = Value<TElementType>::checkpointForNeurons();

	constexpr size_t kIterations = 200 * 1000;// *0 + 1;
	std::cout << "iterations: " << kIterations << '\n';

	// run without pre-allocated buffers
	{
		burt::HighPrecisionTimer timer_main;
		for (size_t i = 0; ; ++i)
		{
			{
				// https://github.com/karpathy/micrograd
				auto a = Value(-4.0);
				a.setLabel("a");
				auto b = Value(2.0);
				b.setLabel("b");

				auto c = a + b;
				c.setLabel("c=a+b");

				auto d = a * b + pow3(b);
				d.setLabel("d=a*b + pow3(b)");

				c += c + Value(1.0);
				c.setLabel("c+=c+1");

				c += Value(1.0) + c - a;
				c.setLabel("c+=1+c-a");

				d += d * Value(2.0) + relu(b + a);
				d.setLabel("d+=d*2 + relu(b+a)");

				d += Value(3.0) * d + relu(b - a);
				d.setLabel("d+=3*d + relu(b-a)");

				auto e = c - d;
				e.setLabel("e=c-d");
				
				auto f = sqr(e);
				f.setLabel("f=sqr(e)");

				auto g = f / Value(2.0);
				g.setLabel("g=f/2");

				g += Value(10.0) / f;
				g.setLabel("g+=10/f");

				//Value<TElementType>::cleanAllGradFrom(chk);
				backward(g);
				
				if (i == kIterations - 1)
				{
					std::cout << "g.data: " << g.dataCopy() << "\n";
					std::cout << "a.grad: " << a.gradRef() << "\n";
					std::cout << "b.grad: " << b.gradRef() << "\n";

					if constexpr (save_graph_to_dot_file)
					{
						std::string res = buildDotGraph(g, "manual-autograd-auto");
						burt::FileSystemHelpers::saveFile("test_2.txt", res.data(), res.size());
					}

					break;
				}
			}
			Value<TElementType>::restoreCheckpoint(chk);
			//			Value<TElementType>::restoreCheckpoint(chk);
						//Value<TElementType>::deactiveUnusedNodes();
		}
		double deltaSec = timer_main.getTimeSec();
		std::cout << std::fixed << std::setprecision(DBL_DECIMAL_DIG) << "without pre-allocated buffers: " << deltaSec << '\n';
	}

	//getchar();
	return 0;

	burt::MutableData res_seq_topo;
	burt::MutableData res_set_topo;
	burt::MutableData recursion;

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
    burt_assert(Value<TElementType>::numActiveNodes() == 0);
	chk = Value<TElementType>::checkpointForNeurons();

	// reserve 1KByte of memory of draft storage
	res_seq_topo.reserveMemory(128 * 8);
	res_set_topo.reserveMemory(128 * 8);
	recursion.reserveMemory(128 * 8);
	Value<TElementType>::reserveMemoryForNodes(128);

	{
		burt::HighPrecisionTimer timer_main;
		for (size_t i = 0; i < kIterations; ++i)
		{
			{
				// https://github.com/karpathy/micrograd
				auto a = Value(-4.0);
				auto b = Value(2.0);
				auto c = a + b;
				auto d = a * b + pow3(b);

				c += c + Value(1.0);
				c += Value(1.0) + c - a;
				d += d * Value(2.0) + relu(b + a);
				d += Value(3.0) * d + relu(b - a);
				auto e = c - d;
				auto f = sqr(e);
				auto g = f / Value(2.0);
				g += Value(10.0) / f;

				backward(g);
				//std::cout << "g.data: " << g.dataCopy() << "\n";
				//std::cout << "a.grad: " << a.gradRef() << "\n";
				//std::cout << "b.grad: " << b.gradRef() << "\n";


				backwardWithScratchStorage(g, res_seq_topo, res_set_topo, recursion);
			}
			Value<TElementType>::restoreCheckpoint(chk);
			//Value<TElementType>::deactiveUnusedNodes();
		}
		double deltaSec = timer_main.getTimeSec();
		std::cout << "with pre-allocated buffers: " << deltaSec << '\n';
	}
	// run with using topo sort

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
    burt_assert(Value<TElementType>::numActiveNodes() == 0);
	chk = Value<TElementType>::checkpointForNeurons();

	{
		burt::HighPrecisionTimer timer_main;
		for (size_t i = 0; i < kIterations; ++i)
		{
			{
				// https://github.com/karpathy/micrograd
				auto a = Value(-4.0);
				auto b = Value(2.0);
				auto c = a + b;
				auto d = a * b + pow3(b);

				c += c + Value(1.0);
				c += Value(1.0) + c - a;
				d += d * Value(2.0) + relu(b + a);
				d += Value(3.0) * d + relu(b - a);
				auto e = c - d;
				auto f = sqr(e);
				auto g = f / Value(2.0);
				g += Value(10.0) / f;

				backward(g);
				//std::cout << "g.data: " << g.dataCopy() << "\n";
				//std::cout << "a.grad: " << a.gradRef() << "\n";
				//std::cout << "b.grad: " << b.gradRef() << "\n";

				backwardWithScratchStorage<decltype(a), false, true, false>(g, res_seq_topo, res_set_topo, recursion);
			}
			Value<TElementType>::restoreCheckpoint(chk);
			//Value<TElementType>::deactiveUnusedNodes();
		}
		double deltaSec = timer_main.getTimeSec();
		std::cout << "with precalculated topo-sort: " << deltaSec << '\n';
	}
	std::cout << "================================\n\n";

	return 0;
}


template<class TElementType, bool save_action, bool load_action, size_t kIterations = 1>
int save_benchmark(int argc, char** argv, const char* test_name)
{
	std::cout << "================================\n";
	std::cout << test_name << '\n';

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
	burt_assert(Value<TElementType>::numActiveNodes() == 0);
	auto chk = Value<TElementType>::checkpointForNeurons();
	std::cout << "iterations: " << kIterations << '\n';

	// https://github.com/karpathy/micrograd
	auto a = Value(-4.0);
	auto b = Value(2.0);
	auto c = a + b;
	auto d = a * b + pow3(b);
	c += c + Value(1.0);
	c += Value(1.0) + c - a;
	d += d * Value(2.0) + relu(b + a);
	d += Value(3.0) * d + relu(b - a);

	auto e = c - d;
	auto f = sqr(e);

	auto g = f / Value(2.0);
	g += Value(10.0) / f;
	backward(g);

	burt::HighPrecisionTimer timer_main;
	//std::ostringstream fname;

	for (size_t i = 0; i < kIterations; ++i)
	{
		//fname << "my_file_f" << i << ".bin";
		saveToFile<true/*save_vaues*/, false/*save_gradients*/>({a,b,c,d,e,f,g}, "my_file_f.bin");
		//fname.seekp(0);
	}
	double deltaSec = timer_main.getTimeSec();
	std::cout << std::fixed << std::setprecision(DBL_DECIMAL_DIG) << "without pre-allocated buffers: " << deltaSec << '\n';

	return 0;
}



template<class TElementType, bool save_action, bool load_action, size_t kIterations = 1>
int save_benchmark_light(int argc, char** argv, const char* test_name)
{
	std::cout << "================================\n";
	std::cout << test_name << '\n';

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
	burt_assert(Value<TElementType>::numActiveNodes() == 0);
	auto chk = Value<TElementType>::checkpointForNeurons();
	std::cout << "iterations: " << kIterations << '\n';

	// https://github.com/karpathy/micrograd
	auto a = Value(-4.0);
	auto b = Value(2.0);
	auto c = a + b;
	auto d = a * b + pow3(b);
	c += c + Value(1.0);
	c += Value(1.0) + c - a;
	d += d * Value(2.0) + relu(b + a);
	d += Value(3.0) * d + relu(b - a);

	auto e = c - d;
	auto f = sqr(e);

	auto g = f / Value(2.0);
	g += Value(10.0) / f;
	backward(g);

	burt::HighPrecisionTimer timer_main;
	std::ostringstream fname;

	for (size_t i = 0; i < kIterations; ++i)
	{
		fname << "my_file_l" << i << ".bin";
		saveComputeGraphContextToFile<true/*save_vaues*/, false/*save_gradients*/, decltype(g)>(fname.str().c_str());
		fname.seekp(0);
	}
	double deltaSec = timer_main.getTimeSec();
	std::cout << std::fixed << std::setprecision(DBL_DECIMAL_DIG) << "without pre-allocated buffers: " << deltaSec << '\n';

	return 0;
}


template<class TElementType, bool save_action, bool load_action, size_t kIterations = 1>
int load_benchmark(int argc, char** argv, const char* test_name)
{
	std::cout << "================================\n";
	std::cout << test_name << '\n';

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
	burt_assert(Value<TElementType>::numActiveNodes() == 0);
	auto chk = Value<TElementType>::checkpointForNeurons();
	std::cout << "iterations: " << kIterations << '\n';

	// https://github.com/karpathy/micrograd
	auto a = Value(-4.0);
	auto b = Value(2.0);
	auto c = a + b;
	auto d = a * b + pow3(b);
	c += c + Value(1.0);
	c += Value(1.0) + c - a;
	d += d * Value(2.0) + relu(b + a);
	d += Value(3.0) * d + relu(b - a);

	auto e = c - d;
	auto f = sqr(e);

	auto g = f / Value(2.0);
	g += Value(10.0) / f;
	backward(g);

	burt::HighPrecisionTimer timer_main;
	std::ostringstream fname;

	for (size_t i = 0; i < kIterations; ++i)
	{
		loadFromFile<true/*load_vaues*/, false/*load_grads*/>({a,b,c,d,e,f,g}, "my_file_f.bin");
	}
	double deltaSec = timer_main.getTimeSec();
	std::cout << std::fixed << std::setprecision(DBL_DECIMAL_DIG) << "without pre-allocated buffers: " << deltaSec << '\n';

	return 0;
}


template<class TElementType, bool save_action, bool load_action, size_t kIterations = 1>
int load_benchmark_light(int argc, char** argv, const char* test_name)
{
	std::cout << "================================\n";
	std::cout << test_name << '\n';

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
	burt_assert(Value<TElementType>::numActiveNodes() == 0);
	auto chk = Value<TElementType>::checkpointForNeurons();
	std::cout << "iterations: " << kIterations << '\n';

	// https://github.com/karpathy/micrograd
	auto a = Value(-4.0);
	auto b = Value(2.0);
	auto c = a + b;
	auto d = a * b + pow3(b);
	c += c + Value(1.0);
	c += Value(1.0) + c - a;
	d += d * Value(2.0) + relu(b + a);
	d += Value(3.0) * d + relu(b - a);

	auto e = c - d;
	auto f = sqr(e);

	auto g = f / Value(2.0);
	g += Value(10.0) / f;
	backward(g);

	burt::HighPrecisionTimer timer_main;
	std::ostringstream fname;

	for (size_t i = 0; i < kIterations; ++i)
	{
		std::ostringstream fname;
		fname << "my_file_l" << i << ".bin";
		loadComputeGraphContextFromFile<true/*load_vaues*/, false/*load_grads*/, decltype(g)>(fname.str().c_str());
		fname.seekp(0);
	}
	double deltaSec = timer_main.getTimeSec();
	std::cout << std::fixed << std::setprecision(DBL_DECIMAL_DIG) << "without pre-allocated buffers: " << deltaSec << '\n';

	return 0;
}

int main(int argc, char** argv)
{
//	main_backprop_almost(argc, argv);
//	return 0;

#if 0
    int main_test_scalar(int argc, char** argv);
    main_test_scalar(argc, argv);

	int main_test_simd(int argc, char** argv);
    main_test_simd(argc, argv);
#endif

    main_benchmark<double>(argc, argv, ">>tiny benchmark with scalar fp64\n");
    //main_full_benchmark<double, false/*save_graph_to_dot_file*/> (argc, argv, ">>small benchmark full with scalar fp64\n");
	//save_benchmark<double, true, false, 5000>(argc, argv, ">>save_benchmark");
	//load_benchmark<double, true, false, 5000>(argc, argv, ">>load_benchmark");

	//std::cout << ">> vector size in items in fp64: " << burt::VectorSimdTraits<double, burt::cpu_extension>::VecType::size() << '\n';
	//main_benchmark<burt::VectorSimdTraits<double, burt::cpu_extension>::VecType>(argc, argv, ">>benchmark with vectorized fp64\n");
	//std::cout << "=================================\n";

	//save_benchmark_light<double, true, false, 5000>(argc, argv, "save_benchmark_light");
	//load_benchmark_light<double, true, false, 5000>(argc, argv, "load_benchmark_light");

	//std::cout << ">> vector size in items in fp64: " << burt::VectorSimdTraits<double, burt::cpu_extension>::VecType::size() << '\n';
	//main_full_benchmark<burt::VectorSimdTraits<double, burt::cpu_extension>::VecType>(argc, argv, ">>benchmark with vectorized fp64\n");

	//std::cout << ">> vector size in items in fp32: " << burt::VectorSimdTraits<float, burt::cpu_extension>::VecType::size() << '\n';
	//main_benchmark<burt::VectorSimdTraits<float, burt::cpu_extension>::VecType>(argc, argv, ">>benchmark with vectorized fp32\n");

    if (1)
    {
        Value<double>::cleanFull();
        Value<float>::cleanFull();
        typedef burt::VectorSimdTraits<double, burt::cpu_extension>::VecType VecTypeFP64;
        typedef burt::VectorSimdTraits<float, burt::cpu_extension>::VecType VecTypeFP32;
        Value<VecTypeFP64>::cleanFull();
        Value<VecTypeFP32>::cleanFull();
    }
	//getchar();
    return 0;
}

int main_test_scalar(int argc, char** argv)
{
	// TEST ARRAYS
	{
		SpecialArray<uint32_t> array;
		burt_assert(array.size() == 0);
		burt_assert(array.push_back(12));
		burt_assert(array.isTinyArray());
		burt_assert(array.size() == 1);
		burt_assert(array[0] == 12);
		burt_assert(array.push_back(16));
		burt_assert(array[0] == 12);
		burt_assert(array[1] == 16);
		burt_assert(array.isTinyArray());
		burt_assert(array.push_back_two_items(17, 18) == true);
		burt_assert(array[0] == 12);
		burt_assert(array[1] == 16);
		burt_assert(array[2] == 17);
		burt_assert(array[3] == 18);
		burt_assert(array.isLongArray());
		burt_assert(array.size() == 4);
        array.sysClearWithErase();
		burt_assert(array.size() == 0);

		array.sysArrayResizeLossyWithoutAnyInit(5);
		burt_assert(array.isLongArray());

		burt_assert(array.size() == 5);
		burt_assert(array.push_back(16));
		burt_assert(array.size() == 6);
	}

	// TEST EXPRESSIONS INPLACE
	{
		Value a = Value(2.0);
		Value b = Value(3.0);
		Value c = Value(70.0);

		std::vector<decltype(a)> abc_vec = { a, b, c };

		Value abc_mean = reduceMean(abc_vec.data(), abc_vec.size());
        burt_assert(fabs(abc_mean.dataCopy() - 25) < 1e-10);

		Value abc_sum = reduceSum(abc_vec.data(), abc_vec.size());
        burt_assert(fabs(abc_sum.dataCopy() - 75) < 1e-10);

		Value abc_var = variance (abc_vec.data(), abc_vec.size());
        burt_assert(fabs(abc_var.dataCopy() - 1519) < 1e-10);

		Value abc_var_biased = varianceBiased (abc_vec.data(), abc_vec.size());
        burt_assert(fabs(abc_var_biased.dataCopy() - ( ((2.-25.)*(2.-25.) + (3.-25.)* (3.-25.) + (70.-25.)*(70.-25.))/3.0) ) < 1e-10);
	}

	{
		Value a = Value(2.0);
		Value b = Value(3.0);
		Value c = Value(5.0);

		Value d_mult = Value(10.0);
		d_mult *= a;
        burt_assert(fabs(d_mult.dataCopy() - 20.0) < 1e-10);
		d_mult *= b;
        burt_assert(fabs(d_mult.dataCopy() - 60.0) < 1e-10);
		d_mult *= c;
        burt_assert(fabs(d_mult.dataCopy() - 300.0) < 1e-10);
		d_mult *= c;
        burt_assert(fabs(d_mult.dataCopy() - 1500.0) < 1e-10);

		Value ab_avg = mean(a, b);
        burt_assert(fabs(ab_avg.dataCopy() - 5.0/2) < 1e-10);

		Value<double> arr[3] = {a, b, c};
		Value abc_avg = reduceMean(arr, 3);
        burt_assert(fabs(abc_avg.dataCopy() - 10.0 / 3) < 1e-10);

		Value ab_avg_neg = negativeMean(a, b);
        burt_assert(fabs(ab_avg_neg.dataCopy() - (-5.0/2) ) < 1e-10);

		Value<double> arr_neg[3] = { a, b, c };
		Value abc_avg_neg = reduceNegativeMean(arr_neg, 3);
        burt_assert(fabs(abc_avg_neg.dataCopy() - (-10.0/3)) < 1e-10);
	}

	// TEST EXPRESSIONS INPLACE
	{
		Value a = Value(2.0);
		Value b = Value(3.0);
		Value c = Value(5.0);

		Value d_add = Value(10.0);
		d_add += a;
        burt_assert(fabs(d_add.dataCopy() - 12.0) < 1e-10);
		d_add += b;
        burt_assert(fabs(d_add.dataCopy() - 15.0) < 1e-10);
		d_add += c;
        burt_assert(fabs(d_add.dataCopy() - 20.0) < 1e-10);
		d_add += c;
        burt_assert(fabs(d_add.dataCopy() - 25.0) < 1e-10);
		d_add *= a;
        burt_assert(fabs(d_add.dataCopy() - 50.0) < 1e-10);
		d_add /= a;
        burt_assert(fabs(d_add.dataCopy() - 25.0) < 1e-10);
	}

	// TEST EXPRESSIONS INPLACE
	{
		Value a = Value(2.0);
		Value b = Value(3.0);
		Value c = Value(5.0);

		Value d_sub = Value(10.0);
		d_sub -= a;
        burt_assert(fabs(d_sub.dataCopy() - 8.0) < 1e-10);
		d_sub -= b;
        burt_assert(fabs(d_sub.dataCopy() - 5.0) < 1e-10);
		d_sub -= c;
        burt_assert(fabs(d_sub.dataCopy() - 0.0) < 1e-10);
		d_sub -= c;
        burt_assert(fabs(d_sub.dataCopy() - (-5.0)) < 1e-10);
		d_sub *= a;
        burt_assert(fabs(d_sub.dataCopy() - (-10.0)) < 1e-10);
	}

	{
		Value b = Value(3.0);
		backward(b);
        burt_assert(fabs(b.dataCopy() - (3.0)) < 1e-10);
        burt_assert(fabs(b.gradCopy() - (1.0)) < 1e-10);
	}

	{
		Value b = Value(3.0);

		Value x1 = Value(-41.0);
		Value x2 = Value(2.5);

		Value w1 = Value(5.0);
		Value w2 = Value(4.0);

		Value in = innerProduct({ x1, x2 }, {w1, w2});
        burt_assert(fabs(in.dataCopy() - (-41.0 * 5.0 + 2.5 * 4.0)) < 1e-10);

		Value in_with_bias = innerProductWithBias(b, { x1, x2 }, { w1, w2 });
        burt_assert(fabs(in_with_bias.dataCopy() - (3.0 + -41.0 * 5.0 + 2.5 * 4.0)) < 1e-10);
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
        burt_assert(fabs(g.dataCopy() - 612.50) < 1e-10);
        burt_assert(fabs(a.gradCopy() - (-35.0)) < 1e-10);
        burt_assert(fabs(b.gradCopy() - 1050.0) < 1e-10);
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

		Value g = mul<OpHint::eOpHintNotEvaluateValue> (f, const_0_5);

		backward(g);
        burt_assert(fabs(g.dataCopy() - 0.0) < 1e-10);
        burt_assert(fabs(a.gradCopy() - (-35.0)) < 1e-10);
        burt_assert(fabs(b.gradCopy() - 1050.0) < 1e-10);
	}

	// TEST VALUES
	{
		auto c1 = Value<double>::numActiveNodes();
		Value a = Value(-41.0);
		auto c2 = Value<double>::numActiveNodes();
		Value b = a;
		auto c3 = Value<double>::numActiveNodes();
        burt_assert(c1 + 1 == c2);
        burt_assert(c3 == c2);
	}

	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a + b;
        burt_assert(fabs(c.dataCopy() - (-41.0 + 2.0)) < 1e-6);
	}

	{
		Value a = Value(16.0);
		Value csqrt = sqrt(a);
		burt_assert(fabs(csqrt.dataCopy() - (4.0)) < 1e-6);
		backward(csqrt);

		{
			double dh = 1e-3;
			Value a_ = Value(16.0 + dh);
			Value csqrt_ = sqrt(a_);
			double grad_num = (csqrt_.dataCopy() - csqrt.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}

	{
		Value a = Value(1/16.0);
		Value icsqrt = invSqrt(a);
		burt_assert(fabs(icsqrt.dataCopy() - (4.0)) < 1e-6);
		backward(icsqrt);

		{
			double dh = 1e-6;
			Value a_ = Value(1/16.0 + dh);
			Value icsqrt_ = invSqrt(a_);
			double grad_num = (icsqrt_.dataCopy() - icsqrt.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
			burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
		}
	}


	{
		Value a_p_one = Value(1.0);
		Value a_n_one = Value(-1.0);

		Value s_p_one = sigmoid(a_p_one);
		Value s_n_one = sigmoid(a_n_one);
        burt_assert(fabs(s_p_one.dataCopy() - (0.7310585786300049)) < 1e-6);
        burt_assert(fabs(s_n_one.dataCopy() - (0.2689414213699951)) < 1e-6);

		Value r_p_one = relu(a_p_one);
		Value r_n_one = relu(a_n_one);
        burt_assert(fabs(r_p_one.dataCopy() - (1.0)) < 1e-6);
        burt_assert(fabs(r_n_one.dataCopy() - (0.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a / b;
        burt_assert(fabs(c.dataCopy() - (-41.0 / 2.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a - b;
        burt_assert(fabs(c.dataCopy() - (-41.0 - 2.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = Value(2.0);
		Value c = a * b;
        burt_assert(fabs(c.dataCopy() - (-41.0 * 2.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = exp(a);
        burt_assert( fabs(b.dataCopy() - exp(-41.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = inv(a);
        burt_assert(fabs(b.dataCopy() - 1.0/(-41.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = sqr(a);
        burt_assert(fabs(b.dataCopy() -(41.0 * 41.0)) < 1e-6);
	}
	{
		Value a = Value(-41.0);
		Value b = pow3(a);
        burt_assert(fabs(b.dataCopy() - (-41.0 * 41.0 * 41.0)) < 1e-6);
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
		
		Value<double> a_arr[8] = {a1, a2, a3, a4, a5, a6, a7, a8};
		double a_sum = (-41.0) + (-42.0) + (+11.0) + (-43.0) + (-44.5) + (-1.0) + (+2.0) + (+3.5);
		Value b = reduceSum(a_arr, sizeof(a_arr)/sizeof(a_arr[0]));
        burt_assert(fabs(b.dataCopy() - a_sum) < 1e-6);

//		Value e = reduceSumExp(a_arr, sizeof(a_arr) / sizeof(a_arr[0]));
//		double e_sum = exp(-41.0) + exp(-42.0) + exp(+11.0) + exp(-43.0) + exp(-44.5) + exp(-1.0) + exp(+2.0) + exp(+3.5);
//		burt_assert(fabs(e.dataCopy() - e_sum) < 1e-6);
	}

	{
		Value a = Value(11.0);
		{
			Value b = negativeLog(a);
            burt_assert(fabs(b.dataCopy() - (-log(11.0))) < 1e-6);
		}
		{
			Value b = logarithm(a);
            burt_assert(fabs(b.dataCopy() - (log(11.0))) < 1e-6);
		}
	}

	// TEST GRADS
	double dh = 1e-6;
    burt_assert(dh > DBL_EPSILON);

	// TEST EXPRESSIONS INPLACE
	{
		Value a = Value(2.0);
		Value c = Value(10.0);
		c *= a + Value(1.0);
        burt_assert(fabs(c.dataCopy() - 10.0 * 3.0) < 1e-10);
		backward(c);
		{
			Value a_ = Value(2.0 + dh);
			Value c_ = Value(10.0);
			c_ *= a_ + Value(1.0);
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = a.gradCopy();
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
		}
		{
			Value a_ = Value(-41.0);
			Value b_ = Value(2.0 + dh);
			Value c_ = a_ + b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = b.gradCopy();
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
		}
		{
			Value a_ = Value(-41.0);
			Value b_ = Value(2.0 + dh);
			Value c_ = a_ - b_;
			double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
			double grad_adiff = b.gradCopy();
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(grad_num - grad_bdiff) < 1e-3);
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
		
		for (size_t i = 0; i < sizeof(a_arr)/sizeof(a_arr[0]); ++i)
		{
            burt_assert(fabs(a_arr[i].gradRef() - 1.0) <= 1e-6);
            burt_assert(fabs(a_arr[i].gradCopy() - 1.0) <= 1e-6);
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

		double a_raw_data[8] = { -41.0, -42.0, +11.0, -43.0, -44.5, -1.0, +2.0, +3.5};
		Value<double> a_arr[8] = { a1, a2, a3, a4, a5, a6, a7, a8 };
		
		/*
		Value b = reduceSumExp(a_arr, sizeof(a_arr) / sizeof(a_arr[0]));
		backward(b);

		for (size_t i = 0; i < sizeof(a_arr) / sizeof(a_arr[0]); ++i)
		{
            burt_assert(fabs(a_arr[i].gradCopy() - exp(a_raw_data[i])) <= 1e-6);
		}
		*/
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
            burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
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
            burt_assert(fabs(x1.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5 + dh);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0);
			Value in_ = innerProduct({ x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
            burt_assert(fabs(x2.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0 + dh);
			Value w2_ = Value(4.0);
			Value in_ = innerProduct({ x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
            burt_assert(fabs(w1.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0 + dh);
			Value in_ = innerProduct({ x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
            burt_assert(fabs(w2.gradCopy() - grad_num) < 1e-3);
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
            burt_assert(fabs(bias.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value bias_ = Value(3.5);
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5 + dh);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0);
			Value in_ = innerProductWithBias(bias_, { x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
            burt_assert(fabs(x2.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value bias_ = Value(3.5);
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0 + dh);
			Value w2_ = Value(4.0);
			Value in_ = innerProductWithBias(bias_, { x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
            burt_assert(fabs(w1.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value bias_ = Value(3.5);
			Value x1_ = Value(-41.0);
			Value x2_ = Value(2.5);
			Value w1_ = Value(5.0);
			Value w2_ = Value(4.0 + dh);
			Value in_ = innerProductWithBias(bias_, { x1_, x2_ }, { w1_, w2_ });
			double grad_num = (in_.dataCopy() - in.dataCopy()) / dh;
            burt_assert(fabs(w2.gradCopy() - grad_num) < 1e-3);
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
        burt_assert(fabs(x.gradCopy() - (2.0*-4.0)) < 1e-3);
		cleanGrad(v);
        burt_assert(fabs(x.gradCopy() - (0.0)) < 1e-3);
		backward(v);
        burt_assert(fabs(x.gradCopy() - (2.0 * -4.0)) < 1e-3);
		cleanGradForNonLeafNodes(v);
        burt_assert(fabs(x.gradCopy() - (2.0 * -4.0)) < 1e-3);
        burt_assert(fabs(v.gradCopy() - (0.0)) < 1e-3);
        burt_assert(fabs(z.gradCopy() - (0.0)) < 1e-3);
        burt_assert(fabs(x2.gradCopy() - (0.0)) < 1e-3);
        burt_assert(fabs(y2.gradCopy() - (0.0)) < 1e-3);

		backward(v);
        burt_assert(fabs(x.gradCopy() - 2*(2.0 * -4.0)) < 1e-3);
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
            burt_assert(fabs(a.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value a = Value(-1.0);
			Value s = sigmoid(a);
			backward(s);

			Value a_ = Value(-1.0 + dh);
			Value s_ = sigmoid(a_);
			double grad_num = (s_.dataCopy() - s.dataCopy()) / dh;
            burt_assert(fabs(a.gradCopy() - grad_num) < 1e-3);
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
            burt_assert(fabs(a.gradCopy() - grad_num) < 1e-3);
		}
		{
			Value a = Value(-1.0);
			Value r = relu(a);
			backward(r);

			Value a_ = Value(-1.0 + dh);
			Value r_ = relu(a_);
			double grad_num = (r_.dataCopy() - r.dataCopy()) / dh;
            burt_assert(fabs(a.gradCopy() - grad_num) < 1e-3);
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

        burt_assert(fabs(c.dataCopy() - (11.0 * 11.0 + 23.0 * 23)) < 1e-3);

		backward(c);
		double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
		double grad_adiff = a.gradCopy();
        burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
		
		Value<double> in[2] = { a, b };
		Value c_var = reduceSumOfSquares(in, 2);
        burt_assert(fabs(c_var.dataCopy() - (11.0 * 11.0 + 23.0 * 23)) < 1e-3);

		cleanGrad(c_var);
		backward(c_var);
		grad_adiff = a.gradCopy();
        burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
	}

	{
		double dh = 1e-6;
		Value a = Value(11.0);
		Value b = Value(23.0);
		Value c = meanSquares(a, b);

		Value a_ = Value(11.0 + dh);
		Value b_ = Value(23.0);
		Value c_ = meanSquares(a_, b_);

        burt_assert(fabs(c.dataCopy() - (11.0 * 11.0/2.0 + 23.0 * 23/2.0)) < 1e-3);

		backward(c);
		double grad_num = (c_.dataCopy() - c.dataCopy()) / dh;
		double grad_adiff = a.gradCopy();
        burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
		backward(c);
		grad_adiff = a.gradCopy() / 2.0;
        burt_assert(fabs(grad_num - grad_adiff) < 1e-3);

		Value<double> in[2] = { a, b };
		Value c_var = reduceMeanSquares(in, 2);
        burt_assert(fabs(c_var.dataCopy() - (11.0 * 11.0 / 2.0 + 23.0 * 23 / 2.0)) < 1e-3);

		cleanGrad(c_var);
		backward(c_var);
		grad_adiff = a.gradCopy();
        burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
	}
	std::cout << "[pass tests]\n";

	return 0;
}

int main_test_simd(int argc, char** argv)
{
	using TElementType = double;
	typedef burt::VectorSimdTraits<TElementType, burt::cpu_extension>::VecType VecType;
	

//	VecType a(1,2,3,4);
//	a.cutoff(3); // set last element to 0
//	auto zz_1 = a[1]; // 2
//	auto zz_2 = a.extract(1); // 2

	{
		Value<VecType> a = Value(VecType(1.0));
        burt_assert(fabs(::horizontal_add(a.dataCopy() - VecType(1.0))) < 1e-6);

		Value<VecType> b = Value(VecType(2.0));
        burt_assert(fabs(::horizontal_add(b.dataCopy() - VecType(2.0))) < 1e-6);

        burt_assert(Value<VecType>::numActiveNodes() == 2);
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
            burt_assert(fabs(s_p_one.dataCopy()[i] - (0.7310585786300049)) < 1e-6);
            burt_assert(fabs(s_n_one.dataCopy()[i] - (0.2689414213699951)) < 1e-6);
            burt_assert(fabs(r_p_one.dataCopy()[i] - (1.0)) < 1e-6);
            burt_assert(fabs(r_n_one.dataCopy()[i] - (0.0)) < 1e-6);
		}
	}

	{
		Value<VecType> a = Value(VecType(3.0));
		Value<VecType> b = Value(VecType(2.0));
		Value<VecType> c = a - b;
		
		for (size_t i = 0; i < c.dataRef().size(); ++i)
		{
            burt_assert(fabs(c.dataRef()[i] - 1.0) < 1e-6);
            burt_assert(fabs(a.dataRef()[i] - 3.0) < 1e-6);
            burt_assert(fabs(b.dataRef()[i] - 2.0) < 1e-6);

            burt_assert(fabs(c.dataCopy()[i] - 1.0) < 1e-6);
            burt_assert(fabs(a.dataCopy()[i] - 3.0) < 1e-6);
            burt_assert(fabs(b.dataCopy()[i] - 2.0) < 1e-6);
		}
	}

	{
		Value<VecType> a = Value(VecType(3.0));
		Value<VecType> b = Value(VecType(2.0));
		
		Value<VecType> c_mult = a * b;
		for (size_t i = 0; i < a.dataRef().size(); ++i) {
            burt_assert(fabs(c_mult.dataRef()[i] - 3.0 * 2.0) < 1e-6);
		}
		Value<VecType> c_div = a / b;
		for (size_t i = 0; i < a.dataRef().size(); ++i) {
            burt_assert(fabs(c_div.dataRef()[i] - 3.0 / 2.0) < 1e-6);
		}
	}

	{
		Value<VecType> b = Value(VecType(3.0));
		backward(b);
		for (size_t i = 0; i < b.dataRef().size(); ++i) {
            burt_assert(fabs(b.dataCopy()[i] - (3.0)) < 1e-10);
            burt_assert(fabs(b.gradCopy()[i] - (1.0)) < 1e-10);
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
        burt_assert(fabs(g.dataCopy()[0] - 612.50) < 1e-10);
        burt_assert(fabs(a.gradCopy()[0] - (-35.0)) < 1e-10);
        burt_assert(fabs(b.gradCopy()[0] - 1050.0) < 1e-10);
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
                burt_assert(fabs(grad_num[ii] - grad_bdiff[ii]) < 1e-3);
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
                burt_assert(fabs(a_arr[i].gradRef()[ii] - 1.0) <= 1e-6);
                burt_assert(fabs(a_arr[i].gradCopy()[ii] - 1.0) <= 1e-6);
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
                burt_assert(fabs(grad_num - grad_adiff) < 1e-3);
			}
		}
	}

	return 0;
}

