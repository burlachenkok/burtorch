#include "burtcore/include/burtorch.h"

#include "burt/fs/include/FileSystemHelpers.h"
#include "burt/linalg_vectors/include_internal/VectorSimdTraits.h"
#include "burt/timers/include/HighPrecisionTimer.h"

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
				backward(g);
			}
			Value<TElementType>::restoreCheckpoint(chk);
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
				Value g = f * const_0_5;
				backwardWithScratchStorage(g, res_seq_topo, res_set_topo, recursion);
			}
			Value<TElementType>::restoreCheckpoint(chk);
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
				backwardWithScratchStorage<decltype(a), false, true, false>(g, res_seq_topo, res_set_topo, recursion);
			}
			Value<TElementType>::restoreCheckpoint(chk);
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

		backward(g);

		std::string res = buildDotGraph(g, "manual-autograd");
		burt::FileSystemHelpers::saveFile("test.txt", res.data(), res.size());
		std::cout << "compute graph representation saved to: " << "test.txt" << "\n";
		std::cout << "================================\n\n";
	}

	return 0;
}

int main(int argc, char** argv)
{
    main_benchmark<double>(argc, argv, ">>tiny benchmark with scalar fp64\n");

	Value<double>::cleanFull();
	Value<float>::cleanFull();
	typedef burt::VectorSimdTraits<double, burt::cpu_extension>::VecType VecTypeFP64;
	typedef burt::VectorSimdTraits<float, burt::cpu_extension>::VecType VecTypeFP32;
	Value<VecTypeFP64>::cleanFull();
	Value<VecTypeFP32>::cleanFull();

	//getchar();
    return 0;
}
