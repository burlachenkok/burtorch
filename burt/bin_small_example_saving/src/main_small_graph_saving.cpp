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

#include "burt/timers/include/HighPrecisionTimer.h"

template class Value<double>;
template class Value<float>;
template class SpecialArray<int, size_t>;


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

	for (size_t i = 0; i < kIterations; ++i)
	{
		bool saved = saveToFile<true/*save_vaues*/, false/*save_gradients*/>({a,b,c,d,e,f,g}, "my_file_f.bin");
		assert(saved);
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
		bool saved = saveComputeGraphContextToFile<true/*save_vaues*/, false/*save_gradients*/, decltype(g)>(fname.str().c_str());
		assert(saved);
		fname.seekp(0);
	}
	double deltaSec = timer_main.getTimeSec();
	std::cout << std::fixed << std::setprecision(DBL_DECIMAL_DIG) << "without pre-allocated buffers: " << deltaSec << '\n';

	return 0;
}

int main(int argc, char** argv)
{
	save_benchmark<double, true, false, 5000>(argc, argv, ">>save_benchmark");
	//save_benchmark_light<double, true, false, 5000>(argc, argv, "save_benchmark_light");

    Value<double>::cleanFull();
	Value<float>::cleanFull();

	//getchar();
    return 0;
}
