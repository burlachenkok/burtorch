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

template class Value<double>;
template class Value<float>;
template class SpecialArray<int, size_t>;


template<class TElementType, bool save_graph_to_dot_file>
int main_full_benchmark(int argc, char** argv, const char* test_name)
{
	std::cout << "================================\n";
	std::cout << test_name << '\n';

	// deactivate unused
	Value<TElementType>::deactiveUnusedNodes();
    burt_assert(Value<TElementType>::numActiveNodes() == 0);
	auto chk = Value<TElementType>::checkpointForNeurons();

	constexpr size_t kIterations = 20 * 1000;
	std::cout << "iterations: " << kIterations << '\n';

	// run without pre-allocated buffers
	{
		burt::HighPrecisionTimer timer_main;
		for (size_t i = 0; ; ++i)
		{
			{
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
		}
		double deltaSec = timer_main.getTimeSec();
		std::cout << std::fixed << std::setprecision(DBL_DECIMAL_DIG) << "without pre-allocated buffers: " << deltaSec << '\n';
	}

	return 0;
}

int main(int argc, char** argv)
{    
    main_full_benchmark<double, false/*save_graph_to_dot_file*/> (argc, argv, ">>small benchmark full with scalar fp64\n");

	Value<double>::cleanFull();
	Value<float>::cleanFull();
	typedef burt::VectorSimdTraits<double, burt::cpu_extension>::VecType VecTypeFP64;
	typedef burt::VectorSimdTraits<float, burt::cpu_extension>::VecType VecTypeFP32;
	Value<VecTypeFP64>::cleanFull();
	Value<VecTypeFP32>::cleanFull();

	//getchar();
    return 0;
}
