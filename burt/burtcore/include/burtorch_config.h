#pragma once


/**
* Auxiliry meta-information
*/

#define BURTORCH_NODES_LABEL_SUPPORT   0           ///< Support labels for nodes.This allows users to associate labels with nodes for easier tracking and debugging.

/**
* These settings control the type of index used to reference nodes in the graph.
*
* @note Only one of the following index settings should be enabled at a time.
*/

#define BURTORCH_NODES_UINT8_INDICIES  0

#define BURTORCH_NODES_UINT16_INDICIES 0

#define BURTORCH_NODES_UINT32_INDICIES 1

#define BURTORCH_NODES_UINT64_INDICIES 0

#define BURTORCH_USE_GC_COUNTERS 1                 ///< Enable or disable garbage collection (GC) counters for node memory management.

#define BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META 1  ///< If set to 1, the high bit of the node index will be used to store metadata (inside containers). This can be useful for encoding additional information within the index itself

#define BURTORCH_INIT_GRADS_TO_ZERO 1              ///< If set to 1, all gradients will be initialized to zero when a node node is created. This prevents of accidental usage of uninitialized gradients at price of intialization time.

#define BURTORCH_MAKE_COMPUTE_GRAPHS_PER_THREAD 0  ///< If set to 1 then BurTorch will be thread safe

#define BURTORCH_ALLOW_SEVERAL_THREADS_WORK_ON_THE_SAME_GRAPH 0  ///< If set to 1 then if different threads construct different part of the graph it's fine. Still thread-safe.

//===================================================================================================================================================//
static_assert(BURTORCH_NODES_UINT8_INDICIES + BURTORCH_NODES_UINT16_INDICIES + BURTORCH_NODES_UINT32_INDICIES + BURTORCH_NODES_UINT64_INDICIES == 1);

#if BURTORCH_MAKE_COMPUTE_GRAPHS_PER_THREAD
	#define BURTORCH_INTERNAL_STATIC_STORAGE thread_local
#else
	#define BURTORCH_INTERNAL_STATIC_STORAGE
#endif

#if BURTORCH_ALLOW_SEVERAL_THREADS_WORK_ON_THE_SAME_GRAPH
	#define BURTORCH_INDEX_WRAPPER(TYPE) std::atomic<TYPE>
	static_assert(BURTORCH_MAKE_COMPUTE_GRAPHS_PER_THREAD == 0, "In this regime threads share the same compute graph");
#else
	#define BURTORCH_INDEX_WRAPPER(TYPE) TYPE
	static_assert(BURTORCH_MAKE_COMPUTE_GRAPHS_PER_THREAD == 0 || BURTORCH_MAKE_COMPUTE_GRAPHS_PER_THREAD == 1, "In this regime threads may or may not share the same compute graph");
#endif
//===================================================================================================================================================//
