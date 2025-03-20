#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"

/**
 * @brief Structure for BurTorch statistics related to backpropagation and sorting.
 */
struct BurTorchCallbacks
{
public:
    /**
     * @brief Called when backpropagation starts.
     */
    void onStartBackpropagation() {}

    /**
     * @brief Called when topological sorting starts.
     */
    void onStartTopologicalSort() {}

    /**
     * @brief Called when backpropagation starts for each individual node.
     */
    void onStartBackpropagationPerNode() {}

    /**
     * @brief Called when backpropagation ends for each individual node.
     */
    void onEndBackpropagationPerNode() {}

    /**
     * @brief Called when a compute node is visited.
     */
    void onVisitingComputeNode() {}

    /**
     * @brief Called when backpropagation ends.
     */
    void onEndBackpropagation() {}
};
