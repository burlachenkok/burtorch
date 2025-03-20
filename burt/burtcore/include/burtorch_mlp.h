/**
 * @file MLP.h
 *
 * @brief A simple implementation of a Multi-Layer Perceptron (MLP).
 *
 * This header file defines a template struct for a Multi-Layer Perceptron (MLP) that supports a forward pass
 * and the retrieval of model parameters. The MLP layers are dynamically created based on the input and output sizes.
 *
 * @tparam DataType The data type used for the values in the model (e.g., float, double).
 */

#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"

#include "burtcore/include/burtorch_node.h"
#include "burtcore/include/burtorch_mlp_layer.h"

#include <vector>

/**
* A simple Multi-Layer Perceptron (MLP) structure.
*
* This structure defines a Multi-Layer Perceptron with a flexible number of layers and a customizable
* data type for the values in the network. The MLP has methods for performing a forward pass through
* the network and retrieving all parameters (weights and biases) of the network.
*
* @tparam DataType The data type used for values in the MLP, typically a numeric type (e.g., float or double).
*/
template<class DataType>
struct MLP
{
    /// @brief The data type used for the values in the MLP.
    using TDataType = DataType;

    /// @brief The scalar type used for each value in the model.
    using Scalar = Value<DataType>;

    /**
     *  Constructs an MLP with a given number of input features and output sizes for each layer.
     *
     * This constructor initializes the layers of the MLP based on the number of input features and the
     * output sizes specified for each layer.
     *
     * @param nin The number of input features for the first layer.
     * @param nout A vector specifying the number of output features for each layer.
     */
    MLP(size_t nin, std::vector<size_t> nout) noexcept
    {
        layers.reserve(nout.size());

        size_t nin_prev = nin;
        for (size_t i = 0; i < nout.size(); ++i)
        {
            layers.emplace_back(nin_prev, nout[i]);
            nin_prev = nout[i];
        }
    }

    /**
     * Performs a forward pass through the MLP.
     *
     * This method takes an input vector and propagates it through all layers of the MLP, returning the output
     * of the final layer.
     *
     * @param x The input vector to the MLP.
     *
     * @return A vector of Scalar values representing the output of the final layer.
     */
    std::vector<Scalar> forward(const std::vector<Scalar>& x) noexcept
    {
        std::vector<Scalar> in = x;
        for (size_t i = 0; i < layers.size(); ++i)
        {
            std::vector<Scalar> out = layers[i].forward(in);
            in = std::move(out);
        }
        return in;
    }

    /**
     * Retrieves the parameters (weights and biases) of all layers in the MLP.
     *
     * This method collects and returns all the parameters (weights and biases) from all layers of the MLP.
     *
     * @return A vector of Scalar values representing the parameters of all layers.
     */
    std::vector<Scalar> parameters() noexcept
    {
        std::vector<Scalar> res;

        for (size_t i = 0; i < layers.size(); ++i)
        {
            std::vector<Scalar> extra = layers[i].parameters();
            res.insert(res.end(), extra.begin(), extra.end());
        }
        return res;
    }

    std::vector<MLPLayer<DataType>> layers; /// A vector of layers that make up the MLP.
};
