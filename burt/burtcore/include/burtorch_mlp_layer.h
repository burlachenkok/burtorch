#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burtcore/include/burtorch_mlp_neuron.h"

#include <vector>
#include <limits>
#include <stddef.h>
#include <math.h>

template<class DataType,
    bool bias = true,
    ActivationType actType = ActivationType::eTanh>
/**
 * @brief A class representing a fully connected multi-layer perceptron (MLP) layer.
 *
 * This template class represents a single layer in a multi-layer perceptron, which can include
 * neurons with different activation functions and bias terms. It provides methods for forward
 * propagation and parameter access, as well as statistics on gradients and parameters.
 *
 * @tparam DataType The data type used for computations (e.g., float, double).
 * @tparam bias Whether to include a bias term for each neuron in the layer.
 * @tparam actType The activation function used in the layer (default is Tanh).
 */
struct MLPLayer
{
    using TDataType = DataType;   ///< Type alias for DataType
    using Scalar = Value<DataType>;   ///< Type alias for the scalar value used in the layer
    using NeuronType = Neuron<DataType, bias, actType>;   ///< Type alias for the neuron class

    /**
     * @brief Returns whether the layer has a bias term.
     *
     * @return True if the layer has a bias term, false otherwise.
     */
    consteval bool hasBias() noexcept {
        return bias;
    }

    /**
     * @brief Returns the activation type used in the layer.
     *
     * @return The activation type.
     */
    consteval ActivationType activationType() noexcept {
        return actType;
    }

    /**
     * @brief Returns the number of inputs (fanin) to the layer.
     *
     * @return The number of input connections to the layer.
     */
    constexpr size_t fanin() noexcept {
        return neurons[0].fanin();
    }

    /**
     * @brief Returns the number of outputs (fanout) from the layer.
     *
     * @return The number of output connections from the layer.
     */
    constexpr size_t fanout() noexcept {
        return neurons.size();
    }

    /**
     * @brief Computes the standard deviation scaling factor for layer initialization.
     *
     * This is based on the activation function used in the layer.
     *
     * @tparam TFloatType The type of the scaling factor (default is double).
     * @return The standard deviation scaling factor for initialization.
     */
    template<class TFloatType = double>
    TFloatType std_for_initialization()
    {
        double gain = 1.0;

        switch (actType)
        {
        case ActivationType::eIdent:
            gain = 1.0;
            break;
        case ActivationType::eSigmoid:
            gain = 1.0;
            break;
        case ActivationType::eTanh:
            gain = 5.0 / 3.0;
            break;
        case ActivationType::eRelu:
            gain = 1.4142135623730951; // sqrt(2.0)
            break;
        default:
            burt_unreahable(); // Error handling for unknown activation types
            break;
        }

        return TFloatType(gain / sqrt(double(fanin())));
    }

    /**
     * @brief Constructor that initializes the MLP layer with a specified number of inputs and outputs.
     *
     * @param nin The number of input neurons.
     * @param nout A vector specifying the number of neurons in each layer.
     * @param init_type The initialization method for neurons (default is uniform).
     */
    MLPLayer(size_t nin, size_t nout, NeuronInitType init_type = NeuronInitType::uniform_neg_one_plus_one) noexcept
    {
        neurons.reserve(nout);
        for (size_t i = 0; i < nout; ++i)
        {
            neurons.emplace_back(nin, init_type);
        }
    }

    /**
     * @brief Copy constructor for the MLP layer.
     *
     * @param rhs The MLP layer to copy.
     */
    MLPLayer(const MLPLayer& rhs) noexcept
        : neurons(rhs.neurons)
    {}

    /**
     * @brief Returns all parameters of the layer (weights and biases).
     *
     * @return A vector of Scalar values representing the parameters.
     */
    std::vector<Scalar> parameters() noexcept
    {
        std::vector<Scalar> res;

        for (size_t i = 0; i < neurons.size(); ++i)
        {
            std::vector<Scalar> extra = neurons[i].parameters();
            res.insert(res.end(), extra.begin(), extra.end());
        }
        return res;
    }

    /**
     * @brief A struct to hold statistical information about the layer's parameters and gradients.
     */
    struct Statistics
    {
        TDataType min_grad_value;   ///< Minimum gradient value
        TDataType max_grad_value;   ///< Maximum gradient value
        TDataType mx_grad_value;    ///< Mean gradient value
        TDataType dx_grad_value;    ///< Standard deviation of gradient values

        TDataType min_param_value;  ///< Minimum parameter value
        TDataType max_param_value;  ///< Maximum parameter value
        TDataType mx_param_value;   ///< Mean parameter value
        TDataType dx_param_value;   ///< Standard deviation of parameter values

        size_t number_of_params;    ///< Number of parameters in the layer
    };

    /**
     * @brief Computes and returns statistics on the layer's parameters and gradients.
     *
     * @return A Statistics struct containing the computed statistics.
     */
    Statistics statistics() noexcept
    {
        Statistics result;
        std::vector<Scalar> params = parameters();
        size_t n = params.size();

        result.min_grad_value = TDataType();
        result.max_grad_value = TDataType();
        result.mx_grad_value = TDataType();
        result.dx_grad_value = TDataType();

        result.min_param_value = TDataType();
        result.max_param_value = TDataType();
        result.mx_param_value = TDataType();
        result.dx_param_value = TDataType();

        result.number_of_params = n;

        if (n == 0)
            return result;

        result.min_grad_value = params[0].gradRef();
        result.max_grad_value = params[0].gradRef();
        result.min_param_value = params[0].dataRef();
        result.max_param_value = params[0].dataRef();

        for (size_t i = 0; i < n; ++i)
        {
            if (params[i].gradRef() > result.max_grad_value)
                result.max_grad_value = params[i].gradRef();
            if (params[i].gradRef() < result.min_grad_value)
                result.min_grad_value = params[i].gradRef();

            if (params[i].dataRef() > result.max_param_value)
                result.max_param_value = params[i].dataRef();
            if (params[i].dataRef() < result.min_param_value)
                result.min_param_value = params[i].dataRef();

            result.mx_grad_value += params[i].gradRef() / TDataType(n);
            result.mx_param_value += params[i].dataRef() / TDataType(n);
        }

        for (size_t i = 0; i < n; ++i)
        {
            result.dx_grad_value += (params[i].gradRef() - result.mx_grad_value) * (params[i].gradRef() - result.mx_grad_value) / TDataType(n);
            result.dx_param_value += (params[i].dataRef() - result.mx_param_value) * (params[i].dataRef() - result.mx_param_value) / TDataType(n);
        }

        return result;
    }

    /**
     * @brief Destructor for the MLP layer.
     */
    ~MLPLayer() noexcept = default;
    std::vector<Scalar> forward(const std::vector<Scalar>& x) noexcept
    {
        std::vector<Scalar> res;
        res.reserve(neurons.size());
        for (size_t i = 0; i < neurons.size(); ++i)
            res.emplace_back(neurons[i].forward(x));
        return res;
    }

    /**
    * Performs a forward pass through the layer with the input vector and stores the result in an existing vector memory for which was pre-allocated.
    *
    * @param result The output vector.
    * @param x The input vector.
    */
    void forward(std::vector<Scalar>& result, const std::vector<Scalar>& x) noexcept
    {
        if (result.size() != neurons.size()) [[unlikely]]
            result.resize(neurons.size());

            for (size_t i = 0; i < neurons.size(); ++i)
            {
                result[i] = std::move(neurons[i].forward(x));
            }

            return;
    }

    /**
    * Performs a forward pass through the layer and returns the maximum value of the output.
    *
    * @param result The output vector.
    * @param x The input vector.
    * @return The maximum value in the result vector.
    */
    DataType forwardAndReportMax(std::vector<Scalar>& result, const std::vector<Scalar>& x) noexcept
    {
        DataType max_value = std::numeric_limits<DataType>::min();

        if (result.size() != neurons.size()) [[unlikely]]
            result.resize(neurons.size());

            for (size_t i = 0; i < neurons.size(); ++i)
            {
                result[i] = std::move(neurons[i].forward(x));

                {
                    auto to_cmp = result[i].dataCopy();
                    if (to_cmp > max_value)
                        max_value = to_cmp;
                }
            }

            return max_value;
    }

    /**
    * Performs a forward pass through the layer using a fixed-size input array.
    *
    * @tparam N The size of the input array.
    * @param result The output vector.
    * @param x The input array.
    */
    template <size_t N>
    void forward(std::vector<Scalar>& result, const Scalar* x) noexcept
    {
        size_t neurons_num = neurons.size();
        if (result.size() != neurons_num) [[unlikely]]
            result.resize(neurons_num);

            Scalar* resultRaw = result.data();
            NeuronType* neuronsRaw = neurons.data();
            NeuronType* neuronsEnd = neuronsRaw + neurons_num;

            for (; neuronsRaw != neuronsEnd; ++neuronsRaw, ++resultRaw)
            {
                *resultRaw = std::move(neuronsRaw->template forward<N>(x));
            }

            return;
    }

    /**
    * Performs a forward pass through the layer using a fixed-size input array.
    *
    * @tparam N The size of the input array.
    * @param result The output vector.
    * @param x The input array.
    */
    template <size_t NItemsTotal, size_t NItemsPerArray>
    void forward(std::vector<Scalar>& result, const std::initializer_list<const Scalar*> x) noexcept
    {
        size_t neurons_num = neurons.size();
        if (result.size() != neurons_num) [[unlikely]]
            result.resize(neurons_num);

            Scalar* resultRaw = result.data();
            NeuronType* neuronsRaw = neurons.data();
            NeuronType* neuronsEnd = neuronsRaw + neurons_num;

            for (; neuronsRaw != neuronsEnd; ++neuronsRaw, ++resultRaw)
            {
                *resultRaw = std::move(neuronsRaw->template forward<NItemsTotal, NItemsPerArray>(x));
            }

            return;
    }

    /**
    * @brief Performs a forward pass through the layer and returns the maximum value of the output.
    *
    * @param result The output vector.
    * @param x The input vector.
    * @return The maximum value in the result vector.
    */
    template <size_t N>
    typename Scalar::TDataType forwardAndReportMax(std::vector<Scalar>& result, const Scalar* x) noexcept
    {
        DataType max_value = std::numeric_limits<DataType>::min();

        if (result.size() != neurons.size()) [[unlikely]]
            result.resize(neurons.size());

            for (size_t i = 0; i < neurons.size(); ++i)
            {
                result[i] = std::move(neurons[i].forward(x));
                {
                    auto to_cmp = result[i].dataCopy();
                    if (to_cmp > max_value)
                        max_value = to_cmp;
                }
            }

            return max_value;
    }

private:
    std::vector<NeuronType> neurons;   ///< A vector of neurons in the layer
};
