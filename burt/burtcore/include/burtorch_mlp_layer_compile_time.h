#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burtcore/include/burtorch_mlp_neuron_compile_time.h"
#include <vector>
#include <array>
#include <limits>
#include <stddef.h>
#include <math.h>

/**
 * @brief Represents a layer of neurons at compile-time with specified configuration.
 *
 * The `LayerAtCompileTime` structure defines a layer of neurons with compile-time parameters
 * such as data type, activation function, input and output sizes, and initialization type.
 * It provides methods for layer initialization, forward pass computation, and statistics collection.
 *
 * @tparam DataType The data type of the neuron values (e.g., float, double).
 * @tparam bias Whether the layer includes a bias term (true/false).
 * @tparam actType The activation type used in the neurons (e.g., ReLU, Sigmoid).
 * @tparam nin The number of inputs to the layer.
 * @tparam nout The number of outputs (neurons) in the layer.
 * @tparam init_type The initialization type for the neurons.
 */
template<class DataType,
    bool bias,
    ActivationType actType,
    size_t nin,
    size_t nout,
    NeuronInitType init_type
>
struct LayerAtCompileTime
{
    using TDataType = DataType; /**< Alias for the data type. */
    using Scalar = Value<DataType>; /**< Alias for the scalar value type used in computations. */
    using NeuronType = NeuronAtCompileTime<DataType, bias, actType, nin, init_type>; /**< Alias for the neuron type. */

    /**
     * @brief Checks if the layer includes a bias term.
     * @return True if the layer has a bias, otherwise false.
     */
    consteval bool hasBias() noexcept {
        return bias;
    }

    /**
     * @brief Returns the activation type used in the layer.
     * @return The activation type.
     */
    consteval ActivationType activationType() noexcept {
        return actType;
    }

    /**
     * @brief Returns the number of inputs to the layer.
     * @return The number of inputs (fan-in).
     */
    constexpr size_t fanin() noexcept {
        return neurons[0].fanin();
    }

    /**
     * @brief Returns the number of outputs from the layer.
     * @return The number of outputs (fan-out).
     */
    constexpr size_t fanout() noexcept {
        return neurons.size();
    }

    /**
     * @brief Computes the standard deviation multiplier for layer initialization based on the activation function.
     *
     * This follows the initialization scheme proposed by He et al. (2015) for layers using ReLU activation.
     * The function adjusts the scaling based on the activation type.
     *
     * @tparam TFloatType The floating-point type used for the scaling multiplier (default: double).
     * @tparam act_type The activation type (default: ReLU).
     * @return The scaling factor for initialization.
     */
    template<class TFloatType = double, ActivationType act_type = ActivationType::eRelu>
    TFloatType std_for_initialization()
    {
        double gain = 1.0;

        switch (act_type)
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
            burt_unreahable();
            break;
        }

        return TFloatType(gain / sqrt(double(fanin())));
    }

    /**
     * @brief Default constructor for the layer.
     * Initializes the neurons in the layer.
     */
    LayerAtCompileTime() noexcept
    {
        for (size_t i = 0; i < nout; ++i)
        {
            neurons[i] = NeuronType();
        }
    }

    /**
     * @brief Copy constructor for the layer.
     * @param rhs The layer to copy.
     */
    LayerAtCompileTime(const LayerAtCompileTime& rhs) noexcept
        : neurons(rhs.neurons)
    {}

    /**
     * @brief Returns the parameters of the layer.
     * @return A vector containing all parameters (weights and biases) of the neurons in the layer.
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

    struct Statistics
    {
        TDataType min_grad_value;
        TDataType max_grad_value;
        TDataType mx_grad_value;
        TDataType dx_grad_value;

        TDataType min_param_value;
        TDataType max_param_value;
        TDataType mx_param_value;
        TDataType dx_param_value;

        size_t number_of_params;
    };


    /**
     * @brief Computes statistics for the layer's parameters and gradients.
     * @return A Statistics struct containing min, max, mean, and variance of parameters and gradients.
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
     * @brief Destructor for the layer.
     */
    ~LayerAtCompileTime() noexcept = default;

    /**
     * @brief Performs a forward pass through the layer for a given input vector.
     * @param x The input vector.
     * @return The output vector after applying the neurons' forward pass.
     */
    std::vector<Scalar> forward(const std::vector<Scalar>& x) noexcept
    {
        std::vector<Scalar> res;
        res.reserve(neurons.size());
        for (size_t i = 0; i < neurons.size(); ++i)
            res.emplace_back(neurons[i].forward(x));
        return res;
    }

    /**
     * @brief Performs a forward pass through the layer and stores the result in the provided output vector.
     * @param result The vector where the output will be stored.
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
     * @brief Performs a forward pass through the layer and returns the maximum value encountered.
     * @param result The vector where the output will be stored.
     * @param x The input vector.
     * @return The maximum value encountered during the forward pass.
     */
    DataType forwardAndReportMax(std::vector<Scalar>& result, const std::vector<Scalar>& x) noexcept
    {
        DataType max_value = std::numeric_limits<DataType>::min();

        if (result.size() != neurons.size()) [[unlikely]]
            result.resize(neurons.size());

            for (size_t i = 0; i < neurons.size(); ++i)
            {
                result[i] = std::move(neurons[i].forward(x));

                auto to_cmp = result[i].dataCopy();
                if (to_cmp > max_value)
                    max_value = to_cmp;
            }

            return max_value;
    }

    /**
     * @brief Template method for performing a forward pass using an input array of a fixed size.
     * @tparam N The size of the input array.
     * @param x The input array.
     * @return The output vector after applying the neurons' forward pass.
     */
    template <size_t N>
    std::vector<Scalar> forward(const std::array<Scalar, N>& x) noexcept
    {
        std::vector<Scalar> res;
        res.reserve(neurons.size());
        for (size_t i = 0; i < neurons.size(); ++i)
            res.emplace_back(neurons[i].forward(x));
        return res;
    }

    /**
     * @brief Template method for performing a forward pass using an input array of a fixed size and storing the result.
     * @tparam N The size of the input array.
     * @param result The vector where the output will be stored.
     * @param x The input array.
     */
    template <size_t N>
    void forward(std::vector<Scalar>& result, const std::array<Scalar, N>& x) noexcept
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
     * @brief Template method for performing a forward pass using an input array and returning the maximum value encountered.
     * @tparam N The size of the input array.
     * @param result The vector where the output will be stored.
     * @param x The input array.
     * @return The maximum value encountered during the forward pass.
     */
    template <size_t N>
    typename Scalar::TDataType forwardAndReportMax(std::vector<Scalar>& result, const std::array<Scalar, N>& x) noexcept
    {
        DataType max_value = std::numeric_limits<DataType>::min();

        if (result.size() != neurons.size()) [[unlikely]]
            result.resize(neurons.size());

            for (size_t i = 0; i < neurons.size(); ++i)
            {
                result[i] = std::move(neurons[i].forward(x));
                auto to_cmp = result[i].dataCopy();
                if (to_cmp > max_value)
                    max_value = to_cmp;
            }

            return max_value;
    }

    std::array<NeuronType, nout> neurons; /**< Array of neurons in the layer. */
};
