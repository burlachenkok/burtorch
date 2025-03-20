#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/random/include/RandomVariable.h"
#include "burtcore/include/burtorch_node.h"
#include "burtcore/include/burtorch_mlp_neuron_enums.h"
#include <vector>
#include <array>
#include <assert.h>

/**
 * @brief A template for a neuron model at compile-time.
 *
 * This structure represents a single neuron with compile-time determined parameters such as the data type, activation function, input dimension, and initialization method.
 * It allows the configuration of a neuron with or without a bias term, a specific activation function (e.g., Tanh, Sigmoid), and a selected initialization type.
 * The forward pass computation is also done at compile-time for optimizations when possible.
 *
 * @tparam DataType The data type for weights and biases.
 * @tparam bias Boolean flag indicating whether the neuron has a bias term.
 * @tparam actType The activation function type used by the neuron.
 * @tparam input_dim The number of inputs (dimension) for the neuron.
 * @tparam init_type The initialization method for the neuron's weights and bias.
 */
template<class DataType,
    bool bias,
    ActivationType actType,
    size_t input_dim,
    NeuronInitType init_type>
struct NeuronAtCompileTime
{
    /// Type alias for the data type used in the neuron.
    using TDataType = DataType;

    /// Scalar type representing a value for the neuron.
    using Scalar = Value<DataType>;

    /**
     * @brief Determines if the neuron has a bias.
     *
     * @return `true` if the neuron has a bias, `false` otherwise.
     */
    consteval bool hasBias() noexcept {
        return bias;
    }

    /**
     * @brief Retrieves the activation type of the neuron.
     *
     * @return The activation function type for the neuron.
     */
    consteval ActivationType activationType() noexcept {
        return actType;
    }

    /**
     * @brief Returns the number of inputs to the neuron (fanin).
     *
     * @return The number of inputs to the neuron.
     */
    constexpr size_t fanin() noexcept {
        return w.size();
    }

    /**
     * @brief Returns the number of outputs from the neuron (fanout), which is always 1 for a single neuron.
     *
     * @return 1, as the output of the neuron is a single scalar.
     */
    consteval size_t fanout() noexcept {
        return 1;
    }

    /**
     * @brief Constructs a neuron with the given input dimension and initialization type.
     *
     * The constructor initializes the neuron's weights and bias (if applicable) based on the provided initialization type.
     *
     * @param input_dim The number of inputs to the neuron.
     * @param init_type The initialization type for weights and bias.
     */
    NeuronAtCompileTime() noexcept
    {
        static burt::RandomVariable rv;

        switch (init_type)
        {
        case NeuronInitType::zero_init:
        {
            if constexpr (bias)
            {
                b = Scalar(DataType());
            }

            for (size_t i = 0; i < input_dim; ++i)
            {
                w[i] = Scalar(DataType());
            }

            break;
        }

        case NeuronInitType::uniform_neg_one_plus_one:
        {
            if constexpr (bias)
            {
                auto value = rv.generateUniform(-1.0, 1.0);
                b = Scalar(value);
            }

            for (size_t i = 0; i < input_dim; ++i)
            {
                auto value = rv.generateUniform(-1.0, 1.0);
                w[i] = Scalar(value);
            }

            break;
        }

        case NeuronInitType::normal_zero_mean_variance_one:
        {
            if constexpr (bias)
            {
                auto value = rv.generateNorm(0.0, 1.0);
                b = Scalar(value);
            }

            for (size_t i = 0; i < input_dim; ++i)
            {
                auto value = rv.generateNorm(0.0, 1.0);
                w[i] = Scalar(value);
            }

            break;
        }

        default:
        {
            burt_unreahable();
            break;
        }
        }
    }

    /**
     * @brief Copy constructor for the neuron.
     *
     * @param rhs The neuron to copy from.
     */
    NeuronAtCompileTime(const NeuronAtCompileTime& rhs) noexcept
        : b(rhs.b)
        , w(rhs.w)
    {}

    /**
     * @brief Computes the forward pass for the neuron using an array of input values.
     *
     * This method computes the weighted sum of inputs and applies the activation function.
     *
     * @tparam N The size of the input array.
     * @param x The input array.
     * @return The result of the forward pass after activation.
     */
    template <size_t N>
    Scalar forward(const std::array<Scalar, N>& x) noexcept
    {
        burt_assert(w.size() == x.size());
        burt_assert(N == x.size());
        burt_assert(Scalar::isSequentialIndicies(w));
        if constexpr (bias)
        {
            burt_assert(w[0].sysGetRawNodeIndex() == 1 + b.sysGetRawNodeIndex());
        }

        if constexpr (bias)
        {
            Scalar sum_ = innerProductWithBiasInternal(&b, w.data(), x.data(), N);

            switch (actType)
            {
            case ActivationType::eIdent:
                return sum_;
            case ActivationType::eTanh:
                return tanh(sum_);
            case ActivationType::eSigmoid:
                return sigmoid(sum_);
            case ActivationType::eRelu:
                return relu(sum_);

            default:
            {
                burt_unreahable();
                return Scalar();
            }
            }
        }
        else
        {
            Scalar sum_ = innerProductInternal(&b, w.data(), x.data(), N);

            switch (actType)
            {
            case ActivationType::eIdent:
                return sum_;
            case ActivationType::eTanh:
                return tanh(sum_);
            case ActivationType::eSigmoid:
                return sigmoid(sum_);
            case ActivationType::eRelu:
                return relu(sum_);

            default:
            {
                burt_unreahable();
                return Scalar();
            }
            }
        }

        {
            burt_unreahable();
            return Scalar();
        }
    }

    /**
     * @brief Computes the forward pass for the neuron using a vector of input values.
     *
     * This method computes the weighted sum of inputs and applies the activation function.
     *
     * @param x The input vector.
     * @return The result of the forward pass after activation.
     */
    Scalar forward(const std::vector<Scalar>& x) noexcept
    {
        burt_assert(w.size() == x.size());

        if constexpr (bias)
        {
            Scalar sum_ = innerProductWithBias(&b, x.data(), w.data(), w.size());

            switch (actType)
            {
            case ActivationType::eIdent:
                return sum_;
            case ActivationType::eTanh:
                return tanh(sum_);
            case ActivationType::eSigmoid:
                return sigmoid(sum_);
            case ActivationType::eRelu:
                return relu(sum_);

            default:
            {
                burt_unreahable();
                return Scalar();
            }
            }
        }
        else
        {
            Scalar sum_ = innerProduct(&b, x.data(), w.data(), w.size());

            switch (actType)
            {
            case ActivationType::eIdent:
                return sum_;
            case ActivationType::eTanh:
                return tanh(sum_);
            case ActivationType::eSigmoid:
                return sigmoid(sum_);
            case ActivationType::eRelu:
                return relu(sum_);

            default:
            {
                burt_unreahable();
                return Scalar();
            }
            }
        }

        {
            burt_unreahable();
            return Scalar();
        }
    }

    /**
     * @brief Retrieves the parameters of the neuron (weights and bias).
     *
     * @return A vector containing the neuron's weights and, if applicable, its bias.
     */
    std::vector<Scalar> parameters() noexcept
    {
        std::vector<Scalar> res(w.begin(), w.end());
        if constexpr (bias)
            res.push_back(b);
        return res;
    }

    /// Destructor for the neuron.
    ~NeuronAtCompileTime() noexcept = default;

    /// The bias of the neuron, if applicable.
    Scalar b;

    /// The weights of the neuron, represented as an array.
    std::array<Scalar, input_dim> w;
};