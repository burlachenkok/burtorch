#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/random/include/RandomVariable.h"

#include "burtcore/include/burtorch_node.h"
#include "burtcore/include/burtorch_mlp_neuron_enums.h"

#include <vector>
#include <array>
#include <assert.h>

/* This structure defines a single neuron that can perform a forward pass with specified activation functions.It supports various initialization schemes and handles parameters such as weights and bias.
*
* @tparam DataType The data type of the neuron(e.g., `float` or `double`).
* @tparam bias A boolean that specifies whether the neuron includes a bias term.Defaults to true.
* @tparam actType The activation type for the neuron.Defaults to `ActivationType::eTanh`.
*/
template<class DataType, bool bias = true, ActivationType actType = ActivationType::eTanh>
struct Neuron
{
    /**
     * @brief Alias for the data type used in the neuron (e.g., `float`, `double`).
     */
    using TDataType = DataType;

    /**
     * @brief Alias for the scalar value type of the neuron.
     */
    using Scalar = Value<DataType>;

    /**
     * @brief Checks if the neuron has a bias.
     *
     * @return `true` if the neuron includes a bias term, `false` otherwise.
     */
    consteval bool hasBias() noexcept {
        return bias;
    }

    /**
     * @brief Returns the activation type of the neuron.
     *
     * @return The activation type (e.g., `eTanh`, `eSigmoid`, etc.).
     */
    consteval ActivationType activationType() noexcept {
        return actType;
    }

    /**
     * @brief Returns the number of input connections (fan-in) for the neuron.
     *
     * @return The size of the weight vector, representing the number of input connections.
     */
    constexpr size_t fanin() noexcept {
        return w.size();
    }

    /**
     * @brief Returns the number of output connections (fan-out) for the neuron.
     *
     * @return Always returns 1, as each neuron has a single output.
     */
    consteval size_t fanout() noexcept {
        return 1;
    }

    /**
    * @brief Constructor that initializes the neuron with the given input dimension and initialization type.
    *
    * @param input_dim The number of input connections (i.e., the size of the weight vector).
    * @param init_type The initialization type for the neuron (e.g., `NeuronInitType::uniform_neg_one_plus_one`).
    */
	Neuron(size_t input_dim, NeuronInitType init_type = NeuronInitType::uniform_neg_one_plus_one) noexcept
	{
		static burt::RandomVariable rv;
		w.reserve(input_dim);

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
                    w.emplace_back(DataType());
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
                    w.emplace_back(value);
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
                    w.emplace_back(value);
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
    * @brief Copy constructor that creates a neuron by copying another.
    *
    * @param rhs The neuron to copy.
    */
	Neuron(const Neuron& rhs) noexcept
	: b(rhs.b)
    , w(rhs.w) 
	{}

    /**
     * Performs a forward pass using the input values provided as an initializer list of scalars.
     *
     * @tparam NItemsTotal The total number of items in the input.
     * @tparam NItemsPerArray The number of items per array in the forward pass.
     * @param x A list of pointers to scalar values representing the inputs to the neuron.
     *
     * @return The output of the neuron after applying the activation function.
     */
    template <size_t NItemsTotal, size_t NItemsPerArray>
    Scalar forward(const std::initializer_list<const Scalar*>& x) noexcept
    {
        burt_assert(w.size() == NItemsTotal);
        burt_assert(Scalar::isSequentialIndicies(w));

        if constexpr (bias)
        {
            burt_assert(w[0].sysGetRawNodeIndex() == 1 + b.sysGetRawNodeIndex());
        }

        if constexpr (bias)
        {
            Scalar sum_ = innerProductWithBiasInternalWithXView<NItemsTotal, NItemsPerArray>(&b, w.data(), x);

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
            Scalar sum_ = innerProductInternalWithXView<NItemsTotal, NItemsPerArray>(&b, w.data(), x);

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
    * Performs a forward pass using the input values provided as a pointer to scalar values.
    *
    * @tparam N The number of items in the input.
    * @param x A pointer to the array of scalar values representing the inputs.
    *
    * @return The output of the neuron after applying the activation function.
    */
    template <size_t N>
    Scalar forward(const Scalar* x) noexcept
    {
        burt_assert(w.size() == N);
        burt_assert(Scalar::isSequentialIndicies(w));

        if constexpr (bias)
        {
            burt_assert(w[0].sysGetRawNodeIndex() == 1 + b.sysGetRawNodeIndex());
        }

        if constexpr (bias)
        {
            Scalar sum_ = innerProductWithBiasInternal<N>(&b, w.data(), x);

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
            Scalar sum_ = innerProductInternal<N>(w.data(), x);

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
    * @brief Performs a forward pass using the input values provided as a vector of scalars.
    *
    * @param x A vector of scalar values representing the inputs.
    *
    * @return The output of the neuron after applying the activation function.
    */
	Scalar forward(const std::vector<Scalar>& x) noexcept
	{
        burt_assert(w.size() == x.size());
		
		if constexpr (bias)
		{
			Scalar sum_ = innerProductWithBiasInternal(&b, w.data(), x.data(), w.size());
            //Scalar sum_ = innerProductWithBias(&b, w.data(), x.data(), w.size());

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
			Scalar sum_ = innerProductInternal(w.data(), x.data(), w.size());

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
    * @brief Returns the parameters of the neuron (weights and bias, if applicable).
    *
    * @return A vector containing the weights and the bias (if the neuron has one).
    */
	std::vector<Scalar> parameters() noexcept
	{
		std::vector<Scalar> res = w;
		if constexpr (bias)
			res.push_back(b);
		return res;
	}

    /**
     * @brief Destructor for the Neuron structure.
     */

    ~Neuron() noexcept = default;

    Scalar b;              ///< The bias of the neuron, used if `bias` is enabled.
	std::vector<Scalar> w; ///< The weight vector of the neuron.
};
