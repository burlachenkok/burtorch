#pragma once

enum class NeuronInitType
{
    zero_init,                    ///< Initializes neuron weights and biases to zero.
    uniform_neg_one_plus_one,     ///< Initializes neuron weights and biases with random values in the range of -1 to 1.
    normal_zero_mean_variance_one ///< Initializes neuron weights and biases with random values from a normal distribution (mean=0, variance=1).
};

enum class ActivationType
{
    eIdent,    ///< Identity activation function, returns the input as is.
    eSigmoid,  ///< Sigmoid activation function, maps inputs to the range (0, 1).
    eRelu,     ///< ReLU (Rectified Linear Unit) activation function, returns the input or zero if input is negative.
    eTanh,     ///< Tanh (hyperbolic tangent) activation function, maps inputs to the range (-1, 1).
    eActNumber ///< A placeholder indicating the total number of activation types. This is not an activation type itself but helps in managing the range of activation types.
};
