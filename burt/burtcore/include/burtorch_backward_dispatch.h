#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/copylocal/include/MutableData.h"
#include "burt/fs/include/FileSystemHelpers.h"

#include "burtcore/include/burtorch_op_metainfo.h"
#include "burtcore/include/burtorch_op_types.h"

#include <algorithm>
#include <stdint.h>

/**
 * Dispatches the backward step for various operations in a computation graph.
 *
 * This function handles the backward propagation of gradients for different types of operations
 * in a computation graph. It updates the gradients of input nodes based on the operation type
 * and the gradient of the output node. The operations supported include common activation functions
 * and arithmetic operations. The behavior varies depending on the provided hint, such as whether
 * gradients should be added or replaced gradient value in child nodes.
 *
 * @tparam Value Type representing the node value. Must support the `gradRef` and `dataRef` methods.
 * @tparam Container Type representing a container of input nodes.
 * @tparam hint Optional parameter specifying the behavior of the backward step. Default is `eNoHints`.
 *
 * @param[out] outNode The output node whose gradient with respect to which already has been computed before the call. And gradient can start flow from it.
 * @param[in] inputNodes A container of input nodes that contribute to the output node's value.
 * @param[in] opType The type of operation for which the backward pass is to be performed.
 *
 * @note The function assumes that the backward operation is only responsible for updating gradients with respect to the input nodes.
 * @note It does not compute the partial derivative with respect to the output node itself. However once all parents nodes (possible more than 1) will finish adding partial derivatives the gradient will be ready.
 *
 * @warning This function operates under the assumption that certain operations, such as element-wise multiplication or division, are defined for the involved node types.
 *
 */

template <class Value, class Container, uint32_t hint = BackwardDispatchHint::eNoHints>
inline void backwardDispatch(Value* outNode, Container& inputNodes, OpType opType) noexcept
{
	using TGradDataType  = typename Value::TGradDataType;
	using TActDataType   = typename Value::TActDataType;
	using TNodeIndexType = typename Value::TNodeIndexType;

	// nuance: Const-Reference                                                       LValue object         Reference
	//            v
	// The C++ standard guarantees that the life of a temporary object if it is LValue (the temporary object that occupies memory) is extended to the life of any reference that refers to it.
	const TGradDataType outGrad = ( (hint & BackwardDispatchHint::eOutGradIsOne) ? TGradDataType(1) : outNode->gradRef());
	constexpr bool theAddGradChildMode = !(hint & BackwardDispatchHint::eReplaceGradsInChilds);

	switch (opType)
    {
        // zero-argumnet ops
		case OpType::eLeaf:
		{
			// leaf is node without any children
            burt_assert(inputNodes.size() == 0);
			break;
        }

        // 1-argumnet ops
        case OpType::eRelu:
        {
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            const TActDataType& outData = outNode->dataRef();

            // auto cond = (outData > TActDataType());
            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad(outData * outGrad );
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad(outData * outGrad);
            }
            break;
        }

        case OpType::eTanh:
        {
            // tanh(x)'=1-tanh(x)^2
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            const TActDataType& outData = outNode->dataRef();

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad((TGradDataType(1) - outData * outData) * outGrad);
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad((TGradDataType(1) - outData * outData) * outGrad);
            }
            break;
        }

        case OpType::eExp:
        {
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            const TActDataType& outData = outNode->dataRef();

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad(outData * outGrad);
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad(outData * outGrad);
            }
            break;
        }

        case OpType::eNegLog:
        {
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->subFromGrad( (outGrad / Value::sysViewMemoryAsNode(&in_index_0)->dataRef()) );
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad( (-outGrad / Value::sysViewMemoryAsNode(&in_index_0)->dataRef()) );
            }
            break;
        }

        case OpType::eSigmoid:
        {
            // sigmoid(x)'=(1-sigmoid(x)) * sigmoid(x)
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            const TActDataType& outData = outNode->dataRef();

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad( (TGradDataType(1) - outData) * outData * outGrad);
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad( (TGradDataType(1) - outData) * outData * outGrad);
            }
            break;
        }

        case OpType::eInv:
        {
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index_0)->dataRef();

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->subFromGrad( outGrad / (inputData * inputData) );
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad( -outGrad / (inputData * inputData) );
            }
            break;
        }

        case OpType::eSqr:
        {
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index_0)->dataRef();

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad((inputData + inputData) * outGrad);
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad((inputData + inputData) * outGrad);
            }
            break;
        }

        case OpType::eCub:
        {
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index_0)->dataRef();

            if constexpr (theAddGradChildMode)
            {
                const TActDataType inputDataSqr = inputData * inputData;
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad((inputDataSqr + inputDataSqr + inputDataSqr)*outGrad);
            }
            else
            {
                const TActDataType inputDataSqr = inputData * inputData;
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad((inputDataSqr + inputDataSqr + inputDataSqr)*outGrad);
            }
            break;
        }
        case OpType::eLog:
        {
            burt_assert(inputNodes.size() == 1);
            auto in_index_0 = inputNodes.fromTinyArray(0);

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad((outGrad / Value::sysViewMemoryAsNode(&in_index_0)->dataRef()));
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad((outGrad / Value::sysViewMemoryAsNode(&in_index_0)->dataRef()));
            }
            break;
        }
		case OpType::eSqrt:
		{
			burt_assert(inputNodes.size() == 1);
			auto in_index_0 = inputNodes.fromTinyArray(0);		
			//const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index_0)->dataRef();
			// const TActDataType inputDataSqrt = sqrt(inputData);
			const TActDataType& outData = outNode->dataRef();

			if constexpr (theAddGradChildMode)
			{
				Value::sysViewMemoryAsNode(&in_index_0)->addToGrad( outGrad / (outData + outData) );
			}
			else
			{
				Value::sysViewMemoryAsNode(&in_index_0)->setGrad(outGrad / (outData + outData) );
			}
			break;
		}
		case OpType::eInvSqrt:
		{
			burt_assert(inputNodes.size() == 1);
			auto in_index_0 = inputNodes.fromTinyArray(0);
			const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index_0)->dataRef(); // (x)
			const TActDataType& outData = outNode->dataRef();                                   // 1.0/sqrt(x)

			if constexpr (theAddGradChildMode)
			{
				Value::sysViewMemoryAsNode(&in_index_0)->subFromGrad(outGrad * outData / (inputData + inputData) );
			}
			else
			{
				Value::sysViewMemoryAsNode(&in_index_0)->setGrad(outGrad * outData / (-(inputData + inputData)));
			}
			break;
		}
        // 2-argumnet ops

        case OpType::eBinaryAdd:
        {
            burt_assert(inputNodes.size() == 2);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            auto in_index_1 = inputNodes.fromTinyArray(1);

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad(/* DataType(1) * */ outGrad);
                Value::sysViewMemoryAsNode(&in_index_1)->addToGrad(/* DataType(1) * */ outGrad);
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad(/* DataType(1) * */ outGrad);
                Value::sysViewMemoryAsNode(&in_index_1)->setGrad(/* DataType(1) * */ outGrad);
            }
            break;
        }

        case OpType::eBinarySub:
        {
            burt_assert(inputNodes.size() == 2);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            auto in_index_1 = inputNodes.fromTinyArray(1);

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad(outGrad);
                Value::sysViewMemoryAsNode(&in_index_1)->subFromGrad(outGrad);
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad(outGrad);
                Value::sysViewMemoryAsNode(&in_index_1)->setGrad(-outGrad);
            }
            break;
        }

        case OpType::eBinaryMult:
        {
            burt_assert(inputNodes.size() == 2);
            auto in_index_0 = inputNodes.fromTinyArray(0);
            auto in_index_1 = inputNodes.fromTinyArray(1);

            if constexpr (theAddGradChildMode)
            {
                Value::sysViewMemoryAsNode(&in_index_0)->addToGrad(Value::sysViewMemoryAsNode(&in_index_1)->dataRef() * outGrad);
                Value::sysViewMemoryAsNode(&in_index_1)->addToGrad(Value::sysViewMemoryAsNode(&in_index_0)->dataRef() * outGrad);
            }
            else
            {
                Value::sysViewMemoryAsNode(&in_index_0)->setGrad(Value::sysViewMemoryAsNode(&in_index_1)->dataRef() * outGrad);
                Value::sysViewMemoryAsNode(&in_index_1)->setGrad(Value::sysViewMemoryAsNode(&in_index_0)->dataRef() * outGrad);
            }
            break;
        }

		case OpType::eBinaryMultByConst:
		{
			burt_assert(inputNodes.size() == 2);
			auto in_index_0_value = inputNodes.fromTinyArray(0);
			auto in_index_1_const = inputNodes.fromTinyArray(1);

			if constexpr (theAddGradChildMode)
			{
				Value::sysViewMemoryAsNode(&in_index_0_value)->addToGrad(Value::sysViewMemoryAsNode(&in_index_1_const)->dataRef() * outGrad);
				//Value::sysViewMemoryAsNode(&in_index_1_const)->addToGrad(Value::sysViewMemoryAsNode(&in_index_0_value)->dataRef() * outGrad);
			}
			else
			{
				Value::sysViewMemoryAsNode(&in_index_0_value)->setGrad(Value::sysViewMemoryAsNode(&in_index_1_const)->dataRef() * outGrad);
				//Value::sysViewMemoryAsNode(&in_index_1_const)->setGrad(Value::sysViewMemoryAsNode(&in_index_0_value)->dataRef() * outGrad);
			}
			break;
		}

        case OpType::eBinaryDiv:
        {
            burt_assert(inputNodes.size() == 2);

            auto in_index_0 = inputNodes.fromTinyArray(0);
            auto in_index_1 = inputNodes.fromTinyArray(1);

			Value* in_index_0_node = Value::sysViewMemoryAsNode(&in_index_0);
			Value* in_index_1_node = Value::sysViewMemoryAsNode(&in_index_1);

            if constexpr (theAddGradChildMode)
            {
				in_index_0_node->addToGrad( outGrad / in_index_1_node->dataRef() );
				in_index_1_node->subFromGrad( outGrad * in_index_0_node->dataRef() / in_index_1_node->dataRef() / in_index_1_node->dataRef());
            }
            else
            {
				in_index_0_node->setGrad(TGradDataType(1) / in_index_1_node->dataRef() * outGrad);
				in_index_1_node->setGrad(-in_index_0_node->dataRef() / in_index_1_node->dataRef() / in_index_1_node->dataRef() * outGrad);
            }
            break;
        }

        case OpType::eBinaryMean:
        {
            burt_assert(inputNodes.size() == 2);

            auto in_index_0 = inputNodes.fromTinyArray(0);
            auto in_index_1 = inputNodes.fromTinyArray(1);

			Value* in_index_0_node = Value::sysViewMemoryAsNode(&in_index_0);
			Value* in_index_1_node = Value::sysViewMemoryAsNode(&in_index_1);

            constexpr double divider = 1.0 / 2.0;

            if constexpr (theAddGradChildMode)
            {
				in_index_0_node->addToGrad(TGradDataType(divider) * outGrad);
				in_index_1_node->addToGrad(TGradDataType(divider) * outGrad);
            }
            else
            {
				in_index_0_node->setGrad(TGradDataType(divider) * outGrad);
				in_index_1_node->setGrad(TGradDataType(divider) * outGrad);
            }
            break;
        }


        case OpType::eBinaryAddSquares:
        {
            burt_assert(inputNodes.size() == 2);

            auto in_index_0 = inputNodes.fromTinyArray(0);
            auto in_index_1 = inputNodes.fromTinyArray(1);

			Value* in_index_0_node = Value::sysViewMemoryAsNode(&in_index_0);
			Value* in_index_1_node = Value::sysViewMemoryAsNode(&in_index_1);

            const auto& in_data_0 = in_index_0_node->dataRef();
            const auto& in_data_1 = in_index_1_node->dataRef();

            if constexpr (theAddGradChildMode)
            {
				in_index_0_node->addToGrad( (in_data_0 + in_data_0) * outGrad );
				in_index_1_node->addToGrad( (in_data_1 + in_data_1) * outGrad );
            }
            else
            {
				in_index_0_node->setGrad((in_data_0 + in_data_0) * outGrad);
				in_index_1_node->setGrad((in_data_1 + in_data_1) * outGrad);
            }
            break;
        }

        case OpType::eBinaryMeanSquares:
        {
            burt_assert(inputNodes.size() == 2);

            auto in_index_0 = inputNodes.fromTinyArray(0);
            auto in_index_1 = inputNodes.fromTinyArray(1);

			Value* in_index_0_node = Value::sysViewMemoryAsNode(&in_index_0);
			Value* in_index_1_node = Value::sysViewMemoryAsNode(&in_index_1);

            const auto& in_data_0 = in_index_0_node->dataRef();
            const auto& in_data_1 = in_index_1_node->dataRef();

            constexpr double divider = 1.0 / 2.0;

            if constexpr (theAddGradChildMode)
            {

				in_index_0_node->addToGrad((in_data_0 + in_data_0) * TGradDataType(divider) * outGrad);
				in_index_1_node->addToGrad((in_data_1 + in_data_1) * TGradDataType(divider) * outGrad);
            }
            else
            {
				in_index_0_node->setGrad((in_data_0 + in_data_0)* TGradDataType(divider) * outGrad);
				in_index_1_node->setGrad((in_data_1 + in_data_1)* TGradDataType(divider) * outGrad);
            }
            break;
        }


        case OpType::eBinaryNegativeMean:
        {
            burt_assert(inputNodes.size() == 2);

            auto in_index_0 = inputNodes.fromTinyArray(0);
            auto in_index_1 = inputNodes.fromTinyArray(1);

			Value* in_index_0_node = Value::sysViewMemoryAsNode(&in_index_0);
			Value* in_index_1_node = Value::sysViewMemoryAsNode(&in_index_1);

            constexpr double divider = (-1.0/2.0);

            if constexpr (theAddGradChildMode)
            {
				in_index_0_node->addToGrad(TGradDataType(divider) * outGrad);
				in_index_1_node->addToGrad(TGradDataType(divider) * outGrad);
            }
            else
            {
				in_index_0_node->setGrad(TGradDataType(divider) * outGrad);
				in_index_1_node->setGrad(TGradDataType(divider) * outGrad);
            }
            break;
        }

        // n-argumnet ops

		case OpType::eAddVarying:
		{
            auto inputNodesNumber = inputNodes.size();
            burt_assert(inputNodesNumber >= 1);

            if constexpr (theAddGradChildMode)
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);

					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
					{
						inputNode->addToGrad( /* DataType(1) * */ outGrad);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
					{
						Value::sysViewMemoryAsNode(&in_index)->addToGrad( /* DataType(1) * */ outGrad);
					}
				}
			}
			else
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);

					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
						inputNode->setGrad( /* DataType(1) * */ outGrad);
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());
					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
						Value::sysViewMemoryAsNode(&in_index)->setGrad( /* DataType(1) * */ outGrad);
				}
			}
			break;
		}

        case OpType::eSubVarying:
        {
			auto inputNodesNumber = inputNodes.size();
			burt_assert(inputNodesNumber >= 1);

			if constexpr (theAddGradChildMode)
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					inputNode->addToGrad( /* DataType(1) * */ outGrad);
					++inputNode;

					for (size_t i = 1; i < inputNodesNumber; ++i, ++inputNode)
					{
						inputNode->subFromGrad( /* DataType(1) * */ outGrad);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					Value::sysViewMemoryAsNode(&in_index)->addToGrad( /* DataType(1) * */ outGrad);
					in_index += in_index_step;

					for (size_t i = 1; i < inputNodesNumber; ++i, in_index += in_index_step)
					{
						Value::sysViewMemoryAsNode(&in_index)->subFromGrad( /* DataType(1) * */ outGrad);
					}
				}
			}
			else
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					inputNode->setGrad( /* DataType(1) * */ outGrad);
					++inputNode;

					for (size_t i = 1; i < inputNodesNumber; ++i, ++inputNode)
						inputNode->setGrad( /* DataType(1) * */ -outGrad);
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();
					
					Value::sysViewMemoryAsNode(&in_index)->setGrad( /* DataType(1) * */ outGrad);
					in_index += in_index_step;
					for (size_t i = 1; i < inputNodesNumber; ++i, in_index += in_index_step)
						Value::sysViewMemoryAsNode(&in_index)->setGrad( /* DataType(1) * */ -outGrad);
				}
			}
            break;
        }

        case OpType::eMulVarying:
        {
            auto inputNodesNumber = inputNodes.size();
            burt_assert(inputNodesNumber >= 1);

            for (size_t i = 0; i < inputNodesNumber; ++i)
            {
                TGradDataType grad = outGrad;

				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);

					for (size_t ii = 0; ii < i; ++ii, ++inputNode)
						grad *= inputNode->dataRef();

					Value* outNode = inputNode;
					++inputNode;

					for (size_t ii = i + 1; ii < inputNodesNumber; ++ii, ++inputNode)
					{
						grad *= inputNode->dataRef();
					}

					if constexpr (theAddGradChildMode)
						outNode->addToGrad(grad);
					else
						outNode->setGrad(grad);
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t ii = 0; ii < i; ++ii, in_index += in_index_step)
						grad *= Value::sysViewMemoryAsNode(&in_index)->dataRef();

					auto in_index_result = in_index;
					in_index += in_index_step;

					for (size_t ii = i + 1; ii < inputNodesNumber; ++ii, ++in_index)
						grad *= Value::sysViewMemoryAsNode(&in_index)->dataRef();

					if constexpr (theAddGradChildMode)
						Value::sysViewMemoryAsNode(&in_index_result)->addToGrad(grad);
					else
						Value::sysViewMemoryAsNode(&in_index_result)->addToGrad(grad);
				}
			}
            break;
        }

		case OpType::eMeanVarying:
		{
            auto inputNodesNumber = inputNodes.size();
            burt_assert(inputNodesNumber >= 1);

            double divider = 1.0 / double(inputNodesNumber);
			
            if constexpr (theAddGradChildMode)
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
						inputNode->addToGrad(divider * outGrad);
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
						Value::sysViewMemoryAsNode(&in_index)->addToGrad(divider * outGrad);
				}
			}
			else
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
						inputNode->setGrad(divider * outGrad);
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
						Value::sysViewMemoryAsNode(&in_index)->setGrad(divider * outGrad);
				}
			}
			break;
		}

        case OpType::eSumOfSquaresVarying:
        {
			auto inputNodesNumber = inputNodes.size();
			burt_assert(inputNodesNumber >= 1);
			const TGradDataType multiplier_total = (outGrad + outGrad);

			if constexpr (theAddGradChildMode)
			{

				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
					{
						const TActDataType& inputData = inputNode->dataRef();
						inputNode->addToGrad((inputData) * multiplier_total);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
					{
						const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index)->dataRef();
						Value::sysViewMemoryAsNode(&in_index)->addToGrad((inputData) * multiplier_total);
					}
				}
			}
			else
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
					{
						const TActDataType& inputData = inputNode->dataRef();
						inputNode->setGrad((inputData) * multiplier_total);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
					{
						const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index)->dataRef();
						Value::sysViewMemoryAsNode(&in_index)->setGrad((inputData) * multiplier_total);
					}
				}
			}
			break;
        }

		case OpType::eMeanSquaresVarying:
		{
			auto inputNodesNumber = inputNodes.size();
			burt_assert(inputNodesNumber >= 1);
			TGradDataType multiplier_total = (outGrad + outGrad) / TGradDataType(inputNodesNumber);

			if constexpr (theAddGradChildMode)
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
					{
						const TActDataType& inputData = inputNode->dataRef();
						inputNode->addToGrad(inputData * multiplier_total);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
					{
						const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index)->dataRef();
						Value::sysViewMemoryAsNode(&in_index)->addToGrad(inputData * multiplier_total);
					}
				}
			}
			else
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
					{
						const TActDataType& inputData = inputNode->dataRef();
						inputNode->setGrad(inputData * multiplier_total);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
					{
						const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index)->dataRef();
						Value::sysViewMemoryAsNode(&in_index)->setGrad(inputData * multiplier_total);
					}
				}
			}
			break;
		}

        case OpType::eNegativeMeanVarying:
        {
			auto inputNodesNumber = inputNodes.size();
			burt_assert(inputNodesNumber >= 1);

			TGradDataType multiplier_total = -(outGrad + outGrad) / TGradDataType(inputNodesNumber);
			if constexpr (theAddGradChildMode)
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
					{
						const TActDataType& inputData = inputNode->dataRef();
						inputNode->addToGrad(inputData * multiplier_total);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
					{
						const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index)->dataRef();
						Value::sysViewMemoryAsNode(&in_index)->addToGrad(inputData * multiplier_total);
					}
				}
			}
			else
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNode = Value::sysViewMemoryAsNode(inputNodesRaw);
					for (size_t i = 0; i < inputNodesNumber; ++i, ++inputNode)
					{
						const TActDataType& inputData = inputNode->dataRef();
						inputNode->setGrad(inputData * multiplier_total);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					for (size_t i = 0; i < inputNodesNumber; ++i, in_index += in_index_step)
					{
						const TActDataType& inputData = Value::sysViewMemoryAsNode(&in_index)->dataRef();
						Value::sysViewMemoryAsNode(&in_index)->setGrad(inputData * multiplier_total);
					}
				}
			}
            break;
        }

		case OpType::eInnerProductNoBias:
		{
            auto inputNodesNumber = inputNodes.size();
            burt_assert(inputNodesNumber % 2 == 0);
			auto inputNodesNumberHalf = (inputNodesNumber  >> 1);

            if constexpr (theAddGradChildMode)
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNodeW = Value::sysViewMemoryAsNode(inputNodesRaw);
					Value* inputNodeX = inputNodeW + inputNodesNumberHalf;

					for (size_t i = 0; i < inputNodesNumberHalf; i++, inputNodeW++, inputNodeX++)
					{
						const auto& in_index_w_data = inputNodeW->dataRef();
						const auto& in_index_x_data = inputNodeX->dataRef();
						inputNodeW->addToGrad(in_index_x_data * outGrad);
						inputNodeX->addToGrad(in_index_w_data * outGrad);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					auto in_index_w_1 = in_index;
					auto in_index_x_1 = static_cast<decltype(in_index_w_1)>(in_index + in_index_step * inputNodesNumberHalf);

					for (size_t i = 0; i < inputNodesNumberHalf; ++i, in_index_w_1 += in_index_step, in_index_x_1 += in_index_step)
					{
						Value* in_w_node_1 = Value::sysViewMemoryAsNode(&in_index_w_1);
						Value* in_x_node_1 = Value::sysViewMemoryAsNode(&in_index_x_1);
						const auto in_index_w_data_1 = in_w_node_1->dataCopy();
						const auto in_index_x_data_1 = in_x_node_1->dataCopy();
						in_w_node_1->addToGrad(in_index_x_data_1 * outGrad);
						in_x_node_1->addToGrad(in_index_w_data_1 * outGrad);
					}
				}
			}
			else
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNodeW = Value::sysViewMemoryAsNode(inputNodesRaw);
					Value* inputNodeX = inputNodeW + inputNodesNumberHalf;

					for (size_t i = 0; i < inputNodesNumberHalf; i++, inputNodeW++, inputNodeX++)
					{
						const auto& in_index_w_data = inputNodeW->dataRef();
						const auto& in_index_x_data = inputNodeX->dataRef();
						inputNodeW->setGrad(in_index_x_data * outGrad);
						inputNodeX->setGrad(in_index_w_data * outGrad);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					auto in_index_w_1 = in_index;
					auto in_index_x_1 = static_cast<decltype(in_index_w_1)>(in_index + in_index_step * inputNodesNumberHalf);

					for (size_t i = 0; i < inputNodesNumberHalf; ++i, in_index_w_1 += in_index_step, in_index_x_1 += in_index_step)
					{
						Value* in_w_node_1 = Value::sysViewMemoryAsNode(&in_index_w_1);
						Value* in_x_node_1 = Value::sysViewMemoryAsNode(&in_index_x_1);
						const auto in_index_w_data_1 = in_w_node_1->dataCopy();
						const auto in_index_x_data_1 = in_x_node_1->dataCopy();
						in_w_node_1->setGrad(in_index_x_data_1 * outGrad);
						in_x_node_1->setGrad(in_index_w_data_1 * outGrad);
					}
				}
			}
			break;
		}
		case OpType::eInnerProductWithBias:
		{
			auto inputNodesNumber = inputNodes.size();
			burt_assert(inputNodesNumber % 2 == 1);
			auto inputNodesNumberHalf = (inputNodesNumber >> 1);

			if constexpr (theAddGradChildMode)
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNodeBias = Value::sysViewMemoryAsNode(inputNodesRaw);
					Value* inputNodeW = inputNodeBias + 1;
					Value* inputNodeX = inputNodeBias + 1 + inputNodesNumberHalf;
					inputNodeBias->addToGrad( /*1*/ outGrad);

					for (size_t i = 0; i < inputNodesNumberHalf; i++, inputNodeW++, inputNodeX++)
					{
						const auto& in_index_w_data = inputNodeW->dataRef();
						const auto& in_index_x_data = inputNodeX->dataRef();
						inputNodeW->addToGrad(in_index_x_data * outGrad);
						inputNodeX->addToGrad(in_index_w_data * outGrad);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					auto i_bias_index = in_index;
					Value::sysViewMemoryAsNode(&i_bias_index)->addToGrad( /*1*/ outGrad);
					in_index++;

					auto in_index_w_1 = in_index;
					auto in_index_x_1 = static_cast<decltype(in_index_w_1)>(in_index + in_index_step * inputNodesNumberHalf);

					for (size_t i = 0; i < inputNodesNumberHalf; ++i, in_index_w_1 += in_index_step, in_index_x_1 += in_index_step)
					{
						Value* in_w_node_1 = Value::sysViewMemoryAsNode(&in_index_w_1);
						Value* in_x_node_1 = Value::sysViewMemoryAsNode(&in_index_x_1);
						const auto in_index_w_data_1 = in_w_node_1->dataCopy();
						const auto in_index_x_data_1 = in_x_node_1->dataCopy();
						in_w_node_1->addToGrad(in_index_x_data_1 * outGrad);
						in_x_node_1->addToGrad(in_index_w_data_1 * outGrad);
					}
				}
			}
			else
			{
				if (TNodeIndexType* inputNodesRaw = inputNodes.data())
				{
					Value* inputNodeBias = Value::sysViewMemoryAsNode(inputNodesRaw);
					Value* inputNodeW = inputNodeBias + 1;
					Value* inputNodeX = inputNodeBias + 1 + inputNodesNumberHalf;

					inputNodeBias->setGrad( /*1*/ outGrad);

					for (size_t i = 0; i < inputNodesNumberHalf; i++, inputNodeW++, inputNodeX++)
					{
						const auto in_index_w_data = inputNodeW->dataCopy();
						const auto in_index_x_data = inputNodeX->dataCopy();
						inputNodeW->setGrad(in_index_x_data * outGrad);
						inputNodeX->setGrad(in_index_w_data * outGrad);
					}
				}
				else
				{
					burt_assert(inputNodes.isArithmProgressArray());

					auto in_index = inputNodes.getArithmProgressFirstItem();
					auto in_index_step = inputNodes.getArithmProgressStep();

					auto i_bias_index = in_index;
					Value::sysViewMemoryAsNode(&i_bias_index)->addToGrad( /*1*/ outGrad);
					in_index++;

					auto in_index_w_1 = in_index;
					auto in_index_x_1 = static_cast<decltype(in_index_w_1)>(in_index + in_index_step * inputNodesNumberHalf);

					for (size_t i = 0; i < inputNodesNumberHalf; ++i, in_index_w_1 += in_index_step, in_index_x_1 += in_index_step)
					{
						Value* in_w_node_1 = Value::sysViewMemoryAsNode(&in_index_w_1);
						Value* in_x_node_1 = Value::sysViewMemoryAsNode(&in_index_x_1);
						const auto in_index_w_data_1 = in_w_node_1->dataCopy();
						const auto in_index_x_data_1 = in_x_node_1->dataCopy();
						in_w_node_1->setGrad(in_index_x_data_1 * outGrad);
						in_x_node_1->setGrad(in_index_w_data_1 * outGrad);
					}
				}
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

//================================================================================================================================================================//
// BEST IMPLEMENTATIONS OF BACKWARD SO FAR. PRETTY LOW-LEVEL
//================================================================================================================================================================//

/** Stack entry type -- for post-order in non-recursive way we need to distinguish two types
*/
enum class StackEntryType : std::uint8_t
{
	k_stack_entry_unmarked = 0,
	k_stack_entry_marked = 1
};

/** Stack entry which hide the bit of type in highest bit of address
* @warning This type of optimization is not-perfectly but fine for all modern OS which has Kernel Space and User Space. And addresses in user space can not use highest bit
*/
template<class TItemType>
struct StackEntry
{
	StackEntry()
	: node(TItemType())
#if BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META == 0
	, type(StackEntryType())
#endif
	{}

	static StackEntry constructUnmarkedStackEntry(TItemType node)
	{
		StackEntry res;
		res.node = node;

#if BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META == 0
		res.type = StackEntryType::k_stack_entry_unmarked;
#endif
		return res;
	}

	static StackEntry constructMarkedStackEntry(TItemType node)
	{
		StackEntry res;
		res.node = node;

#if BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META
		constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
		res.node |= (((TItemType)0x1) << (kBitInNode));
#else
		res.type = StackEntryType::k_stack_entry_marked;
#endif
		return res;
	}

	constexpr bool isMarked() const
	{
#if BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META
		constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
		TItemType mask = ( ((TItemType)0x1) << (kBitInNode) );
		return node >= mask;
#else
		return type == StackEntryType::k_stack_entry_marked;
#endif
	}

	constexpr StackEntryType getType() const
	{
#if BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META
		constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
		if (node & (((TItemType)0x1) << (kBitInNode)))
		{
			return StackEntryType::k_stack_entry_marked;
		}
		else
		{
			return StackEntryType::k_stack_entry_unmarked;
		}
#else
		return type;
#endif
	}

	template<bool promiseToTackUnmarked>
	constexpr TItemType getCleanNode() const
	{
#if BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META
		if constexpr (promiseToTackUnmarked)
		{
			return node;
		}
		else
		{
			constexpr size_t kBitInNode = sizeof(TItemType) * 8 - 1;
			return (node & ~(((TItemType)0x1) << (kBitInNode)));
		}
#else
		return node;
#endif
	}

	TItemType node;

#if BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META == 0
	StackEntryType type;
#endif
};

//================================================================================================================================================================//
// VERSION: 4 [better memory allocations]
template <class TValueType,
		  bool execute_reverse_topo_order = true,
		  bool execute_backward_for_internal_nodes = true,
	      bool execute_backward_for_leafs = false
          >

inline void backwardWithScratchStorage(TValueType& root, 
									   burt::MutableData& reverse_topo_order,
									   burt::MutableData& leafs_which_can_computed_in_any_order,
  	                                   burt::MutableData& recursion_stack) noexcept
{
	using TValueTypePtr = TValueType*;
	using TNodeIndexType = typename TValueType::TNodeIndexType;

	static_assert(std::is_trivially_copyable<StackEntry<TNodeIndexType>>::value);
	static_assert(std::is_trivially_copyable<TValueTypePtr>::value);
	//static_assert(sizeof(StackEntry<TValueType>) >= sizeof(TValueTypePtr));

	// bool each_node_has_single_income = false;

	// TOPOLOGICAL SORT PART
	if constexpr (execute_reverse_topo_order)
	{
		//each_node_has_single_income = true;

		// reset seek position for result
		reverse_topo_order.rewindToStart();

		if constexpr (execute_backward_for_leafs)
		{
			leafs_which_can_computed_in_any_order.rewindToStart();
		}

		unsigned int new_maker = root.backwardOptVisitNumberForBackpropCopy() + 1;
		StackEntry<TNodeIndexType> v = StackEntry<TNodeIndexType>::constructUnmarkedStackEntry(root.sysGetRawNodeIndex());
		burt_assert(v.isMarked() == false);

		// unmark -- still to process
		// marked -- dfs for children was finished
		
		//root.backwardOptVisitNumberSet(new_maker);

		// stack: [recursion, v] <<< head
		for (;;)
		{
			if (v.isMarked())
			{
				// PUT TO RESULT:cNo need to reserve memory -- we are for sure have enough memory right now
                TNodeIndexType vNodeIndex = v.template getCleanNode<false>();
				TValueType* vNode = TValueType::sysViewMemoryAsNode(&vNodeIndex);
                burt_assert(vNode->childrenNum() > 0);

				auto cBackprops = vNode->backwardOptVisitNumberForBackpropCopy();
				
				if (new_maker != cBackprops) [[likely]]
				{
					// it can be a situation that during waiting to be postprocessed
					vNode->backwardOptVisitNumberSet(new_maker);
					reverse_topo_order.putPOD(vNode->sysGetRawNodeIndex());
				}
				else
				{
					// each_node_has_single_income = false;
				}

				if (!recursion_stack.isEmpty())
				{
					// pop the stack and copy last item into v
					size_t wPos = recursion_stack.seekBackwardAtCompileTime</*size_t delta*/ sizeof(StackEntry<TNodeIndexType>), /*bool promise_pos_is_ge_delta*/ true>();
					v = *(StackEntry<TNodeIndexType>*)(recursion_stack.getPtr() + wPos);
					continue;
				}
			}
			else // v.getType() == StackEntryType::kStackEntry_process_unmarked
			{
                TNodeIndexType vNodeIndex = v.template getCleanNode<true>();
				TValueType* vNode = TValueType::sysViewMemoryAsNode(&vNodeIndex);
				
				// mark as processed right now => nobody should process it futher
				//vNode->backwardOptVisitNumberSet(new_maker);
				const auto& childSet = vNode->childrenSet();
				auto children_number = childSet.size();
                
				if (children_number == 0)
				{
                    burt_assert(strcmp("leaf", vNode->getHelpString()) == 0);

					// Special case: 0 children
					//  reasons for reserving this amount of memory: 
					//    - recursion_stack stack contains potential to handle records with postprocess
					//    - one node is current node (not in stack) it is vNode
					//    - one extra is valid lookahead for seeking

					if constexpr (execute_backward_for_leafs)
					{
						// Do not put node more then once into leafs_which_can_computed_in_any_order
						// In principle it's possible to put result into reverse_topo_order, but it has no sense because <reverse_topo_order> is used for backward pass next. 
						// And backward pass for leaf nodes contains no real code to execute
						auto vBackprops = vNode->backwardOptVisitNumberForBackpropCopy();
						if (new_maker != vBackprops) [[likely]]
						{
							vNode->backwardOptVisitNumberSet(new_maker);
							leafs_which_can_computed_in_any_order.reserveMemory(recursion_stack.getFilledSize());
							leafs_which_can_computed_in_any_order.putPOD(vNode->sysGetRawNodeIndex());
						}
					}

					if (!recursion_stack.isEmpty())
					{
						// stack is not empty
						// pop the stack: use knowledge that recursion_stack contains only StackEntry
						size_t wPos = recursion_stack.seekBackwardAtCompileTime</*size_t delta*/ sizeof(StackEntry<TNodeIndexType>), /*bool promise_pos_is_ge_delta*/ true>();
						// Copy entry from stack -- use pointers can be a bit unsafe
						v = *(StackEntry<TNodeIndexType>*)(recursion_stack.getPtr() + wPos);
						continue;
					}
				}
				else
				{
					//auto children_number = childSet.size();
                    //auto children_first = vNode->childrenSet().cbegin();
                    //auto children_last = children_first + children_number - 1;

					// Case: 1 children or more
					// Post-process entry -- we can not process right now with parent node. We do it later. And to do it later we put postprocess stack entry.
					//
					recursion_stack.putPOD<StackEntry<TNodeIndexType>, true, burt::MutableDataRelocationPolicy::eIncreaseByTwoAndRoundUpToChunk>(StackEntry<TNodeIndexType>::constructMarkedStackEntry(vNodeIndex));
					// remark-2: ... however do it later implies that recursion_stack is for sure non-empty right now
					// note-3: "process" via puting into the stack all child up to the last (the last will virtuallly put and immediately pop from the stack)
					//  important it's not exactly DFS
					//    a  
					//  b  c
					// e f
					// DFS order: a->b->e->e'->f->f'->b'->c->c'->a'  [only postorder: e'->f'->b'->c'->a']
					// order for this traverals: a->c->c'->b->f->f'->e->e'->b'->a' [only postorder:: c'->f'->e'->b'->a']
					// However it does not matter: both post-order traversal garantees the following: [root need be processed => all child are processed]
					//  order does not really matter: processing in this order implies visiting in reverses order (e,f) => memory access is more sequential

					if (const TNodeIndexType* rawChildArray = childSet.dataConst())
					{
						const TNodeIndexType* rawChildLast = rawChildArray + children_number - 1;

						for (;rawChildArray != rawChildLast; ++rawChildArray)
						{
							TNodeIndexType cIndex = *rawChildArray;
							TValueType* cNode = TValueType::sysViewMemoryAsNode(&cIndex);

							auto cBackprops = cNode->backwardOptVisitNumberForBackpropCopy();

							// checking is important, because checks for marking happens only on 2 places: during post-order processing -- to be sure that nodes has not been processed.
							// here to be sure that we are not recursively pushing into the stack already processed computation graph subgraphs
							if (new_maker != cBackprops) [[likely]]
							{
								if (cNode->childrenSet().isEmpty())
								{
									if constexpr (execute_backward_for_leafs)
									{
										// Do not put node more then once into leafs_which_can_computed_in_any_order
										// In principle it's possible to put result into reverse_topo_order, but it has no sense because <reverse_topo_order> is used for backward pass next. 
										// And backward pass for leaf nodes contains no real code to execute
										auto vBackprops = vNode->backwardOptVisitNumberForBackpropCopy();
										if (new_maker != vBackprops) [[likely]]
										{
											vNode->backwardOptVisitNumberSet(new_maker);
											leafs_which_can_computed_in_any_order.reserveMemory(recursion_stack.getFilledSize());
											leafs_which_can_computed_in_any_order.putPOD(vNode->sysGetRawNodeIndex());
										}
									}
								}
								else
								{
									recursion_stack.putPOD </*class T*/ StackEntry<TNodeIndexType>,
															/*bool moveDataWindow*/ true,
															/*MutableDataRelocationPolicy relocation*/ burt::MutableDataRelocationPolicy::eIncreaseByTwoAndRoundUpToChunk
															>
										(
											StackEntry<TNodeIndexType>::constructUnmarkedStackEntry(cIndex)
										);
								}
							}
							else
							{
								// each_node_has_single_income = false;
							}
						}

						// process last
						{
							TNodeIndexType cIndex = *rawChildLast;
							TValueType* cNode = TValueType::sysViewMemoryAsNode(&cIndex);
							auto cBackprops = cNode->backwardOptVisitNumberForBackpropCopy();

							if (new_maker != cBackprops) [[likely]]
							{
								// optimization: do not push/pop from the stack -- process directly
								v = StackEntry<TNodeIndexType>::constructUnmarkedStackEntry(cIndex);
								continue;
							}
							else
							{
								// last node has been processed already. pop the stack and copy last item into v. SAFE TO REMOVE DUE TO REMARK-2.
								size_t wPos = recursion_stack.seekBackwardAtCompileTime</*size_t delta*/ sizeof(StackEntry<TNodeIndexType>), /*bool promise_pos_is_ge_delta*/ true>();
								v = *(StackEntry<TNodeIndexType>*)(recursion_stack.getPtr() + wPos);
								// each_node_has_single_income = false;
								continue;
							}
						}
					}
					else
					{
						burt_assert(childSet.isArithmProgressArray());

						auto children_number_last = children_number - 1;

						TNodeIndexType cIndex = childSet.getArithmProgressFirstItem();
						TNodeIndexType cIndexStep = childSet.getArithmProgressStep();

						for (size_t i = 0; i < children_number_last; ++i, cIndex += cIndexStep)
						{
							TValueType* cNode = TValueType::sysViewMemoryAsNode(&cIndex);

							auto cBackprops = cNode->backwardOptVisitNumberForBackpropCopy();

							// checking is important, because checks for marking happens only on 2 places: during post-order processing -- to be sure that nodes has not been processed.
							// here to be sure that we are not recursively pushing into the stack already processed computation graph subgraphs
							if (new_maker != cBackprops) [[likely]]
							{								
								if (cNode->childrenSet().isEmpty())
								{
									if constexpr (execute_backward_for_leafs)
									{
										// Do not put node more then once into leafs_which_can_computed_in_any_order
										// In principle it's possible to put result into reverse_topo_order, but it has no sense because <reverse_topo_order> is used for backward pass next. 
										// And backward pass for leaf nodes contains no real code to execute
										auto vBackprops = vNode->backwardOptVisitNumberForBackpropCopy();
										if (new_maker != vBackprops) [[likely]]
										{
											vNode->backwardOptVisitNumberSet(new_maker);
											leafs_which_can_computed_in_any_order.reserveMemory(recursion_stack.getFilledSize());
											leafs_which_can_computed_in_any_order.putPOD(vNode->sysGetRawNodeIndex());
										}
									}
								}
								else
								{
									recursion_stack.putPOD </*class T*/ StackEntry<TNodeIndexType>,
															/*bool moveDataWindow*/ true,
															/*MutableDataRelocationPolicy relocation*/ burt::MutableDataRelocationPolicy::eIncreaseByTwoAndRoundUpToChunk
														   >
										(
											StackEntry<TNodeIndexType>::constructUnmarkedStackEntry(cIndex)
										);
								}
							}
						}

						// process last
						{
							//TNodeIndexType cIndex = vNode->childrenSet().get(children_number_last);
							TValueType* cNode = TValueType::sysViewMemoryAsNode(&cIndex);

							auto cBackprops = cNode->backwardOptVisitNumberForBackpropCopy();
							if (new_maker != cBackprops) [[likely]]
							{
								// optimization: do not push/pop from the stack -- process directly
								v = StackEntry<TNodeIndexType>::constructUnmarkedStackEntry(cIndex);
								burt_assert(v.isMarked() == false);
								continue;
							}
							else
							{
								// last node has been processed already. pop the stack and copy last item into v. SAFE TO REMOVE DUE TO REMARK-2.
								size_t wPos = recursion_stack.seekBackwardAtCompileTime</*size_t delta*/ sizeof(StackEntry<TNodeIndexType>), /*bool promise_pos_is_ge_delta*/ true>();
								v = *(StackEntry<TNodeIndexType>*)(recursion_stack.getPtr() + wPos);
								// each_node_has_single_income = false;
								continue;
							}
						}
					}
				}
			}

			// break the loop
			break;
		}

	}

	// FORMALLY SET GRAD OF FINAL NOE W.R.T. TO FINAL NODE AS 1.0
	root.setGradToOne();

	// BACKWARD PHASE
	if constexpr (execute_backward_for_internal_nodes)
	{
		// Execute dependencies in topological order: therefore there is a garantee not to start node execution until it's parents in parent->child relation has been finished
		TNodeIndexType* basePtr = (TNodeIndexType*)(reverse_topo_order.getPtr());
		TNodeIndexType* curPtr = (TNodeIndexType*)(reverse_topo_order.getPtr() + reverse_topo_order.getFilledSize());

		if (basePtr != curPtr)
		{
			curPtr--;
			burt_assert(*curPtr == root.sysGetRawNodeIndex());
			TValueType::sysViewMemoryAsNode(curPtr)->template backward <BackwardDispatchHint::eOutGradIsOne>();
		}
		
		for (; curPtr != basePtr;)
		{
			--curPtr;
			TValueType::sysViewMemoryAsNode(curPtr)->template backward <BackwardDispatchHint::eNoHints> ();
		}
	}

	if constexpr (execute_backward_for_leafs)
	{
		// Leafs can be executed in any order it does not matter. They have no children and can not affect correctness
		TNodeIndexType* basePtrLeafs = (TNodeIndexType*)(leafs_which_can_computed_in_any_order.getPtr());
		TNodeIndexType* endPtrLeafs = (TNodeIndexType*)(leafs_which_can_computed_in_any_order.getPtr() + leafs_which_can_computed_in_any_order.getFilledSize());

		for (; basePtrLeafs != endPtrLeafs; ++basePtrLeafs)
			TValueType::sysViewMemoryAsNode(basePtrLeafs)->template backward <BackwardDispatchHint::eNoHints> ();
	}
}

template <class TValueType>
inline void backward(TValueType& root) noexcept
{
	burt::MutableData res_seq_topo;
	burt::MutableData res_set_topo;
	burt::MutableData recursion;
	return backwardWithScratchStorage<TValueType, /*execute_reverse_topo_order*/ true, /*execute_backward*/ true> (root, res_seq_topo, res_set_topo, recursion);
}

template <bool save_vaues, bool save_grads, class TValueType>
inline bool saveToFile(std::initializer_list<TValueType> nodes, const char* filename) noexcept
{
	using TActDataType = typename TValueType::TActDataType;
	using TGradDataType = typename TValueType::TGradDataType;

	constexpr size_t infoSizePerNode_Data = sizeof(nodes.begin()->dataCopy());
	constexpr size_t infoSizePerNode_Grad = sizeof(nodes.begin()->gradCopy());
	constexpr size_t infoSizePerNodeTotal = size_t(save_vaues) * infoSizePerNode_Data + size_t(save_grads) * infoSizePerNode_Grad;

	size_t sz = nodes.size();
	burt::FileSystemHelpers::FileMappingResult mapping_res = burt::FileSystemHelpers::mapFileToMemoryForWrite(filename, sz * (infoSizePerNodeTotal));
	if (!mapping_res.isOk) [[unlikely]]
		return false;

	uint8_t* restrict_ext rawMemory = (uint8_t*)mapping_res.memory;

	if constexpr (save_vaues)
	{
		const TValueType* restrict_ext nodesPtr = nodes.begin();
		const TValueType* restrict_ext endPtr = nodes.end();

		for (; nodesPtr != endPtr; ++nodesPtr)
		{
			*reinterpret_cast<TActDataType*>(rawMemory) = nodesPtr->dataRef();
			rawMemory += infoSizePerNode_Data;
		}
	}

	if constexpr (save_grads)
	{
		const TValueType* restrict_ext nodesPtr = nodes.begin();
		const TValueType* restrict_ext endPtr = nodes.end();

		for (; nodesPtr != endPtr; ++nodesPtr)
		{
			*reinterpret_cast<TGradDataType*>(rawMemory) = nodesPtr->gradRef();
			rawMemory += infoSizePerNode_Grad;
		}
	}

	if (!burt::FileSystemHelpers::unmapFileFromMemory(mapping_res)) [[unlikely]]
		return false;

	return true;
}

template <bool save_vaues, bool save_grads, class TValueType, class TNodeIndexType>
inline bool saveCreatedTensorsToFileHelper(TNodeIndexType first_index, TNodeIndexType last_index, const char* filename) noexcept
{
	constexpr size_t infoSizePerNode_Data = sizeof(TValueType::sysViewMemoryAsNode(&first_index)->dataCopy());
	constexpr size_t infoSizePerNode_Grad = sizeof(TValueType::sysViewMemoryAsNode(&first_index)->gradCopy());

	constexpr size_t infoSizePerNodeTotal = size_t(save_vaues) * infoSizePerNode_Data + size_t(save_grads) * infoSizePerNode_Grad;
	size_t sz = last_index - first_index + 1;
	burt::FileSystemHelpers::FileMappingResult mapping_res = burt::FileSystemHelpers::mapFileToMemoryForWrite(filename, sz * (infoSizePerNodeTotal));

	if (!mapping_res.isOk) [[unlikely]]
		return false;

	uint8_t* restrict_ext rawMemory = (uint8_t*)mapping_res.memory;

	if constexpr (save_vaues)
	{
		auto* first_data_pointer = &(TValueType::sysViewMemoryAsNode(&first_index)->dataRef());
		size_t bytes_to_copy = infoSizePerNode_Data * sz;
		memcpy(rawMemory, first_data_pointer, infoSizePerNode_Data * sz);

		if constexpr (save_grads)
		{
			rawMemory += bytes_to_copy;
		}
	}

	if constexpr (save_grads)
	{
		auto* first_data_pointer = &(TValueType::sysViewMemoryAsNode(&first_index)->gradRef());
		size_t bytes_to_copy = infoSizePerNode_Grad * sz;
		memcpy(rawMemory, first_data_pointer, infoSizePerNode_Grad * sz);
		if constexpr (false)
		{
			rawMemory += bytes_to_copy;
		}
	}
	
	if (!burt::FileSystemHelpers::unmapFileFromMemory(mapping_res)) [[unlikely]]
		return false;
		
	return true;
}


template <bool save_vaues, bool save_grads, class TValueType>
inline bool saveCreatedTensorsToFile(TValueType first, TValueType last, const char* filename) noexcept
{
	auto first_index = first.sysGetRawNodeIndex();
	auto last_index = last.sysGetRawNodeIndex();
	if (last_index < first_index)
		return false;
	return saveCreatedTensorsToFileHelper<save_vaues, save_grads, TValueType> (first_index, last_index, filename);
}

template <bool save_vaues, bool save_grads, class TValueType>
inline bool saveComputeGraphContextToFile(const char* filename) noexcept
{
	auto first_index = typename TValueType::TNodeIndexType(0);
	auto last_index = typename TValueType::TNodeIndexType(TValueType::checkpointForNeurons());

	if (first_index == last_index)
		return true;
	last_index -= 1;
	return saveCreatedTensorsToFileHelper<save_vaues, save_grads, TValueType>(first_index, last_index, filename);
}

template <bool load_vaues, bool load_grads, class TValueType>
inline bool loadFromFile(std::initializer_list<TValueType> nodes, const char* filename) noexcept
{
	using TActDataType = typename TValueType::TActDataType;
	using TGradDataType = typename TValueType::TGradDataType;

	constexpr size_t infoSizePerNode_Data = sizeof(nodes.begin()->dataCopy());
	constexpr size_t infoSizePerNode_Grad = sizeof(nodes.begin()->gradCopy());
	constexpr size_t infoSizePerNodeTotal = size_t(load_vaues) * infoSizePerNode_Data + size_t(load_grads) * infoSizePerNode_Grad;

	size_t sz = nodes.size();
	burt::FileSystemHelpers::FileMappingResult mapping_res = burt::FileSystemHelpers::mapFileToMemory(filename, true /*read only*/, false /*create if not exist*/);

	if (!mapping_res.isOk) [[unlikely]]
		return false;

	if (mapping_res.fileSizeInBytes != infoSizePerNodeTotal * nodes.size()) [[unlikely]]
		return false;
	
	const uint8_t* restrict_ext rawMemory = (const uint8_t*)mapping_res.memory;
	
	if constexpr (load_vaues)
	{
		const TValueType* restrict_ext nodesPtr = nodes.begin();
		const TValueType* restrict_ext endPtr = nodes.end();

		for (; nodesPtr != endPtr; ++nodesPtr)
		{
			const_cast<TActDataType&>(nodesPtr->dataRef()) = *reinterpret_cast<const TActDataType*>(rawMemory);
			rawMemory += infoSizePerNode_Data;
		}
	}
	
	if constexpr (load_grads)
	{
		const TValueType* restrict_ext nodesPtr = nodes.begin();
		const TValueType* restrict_ext endPtr = nodes.end();

		for (; nodesPtr != endPtr; ++nodesPtr)
		{
			const_cast<TGradDataType&>(nodesPtr->gradRef()) = *reinterpret_cast<const TGradDataType*>(rawMemory);
			rawMemory += infoSizePerNode_Grad;
		}
	}
	
	if (!burt::FileSystemHelpers::unmapFileFromMemory(mapping_res)) [[unlikely]]
		return false;
		
	return true;
}

template <bool load_vaues, bool load_grads, class TValueType, class TNodeIndexType>
inline bool loadCreatedTensorsFromFileHelper(TNodeIndexType first_index, TNodeIndexType last_index, const char* filename) noexcept
{
	constexpr size_t infoSizePerNode_Data = sizeof(TValueType::sysViewMemoryAsNode(&first_index)->dataCopy());
	constexpr size_t infoSizePerNode_Grad = sizeof(TValueType::sysViewMemoryAsNode(&first_index)->gradCopy());

	constexpr size_t infoSizePerNodeTotal = size_t(load_vaues) * infoSizePerNode_Data + size_t(load_grads) * infoSizePerNode_Grad;
	size_t sz = last_index - first_index + 1;

	burt::FileSystemHelpers::FileMappingResult mapping_res = burt::FileSystemHelpers::mapFileToMemoryForWrite(filename, sz * (infoSizePerNodeTotal));

	if (!mapping_res.isOk) [[unlikely]]
		return false;
		
	uint8_t* restrict_ext rawMemory = (uint8_t*)mapping_res.memory;
	
	if constexpr (load_vaues)
	{
		auto* first_data_pointer = &(TValueType::sysViewMemoryAsNode(&first_index)->dataRef());
		size_t bytes_to_copy = infoSizePerNode_Data * sz;
		memcpy(first_data_pointer, rawMemory, infoSizePerNode_Data * sz);
		
		if constexpr (load_grads)
		{
			rawMemory += bytes_to_copy;
		}
	}
	
	if constexpr (load_grads)
	{
		auto* first_data_pointer = &(TValueType::sysViewMemoryAsNode(&first_index)->gradRef());
		size_t bytes_to_copy = infoSizePerNode_Grad * sz;
		memcpy(first_data_pointer, rawMemory, infoSizePerNode_Grad * sz);
		
		if constexpr (false)
		{
			rawMemory += bytes_to_copy;
		}
	}
	
	if (!burt::FileSystemHelpers::unmapFileFromMemory(mapping_res)) [[unlikely]]
		return false;

	return true;
}

template <bool load_vaues, bool load_grads, class TValueType>
inline bool loadCreatedTensorsFromFile(TValueType first, TValueType last, const char* filename) noexcept
{
	auto first_index = first.sysGetRawNodeIndex();
	auto last_index = last.sysGetRawNodeIndex();
	if (last_index < first_index)
		return false;
	return loadCreatedTensorsFromFileHelper<load_vaues, load_grads, TValueType> (first_index, last_index, filename);
}

template <bool load_vaues, bool load_grads, class TValueType>
inline bool loadComputeGraphContextFromFile(const char* filename) noexcept
{
	auto first_index = typename TValueType::TNodeIndexType(0);
	auto last_index = typename TValueType::TNodeIndexType(TValueType::checkpointForNeurons());
	
	if (first_index == last_index)
		return true;
	last_index -= 1;
	return loadCreatedTensorsFromFileHelper<load_vaues, load_grads, TValueType>(first_index, last_index, filename);
}
