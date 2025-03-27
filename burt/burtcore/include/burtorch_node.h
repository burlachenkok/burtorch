#pragma once

#include "burt/system/include/PlatformSpecificMacroses.h"

#include "burt/linalg_vectors/include/VectorND_Raw.h"
#include "burt/linalg_vectors/include/LightVectorND.h"

#include "burtcore/include/burtorch_op_types.h"
#include "burtcore/include/burtorch_op_metainfo.h"
#include "burtcore/include/burtorch_array4node.h"
#include "burtcore/include/burtorch_config.h"


#include <initializer_list>
#include <string>
#include <sstream>

#include <stdlib.h>
#include <memory.h>

template <class Value, class Container, BackwardDispatchHint hint>
void backwardDispatch(Value* outNode, Container& inputNodes, OpType opType) noexcept;

enum ValueInitHints : std::uint32_t
{
	eInitHint_Empty                                    = 0x0,
	eInitHint_Promise_Resize_Child_and_InitValue_Later = 0x1 << 0
};

template <class DataType>
class Value
{
public:
	using TValue = Value<DataType>;
	using TActDataType = DataType;
	using TGradDataType = DataType;
	using TDataType = DataType;

#if BURTORCH_NODES_UINT8_INDICIES
	using TNodeIndexType = uint8_t;
#elif BURTORCH_NODES_UINT16_INDICIES
	using TNodeIndexType = uint16_t;
#elif BURTORCH_NODES_UINT32_INDICIES
	using TNodeIndexType = uint32_t;
#elif BURTORCH_NODES_UINT64_INDICIES
	using TNodeIndexType = uint64_t;
#else
	#error please specify used type for nodes indicies
#endif

	static inline TNodeIndexType invalid_node_index = TNodeIndexType(-1);

	using TChildVec = SpecialArray<TNodeIndexType, TNodeIndexType>;
	using TStringType = const char*;

	consteval bool isStringNamesAreSupported()
	{
#if BURTORCH_NODES_LABEL_SUPPORT
		return 1;
#else
		return 0;
#endif
	}

	static Value getConstant(const TActDataType& theValue) noexcept
	{
		TNodeIndexType node_index = reserveOneIndex();

		Value res;
		res.node_index = node_index;
		
		bwdOpDescr[node_index] = createValidOpDescriptorCompileTime<OpType::eLeaf>();
		value[node_index] = theValue;

#if BURTORCH_INIT_GRADS_TO_ZERO
		grad[node_index] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		label[node_index] = TStringType("");
#endif
		children[node_index].sysClearWithErase();
		
		return res;
	}

	Value(const TActDataType& theValue) noexcept
	{		
		auto cur_idx_counter = reserveOneIndex();
		
		node_index = cur_idx_counter;
		bwdOpDescr[cur_idx_counter] = createValidOpDescriptorCompileTime<OpType::eLeaf>();
		value[cur_idx_counter] = theValue;

#if BURTORCH_INIT_GRADS_TO_ZERO
		grad[cur_idx_counter] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		label[cur_idx_counter] = TStringType("");
#endif
		children[cur_idx_counter].sysClearWithErase();
	}

	Value(const TActDataType& theValue, TStringType theLabel) noexcept
	{
		auto cur_idx_counter = reserveOneIndex();

		node_index = cur_idx_counter;
		bwdOpDescr[cur_idx_counter] = createValidOpDescriptorCompileTime<OpType::eLeaf>();
		value[cur_idx_counter] = theValue;

#if BURTORCH_INIT_GRADS_TO_ZERO
		grad[cur_idx_counter] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		label[cur_idx_counter] = theLabel;
#endif
		children[cur_idx_counter].sysClearWithErase();
	}

	template <ValueInitHints hints, OpType theOpType>
	inline static Value sysCreateRawValue() noexcept
	{
		Value res;

		auto cur_idx_counter = reserveOneIndex();

		res.node_index = cur_idx_counter;
		res.bwdOpDescr[cur_idx_counter] = createValidOpDescriptorCompileTime<theOpType>();

		if constexpr (hints & ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later)
		{
		}
		else
		{
			res.value[cur_idx_counter] = TValue();
		}

#if BURTORCH_INIT_GRADS_TO_ZERO
		res.grad[cur_idx_counter] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		res.label[cur_idx_counter] = theLabel;
#endif

		if constexpr (hints & ValueInitHints::eInitHint_Promise_Resize_Child_and_InitValue_Later)
		{
		}
		else
		{
			res.children[cur_idx_counter].sysClearWithErase();
		}

		return res;
	}

	Value(const TActDataType& theValue, OpType theOpType) noexcept
	{
		auto cur_idx_counter = reserveOneIndex();

		node_index = cur_idx_counter;
		bwdOpDescr[cur_idx_counter] = createValidOpDescriptor(theOpType);
		value[cur_idx_counter] = theValue;

#if BURTORCH_INIT_GRADS_TO_ZERO
		grad[cur_idx_counter] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		label[cur_idx_counter] = TStringType("");
#endif
		children[cur_idx_counter].sysClearWithErase();

	}

	Value(const TActDataType& theValue, OpType theOpType, TNodeIndexType theChildrenA) noexcept
	{
		auto cur_idx_counter = reserveOneIndex();

		node_index = cur_idx_counter;
		bwdOpDescr[cur_idx_counter] = createValidOpDescriptor(theOpType);
		value[cur_idx_counter] = theValue;

#if BURTORCH_INIT_GRADS_TO_ZERO
		grad[cur_idx_counter] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		label[cur_idx_counter] = TStringType("");
#endif

		children[cur_idx_counter] = TChildVec(theChildrenA);
	}

	Value(const TActDataType& theValue, OpType theOpType, TNodeIndexType theChildrenA, TNodeIndexType theChildrenB) noexcept
	{
		auto cur_idx_counter = reserveOneIndex();

		node_index = cur_idx_counter;
		bwdOpDescr[cur_idx_counter] = createValidOpDescriptor(theOpType);
		value[cur_idx_counter] = theValue;

#if BURTORCH_INIT_GRADS_TO_ZERO
		grad[cur_idx_counter] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		label[cur_idx_counter] = TStringType("");
#endif

		children[cur_idx_counter] = TChildVec(theChildrenA, theChildrenB);
	}

	Value(const TActDataType& theValue, OpType theOpType, std::initializer_list<TNodeIndexType> theChildrenIndicies) noexcept
	{
		auto cur_idx_counter = reserveOneIndex();

		node_index = cur_idx_counter;
		bwdOpDescr[cur_idx_counter] = createValidOpDescriptor(theOpType);
		value[cur_idx_counter] = theValue;

#if BURTORCH_INIT_GRADS_TO_ZERO
		grad[cur_idx_counter] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		label[cur_idx_counter] = TStringType("");
#endif

		children[cur_idx_counter] = TChildVec(theChildrenIndicies);
	}


	Value(const TActDataType& theValue, 
		  OpType theOpType, 
		  TStringType theLabel, 
		  std::initializer_list<TNodeIndexType> theChildrenIndicies) noexcept
	{
		auto cur_idx_counter = reserveOneIndex();

		node_index = cur_idx_counter;
		bwdOpDescr[cur_idx_counter] = createValidOpDescriptor(theOpType);
		value[cur_idx_counter] = theValue;

#if BURTORCH_INIT_GRADS_TO_ZERO
		grad[cur_idx_counter] = TGradDataType();
#endif

#if BURTORCH_NODES_LABEL_SUPPORT
		label[cur_idx_counter] = theLabel;
#endif

		children[cur_idx_counter] = TChildVec(theChildrenIndicies);
	}

	template <size_t kItems>
	static void sysCreateLightView(Value* dst, const Value* src) noexcept
	{
		static_assert(sizeof(Value) == sizeof(dst->node_index));
		static_assert(sizeof(Value) == sizeof(dst->node_index));

		TNodeIndexType* dst_node_index = &(dst->node_index);
		const TNodeIndexType* src_nodes_index = &(src->node_index);

		for (size_t i = 0; i < kItems; ++i)
			dst_node_index[i] = src_nodes_index[i];
	}

	template <size_t kItems>
	static void sysInvalidateLightView(Value* dst) noexcept
	{
		static_assert(sizeof(Value) == sizeof(dst->node_index));

		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{
			TNodeIndexType* dst_node_index = &(dst->node_index);
			for (size_t i = 0; i < kItems; ++i)
				dst_node_index[i] = invalid_node_index;
		}
	}

	static void sysDestructManually(Value* dst, size_t kItems) noexcept
	{
		static_assert(sizeof(Value) == sizeof(dst->node_index));

		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{
			TNodeIndexType* dst_node_index = &(dst->node_index);
			memset(dst_node_index, 0xff, kItems * sizeof(Value));
		}
	}

	Value(const Value& rhs) noexcept
	{
		node_index = rhs.node_index;

		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{
			bwdOpDescr[node_index].node_gc_counter += 1;
		}
	}

	Value(Value&& rhs) noexcept
	{
		burt_assert(this != &rhs);

		node_index = rhs.node_index;

		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{
			rhs.node_index = invalid_node_index;
		}
	}

	Value& operator = (const Value& rhs) noexcept
	{
		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{
			if (node_index != invalid_node_index)
			{
				bwdOpDescr[node_index].node_gc_counter -= 1;
			}
		}

		node_index = rhs.node_index;

		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{
			burt_assert(isValid());
			bwdOpDescr[node_index].node_gc_counter += 1;
		}
		return *this;
	}

	Value& operator = (Value&& rhs) noexcept
	{
		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{
			burt_assert(node_index != rhs.node_index);

			if (node_index != invalid_node_index)
			{
				bwdOpDescr[node_index].node_gc_counter -= 1;
			}
		}

		node_index = rhs.node_index;
		
		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{
			burt_assert(isValid());
			rhs.node_index = invalid_node_index;
		}

		return *this;
	}
	
	constexpr bool isValid() noexcept
	{
		return node_index != invalid_node_index;
	}

	~Value() noexcept
	{
		if constexpr (BURTORCH_USE_GC_COUNTERS)
		{			
			if (node_index == invalid_node_index)
			{
				return;
			}
			else
			{
				burt_assert(bwdOpDescr[node_index].node_gc_counter > 0);
				bwdOpDescr[node_index].node_gc_counter -= 1;
			}
		}
	}

    std::string asString() const noexcept
	{
		std::ostringstream s;
		s << "Value(data=" << value[node_index] << ")";
		s << getHelpString();
		s << ' ';
		s << '{';

        size_t children_num = childrenNum();
        for (size_t i = 0; i < children_num; ++i)
		{
            auto cindex = children[node_index].get(i);
            s << "child_value=" << value[cindex];
			s << ' ';
		}
		s << '}';

		return s.str();
	}

    operator std::string() const noexcept {
		return asString();
	}

	forceinline_ext constexpr const TActDataType& dataRef() const noexcept {
		return value[node_index];
	}

	forceinline_ext constexpr TActDataType& dataRef() noexcept {
		return value[node_index];
	}
	forceinline_ext constexpr TActDataType dataCopy() const noexcept {
		return value[node_index];
	}
	forceinline_ext void resetData() noexcept {
		value[node_index] = TActDataType();
	}

	forceinline_ext constexpr const TGradDataType& gradRef() const noexcept {
		return grad[node_index];
	}
	forceinline_ext constexpr TGradDataType gradCopy() const noexcept {
		return grad[node_index];
	}
	forceinline_ext constexpr void setGrad(const TGradDataType& gradValue) noexcept {
		grad[node_index] = gradValue;
	}
	forceinline_ext constexpr void setGradToOne() noexcept {
		grad[node_index] = TGradDataType(1);
	}
	forceinline_ext constexpr void setGradToZero() noexcept {
		grad[node_index] = TGradDataType();
	}
	forceinline_ext constexpr void resetGrad() noexcept {
		grad[node_index] = TGradDataType();
	}


	forceinline_ext void addToGrad(const TGradDataType& gradValue) noexcept {
		grad[node_index] += gradValue;
	}

	forceinline_ext void subFromGrad(const TGradDataType& gradValue) noexcept {
		grad[node_index] -= gradValue;
	}

	forceinline_ext const TChildVec& childrenSet() const noexcept {
		return children[node_index];
	}

	forceinline_ext constexpr bool childrenSetIsEmpty() const noexcept {
		return children[node_index].size() == 0;
	}
	
	forceinline_ext constexpr size_t childrenNum() const noexcept {
		return children[node_index].size();
	}

	forceinline_ext constexpr bool hasChildren() const noexcept {
		return !children[node_index].isEmpty();
	}

	forceinline_ext constexpr bool isLeaf() const noexcept {
		return children[node_index].isEmpty();
	}

	forceinline_ext void addChild(const TValue* childNode) noexcept {
		children[node_index].push_back(childNode->sysGetRawNodeIndex());
	}

	forceinline_ext void addTwoChild(const TValue* childNodeA, const TValue* childNodeB) noexcept {
		children[node_index].push_back_two_items(childNodeA->sysGetRawNodeIndex(), childNodeB->sysGetRawNodeIndex());
	}

	forceinline_ext void setLabel(TStringType theLabel) noexcept
	{
#if BURTORCH_NODES_LABEL_SUPPORT
		label[node_index] = theLabel;
#else
		return;
#endif
	}

	forceinline_ext constexpr std::string_view getLabel() const noexcept
	{
#if BURTORCH_NODES_LABEL_SUPPORT
		return std::string_view(label[node_index]);
#else
		return std::string_view();
#endif
	}

	forceinline_ext constexpr const char* getHelpString() const noexcept
	{
		return opTypeToString((OpType)(bwdOpDescr[node_index].op_type));
	}

	template <uint32_t hint = BackwardDispatchHint::eNoHints>
	forceinline_ext void backward() noexcept {
		backwardDispatch<Value, decltype(children[node_index]), hint>(this, children[node_index], (OpType)bwdOpDescr[node_index].op_type);
	}


	void setupBackwardFuncType(OpType theOperationType) noexcept
	{
		bwdOpDescr[node_index].op_type = (unsigned int)theOperationType;
	}

	forceinline_ext constexpr unsigned int backwardOptVisitNumberForBackpropCopy() const noexcept
	{
		return bwdOpDescr[node_index].visiting_number_for_backprop;
	}

	forceinline_ext void backwardOptVisitNumberSet(unsigned int counter) noexcept {
		bwdOpDescr[node_index].visiting_number_for_backprop = counter;
	}

	forceinline_ext static void reserveMemoryForNodes(size_t expectedTotalNumber) noexcept
	{
		// Only compile-time checks
		//==============================================================================
		#if	BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META
			// check that high bit is still not used
			constexpr size_t mask = (0x1 << ((sizeof(idx_counter) * 8) - 1));
            burt_assert( (expectedTotalNumber & mask) == 0);
		#endif

		// check that there is no overflow with indicies
        burt_assert(expectedTotalNumber <= size_t(TNodeIndexType(-1)));
		//==============================================================================
		// Redirect to resizeArrayToSatisfyNewSize
		resizeArrayToSatisfyNewSize(expectedTotalNumber);
	}

	forceinline_ext static void deactiveUnusedNodes() noexcept 
	{

#if BURTORCH_USE_GC_COUNTERS
        burt_assert(idx_counter == ctrSize());
		size_t new_compressed_size = ctrSize();
		
		for (; new_compressed_size != 0; --new_compressed_size)
		{
			if (bwdOpDescr[new_compressed_size - 1].node_gc_counter == 0)
				continue;
			else
				break;
		}
		resizeArrayToSatisfyNewSize((TNodeIndexType)new_compressed_size);
		idx_counter = (TNodeIndexType)new_compressed_size;
#endif
		
		return;
	}

    inline static void* allocateBytes(size_t sz) noexcept
    {
		burt_assert(sz > 0);

#if 0
            return malloc(sz);
#else
        #if BURT_WINDOWS
            {
				constexpr size_t kCacheLizeSizeInBytes = 64;
				burt_assert(sz % kCacheLizeSizeInBytes == 0);
				return _aligned_malloc(sz, kCacheLizeSizeInBytes);
            }
        #else
            {
				constexpr size_t kCacheLizeSizeInBytes = 64;
				burt_assert(sz % kCacheLizeSizeInBytes == 0);
                return aligned_alloc(kCacheLizeSizeInBytes, sz);
            }
        #endif
#endif
    }

    inline static void deallocateBytes(void* ptr) noexcept
    {
        #if 0
            free(ptr);             // // If ptr is a null pointer, the function does nothing.
        #else
            #if BURT_WINDOWS
                _aligned_free(ptr); //  If memblock is a NULL pointer, this function simply performs no actions according to MSVC docs
            #else
                free(ptr);          // If ptr is a null pointer, the function does nothing.
            #endif
        #endif
    }


    inline static void resizeArrayToSatisfyNewSize(size_t new_size) noexcept
	{
		if (ctrReservedMemory() >= new_size)
			return;

		size_t old_num_items = idx_counter;
		size_t new_num_items = new_size;

        constexpr size_t kCacheLizeSizeInBytes = 64;
        ctrs_reserved_mem_in_items = burt::roundToNearestMultipleUp<kCacheLizeSizeInBytes>(new_num_items * 2);

		{
			static_assert(std::is_trivially_copyable<OperationDescriptor>::value);
            OperationDescriptor* bwdOpDescrNew = (OperationDescriptor*) allocateBytes(ctrs_reserved_mem_in_items * sizeof(bwdOpDescrNew[0]));
			memcpy(bwdOpDescrNew, bwdOpDescr, old_num_items * sizeof(bwdOpDescrNew[0]));
			//memset(bwdOpDescrNew + old_num_items, 0, (ctrs_reserved_mem_in_items - old_num_items) * sizeof(bwdOpDescrNew[0]));
            deallocateBytes(bwdOpDescr);
			bwdOpDescr = bwdOpDescrNew;
		}
#if BURTORCH_NODES_LABEL_SUPPORT
		{
			static_assert(std::is_trivially_copyable<TStringType>::value);
            TStringType* labelNew = (TStringType*) allocateBytes(ctrs_reserved_mem_in_items * sizeof(labelNew[0]));
			memcpy(labelNew, label, old_num_items * sizeof(labelNew[0]));
			//memset(labelNew + old_num_items, 0, (ctrs_reserved_mem_in_items - old_num_items) * sizeof(labelNew[0]));
            deallocateBytes(label);
			label = labelNew;
		}
#endif

		{
			// Exploit information that really TChildVec are fundamental types + raw poiters
            TChildVec* childrenNew = (TChildVec*)allocateBytes(ctrs_reserved_mem_in_items * sizeof(childrenNew[0]));
			memcpy(childrenNew, children, old_num_items * sizeof(childrenNew[0]));
#if 1
			{
				// Manual reset number of children to zero from non-init part -- no dtor no move is needed, it's just raw memory
                TChildVec* childToResetBegin = childrenNew + old_num_items;
                TChildVec* childToResetEnd = childrenNew + ctrs_reserved_mem_in_items;
				for (; childToResetBegin != childToResetEnd; ++childToResetBegin)
                    childToResetBegin->sysInitToDefault();
			}
#else
			{
				// A bit dangerous, but fine: exploits internal representation of nodes and size_tag. If size_tag is zero it does not matter.			
				memset(childrenNew + old_num_items,
		   			   0,
					   (ctrs_reserved_mem_in_items - old_num_items) * sizeof(childrenNew[0])
				       );
		    }
#endif
            deallocateBytes(children);
			children = childrenNew;
		}

		{
			//static_assert(std::is_trivially_copyable<TActDataType>::value);
            TActDataType* valueNew = (TActDataType*) allocateBytes(ctrs_reserved_mem_in_items * sizeof(valueNew[0]));
			memcpy(valueNew, value, old_num_items * sizeof(valueNew[0]));
			//memset(valueNew + old_num_items, 0, (ctrs_reserved_mem_in_items - old_num_items) * sizeof(valueNew[0]));
            deallocateBytes(value);
			value = valueNew;
		}

		{
			//static_assert(std::is_trivially_copyable<TGradDataType>::value);
            TGradDataType* gradNew = (TGradDataType*) allocateBytes(ctrs_reserved_mem_in_items * sizeof(gradNew[0]));
			memcpy(gradNew, grad, old_num_items * sizeof(gradNew[0]));
			//memset(gradNew + old_num_items, 0, (ctrs_reserved_mem_in_items - old_num_items) * sizeof(gradNew[0]));
            deallocateBytes(grad);
			grad = gradNew;
		}
	}

	forceinline_ext static TNodeIndexType checkpointForNeurons() noexcept
	{
		// the place where next neuron will be placed
		return idx_counter;
	}

	forceinline_ext static void restoreCheckpoint(TNodeIndexType ckeckpoint) noexcept
	{
        burt_assert(idx_counter == ctrSize());
		resizeArrayToSatisfyNewSize(ckeckpoint);
		idx_counter = ckeckpoint;
	}

	forceinline_ext static void setGradToZeroFrom(TNodeIndexType fromCkeckpoint) noexcept
	{
		// the place where next neuron will be placed
        burt_assert(fromCkeckpoint <= idx_counter);
		memset(&grad[fromCkeckpoint], 0, (idx_counter - fromCkeckpoint) * sizeof(grad[0]));
	}

	/** Clean gradients for neurons in interval [startCkeckpoint, endCkeckpoint)
	*/
	forceinline_ext static void setGradToZeroIn(TNodeIndexType startCkeckpoint, TNodeIndexType endCkeckpoint) noexcept
	{
		burt_assert(startCkeckpoint <= endCkeckpoint);
		memset(&grad[startCkeckpoint], 0, (endCkeckpoint - startCkeckpoint) * sizeof(grad[0]));
	}

	forceinline_ext static TGradDataType computeGradL2NormSquare(TNodeIndexType startCkeckpoint, TNodeIndexType endCkeckpoint, TGradDataType oneInvProcessedSamples) noexcept
	{
		burt_assert(startCkeckpoint <= endCkeckpoint);
		TGradDataType res = TGradDataType();
		for (TNodeIndexType i = startCkeckpoint; i < endCkeckpoint; ++i)
		{
			auto add_ = grad[i] * grad[i];
			res += add_;
		}
		return res * (oneInvProcessedSamples * oneInvProcessedSamples);
	}

	forceinline_ext static void applyGDStep(TNodeIndexType startCkeckpoint, TNodeIndexType endCkeckpoint, TGradDataType oneInvProcessedSamplesTimesLearningRate) noexcept
	{
		burt_assert(startCkeckpoint <= endCkeckpoint);
		for (TNodeIndexType i = startCkeckpoint; i < endCkeckpoint; ++i)
		{
			value[i] -= (oneInvProcessedSamplesTimesLearningRate * grad[i]);
		}
		return;
	}
	
	forceinline_ext static TGradDataType applyGDStepAndComputeGradL2NormSquareWithSIMD(TNodeIndexType startCkeckpoint, TNodeIndexType endCkeckpoint, 
																					   TGradDataType oneInvProcessedSamples, TGradDataType lr) noexcept
	{
		const TGradDataType oneInvProcessedSamplesTimesLearningRate = oneInvProcessedSamples * lr;
		burt_assert(startCkeckpoint <= endCkeckpoint);

		const size_t items = endCkeckpoint - startCkeckpoint;

		burt::LightVectorND<burt::VectorNDRaw<TGradDataType>> light_vec_data(&value[startCkeckpoint], items);
		burt::LightVectorND<burt::VectorNDRaw<TGradDataType>> light_vec_grad(&grad[startCkeckpoint], items);

		auto grad_l2_norm_unnormalized = light_vec_data.subInPlaceVectorWithMultipleAndReportL2NormSqr(oneInvProcessedSamples * lr, light_vec_grad);
		return grad_l2_norm_unnormalized * (oneInvProcessedSamples * oneInvProcessedSamples);
	}

	forceinline_ext static void applyGDStepWithSIMD(TNodeIndexType startCkeckpoint, TNodeIndexType endCkeckpoint, TGradDataType oneInvProcessedSamples, TGradDataType lr) noexcept
	{
		const TGradDataType oneInvProcessedSamplesTimesLearningRate = oneInvProcessedSamples * lr;
		burt_assert(startCkeckpoint <= endCkeckpoint);

		const size_t items = endCkeckpoint - startCkeckpoint;

		burt::LightVectorND<burt::VectorNDRaw<TGradDataType>> light_vec_data(&value[startCkeckpoint], items);
		burt::LightVectorND<burt::VectorNDRaw<TGradDataType>> light_vec_grad(&grad[startCkeckpoint], items);

		light_vec_data.subInPlaceVectorWithMultiple(oneInvProcessedSamples * lr, light_vec_grad);
	}


	forceinline_ext static void applyGDStep(TNodeIndexType startCkeckpoint, TNodeIndexType endCkeckpoint, TGradDataType oneInvProcessedSamples, TGradDataType lr) noexcept
	{
		const TGradDataType oneInvProcessedSamplesTimesLearningRate = oneInvProcessedSamples * lr;
		burt_assert(startCkeckpoint <= endCkeckpoint);

		for (TNodeIndexType i = startCkeckpoint; i < endCkeckpoint; ++i)
		{
			value[i] -= (oneInvProcessedSamplesTimesLearningRate * grad[i]);
		}
	}

	forceinline_ext static TGradDataType applyGDStepAndComputeGradL2NormSquare(TNodeIndexType startCkeckpoint, TNodeIndexType endCkeckpoint, 
																			   TGradDataType oneInvProcessedSamples, TGradDataType lr) noexcept
	{
		const TGradDataType oneInvProcessedSamplesTimesLearningRate = oneInvProcessedSamples * lr;
		burt_assert(startCkeckpoint <= endCkeckpoint);
		
		TGradDataType res = TGradDataType();

#if 0
		size_t items = endCkeckpoint - startCkeckpoint;

		size_t packed_sz_by_16 = burt::roundToNearestMultipleDown<16>(items);
		size_t packed_sz_by_8 = burt::roundToNearestMultipleDown<8>(items);
		size_t packed_sz_by_4 = burt::roundToNearestMultipleDown<4>(items);
		size_t packed_sz_by_2 = burt::roundToNearestMultipleDown<2>(items);


		TGradDataType* restrict_ext grad_ptr  = &grad[startCkeckpoint];
		TActDataType* restrict_ext value_ptr  = &value[startCkeckpoint];
		size_t i = 0;

		for (; i < packed_sz_by_16; i += 16, grad_ptr += 16, value_ptr += 16)
		{
			for (size_t j = 0; j < 16; ++j)
			{
				res += grad_ptr[j] * grad_ptr[j];
				value_ptr[j] -= (oneInvProcessedSamplesTimesLearningRate * grad_ptr[j]);
			}
		}

		for (; i < packed_sz_by_8; i += 8, grad_ptr += 8, value_ptr += 8)
		{
			for (size_t j = 0; j < 8; ++j)
			{
				res += grad_ptr[j] * grad_ptr[j];
				value_ptr[j] -= (oneInvProcessedSamplesTimesLearningRate * grad_ptr[j]);
			}
		}

		for (; i < packed_sz_by_4; i += 4, grad_ptr += 4, value_ptr += 4)
		{
			for (size_t j = 0; j < 4; ++j)
			{
				res += grad_ptr[j] * grad_ptr[j];
				value_ptr[j] -= (oneInvProcessedSamplesTimesLearningRate * grad_ptr[j]);
			}
		}

		for (; i < packed_sz_by_2; i += 2, grad_ptr += 2, value_ptr += 2)
		{
			for (size_t j = 0; j < 2; ++j)
			{
				res += grad_ptr[j] * grad_ptr[j];
				value_ptr[j] -= (oneInvProcessedSamplesTimesLearningRate * grad_ptr[j]);
			}
		}

		for (; i < items; i++, grad_ptr++, value_ptr++)
		{
			res += grad_ptr[0] * grad_ptr[0];
			value_ptr[0] -= (oneInvProcessedSamplesTimesLearningRate * grad_ptr[0]);
		}

		#else
			for (TNodeIndexType i = startCkeckpoint; i < endCkeckpoint; ++i)
			{
				res += grad[i] * grad[i];
				value[i] -= (oneInvProcessedSamplesTimesLearningRate * grad[i]);
			}
		#endif

		return res * (oneInvProcessedSamples * oneInvProcessedSamples);
	}

	template<class TCtr>
	forceinline_ext static bool isSequentialIndicies(const TCtr& values)
	{
		if (values.size() <= 1)
			return true;

		auto base_index = values[0].sysGetRawNodeIndex();

		for (size_t i = 1; i < values.size(); ++i)
		{
			auto cur_index = values[i].sysGetRawNodeIndex();

			if (base_index + i != cur_index)
			{
				return false;
			}
		}

		return true;
	}

	forceinline_ext static void cleanupBackpropCounters() noexcept
	{
		;
//		memset(visitingNumberForBackprop.data(), 
//			   visitingNumberForBackprop.size() * sizeof(visitingNumberForBackprop[0]), 
//			   0);
	}

	forceinline_ext static void deactiveAllNodes() noexcept
	{
		restoreCheckpoint(0);
	}

	forceinline_ext static constexpr TNodeIndexType numActiveNodes() noexcept
	{
        burt_assert(idx_counter == ctrSize());
		return idx_counter;
	}

	forceinline_ext TChildVec& sysChildrenSet() noexcept {
		return children[node_index];
	}

	constexpr TNodeIndexType sysGetRawNodeIndex() const {
		return node_index;
	}

	const_func_ext static constexpr Value* sysViewMemoryAsNode(TNodeIndexType* index) noexcept
	{
		static_assert(sizeof(Value) == sizeof(TNodeIndexType));
		Value* res = reinterpret_cast<Value*>(index);
		return res;
	}
	
	const_func_ext static constexpr const Value* sysViewMemoryAsNode(const TNodeIndexType* index) noexcept
	{
		static_assert(sizeof(Value) == sizeof(TNodeIndexType));
		const Value* res = reinterpret_cast<const Value*>(index);
		return res;
	}

	Value() noexcept
	: node_index(invalid_node_index)
	{
	}

	struct MemoryStatistics
	{
		size_t labelsMemory;
		size_t bwdOpDescrMemory;
		size_t childrenTopologyMemory;
		size_t auxChildrenTopologyMemory;
		size_t activationsMemory;
		size_t gradsMemory;
	};

	struct NodeStatistics
	{
		bool node_types[255];
		size_t node_types_counts[255];
		size_t total_number_of_nodes;
		size_t min_number_of_children;
		size_t max_number_of_children;
		size_t avg_number_of_children;
	};

	struct RuntimeConfiguration
	{
		bool IS_BURTORCH_NODES_LABEL_SUPPORT;
		bool IS_BURTORCH_NODES_UINT8_INDICIES;
		bool IS_BURTORCH_NODES_UINT16_INDICIES;
		bool IS_BURTORCH_NODES_UINT32_INDICIES;
		bool IS_BURTORCH_NODES_UINT64_INDICIES;
		bool IS_BURTORCH_USE_GC_COUNTERS;
		bool IS_BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META;
		bool IS_BURTORCH_INIT_GRADS_TO_ZERO;
	};

	struct Statistics
	{
		MemoryStatistics occupied_memory;
		MemoryStatistics reserved_memory;

		NodeStatistics alive_nodes_info;
		NodeStatistics all_nodes_info;

		size_t total_numer_of_currently_used_nodes;
		size_t total_numer_of_currently_reserved_nodes;

		RuntimeConfiguration runtime_cfg;
	};

	forceinline_ext static void sysCollectStatistics(Statistics& stats) noexcept
	{
		{
			#if BURTORCH_NODES_LABEL_SUPPORT
				stats.occupied_memory.labelsMemory = ctrSize() * sizeof(label[0]);
			#else
				stats.occupied_memory.labelsMemory = 0;
			#endif

			stats.occupied_memory.bwdOpDescrMemory = ctrSize() * sizeof(bwdOpDescr[0]);
			stats.occupied_memory.childrenTopologyMemory = ctrSize() * sizeof(children[0]);
			stats.occupied_memory.activationsMemory = ctrSize() * sizeof(value[0]);
			stats.occupied_memory.gradsMemory = ctrSize() * sizeof(grad[0]);
		}

		{
			#if BURTORCH_NODES_LABEL_SUPPORT
				stats.reserved_memory.labelsMemory = ctrReservedMemory() * sizeof(label[0]);
			#else
				stats.reserved_memory.labelsMemory = 0;
			#endif

			stats.reserved_memory.bwdOpDescrMemory = ctrReservedMemory() * sizeof(bwdOpDescr[0]);
			stats.reserved_memory.childrenTopologyMemory = ctrReservedMemory() * sizeof(children[0]);
			stats.reserved_memory.activationsMemory = ctrReservedMemory() * sizeof(value[0]);
			stats.reserved_memory.gradsMemory = ctrReservedMemory() * sizeof(grad[0]);
		}

		{
			stats.total_numer_of_currently_used_nodes = idx_counter;
			stats.total_numer_of_currently_reserved_nodes = ctrs_reserved_mem_in_items;
		}

		{
			stats.runtime_cfg.IS_BURTORCH_NODES_LABEL_SUPPORT = BURTORCH_NODES_LABEL_SUPPORT;
			stats.runtime_cfg.IS_BURTORCH_NODES_UINT8_INDICIES = BURTORCH_NODES_UINT8_INDICIES;
			stats.runtime_cfg.IS_BURTORCH_NODES_UINT16_INDICIES = BURTORCH_NODES_UINT16_INDICIES;
			stats.runtime_cfg.IS_BURTORCH_NODES_UINT32_INDICIES = BURTORCH_NODES_UINT32_INDICIES;
			stats.runtime_cfg.IS_BURTORCH_NODES_UINT64_INDICIES = BURTORCH_NODES_UINT64_INDICIES;
			stats.runtime_cfg.IS_BURTORCH_USE_GC_COUNTERS = BURTORCH_USE_GC_COUNTERS;
			stats.runtime_cfg.IS_BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META = BURTORCH_ALLOW_USE_IND_HIGHBIT_FOR_META;
			stats.runtime_cfg.IS_BURTORCH_INIT_GRADS_TO_ZERO = BURTORCH_INIT_GRADS_TO_ZERO;
		}
		return;
	}

    inline static void cleanFull() noexcept
    {
        static_assert(std::is_trivially_copyable<OperationDescriptor>::value);
        static_assert(std::is_trivially_copyable<TStringType>::value);
        //static_assert(std::is_trivially_copyable<TActDataType>::value);
        //static_assert(std::is_trivially_copyable<TGradDataType>::value);

        {
            for (size_t i = 0; i < ctrs_reserved_mem_in_items; ++i)
                children[i].~TChildVec();
        }

#if BURTORCH_NODES_LABEL_SUPPORT
        deallocateBytes(label);
#endif
        deallocateBytes(bwdOpDescr);
        deallocateBytes(children);
        deallocateBytes(value);
        deallocateBytes(grad);

#if BURTORCH_NODES_LABEL_SUPPORT
        label = nullptr;
#endif
        idx_counter = 0;
        ctrs_reserved_mem_in_items = 0;

        bwdOpDescr = nullptr;
        children = nullptr;
        value = nullptr;
        grad = nullptr;
    }
private:

	inline static TNodeIndexType reserveOneIndex()
	{
		reserveMemoryForNodes(idx_counter + 1);
		return idx_counter++;
	}

    inline static TNodeIndexType ctrSize() noexcept
    {
		return idx_counter;
	}

    inline static TNodeIndexType ctrReservedMemory() noexcept
    {
		return ctrs_reserved_mem_in_items;
	}


	BURTORCH_INTERNAL_STATIC_STORAGE inline static BURTORCH_INDEX_WRAPPER(TNodeIndexType) idx_counter = 0;                              ///< Global index counter for nodes of specific type

	BURTORCH_INTERNAL_STATIC_STORAGE inline static TNodeIndexType ctrs_reserved_mem_in_items = 0;               ///< Global index counter for nodes of specific type

#if BURTORCH_NODES_LABEL_SUPPORT
	BURTORCH_INTERNAL_STATIC_STORAGE inline static TStringType* label;                                          ///< Label per node
#endif

	BURTORCH_INTERNAL_STATIC_STORAGE inline static OperationDescriptor* bwdOpDescr = nullptr;                 ///< Backward operation type per node
																             // | POSSIBLE EXTENSIONS: eliminate visitingNumberForBackprop -- add flag visitingNumberForBackprop into NODE

	BURTORCH_INTERNAL_STATIC_STORAGE inline static TChildVec* children = nullptr;                              ///< Connection topology. direct children.

	BURTORCH_INTERNAL_STATIC_STORAGE inline static TActDataType* value = nullptr;                              ///< Activation values per node

	BURTORCH_INTERNAL_STATIC_STORAGE inline static TGradDataType* grad = nullptr;                              ///< Gradient values per node

private:

	TNodeIndexType node_index;

	//TChildVec children;                 ///< Children of compute node. They come also as backward arguments.... If topology is kind the same of connection then can be store effectively index mappings... [good]

	//String label;                      ///< Label for compute node. Primal goal is to have user-defined name of the node. [not needed during backprop really -- better to store separately in specific map]
									     /// However for map potenially nice to have for 4B model paramaters 32-bit index

	//TDataType value;                   ///< Value (activation)...for better memory ops it can be nice to store things separately in something like activation buffer...

	//TDataType grad;                    ///< Accumulated partial derivative of final node value w.r.t. to this variable/node...for better memory ops it can be nice to store things separately in something like grad buffer..
	
	//size_t visitingNumberForBackprop; ///< Number of visits during backprop [possibly just binary counter -- 1 bit]

	// node_index is pure TNodeIndexType index to navigate between nodes and children, copying of nodes is shallow copying
	//    OPEN QUESTION: reference counts to invalidate handles. Should be used or not?
	//             CHOICE-1: make copy with ref.counts
	//             CHOICE-2: all copies are soft copies. invalidation of handles is separate process
	//     BETTER SUPPORT BOTH STYLES
};
