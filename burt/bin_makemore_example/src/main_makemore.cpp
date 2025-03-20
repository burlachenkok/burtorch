#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/system/include/threads/Thread.h"

#include "burt/timers/include/HighPrecisionTimer.h"
#include "burt/fs/include/FileSystemHelpers.h"
#include "burt/copylocal/include/Data.h"

#include "burt/random/include/RandomGenRealLinear.h"
#include "burt/random/include/RandomGenIntegerLinear.h"
#include "burt/random/include/Shuffle.h"

#include "burtcore/include/burtorch.h"

#include <vector>
#include <array>
#include <string_view>
#include <string>
#include <map>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <stddef.h>
#include <assert.h>

namespace
{
    template<size_t sz_in_bytes>
    void my_memset_to_zero(void* dst)
    {
        memset(dst, 0, sz_in_bytes);
    }

    inline std::ostream& my_log_stream() {
        return std::cout;
    }
}

int main(int argc, char** argv)
{
    constexpr bool kMakePauseAtEnd = true;
    burt::DefaultThread::setThreadAffinityMaskForCurrentTh(0x1 << 1);

    if (argc == 1)
    {
        std::cout << "Please specify name of input dataset as first argument (like exp2_names.txt)";
        return -1;
    }    
    const char* fname_open = argv[1];

    Value<float>::deactiveUnusedNodes();
    auto nodes_at_start = Value<float>::numActiveNodes();

    {
        constexpr bool kDebugPrint = false;                                                     // debug printing
        using TChar = char;                                                                     // character type
        using TTokenIndex = uint8_t;                                                            // type to index indidual tokens
        constexpr TChar sysToken = TChar('.');                                                  // token printable representation for <START> <END> <PAD>
    
        constexpr size_t kBlockSize = 16;                                                       // context length (sometimes denote as block length) how many chars we take to predict next one (16 in makemore). How many previous characters we used to predict the next one.
        typedef TTokenIndex index_type;                                                         // index type to index individual characters
        typedef float emb_type;                                                                 // type of element for single embeded item
        constexpr size_t kEmbeddingSize = 64;                                                   // embedded vector dimension to encode invidual characters in space emb_type**64
        
        //constexpr size_t kBatchSize = 64;                                                    // batch size for one iteration [TUNE]
        constexpr size_t kBatchSize = 64;                                                   // batch size for one iteration [TUNE]
        constexpr size_t hidden_dim = 1024;                                                       // model specific hidden layer dimension [TUNE]

        constexpr bool kPrintDetailedInfo = !true;                                              // print detailed information
        constexpr size_t kIterationsPrintFreq = 10;                                             // print frequency for memory reports

        constexpr size_t kTotalIterations = 4000;                                                 // fixed number of steps/iterations to process batches, where one iteration -- process one batch

        constexpr bool kUsedSIMD4Compute = !true;                                               // used CPU SIMD
#if 0
        ->Train: [4000 / 4000] | Loss : 3.002 | Grad l2 norm sqr : -1 | Avg Time : 13.4881 msec.
            | Std.dev for time : 0.62586 msec.
            | Params : 136411

            Total computate and parse time : 54549.2 msec.

            ~batch : 64
            ~hidden dim : 128
#endif

        burt::HighPrecisionTimer timer_main;
        my_log_stream() << "step-1: read file for character based generation model from: " << fname_open << '\n';
        burt::FileSystemHelpers::FileMappingResult names_files = burt::FileSystemHelpers::mapFileToMemory(fname_open, true);
        if (!names_files.isOk)
        {
            my_log_stream() << "File can not be opened: " << fname_open << '\n';
            my_log_stream() << "Error message: " << names_files.errorMsg << '\n';
            return -1;
        }

        // We read everything not in the list, but put words as <pointer start, length> string view tuples.
        burt::Data trainDatasetRaw(names_files.memory, 
                                   names_files.memorySizeInBytes, 
                                   burt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);

        std::vector<std::string_view> words;
        while (!trainDatasetRaw.isEmpty())
        {
            words.push_back(trainDatasetRaw.getStringView());
            assert(!words.back().empty());
        }
        my_log_stream() << "in file: " << fname_open << '\n';
        my_log_stream() << "in fsize in bytes: " << names_files.fileSizeInBytes << '\n';
        my_log_stream() << "in words: " << words.size() << '\n';

        my_log_stream() << "step-2: build symbols map\n";
        size_t totalCharacters = 1;  // total number of unique characters. zero-index reserved for <START> <END> <PAD> token.
        TTokenIndex stoi[256] = {0}; // character => index
        TChar itos[256] = {0};       // index => raw character

        size_t total_number_of_words = words.size(); // total number of words in train set

        for (size_t i = 0; i < total_number_of_words; ++i)
        {
            const std::string_view wi = words[i];
            const size_t wi_size = wi.size();

            for (size_t j = 0; j < wi_size; ++j)
            {
                TChar c_char = wi[j];
                TTokenIndex c_unit = TTokenIndex(c_char);
            
                if (stoi[c_unit] == 0)
                {
                    stoi[c_unit] = totalCharacters;
                    itos[totalCharacters] = c_unit;
                    totalCharacters++;
                }
            }
        }

        constexpr size_t kTotalCharacters = 27;
        assert(totalCharacters == kTotalCharacters);
        assert(stoi[sysToken] == 0);
        assert(itos[0] == 0);

        stoi[sysToken] = 0;
        itos[0] = sysToken;

        my_log_stream() << "total symbols with pad: " << totalCharacters << '\n';

        typedef std::array<index_type, kBlockSize> IndexVecContext; // IndexVecContext contains the last kBlockSize characters to predict next one. The last character in a stream is in kBlockSize - 1 pos.
        typedef std::vector<index_type> IndexVecGeneral;            // Lineary resizable array with indicies

        std::vector<IndexVecContext> X;
        IndexVecGeneral Y;

        // in fact it's exact number of items in X,Y but to exploit this some underlying informatin have to be used
        //  bytes encode words in formar "words<LF>" [e.g. 6 bytes]
        //  each word [e.g. 5 bytes] produces 5+1 words, [e.g. 6 words]
        //  
        X.reserve(names_files.memorySizeInBytes + 1);
        Y.reserve(names_files.memorySizeInBytes + 1);

        {
            IndexVecContext context;
            for (size_t i = 0; i < total_number_of_words; ++i)
            {
                // reset context
                my_memset_to_zero<sizeof(context[0]) * kBlockSize> (context.data());

                std::string_view wi = words[i];
                size_t wordLen = wi.size();

                for (size_t j = 0; j < wordLen; ++j)
                {
                    X.push_back(context);

                    auto ix = stoi[wi[j]];
                    Y.push_back(ix);

                    if constexpr (kDebugPrint)
                    {
                        for (size_t k = 0; k < kBlockSize; ++k)
                        {
                            my_log_stream() << itos[context[k]];
                        }
                        my_log_stream() << "--->";
                        my_log_stream() << itos[ix] << '\n';
                    }

                    static_assert(std::is_trivially_copyable_v<index_type>);
                    memmove(context.data(), context.data() + 1, sizeof(context[0]) * (kBlockSize - 1));
                    context[kBlockSize - 1] = ix;
                }

                {
                    // padding symbol is used also for encoding the <end>
                    X.push_back(context);
                    Y.push_back(0);

                    if constexpr (kDebugPrint)
                    {
                        for (size_t k = 0; k < kBlockSize; ++k)
                        {
                            my_log_stream() << itos[context[k]];
                        }
                        my_log_stream() << "--->";
                        my_log_stream() << itos[0] << '\n';
                    }
                    // no need to update context in rolling way because it's the last step for processing word
                }
            }
        }

        const size_t total_train_set_size = X.size();
        burt::RandomGenIntegerLinear generatorForSampleItems;
        std::vector<uint32_t> seq_indicies(total_train_set_size);
        std::vector<uint32_t> indicies(total_train_set_size);
        size_t indicies_sz_in_bytes = total_train_set_size * sizeof(indicies[0]);

        assert(total_train_set_size <= std::numeric_limits<uint32_t>::max());
        for (size_t i = 0; i < total_train_set_size; ++i)
        {
            seq_indicies[i] = i;
        }
   
        my_log_stream() << " ~ batch: " << kBatchSize << '\n';
        my_log_stream() << " ~ hidden dim: " << hidden_dim << '\n';
        my_log_stream() << '\n';
        my_log_stream() << " ~ train set size: " << total_train_set_size << '\n';
        my_log_stream() << " ~ context len: " << kBlockSize << '\n';
        my_log_stream() << " ~ embedded dim: " << kEmbeddingSize << '\n';

        // Initialized Embedding Matrix C: ROWS -- token index | COLUMNDS -- flattened embedding with indicies from 0 to kEmbeddingSize
        using ValueWithEmbItem = Value<emb_type>;
        using VecWithEmbItem = std::array<ValueWithEmbItem, kEmbeddingSize>;
    
        auto first_trainable_neuron = ValueWithEmbItem::checkpointForNeurons();

        std::vector<VecWithEmbItem> C; // [Tokens] x [Embedding Size]
        C.resize(totalCharacters);     // Preallocate rows

        burt::RandomVariable rv;       // Used RV to inialize original embeddings with N(0,1)
        for (size_t i = 0; i < totalCharacters; ++i)
        {
            VecWithEmbItem& ci = C[i];
            for (size_t k = 0; k < kEmbeddingSize; ++k)
            {
                auto value = rv.generateNorm();
                ci[k] = std::move(ValueWithEmbItem(value));
            }
        }

        // Get C_at_x [i,j,k]
        //  i -- number of sample/example
        //  j -- index of separate words in "context" or in "block"
        //  k -- index separate scalar embeddings
        // But essentiall C_at_x[i,j] => points to need embedding

        std::vector<std::array<VecWithEmbItem*, kBlockSize>> C_at_x; // [Sample, BlockSize, Emdedding]
        C_at_x.resize(total_train_set_size);
        for (size_t i = 0; i < total_train_set_size; ++i)
        {
            for (size_t j = 0; j < kBlockSize; ++j) 
            {
                C_at_x[i][j] = &(C[X[i][j]]);
            }
        }

        // Random thoughts:
        //   - Activation consistent with makemore
        //   - Used SIMD vector for vector-vector ops inside
        //   - Add ability to view raw buffer [unmap not needed]
        //   - To think on how to enforce
        //   - To think if there is some matrix multiply then CO is very good choice
        //   - Compute Exp in parallel
        //   - Compute Activations in parallel
        //   - Compute Softmax potentially separately

        // input for W1 is essentially collection of embeddings from "kBlockSize" blocks each one contains "kEmbeddingSize" scalars
        
        MLPLayer<emb_type, /*bias*/ true> W1(kBlockSize * kEmbeddingSize, hidden_dim, NeuronInitType::uniform_neg_one_plus_one);
        MLPLayer<emb_type, /*bias*/ true> W2(hidden_dim, totalCharacters, NeuronInitType::uniform_neg_one_plus_one);

        //LayerAtCompileTime<emb_type, /*bias*/ true, ActivationType::eTanh, kBlockSize* kEmbeddingSize, hidden_dim, NeuronInitType::uniform_neg_one_plus_one> W1;
        //LayerAtCompileTime<emb_type, /*bias*/ true, ActivationType::eTanh, hidden_dim, kTotalCharacters, NeuronInitType::uniform_neg_one_plus_one> W2;

        auto end_trainable_neuron = ValueWithEmbItem::checkpointForNeurons();

        // system-like buffer to opimize backpropagation
        burt::MutableData reverse_topo_order_seq;
        burt::MutableData reverse_topo_order_set;
        burt::MutableData recursion;
        //reverse_topo_order_seq.reserveMemory(1 * 1024 * 1024);
        //reverse_topo_order_set.reserveMemory(1 * 1024 * 1024);
        //recursion.reserveMemory(1 * 1024 * 1024);

        const size_t num_classes = totalCharacters;

        my_log_stream() << "--------------------------------------------\n";
        my_log_stream() << " ~ number of token classes: " << num_classes << "\n";
        my_log_stream() << " ~ reserved memory for topological order: " << reverse_topo_order_seq.getAllocBytesToStoreData() / 1024.0 / 1024.0 << " MBytes\n";
        my_log_stream() << " ~ reserved memory for topological order leafs: " << reverse_topo_order_set.getAllocBytesToStoreData() / 1024.0 / 1024.0 << " MBytes\n";
        my_log_stream() << " ~ reserved memory for recursive traverse: " << recursion.getAllocBytesToStoreData() / 1024.0 / 1024.0 << " MBytes\n";
        my_log_stream() << " ~ train set - inputs memory: " << X.size() * X[0].size() * sizeof(X[0][0]) / 1024.0 / 1024.0 << " MBytes\n";
        my_log_stream() << " ~ train set - outputs memory: " << Y.size() * sizeof(Y[0]) / 1024.0 / 1024.0 << " MBytes\n";
        my_log_stream() << " ~ train set - samples: " << X.size() << '\n';
        my_log_stream() << "--------------------------------------------\n";

        burt::HighPrecisionTimer timer_to_process;

        constexpr size_t x_in_length = kBlockSize * kEmbeddingSize;
        std::array<ValueWithEmbItem, x_in_length> x_in;

        assert(end_trainable_neuron == ValueWithEmbItem::checkpointForNeurons());
        auto trainable_variables = (end_trainable_neuron - first_trainable_neuron);

        index_type prev_true_label = std::numeric_limits<index_type>::max();
    
        double time_to_process_avg = 0.0;
        double time_to_process_sqr_avg = 0.0;

        std::vector<ValueWithEmbItem> fwd_value_after_tanh_cached;
        std::vector<ValueWithEmbItem> counts_cached;
        std::vector<ValueWithEmbItem> counts_exp_cached;
    
        counts_cached.resize(totalCharacters);
        counts_exp_cached.resize(totalCharacters);

        uint32_t chkpoint = ValueWithEmbItem::checkpointForNeurons();

        for (size_t e = 1; e <= kTotalIterations; ++e)
        {
            timer_to_process.reset();

            ValueWithEmbItem::setGradToZeroIn(first_trainable_neuron, end_trainable_neuron);
            double loss_avg = double();

            {
                memcpy(indicies.data(), seq_indicies.data(), indicies_sz_in_bytes);
                burt::shuffle(indicies, kBatchSize, generatorForSampleItems);

                // Sort usefull for 2 reasons: seq. mem.fecthing AND reusing compute graphs already build for specific label. after embedding the used neurons are the same and topo sort can be skipped
                sort(indicies.begin(), 
                     indicies.begin() + kBatchSize, 
                     [&Y]<class TIndex>(TIndex a, TIndex b)
                     {
                        auto ya = Y[a];
                        auto yb = Y[b];

                        if (ya == yb) [[unlikely]]
                        {
                            return a < b;
                        }
                        else
                        {
                            return ya < yb;
                        }
                     }
                );

                for (size_t iiSample = 0; iiSample < kBatchSize; ++iiSample)
                {
                    uint32_t iSample = indicies[iiSample];

                    index_type true_label = Y[iSample];

#if 1
                    fwd_value_after_tanh_cached.clear();
                    counts_cached.clear();
                    counts_exp_cached.clear();
#else
                    ValueWithEmbItem::sysDestructManually(fwd_value_after_tanh_cached.data(), fwd_value_after_tanh_cached.size());
                    ValueWithEmbItem::sysDestructManually(counts_cached.data(), counts_cached.size());
                    ValueWithEmbItem::sysDestructManually(counts_exp_cached.data(), counts_exp_cached.size());
#endif
                    ValueWithEmbItem::restoreCheckpoint(chkpoint);

                    {
                        size_t w_offset = 0;
                        for (size_t j = 0; j < kBlockSize; ++j, w_offset += kEmbeddingSize)
                        {
                            ValueWithEmbItem::sysCreateLightView<kEmbeddingSize> (&x_in[w_offset], C_at_x[iSample][j]->data());
                        }
                        // x_in is concatenation of need embedding scalars
                    }

                    if constexpr (W2.activationType() == ActivationType::eIdent || W2.activationType() == ActivationType::eRelu)
                    {
                        // normalization tricks is really needed only in case use activation with unbounded image / co-domain
                        W1.forward<x_in.size()> (fwd_value_after_tanh_cached, x_in.data());
                        emb_type max_value = W2.forwardAndReportMax<hidden_dim> (counts_cached, fwd_value_after_tanh_cached.data());

                        for (size_t k = 0; k < totalCharacters; ++k)
                        {
                            counts_exp_cached[k] = std::move(exp_shifted(counts_cached[k], max_value));
                        }
                    }
                    else
                    {
                        W1.forward<x_in.size()> (fwd_value_after_tanh_cached, x_in.data());
                        //W2.forward(counts_cached, fwd_value_after_tanh_cached);
                        W2.forward<hidden_dim> (counts_cached, fwd_value_after_tanh_cached.data());

                        {
    #if 0
                            size_t totalCharacters_Div_4 = burt::roundToNearestMultipleDown<4>(totalCharacters);

                            for (size_t k = 0; k != totalCharacters_Div_4; k += 4)
                            {
                                counts_exp_cached[k]   = std::move(exp(counts_cached[k]));
                                counts_exp_cached[k+1] = std::move(exp(counts_cached[k+1]));
                                counts_exp_cached[k+2] = std::move(exp(counts_cached[k+2]));
                                counts_exp_cached[k+3] = std::move(exp(counts_cached[k+3]));
                            }
    #endif
                            if (counts_exp_cached.size() != counts_cached.size())
                                counts_exp_cached.resize(counts_cached.size());

                            for (size_t k = 0; k != totalCharacters; k++)
                            {
                                counts_exp_cached[k] = std::move(exp(counts_cached[k]));
                            }
                        }
                    }

                    ValueWithEmbItem loss;

                    if constexpr (kUsedSIMD4Compute)
                    {
                        ValueWithEmbItem countsExpSum = reduceSumForSequnetialAllocatedNeurons(counts_exp_cached.data(), counts_exp_cached.size());
                        ValueWithEmbItem pi = counts_exp_cached[true_label] / countsExpSum;
                        loss = negativeLog(pi);
                        loss_avg += loss.dataCopy();
                    }
                    else
                    {
                        ValueWithEmbItem countsExpSum = reduceSum(counts_exp_cached.data(), counts_exp_cached.size());
                        ValueWithEmbItem pi = counts_exp_cached[true_label] / countsExpSum;
                        loss = negativeLog(pi);
                        loss_avg += loss.dataCopy();
                    }

                    // We don't evaluate countsExp[not-true_label] because we actually do not need it at all.

                    // KL(p,q) = \sum (pi * log(pi/qi))          H(p) = -\sum pi * log(pi)        KL(p,q) + H(p) = -\sum (pi * log(qi))
                    // CE(p,q) = -\sum (pi * log(qi))         =>   CE(one-hot)=-log(qi)

                    // backward pass: run backward
                    if (prev_true_label != true_label)
                    {
                        backwardWithScratchStorage<decltype(loss), /*execute_reverse_topo_order*/ true, /*execute_backward_for_internal_nodes*/ true, /*execute_backward_for_leafs*/ false>(loss, reverse_topo_order_seq, reverse_topo_order_set, recursion);
                        prev_true_label = true_label;
                    }
                    else
                    {
                        backwardWithScratchStorage<decltype(loss), /*execute_reverse_topo_order*/ false, /*execute_backward_for_internal_nodes*/ true, /*execute_backward_for_leafs*/ false>(loss, reverse_topo_order_seq, reverse_topo_order_set, recursion);
                    }
                }
            }

            constexpr size_t processed_samples = kBatchSize;
            constexpr emb_type one_inv_processed_samples = 1.0/double(kBatchSize);
            constexpr emb_type lr = 0.1;
            constexpr emb_type one_inv_processed_samples_times_lr = one_inv_processed_samples * lr;

            // TODO: add support of multi-threaded

            emb_type grad_len_sqr = emb_type(-1);
            if constexpr (kUsedSIMD4Compute)
            {
                // grad_len_sqr = ValueWithEmbItem::applyGDStepAndComputeGradL2NormSquareWithSIMD(first_trainable_neuron, end_trainable_neuron, one_inv_processed_samples, lr);
                ValueWithEmbItem::applyGDStepWithSIMD(first_trainable_neuron, end_trainable_neuron, one_inv_processed_samples, lr);
            }
            else
            {
                // grad_len_sqr = ValueWithEmbItem::applyGDStepAndComputeGradL2NormSquare(first_trainable_neuron, end_trainable_neuron, one_inv_processed_samples, lr);
                ValueWithEmbItem::applyGDStep(first_trainable_neuron, end_trainable_neuron, one_inv_processed_samples, lr);
            }

            double time_to_process = timer_to_process.getTimeSec();

            time_to_process_avg = (double(e - 1)/double(e) * time_to_process_avg + time_to_process/double(e));
            time_to_process_sqr_avg = (double(e - 1) / double(e) * time_to_process_sqr_avg + time_to_process * time_to_process / double(e));
            //time_to_process_avg = time_to_process;

            if (e % kIterationsPrintFreq == 0)
            {
                my_log_stream() << "  ->Train: [" << e << '/' << kTotalIterations << "] "
                                << " | Loss: " << loss_avg * one_inv_processed_samples
                                << " | Grad l2 norm sqr: " << grad_len_sqr
                                << " | Avg Time: " << time_to_process_avg * 1000 << " msec.\n"
                                << " | Std.dev for time: " << sqrt(fabs(time_to_process_sqr_avg - time_to_process_avg * time_to_process_avg)) * 1000 << " msec.\n"
                                << " | Params: " << (end_trainable_neuron - first_trainable_neuron) << "\n";

                if (kPrintDetailedInfo)
                {
                    auto W1_stats = W1.statistics();
                    auto W2_stats = W2.statistics();

                    my_log_stream() << "| kilo-samples/sec.: " << processed_samples / time_to_process / 1000.0
                                    << "| nodes: " << Value<float>::numActiveNodes() / 1000.0 << "K"
                                    << "| processed samples: " << processed_samples << "\n"
                                    << "| trainable vars: " << trainable_variables << "\n"
                                    << '\n';

                    Value<float>::Statistics stats;

                    Value<float>::sysCollectStatistics(stats);

                    my_log_stream() << "  ->Train Mem. Consumption: [" << e << '/' << kTotalIterations << "]\n"
                                    << "| nodes names mem.: " << stats.occupied_memory.labelsMemory / 1024. << " Kb\n"
                                    << "| nodes backward type mem.: " << stats.occupied_memory.bwdOpDescrMemory / 1024. << " Kb\n"
                                    //<< "| nodes visiting number for backprop mem.: " << memory_stats.visitingNumberForBackpropMemory / 1024. << " KBytes \n"
                                    << "| nodes children topology mem.: " << stats.occupied_memory.childrenTopologyMemory / 1024. << " Kb\n"
                                    << "| nodes activations mem.: " << stats.occupied_memory.activationsMemory / 1024. << " Kb\n"
                                    << "| nodes grads mem.: " << stats.occupied_memory.gradsMemory / 1024. << " Kb\n\n\n";
                }
            }
        }

        ValueWithEmbItem::sysInvalidateLightView<x_in_length>(x_in.data());

        bool unmapInputFile = burt::FileSystemHelpers::unmapFileFromMemory(names_files);
        assert(unmapInputFile == true);
        double deltaSec = timer_main.getTimeSec();
        my_log_stream() << "\nTotal computate and parse time: " << deltaSec * 1000 << " msec.\n";

        my_log_stream() << '\n';
        my_log_stream() << " ~ batch: " << kBatchSize << '\n';
        my_log_stream() << " ~ hidden dim: " << hidden_dim << '\n';
        my_log_stream() << '\n';
    }

    Value<float>::deactiveUnusedNodes();
    auto nodes_at_end = Value<float>::numActiveNodes();
    assert(nodes_at_end == nodes_at_start);


    if (kMakePauseAtEnd)
        getchar();

    return 0;
}
