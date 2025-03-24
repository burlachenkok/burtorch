
#include "burt/system/include/PlatformSpecificMacroses.h"
#include "burt/system/include/threads/Thread.h"

#include "burt/timers/include/HighPrecisionTimer.h"
#include "burt/fs/include/FileSystemHelpers.h"
#include "burt/copylocal/include/Data.h"

#include "burt/random/include/RandomGenIntegerLinear.h"
#include "burt/random/include/RandomGenRealLinear.h"

#include "burtcore/include/burtorch.h"

#include <vector>
#include <string_view>
#include <iostream>

#include <stddef.h>
#include <assert.h>

namespace
{
    inline std::ostream& my_log_stream() {
        return std::cout;
    }

    struct Tokenizer
    {
        bool constructFromParsingTheFile(const char* fname_open)
        {
            burt::FileSystemHelpers::FileMappingResult names_files = burt::FileSystemHelpers::mapFileToMemory(fname_open, true);

            if (!names_files.isOk)
            {
                std::cout << "File can not be opened: " << fname_open << '\n';
                std::cout << "Error message: " << names_files.errorMsg << '\n';
                return false;
            }

            std::cout << "Train dataset: " << fname_open << '\n';
            std::cout << "Train dataset size in bytes: " << names_files.memorySizeInBytes << '\n';

            burt::Data trainDatasetRaw(names_files.memory, names_files.memorySizeInBytes, burt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
            std::string_view chars = std::string_view((char*)trainDatasetRaw.getPtr(), trainDatasetRaw.getTotalLength());

            vocab_size = 0;
            for (size_t i = 0; i < totalPossibleCharacters; ++i)
            {
                characters_in_text[i] = false;
                stoi[i] = -1;
                itos[i] = char(-1);
            }

            for (size_t i = 0; i < chars.size(); ++i)
            {
                unsigned char character_ = (unsigned char)(chars[i]);
                characters_in_text[character_] = true;
            }

            for (size_t i = 0; i < totalPossibleCharacters; ++i)
            {
                if (characters_in_text[i])
                    vocab_size++;
            }

            std::cout << "vocab size: " << vocab_size << '\n';
            std::cout << "characters in the entire dataset:\n";
            for (size_t i = 0; i < totalPossibleCharacters; ++i)
            {
                if (characters_in_text[i])
                {
                    std::cout << (char)i;
                }
            }
            std::cout << '\n';

            {
                int index = 0;
                for (size_t i = 0; i < totalPossibleCharacters; ++i)
                {
                    if (characters_in_text[i])
                    {
                        stoi[i] = index;
                        itos[index] = (char)i;
                        index++;
                    }
                }
                assert(index == vocab_size);
            }
            bool unmapInputFile = burt::FileSystemHelpers::unmapFileFromMemory(names_files);
            assert(unmapInputFile == true);

            return true;
        }

        std::vector<uint64_t> encode(std::string_view s)
        {
            std::vector<uint64_t> encoding;
            encoding.reserve(s.size());

            for (size_t i = 0; i < s.size(); ++i)
                encoding.push_back(stoi[s[i]]);
            return encoding;
        }

        std::string decode(const std::vector<uint64_t>& s_encoded)
        {
            std::ostringstream out_str;            
            for (size_t i = 0; i < s_encoded.size(); ++i)
                out_str << itos[s_encoded[i]];
            return out_str.str();
        }

        static inline constexpr size_t totalPossibleCharacters = 255;

        bool characters_in_text[totalPossibleCharacters] = { false };
        size_t vocab_size = 0;
        int stoi[totalPossibleCharacters] = {};
        char itos[totalPossibleCharacters] = {};
    };

    bool createTrainAndValidationData(const char* fname_open,
                                      Tokenizer& tk,
                                      double train_data_fraction, 
                                      std::vector<uint64_t>& train_data, 
                                      std::vector<uint64_t>& val_data)
    {
        burt::FileSystemHelpers::FileMappingResult names_files = burt::FileSystemHelpers::mapFileToMemory(fname_open, true);

        if (!names_files.isOk)
        {
            std::cout << "File can not be opened: " << fname_open << '\n';
            std::cout << "Error message: " << names_files.errorMsg << '\n';
            return false;
        }

        burt::Data trainDatasetRaw(names_files.memory, names_files.memorySizeInBytes, burt::Data::MemInitializedType::eGiftWholeMemoryPleaseNotFree);
        std::string_view chars = std::string_view((char*)trainDatasetRaw.getPtr(), trainDatasetRaw.getTotalLength());

        const size_t total_sz = chars.size();
        const size_t train_data_sz = size_t(total_sz * 0.9);
        const size_t val_data_sz = total_sz - train_data_sz;

        train_data.reserve(train_data.size() + train_data_sz);
        val_data.reserve(val_data.size() + val_data_sz);

        {
            size_t i = 0;
            for (; i < train_data_sz; ++i)
                train_data.push_back(tk.stoi[chars[i]]);
            for (; i < total_sz; ++i)
                val_data.push_back(tk.stoi[chars[i]]);
        }

        bool unmapInputFile = burt::FileSystemHelpers::unmapFileFromMemory(names_files);
        assert(unmapInputFile == true);

        return true;
    }

    void generateBatch(std::vector<std::vector<uint64_t>>& X, 
                       std::vector<std::vector<uint64_t>>& Y,
                       size_t k_batch_size, 
                       size_t k_block_size,
                       burt::RandomGenIntegerLinear& gen, 
                       std::vector<uint64_t>& input_data)
    {
        X.clear();
        Y.clear();

        X.reserve(k_batch_size);
        Y.reserve(k_batch_size);

        const size_t input_data_sz = input_data.size();

        for (size_t e = 0; e < k_batch_size; ++e)
        {
            size_t ix = gen.generateInteger();
            ix = ix % (input_data_sz - k_block_size);

            std::vector<uint64_t> input_;
            input_.reserve(k_block_size);

            std::vector<uint64_t> output_;
            output_.reserve(k_block_size);

            for (size_t ii = 0; ii < k_block_size; ++ii)
            {
                input_.push_back(input_data[ix + ii]);
                output_.push_back(input_data[ix + ii + 1]);
            }
            X.push_back(input_);
            Y.push_back(output_);
        }

        return;
    }

    template<class element_type, size_t the_n_emb, size_t the_head_size>
    struct SelfAttentionHead
    {
        SelfAttentionHead()
        : key(the_n_emb, the_head_size, NeuronInitType::zero_init)
        , query(the_n_emb, the_head_size, NeuronInitType::zero_init)
        , value(the_n_emb, the_head_size, NeuronInitType::zero_init)
        {
            burt::RandomVariable rv_for_init;

            // initialize k,q,v
            element_type std_for_variance_k = key.template std_for_initialization<element_type>();
            for (auto p : key.parameters())
                p.dataRef() = rv_for_init.generateUniform(-std_for_variance_k, +std_for_variance_k);

            element_type std_for_variance_q = query.template std_for_initialization<element_type>();
            for (auto p : query.parameters())
                p.dataRef() = rv_for_init.generateUniform(-std_for_variance_q, +std_for_variance_q);

            element_type std_for_variance_v = value.template std_for_initialization<element_type>();
            for (auto p : value.parameters())
                p.dataRef() = rv_for_init.generateUniform(-std_for_variance_v, +std_for_variance_v);
        }

        inline static constexpr double constSqrt(double x, double curr = 1.0) 
        {
            // find x^2-a via Newton method
            constexpr double acc = 1e-10;
            if (curr * curr - x > acc || curr * curr - x < -acc)
                return constSqrt(x, (curr + x / curr) / 2);
            else
                return curr;
        }

        template<size_t block_size,
                 size_t channels,
                 class TCtr,
                 bool allow_complete_attention = false, 
                 bool scale_att = true>
        TCtr forwardSample(const TCtr& x) noexcept
        {
            // allow_complete_attention -- allow all nodes/tokens to talk to each other completely
            //   allow_complete_attention = true => Encoder block [no mask]
            // allow_complete_attention = false => Decoder block [triangular attention mask] which is used in auto-regressive tasks

            constexpr element_type one_inv_sqrt_head_size = element_type(1) / constSqrt(double(the_head_size));
            assert(fabs(constSqrt(double(the_head_size)) - sqrt(double(the_head_size))) < 1e-6);
            Value<element_type> one_inv_sqrt_head_size_s(one_inv_sqrt_head_size);

            // Not-scaled attention
            constexpr size_t T = block_size;  // time
            constexpr size_t C = channels;    // channels

            assert(T == x.size());
            assert(C == x[0].size());

            // No commucation between samples for now and between any token => all tokens processed in parallel

            // <NO-B>, T, head_size
            TCtr k;
            k.resize(T);
            // <NO-B>, T, head_size
            TCtr q;
            q.resize(T);
            // NO-B, T, head_size ==>>> propagate x through values
            TCtr v;
            v.resize(T);

            {
                for (size_t t = 0; t < T; ++t)
                {
                    auto xDataPtr = x[t].data();
                    key.template forward<C>(k[t], xDataPtr);
                    query.template forward<C>(q[t], xDataPtr);
                    value.template forward<C>(v[t], xDataPtr);
                }
            }

            // Communication between tokens happens NOW.
            // We performed batched matrix multiplication with q (NO-B,T,Heads) and k.Transpose (NO-B,Heads,T) ---> (B,T,T)      
            //   k (<NO-B>,T,Heads) 
            //   We perform masking explicitly instead of any tricks

            //TCtr wei;
            //TCtr wei_afer_softmax;
            //TCtr wei_afer_exp;

            typename TCtr::value_type wei;
            typename TCtr::value_type wei_afer_softmax;
            typename TCtr::value_type wei_afer_exp;
            typename TCtr::value_type::value_type wei_afer_exp_sum_by_rows;

            wei.reserve(T);
            wei_afer_exp.reserve(T);
            wei_afer_softmax.reserve(T);

            // finally multiply [wei (B,T,T) ] by [v (B, T, head_size) ] ====> B, T, head_size
            // <NO-B>, T, head_size
            TCtr out;
            out.resize(T);

            {
                for (size_t t1 = 0; t1 < T; ++t1)
                {
                    size_t T2_bound = 0;
                    if (allow_complete_attention)
                        T2_bound = T;
                    else
                        T2_bound = t1 + 1;

                    const auto& row_t1_of_q = q[t1];

                    auto& row_t1_of_wei = wei;
                    auto& row_t1_of_wei_afer_exp = wei_afer_exp;
                    auto& row_t1_of_wei_afer_softmax = wei_afer_softmax;

                    row_t1_of_wei.resize(T2_bound);
                    row_t1_of_wei_afer_exp.resize(T2_bound);
                    row_t1_of_wei_afer_softmax.resize(T2_bound);

                    element_type max_item = -(std::numeric_limits<element_type>::max());
                    for (size_t t2 = 0; t2 < T2_bound; ++t2)
                    {
                        assert(row_t1_of_q.size() == k[t2].size());
                        const auto& col_t2_of_k_tr = k[t2];
                        row_t1_of_wei[t2] = innerProduct<the_head_size>(row_t1_of_q.data(), col_t2_of_k_tr.data());

                        if constexpr (scale_att)
                        {
                            row_t1_of_wei[t2] = mulByConstant(row_t1_of_wei[t2], one_inv_sqrt_head_size_s);
                        }
                        auto item = row_t1_of_wei[t2].dataCopy();
                        if (item > max_item)
                            max_item = item;
                    }

                    for (size_t t2 = 0; t2 < T2_bound; ++t2) {
                        row_t1_of_wei_afer_exp[t2] = exp_shifted(row_t1_of_wei[t2], max_item);
                    }

                    wei_afer_exp_sum_by_rows = reduceSum(row_t1_of_wei_afer_exp.data(), T2_bound);

                    for (size_t t2 = 0; t2 < T2_bound; ++t2)
                        row_t1_of_wei_afer_softmax[t2] = row_t1_of_wei_afer_exp[t2] / wei_afer_exp_sum_by_rows;

                    out[t1].resize(the_head_size);

                    for (size_t k = 0; k < the_head_size; ++k)
                    {
                        std::vector<Value<float>> v_k_column;
                        v_k_column.reserve(T2_bound);
                        for (size_t kk = 0; kk < T2_bound; ++kk)
                            v_k_column.emplace_back(v[kk][k]);

                        assert(v_k_column.size() == T2_bound);
                        assert(row_t1_of_wei_afer_softmax.size() == T2_bound);
                        out[t1][k] = innerProduct(row_t1_of_wei_afer_softmax.data(), v_k_column.data(), T2_bound);
                    }
                }
            }

            return out;
        }

        template<class TCtr, bool allow_complete_attention = false, bool scale_att = true>
        TCtr forward(const TCtr& x) noexcept
        {
            // allow_complete_attention
            //   allow all nodes/tokens to talk to each other completely
            //   allow_complete_attention = true => Encoder block [no mask]
            // allow_complete_attention = false => Decoder block [triangular attention mask] which is used in auto-regressive tasks

            const element_type one_inv_sqrt_head_size = element_type(1) / sqrt(element_type(the_head_size));

            // not-scaled attention

            const size_t B = x.size();       // batches
            const size_t T = x[0].size();    // time
            const size_t C = x[0][0].size(); // channels

            // No commucation between samples for now and between any token => all tokens processed in parallel
            
            // B, T, head_size
            TCtr k;

            k.resize(B);
            for (size_t b = 0; b < B; ++b)
            {
                k[b].resize(T);
                for (size_t t = 0; t < T; ++t)
                    k[b][t] = key.forward(x[b][t]);
            }

            // B, T, head_size
            TCtr q;

            q.resize(B);
            for (size_t b = 0; b < B; ++b)
            {
                q[b].resize(T);
                for (size_t t = 0; t < T; ++t)
                    q[b][t] = query.forward(x[b][t]);
            }

            // Communication between tokens happens NOW.
            // We performed batched matrix multiplication with q (B,T,Heads) and k.Transpose (B,Heads,T) ---> (B,T,T)      
            //   k (B,T,Heads) 
            //   We perform masking explicitly instead of any tricks
            TCtr wei;
            wei.resize(B);
            for (size_t b = 0; b < B; ++b)
            {
                wei[b].resize(T);
                for (size_t t1 = 0; t1 < T; ++t1)
                {
                    const auto& row_t1_of_q = q[b][t1];
                    wei[b][t1].resize(T);
                    for (size_t t2 = 0; t2 < T; ++t2)
                    {
                        const auto& col_t2_of_k_tr = k[b][t2];
                        assert(row_t1_of_q.size() == col_t2_of_k_tr.size());
                        wei[b][t1][t2] = innerProduct<the_head_size> (row_t1_of_q.data(), col_t2_of_k_tr.data());

                        if (scale_att)
                        {
                            wei[b][t1][t2] = wei[b][t1][t2] * Value<element_type>(one_inv_sqrt_head_size);
                        }
                    }
                }
            }

            TCtr wei_afer_softmax;
            TCtr wei_afer_exp;
            typename TCtr::value_type wei_afer_exp_sum_by_rows;


            wei_afer_softmax.resize(B);
            wei_afer_exp.resize(B);
            wei_afer_exp_sum_by_rows.resize(B);

            for (size_t b = 0; b < B; ++b)
            {
                wei_afer_exp[b].resize(T);
                wei_afer_exp_sum_by_rows[b].resize(T);
                wei_afer_softmax[b].resize(T);

                for (size_t t1 = 0; t1 < T; ++t1)
                {
                    size_t T2_bound = 0;
                    if (allow_complete_attention)
                        T2_bound = T;
                    else
                        T2_bound = t1 + 1;

                    wei_afer_exp[b][t1].resize(T2_bound);
                    wei_afer_softmax[b][t1].resize(T2_bound);

                    auto max_item = wei[b][t1][0].dataCopy();
                    for (size_t t2 = 1; t2 < T2_bound; ++t2)
                    {
                        if (wei[b][t1][t2].dataCopy() > max_item)
                            max_item = wei[b][t1][t2].dataCopy();
                    }
                    for (size_t t2 = 0; t2 < T2_bound; ++t2) {
                        wei_afer_exp[b][t1][t2] = exp_shifted(wei[b][t1][t2], max_item);
                    }

                    wei_afer_exp_sum_by_rows[b][t1] = reduceSum(wei_afer_exp[b][t1].data(), T2_bound);

                    for (size_t t2 = 0; t2 < T2_bound; ++t2)
                        wei_afer_softmax[b][t1][t2] = wei_afer_exp[b][t1][t2] / wei_afer_exp_sum_by_rows[b][t1];
                }
            }

            // propagate x through values
            // B, T, head_size
            TCtr v;

            v.resize(B);
            for (size_t b = 0; b < B; ++b)
            {
                v[b].resize(T);
                for (size_t t = 0; t < T; ++t)
                    v[b][t] = value.forward(x[b][t]);
            }

            // finally multiply [wei (B,T,T) ] by [v (B, T, head_size) ] ====> B, T, head_size
            TCtr out; // B, T, head_size
            out.resize(B);
            for (size_t b = 0; b < B; ++b)
            {
                out[b].resize(T);
                for (size_t t = 0; t < T; ++t)
                {
                    out[b][t].resize(the_head_size);

                    const auto& wei_t_row = wei_afer_softmax[b][t];

                    for (size_t k = 0; k < the_head_size; ++k)
                    {
                        size_t T2_bound = 0;
                        if (allow_complete_attention)
                            T2_bound = T;
                        else
                            T2_bound = t + 1;

                        std::vector<Value<float>> v_k_column;
                        v_k_column.reserve(T2_bound);

                        for (size_t kk = 0; kk < T2_bound; ++kk)
                            v_k_column.emplace_back(v[b][kk][k]);

                        assert(v_k_column.size() == wei_t_row.size());
                        out[b][t][k] = innerProduct(wei_t_row.data(), v_k_column.data(), wei_t_row.size());
                    }
                }
            }

            return out;
        }

        MLPLayer<element_type, /*bias*/ false, ActivationType::eIdent> key;
        MLPLayer<element_type, /*bias*/ false, ActivationType::eIdent> query;
        MLPLayer<element_type, /*bias*/ false, ActivationType::eIdent> value;
    };

    struct EmbeddingModel
    {
        EmbeddingModel(size_t theVocabSize, size_t theEmbeddingSize, size_t theBlockSize, burt::RandomVariable& rv)
        : embeddingSize(theEmbeddingSize)
        , vocabSize(theVocabSize)
        , blockSize(theBlockSize)
        {
            token_emb_table.resize(vocabSize);
            for (size_t i = 0; i < vocabSize; ++i)
            {
                for (size_t k = 0; k < embeddingSize; ++k)
                {
                    auto value = rv.generateNorm(0.0, 1.0);
                    token_emb_table[i].emplace_back(value);
                }
            }

            pos_emb_table.resize(blockSize);
            for (size_t i = 0; i < blockSize; ++i)
            {
                for (size_t k = 0; k < embeddingSize; ++k)
                {
                    auto value = rv.generateNorm(0.0, 1.0);
                    pos_emb_table[i].emplace_back(value);
                }
            }
        }

        std::vector<std::vector<std::vector<Value<float>>*>> embedding(std::vector<std::vector<uint64_t>>& X)
        {
            // X:
            //  axe-0: B samples
            //  axe-1: T tokens (integer indicies)

            // OUT:
            //  axe-0: B samples
            //  axe-1: T tokens
            //  axe-2: Embedding vector items

            std::vector<std::vector<std::vector<Value<float>>*>> C_at_x;
            C_at_x.resize(X.size());
            for (size_t i = 0; i < X.size(); ++i)
            {
                C_at_x[i].resize(X[i].size());
                for (size_t j = 0; j < X[i].size(); ++j)
                {
                    C_at_x[i][j] = &(token_emb_table[X[i][j]]);
                }
            }
            return C_at_x;
        }


        void embedding(std::vector<std::vector<std::vector<Value<float>>*>>& C_at_x, std::vector<std::vector<uint64_t>>& X)
        {
            // X:
            //  axe-0: B samples
            //  axe-1: T tokens (integer indicies)

            // OUT:
            //  axe-0: B samples
            //  axe-1: T tokens
            //  axe-2: Embedding vector items

            C_at_x.resize(X.size());
            for (size_t i = 0; i < X.size(); ++i)
            {
                C_at_x[i].resize(X[i].size());
                for (size_t j = 0; j < X[i].size(); ++j)
                {
                    C_at_x[i][j] = &(token_emb_table[X[i][j]]);
                }
            }

            return;
        }

        void extendAutoRegressively(std::vector<std::vector<uint64_t>>& X, size_t kMaxSequence, burt::RandomGenRealLinear& gen)
        {
            std::vector<std::vector<std::vector<Value<float>>*>> C_at_x = embedding(X);

            const size_t B = C_at_x.size();

            for (size_t seq = 0; seq < kMaxSequence; ++seq)
            {
                for (size_t b = 0; b < B; ++b)
                {
                    // last time step
                    const size_t t = C_at_x[b].size() - 1;

                    //logits from last time step
                    auto& logits = *(C_at_x[b][t]);

                    // compute PDF and CDF for sampling
                    std::vector<double> pdf;
                    std::vector<double> cdf;
                    double logits_exp_sum = 0.0;

                    for (size_t c = 0; c < logits.size(); ++c)
                    {
                        double v = exp(logits[c].dataCopy());
                        logits_exp_sum += v;
                        pdf.emplace_back(v);
                    }

                    double sum_ = double();
                    for (size_t c = 0; c < logits.size(); ++c)
                    {
                        pdf[c] /= logits_exp_sum;
                        cdf.emplace_back(sum_);
                        sum_ += pdf[c];
                    }

                    cdf.emplace_back(1.0 + 1e-9);

                    // Sampling from CDF
                    size_t jSampled = 0;
                    double u = gen.generateReal();

                    for (size_t j = 0; j < cdf.size() - 1; ++j)
                    {
                        if (cdf[j + 1] >= u)
                        {
                            jSampled = j;
                            break;
                        }
                    }

                    // Extend C_at_x
                    auto* embedding = &(token_emb_table[jSampled]);
                    C_at_x[b].emplace_back(embedding);
                    X[b].emplace_back(jSampled);
                }
            }
        }

        size_t embeddingSize;                     // Embedding size in C
        size_t vocabSize;                         // Vocabulary size for symbols
        size_t blockSize;                         // Block size to process

        std::vector<std::vector<Value<float>>> token_emb_table; // Embedding matrix for symbols
        std::vector<std::vector<Value<float>>> pos_emb_table;   // Embedding matrix for postions
    };
}

typedef std::vector<Value<float>> tensor_1d_fp32;
typedef std::vector<std::vector<Value<float>>> tensor_2d_fp32;
typedef std::vector<std::vector<std::vector<Value<float>>>> tensor_3d_fp32;

tensor_3d_fp32 rand_fp32(size_t axe_0_items, size_t axe_1_items, size_t axe_2_items, burt::RandomVariable& rv)
{
    tensor_3d_fp32 res;
    res.resize(axe_0_items);
    for (size_t i = 0; i < axe_0_items; ++i)
    {
        res[i].resize(axe_1_items);
        for (size_t j = 0; j < axe_1_items; ++j)
        {
            res[i][j].reserve(axe_2_items);
            for (size_t k = 0; k < axe_2_items; ++k)
            {
                auto value = rv.generateUniform();
                res[i][j].emplace_back(value);
            }
        }
    }

    return res;
}

void print_shape(const tensor_3d_fp32& tensor) {
    std::cout << "[" << tensor.size() << ',' << tensor[0].size() << ',' << tensor[0][0].size() << ']' << '\n';
}

void print_content(const tensor_2d_fp32& tensor) {

    std::cout << '[';
    std::cout << '\n';
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        std::cout << '[';
        for (size_t j = 0; j < tensor[i].size(); ++j)
        {
            auto item = tensor[i][j].dataCopy();
            std::cout << item;

            if (j != tensor[i].size() - 1)
                std::cout << ',';
        }
        std::cout << ']';

        if (i != tensor.size() - 1)
            std::cout << ',';

        std::cout << '\n';
    }
    std::cout << ']';
}


int main(int argc, char** argv)
{
    constexpr bool kMakePauseAtEnd = false;
    burt::DefaultThread::setThreadAffinityMaskForCurrentTh(0x1 << 1);

    if (argc == 1)
    {
        std::cout << "Please specify name of input dataset as first argument (like exp3_input.txt)";
        return -1;
    }
    const char* fname = argv[1];

    burt::MutableData reverse_topo_order_seq;
    burt::MutableData reverse_topo_order_set;
    burt::MutableData recursion;


    burt::HighPrecisionTimer timer_main;
    Tokenizer tk;
    tk.constructFromParsingTheFile(fname);

    // build encoder-decoder for tokenization  

    auto res = tk.encode("hii there");
    assert(res == std::vector<uint64_t>({46, 47, 47, 1, 58, 46, 43, 56, 43}));
    assert(std::string("hii there") == tk.decode(tk.encode("hii there")));
    
    // https://youtu.be/kCc8FmEb1nY?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=765

    // now tokenize all entire input dataset
    // default std::vector for uint64 as in torch [COMMENT-1: UINT64 is not really needed, COMMENT-2: MAYBE POSSIBLE TO MAKE LOOKUP WITH SIMD]
    std::vector<uint64_t> train_data;
    std::vector<uint64_t> val_data;
    createTrainAndValidationData(fname, tk, 0.9, train_data, val_data);

    
    constexpr size_t k_block_size = 8; // block_size or context_length -- maximum lenth of sequence to feed.
    constexpr size_t k_batch_size = 64;// 1 + 0 * 32;

    std::cout << "BATCH SZ: " << k_batch_size << "\n";

    // test
    {
        // X[pos_in_train_data,...,pos_in_train_data+j] => X[pos_in_train_data+j+1]
        size_t pos_in_train_data = 0;
        for (size_t t = 0; t < k_block_size; ++t)
        {
            std::cout << "when input: ";
            for (size_t j = 0; j <= t; ++j)
            {
                std::cout << train_data[pos_in_train_data + j] << ' ';
            }
            std::cout << "the target: " << train_data[pos_in_train_data + t + 1];
            std::cout << '\n';
        }
    }

    burt::RandomGenIntegerLinear gen_sampler_val;
    gen_sampler_val.setSeed(123);
    
    // generate validation batch
    {
        std::vector<std::vector<uint64_t>> X, Y;
        generateBatch(X, Y, k_batch_size, k_block_size, gen_sampler_val, val_data);
        assert(X.size() == k_batch_size);
        assert(Y.size() == k_batch_size);
        assert(X[0].size() == k_block_size);
        assert(Y[0].size() == k_block_size);
    }

    burt::RandomVariable init_rv;

    constexpr size_t n_emb = 24;
    constexpr size_t k_heads = 6;
    constexpr size_t n_layer = 6; // number of TRANSFORMER encoder blocks

    auto first_trainable_neuron = Value<float>::checkpointForNeurons();

    // emb models
    EmbeddingModel model(tk.vocab_size, n_emb, k_block_size, init_rv);


    assert(n_emb % k_heads == 0);

    // block internals
    typedef SelfAttentionHead<float, n_emb, n_emb / k_heads> SaHead;
    std::vector<SaHead*> sa_heads;
    std::vector<MLPLayer<float, true, ActivationType::eRelu>*> fwd_a;
    std::vector<MLPLayer<float, true, ActivationType::eIdent>*> fwd_b_projection;
    std::vector<MLPLayer<float, true, ActivationType::eIdent>*> fwd_projection;

    for (size_t k = 0; k < k_heads * n_layer; ++k)
    {
        sa_heads.emplace_back(new SaHead());
    }

    constexpr size_t n_project_emb = 4 * n_emb;

    for (size_t k = 0; k < n_layer; ++k)
    {
        // due to transofmer paper
        fwd_a.emplace_back(new MLPLayer<float, true, ActivationType::eRelu>(n_emb, n_project_emb, NeuronInitType::uniform_neg_one_plus_one));
        fwd_b_projection.emplace_back(new MLPLayer<float, true, ActivationType::eIdent>(n_project_emb, n_emb, NeuronInitType::uniform_neg_one_plus_one));
        
        fwd_projection.emplace_back(new MLPLayer<float, true, ActivationType::eIdent>(n_emb, n_emb, NeuronInitType::uniform_neg_one_plus_one));
    }
    // layer norm params
    burt::RandomVariable rv_layer_norm_init;

    std::vector<std::vector<Value<float>>> gamma_ln_1;
    std::vector<std::vector<Value<float>>> betta_ln_1;

    std::vector<std::vector<Value<float>>> gamma_ln_2;   
    std::vector<std::vector<Value<float>>> betta_ln_2;
    
    for (size_t k = 0; k < n_layer; ++k)
    {
        gamma_ln_1.push_back(std::vector<Value<float>>());
        betta_ln_1.push_back(std::vector<Value<float>>());
        for (size_t c = 0; c < n_emb; ++c)
        {
            gamma_ln_1.back().push_back(Value<float>(rv_layer_norm_init.generateNorm(0.0, 1.0)));
            betta_ln_1.back().push_back(Value<float>(rv_layer_norm_init.generateNorm(0.0, 1.0)));
        }

        gamma_ln_2.push_back(std::vector<Value<float>>());
        betta_ln_2.push_back(std::vector<Value<float>>());
        for (size_t c = 0; c < n_emb; ++c)
        {
            gamma_ln_2.back().push_back(Value<float>(rv_layer_norm_init.generateNorm(0.0, 1.0)));
            betta_ln_2.back().push_back(Value<float>(rv_layer_norm_init.generateNorm(0.0, 1.0)));
        }
    }

    // head
    MLPLayer<float> lm_head = MLPLayer<float>(n_emb, tk.vocab_size, NeuronInitType::uniform_neg_one_plus_one);

    auto end_trainable_neuron = Value<float>::checkpointForNeurons();

    auto trainable_variables = (end_trainable_neuron - first_trainable_neuron);

    uint32_t chkpoint = Value<float>::checkpointForNeurons();

    burt::RandomGenIntegerLinear gen_sampler_train;
    gen_sampler_train.setSeed(123);


    constexpr size_t kMaxIterations = 3000;
    constexpr size_t kValFreq = 200;
    constexpr size_t kPrintFreq = 500;

    constexpr bool kPrintDetailedInfo = !true;                                              // print detailed information
    constexpr size_t kIterationsPrintFreq = 100;                                            // print frequency for memory reports

    double time_to_process_avg = 0.0;
    double time_to_process_sqr_avg = 0.0;

    burt::HighPrecisionTimer timer_to_process;
    std::vector<std::vector<   std::vector<Value<float>>*    >> C_at_x;

    for (size_t e = 1; e <= kMaxIterations; ++e)
    {
        //        for debug
        //        gen_sampler_train.setSeed(123);
        std::vector<std::vector<uint64_t>> X, Y;
        bool vallidation = false;

        if (false && e % kValFreq == 0)
        {
            // slightly different compare to Karpathy
            generateBatch(X, Y, k_batch_size, k_block_size, gen_sampler_val, val_data);
            vallidation = true;
        }
        else
        {
            generateBatch(X, Y, k_batch_size, k_block_size, gen_sampler_train, train_data);
            vallidation = false;
        }

        timer_to_process.reset();

        Value<float>::setGradToZeroIn(first_trainable_neuron, end_trainable_neuron);

        model.embedding(C_at_x, X);

        size_t processed_samples = 0;

        float loss_avg = 0.0;
        {
            std::vector<Value<float>> logits;
            logits.reserve(model.vocabSize);

            std::vector<Value<float>> countsExp;
            countsExp.reserve(model.vocabSize);

            std::vector<Value<float>> x_in_after_fwd_a;
            x_in_after_fwd_a.reserve(n_project_emb);

            std::vector<Value<float>> x_in_after_fwd_b;
            x_in_after_fwd_b.reserve(n_emb);

            std::vector<std::vector<Value<float>>> x_in_tc; // T,C
            x_in_tc.reserve(k_block_size);

            std::vector<std::vector<Value<float>>> x_in_tc_orig_; // T,C
            x_in_tc_orig_.reserve(k_block_size);

            std::vector<Value<float>> x_in_after_proj;
            x_in_after_proj.reserve(n_emb);

            std::vector<Value<float>> x_in;
            //x_in.reserve(n_emb);

            for (size_t iSample = 0; iSample < k_batch_size; ++iSample)
            {
                {
                    logits.clear();
                    countsExp.clear();
                    x_in_after_fwd_a.clear();
                    x_in_after_fwd_b.clear();
                    x_in_tc.clear();
                    x_in_tc_orig_.clear();
                    x_in_after_proj.clear();
                }

                Value<float>::restoreCheckpoint(chkpoint);

                for (size_t t = 0; t < k_block_size; ++t)
                {
                    {
                        x_in.resize(n_emb);
                        // form embeddingSize items with mixing token embedding and position embedding
                        for (size_t k = 0; k < n_emb; ++k)
                        {
                            const auto& x_in_token = (*(C_at_x[iSample][t]))[k];
                            const auto& x_in_pos = model.pos_emb_table[t][k];
                            x_in[k] = (x_in_token + x_in_pos);
                        }
                        x_in_tc.emplace_back(std::move(x_in));                        
                    }
                }

                for (size_t BLOCK = 0, HEAD = 0, FWD = 0; BLOCK < n_layer; ++BLOCK, HEAD += k_heads, ++FWD)
                {
                    assert(k_heads == 6);
                    assert(x_in_tc.size() == k_block_size);

                    constexpr size_t T_ = k_block_size;

                    // layer norm before SA
                    x_in_tc_orig_ = x_in_tc;

                    if (true)
                    {
                        const auto& gamma = gamma_ln_1[BLOCK];
                        const auto& beta = betta_ln_1[BLOCK];

                        for (size_t t = 0; t < T_; ++t)
                        {
                            auto& x = x_in_tc[t];
                            constexpr size_t C = n_emb;
                            assert(x.size() == n_emb);

                            Value<float> x_mean_, mean_square_;
                            assert(!x_mean_.isValid());
                            assert(!mean_square_.isValid());
                            reduceMeanAndMeanSquares<C>(x_mean_, mean_square_, x.data());
                            assert(x_mean_.isValid());
                            assert(mean_square_.isValid());

                            auto x_var_ = mean_square_ - sqr(x_mean_);
                            auto x_sigma_inv_ = invSqrt(x_var_);

                            //#pragma omp simd
                            for (size_t c = 0; c < C; ++c)
                            {
                                auto xhat = (x[c] - x_mean_) * x_sigma_inv_;
                                x[c] = gamma[c] * xhat + beta[c];
                            }
                        }
                    }

                    auto x_in_tc_after_satt_0 = sa_heads[HEAD + 0]->forwardSample<k_block_size, n_emb>(x_in_tc);
                    auto x_in_tc_after_satt_1 = sa_heads[HEAD + 1]->forwardSample<k_block_size, n_emb>(x_in_tc);
                    auto x_in_tc_after_satt_2 = sa_heads[HEAD + 2]->forwardSample<k_block_size, n_emb>(x_in_tc);
                    auto x_in_tc_after_satt_3 = sa_heads[HEAD + 3]->forwardSample<k_block_size, n_emb>(x_in_tc);
                    auto x_in_tc_after_satt_4 = sa_heads[HEAD + 4]->forwardSample<k_block_size, n_emb>(x_in_tc);
                    auto x_in_tc_after_satt_5 = sa_heads[HEAD + 5]->forwardSample<k_block_size, n_emb>(x_in_tc);

                    {
                        const auto& gamma = gamma_ln_2[BLOCK];
                        const auto& beta = betta_ln_2[BLOCK];

                        size_t t_sz = x_in_tc_after_satt_0.size();
                        for (size_t t = 0; t < t_sz; ++t)
                        {
                            auto& x_in_0 = x_in_tc_after_satt_0[t];
                            auto& x_in_1 = x_in_tc_after_satt_1[t];
                            auto& x_in_2 = x_in_tc_after_satt_2[t];
                            auto& x_in_3 = x_in_tc_after_satt_3[t];
                            auto& x_in_4 = x_in_tc_after_satt_4[t];
                            auto& x_in_5 = x_in_tc_after_satt_5[t];

                            assert(x_in_0.size() + x_in_1.size() + x_in_2.size() + x_in_3.size() + x_in_4.size() + x_in_5.size() == n_emb);
                            fwd_projection[FWD]->forward<n_emb, n_emb / k_heads>(x_in_after_proj, { x_in_0.data(), x_in_1.data(), x_in_2.data(), x_in_3.data(), x_in_4.data(), x_in_5.data() });


                            // SA is finished

                            // residual part
                            {
                                assert(x_in_after_proj.size() == n_emb);
                                constexpr size_t C = n_emb;

                                for (size_t c = 0; c < C; ++c)
                                {
                                    // NEW                LN+SA                ORIG
                                    //x_in_after_proj[c] = x_in_after_proj[c] + x_in_tc_orig_[t][c];
                                    x_in_after_proj[c] += x_in_tc_orig_[t][c];
                                }
                            }

                            // FWD MAIN START
                            auto x_in_after_proj_orig = x_in_after_proj;

                            // layer norm before FWD
                            if (true)
                            {
                                auto& x = x_in_after_proj;

                                assert(x.size() == n_emb);
                                constexpr size_t C = n_emb;

                                Value<float> x_mean_, mean_square_;
                                assert(!x_mean_.isValid());
                                assert(!mean_square_.isValid());
                                reduceMeanAndMeanSquares<C>(x_mean_, mean_square_, x.data());
                                assert(x_mean_.isValid());
                                assert(mean_square_.isValid());

                                auto x_var_ = mean_square_ - sqr(x_mean_);
                                auto x_sigma_inv_ = invSqrt(x_var_);

                                //#pragma omp simd
                                for (size_t c = 0; c < C; ++c)
                                {
                                    auto xhat = (x[c] - x_mean_) * x_sigma_inv_;
                                    x[c] = gamma[c] * xhat + beta[c];
                                }
                            }

                            assert(x_in_after_proj.size() == n_emb);
                            fwd_a[FWD]->forward<n_emb>(x_in_after_fwd_a, x_in_after_proj.data());

                            assert(x_in_after_fwd_a.size() == n_project_emb);
                            fwd_b_projection[FWD]->forward<n_project_emb>(x_in_after_fwd_b, x_in_after_fwd_a.data());

                            // FWD MAIN END
                            //  residual part
                            {
                                size_t C = x_in_after_fwd_b.size();
                                for (size_t c = 0; c < C; ++c)
                                {
                                    // NEW                   LN+SA                ORIG
                                    //x_in_after_fwd_b[c] = x_in_after_fwd_b[c] + x_in_after_proj_orig[c];
                                    x_in_after_fwd_b[c] += x_in_after_proj_orig[c];
                                }
                            }

                            // REWRITE X_in_BTC
                            x_in_tc[t] = x_in_after_fwd_b;
                        }
                    }
                }

                assert(x_in_tc.size() == k_block_size);

                constexpr size_t t_sz = k_block_size;              

                for (size_t t = 0; t < t_sz; ++t)
                {
                    const auto& x_in_after_fwd_b = x_in_tc[t];

                    // HEAD
                    assert(x_in_after_fwd_b.size() == n_emb);
                    lm_head.forward<n_emb>(logits, x_in_after_fwd_b.data());

                    for (size_t k = 0; k < model.vocabSize; ++k)
                        countsExp.emplace_back(exp(logits[k]));
                    Value<float> countsExpSum = reduceSum(countsExp.data(), model.vocabSize);

                    // KL(p,q) = \sum (pi * log(pi/qi))
                    // H(p) = -\sum pi * log(pi)
                    // KL(p,q) + H(p) = -\sum (pi * log(qi))
                    // CE(p,q) = -\sum (pi * log(qi))
                    // CE(one-hot) = -log(qi)
                    auto true_label = Y[iSample][t];
                    Value<float> pi = countsExp[true_label] / countsExpSum;
                    Value<float> loss = negativeLog(pi);
                    loss_avg += loss.dataCopy();
                    processed_samples++;

                    {
                        backwardWithScratchStorage<decltype(loss),
                            /*execute_reverse_topo_order*/ true,
                            /*execute_backward_for_internal_nodes*/ true,
                            /*execute_backward_for_leafs*/ false>(loss, reverse_topo_order_seq, reverse_topo_order_set, recursion);
                    }
                }
            }
        }

        float one_inv_processed_samples = 1.0 / float(processed_samples);
        float lr = 3e-4;
        float one_inv_processed_samples_times_lr = one_inv_processed_samples * lr;
        bool kUsedSIMD4Compute = false;

        float grad_len_sqr = float(-1);
        if (kUsedSIMD4Compute)
        {
            Value<float>::applyGDStepWithSIMD(first_trainable_neuron, end_trainable_neuron, one_inv_processed_samples, lr);
        }
        else
        {
            Value<float>::applyGDStep(first_trainable_neuron, end_trainable_neuron, one_inv_processed_samples, lr);
        }

        double time_to_process = timer_to_process.getTimeSec();
        time_to_process_avg = (double(e - 1) / double(e) * time_to_process_avg + time_to_process / double(e));
        time_to_process_sqr_avg = (double(e - 1) / double(e) * time_to_process_sqr_avg + time_to_process * time_to_process / double(e));

        if (e % kIterationsPrintFreq == 0)
        {
            my_log_stream() << "  ->Train: [" << e << '/' << kMaxIterations << "] "
                << " | Loss: " << loss_avg * one_inv_processed_samples
                << " | Grad l2 norm sqr: " << grad_len_sqr << "\n"
                << " | Batch Time: " << time_to_process * 1000 << " msec.\n"
                << " | Avg Time: " << time_to_process_avg * 1000 << " msec.\n"
                << " | Std.dev for time: " << sqrt(fabs(time_to_process_sqr_avg - time_to_process_avg * time_to_process_avg)) * 1000 << " msec.\n"
                << " | Params: " << (end_trainable_neuron - first_trainable_neuron) << "\n";

            if (kPrintDetailedInfo)
            {
                my_log_stream() << "| kilo-samples/sec.: " << processed_samples / time_to_process / 1000.0
                    << "| nodes: " << Value<float>::numActiveNodes() / 1000.0 << "K"
                    << "| processed samples: " << processed_samples << "\n"
                    << "| trainable vars: " << trainable_variables << "\n"
                    << '\n';

                Value<float>::Statistics stats;

                Value<float>::sysCollectStatistics(stats);

                my_log_stream() << "  ->Train Mem. Consumption: [" << e << '/' << kMaxIterations << "]\n"
                    << "| nodes names mem.: " << stats.occupied_memory.labelsMemory / 1024. << " Kb\n"
                    << "| nodes backward type mem.: " << stats.occupied_memory.bwdOpDescrMemory / 1024. << " Kb\n"
                    << "| nodes children topology mem.: " << stats.occupied_memory.childrenTopologyMemory / 1024. << " Kb\n"
                    << "| nodes activations mem.: " << stats.occupied_memory.activationsMemory / 1024. << " Kb\n"
                    << "| nodes grads mem.: " << stats.occupied_memory.gradsMemory / 1024. << " Kb\n\n\n";
            }
        }
    }

    if (kMakePauseAtEnd)
        getchar();

    return 0;
}
