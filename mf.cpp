#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <limits>

#include "mf.h"

#if defined USESSE
#include <pmmintrin.h>
#endif

#if defined USEAVX
#include <immintrin.h>
#endif

#if defined USEOMP
#include <omp.h>
#endif

namespace mf
{

using namespace std;

namespace // unnamed namespace
{

mf_int const kALIGNByte = 32;
mf_int const kALIGN = kALIGNByte/sizeof(mf_float);

//--------------------------------------
//---------Scheduler of Blocks----------
//--------------------------------------

class Scheduler
{
public:
    Scheduler(mf_int nr_bins, mf_int nr_threads, vector<mf_int> cv_blocks);
    mf_int get_job();
    mf_int get_bpr_job(mf_int first_block, bool is_column_oriented);
    void put_job(mf_int block, mf_double loss, mf_double error);
    void put_bpr_job(mf_int first_block, mf_int second_block);
    mf_double get_loss();
    mf_double get_error();
    mf_int get_negative(mf_int first_block, mf_int second_block,
                        mf_int m, mf_int n, bool is_column_oriented);
    void wait_for_jobs_done();
    void resume();
    void terminate();
    bool is_terminated();

private:
    mf_int nr_bins;
    mf_int nr_threads;
    mf_int nr_done_jobs;
    mf_int target;
    mf_int nr_paused_threads;
    bool terminated;
    vector<mf_int> counts;
    vector<mf_int> busy_p_blocks;
    vector<mf_int> busy_q_blocks;
    vector<mf_double> block_losses;
    vector<mf_double> block_errors;
    vector<minstd_rand0> block_generators;
    unordered_set<mf_int> cv_blocks;
    mutex mtx;
    condition_variable cond_var;
    default_random_engine generator;
    uniform_real_distribution<mf_float> distribution;
    priority_queue<pair<mf_float, mf_int>,
                   vector<pair<mf_float, mf_int>>,
                   greater<pair<mf_float, mf_int>>> pq;
};

Scheduler::Scheduler(mf_int nr_bins, mf_int nr_threads,
    vector<mf_int> cv_blocks)
    : nr_bins(nr_bins),
      nr_threads(nr_threads),
      nr_done_jobs(0),
      target(nr_bins*nr_bins),
      nr_paused_threads(0),
      terminated(false),
      counts(nr_bins*nr_bins, 0),
      busy_p_blocks(nr_bins, 0),
      busy_q_blocks(nr_bins, 0),
      block_losses(nr_bins*nr_bins, 0),
      block_errors(nr_bins*nr_bins, 0),
      cv_blocks(cv_blocks.begin(), cv_blocks.end()),
      distribution(0.0, 1.0)
{
    for(mf_int i = 0; i < nr_bins*nr_bins; i++)
    {
        if(this->cv_blocks.find(i) == this->cv_blocks.end())
            pq.emplace(distribution(generator), i);
        block_generators.push_back(minstd_rand0(rand()));
    }
}

mf_int Scheduler::get_job()
{
    bool is_found = false;
    pair<mf_float, mf_int> block;

    while(!is_found)
    {
        lock_guard<mutex> lock(mtx);
        vector<pair<mf_float, mf_int>> locked_blocks;
        mf_int p_block = 0;
        mf_int q_block = 0;

        while(!pq.empty())
        {
            block = pq.top();
            pq.pop();

            p_block = block.second/nr_bins;
            q_block = block.second%nr_bins;

            if(busy_p_blocks[p_block] || busy_q_blocks[q_block])
                locked_blocks.push_back(block);
            else
            {
                busy_p_blocks[p_block] = 1;
                busy_q_blocks[q_block] = 1;
                counts[block.second]++;
                is_found = true;
                break;
            }
        }

        for(auto &block : locked_blocks)
            pq.push(block);
    }

    return block.second;
}

mf_int Scheduler::get_bpr_job(mf_int first_block, bool is_column_oriented)
{
    lock_guard<mutex> lock(mtx);
    mf_int another = first_block;
    vector<pair<mf_float, mf_int>> locked_blocks;

    while(!pq.empty())
    {
        pair<mf_float, mf_int> block = pq.top();
        pq.pop();

        mf_int p_block = block.second/nr_bins;
        mf_int q_block = block.second%nr_bins;

        auto is_rejected = [&] ()
        {
            if(is_column_oriented)
                return first_block%nr_bins != q_block ||
                       busy_p_blocks[p_block];
            else
                return first_block/nr_bins != p_block ||
                         busy_q_blocks[q_block];
        };

        if(is_rejected())
            locked_blocks.push_back(block);
        else
        {
            busy_p_blocks[p_block] = 1;
            busy_q_blocks[q_block] = 1;
            another = block.second;
            break;
        }
    }

    for(auto &block : locked_blocks)
        pq.push(block);

    return another;
}

void Scheduler::put_job(mf_int block_idx, mf_double loss, mf_double error)
{
    {
        lock_guard<mutex> lock(mtx);
        busy_p_blocks[block_idx/nr_bins] = 0;
        busy_q_blocks[block_idx%nr_bins] = 0;
        block_losses[block_idx] = loss;
        block_errors[block_idx] = error;
        nr_done_jobs++;
        mf_float priority =
            (mf_float)counts[block_idx]+distribution(generator);
        pq.emplace(priority, block_idx);
        nr_paused_threads++;
        cond_var.notify_all();
    }

    {
        unique_lock<mutex> lock(mtx);
        cond_var.wait(lock, [&] {
            return nr_done_jobs < target;
        });
    }

    {
        lock_guard<mutex> lock(mtx);
        --nr_paused_threads;
    }
}

void Scheduler::put_bpr_job(mf_int first_block, mf_int second_block)
{
    if(first_block == second_block)
        return;

    lock_guard<mutex> lock(mtx);
    {
        busy_p_blocks[second_block/nr_bins] = 0;
        busy_q_blocks[second_block%nr_bins] = 0;
        mf_float priority =
            (mf_float)counts[second_block]+distribution(generator);
        pq.emplace(priority, second_block);
    }
}

mf_double Scheduler::get_loss()
{
    lock_guard<mutex> lock(mtx);
    return accumulate(block_losses.begin(), block_losses.end(), 0.0);
}

mf_double Scheduler::get_error()
{
    lock_guard<mutex> lock(mtx);
    return accumulate(block_errors.begin(), block_errors.end(), 0.0);
}

mf_int Scheduler::get_negative(mf_int first_block, mf_int second_block,
        mf_int m, mf_int n, bool is_column_oriented)
{
    mf_int rand_val = (mf_int)block_generators[first_block]();

    auto gen_random = [&] (mf_int block_id)
    {
        mf_int v_min, v_max;

        if(is_column_oriented)
        {
            mf_int seg_size = (mf_int)ceil((double)m/nr_bins);
            v_min = min((block_id/nr_bins)*seg_size, m-1);
            v_max = min(v_min+seg_size, m-1);
        }
        else
        {
            mf_int seg_size = (mf_int)ceil((double)n/nr_bins);
            v_min = min((block_id%nr_bins)*seg_size, n-1);
            v_max = min(v_min+seg_size, n-1);
        }
        if(v_max == v_min)
            return v_min;
        else
            return rand_val%(v_max-v_min)+v_min;
    };

    if (rand_val % 2)
        return (mf_int)gen_random(first_block);
    else
        return (mf_int)gen_random(second_block);
}

void Scheduler::wait_for_jobs_done()
{
    unique_lock<mutex> lock(mtx);

    cond_var.wait(lock, [&] {
        return nr_done_jobs >= target;
    });

    cond_var.wait(lock, [&] {
        return nr_paused_threads == nr_threads;
    });
}

void Scheduler::resume()
{
    lock_guard<mutex> lock(mtx);
    target += nr_bins*nr_bins;
    cond_var.notify_all();
}

void Scheduler::terminate()
{
    lock_guard<mutex> lock(mtx);
    terminated = true;
}

bool Scheduler::is_terminated()
{
    lock_guard<mutex> lock(mtx);
    return terminated;
}

//--------------------------------------
//------------Block of matrix-----------
//--------------------------------------

class BlockBase
{
public:
    virtual bool move_next() { return false; };
    virtual mf_node* get_current() { return nullptr; }
    virtual void reload() {};
    virtual void free() {};
    virtual mf_long get_nnz() { return 0; };
    virtual ~BlockBase() {};
};

class Block : public BlockBase
{
public:
    Block() : first(nullptr), last(nullptr), current(nullptr) {};
    Block(mf_node *first_, mf_node *last_)
        : first(first_), last(last_), current(nullptr) {};
    bool move_next() { return ++current != last; }
    mf_node* get_current() { return current; }
    void tie_to(mf_node *first_, mf_node *last_);
    void reload() { current = first-1; };
    mf_long get_nnz() { return last-first; };

private:
    mf_node* first;
    mf_node* last;
    mf_node* current;
};

void Block::tie_to(mf_node *first_, mf_node *last_)
{
    first = first_;
    last = last_;
};

class BlockOnDisk : public BlockBase
{
public:
    BlockOnDisk() : first(0), last(0), current(0),
                    source_path(""), buffer(0) {};
    bool move_next() { return ++current < last-first; }
    mf_node* get_current() { return &buffer[current]; }
    void tie_to(string source_path_, mf_long first_, mf_long last_);
    void reload();
    void free() { buffer.resize(0); };
    mf_long get_nnz() { return last-first; };

private:
    mf_long first;
    mf_long last;
    mf_long current;
    string source_path;
    vector<mf_node> buffer;
};

void BlockOnDisk::tie_to(string source_path_, mf_long first_, mf_long last_)
{
    source_path = source_path_;
    first = first_;
    last = last_;
}

void BlockOnDisk::reload()
{
    ifstream source(source_path, ifstream::in|ifstream::binary);
    if(!source)
        throw runtime_error("can not open "+source_path);

    buffer.resize(last-first);
    source.seekg(first*sizeof(mf_node));
    source.read((char*)buffer.data(), (last-first)*sizeof(mf_node));
    current = -1;
}

//--------------------------------------
//-------------Miscellaneous------------
//--------------------------------------

struct sort_node_by_p
{
    bool operator() (mf_node const &lhs, mf_node const &rhs)
    {
        return tie(lhs.u, lhs.v) < tie(rhs.u, rhs.v);
    }
};

struct sort_node_by_q
{
    bool operator() (mf_node const &lhs, mf_node const &rhs)
    {
        return tie(lhs.v, lhs.u) < tie(rhs.v, rhs.u);
    }
};

class Utility
{
public:
    Utility(mf_int f, mf_int n) : fun(f), nr_threads(n) {};
    void collect_info(mf_problem &prob, mf_float &avg, mf_float &std_dev);
    void collect_info_on_disk(string data_path, mf_problem &prob,
                              mf_float &avg, mf_float &std_dev);
    void shuffle_problem(mf_problem &prob, vector<mf_int> &p_map,
                         vector<mf_int> &q_map);
    vector<mf_node*> grid_problem(mf_problem &prob, mf_int nr_bins,
                                  vector<mf_int> &omega_p,
                                  vector<mf_int> &omega_q,
                                  vector<Block> &blocks);
    void grid_shuffle_scale_problem_on_disk(mf_int m, mf_int n, mf_int nr_bins,
                                            mf_float scale, string data_path,
                                            vector<mf_int> &p_map,
                                            vector<mf_int> &q_map,
                                            vector<mf_int> &omega_p,
                                            vector<mf_int> &omega_q,
                                            vector<BlockOnDisk> &blocks);
    void scale_problem(mf_problem &prob, mf_float scale);
    mf_double calc_reg1(mf_model &model, mf_float lambda_p, mf_float lambda_q,
                        vector<mf_int> &omega_p, vector<mf_int> &omega_q);
    mf_double calc_reg2(mf_model &model, mf_float lambda_p, mf_float lambda_q,
                        vector<mf_int> &omega_p, vector<mf_int> &omega_q);
    string get_error_legend();
    mf_double calc_error(vector<BlockBase*> &blocks,
                         vector<mf_int> &cv_block_ids,
                         mf_model const &model);
    void scale_model(mf_model &model, mf_float scale);

    static mf_problem* copy_problem(mf_problem const *prob, bool copy_data);
    static vector<mf_int> gen_random_map(mf_int size);
    static mf_float* malloc_aligned_float(mf_long size);
    static mf_model* init_model(mf_int loss, mf_int m, mf_int n,
                                mf_int k, mf_float avg,
                                vector<mf_int> &omega_p,
                                vector<mf_int> &omega_q);
    static mf_float inner_product(mf_float *p, mf_float *q, mf_int k);
    static vector<mf_int> gen_inv_map(vector<mf_int> &map);
    static void shrink_model(mf_model &model, mf_int k_new);
    static void shuffle_model(mf_model &model,
                              vector<mf_int> &p_map,
                              vector<mf_int> &q_map);

private:
    mf_int fun;
    mf_int nr_threads;
};

void Utility::collect_info(
    mf_problem &prob,
    mf_float &avg,
    mf_float &std_dev)
{
    mf_double ex = 0;
    mf_double ex2 = 0;

#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:ex,ex2)
#endif
    for(mf_long i = 0; i < prob.nnz; i++)
    {
        mf_node &N = prob.R[i];
        ex += (mf_double)N.r;
        ex2 += (mf_double)N.r*N.r;
    }

    ex /= (mf_double)prob.nnz;
    ex2 /= (mf_double)prob.nnz;
    avg = (mf_float)ex;
    std_dev = (mf_float)sqrt(ex2-ex*ex);
}

void Utility::collect_info_on_disk(
    string data_path,
    mf_problem &prob,
    mf_float &avg,
    mf_float &std_dev)
{
    mf_double ex = 0;
    mf_double ex2 = 0;

    ifstream source(data_path);
    if(!source.is_open())
        throw runtime_error("cannot open " + data_path);

    for(mf_node N; source >> N.u >> N.v >> N.r;)
    {
        if(N.u+1 > prob.m)
            prob.m = N.u+1;
        if(N.v+1 > prob.n)
            prob.n = N.v+1;
        prob.nnz++;
        ex += (mf_double)N.r;
        ex2 += (mf_double)N.r*N.r;
    }
    source.close();

    ex /= (mf_double)prob.nnz;
    ex2 /= (mf_double)prob.nnz;
    avg = (mf_float)ex;
    std_dev = (mf_float)sqrt(ex2-ex*ex);
}

void Utility::scale_problem(mf_problem &prob, mf_float scale)
{
    if(scale == 1.0)
        return;

#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static)
#endif
    for(mf_long i = 0; i < prob.nnz; i++)
        prob.R[i].r *= scale;
}

void Utility::scale_model(mf_model &model, mf_float scale)
{
    if(scale == 1.0)
        return;

    mf_int k = model.k;

    model.b *= scale;

    auto scale1 = [&] (mf_float *ptr, mf_int size, mf_float factor_scale)
    {
#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static)
#endif
        for(mf_int i = 0; i < size; i++)
        {
            mf_float *ptr1 = ptr+(mf_long)i*model.k;
            for(mf_int d = 0; d < k; d++)
                ptr1[d] *= factor_scale;
        }
    };

    scale1(model.P, model.m, sqrt(scale));
    scale1(model.Q, model.n, sqrt(scale));
}

mf_float Utility::inner_product(mf_float *p, mf_float *q, mf_int k)
{
#if defined USESSE
    __m128 XMM = _mm_setzero_ps();
    for(mf_int d = 0; d < k; d += 4)
        XMM = _mm_add_ps(XMM, _mm_mul_ps(
                  _mm_load_ps(p+d), _mm_load_ps(q+d)));
    XMM = _mm_hadd_ps(XMM, XMM);
    XMM = _mm_hadd_ps(XMM, XMM);
    mf_float product;
    _mm_store_ss(&product, XMM);
    return product;
#elif defined USEAVX
    __m256 XMM = _mm256_setzero_ps();
    for(mf_int d = 0; d < k; d += 8)
        XMM = _mm256_add_ps(XMM, _mm256_mul_ps(
                  _mm256_load_ps(p+d), _mm256_load_ps(q+d)));
    XMM = _mm256_add_ps(XMM, _mm256_permute2f128_ps(XMM, XMM, 1));
    XMM = _mm256_hadd_ps(XMM, XMM);
    XMM = _mm256_hadd_ps(XMM, XMM);
    mf_float product;
    _mm_store_ss(&product, _mm256_castps256_ps128(XMM));
    return product;
#else
    return std::inner_product(p, p+k, q, (mf_float)0.0);
#endif
}

mf_double Utility::calc_reg1(mf_model &model,
                             mf_float lambda_p, mf_float lambda_q,
                             vector<mf_int> &omega_p, vector<mf_int> &omega_q)
{
    auto calc_reg1_core = [&] (mf_float *ptr, mf_int size,
                               vector<mf_int> &omega)
    {
        mf_double reg = 0;
        for(mf_int i = 0; i < size; i++)
        {
            if(omega[i] <= 0)
                continue;

            mf_float tmp = 0;
            for(mf_int j = 0; j < model.k; j++)
                tmp += abs(ptr[i*model.k+j]);
            reg += omega[i]*tmp;
        }
        return reg;
    };

    return lambda_p*calc_reg1_core(model.P, model.m, omega_p)+
           lambda_q*calc_reg1_core(model.Q, model.n, omega_q);
}

mf_double Utility::calc_reg2(mf_model &model,
                             mf_float lambda_p, mf_float lambda_q,
                             vector<mf_int> &omega_p, vector<mf_int> &omega_q)
{
    auto calc_reg2_core = [&] (mf_float *ptr, mf_int size,
                               vector<mf_int> &omega)
    {
        mf_double reg = 0;
#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:reg)
#endif
        for(mf_int i = 0; i < size; i++)
        {
            if(omega[i] <= 0)
                continue;

            mf_float *ptr1 = ptr+(mf_long)i*model.k;
            reg += omega[i]*Utility::inner_product(ptr1, ptr1, model.k);
        }

        return reg;
    };

    return lambda_p*calc_reg2_core(model.P, model.m, omega_p) +
           lambda_q*calc_reg2_core(model.Q, model.n, omega_q);
}

mf_double Utility::calc_error(
    vector<BlockBase*> &blocks,
    vector<mf_int> &cv_block_ids,
    mf_model const &model)
{
    mf_double error = 0;
    if(fun == P_L2_MFR || fun == P_L1_MFR || fun == P_KL_MFR ||
       fun == P_LR_MFC || fun == P_L2_MFC || fun == P_L1_MFC)
    {
#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:error)
#endif
        for(mf_int i = 0; i < (mf_long)cv_block_ids.size(); i++)
        {
            BlockBase *block = blocks[cv_block_ids[i]];
            block->reload();
            while(block->move_next())
            {
                mf_node const &N = *(block->get_current());
                mf_float z = mf_predict(&model, N.u, N.v);
                switch(fun)
                {
                    case P_L2_MFR:
                        error += pow(N.r-z, 2);
                        break;
                    case P_L1_MFR:
                        error += abs(N.r-z);
                        break;
                    case P_KL_MFR:
                        error += N.r*log(N.r/z)-N.r+z;
                        break;
                    case P_LR_MFC:
                        if(N.r > 0)
                            error += log(1.0+exp(-z));
                        else
                            error += log(1.0+exp(z));
                        break;
                    case P_L2_MFC:
                    case P_L1_MFC:
                        if(N.r > 0)
                            error += z > 0? 1: 0;
                        else
                            error += z < 0? 1: 0;
                        break;
                    default:
                        throw invalid_argument("unknown error function");
                        break;
                }
            }
            block->free();
        }
    }
    else
    {
        minstd_rand0 generator(rand());
        switch(fun)
        {
            case P_ROW_BPR_MFOC:
            {
                uniform_int_distribution<mf_int> distribution(0, model.n-1);
#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:error)
#endif
                for(mf_int i = 0; i < (mf_long)cv_block_ids.size(); i++)
                {
                    BlockBase *block = blocks[cv_block_ids[i]];
                    block->reload();
                    while(block->move_next())
                    {
                        mf_node const &N = *(block->get_current());
                        mf_int w = distribution(generator);
                        error += log(1+exp(mf_predict(&model, N.u, w)-
                                           mf_predict(&model, N.u, N.v)));
                    }
                    block->free();
                }
                break;
            }
            case P_COL_BPR_MFOC:
            {
                uniform_int_distribution<mf_int> distribution(0, model.m-1);
#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static) reduction(+:error)
#endif
                for(mf_int i = 0; i < (mf_long)cv_block_ids.size(); i++)
                {
                    BlockBase *block = blocks[cv_block_ids[i]];
                    block->reload();
                    while(block->move_next())
                    {
                        mf_node const &N = *(block->get_current());
                        mf_int w = distribution(generator);
                        error += log(1+exp(mf_predict(&model, w, N.v)-
                                           mf_predict(&model, N.u, N.v)));
                    }
                    block->free();
                }
                break;
            }
            default:
            {
                throw invalid_argument("unknown error function");
                break;
            }
        }
    }

    return error;
}

string Utility::get_error_legend()
{
    switch(fun)
    {
        case P_L2_MFR:
            return string("rmse");
            break;
        case P_L1_MFR:
            return string("mae");
            break;
        case P_KL_MFR:
            return string("gkl");
            break;
        case P_LR_MFC:
            return string("logloss");
            break;
        case P_L2_MFC:
        case P_L1_MFC:
            return string("accuracy");
            break;
        case P_ROW_BPR_MFOC:
        case P_COL_BPR_MFOC:
            return string("bprloss");
            break;
        default:
            return string();
            break;
     }
}

void Utility::shuffle_problem(
    mf_problem &prob,
    vector<mf_int> &p_map,
    vector<mf_int> &q_map)
{
#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(static)
#endif
    for(mf_long i = 0; i < prob.nnz; i++)
    {
        mf_node &N = prob.R[i];
        if(N.u < (mf_long)p_map.size())
            N.u = p_map[N.u];
        if(N.v < (mf_long)q_map.size())
            N.v = q_map[N.v];
    }
}

vector<mf_node*> Utility::grid_problem(
    mf_problem &prob,
    mf_int nr_bins,
    vector<mf_int> &omega_p,
    vector<mf_int> &omega_q,
    vector<Block> &blocks)
{
    vector<mf_long> counts(nr_bins*nr_bins, 0);

    mf_int seg_p = (mf_int)ceil((double)prob.m/nr_bins);
    mf_int seg_q = (mf_int)ceil((double)prob.n/nr_bins);

    auto get_block_id = [=] (mf_int u, mf_int v)
    {
        return (u/seg_p)*nr_bins+v/seg_q;
    };

    for(mf_long i = 0; i < prob.nnz; i++)
    {
        mf_node &N = prob.R[i];
        mf_int block = get_block_id(N.u, N.v);
        counts[block]++;
        omega_p[N.u]++;
        omega_q[N.v]++;
    }

    vector<mf_node*> ptrs(nr_bins*nr_bins+1);
    mf_node *ptr = prob.R;
    ptrs[0] = ptr;
    for(mf_int block = 0; block < nr_bins*nr_bins; block++)
        ptrs[block+1] = ptrs[block] + counts[block];

    vector<mf_node*> pivots(ptrs.begin(), ptrs.end()-1);
    for(mf_int block = 0; block < nr_bins*nr_bins; block++)
    {
        for(mf_node* pivot = pivots[block]; pivot != ptrs[block+1];)
        {
            mf_int curr_block = get_block_id(pivot->u, pivot->v);
            if(curr_block == block)
            {
                pivot++;
                continue;
            }

            mf_node *next = pivots[curr_block];
            swap(*pivot, *next);
            pivots[curr_block]++;
        }
    }

#if defined USEOMP
#pragma omp parallel for num_threads(nr_threads) schedule(dynamic)
#endif
    for(mf_int block = 0; block < nr_bins*nr_bins; block++)
    {
        if(prob.m > prob.n)
            sort(ptrs[block], ptrs[block+1], sort_node_by_p());
        else
            sort(ptrs[block], ptrs[block+1], sort_node_by_q());
    }

    for(mf_int i = 0; i < (mf_long)blocks.size(); i++)
        blocks[i].tie_to(ptrs[i], ptrs[i+1]);

    return ptrs;
}

void Utility::grid_shuffle_scale_problem_on_disk(
    mf_int m, mf_int n, mf_int nr_bins,
    mf_float scale, string data_path,
    vector<mf_int> &p_map, vector<mf_int> &q_map,
    vector<mf_int> &omega_p, vector<mf_int> &omega_q,
    vector<BlockOnDisk> &blocks)
{
    string const buffer_path = data_path+string(".disk");
    mf_int seg_p = (mf_int)ceil((double)m/nr_bins);
    mf_int seg_q = (mf_int)ceil((double)n/nr_bins);
    vector<mf_long> counts(nr_bins*nr_bins+1, 0);
    vector<mf_long> pivots(nr_bins*nr_bins, 0);
    ifstream source(data_path);
    fstream buffer(buffer_path, fstream::in|fstream::out|
                   fstream::binary|fstream::trunc);
    auto get_block_id = [=] (mf_int u, mf_int v)
    {
        return (u/seg_p)*nr_bins+v/seg_q;
    };

    if(!source)
        throw ios::failure(string("cannot to open ")+data_path);
    if(!buffer)
        throw ios::failure(string("cannot to open ")+buffer_path);

    for(mf_node N; source >> N.u >> N.v >> N.r;)
    {
        N.u = p_map[N.u];
        N.v = q_map[N.v];
        mf_int bid = get_block_id(N.u, N.v);
        omega_p[N.u]++;
        omega_q[N.v]++;
        counts[bid+1]++;
    }

    for(mf_int i = 1; i < nr_bins*nr_bins+1; i++)
    {
        counts[i] += counts[i-1];
        pivots[i-1] = counts[i-1];
    }

    source.clear();
    source.seekg(0);
    for(mf_node N; source >> N.u >> N.v >> N.r;)
    {
        N.u = p_map[N.u];
        N.v = q_map[N.v];
        N.r /= scale;
        mf_int bid = get_block_id(N.u, N.v);
        buffer.seekp(pivots[bid]*sizeof(mf_node));
        buffer.write((char*)&N, sizeof(mf_node));
        pivots[bid]++;
    }

    for(mf_int i = 0; i < nr_bins*nr_bins; i++)
    {
        vector<mf_node> nodes(counts[i+1]-counts[i]);
        buffer.clear();
        buffer.seekg(counts[i]*sizeof(mf_node));
        buffer.read((char*)nodes.data(), sizeof(mf_node)*nodes.size());

        if(m > n)
            sort(nodes.begin(), nodes.end(), sort_node_by_p());
        else
            sort(nodes.begin(), nodes.end(), sort_node_by_q());

        buffer.clear();
        buffer.seekp(counts[i]*sizeof(mf_node));
        buffer.write((char*)nodes.data(), sizeof(mf_node)*nodes.size());
        buffer.read((char*)nodes.data(), sizeof(mf_node)*nodes.size());
    }

    for(mf_int i = 0; i < (mf_long)blocks.size(); i++)
        blocks[i].tie_to(buffer_path, counts[i], counts[i+1]);
}

mf_float* Utility::malloc_aligned_float(mf_long size)
{
    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size*sizeof(mf_float), kALIGNByte);
    if(ptr == nullptr)
        throw bad_alloc();
#else
    int status = posix_memalign(&ptr, kALIGNByte, size*sizeof(mf_float));
    if(status != 0)
        throw bad_alloc();
#endif

    return (mf_float*)ptr;
}

mf_model* Utility::init_model(mf_int fun,
                              mf_int m, mf_int n,
                              mf_int k, mf_float avg,
                              vector<mf_int> &omega_p,
                              vector<mf_int> &omega_q)
{
    mf_int k_real = k;
    mf_int k_aligned = (mf_int)ceil(mf_double(k)/kALIGN)*kALIGN;

    mf_model *model = new mf_model;

    model->fun = fun;
    model->m = m;
    model->n = n;
    model->k = k_aligned;
    model->b = avg;
    model->P = nullptr;
    model->Q = nullptr;

    mf_float scale = (mf_float)sqrt(1.0/k_real);
    default_random_engine generator;
    uniform_real_distribution<mf_float> distribution(0.0, 1.0);

    try
    {
        model->P = Utility::malloc_aligned_float((mf_long)model->m*model->k);
        model->Q = Utility::malloc_aligned_float((mf_long)model->n*model->k);
    }
    catch(bad_alloc const &e)
    {
        cerr << e.what() << endl;
        mf_destroy_model(&model);
        throw;
    }

    auto init1 = [&](mf_float *start_ptr, mf_long size, vector<mf_int> counts)
    {
        memset(start_ptr, 0, sizeof(mf_float)*size*model->k);
        for(mf_long i = 0; i < size; i++)
        {
            mf_float * ptr = start_ptr + i*model->k;
            if(counts[i] > 0)
                for(mf_long d = 0; d < k_real; d++, ptr++)
                    *ptr = (mf_float)(distribution(generator)*scale);
            else
                if(fun != P_ROW_BPR_MFOC && fun != P_COL_BPR_MFOC) // unseen for bpr is 0
                    for(mf_long d = 0; d < k_real; d++, ptr++)
                        *ptr = numeric_limits<mf_float>::quiet_NaN();
        }
    };

    init1(model->P, m, omega_p);
    init1(model->Q, n, omega_q);

    return model;
}

vector<mf_int> Utility::gen_random_map(mf_int size)
{
    srand(0);
    vector<mf_int> map(size, 0);
    for(mf_int i = 0; i < size; i++)
        map[i] = i;
    random_shuffle(map.begin(), map.end());
    return map;
}

vector<mf_int> Utility::gen_inv_map(vector<mf_int> &map)
{
    vector<mf_int> inv_map(map.size());
    for(mf_int i = 0; i < (mf_long)map.size(); i++)
      inv_map[map[i]] = i;
    return inv_map;
}

void Utility::shuffle_model(
    mf_model &model,
    vector<mf_int> &p_map,
    vector<mf_int> &q_map)
{
    auto inv_shuffle1 = [] (mf_float *vec, vector<mf_int> &map,
                            mf_int size, mf_int k)
    {
        for(mf_int pivot = 0; pivot < size;)
        {
            if(pivot == map[pivot])
            {
                ++pivot;
                continue;
            }

            mf_int next = map[pivot];

            for(mf_int d = 0; d < k; d++)
                swap(*(vec+(mf_long)pivot*k+d), *(vec+(mf_long)next*k+d));

            map[pivot] = map[next];
            map[next] = next;
        }
    };

    inv_shuffle1(model.P, p_map, model.m, model.k);
    inv_shuffle1(model.Q, q_map, model.n, model.k);
}

void Utility::shrink_model(mf_model &model, mf_int k_new)
{
    mf_int k_old = model.k;
    model.k = k_new;

    auto shrink1 = [&] (mf_float *ptr, mf_int size)
    {
        for(mf_int i = 0; i < size; i++)
        {
            mf_float *src = ptr+(mf_long)i*k_old;
            mf_float *dst = ptr+(mf_long)i*k_new;
            copy(src, src+k_new, dst);
        }
    };

    shrink1(model.P, model.m);
    shrink1(model.Q, model.n);
}

mf_problem* Utility::copy_problem(mf_problem const *prob, bool copy_data)
{
    mf_problem *new_prob = new mf_problem;

    if(prob == nullptr)
    {
        new_prob->m = 0;
        new_prob->n = 0;
        new_prob->nnz = 0;
        new_prob->R = nullptr;

        return new_prob;
    }

    new_prob->m = prob->m;
    new_prob->n = prob->n;
    new_prob->nnz = prob->nnz;

    if(copy_data)
    {
        try
        {
            new_prob->R = new mf_node[prob->nnz];
            copy(prob->R, prob->R+prob->nnz, new_prob->R);
        }
        catch(...)
        {
            delete new_prob;
            throw;
        }
    }
    else
    {
        new_prob->R = prob->R;
    }

    return new_prob;
}

//--------------------------------------
//-----The base class of all solvers----
//--------------------------------------

class SolverBase
{
public:
    SolverBase(Scheduler &scheduler, vector<BlockBase*> &blocks,
               mf_float *PG, mf_float *QG, mf_model &model, mf_parameter param,
               bool &slow_only)
        : scheduler(scheduler), blocks(blocks), PG(PG), QG(QG),
          model(model), param(param), slow_only(slow_only) {}
    void run();
    SolverBase(const SolverBase&) = delete;
    SolverBase& operator=(const SolverBase&) = delete;

protected:
#if defined USESSE
    static void calc_z(__m128 &XMMz, mf_int k, mf_float *p, mf_float *q);
    virtual void load_fixed_variables(
        __m128 &XMMlambda_p1, __m128 &XMMlambda_q1,
        __m128 &XMMlambda_p2, __m128 &XMMlabmda_q2,
        __m128 &XMMeta, __m128 &XMMrk_slow,
        __m128 &XMMrk_fast);
    virtual void arrange_block(__m128d &XMMloss, __m128d &XMMerror);
    virtual void prepare_for_sg_update(
        __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror) = 0;
    virtual void sg_update(mf_int d_begin, mf_int d_end, __m128 XMMz,
                           __m128 XMMlambda_p1, __m128 XMMlambda_q1,
                           __m128 XMMlambda_p2, __m128 XMMlamdba_q2,
                           __m128 XMMeta, __m128 XMMrk) = 0;
    virtual void finalize(__m128d XMMloss, __m128d XMMerror);
#elif defined USEAVX
    static void calc_z(__m256 &XMMz, mf_int k, mf_float *p, mf_float *q);
    virtual void load_fixed_variables(
        __m256 &XMMlambda_p1, __m256 &XMMlambda_q1,
        __m256 &XMMlambda_p2, __m256 &XMMlabmda_q2,
        __m256 &XMMeta, __m256 &XMMrk_slow,
        __m256 &XMMrk_fast);
    virtual void arrange_block(__m128d &XMMloss, __m128d &XMMerror);
    virtual void prepare_for_sg_update(
        __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror) = 0;
    virtual void sg_update(mf_int d_begin, mf_int d_end, __m256 XMMz,
                           __m256 XMMlambda_p1, __m256 XMMlambda_q1,
                           __m256 XMMlambda_p2, __m256 XMMlamdba_q2,
                           __m256 XMMeta, __m256 XMMrk) = 0;
    virtual void finalize(__m128d XMMloss, __m128d XMMerror);
#else
    static void calc_z(mf_float &z, mf_int k, mf_float *p, mf_float *q);
    virtual void load_fixed_variables();
    virtual void arrange_block();
    virtual void prepare_for_sg_update() = 0;
    virtual void sg_update(mf_int d_begin, mf_int d_end, mf_float rk) = 0;
    virtual void finalize();
    static float qrsqrt(float x);
#endif
    virtual void update() { pG++; qG++; };

    Scheduler &scheduler;
    vector<BlockBase*> &blocks;
    BlockBase *block;
    mf_float *PG;
    mf_float *QG;
    mf_model &model;
    mf_parameter param;
    bool &slow_only;

    mf_node *N;
    mf_float z;
    mf_double loss;
    mf_double error;
    mf_float *p;
    mf_float *q;
    mf_float *pG;
    mf_float *qG;
    mf_int bid;

    mf_float lambda_p1;
    mf_float lambda_q1;
    mf_float lambda_p2;
    mf_float lambda_q2;
    mf_float rk_slow;
    mf_float rk_fast;
};

#if defined USESSE
inline void SolverBase::run()
{
    __m128d XMMloss;
    __m128d XMMerror;
    __m128 XMMz;
    __m128 XMMlambda_p1;
    __m128 XMMlambda_q1;
    __m128 XMMlambda_p2;
    __m128 XMMlambda_q2;
    __m128 XMMeta;
    __m128 XMMrk_slow;
    __m128 XMMrk_fast;
    load_fixed_variables(XMMlambda_p1, XMMlambda_q1,
                         XMMlambda_p2, XMMlambda_q2,
                         XMMeta, XMMrk_slow,
                         XMMrk_fast);
    while(!scheduler.is_terminated())
    {
        arrange_block(XMMloss, XMMerror);
        while(block->move_next())
        {
            N = block->get_current();
            p = model.P+(mf_long)N->u*model.k;
            q = model.Q+(mf_long)N->v*model.k;
            pG = PG+N->u*2;
            qG = QG+N->v*2;
            prepare_for_sg_update(XMMz, XMMloss, XMMerror);
            sg_update(0, kALIGN, XMMz, XMMlambda_p1, XMMlambda_q1,
                    XMMlambda_p2, XMMlambda_q2, XMMeta, XMMrk_slow);
            if(slow_only)
                continue;
            update();
            sg_update(kALIGN, model.k, XMMz, XMMlambda_p1, XMMlambda_q1,
                    XMMlambda_p2, XMMlambda_q2, XMMeta, XMMrk_slow);
        }
        finalize(XMMloss, XMMerror);
    }
}

void SolverBase::load_fixed_variables(
    __m128 &XMMlambda_p1, __m128 &XMMlambda_q1,
    __m128 &XMMlambda_p2, __m128 &XMMlambda_q2,
    __m128 &XMMeta, __m128 &XMMrk_slow,
    __m128 &XMMrk_fast)
{
    XMMlambda_p1 = _mm_set1_ps(param.lambda_p1);
    XMMlambda_q1 = _mm_set1_ps(param.lambda_q1);
    XMMlambda_p2 = _mm_set1_ps(param.lambda_p2);
    XMMlambda_q2 = _mm_set1_ps(param.lambda_q2);
    XMMeta = _mm_set1_ps(param.eta);
    XMMrk_slow = _mm_set1_ps((mf_float)1.0/kALIGN);
    XMMrk_fast = _mm_set1_ps((mf_float)1.0/(model.k-kALIGN));
}

void SolverBase::arrange_block(__m128d &XMMloss, __m128d &XMMerror)
{
    XMMloss = _mm_setzero_pd();
    XMMerror = _mm_setzero_pd();
    bid = scheduler.get_job();
    block = blocks[bid];
    block->reload();
}

inline void SolverBase::calc_z(
    __m128 &XMMz, mf_int k, mf_float *p, mf_float *q)
{
    XMMz = _mm_setzero_ps();
    for(mf_int d = 0; d < k; d += 4)
        XMMz = _mm_add_ps(XMMz, _mm_mul_ps(
               _mm_load_ps(p+d), _mm_load_ps(q+d)));
    XMMz = _mm_hadd_ps(XMMz, XMMz);
    XMMz = _mm_hadd_ps(XMMz, XMMz);
}

void SolverBase::finalize(__m128d XMMloss, __m128d XMMerror)
{
    _mm_store_sd(&loss, XMMloss);
    _mm_store_sd(&error, XMMerror);
    block->free();
    scheduler.put_job(bid, loss, error);
}
#elif defined USEAVX
inline void SolverBase::run()
{
    __m128d XMMloss;
    __m128d XMMerror;
    __m256 XMMz;
    __m256 XMMlambda_p1;
    __m256 XMMlambda_q1;
    __m256 XMMlambda_p2;
    __m256 XMMlambda_q2;
    __m256 XMMeta;
    __m256 XMMrk_slow;
    __m256 XMMrk_fast;
    load_fixed_variables(XMMlambda_p1, XMMlambda_q1,
                         XMMlambda_p2, XMMlambda_q2,
                         XMMeta, XMMrk_slow, XMMrk_fast);
    while(!scheduler.is_terminated())
    {
        arrange_block(XMMloss, XMMerror);
        while(block->move_next())
        {
            N = block->get_current();
            p = model.P+(mf_long)N->u*model.k;
            q = model.Q+(mf_long)N->v*model.k;
            pG = PG+N->u*2;
            qG = QG+N->v*2;
            prepare_for_sg_update(XMMz, XMMloss, XMMerror);
            sg_update(0, kALIGN, XMMz, XMMlambda_p1, XMMlambda_q1,
                      XMMlambda_p2, XMMlambda_q2, XMMeta, XMMrk_slow);
            if(slow_only)
                continue;
            update();
            sg_update(kALIGN, model.k, XMMz, XMMlambda_p1, XMMlambda_q1,
                      XMMlambda_p2, XMMlambda_q2, XMMeta, XMMrk_fast);
        }
        finalize(XMMloss, XMMerror);
    }
}

void SolverBase::load_fixed_variables(
    __m256 &XMMlambda_p1, __m256 &XMMlambda_q1,
    __m256 &XMMlambda_p2, __m256 &XMMlambda_q2,
    __m256 &XMMeta, __m256 &XMMrk_slow,
    __m256 &XMMrk_fast)
{
    XMMlambda_p1 = _mm256_set1_ps(param.lambda_p1);
    XMMlambda_q1 = _mm256_set1_ps(param.lambda_q1);
    XMMlambda_p2 = _mm256_set1_ps(param.lambda_p2);
    XMMlambda_q2 = _mm256_set1_ps(param.lambda_q2);
    XMMeta = _mm256_set1_ps(param.eta);
    XMMrk_slow = _mm256_set1_ps((mf_float)1.0/kALIGN);
    XMMrk_fast = _mm256_set1_ps((mf_float)1.0/(model.k-kALIGN));
}

void SolverBase::arrange_block(__m128d &XMMloss, __m128d &XMMerror)
{
    XMMloss = _mm_setzero_pd();
    XMMerror = _mm_setzero_pd();
    bid = scheduler.get_job();
    block = blocks[bid];
    block->reload();
}

inline void SolverBase::calc_z(
    __m256 &XMMz, mf_int k, mf_float *p, mf_float *q)
{
    XMMz = _mm256_setzero_ps();
    for(mf_int d = 0; d < k; d += 8)
        XMMz = _mm256_add_ps(XMMz, _mm256_mul_ps(
               _mm256_load_ps(p+d), _mm256_load_ps(q+d)));
    XMMz = _mm256_add_ps(XMMz, _mm256_permute2f128_ps(XMMz, XMMz, 0x1));
    XMMz = _mm256_hadd_ps(XMMz, XMMz);
    XMMz = _mm256_hadd_ps(XMMz, XMMz);
}

void SolverBase::finalize(__m128d XMMloss, __m128d XMMerror)
{
    _mm_store_sd(&loss, XMMloss);
    _mm_store_sd(&error, XMMerror);
    block->free();
    scheduler.put_job(bid, loss, error);
}
#else
inline void SolverBase::run()
{
    load_fixed_variables();
    while(!scheduler.is_terminated())
    {
        arrange_block();
        while(block->move_next())
        {
            N = block->get_current();
            p = model.P+(mf_long)N->u*model.k;
            q = model.Q+(mf_long)N->v*model.k;
            pG = PG+N->u*2;
            qG = QG+N->v*2;
            prepare_for_sg_update();
            sg_update(0, kALIGN, rk_slow);
            if(slow_only)
                continue;
            update();
            sg_update(kALIGN, model.k, rk_fast);
        }
        finalize();
    }
}

inline float SolverBase::qrsqrt(float x)
{
    float xhalf = 0.5f*x;
    uint32_t i;
    memcpy(&i, &x, sizeof(i));
    i = 0x5f375a86 - (i>>1);
    memcpy(&x, &i, sizeof(i));
    x = x*(1.5f - xhalf*x*x);
    return x;
}

void SolverBase::load_fixed_variables()
{
    lambda_p1 = param.lambda_p1;
    lambda_q1 = param.lambda_q1;
    lambda_p2 = param.lambda_p2;
    lambda_q2 = param.lambda_q2;
    rk_slow = (mf_float)1.0/kALIGN;
    rk_fast = (mf_float)1.0/(model.k-kALIGN);
}

void SolverBase::arrange_block()
{
    loss = 0.0;
    error = 0.0;
    bid = scheduler.get_job();
    block = blocks[bid];
    block->reload();
}

inline void SolverBase::calc_z(mf_float &z, mf_int k, mf_float *p, mf_float *q)
{
    z = 0;
    for(mf_int d = 0; d < k; d++)
        z += p[d]*q[d];
}

void SolverBase::finalize()
{
    block->free();
    scheduler.put_job(bid, loss, error);
}
#endif

//--------------------------------------
//-----Real-valued MF and binary MF-----
//--------------------------------------

class MFSolver: public SolverBase
{
public:
    MFSolver(Scheduler &scheduler, vector<BlockBase*> &blocks,
             mf_float *PG, mf_float *QG, mf_model &model,
             mf_parameter param, bool &slow_only)
        : SolverBase(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
#if defined USESSE
    void sg_update(mf_int d_begin, mf_int d_end, __m128 XMMz,
                   __m128 XMMlambda_p1, __m128 XMMlambda_q1,
                   __m128 XMMlambda_p2, __m128 XMMlambda_q2,
                   __m128 XMMeta, __m128 XMMrk);
#elif defined USEAVX
    void sg_update(mf_int d_begin, mf_int d_end, __m256 XMMz,
                   __m256 XMMlambda_p1, __m256 XMMlambda_q1,
                   __m256 XMMlambda_p2, __m256 XMMlambda_q2,
                   __m256 XMMeta, __m256 XMMrk);
#else
    void sg_update(mf_int d_begin, mf_int d_end, mf_float rk);
#endif
};

#if defined USESSE
void MFSolver::sg_update(mf_int d_begin, mf_int d_end, __m128 XMMz,
                                __m128 XMMlambda_p1, __m128 XMMlambda_q1,
                                __m128 XMMlambda_p2, __m128 XMMlambda_q2,
                                __m128 XMMeta, __m128 XMMrk)
{
    __m128 XMMpG = _mm_load1_ps(pG);
    __m128 XMMqG = _mm_load1_ps(qG);
    __m128 XMMeta_p = _mm_mul_ps(XMMeta, _mm_rsqrt_ps(XMMpG));
    __m128 XMMeta_q = _mm_mul_ps(XMMeta, _mm_rsqrt_ps(XMMqG));
    __m128 XMMpG1 = _mm_setzero_ps();
    __m128 XMMqG1 = _mm_setzero_ps();

    for(mf_int d = d_begin; d < d_end; d += 4)
    {
        __m128 XMMp = _mm_load_ps(p+d);
        __m128 XMMq = _mm_load_ps(q+d);

        __m128 XMMpg = _mm_sub_ps(_mm_mul_ps(XMMlambda_p2, XMMp),
                       _mm_mul_ps(XMMz, XMMq));
        __m128 XMMqg = _mm_sub_ps(_mm_mul_ps(XMMlambda_q2, XMMq),
                       _mm_mul_ps(XMMz, XMMp));

        XMMpG1 = _mm_add_ps(XMMpG1, _mm_mul_ps(XMMpg, XMMpg));
        XMMqG1 = _mm_add_ps(XMMqG1, _mm_mul_ps(XMMqg, XMMqg));

        XMMp = _mm_sub_ps(XMMp, _mm_mul_ps(XMMeta_p, XMMpg));
        XMMq = _mm_sub_ps(XMMq, _mm_mul_ps(XMMeta_q, XMMqg));

        _mm_store_ps(p+d, XMMp);
        _mm_store_ps(q+d, XMMq);
    }

    mf_float tmp = 0;
    _mm_store_ss(&tmp, XMMlambda_p1);
    if(tmp > 0)
    {
        for(mf_int d = d_begin; d < d_end; d += 4)
        {
            __m128 XMMp = _mm_load_ps(p+d);
            __m128 XMMflip = _mm_and_ps(_mm_cmple_ps(XMMp, _mm_set1_ps(0.0f)),
                             _mm_set1_ps(-0.0f));
            XMMp = _mm_xor_ps(XMMflip,
                   _mm_max_ps(_mm_sub_ps(_mm_xor_ps(XMMp, XMMflip),
                   _mm_mul_ps(XMMeta_p, XMMlambda_p1)), _mm_set1_ps(0.0f)));
            _mm_store_ps(p+d, XMMp);
        }
    }

    _mm_store_ss(&tmp, XMMlambda_q1);
    if(tmp > 0)
    {
        for(mf_int d = d_begin; d < d_end; d += 4)
        {
            __m128 XMMq = _mm_load_ps(q+d);
            __m128 XMMflip = _mm_and_ps(_mm_cmple_ps(XMMq, _mm_set1_ps(0.0f)),
                             _mm_set1_ps(-0.0f));
            XMMq = _mm_xor_ps(XMMflip,
                   _mm_max_ps(_mm_sub_ps(_mm_xor_ps(XMMq, XMMflip),
                   _mm_mul_ps(XMMeta_q, XMMlambda_q1)), _mm_set1_ps(0.0f)));
            _mm_store_ps(q+d, XMMq);
        }
    }

    if(param.do_nmf)
    {
        for(mf_int d = d_begin; d < d_end; d += 4)
        {
            __m128 XMMp = _mm_load_ps(p+d);
            __m128 XMMq = _mm_load_ps(q+d);
            XMMp = _mm_max_ps(XMMp, _mm_set1_ps(0.0f));
            XMMq = _mm_max_ps(XMMq, _mm_set1_ps(0.0f));
            _mm_store_ps(p+d, XMMp);
            _mm_store_ps(q+d, XMMq);
        }
    }

    XMMpG1 = _mm_hadd_ps(XMMpG1, XMMpG1);
    XMMpG1 = _mm_hadd_ps(XMMpG1, XMMpG1);
    XMMqG1 = _mm_hadd_ps(XMMqG1, XMMqG1);
    XMMqG1 = _mm_hadd_ps(XMMqG1, XMMqG1);

    XMMpG = _mm_add_ps(XMMpG, _mm_mul_ps(XMMpG1, XMMrk));
    XMMqG = _mm_add_ps(XMMqG, _mm_mul_ps(XMMqG1, XMMrk));

    _mm_store_ss(pG, XMMpG);
    _mm_store_ss(qG, XMMqG);
}
#elif defined USEAVX
void MFSolver::sg_update(mf_int d_begin, mf_int d_end, __m256 XMMz,
                                __m256 XMMlambda_p1, __m256 XMMlambda_q1,
                                __m256 XMMlambda_p2, __m256 XMMlambda_q2,
                                __m256 XMMeta, __m256 XMMrk)
{
    __m256 XMMpG = _mm256_broadcast_ss(pG);
    __m256 XMMqG = _mm256_broadcast_ss(qG);
    __m256 XMMeta_p = _mm256_mul_ps(XMMeta, _mm256_rsqrt_ps(XMMpG));
    __m256 XMMeta_q = _mm256_mul_ps(XMMeta, _mm256_rsqrt_ps(XMMqG));
    __m256 XMMpG1 = _mm256_setzero_ps();
    __m256 XMMqG1 = _mm256_setzero_ps();

    for(mf_int d = d_begin; d < d_end; d += 8)
    {
        __m256 XMMp = _mm256_load_ps(p+d);
        __m256 XMMq = _mm256_load_ps(q+d);

        __m256 XMMpg = _mm256_sub_ps(_mm256_mul_ps(XMMlambda_p2, XMMp),
                                     _mm256_mul_ps(XMMz, XMMq));
        __m256 XMMqg = _mm256_sub_ps(_mm256_mul_ps(XMMlambda_q2, XMMq),
                                     _mm256_mul_ps(XMMz, XMMp));

        XMMpG1 = _mm256_add_ps(XMMpG1, _mm256_mul_ps(XMMpg, XMMpg));
        XMMqG1 = _mm256_add_ps(XMMqG1, _mm256_mul_ps(XMMqg, XMMqg));

        XMMp = _mm256_sub_ps(XMMp, _mm256_mul_ps(XMMeta_p, XMMpg));
        XMMq = _mm256_sub_ps(XMMq, _mm256_mul_ps(XMMeta_q, XMMqg));
        _mm256_store_ps(p+d, XMMp);
        _mm256_store_ps(q+d, XMMq);
    }

    mf_float tmp = 0;
    _mm_store_ss(&tmp, _mm256_castps256_ps128(XMMlambda_p1));
    if(tmp > 0)
    {
        for(mf_int d = d_begin; d < d_end; d += 8)
        {
            __m256 XMMp = _mm256_load_ps(p+d);
            __m256 XMMflip = _mm256_and_ps(_mm256_cmp_ps(XMMp,
                             _mm256_set1_ps(0.0f), _CMP_LE_OS),
                             _mm256_set1_ps(-0.0f));
            XMMp = _mm256_xor_ps(XMMflip,
                   _mm256_max_ps(_mm256_sub_ps(
                   _mm256_xor_ps(XMMp, XMMflip),
                   _mm256_mul_ps(XMMeta_p, XMMlambda_p1)),
                   _mm256_set1_ps(0.0f)));
            _mm256_store_ps(p+d, XMMp);
        }
    }

    _mm_store_ss(&tmp, _mm256_castps256_ps128(XMMlambda_q1));
    if(tmp > 0)
    {
        for(mf_int d = d_begin; d < d_end; d += 8)
        {
            __m256 XMMq = _mm256_load_ps(q+d);
            __m256 XMMflip = _mm256_and_ps(_mm256_cmp_ps(XMMq,
                             _mm256_set1_ps(0.0f), _CMP_LE_OS),
                             _mm256_set1_ps(-0.0f));
            XMMq = _mm256_xor_ps(XMMflip,
                   _mm256_max_ps(_mm256_sub_ps(
                   _mm256_xor_ps(XMMq, XMMflip),
                   _mm256_mul_ps(XMMeta_q, XMMlambda_q1)),
                   _mm256_set1_ps(0.0f)));
            _mm256_store_ps(q+d, XMMq);
        }
    }

    if(param.do_nmf)
    {
        for(mf_int d = d_begin; d < d_end; d += 8)
        {
            __m256 XMMp = _mm256_load_ps(p+d);
            __m256 XMMq = _mm256_load_ps(q+d);
            XMMp = _mm256_max_ps(XMMp, _mm256_set1_ps(0));
            XMMq = _mm256_max_ps(XMMq, _mm256_set1_ps(0));
            _mm256_store_ps(p+d, XMMp);
            _mm256_store_ps(q+d, XMMq);
        }
    }

    XMMpG1 = _mm256_add_ps(XMMpG1,
             _mm256_permute2f128_ps(XMMpG1, XMMpG1, 0x1));
    XMMpG1 = _mm256_hadd_ps(XMMpG1, XMMpG1);
    XMMpG1 = _mm256_hadd_ps(XMMpG1, XMMpG1);

    XMMqG1 = _mm256_add_ps(XMMqG1,
             _mm256_permute2f128_ps(XMMqG1, XMMqG1, 0x1));
    XMMqG1 = _mm256_hadd_ps(XMMqG1, XMMqG1);
    XMMqG1 = _mm256_hadd_ps(XMMqG1, XMMqG1);

    XMMpG = _mm256_add_ps(XMMpG, _mm256_mul_ps(XMMpG1, XMMrk));
    XMMqG = _mm256_add_ps(XMMqG, _mm256_mul_ps(XMMqG1, XMMrk));

    _mm_store_ss(pG, _mm256_castps256_ps128(XMMpG));
    _mm_store_ss(qG, _mm256_castps256_ps128(XMMqG));
}
#else
void MFSolver::sg_update(mf_int d_begin, mf_int d_end, mf_float rk)
{
    mf_float eta_p = param.eta*qrsqrt(*pG);
    mf_float eta_q = param.eta*qrsqrt(*qG);

    mf_float pG1 = 0;
    mf_float qG1 = 0;

    for(mf_int d = d_begin; d < d_end; d++)
    {
        mf_float gp = -z*q[d]+lambda_p2*p[d];
        mf_float gq = -z*p[d]+lambda_q2*q[d];

        pG1 += gp*gp;
        qG1 += gq*gq;

        p[d] -= eta_p*gp;
        q[d] -= eta_q*gq;
    }

    if(lambda_p1 > 0)
    {
        for(mf_int d = d_begin; d < d_end; d++)
        {
            mf_float p1 = max(abs(p[d])-lambda_p1*eta_p, 0.0f);
            p[d] = p[d] >= 0? p1: -p1;
        }
    }

    if(lambda_q1 > 0)
    {
        for(mf_int d = d_begin; d < d_end; d++)
        {
            mf_float q1 = max(abs(q[d])-lambda_q1*eta_q, 0.0f);
            q[d] = q[d] >= 0? q1: -q1;
        }
    }

    if(param.do_nmf)
    {
        for(mf_int d = d_begin; d < d_end; d++)
        {
            p[d] = max(p[d], (mf_float)0.0f);
            q[d] = max(q[d], (mf_float)0.0f);
        }
    }

    *pG += pG1*rk;
    *qG += qG1*rk;
}
#endif

class L2_MFR : public MFSolver
{
public:
    L2_MFR(Scheduler &scheduler, vector<BlockBase*> &blocks, mf_float *PG, mf_float *QG,
           mf_model &model, mf_parameter param, bool &slow_only)
        : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
#if defined USESSE
    void prepare_for_sg_update(
        __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#elif defined USEAVX
    void prepare_for_sg_update(
        __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#else
    void prepare_for_sg_update();
#endif
};

#if defined USESSE
void L2_MFR::prepare_for_sg_update(
    __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    XMMz = _mm_sub_ps(_mm_set1_ps(N->r), XMMz);
    XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(
              _mm_mul_ps(XMMz, XMMz)));
    XMMerror = XMMloss;
}
#elif defined USEAVX
void L2_MFR::prepare_for_sg_update(
    __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    XMMz = _mm256_sub_ps(_mm256_set1_ps(N->r), XMMz);
    XMMloss = _mm_add_pd(XMMloss,
              _mm_cvtps_pd(_mm256_castps256_ps128(
              _mm256_mul_ps(XMMz, XMMz))));
    XMMerror = XMMloss;
}
#else
void L2_MFR::prepare_for_sg_update()
{
    calc_z(z, model.k, p, q);
    z = N->r-z;
    loss += z*z;
    error = loss;
}
#endif
class L1_MFR : public MFSolver
{
public:
    L1_MFR(Scheduler &scheduler, vector<BlockBase*> &blocks, mf_float *PG, mf_float *QG,
           mf_model &model, mf_parameter param, bool &slow_only)
        : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
#if defined USESSE
    void prepare_for_sg_update(
        __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#elif defined USEAVX
    void prepare_for_sg_update(
        __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#else
    void prepare_for_sg_update();
#endif
};

#if defined USESSE
void L1_MFR::prepare_for_sg_update(
    __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    XMMz = _mm_sub_ps(_mm_set1_ps(N->r), XMMz);
    XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(
              _mm_andnot_ps(_mm_set1_ps(-0.0f), XMMz)));
    XMMerror = XMMloss;
    XMMz = _mm_add_ps(_mm_and_ps(_mm_cmpgt_ps(XMMz, _mm_set1_ps(0.0f)),
           _mm_set1_ps(1.0f)),
           _mm_and_ps(_mm_cmplt_ps(XMMz, _mm_set1_ps(0.0f)),
           _mm_set1_ps(-1.0f)));
}
#elif defined USEAVX
void L1_MFR::prepare_for_sg_update(
    __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    XMMz = _mm256_sub_ps(_mm256_set1_ps(N->r), XMMz);
    XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(_mm256_castps256_ps128(
              _mm256_andnot_ps(_mm256_set1_ps(-0.0f), XMMz))));
    XMMerror = XMMloss;
    XMMz = _mm256_add_ps(_mm256_and_ps(_mm256_cmp_ps(XMMz,
           _mm256_set1_ps(0.0f), _CMP_GT_OS), _mm256_set1_ps(1.0f)),
           _mm256_and_ps(_mm256_cmp_ps(XMMz,
           _mm256_set1_ps(0.0f), _CMP_LT_OS), _mm256_set1_ps(-1.0f)));
}
#else
void L1_MFR::prepare_for_sg_update()
{
    calc_z(z, model.k, p, q);
    z = N->r-z;
    loss += abs(z);
    error = loss;
    if(z > 0)
        z = 1;
    else if(z < 0)
        z = -1;
}
#endif

class KL_MFR : public MFSolver
{
public:
    KL_MFR(Scheduler &scheduler, vector<BlockBase*> &blocks, mf_float *PG, mf_float *QG,
           mf_model &model, mf_parameter param, bool &slow_only)
        : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
#if defined USESSE
    void prepare_for_sg_update(
        __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#elif defined USEAVX
    void prepare_for_sg_update(
        __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#else
    void prepare_for_sg_update();
#endif
};

#if defined USESSE
void KL_MFR::prepare_for_sg_update(
    __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    XMMz = _mm_div_ps(_mm_set1_ps(N->r), XMMz);
    _mm_store_ss(&z, XMMz);
    XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(
              _mm_set1_ps(N->r*(log(z)-1+1/z))));
    XMMerror = XMMloss;
    XMMz = _mm_sub_ps(XMMz, _mm_set1_ps(1.0f));
}
#elif defined USEAVX
void KL_MFR::prepare_for_sg_update(
    __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    XMMz = _mm256_div_ps(_mm256_set1_ps(N->r), XMMz);
    _mm_store_ss(&z, _mm256_castps256_ps128(XMMz));
    XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(
              _mm_set1_ps(N->r*(log(z)-1+1/z))));
    XMMerror = XMMloss;
    XMMz = _mm256_sub_ps(XMMz, _mm256_set1_ps(1.0f));
}
#else
void KL_MFR::prepare_for_sg_update()
{
    calc_z(z, model.k, p, q);
    z = N->r/z;
    loss += N->r*(log(z)-1+1/z);
    error = loss;
    z -= 1;
}
#endif

class LR_MFC : public MFSolver
{
public:
    LR_MFC(Scheduler &scheduler, vector<BlockBase*> &blocks,
           mf_float *PG, mf_float *QG, mf_model &model,
           mf_parameter param, bool &slow_only)
        : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
#if defined USESSE
    void prepare_for_sg_update(
        __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#elif defined USEAVX
    void prepare_for_sg_update(
        __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#else
    void prepare_for_sg_update();
#endif
};

#if defined USESSE
void LR_MFC::prepare_for_sg_update(
    __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    _mm_store_ss(&z, XMMz);
    if(N->r > 0)
    {
        z = exp(-z);
        XMMloss = _mm_add_pd(XMMloss, _mm_set1_pd(log(1+z)));
        XMMz = _mm_set1_ps(z/(1+z));
    }
    else
    {
        z = exp(z);
        XMMloss = _mm_add_pd(XMMloss, _mm_set1_pd(log(1+z)));
        XMMz = _mm_set1_ps(-z/(1+z));
    }
    XMMerror = XMMloss;
}
#elif defined USEAVX
void LR_MFC::prepare_for_sg_update(
    __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    _mm_store_ss(&z, _mm256_castps256_ps128(XMMz));
    if(N->r > 0)
    {
        z = exp(-z);
        XMMloss = _mm_add_pd(XMMloss, _mm_set1_pd(log(1.0+z)));
        XMMz = _mm256_set1_ps(z/(1+z));
    }
    else
    {
        z = exp(z);
        XMMloss = _mm_add_pd(XMMloss, _mm_set1_pd(log(1.0+z)));
        XMMz = _mm256_set1_ps(-z/(1+z));
    }
    XMMerror = XMMloss;
}
#else
void LR_MFC::prepare_for_sg_update()
{
    calc_z(z, model.k, p, q);
    if(N->r > 0)
    {
        z = exp(-z);
        loss += log(1+z);
        error = loss;
        z = z/(1+z);
    }
    else
    {
        z = exp(z);
        loss += log(1+z);
        error = loss;
        z = -z/(1+z);
    }
}
#endif

class L2_MFC : public MFSolver
{
public:
    L2_MFC(Scheduler &scheduler, vector<BlockBase*> &blocks,
           mf_float *PG, mf_float *QG, mf_model &model,
           mf_parameter param, bool &slow_only)
        : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
#if defined USESSE
    void prepare_for_sg_update(
        __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#elif defined USEAVX
    void prepare_for_sg_update(
        __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#else
    void prepare_for_sg_update();
#endif
};

#if defined USESSE
void L2_MFC::prepare_for_sg_update(
    __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    if(N->r > 0)
    {
        __m128 mask = _mm_cmpgt_ps(XMMz, _mm_set1_ps(0.0f));
        XMMerror = _mm_add_pd(XMMerror, _mm_cvtps_pd(
                   _mm_and_ps(_mm_set1_ps(1.0f), mask)));
        XMMz = _mm_max_ps(_mm_set1_ps(0.0f), _mm_sub_ps(
               _mm_set1_ps(1.0f), XMMz));
    }
    else
    {
        __m128 mask = _mm_cmplt_ps(XMMz, _mm_set1_ps(0.0f));
        XMMerror = _mm_add_pd(XMMerror, _mm_cvtps_pd(
                   _mm_and_ps(_mm_set1_ps(1.0f), mask)));
        XMMz = _mm_min_ps(_mm_set1_ps(0.0f), _mm_sub_ps(
               _mm_set1_ps(-1.0f), XMMz));
    }
    XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(
              _mm_mul_ps(XMMz, XMMz)));
}
#elif defined USEAVX
void L2_MFC::prepare_for_sg_update(
    __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    if(N->r > 0)
    {
        __m128 mask = _mm_cmpgt_ps(_mm256_castps256_ps128(XMMz),
                      _mm_set1_ps(0.0f));
        XMMerror = _mm_add_pd(XMMerror, _mm_cvtps_pd(
                   _mm_and_ps(_mm_set1_ps(1.0f), mask)));
        XMMz = _mm256_max_ps(_mm256_set1_ps(0.0f),
               _mm256_sub_ps(_mm256_set1_ps(1.0f), XMMz));
    }
    else
    {
        __m128 mask = _mm_cmplt_ps(_mm256_castps256_ps128(XMMz),
                      _mm_set1_ps(0.0f));
        XMMerror = _mm_add_pd(XMMerror, _mm_cvtps_pd(
                   _mm_and_ps(_mm_set1_ps(1.0f), mask)));
        XMMz = _mm256_min_ps(_mm256_set1_ps(0.0f),
               _mm256_sub_ps(_mm256_set1_ps(-1.0f), XMMz));
    }
    XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(
              _mm_mul_ps(_mm256_castps256_ps128(XMMz),
              _mm256_castps256_ps128(XMMz))));
}
#else
void L2_MFC::prepare_for_sg_update()
{
    calc_z(z, model.k, p, q);
    if(N->r > 0)
    {
        error += z > 0? 1: 0;
        z = max(0.0f, 1-z);
    }
    else
    {
        error += z < 0? 1: 0;
        z = min(0.0f, -1-z);
    }
    loss += z*z;
}
#endif

class L1_MFC : public MFSolver
{
public:
    L1_MFC(Scheduler &scheduler, vector<BlockBase*> &blocks, mf_float *PG, mf_float *QG,
           mf_model &model, mf_parameter param, bool &slow_only)
        : MFSolver(scheduler, blocks, PG, QG, model, param, slow_only) {}

protected:
#if defined USESSE
    void prepare_for_sg_update(
        __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#elif defined USEAVX
    void prepare_for_sg_update(
        __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
#else
    void prepare_for_sg_update();
#endif
};

#if defined USESSE
void L1_MFC::prepare_for_sg_update(
    __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    if(N->r > 0)
    {
        XMMerror = _mm_add_pd(XMMerror, _mm_cvtps_pd(
                   _mm_and_ps(_mm_cmpge_ps(XMMz, _mm_set1_ps(0.0f)),
                   _mm_set1_ps(1.0f))));
        XMMz = _mm_sub_ps(_mm_set1_ps(1.0f), XMMz);
        XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(
                  _mm_max_ps(_mm_set1_ps(0.0f), XMMz)));
        XMMz = _mm_and_ps(_mm_cmpge_ps(XMMz, _mm_set1_ps(0.0f)),
               _mm_set1_ps(1.0f));
    }
    else
    {
        XMMerror = _mm_add_pd(XMMerror, _mm_cvtps_pd(
                   _mm_and_ps(_mm_cmplt_ps(XMMz, _mm_set1_ps(0.0f)),
                   _mm_set1_ps(1.0f))));
        XMMz = _mm_add_ps(_mm_set1_ps(1.0f), XMMz);
        XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(
                  _mm_max_ps(_mm_set1_ps(0.0f), XMMz)));
        XMMz = _mm_and_ps(_mm_cmpge_ps(XMMz, _mm_set1_ps(0.0f)),
               _mm_set1_ps(-1.0f));
    }
}
#elif defined USEAVX
void L1_MFC::prepare_for_sg_update(
    __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    calc_z(XMMz, model.k, p, q);
    if(N->r > 0)
    {
        XMMerror = _mm_add_pd(XMMerror, _mm_cvtps_pd(_mm_and_ps(
                   _mm_cmpge_ps(_mm256_castps256_ps128(XMMz),
                   _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f))));
        XMMz = _mm256_sub_ps(_mm256_set1_ps(1.0f), XMMz);
        XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(_mm_max_ps(
                  _mm_set1_ps(0.0f), _mm256_castps256_ps128(XMMz))));
        XMMz = _mm256_and_ps(_mm256_cmp_ps(XMMz, _mm256_set1_ps(0.0f),
               _CMP_GE_OS), _mm256_set1_ps(1.0f));
    }
    else
    {
        XMMerror = _mm_add_pd(XMMerror, _mm_cvtps_pd(_mm_and_ps(
                   _mm_cmplt_ps(_mm256_castps256_ps128(XMMz),
                   _mm_set1_ps(0.0f)), _mm_set1_ps(1.0f))));
        XMMz = _mm256_add_ps(_mm256_set1_ps(1.0f), XMMz);
        XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(_mm_max_ps(
                  _mm_set1_ps(0.0f), _mm256_castps256_ps128(XMMz))));
        XMMz = _mm256_and_ps(_mm256_cmp_ps(XMMz, _mm256_set1_ps(0.0f),
               _CMP_GE_OS), _mm256_set1_ps(-1.0f));
    }
}
#else
void L1_MFC::prepare_for_sg_update()
{
    calc_z(z, model.k, p, q);
    if(N->r > 0)
    {
        loss += max(0.0f, 1-z);
        error += z > 0? 1.0f: 0.0f;
        z = z > 1? 0.0f: 1.0f;
    }
    else
    {
        loss += max(0.0f, 1+z);
        error += z < 0? 1.0f: 0.0f;
        z = z < -1? 0.0f: -1.0f;
    }
}
#endif
//--------------------------------------
//------------One-class MF--------------
//--------------------------------------

class BPRSolver : public SolverBase
{
public:
    BPRSolver(Scheduler &scheduler, vector<BlockBase*> &blocks,
              mf_float *PG, mf_float *QG, mf_model &model, mf_parameter param,
              bool &slow_only, bool is_column_oriented)
        : SolverBase(scheduler, blocks, PG, QG, model, param, slow_only),
                     is_column_oriented(is_column_oriented) {}

protected:
#if defined USESSE
    static void calc_z(__m128 &XMMz, mf_int k,
                       mf_float *p, mf_float *q, mf_float *w);
    void arrange_block(__m128d &XMMloss, __m128d &XMMerror);
    void prepare_for_sg_update(
        __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
    void sg_update(mf_int d_begin, mf_int d_end, __m128 XMMz,
                   __m128 XMMlambda_p1, __m128 XMMlambda_q1,
                   __m128 XMMlambda_p2, __m128 XMMlamdba_q2,
                   __m128 XMMeta, __m128 XMMrk);
    void finalize(__m128d XMMloss, __m128d XMMerror);
#elif defined USEAVX
    static void calc_z(__m256 &XMMz, mf_int k,
                       mf_float *p, mf_float *q, mf_float *w);
    void arrange_block(__m128d &XMMloss, __m128d &XMMerror);
    void prepare_for_sg_update(
        __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror);
    void sg_update(mf_int d_begin, mf_int d_end, __m256 XMMz,
                   __m256 XMMlambda_p1, __m256 XMMlambda_q1,
                   __m256 XMMlambda_p2, __m256 XMMlamdba_q2,
                   __m256 XMMeta, __m256 XMMrk);
    void finalize(__m128d XMMloss, __m128d XMMerror);
#else
    static void calc_z(mf_float &z, mf_int k,
                       mf_float *p, mf_float *q, mf_float *w);
    void arrange_block();
    void prepare_for_sg_update();
    void sg_update(mf_int d_begin, mf_int d_end, mf_float rk);
    void finalize();
#endif
    void update() { pG++; qG++; wG++; };
    virtual void prepare_negative() = 0;

    bool is_column_oriented;
    mf_int bpr_bid;
    mf_float *w;
    mf_float *wG;
};


#if defined USESSE
inline void BPRSolver::calc_z(
    __m128 &XMMz, mf_int k, mf_float *p, mf_float *q, mf_float *w)
{
    XMMz = _mm_setzero_ps();
    for(mf_int d = 0; d < k; d += 4)
        XMMz = _mm_add_ps(XMMz, _mm_mul_ps(_mm_load_ps(p+d),
               _mm_sub_ps(_mm_load_ps(q+d), _mm_load_ps(w+d))));
    XMMz = _mm_hadd_ps(XMMz, XMMz);
    XMMz = _mm_hadd_ps(XMMz, XMMz);
}

void BPRSolver::arrange_block(__m128d &XMMloss, __m128d &XMMerror)
{
    XMMloss = _mm_setzero_pd();
    XMMerror = _mm_setzero_pd();
    bid = scheduler.get_job();
    block = blocks[bid];
    block->reload();
    bpr_bid = scheduler.get_bpr_job(bid, is_column_oriented);
}

void BPRSolver::finalize(__m128d XMMloss, __m128d XMMerror)
{
    _mm_store_sd(&loss, XMMloss);
    _mm_store_sd(&error, XMMerror);
    scheduler.put_job(bid, loss, error);
    scheduler.put_bpr_job(bid, bpr_bid);
}

void BPRSolver::sg_update(mf_int d_begin, mf_int d_end, __m128 XMMz,
                                 __m128 XMMlambda_p1, __m128 XMMlambda_q1,
                                 __m128 XMMlambda_p2, __m128 XMMlambda_q2,
                                 __m128 XMMeta, __m128 XMMrk)
{
    __m128 XMMpG = _mm_load1_ps(pG);
    __m128 XMMqG = _mm_load1_ps(qG);
    __m128 XMMwG = _mm_load1_ps(wG);
    __m128 XMMeta_p = _mm_mul_ps(XMMeta, _mm_rsqrt_ps(XMMpG));
    __m128 XMMeta_q = _mm_mul_ps(XMMeta, _mm_rsqrt_ps(XMMqG));
    __m128 XMMeta_w = _mm_mul_ps(XMMeta, _mm_rsqrt_ps(XMMwG));

    __m128 XMMpG1 = _mm_setzero_ps();
    __m128 XMMqG1 = _mm_setzero_ps();
    __m128 XMMwG1 = _mm_setzero_ps();

    for(mf_int d = d_begin; d < d_end; d += 4)
    {
        __m128 XMMp = _mm_load_ps(p+d);
        __m128 XMMq = _mm_load_ps(q+d);
        __m128 XMMw = _mm_load_ps(w+d);

        __m128 XMMpg = _mm_add_ps(_mm_mul_ps(XMMlambda_p2, XMMp),
                       _mm_mul_ps(XMMz, _mm_sub_ps(XMMw, XMMq)));
        __m128 XMMqg = _mm_sub_ps(_mm_mul_ps(XMMlambda_q2, XMMq),
                       _mm_mul_ps(XMMz, XMMp));
        __m128 XMMwg = _mm_add_ps(_mm_mul_ps(XMMlambda_q2, XMMw),
                       _mm_mul_ps(XMMz, XMMp));

        XMMpG1 = _mm_add_ps(XMMpG1, _mm_mul_ps(XMMpg, XMMpg));
        XMMqG1 = _mm_add_ps(XMMqG1, _mm_mul_ps(XMMqg, XMMqg));
        XMMwG1 = _mm_add_ps(XMMwG1, _mm_mul_ps(XMMwg, XMMwg));

        XMMp = _mm_sub_ps(XMMp, _mm_mul_ps(XMMeta_p, XMMpg));
        XMMq = _mm_sub_ps(XMMq, _mm_mul_ps(XMMeta_q, XMMqg));
        XMMw = _mm_sub_ps(XMMw, _mm_mul_ps(XMMeta_w, XMMwg));

        _mm_store_ps(p+d, XMMp);
        _mm_store_ps(q+d, XMMq);
        _mm_store_ps(w+d, XMMw);
    }

    mf_float tmp = 0;
    _mm_store_ss(&tmp, XMMlambda_p1);
    if(tmp > 0)
    {
        for(mf_int d = d_begin; d < d_end; d += 4)
        {
            __m128 XMMp = _mm_load_ps(p+d);
            __m128 XMMflip = _mm_and_ps(_mm_cmple_ps(XMMp, _mm_set1_ps(0.0f)),
                             _mm_set1_ps(-0.0f));
            XMMp = _mm_xor_ps(XMMflip,
                   _mm_max_ps(_mm_sub_ps(_mm_xor_ps(XMMp, XMMflip),
                   _mm_mul_ps(XMMeta_p, XMMlambda_p1)), _mm_set1_ps(0.0f)));
            _mm_store_ps(p+d, XMMp);
        }
    }

    _mm_store_ss(&tmp, XMMlambda_q1);
    if (tmp > 0)
    {
        for(mf_int d = d_begin; d < d_end; d += 4)
        {
            __m128 XMMq = _mm_load_ps(q+d);
            __m128 XMMw = _mm_load_ps(w+d);
            __m128 XMMflip = _mm_and_ps(_mm_cmple_ps(XMMq, _mm_set1_ps(0.0f)),
                             _mm_set1_ps(-0.0f));
            XMMq = _mm_xor_ps(XMMflip,
                   _mm_max_ps(_mm_sub_ps(_mm_xor_ps(XMMq, XMMflip),
                   _mm_mul_ps(XMMeta_q, XMMlambda_q1)), _mm_set1_ps(0.0f)));
            _mm_store_ps(q+d, XMMq);


            XMMflip = _mm_and_ps(_mm_cmple_ps(XMMw, _mm_set1_ps(0.0f)),
                    _mm_set1_ps(-0.0f));
            XMMw = _mm_xor_ps(XMMflip,
                   _mm_max_ps(_mm_sub_ps(_mm_xor_ps(XMMw, XMMflip),
                   _mm_mul_ps(XMMeta_w, XMMlambda_q1)), _mm_set1_ps(0.0f)));
            _mm_store_ps(w+d, XMMw);
        }
    }

    if(param.do_nmf)
    {
        for(mf_int d = d_begin; d < d_end; d += 4)
        {
            __m128 XMMp = _mm_load_ps(p+d);
            __m128 XMMq = _mm_load_ps(q+d);
            __m128 XMMw = _mm_load_ps(w+d);
            XMMp = _mm_max_ps(XMMp, _mm_set1_ps(0.0f));
            XMMq = _mm_max_ps(XMMq, _mm_set1_ps(0.0f));
            XMMw = _mm_max_ps(XMMw, _mm_set1_ps(0.0f));
            _mm_store_ps(p+d, XMMp);
            _mm_store_ps(q+d, XMMq);
            _mm_store_ps(w+d, XMMw);
        }
    }

    XMMpG1 = _mm_hadd_ps(XMMpG1, XMMpG1);
    XMMpG1 = _mm_hadd_ps(XMMpG1, XMMpG1);
    XMMqG1 = _mm_hadd_ps(XMMqG1, XMMqG1);
    XMMqG1 = _mm_hadd_ps(XMMqG1, XMMqG1);
    XMMwG1 = _mm_hadd_ps(XMMwG1, XMMwG1);
    XMMwG1 = _mm_hadd_ps(XMMwG1, XMMwG1);

    XMMpG = _mm_add_ps(XMMpG, _mm_mul_ps(XMMpG1, XMMrk));
    XMMqG = _mm_add_ps(XMMqG, _mm_mul_ps(XMMqG1, XMMrk));
    XMMwG = _mm_add_ps(XMMwG, _mm_mul_ps(XMMwG1, XMMrk));

    _mm_store_ss(pG, XMMpG);
    _mm_store_ss(qG, XMMqG);
    _mm_store_ss(wG, XMMwG);
}

void BPRSolver::prepare_for_sg_update(
    __m128 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    prepare_negative();
    calc_z(XMMz, model.k, p, q, w);
    _mm_store_ss(&z, XMMz);
    z = exp(-z);
    XMMloss = _mm_add_pd(XMMloss, _mm_set1_pd(log(1+z)));
    XMMerror = XMMloss;
    XMMz = _mm_set1_ps(z/(1+z));
}
#elif defined USEAVX
inline void BPRSolver::calc_z(
    __m256 &XMMz, mf_int k, mf_float *p, mf_float *q, mf_float *w)
{
    XMMz = _mm256_setzero_ps();
    for(mf_int d = 0; d < k; d += 8)
        XMMz = _mm256_add_ps(XMMz, _mm256_mul_ps(
               _mm256_load_ps(p+d), _mm256_sub_ps(
               _mm256_load_ps(q+d), _mm256_load_ps(w+d))));
    XMMz = _mm256_add_ps(XMMz, _mm256_permute2f128_ps(XMMz, XMMz, 0x1));
    XMMz = _mm256_hadd_ps(XMMz, XMMz);
    XMMz = _mm256_hadd_ps(XMMz, XMMz);
}

void BPRSolver::arrange_block(__m128d &XMMloss, __m128d &XMMerror)
{
    XMMloss = _mm_setzero_pd();
    XMMerror = _mm_setzero_pd();
    bid = scheduler.get_job();
    block = blocks[bid];
    block->reload();
    bpr_bid = scheduler.get_bpr_job(bid, is_column_oriented);
}

void BPRSolver::finalize(__m128d XMMloss, __m128d XMMerror)
{
    _mm_store_sd(&loss, XMMloss);
    _mm_store_sd(&error, XMMerror);
    scheduler.put_job(bid, loss, error);
    scheduler.put_bpr_job(bid, bpr_bid);
}

void BPRSolver::sg_update(mf_int d_begin, mf_int d_end, __m256 XMMz,
                                 __m256 XMMlambda_p1, __m256 XMMlambda_q1,
                                 __m256 XMMlambda_p2, __m256 XMMlambda_q2,
                                 __m256 XMMeta, __m256 XMMrk)
{
    __m256 XMMpG = _mm256_broadcast_ss(pG);
    __m256 XMMqG = _mm256_broadcast_ss(qG);
    __m256 XMMwG = _mm256_broadcast_ss(wG);
    __m256 XMMeta_p =
        _mm256_mul_ps(XMMeta, _mm256_rsqrt_ps(XMMpG));
    __m256 XMMeta_q =
        _mm256_mul_ps(XMMeta, _mm256_rsqrt_ps(XMMqG));
    __m256 XMMeta_w =
        _mm256_mul_ps(XMMeta, _mm256_rsqrt_ps(XMMwG));

    __m256 XMMpG1 = _mm256_setzero_ps();
    __m256 XMMqG1 = _mm256_setzero_ps();
    __m256 XMMwG1 = _mm256_setzero_ps();

    for(mf_int d = d_begin; d < d_end; d += 8)
    {
        __m256 XMMp = _mm256_load_ps(p+d);
        __m256 XMMq = _mm256_load_ps(q+d);
        __m256 XMMw = _mm256_load_ps(w+d);
        __m256 XMMpg = _mm256_add_ps(_mm256_mul_ps(XMMlambda_p2, XMMp),
                       _mm256_mul_ps(XMMz, _mm256_sub_ps(XMMw, XMMq)));
        __m256 XMMqg = _mm256_sub_ps(_mm256_mul_ps(XMMlambda_q2, XMMq),
                       _mm256_mul_ps(XMMz, XMMp));
        __m256 XMMwg = _mm256_add_ps(_mm256_mul_ps(XMMlambda_q2, XMMw),
                       _mm256_mul_ps(XMMz, XMMp));

        XMMpG1 = _mm256_add_ps(XMMpG1, _mm256_mul_ps(XMMpg, XMMpg));
        XMMqG1 = _mm256_add_ps(XMMqG1, _mm256_mul_ps(XMMqg, XMMqg));
        XMMwG1 = _mm256_add_ps(XMMwG1, _mm256_mul_ps(XMMwg, XMMwg));

        XMMp = _mm256_sub_ps(XMMp, _mm256_mul_ps(XMMeta_p, XMMpg));
        XMMq = _mm256_sub_ps(XMMq, _mm256_mul_ps(XMMeta_q, XMMqg));
        XMMw = _mm256_sub_ps(XMMw, _mm256_mul_ps(XMMeta_w, XMMwg));

        _mm256_store_ps(p+d, XMMp);
        _mm256_store_ps(q+d, XMMq);
        _mm256_store_ps(w+d, XMMw);
    }

    mf_float tmp = 0;
    _mm_store_ss(&tmp, _mm256_castps256_ps128(XMMlambda_p1));
    if(tmp > 0)
    {
        for(mf_int d = d_begin; d < d_end; d += 8)
        {
            __m256 XMMp = _mm256_load_ps(p+d);
            __m256 XMMflip =
                _mm256_and_ps(
                _mm256_cmp_ps(XMMp, _mm256_set1_ps(0.0f), _CMP_LE_OS),
                _mm256_set1_ps(-0.0f));
            XMMp = _mm256_xor_ps(XMMflip,
                   _mm256_max_ps(_mm256_sub_ps(_mm256_xor_ps(XMMp, XMMflip),
                   _mm256_mul_ps(XMMeta_p, XMMlambda_p1)),
                   _mm256_set1_ps(0.0f)));
            _mm256_store_ps(p+d, XMMp);
        }
    }

    _mm_store_ss(&tmp, _mm256_castps256_ps128(XMMlambda_q1));
    if (tmp > 0)
    {
        for(mf_int d = d_begin; d < d_end; d += 8)
        {
            __m256 XMMq = _mm256_load_ps(q+d);
            __m256 XMMw = _mm256_load_ps(w+d);
            __m256 XMMflip;

            XMMflip = _mm256_and_ps(
                      _mm256_cmp_ps(XMMq, _mm256_set1_ps(0.0f), _CMP_LE_OS),
                      _mm256_set1_ps(-0.0f));
            XMMq = _mm256_xor_ps(XMMflip,
                   _mm256_max_ps(_mm256_sub_ps(_mm256_xor_ps(XMMq, XMMflip),
                   _mm256_mul_ps(XMMeta_q, XMMlambda_q1)),
                   _mm256_set1_ps(0.0f)));
            _mm256_store_ps(q+d, XMMq);


            XMMflip = _mm256_and_ps(
                      _mm256_cmp_ps(XMMw, _mm256_set1_ps(0.0f), _CMP_LE_OS),
                      _mm256_set1_ps(-0.0f));
            XMMw = _mm256_xor_ps(XMMflip,
                   _mm256_max_ps(_mm256_sub_ps(_mm256_xor_ps(XMMw, XMMflip),
                   _mm256_mul_ps(XMMeta_w, XMMlambda_q1)),
                   _mm256_set1_ps(0.0f)));
            _mm256_store_ps(w+d, XMMw);
        }
    }

    if(param.do_nmf)
    {
        for(mf_int d = d_begin; d < d_end; d += 8)
        {
            __m256 XMMp = _mm256_load_ps(p+d);
            __m256 XMMq = _mm256_load_ps(q+d);
            __m256 XMMw = _mm256_load_ps(w+d);
            XMMp = _mm256_max_ps(XMMp, _mm256_set1_ps(0.0f));
            XMMq = _mm256_max_ps(XMMq, _mm256_set1_ps(0.0f));
            XMMw = _mm256_max_ps(XMMw, _mm256_set1_ps(0.0f));
            _mm256_store_ps(p+d, XMMp);
            _mm256_store_ps(q+d, XMMq);
            _mm256_store_ps(w+d, XMMw);
        }
    }

    XMMpG1 = _mm256_add_ps(XMMpG1,
             _mm256_permute2f128_ps(XMMpG1, XMMpG1, 0x1));
    XMMpG1 = _mm256_hadd_ps(XMMpG1, XMMpG1);
    XMMpG1 = _mm256_hadd_ps(XMMpG1, XMMpG1);

    XMMqG1 = _mm256_add_ps(XMMqG1,
             _mm256_permute2f128_ps(XMMqG1, XMMqG1, 0x1));
    XMMqG1 = _mm256_hadd_ps(XMMqG1, XMMqG1);
    XMMqG1 = _mm256_hadd_ps(XMMqG1, XMMqG1);

    XMMwG1 = _mm256_add_ps(XMMwG1,
             _mm256_permute2f128_ps(XMMwG1, XMMwG1, 0x1));
    XMMwG1 = _mm256_hadd_ps(XMMwG1, XMMwG1);
    XMMwG1 = _mm256_hadd_ps(XMMwG1, XMMwG1);

    XMMpG = _mm256_add_ps(XMMpG, _mm256_mul_ps(XMMpG1, XMMrk));
    XMMqG = _mm256_add_ps(XMMqG, _mm256_mul_ps(XMMqG1, XMMrk));
    XMMwG = _mm256_add_ps(XMMwG, _mm256_mul_ps(XMMwG1, XMMrk));

    _mm_store_ss(pG, _mm256_castps256_ps128(XMMpG));
    _mm_store_ss(qG, _mm256_castps256_ps128(XMMqG));
    _mm_store_ss(wG, _mm256_castps256_ps128(XMMwG));
}

void BPRSolver::prepare_for_sg_update(
    __m256 &XMMz, __m128d &XMMloss, __m128d &XMMerror)
{
    prepare_negative();
    calc_z(XMMz, model.k, p, q, w);
    _mm_store_ss(&z, _mm256_castps256_ps128(XMMz));
    z = exp(-z);
    XMMloss = _mm_add_pd(XMMloss, _mm_set1_pd(log(1+z)));
    XMMerror = XMMloss;
    XMMz = _mm256_set1_ps(z/(1+z));
}
#else
inline void BPRSolver::calc_z(
    mf_float &z, mf_int k, mf_float *p, mf_float *q, mf_float *w)
{
    z = 0;
    for(mf_int d = 0; d < k; d++)
        z += p[d]*(q[d]-w[d]);
}

void BPRSolver::arrange_block()
{
    loss = 0.0;
    error = 0.0;
    bid = scheduler.get_job();
    block = blocks[bid];
    block->reload();
    bpr_bid = scheduler.get_bpr_job(bid, is_column_oriented);
}

void BPRSolver::finalize()
{
    scheduler.put_job(bid, loss, error);
    scheduler.put_bpr_job(bid, bpr_bid);
}

void BPRSolver::sg_update(mf_int d_begin, mf_int d_end, mf_float rk)
{
    mf_float eta_p = param.eta*qrsqrt(*pG);
    mf_float eta_q = param.eta*qrsqrt(*qG);
    mf_float eta_w = param.eta*qrsqrt(*wG);

    mf_float pG1 = 0;
    mf_float qG1 = 0;
    mf_float wG1 = 0;

    for(mf_int d = d_begin; d < d_end; d++)
    {
        mf_float gp = z*(w[d]-q[d]) + lambda_p2*p[d];
        mf_float gq = -z*p[d] + lambda_q2*q[d];
        mf_float gw = z*p[d] + lambda_q2*w[d];

        pG1 += gp*gp;
        qG1 += gq*gq;
        wG1 += gw*gw;

        p[d] -= eta_p*gp;
        q[d] -= eta_q*gq;
        w[d] -= eta_w*gw;
    }

    if(lambda_p1 > 0)
    {
        for(mf_int d = d_begin; d < d_end; d++)
        {
            mf_float p1 = max(abs(p[d])-lambda_p1*eta_p, 0.0f);
            p[d] = p[d] >= 0? p1: -p1;
        }
    }

    if (lambda_q1 > 0)
    {
        for(mf_int d = d_begin; d < d_end; d++)
        {
            mf_float q1 = max(abs(w[d])-lambda_q1*eta_w, 0.0f);
            w[d] = w[d] >= 0? q1: -q1;
            q1 = max(abs(q[d])-lambda_q1*eta_q, 0.0f);
            q[d] = q[d] >= 0? q1: -q1;
        }
    }

    if(param.do_nmf)
    {
        for(mf_int d = d_begin; d < d_end; d++)
        {
            p[d] = max(p[d], (mf_float)0.0);
            q[d] = max(q[d], (mf_float)0.0);
            w[d] = max(w[d], (mf_float)0.0);
        }
    }

    *pG += pG1*rk;
    *qG += qG1*rk;
    *wG += wG1*rk;
}

void BPRSolver::prepare_for_sg_update()
{
    prepare_negative();
    calc_z(z, model.k, p, q, w);
    z = exp(-z);
    loss += log(1+z);
    error = loss;
    z = z/(1+z);
}
#endif

class COL_BPR_MFOC : public BPRSolver
{
public:
    COL_BPR_MFOC(Scheduler &scheduler, vector<BlockBase*> &blocks,
                 mf_float *PG, mf_float *QG, mf_model &model,
                 mf_parameter param, bool &slow_only,
                 bool is_column_oriented=true)
        : BPRSolver(scheduler, blocks, PG, QG, model, param,
                    slow_only, is_column_oriented) {}
protected:
#if defined USESSE
    void load_fixed_variables(
        __m128 &XMMlambda_p1, __m128 &XMMlambda_q1,
        __m128 &XMMlambda_p2, __m128 &XMMlabmda_q2,
        __m128 &XMMeta, __m128 &XMMrk_slow,
        __m128 &XMMrk_fast);
#elif defined USEAVX
    void load_fixed_variables(
        __m256 &XMMlambda_p1, __m256 &XMMlambda_q1,
        __m256 &XMMlambda_p2, __m256 &XMMlabmda_q2,
        __m256 &XMMeta, __m256 &XMMrk_slow,
        __m256 &XMMrk_fast);
#else
    void load_fixed_variables();
#endif
    void prepare_negative();
};

void COL_BPR_MFOC::prepare_negative()
{
    mf_int negative = scheduler.get_negative(bid, bpr_bid, model.m, model.n,
                                             is_column_oriented);
    w = model.P + negative*model.k;
    wG = PG + negative*2;
    swap(p, q);
    swap(pG, qG);
}

#if defined USESSE
void COL_BPR_MFOC::load_fixed_variables(
    __m128 &XMMlambda_p1, __m128 &XMMlambda_q1,
    __m128 &XMMlambda_p2, __m128 &XMMlambda_q2,
    __m128 &XMMeta, __m128 &XMMrk_slow,
    __m128 &XMMrk_fast)
{
    XMMlambda_p1 = _mm_set1_ps(param.lambda_q1);
    XMMlambda_q1 = _mm_set1_ps(param.lambda_p1);
    XMMlambda_p2 = _mm_set1_ps(param.lambda_q2);
    XMMlambda_q2 = _mm_set1_ps(param.lambda_p2);
    XMMeta = _mm_set1_ps(param.eta);
    XMMrk_slow = _mm_set1_ps((mf_float)1.0/kALIGN);
    XMMrk_fast = _mm_set1_ps((mf_float)1.0/(model.k-kALIGN));
}
#elif defined USEAVX
void COL_BPR_MFOC::load_fixed_variables(
    __m256 &XMMlambda_p1, __m256 &XMMlambda_q1,
    __m256 &XMMlambda_p2, __m256 &XMMlambda_q2,
    __m256 &XMMeta, __m256 &XMMrk_slow,
    __m256 &XMMrk_fast)
{
    XMMlambda_p1 = _mm256_set1_ps(param.lambda_q1);
    XMMlambda_q1 = _mm256_set1_ps(param.lambda_p1);
    XMMlambda_p2 = _mm256_set1_ps(param.lambda_q2);
    XMMlambda_q2 = _mm256_set1_ps(param.lambda_p2);
    XMMeta = _mm256_set1_ps(param.eta);
    XMMrk_slow = _mm256_set1_ps((mf_float)1.0/kALIGN);
    XMMrk_fast = _mm256_set1_ps((mf_float)1.0/(model.k-kALIGN));
}
#else
void COL_BPR_MFOC::load_fixed_variables()
{
    lambda_p1 = param.lambda_q1;
    lambda_q1 = param.lambda_p1;
    lambda_p2 = param.lambda_q2;
    lambda_q2 = param.lambda_p2;
    rk_slow = (mf_float)1.0/kALIGN;
    rk_fast = (mf_float)1.0/(model.k-kALIGN);
}
#endif

class ROW_BPR_MFOC : public BPRSolver
{
public:
    ROW_BPR_MFOC(Scheduler &scheduler, vector<BlockBase*> &blocks,
                 mf_float *PG, mf_float *QG, mf_model &model,
                 mf_parameter param, bool &slow_only,
                 bool is_column_oriented = false)
        : BPRSolver(scheduler, blocks, PG, QG, model, param,
                    slow_only, is_column_oriented) {}
protected:
    void prepare_negative();
};

void ROW_BPR_MFOC::prepare_negative()
{
    mf_int negative = scheduler.get_negative(bid, bpr_bid, model.m, model.n,
                                             is_column_oriented);
    w = model.Q + negative*model.k;
    wG = QG + negative*2;
}


class SolverFactory
{
public:
    static shared_ptr<SolverBase> get_solver(
        Scheduler &scheduler,
        vector<BlockBase*> &blocks,
        mf_float *PG,
        mf_float *QG,
        mf_model &model,
        mf_parameter param,
        bool &slow_only);
};

shared_ptr<SolverBase> SolverFactory::get_solver(
    Scheduler &scheduler,
    vector<BlockBase*> &blocks,
    mf_float *PG,
    mf_float *QG,
    mf_model &model,
    mf_parameter param,
    bool &slow_only)
{
    shared_ptr<SolverBase> solver;

    switch(param.fun)
    {
        case P_L2_MFR:
            solver = shared_ptr<SolverBase>(new L2_MFR(scheduler, blocks,
                        PG, QG, model, param, slow_only));
            break;
        case P_L1_MFR:
            solver = shared_ptr<SolverBase>(new L1_MFR(scheduler, blocks,
                        PG, QG, model, param, slow_only));
            break;
        case P_KL_MFR:
            solver = shared_ptr<SolverBase>(new KL_MFR(scheduler, blocks,
                        PG, QG, model, param, slow_only));
            break;
        case P_LR_MFC:
            solver = shared_ptr<SolverBase>(new LR_MFC(scheduler, blocks,
                        PG, QG, model, param, slow_only));
            break;
        case P_L2_MFC:
            solver = shared_ptr<SolverBase>(new L2_MFC(scheduler, blocks,
                        PG, QG, model, param, slow_only));
            break;
        case P_L1_MFC:
            solver = shared_ptr<SolverBase>(new L1_MFC(scheduler, blocks,
                        PG, QG, model, param, slow_only));
            break;
        case P_ROW_BPR_MFOC:
            solver = shared_ptr<SolverBase>(new ROW_BPR_MFOC(scheduler,
                        blocks, PG, QG, model, param, slow_only));
            break;
        case P_COL_BPR_MFOC:
            solver = shared_ptr<SolverBase>(new COL_BPR_MFOC(scheduler,
                        blocks, PG, QG, model, param, slow_only));
            break;
        default:
            throw invalid_argument("unknown error function");
    }
    return solver;
}

void fpsg_core(
    Utility &util,
    Scheduler &sched,
    mf_problem *tr,
    mf_problem *va,
    mf_parameter param,
    mf_float scale,
    vector<BlockBase*> &block_ptrs,
    vector<mf_int> &omega_p,
    vector<mf_int> &omega_q,
    shared_ptr<mf_model> &model,
    vector<mf_int> cv_blocks,
    mf_double *cv_error)
{
#if defined USESSE || defined USEAVX
    auto flush_zero_mode = _MM_GET_FLUSH_ZERO_MODE();
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
    if(tr->nnz == 0)
    {
        cout << "warning: train on an empty training set" << endl;
        return;
    }

    if(param.fun == P_L2_MFR ||
       param.fun == P_L1_MFR ||
       param.fun == P_KL_MFR)
    {
        switch(param.fun)
        {
            case P_L2_MFR:
                param.lambda_p2 /= scale;
                param.lambda_q2 /= scale;
                param.lambda_p1 /= (mf_float)pow(scale, 1.5);
                param.lambda_q1 /= (mf_float)pow(scale, 1.5);
                break;
            case P_L1_MFR:
            case P_KL_MFR:
                param.lambda_p1 /= sqrt(scale);
                param.lambda_q1 /= sqrt(scale);
                break;
        }
    }

    if(!param.quiet)
    {
        cout.width(4);
        cout << "iter";
        cout.width(13);
        cout << "tr_"+util.get_error_legend();
        if(va->nnz != 0)
        {
            cout.width(13);
            cout << "va_"+util.get_error_legend();
        }
        cout.width(13);
        cout << "obj";
        cout << "\n";
    }

    bool slow_only = param.lambda_p1 == 0 && param.lambda_q1 == 0? true: false;
    vector<mf_float> PG(model->m*2, 1), QG(model->n*2, 1);

    vector<shared_ptr<SolverBase>> solvers(param.nr_threads);
    vector<thread> threads;
    threads.reserve(param.nr_threads);
    for(mf_int i = 0; i < param.nr_threads; i++)
    {
        solvers[i] = SolverFactory::get_solver(sched, block_ptrs,
                                               PG.data(), QG.data(),
                                               *model, param, slow_only);
        threads.emplace_back(&SolverBase::run, solvers[i].get());
    }

    for(mf_int iter = 0; iter < param.nr_iters; iter++)
    {
        sched.wait_for_jobs_done();

        if(!param.quiet)
        {
            mf_double reg = 0;
            mf_double reg1 = util.calc_reg1(*model, param.lambda_p1,
                             param.lambda_q1, omega_p, omega_q);
            mf_double reg2 = util.calc_reg2(*model, param.lambda_p2,
                             param.lambda_q2, omega_p, omega_q);
            mf_double tr_loss = sched.get_loss();
            mf_double tr_error = sched.get_error()/tr->nnz;

            switch(param.fun)
            {
                case P_L2_MFR:
                    reg = (reg1+reg2)*scale*scale;
                    tr_loss *= scale*scale;
                    tr_error = sqrt(tr_error*scale*scale);
                    break;
                case P_L1_MFR:
                case P_KL_MFR:
                    reg = (reg1+reg2)*scale;
                    tr_loss *= scale;
                    tr_error *= scale;
                    break;
                default:
                    reg = reg1+reg2;
                    break;
            }

            cout.width(4);
            cout << iter;
            cout.width(13);
            cout << fixed << setprecision(4) << tr_error;
            if(va->nnz != 0)
            {
                Block va_block(va->R, va->R+va->nnz);
                vector<BlockBase*> va_blocks(1, &va_block);
                vector<mf_int> va_block_ids(1, 0);
                mf_double va_error =
                    util.calc_error(va_blocks, va_block_ids, *model)/va->nnz;
                switch(param.fun)
                {
                    case P_L2_MFR:
                        va_error = sqrt(va_error*scale*scale);
                        break;
                    case P_L1_MFR:
                    case P_KL_MFR:
                        va_error *= scale;
                        break;
                }

                cout.width(13);
                cout << fixed << setprecision(4) << va_error;
            }
            cout.width(13);
            cout << fixed << setprecision(4) << scientific << reg+tr_loss;
            cout << "\n" << flush;
        }

        if(iter == 0)
            slow_only = false;

        sched.resume();
    }
    sched.terminate();

    for(auto &thread : threads)
        thread.join();

    if(cv_error != nullptr && cv_blocks.size() > 0)
    {
        mf_long cv_count = 0;
        for(auto block : cv_blocks)
            cv_count += block_ptrs[block]->get_nnz();

        *cv_error = util.calc_error(block_ptrs, cv_blocks, *model)/cv_count;

        switch(param.fun)
        {
            case P_L2_MFR:
                *cv_error = sqrt(*cv_error*scale*scale);
                break;
            case P_L1_MFR:
            case P_KL_MFR:
                *cv_error *= scale;
                break;
        }
    }

#if defined USESSE || defined USEAVX
    _MM_SET_FLUSH_ZERO_MODE(flush_zero_mode);
#endif
}

shared_ptr<mf_model> fpsg(
    mf_problem const *tr_,
    mf_problem const *va_,
    mf_parameter param,
    vector<mf_int> cv_blocks=vector<mf_int>(),
    mf_double *cv_error=nullptr)
{
    shared_ptr<mf_model> model;
try
{
    Utility util(param.fun, param.nr_threads);
    Scheduler sched(param.nr_bins, param.nr_threads, cv_blocks);
    shared_ptr<mf_problem> tr;
    shared_ptr<mf_problem> va;
    vector<Block> blocks(param.nr_bins*param.nr_bins);
    vector<BlockBase*> block_ptrs(param.nr_bins*param.nr_bins);
    vector<mf_node*> ptrs;
    vector<mf_int> p_map;
    vector<mf_int> q_map;
    vector<mf_int> inv_p_map;
    vector<mf_int> inv_q_map;
    vector<mf_int> omega_p;
    vector<mf_int> omega_q;
    mf_float avg = 0;
    mf_float std_dev = 0;
    mf_float scale = 1;

    if(param.copy_data)
    {
        struct deleter
        {
            void operator() (mf_problem *prob)
            {
                delete[] prob->R;
                delete prob;
            }
        };

        tr = shared_ptr<mf_problem>(
                Utility::copy_problem(tr_, true), deleter());
        va = shared_ptr<mf_problem>(
                Utility::copy_problem(va_, true), deleter());
    }
    else
    {
        tr = shared_ptr<mf_problem>(Utility::copy_problem(tr_, false));
        va = shared_ptr<mf_problem>(Utility::copy_problem(va_, false));
    }

    util.collect_info(*tr, avg, std_dev);

    if(param.fun == P_L2_MFR ||
       param.fun == P_L1_MFR ||
       param.fun == P_KL_MFR)
        scale = max((mf_float)1e-4, std_dev);

    p_map = Utility::gen_random_map(tr->m);
    q_map = Utility::gen_random_map(tr->n);
    inv_p_map = Utility::gen_inv_map(p_map);
    inv_q_map = Utility::gen_inv_map(q_map);
    omega_p = vector<mf_int>(tr->m, 0);
    omega_q = vector<mf_int>(tr->n, 0);

    util.shuffle_problem(*tr, p_map, q_map);
    util.shuffle_problem(*va, p_map, q_map);
    util.scale_problem(*tr, (mf_float)1.0/scale);
    util.scale_problem(*va, (mf_float)1.0/scale);
    ptrs = util.grid_problem(*tr, param.nr_bins, omega_p, omega_q, blocks);

    model = shared_ptr<mf_model>(Utility::init_model(param.fun,
                tr->m, tr->n, param.k, avg/scale, omega_p, omega_q),
                [] (mf_model *ptr) { mf_destroy_model(&ptr); });

    for(mf_int i = 0; i < (mf_long)blocks.size(); i++)
        block_ptrs[i] = &blocks[i];

    fpsg_core(util, sched, tr.get(), va.get(), param, scale,
              block_ptrs, omega_p, omega_q, model, cv_blocks, cv_error);

    if(!param.copy_data)
    {
        util.scale_problem(*tr, scale);
        util.scale_problem(*va, scale);
        util.shuffle_problem(*tr, inv_p_map, inv_q_map);
        util.shuffle_problem(*va, inv_p_map, inv_q_map);
    }

    util.scale_model(*model, scale);
    Utility::shrink_model(*model, param.k);
    Utility::shuffle_model(*model, inv_p_map, inv_q_map);
}
catch(exception const &e)
{
    cerr << e.what() << endl;
    throw;
}
    return model;
}

shared_ptr<mf_model> fpsg_on_disk(
    const string tr_path,
    const string va_path,
    mf_parameter param,
    vector<mf_int> cv_blocks=vector<mf_int>(),
    mf_double *cv_error=nullptr)
{
    shared_ptr<mf_model> model;
try
{
    Utility util(param.fun, param.nr_threads);
    Scheduler sched(param.nr_bins, param.nr_threads, cv_blocks);
    mf_problem tr = {};
    mf_problem va = read_problem(va_path.c_str());
    vector<BlockOnDisk> blocks(param.nr_bins*param.nr_bins);
    vector<BlockBase*> block_ptrs(param.nr_bins*param.nr_bins);
    vector<mf_int> p_map;
    vector<mf_int> q_map;
    vector<mf_int> inv_p_map;
    vector<mf_int> inv_q_map;
    vector<mf_int> omega_p;
    vector<mf_int> omega_q;
    mf_float avg = 0;
    mf_float std_dev = 0;
    mf_float scale = 1;

    util.collect_info_on_disk(tr_path, tr, avg, std_dev);

    if(param.fun == P_L2_MFR ||
       param.fun == P_L1_MFR ||
       param.fun == P_KL_MFR)
        scale = max((mf_float)1e-4, std_dev);

    p_map = Utility::gen_random_map(tr.m);
    q_map = Utility::gen_random_map(tr.n);
    inv_p_map = Utility::gen_inv_map(p_map);
    inv_q_map = Utility::gen_inv_map(q_map);
    omega_p = vector<mf_int>(tr.m, 0);
    omega_q = vector<mf_int>(tr.n, 0);

    util.shuffle_problem(va, p_map, q_map);
    util.scale_problem(va, (mf_float)1.0/scale);

    util.grid_shuffle_scale_problem_on_disk(
        tr.m, tr.n, param.nr_bins, scale, tr_path,
        p_map, q_map, omega_p, omega_q, blocks);

    model = shared_ptr<mf_model>(Utility::init_model(param.fun,
                tr.m, tr.n, param.k, avg/scale, omega_p, omega_q),
                [] (mf_model *ptr) { mf_destroy_model(&ptr); });

    for(mf_int i = 0; i < (mf_long)blocks.size(); i++)
        block_ptrs[i] = &blocks[i];

    fpsg_core(util, sched, &tr, &va, param, scale,
              block_ptrs, omega_p, omega_q, model, cv_blocks, cv_error);

    delete [] va.R;

    util.scale_model(*model, scale);
    Utility::shrink_model(*model, param.k);
    Utility::shuffle_model(*model, inv_p_map, inv_q_map);
}
catch(exception const &e)
{
    cerr << e.what() << endl;
    throw;
}
    return model;
}

bool check_parameter(mf_parameter param)
{
    if(param.fun != P_L2_MFR &&
       param.fun != P_L1_MFR &&
       param.fun != P_KL_MFR &&
       param.fun != P_LR_MFC &&
       param.fun != P_L2_MFC &&
       param.fun != P_L1_MFC &&
       param.fun != P_ROW_BPR_MFOC &&
       param.fun != P_COL_BPR_MFOC)
    {
        cerr << "unknown loss function" << endl;
        return false;
    }

    if(param.k < 1)
    {
        cerr << "number of factors must be greater than zero" << endl;
        return false;
    }

    if(param.nr_threads < 1)
    {
        cerr << "number of threads must be greater than zero" << endl;
        return false;
    }

    if(param.nr_bins < 1 || param.nr_bins < param.nr_threads)
    {
        cerr << "number of bins must be greater than number of threads"
             << endl;
        return false;
    }

    if(param.nr_iters < 1)
    {
        cerr << "number of iterations must be greater than zero" << endl;
        return false;
    }

    if(param.lambda_p1 < 0 ||
       param.lambda_p2 < 0 ||
       param.lambda_q1 < 0 ||
       param.lambda_q2 < 0)
    {
        cerr << "regularization coefficient must be non-negative" << endl;
        return false;
    }

    if(param.eta <= 0)
    {
        cerr << "learning rate must be greater than zero" << endl;
        return false;
    }

    if(param.fun == P_KL_MFR && !param.do_nmf)
    {
        cerr << "--nmf must be set when using generalized KL-divergence"
             << endl;
        return false;
    }

    if(param.nr_bins <= 2*param.nr_threads)
    {
        cerr << "Warning: insufficient blocks may slow down the training"
             << "process (4*nr_threads^2+1 blocks is suggested)" << endl;
    }

    return true;
}

//--------------------------------------
//-----Classes for cross validation-----
//--------------------------------------

class CrossValidatorBase
{
public:
    CrossValidatorBase(mf_parameter param_, mf_int nr_folds_);
    mf_double do_cross_validation();
    virtual mf_double do_cv1(vector<mf_int> &hidden_blocks) = 0;
protected:
    mf_parameter param;
    mf_int nr_bins;
    mf_int nr_folds;
    mf_int nr_blocks_per_fold;
    bool quiet;
    Utility util;
    mf_double cv_error;
};

CrossValidatorBase::CrossValidatorBase(mf_parameter param_, mf_int nr_folds_)
    : param(param_), nr_bins(param_.nr_bins), nr_folds(nr_folds_),
      nr_blocks_per_fold(nr_bins*nr_bins/nr_folds), quiet(param_.quiet),
      util(param.fun, param.nr_threads), cv_error(0)
{
    param.quiet = true;
}

mf_double CrossValidatorBase::do_cross_validation()
{
    vector<mf_int> cv_blocks;
    srand(0);
    for(mf_int block = 0; block < nr_bins*nr_bins; block++)
        cv_blocks.push_back(block);
    random_shuffle(cv_blocks.begin(), cv_blocks.end());

    if(!quiet)
    {
        cout.width(4);
        cout << "fold";
        cout.width(10);
        cout << util.get_error_legend();
        cout << endl;
    }

    cv_error = 0;

    for(mf_int fold = 0; fold < nr_folds; fold++)
    {
        mf_int begin = fold*nr_blocks_per_fold;
        mf_int end = min((fold+1)*nr_blocks_per_fold, nr_bins*nr_bins);
        vector<mf_int> hidden_blocks(cv_blocks.begin()+begin,
                                    cv_blocks.begin()+end);

        mf_double err = do_cv1(hidden_blocks);
        cv_error += err;

        if(!quiet)
        {
            cout.width(4);
            cout << fold;
            cout.width(10);
            cout << fixed << setprecision(4) << err;
            cout << endl;
        }
    }

    if(!quiet)
    {
        cout.width(14);
        cout.fill('=');
        cout << "" << endl;
        cout.fill(' ');
        cout.width(4);
        cout << "avg";
        cout.width(10);
        cout << fixed << setprecision(4) << cv_error/nr_folds;
        cout << endl;
    }

    return cv_error/nr_folds;
}

class CrossValidator : public CrossValidatorBase
{
public:
    CrossValidator(
        mf_parameter param_, mf_int nr_folds_, mf_problem const *prob_)
        : CrossValidatorBase(param_, nr_folds_), prob(prob_) {};
    mf_double do_cv1(vector<mf_int> &hidden_blocks);
private:
    mf_problem const *prob;
};

mf_double CrossValidator::do_cv1(vector<mf_int> &hidden_blocks)
{
    mf_double err = 0;
    fpsg(prob, nullptr, param, hidden_blocks, &err);
    return err;
}

class CrossValidatorOnDisk : public CrossValidatorBase
{
public:
    CrossValidatorOnDisk(
        mf_parameter param_, mf_int nr_folds_, string data_path_)
        : CrossValidatorBase(param_, nr_folds_), data_path(data_path_) {};
    mf_double do_cv1(vector<mf_int> &hidden_blocks);
private:
    string data_path;
};

mf_double CrossValidatorOnDisk::do_cv1(vector<mf_int> &hidden_blocks)
{
    mf_double err = 0;
    fpsg_on_disk(data_path, string(), param, hidden_blocks, &err);
    return err;
}

} // unnamed namespace

mf_model* mf_train_with_validation(
    mf_problem const *tr,
    mf_problem const *va,
    mf_parameter param)
{
    if(!check_parameter(param))
        return nullptr;

    shared_ptr<mf_model> model = fpsg(tr, va, param);

    mf_model *model_ret = new mf_model;

    model_ret->fun = model->fun;
    model_ret->m = model->m;
    model_ret->n = model->n;
    model_ret->k = model->k;
    model_ret->b = model->b;

    model_ret->P = model->P;
    model->P = nullptr;

    model_ret->Q = model->Q;
    model->Q = nullptr;

    return model_ret;
}

mf_model* mf_train_with_validation_on_disk(
    char const *tr_path,
    char const *va_path,
    mf_parameter param)
{
    if(!check_parameter(param))
        return nullptr;

    shared_ptr<mf_model> model = fpsg_on_disk(
        string(tr_path), string(va_path), param);

    mf_model *model_ret = new mf_model;

    model_ret->fun = model->fun;
    model_ret->m = model->m;
    model_ret->n = model->n;
    model_ret->k = model->k;
    model_ret->b = model->b;

    model_ret->P = model->P;
    model->P = nullptr;

    model_ret->Q = model->Q;
    model->Q = nullptr;

    return model_ret;
}

mf_model* mf_train(mf_problem const *prob, mf_parameter param)
{
    return mf_train_with_validation(prob, nullptr, param);
}

mf_model* mf_train_on_disk(char const *tr_path, mf_parameter param)
{
    return mf_train_with_validation_on_disk(tr_path, "", param);
}

mf_double mf_cross_validation(
    mf_problem const *prob,
    mf_int nr_folds,
    mf_parameter param)
{
    if(!check_parameter(param))
        return 0;

    CrossValidator validator(param, nr_folds, prob);

    return validator.do_cross_validation();
}

mf_double mf_cross_validation_on_disk(
    char const *prob,
    mf_int nr_folds,
    mf_parameter param)
{
    if(!check_parameter(param))
        return 0;

    CrossValidatorOnDisk validator(param, nr_folds, string(prob));

    return validator.do_cross_validation();
}

mf_problem read_problem(string path)
{
    mf_problem prob;
    prob.m = 0;
    prob.n = 0;
    prob.nnz = 0;
    prob.R = nullptr;

    if(path.empty())
        return prob;

    ifstream f(path);
    if(!f.is_open())
        return prob;

    string line;
    while(getline(f, line))
        prob.nnz++;

    mf_node *R = new mf_node[prob.nnz];

    f.close();
    f.open(path);

    mf_long idx = 0;
    for(mf_node N; f >> N.u >> N.v >> N.r;)
    {
        if(N.u+1 > prob.m)
            prob.m = N.u+1;
        if(N.v+1 > prob.n)
            prob.n = N.v+1;
        R[idx] = N;
        idx++;
    }
    prob.R = R;

    f.close();

    return prob;
}

mf_int mf_save_model(mf_model const *model, char const *path)
{
    ofstream f(path);
    if(!f.is_open())
        return 1;

    f << "f " << model->fun << endl;
    f << "m " << model->m << endl;
    f << "n " << model->n << endl;
    f << "k " << model->k << endl;
    f << "b " << model->b << endl;

    auto write = [&] (mf_float *ptr, mf_int size, char prefix)
    {
        for(mf_int i = 0; i < size; i++)
        {
            mf_float *ptr1 = ptr + (mf_long)i*model->k;
            f << prefix << i << " ";
            if(isnan(ptr1[0]))
            {
                f << "F ";
                for(mf_int d = 0; d < model->k; d++)
                    f << 0 << " ";
            }
            else
            {
                f << "T ";
                for(mf_int d = 0; d < model->k; d++)
                    f << ptr1[d] << " ";
            }
            f << endl;
        }

    };

    write(model->P, model->m, 'p');
    write(model->Q, model->n, 'q');

    f.close();

    return 0;
}

mf_model* mf_load_model(char const *path)
{
    ifstream f(path);
    if(!f.is_open())
        return nullptr;

    string dummy;

    mf_model *model = new mf_model;
    model->P = nullptr;
    model->Q = nullptr;

    f >> dummy >> model->fun >> dummy >> model->m >> dummy >> model->n >>
         dummy >> model->k >> dummy >> model->b;

    try
    {
        model->P = Utility::malloc_aligned_float((mf_long)model->m*model->k);
        model->Q = Utility::malloc_aligned_float((mf_long)model->n*model->k);
    }
    catch(bad_alloc const &e)
    {
        cerr << e.what() << endl;
        mf_destroy_model(&model);
        return nullptr;
    }

    auto read = [&] (mf_float *ptr, mf_int size)
    {
        for(mf_int i = 0; i < size; i++)
        {
            mf_float *ptr1 = ptr + (mf_long)i*model->k;
            f >> dummy >> dummy;
            if(dummy.compare("F") == 0) // nan vector starts with "F"
                for(mf_int d = 0; d < model->k; d++)
                {
                    f >> dummy;
                    ptr1[d] = numeric_limits<mf_float>::quiet_NaN();
                }
            else
                for(mf_int d = 0; d < model->k; d++)
                    f >> ptr1[d];
        }
    };

    read(model->P, model->m);
    read(model->Q, model->n);

    f.close();

    return model;
}

void mf_destroy_model(mf_model **model)
{
    if(model == nullptr || *model == nullptr)
        return;
#ifdef _WIN32
    _aligned_free((*model)->P);
    _aligned_free((*model)->Q);
#else
    free((*model)->P);
    free((*model)->Q);
#endif
    delete *model;
    *model = nullptr;
}

mf_float mf_predict(mf_model const *model, mf_int u, mf_int v)
{
    if(u < 0 || u >= model->m || v < 0 || v >= model->n)
        return model->b;

    mf_float *p = model->P+(mf_long)u*model->k;
    mf_float *q = model->Q+(mf_long)v*model->k;

    mf_float z = std::inner_product(p, p+model->k, q, (mf_float)0.0f);

    if(isnan(z))
        z = model->b;

    if(model->fun == P_L2_MFC &&
       model->fun == P_L1_MFC &&
       model->fun == P_LR_MFC)
        z = z > 0.0f? 1.0f: -1.0f;

    return z;
}

mf_double calc_rmse(mf_problem *prob, mf_model *model)
{
    if(prob->nnz == 0)
        return 0;
    mf_double loss = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+:loss)
#endif
    for(mf_long i = 0; i < prob->nnz; i++)
    {
        mf_node &N = prob->R[i];
        mf_float e = N.r - mf_predict(model, N.u, N.v);
        loss += e*e;
    }
    return sqrt(loss/prob->nnz);
}

mf_double calc_mae(mf_problem *prob, mf_model *model)
{
    if(prob->nnz == 0)
        return 0;
    mf_double loss = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+:loss)
#endif
    for(mf_long i = 0; i < prob->nnz; i++)
    {
        mf_node &N = prob->R[i];
        loss += abs(N.r - mf_predict(model, N.u, N.v));
    }
    return loss/prob->nnz;
}

mf_double calc_gkl(mf_problem *prob, mf_model *model)
{
    if(prob->nnz == 0)
        return 0;
    mf_double loss = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+:loss)
#endif
    for(mf_long i = 0; i < prob->nnz; i++)
    {
        mf_node &N = prob->R[i];
        mf_float z = mf_predict(model, N.u, N.v);
        loss += N.r*log(N.r/z)-N.r+z;
    }
    return loss/prob->nnz;
}

mf_double calc_logloss(mf_problem *prob, mf_model *model)
{
    if(prob->nnz == 0)
        return 0;
    mf_double logloss = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+:logloss)
#endif
    for(mf_long i = 0; i < prob->nnz; i++)
    {
        mf_node &N = prob->R[i];
        mf_float z = mf_predict(model, N.u, N.v);
        if(N.r > 0)
            logloss += log(1.0+exp(-z));
        else
            logloss += log(1.0+exp(z));
    }
    return logloss/prob->nnz;
}

mf_double calc_accuracy(mf_problem *prob, mf_model *model)
{
    if(prob->nnz == 0)
        return 0;
    mf_double acc = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+:acc)
#endif
    for(mf_long i = 0; i < prob->nnz; i++)
    {
        mf_node &N = prob->R[i];
        mf_float z = mf_predict(model, N.u, N.v);
        if(N.r > 0)
            acc += z > 0? 1: 0;
        else
            acc += z < 0? 1: 0;
    }
    return acc/prob->nnz;
}

pair<mf_double, mf_double> calc_mpr_auc(mf_problem *prob,
                                        mf_model *model, bool transpose)
{
    mf_int mf_node::*row_ptr;
    mf_int mf_node::*col_ptr;
    mf_int m = 0, n = 0;
    if(!transpose)
    {
        row_ptr = &mf_node::u;
        col_ptr = &mf_node::v;
        m = max(prob->m, model->m);
        n = max(prob->n, model->n);
    }
    else
    {
        row_ptr = &mf_node::v;
        col_ptr = &mf_node::u;
        m = max(prob->n, model->n);
        n = max(prob->m, model->m);
    }

    auto sort_by_id = [&] (mf_node const &lhs, mf_node const &rhs)
    {
        return tie(lhs.*row_ptr, lhs.*col_ptr) <
               tie(rhs.*row_ptr, rhs.*col_ptr);
    };

    sort(prob->R, prob->R+prob->nnz, sort_by_id);

    auto sort_by_pred = [&] (pair<mf_node, mf_float> const &lhs,
        pair<mf_node, mf_float> const &rhs) { return lhs.second < rhs.second; };

    vector<mf_int> pos_cnts(m+1, 0);
    for(mf_int i = 0; i < prob->nnz; i++)
        pos_cnts[prob->R[i].*row_ptr+1]++;
    for(mf_int i = 1; i < m+1; i++)
        pos_cnts[i] += pos_cnts[i-1];

    mf_int total_m = 0;
    mf_long total_pos = 0;
    mf_double all_u_mpr = 0;
    mf_double all_u_auc = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: total_m, total_pos, all_u_mpr, all_u_auc)
#endif
    for(mf_int i = 0; i < m; i++)
    {
        if(pos_cnts[i+1]-pos_cnts[i] < 1)
            continue;

        vector<pair<mf_node, mf_float>> row(n);

        for(mf_int j = 0; j < n; j++)
        {
            mf_node N;
            N.*row_ptr = i;
            N.*col_ptr = j;
            N.r = 0;
            row[j] = make_pair(N, mf_predict(model, N.u, N.v));
        }

        mf_int pos = 0;
        vector<mf_int> index(pos_cnts[i+1]-pos_cnts[i], 0);
        for(mf_int j = pos_cnts[i]; j < pos_cnts[i+1]; j++)
        {
            if(prob->R[j].r <= 0)
                continue;

            mf_int col = prob->R[j].*col_ptr;
            row[col].first.r = prob->R[j].r;
            index[pos] = col;
            pos++;
        }

        if(n-pos < 1 || pos < 1)
            continue;

        total_m++;
        total_pos += pos;

        mf_int count = 0;
        for(mf_int k = 0; k < pos; k++)
        {
            swap(row[count], row[index[k]]);
            count++;
        }
        sort(row.begin(), row.begin()+pos, sort_by_pred);

        mf_double u_mpr = 0;
        mf_double u_auc = 0;
        for(auto neg_it = row.begin()+pos; neg_it != row.end(); neg_it++)
        {
            if(row[pos-1].second <= neg_it->second)
            {
                u_mpr += pos;
                continue;
            }

            mf_int left = 0;
            mf_int right = pos-1;
            while(left < right)
            {
                mf_int mid = (left+right)/2;
                if(row[mid].second > neg_it->second)
                    right = mid;
                else
                    left = mid+1;
            }
            u_mpr += left;
            u_auc += pos-left;
        }

        all_u_mpr += u_mpr/(n-pos);
        all_u_auc += u_auc/(n-pos)/pos;
    }

    all_u_mpr /= total_pos;
    all_u_auc /= total_m;

    return make_pair(all_u_mpr, all_u_auc);
}

mf_double calc_mpr(mf_problem *prob, mf_model *model, bool transpose)
{
    return calc_mpr_auc(prob, model, transpose).first;
}

mf_double calc_auc(mf_problem *prob, mf_model *model, bool transpose)
{
    return calc_mpr_auc(prob, model, transpose).second;
}

mf_parameter mf_get_default_param()
{
    mf_parameter param;

    param.fun = P_L2_MFR;
    param.k = 8;
    param.nr_threads = 12;
    param.nr_bins = 20;
    param.nr_iters = 20;
    param.lambda_p1 = 0.0f;
    param.lambda_q1 = 0.0f;
    param.lambda_p2 = 0.1f;
    param.lambda_q2 = 0.1f;
    param.eta = 0.1f;
    param.do_nmf = false;
    param.quiet = false;
    param.copy_data = true;

    return param;
}

}
