#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <condition_variable>
#include <queue>
#include <algorithm>
#include <thread>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include "mf.h"

#if defined NOSSE && defined USEAVX
#error "NOSSE and USEAVX cannot be define simultaneously"
#endif

#if !defined NOSSE && !defined USEAVX
#include <pmmintrin.h>
#endif

#if defined USEAVX
#include <immintrin.h>
#endif

namespace 
{

struct TrainOption
{
    TrainOption()
        : tr_path(), va_path(), model_path(), en_show_tr_rmse(false),
          en_show_obj(false), en_rand_shuffle(false), en_avg(false),
          param(), nr_gubs(1), nr_gibs(1), nr_threads(1), nr_iter(40) {}
    std::string tr_path, va_path, model_path;
    bool en_show_tr_rmse, en_show_obj, en_rand_shuffle, en_avg;
    Parameter param;
    int nr_gubs, nr_gibs, nr_threads, nr_iter;
};

void train_help()
{
    printf(
    "usage: libmf train [options] binary_train_file [model]\n"
    "\n"
    "options:\n"
    "-k <dimensions>: set the number of dimensions (default 40)\n"
    "-t <iterations>: set the number of iterations (default 40)\n"
    "-s <number of threads>: set the number of threads (default 1)\n"
    "-p <cost>: set the regularization cost for P (default 1)\n"
    "-q <cost>: set the regularization cost for Q (default 1)\n"
    "-ub <cost>: set the regularization cost for user bias. Set <0 to disable. (default -1)\n"
    "-ib <cost>: set the regularization cost for item bias. Set <0 to disable. (default -1)\n"
    "-g <gamma>: set the learning rate for parallel SGD (default 0.001)\n"
    "-v <path>: set the path to validation set\n"
    "    This option may slow down the training procedure.\n"
    "-blk <blocks>: set the number of blocks for parallel SGD (default 2 threads x 2 threads)\n"
    "    For example, if you want 3x4 blocks, then use '-blk 3x4'.\n"
    "--rand-shuffle --no-rand-shuffle: enable / disable random suffle (default disabled)\n"
    "    This options should be used when the data is imbalanced.\n"
    "--tr-rmse --no-tr-rmse: enable / disable show RMSE on training data (default disabled)\n"
    "    This option shows the estimated RMSE on training data.\n"
    "--obj --no-obj: enable / disable show objective value (default disabled)\n"
    "    This option shows the estimated objective value on training data. It may slow down the training procedure.\n"
    "--use-avg --no-use-avg: enable / disable using training data average (default disabled)\n"
    );
}

std::shared_ptr<TrainOption> parse_train_option(const int argc,
                                                char const * const * const argv)
{
    if(argc == 0)
    {
        train_help();
        return std::shared_ptr<TrainOption>(nullptr);
    }

    std::shared_ptr<TrainOption> option(new TrainOption);
    option->nr_gubs = option->nr_gibs = 0;

    int i = 0;
    for(; i < argc; i++)
    {
        if(argv[i][0] != '-')
            break;
        if(i+1 >= argc)
        {
            fprintf(stderr, "Error: Invalid command.\n");
            return std::shared_ptr<TrainOption>(nullptr);
        }
        if(!strcmp(argv[i], "-k"))
        {
            option->param.dim = atoi(argv[++i]);
            if(option->param.dim <= 0)
            {
                fprintf(stderr, "Error: The number of dimensions should be greater than zero.\n");
                return std::shared_ptr<TrainOption>(nullptr);
            }
        }
        else if(!strcmp(argv[i], "-t"))
        {
            option->nr_iter = atoi(argv[++i]);
            if(option->nr_iter <= 0)
            {
                fprintf(stderr, "Error: The number of iterations should be greater than zero.\n");
                return std::shared_ptr<TrainOption>(nullptr);
            }
        }
        else if(!strcmp(argv[i], "-s"))
        {
            option->nr_threads = atoi(argv[++i]);
            if(option->nr_threads <= 0)
            {
                fprintf(stderr, "Error: The number of threads should be greater than zero.\n");
                return std::shared_ptr<TrainOption>(nullptr);
            }
        }
        else if(!strcmp(argv[i], "-p"))
        {
            option->param.lp = atof(argv[++i]);
            if(option->param.lp < 0)
            {
                fprintf(stderr, "Error: The regularization cost for P should not be smaller than zero.\n");
                return std::shared_ptr<TrainOption>(nullptr);
            }
        }
        else if(!strcmp(argv[i], "-q"))
        {
            option->param.lq = atof(argv[++i]);
            if(option->param.lq < 0)
            {
                fprintf(stderr, "Error: The regularization cost for Q should not be smaller than zero.\n");
                return std::shared_ptr<TrainOption>(nullptr);
            }
        }
        else if(!strcmp(argv[i], "-g"))
        {
            option->param.gamma = atof(argv[++i]);
            if(option->param.gamma <= 0)
            {
                fprintf(stderr, "Error: The learning rate should be greater than zero.\n");
                return std::shared_ptr<TrainOption>(nullptr);
            }
        }
        else if(!strcmp(argv[i], "-v"))
        {
            option->va_path = std::string(argv[++i]);
        }
        else if(!strcmp(argv[i], "-blk"))
        {
            std::string blk_str(argv[++i]);
            const char *p = strtok(&*blk_str.begin(), "x");
            option->nr_gubs = atoi(p);

            p = strtok(nullptr, "x");
            if(!p)
            {
                fprintf(stderr, "Error: The format of -blk is incorrect.\n");
                return std::shared_ptr<TrainOption>(nullptr);
            }
            option->nr_gibs = atoi(p);

            if(option->nr_gubs <= 0 || option->nr_gibs <= 0)
            {
                fprintf(stderr, "Error: The number of blocks should be greater than zero.\n");
                return std::shared_ptr<TrainOption>(nullptr);
            }
        }
        else if(!strcmp(argv[i], "--rand-shuffle"))
        {
            option->en_rand_shuffle = true;
        }
        else if(!strcmp(argv[i], "--no-rand-shuffle"))
        {
            option->en_rand_shuffle = false;
        }
        else if(!strcmp(argv[i], "--tr-rmse"))
        {
            option->en_show_tr_rmse = true;
        }
        else if(!strcmp(argv[i], "--no-tr-rmse"))
        {
            option->en_show_tr_rmse = false;
        }
        else if(!strcmp(argv[i], "--obj"))
        {
            option->en_show_obj = true;
        }
        else if(!strcmp(argv[i], "--no-obj"))
        {
            option->en_show_obj = false;
        }
        else if(!strcmp(argv[i], "--use-avg"))
        {
            option->en_avg = true;
        }
        else if(!strcmp(argv[i], "--no-use-avg"))
        {
            option->en_avg = false;
        }
        else if(!strcmp(argv[i], "-ub"))
        {
            option->param.lub = atof(argv[++i]);
        }
        else if(!strcmp(argv[i], "-ib"))
        {
            option->param.lib = atof(argv[++i]);
        }
        else
        {
            fprintf(stderr, "Error: Invalid option %s\n", argv[i]);
            return std::shared_ptr<TrainOption>(nullptr);
        }
    }

    if(option->nr_gubs == 0)
        option->nr_gubs = 2*option->nr_threads;
    if(option->nr_gibs == 0)
        option->nr_gibs = 2*option->nr_threads;

    if(option->nr_gubs <= option->nr_threads)
    {
        fprintf(stderr, "Error: The number of user blocks should be greater than number of threads.\n");
        return std::shared_ptr<TrainOption>(nullptr);
    }

    if(option->nr_gibs <= option->nr_threads)
    {
        fprintf(stderr, "Error: The number of item blocks should be greater than number of threads.\n");
        return std::shared_ptr<TrainOption>(nullptr);
    }

    if(i >= argc)
    {
        fprintf(stderr, "Error: Training data not specified.\n");
        return std::shared_ptr<TrainOption>(nullptr);
    }

    option->tr_path = std::string(argv[i++]);

    if(i < argc)
    {
        option->model_path = std::string(argv[i]);
    }
    else
    {
        std::string train_str(argv[i-1]);
        const char *p = strrchr(&*train_str.begin(),'/');
        if(!p)
            p = argv[i-1];
        else
            ++p;
        option->model_path = std::string(p) + ".model";
    }

    return option;
}

struct TrainMatrix
{
    int nr_us, nr_is, nr_gubs, nr_gibs;
    long nr_rs;
    float avg;
    std::vector<Matrix> GM;
};

std::vector<int> gen_map(int const size, bool const shuffle)
{
    std::vector<int> map(size, 0);
    for(int i = 0; i < size; i++)
        map[i] = i;
    if(shuffle)
        std::random_shuffle(map.begin(), map.end());
    return map;
}

Model generate_initial_model(Parameter const &param, int const nr_us,
                             int const nr_is, float const avg)
{
    int const dim_aligned = get_aligned_dim(param.dim);

    Model model;
    model.param = param;
    model.nr_us = nr_us;
    model.nr_is = nr_is;
    model.avg = avg;
    posix_memalign((void**)&model.P, 32, model.nr_us*dim_aligned*sizeof(float));
    posix_memalign((void**)&model.Q, 32, model.nr_is*dim_aligned*sizeof(float));

    auto initialize = [&] (float *ptr, int const count)
    {
        srand48(0L);
        for(int i = 0; i < count; i++)
        {
            int dx = 0;
            for(; dx < param.dim; dx++, ptr++)
                *ptr = 0.1*drand48();
            for(; dx < dim_aligned; dx++, ptr++)
                *ptr = 0;
        }
    };

    initialize(model.P, model.nr_us);
    initialize(model.Q, model.nr_is);
    if(param.lub >= 0)
        model.UB.assign(nr_us, 0);
    if(param.lib >= 0)
        model.IB.assign(nr_is, 0);

    return model;
}

class Monitor
{
public:
    Monitor(TrainMatrix const &Tr, Matrix const * const Va,
            Model const * const model, bool const en_show_tr_rmse,
            bool const en_show_obj);
    void scan_tr(TrainMatrix const &Tr);
    void print_header();
    void print(int const iter, float const time, double const loss,
               float const tr_rmse);
    double calc_reg();
private:
    Matrix const * const Va;
    Model const *model;
    bool en_show_tr_rmse, en_show_obj;
    std::vector<int> nr_tr_usrs, nr_tr_isrs;
};

Monitor::Monitor(TrainMatrix const &Tr, Matrix const * const Va,
                 Model const * const model, bool const en_show_tr_rmse,
                 bool const en_show_obj)
        : Va(Va), model(model), en_show_tr_rmse(en_show_tr_rmse),
          en_show_obj(en_show_obj), nr_tr_usrs(0), nr_tr_isrs(0)
{
    if(!en_show_obj)
        return;
    nr_tr_usrs.assign(Tr.nr_us, 0);
    nr_tr_isrs.assign(Tr.nr_is, 0);
    for (int ux = 0; ux < Tr.nr_gubs; ux++)
        for (int ix = 0; ix < Tr.nr_gibs; ix++)
        {
            Matrix const * const M = &Tr.GM[ux*Tr.nr_gibs+ix];
            for(long rx = 0; rx < M->nr_rs; rx++)
            {
                nr_tr_usrs[M->R[rx].uid]++;
                nr_tr_isrs[M->R[rx].iid]++;
            }
        }
}

void Monitor::print_header()
{
    char output[1024];
    sprintf(output, "%4s", "iter");
    sprintf(output+strlen(output), " %10s", "time");
    if(en_show_tr_rmse)
        sprintf(output+strlen(output), " %10s", "tr_rmse");
    if(Va != nullptr)
        sprintf(output+strlen(output), " %10s", "va_rmse");
    if(en_show_obj)
        sprintf(output+strlen(output), " %13s %13s %13s", "loss", "reg", "obj");
    printf("%s\n", output);
}

void Monitor::print(int const iter, float const time, double const loss,
                    float const tr_rmse)
{
    char output[1024];
    sprintf(output, "%-4d %10.2f", iter, time);
    if(en_show_tr_rmse)
        sprintf(output+strlen(output), " %10.3f", tr_rmse);
    if(Va != nullptr)
        sprintf(output+strlen(output), " %10.3f", calc_rmse(*model, *Va));
    if(en_show_obj)
    {
        double const reg = calc_reg();
        sprintf(output+strlen(output), " %13.3e %13.3e %13.3e", loss, reg,
                loss+reg);
    }
    printf("%s\n", output); 
    fflush(stdout);
}

double Monitor::calc_reg()
{
    int const dim_aligned = get_aligned_dim(model->param.dim);
    double reg = 0;

    {
        float * const P = model->P;
        double reg_p = 0;
        for(int ux = 0; ux < model->nr_us; ux++)
            reg_p += nr_tr_usrs[ux] * std::inner_product(
                                          P+ux*dim_aligned,
                                          P+ux*dim_aligned+model->param.dim,
                                          P+ux*dim_aligned, 
                                          0.0);
        reg += reg_p*model->param.lp;
    }

    {
        float * const Q = model->Q;
        double reg_q = 0;
        for(int ix = 0; ix < model->nr_is; ix++)
            reg_q += nr_tr_isrs[ix] *
                     std::inner_product(Q+ix*dim_aligned,
                                        Q+ix*dim_aligned+model->param.dim,
                                        Q+ix*dim_aligned, 0.0);
        reg += reg_q*model->param.lq;
    }

    if(model->param.lub >= 0)
    {
        double reg_ub = 0;
        for(int ux = 0; ux < model->nr_us; ux++)
            reg_ub += nr_tr_usrs[ux] * model->UB[ux] * model->UB[ux];
        reg += reg_ub*model->param.lub;
    }

    if(model->param.lib >= 0)
    {
        double reg_ib = 0;
        for(int ix = 0; ix < model->nr_is; ix++)
            reg_ib += nr_tr_isrs[ix] * model->UB[ix] * model->UB[ix];
        reg += reg_ib*model->param.lib;
    }

    return reg;
}

std::shared_ptr<TrainMatrix> read_as_train_matrix(
        TrainOption const &option,
        std::vector<int> const &user_map,
        std::vector<int> const &item_map)
{
    FILE *f = fopen(option.tr_path.c_str(), "rb");
    if (!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.\n", option.tr_path.c_str());
        return std::shared_ptr<TrainMatrix>(nullptr);
    }
    std::shared_ptr<Matrix> Tr_meta = read_matrix_meta(f);

    std::shared_ptr<TrainMatrix> Tr(new TrainMatrix);
    Tr->nr_us = Tr_meta->nr_us;
    Tr->nr_is = Tr_meta->nr_is;
    Tr->nr_rs = Tr_meta->nr_rs;
    Tr->avg = Tr_meta->avg;
    Tr->nr_gubs = option.nr_gubs;
    Tr->nr_gibs = option.nr_gibs;
    Tr->GM.resize(option.nr_gubs*option.nr_gibs);

    std::vector<std::vector<Node>> buffers(Tr->nr_us);
    for(long rx = 0; rx < Tr->nr_rs; rx++)
    {
        Node node_in;
        fread(&node_in, sizeof(Node), 1, f);
        node_in.uid = user_map[node_in.uid];
        node_in.iid = item_map[node_in.iid];
        buffers[node_in.uid].push_back(node_in);
    }

    // It seems that sorting makes no significant difference in performance.
    /*
    std::mutex mtx;
    std::queue<int> tasks;
    for(int ux = 0; ux < Tr->nr_us; ux++)
        tasks.push(ux);

    auto sort_node = [] (Node lhs, Node rhs)
    {
        if(lhs.uid!=rhs.uid)
            return lhs.uid < rhs.uid;
        else
            return lhs.iid < rhs.iid;
    };

    auto sort_worker = [&] ()
    {
        while(true)
        {
            int ux = 0;
            {
                std::lock_guard<std::mutex> lock(mtx);
                if(tasks.empty())
                    break;
                ux = tasks.front();
                tasks.pop();
            }
            std::sort(buffers[ux].begin(), buffers[ux].end(), sort_node);
        }
    };

    std::vector<std::thread> threads;
    for(int tx = 0; tx < option.nr_threads; tx++)
        threads.push_back(std::thread(sort_worker));
    for(auto &thread : threads)
        thread.join();
    */

    int const seg_u = (int)ceil((double)Tr_meta->nr_us/option.nr_gubs);
    int const seg_i = (int)ceil((double)Tr_meta->nr_is/option.nr_gibs);
    for(auto &buffer : buffers)
    {
        for(auto &node : buffer)
        {
            int const bid = (node.uid/seg_u)*option.nr_gibs+node.iid/seg_i;
            Tr->GM[bid].R.push_back(node);
        }
        buffer.clear();
    }

    for(auto &M : Tr->GM)
    {
        M.nr_rs = (long)(M.R.size());
        M.R.shrink_to_fit();
    }

    return Tr;
}

class Scheduler
{
public:
    Scheduler(int const nr_gubs, int const nr_gibs, int const nr_threads);
    int get_job();
    void put_job(int const jid, double const loss);
    double get_loss();
    void wait_for_jobs_done(int const nr_jobs);
    void pause();
    void resume();
    void terminate();
    bool is_terminated();
private:
    void pause_if_needed();

    int const nr_gubs, nr_gibs, nr_blocks, nr_threads;
    int total_jobs, nr_paused_thrs;
    bool paused, terminated;
    std::vector<int> counts, order_u, order_i, blocked_u, blocked_i;
    std::vector<double> losses;
    std::mutex mtx;
    std::condition_variable cv;
};

Scheduler::Scheduler(int const nr_gubs, int const nr_gibs, int const nr_threads)
        : nr_gubs(nr_gubs), nr_gibs(nr_gibs), nr_blocks(nr_gubs*nr_gibs),
          nr_threads(nr_threads), total_jobs(0), nr_paused_thrs(0), 
          paused(false), terminated(false), counts(nr_blocks, 0), 
          order_u(nr_gubs, 0), order_i(nr_gibs, 0), blocked_u(nr_gubs, 0), 
          blocked_i(nr_gibs, 0), losses(nr_blocks, 0)
{
    for(int ux = 0; ux < nr_gubs; ux++)
        order_u[ux] = ux;
    for(int ix = 0; ix < nr_gibs; ix++)
        order_i[ix] = ix;
}

int Scheduler::get_job()
{
    int best_jid = -1;
    int scanned_times = std::numeric_limits<int>::max();
    std::lock_guard<std::mutex> lock(mtx);
    for(int ux = 0; ux < nr_gubs; ux++)
    {
        if(blocked_u[ux] == 1)
            continue;
        for(int ix = 0; ix < nr_gibs; ix++)
        {
            if(blocked_i[ix] == 1)
                continue;
            int const jid = ux*nr_gibs+ix;
            if(counts[jid] < scanned_times)
            {
                best_jid = jid;
                scanned_times = counts[best_jid];
            }
        }
    }
    blocked_u[best_jid/nr_gibs] = 1;
    blocked_i[best_jid%nr_gibs] = 1;
    counts[best_jid]++;
    return best_jid;
}

void Scheduler::put_job(int const jid, double const loss)
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        blocked_u[jid/nr_gibs] = 0;
        blocked_i[jid%nr_gibs] = 0;
        losses[jid] = loss;
        total_jobs++;
        cv.notify_all();
    }
    pause_if_needed();
}

double Scheduler::get_loss()
{
    std::lock_guard<std::mutex> lock(mtx);
    return std::accumulate(losses.begin(), losses.end(), 0.0);
}

void Scheduler::wait_for_jobs_done(int const nr_jobs)
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]{return total_jobs >= nr_jobs;});
}

void Scheduler::pause()
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        paused = true;
    }
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]{return nr_paused_thrs == nr_threads;});
}

void Scheduler::pause_if_needed()
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        if(!paused)
            return;
        nr_paused_thrs++;
        cv.notify_all();
    }

    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{return !paused;});
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        nr_paused_thrs--;
    }
}

void Scheduler::resume()
{
    {
        std::lock_guard<std::mutex> lock(mtx);
        std::random_shuffle(order_u.begin(), order_u.end());
        std::random_shuffle(order_i.begin(), order_i.end());
        paused = false;
    }
    cv.notify_all();
}

void Scheduler::terminate()
{
    std::lock_guard<std::mutex> lock(mtx);
    terminated = true;
}

bool Scheduler::is_terminated()
{
    std::lock_guard<std::mutex> lock(mtx);
    return terminated;
}

void sgd(TrainMatrix const * const Tr, Model * const model,
         Scheduler * const scheduler)
{
    int const dim_aligned = get_aligned_dim(model->param.dim);
    bool const en_ub = (model->param.lub >= 0);
    bool const en_ib = (model->param.lib >= 0);
    float const glp = 1 - model->param.gamma*(model->param.lp);
    float const glq = 1 - model->param.gamma*(model->param.lq);
    float const glub = 1 - model->param.gamma*(model->param.lub);
    float const glib = 1 - model->param.gamma*(model->param.lib);
    float const gamma = model->param.gamma;
    float const avg = model->avg;

    float *const P = model->P;
    float *const Q = model->Q;
    float *const UB = model->UB.data();
    float *const IB = model->IB.data();

#if defined NOSSE
    int const dim = model->param.dim;
    while(true)
    {
        int const jid = scheduler->get_job();
        double loss = 0;
        Matrix const * const M = &Tr->GM[jid];
        float *p, *q, *ub, *ib;
        for(auto r = M->R.begin(); r != M->R.end(); r++)
        {
            p = P + r->uid*dim_aligned;
            q = Q + r->iid*dim_aligned;
            float ge = std::inner_product(p, p+dim_aligned, q, 0.0) + avg;
            if (en_ub)
            {
                ub = UB + r->uid;
                ge += (*ub);
            }
            if (en_ib)
            {
                ib = IB + r->iid;
                ge += (*ib);
            }
            ge = r->rate - ge;
            loss += ge*ge;
            ge *= gamma;
            for(int dx = 0; dx < dim; dx++)
            {
                float const tmp = p[dx];
                p[dx] = ge*q[dx] + glp*p[dx];
                q[dx] = ge*tmp + glq*q[dx];
            }
            if(en_ub)
                *ub = glub*(*ub) + ge;
            if(en_ib)
                *ib = glib*(*ib) + ge;
        }
        scheduler->put_job(jid, loss);
        if(scheduler->is_terminated())
            break;
    }
#elif defined USEAVX
    __m256 const XMMglp = _mm256_broadcast_ss(&glp);
    __m256 const XMMglq = _mm256_broadcast_ss(&glq);
    __m256 const XMMglub = _mm256_broadcast_ss(&glub);
    __m256 const XMMglib = _mm256_broadcast_ss(&glib);
    __m256 const XMMg = _mm256_broadcast_ss(&gamma);
    __m256 const XMMavg = _mm256_broadcast_ss(&avg);

    while(true)
    {
        int const jid = scheduler->get_job();
        Matrix const * const M = &Tr->GM[jid];
        long const nr_rs = M->nr_rs;
        Node const *r;
        float *p, *q, *ub, *ib;
        __m128d XMMloss = _mm_setzero_pd();
        __m256 XMMr;
        __m256 XMMge = _mm256_setzero_ps();;
        if(nr_rs > 0)
        {
            r = M->R.data();
            p = P + r->uid*dim_aligned;
            q = Q + r->iid*dim_aligned;

            XMMr = _mm256_broadcast_ss(&r->rate);

            for(int dx = 0; dx < dim_aligned; dx += 8)
                XMMge = _mm256_add_ps(XMMge, _mm256_mul_ps(
                                                 _mm256_load_ps(p+dx),
                                                 _mm256_load_ps(q+dx)));
        }

        for(long mx = 0; mx < nr_rs; mx++)
        {
            XMMge = _mm256_add_ps(XMMge,
                                  _mm256_permute2f128_ps(XMMge, XMMge, 1));
            XMMge = _mm256_hadd_ps(XMMge, XMMge);
            XMMge = _mm256_hadd_ps(XMMge, XMMge);

            __m256 XMMub;
            if(en_ub)
            {
                ub = UB + r->uid;
                XMMub = _mm256_broadcast_ss(ub);
                XMMge = _mm256_add_ps(XMMge, XMMub);
            }

            __m256 XMMib;
            if(en_ib)
            {
                ib = IB + r->iid;
                XMMib = _mm256_broadcast_ss(ib);
                XMMge = _mm256_add_ps(XMMge, XMMib);
            }

            XMMge = _mm256_sub_ps(XMMr, _mm256_add_ps(XMMge, XMMavg));
            XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(_mm256_castps256_ps128(
                                 _mm256_mul_ps(XMMge, XMMge))));
            XMMge = _mm256_mul_ps(XMMge, XMMg);

            if(en_ub)
                _mm_store_ss(ub, _mm256_castps256_ps128(_mm256_add_ps(XMMge,
                             _mm256_mul_ps(XMMglub, XMMub))));
            if(en_ib)
                _mm_store_ss(ib, _mm256_castps256_ps128(_mm256_add_ps(XMMge,
                             _mm256_mul_ps(XMMglib, XMMib))));
            Node const *r_next = r+1;
            if(mx < nr_rs-1 && r->uid == r_next->uid)
            {
                float *q_next = Q + r_next->iid*dim_aligned;
                __m256 XMMge_next = _mm256_setzero_ps();
                for(int dx = 0; dx < dim_aligned; dx += 8)
                {
                    __m256 XMMp = _mm256_load_ps(p+dx);
                    __m256 XMMq = _mm256_load_ps(q+dx);
                    __m256 XMMt = XMMp;
                    XMMp = _mm256_add_ps(_mm256_mul_ps(XMMge, XMMq),
                                         _mm256_mul_ps(XMMglp, XMMp));
                    XMMq = _mm256_add_ps(_mm256_mul_ps(XMMge, XMMt),
                                         _mm256_mul_ps(XMMglq, XMMq));
                    _mm256_store_ps(p+dx, XMMp);
                    _mm256_store_ps(q+dx, XMMq);

                    XMMq = _mm256_load_ps(q_next+dx);
                    XMMge_next = _mm256_add_ps(XMMge_next,
                                               _mm256_mul_ps(XMMp, XMMq));
                }
                q = q_next;
                XMMge = XMMge_next;
            }
            else
            {
                for(int dx = 0; dx < dim_aligned; dx += 8)
                {
                    __m256 XMMp = _mm256_load_ps(p+dx);
                    __m256 XMMq = _mm256_load_ps(q+dx);
                    __m256 XMMt = XMMp;
                    XMMp = _mm256_add_ps(_mm256_mul_ps(XMMge, XMMq),
                                         _mm256_mul_ps(XMMglp, XMMp));
                    XMMq = _mm256_add_ps(_mm256_mul_ps(XMMge, XMMt),
                                         _mm256_mul_ps(XMMglq, XMMq));
                    _mm256_store_ps(p+dx, XMMp);
                    _mm256_store_ps(q+dx, XMMq);
                }
                if(mx == nr_rs-1)
                    break;
                p = P + r_next->uid*dim_aligned;
                q = Q + r_next->iid*dim_aligned;
                XMMge = _mm256_setzero_ps();
                for(int dx = 0; dx < dim_aligned; dx += 8)
                    XMMge = _mm256_add_ps(XMMge, _mm256_mul_ps(
                                                     _mm256_load_ps(p+dx),
                                                     _mm256_load_ps(q+dx)));
            }
            r = r_next;
            XMMr = _mm256_broadcast_ss(&r->rate);
        }
        double loss = 0;
        _mm_store_sd(&loss, XMMloss);
        scheduler->put_job(jid, loss);
        if(scheduler->is_terminated())
            break;
    }
#else
    __m128 const XMMglp = _mm_load1_ps(&glp);
    __m128 const XMMglq = _mm_load1_ps(&glq);
    __m128 const XMMglub = _mm_load1_ps(&glub);
    __m128 const XMMglib = _mm_load1_ps(&glib);
    __m128 const XMMg = _mm_load1_ps(&gamma);
    __m128 const XMMavg = _mm_load1_ps(&avg);
    while(true)
    {
        int const jid = scheduler->get_job();
        Node const *r = Tr->GM[jid].R.data();
        __m128d XMMloss = _mm_setzero_pd();
        float *p, *q, *ub, *ib;
        for(long mx = 0; mx < Tr->GM[jid].nr_rs; mx++, r++)
        {
            __m128 const XMMr = _mm_load1_ps(&r->rate);
            __m128 XMMge = _mm_setzero_ps();
            p = P + r->uid*dim_aligned;
            q = Q + r->iid*dim_aligned;
            for(int dx = 0; dx < dim_aligned; dx += 4)
                XMMge = _mm_add_ps(XMMge, _mm_mul_ps(_mm_load_ps(p+dx),
                                                     _mm_load_ps(q+dx)));
            XMMge = _mm_hadd_ps(XMMge, XMMge);
            XMMge = _mm_hadd_ps(XMMge, XMMge);
            __m128 XMMub;
            if(en_ub)
            {
                ub = UB + r->uid;
                XMMub = _mm_load1_ps(ub);
                XMMge = _mm_add_ps(XMMge, XMMub);
            }
            __m128 XMMib;
            if(en_ib)
            {
                ib = IB + r->iid;
                XMMib = _mm_load1_ps(ib);
                XMMge = _mm_add_ps(XMMge, XMMib);
            }
            XMMge = _mm_sub_ps(XMMr, _mm_add_ps(XMMge, XMMavg));
            XMMloss = _mm_add_pd(XMMloss, _mm_cvtps_pd(_mm_mul_ps(XMMge,
                                                                  XMMge)));
            XMMge = _mm_mul_ps(XMMge, XMMg);
            for(int dx = 0; dx < dim_aligned; dx += 4)
            {
                __m128 XMMp = _mm_load_ps(p+dx);
                __m128 XMMq = _mm_load_ps(q+dx);
                __m128 XMMt = XMMp;
                XMMp = _mm_add_ps(_mm_mul_ps(XMMge, XMMq),
                                  _mm_mul_ps(XMMglp, XMMp));
                XMMq = _mm_add_ps(_mm_mul_ps(XMMge, XMMt),
                                  _mm_mul_ps(XMMglq, XMMq));
                _mm_store_ps(p+dx, XMMp);
                _mm_store_ps(q+dx, XMMq);
            }
            if(en_ub)
                _mm_store_ss(ub, _mm_add_ps(XMMge, _mm_mul_ps(XMMglub, XMMub)));
            if(en_ib)
                _mm_store_ss(ib, _mm_add_ps(XMMge, _mm_mul_ps(XMMglib, XMMib)));
        }
        double loss;
        _mm_store_sd(&loss, XMMloss);
        scheduler->put_job(jid, loss);
        if(scheduler->is_terminated())
            break;
    }
#endif
}

Model fpsgd(TrainMatrix const &Tr, Matrix const &Va, TrainOption const &option)
{
    Timer timer;
    timer.reset("Initializing model...");
    Model model = generate_initial_model(option.param, Tr.nr_us, Tr.nr_is,
                                         option.en_avg? Tr.avg : 0);
    timer.toc("done.");

    Monitor monitor(Tr, &Va, &model, option.en_show_tr_rmse,
                    option.en_show_obj);

    Scheduler scheduler(option.nr_gubs, option.nr_gibs, option.nr_threads);
    std::vector<std::thread> threads;
    for(int tx = 0; tx < option.nr_threads; tx++)
        threads.push_back(std::thread(sgd, &Tr, &model, &scheduler));
    monitor.print_header();

    timer.reset();
    for(int iter = 1; iter <= option.nr_iter; iter++)
    {
        scheduler.wait_for_jobs_done(iter*option.nr_gubs*option.nr_gibs);
        scheduler.pause();
        float const iter_time = timer.toc();
        double const loss = scheduler.get_loss();
        monitor.print(iter, iter_time, loss, sqrt(loss/Tr.nr_rs));
        timer.tic();
        scheduler.resume();
    }

    scheduler.terminate();
    for(auto thread = threads.begin(); thread != threads.end(); thread++)
        thread->join();

    return model;
}

void inversely_shuffle_model(Model &model, std::vector<int> const &user_map,
                             std::vector<int> const &item_map)
{
    auto gen_inv_map = [] (std::vector<int> const &map)
    {
        std::vector<int> inv_map(map.size());
        for(int i = 0; i < (int)map.size(); i++)
          inv_map[map[i]] = i;
        return inv_map;
    };

    auto shuffle = [] (float * const vec, std::vector<int> const &map,
                       int const count, int const dim)
    {
        std::vector<float> vec_(count*dim);
        std::copy(vec, vec+count*dim, vec_.data());
        for(int idx = 0; idx < count; idx++)
            std::copy(vec_.data()+idx*dim,
                      vec_.data()+idx*dim+dim,
                      vec+map[idx]*dim);
    };

    std::vector<int> const inv_user_map = gen_inv_map(user_map);
    std::vector<int> const inv_item_map = gen_inv_map(item_map);

    int const dim_aligned = get_aligned_dim(model.param.dim);

    shuffle(model.P, inv_user_map, model.nr_us, dim_aligned);
    shuffle(model.Q, inv_item_map, model.nr_is, dim_aligned);
    if(model.param.lub >= 0)
        shuffle(model.UB.data(), inv_user_map, model.nr_us, 1);
    if(model.param.lib >= 0)
        shuffle(model.IB.data(), inv_item_map, model.nr_is, 1);
}

} //namespace

int train(int const argc, char const * const * const argv)
{
#if defined NOSSE
    printf("Warning: SSE is disabled.\n");
#elif defined USEAVX
    printf("Warning: AVX is enabled.\n");
#endif

    std::shared_ptr<const TrainOption> const 
        option = parse_train_option(argc, argv);
    if (!option)
        return EXIT_FAILURE;

    Timer timer;

    std::shared_ptr<const Matrix> const
        Tr_meta = read_matrix_meta(option->tr_path);
    if(!Tr_meta)
        return EXIT_FAILURE;

    std::vector<int> const user_map = gen_map(Tr_meta->nr_us,
                                              option->en_rand_shuffle);
    std::vector<int> const item_map = gen_map(Tr_meta->nr_is,
                                              option->en_rand_shuffle);

    timer.reset("Reading training data...");
    std::shared_ptr<const TrainMatrix> const
        Tr = read_as_train_matrix(*option, user_map, item_map);
    if(!Tr)
        return EXIT_FAILURE;
    timer.toc("done.");

    std::shared_ptr<Matrix> Va;
    if(!option->va_path.empty())
    {
        timer.reset("Reading validation data...");
        Va = read_matrix(option->va_path);
        if(!Va)
            return EXIT_FAILURE;
        timer.toc("done.");
        if(Va->nr_us > Tr_meta->nr_us || Va->nr_is > Tr_meta->nr_is)
        {
            fprintf(stderr, "Error: Validation set out of range.\n");
            return EXIT_FAILURE;
        }
        for (auto &r : Va->R)
        {
            r.uid = user_map[r.uid];
            r.iid = item_map[r.iid];
        }
    }

    Model model = fpsgd(*Tr, *Va, *option);

    if(option->en_rand_shuffle)
        inversely_shuffle_model(model, user_map, item_map);

    timer.reset("Writing model...");
    if(!write_model(model, option->model_path))
        return EXIT_FAILURE;
    timer.toc("done.");

    return EXIT_SUCCESS;
}
