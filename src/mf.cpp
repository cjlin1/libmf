#include <algorithm>
#include <cassert>
#include <cstdlib>
#include "mf.h"

Timer::Timer()
{
    reset();
}

void Timer::reset()
{
    begin = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>
                   (begin-begin);
}

void Timer::reset(std::string const &msg)
{
    printf("%s", msg.c_str());
    fflush(stdout);
    reset();
}

void Timer::tic()
{
    begin = std::chrono::high_resolution_clock::now();
}

void Timer::tic(std::string const &msg)
{
    printf("%s", msg.c_str());
    fflush(stdout);
    tic();
}

float Timer::toc()
{
    duration += std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::high_resolution_clock::now()-begin);
    return (double)duration.count()/1000;
}

float Timer::toc(std::string const &msg)
{
    float duration_one = toc();
    printf("%s  %.2f\n", msg.c_str(), duration_one);
    fflush(stdout);
    return duration_one;
}

std::shared_ptr<Matrix> read_matrix_meta(FILE *f)
{
    std::shared_ptr<Matrix> M(new Matrix);
    fread(&M->nr_users, sizeof(int), 1, f);
    fread(&M->nr_items, sizeof(int), 1, f);
    fread(&M->nr_ratings, sizeof(long), 1, f);
    fread(&M->avg, sizeof(float), 1, f);
    return M;
}

std::shared_ptr<Matrix> read_matrix_meta(std::string const &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if(!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.\n", path.c_str());
        return std::shared_ptr<Matrix>(nullptr);
    }
    std::shared_ptr<Matrix> M = read_matrix_meta(f);
    fclose(f);
    return M;
}

std::shared_ptr<Matrix> read_matrix(std::string const &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if(!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.\n", path.c_str());
        return std::shared_ptr<Matrix>(nullptr);
    }
    std::shared_ptr<Matrix> M = read_matrix_meta(f);
    M->R.resize(M->nr_ratings);
    fread(M->R.data(), sizeof(Node), M->nr_ratings, f);
    fclose(f);
    return M;
}

bool write_matrix(Matrix const &M, std::string const &path)
{
    FILE *f = fopen(path.c_str(), "wb");
    if(!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.\n", path.c_str());
        return false;
    }
    fwrite(&M.nr_users, sizeof(int), 1, f);
    fwrite(&M.nr_items, sizeof(int), 1, f);
    fwrite(&M.nr_ratings, sizeof(long), 1, f);
    fwrite(&M.avg, sizeof(float), 1, f);
    fwrite(M.R.data(), sizeof(Node), M.nr_ratings, f);
    fclose(f);
    return true;
}

Model::~Model()
{
    if(P != nullptr)
        free(P);
    if(Q != nullptr)
        free(Q);
}

std::shared_ptr<Model> read_model_meta(FILE *f)
{
    std::shared_ptr<Model> model(new Model);
    fread(&model->param, sizeof(Parameter), 1, f);
    fread(&model->nr_users, sizeof(int), 1, f);
    fread(&model->nr_items, sizeof(int), 1, f);
    fread(&model->avg, sizeof(float), 1, f);
    return model;
}

std::shared_ptr<Model> read_model_meta(std::string const &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if(!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.\n", path.c_str());
        return std::shared_ptr<Model>(nullptr);
    }
    std::shared_ptr<Model> model = read_model_meta(f);
    fclose(f);
    return model;
}

std::shared_ptr<Model> read_model(std::string const &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if(!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.\n", path.c_str());
        return std::shared_ptr<Model>(nullptr);
    }

    std::shared_ptr<Model> model = read_model_meta(f);
    int const dim_aligned = get_aligned_dim(model->param.dim);

    posix_memalign((void**)&model->P, 32,
                   model->nr_users*dim_aligned*sizeof(float));
    fread(model->P, sizeof(float), model->nr_users*dim_aligned, f);

    posix_memalign((void**)&model->Q, 32,
                   model->nr_items*dim_aligned*sizeof(float));
    fread(model->Q, sizeof(float), model->nr_items*dim_aligned, f);

    if(model->param.lub >= 0)
    {
        model->UB.resize(model->nr_users);
        fread(model->UB.data(), sizeof(float), model->nr_users, f);
    }

    if(model->param.lib >= 0)
    {
        model->IB.resize(model->nr_items);
        fread(model->IB.data(), sizeof(float), model->nr_items, f);
    }

    fclose(f);
    return model;
}

bool write_model(Model const &model, std::string const &path)
{
    FILE *f = fopen(path.c_str(), "wb");
    if(!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.", path.c_str());
        return false;
    }
    int const dim_aligned = get_aligned_dim(model.param.dim);
    fwrite(&model.param, sizeof(Parameter), 1, f);
    fwrite(&model.nr_users, sizeof(int), 1, f);
    fwrite(&model.nr_items, sizeof(int), 1, f);
    fwrite(&model.avg, sizeof(float), 1, f);
    fwrite(model.P, sizeof(float), model.nr_users*dim_aligned, f);
    fwrite(model.Q, sizeof(float), model.nr_items*dim_aligned, f);
    if(model.param.lub >= 0)
        fwrite(model.UB.data(), sizeof(float), model.nr_users, f);
    if(model.param.lib >= 0)
        fwrite(model.IB.data(), sizeof(float), model.nr_items, f);
    fclose(f);
    return true;
}

float calc_rate(Model const &model, Node const &r)
{
    int const dim_aligned = get_aligned_dim(model.param.dim);
    float rate = std::inner_product(
                     model.P+r.uid*dim_aligned,
                     model.P+r.uid*dim_aligned + model.param.dim,
                     model.Q+r.iid*dim_aligned,
                     0.0);
    rate += model.avg;
    if(model.param.lub >= 0)
        rate += model.UB[r.uid];
    if(model.param.lib >= 0)
        rate += model.IB[r.iid];
    return rate;
}

float calc_rmse(Model const &model, Matrix const &M)
{
    double loss = 0;
    for(auto r = M.R.begin(); r != M.R.end(); r++)
    {
        float const e = r->rate - calc_rate(model, *r);
        loss += e*e;
    }
    return sqrt(loss/M.nr_ratings);
}

int get_aligned_dim(int const dim)
{
#if defined NOSSE
  return dim;
#elif defined USEAVX
  return ceil(float(dim)/8)*8;
#else
  return ceil(float(dim)/4)*4;
#endif
}
