#ifndef _LIBMF_H
#define _LIBMF_H

#ifdef __cplusplus
extern "C" 
{

namespace mf
{
#endif

// Changing the following typedef is not allowed in this version.
typedef float mf_float;
typedef double mf_double;
typedef int mf_int;
typedef long long mf_long;

struct mf_node
{
    mf_int u;
    mf_int v;
    mf_float r;
};

struct mf_problem
{
    mf_int m;
    mf_int n;
    mf_long nnz;
    struct mf_node *R;
};

struct mf_parameter
{
    mf_int k; 
    mf_int nr_threads;
    mf_int nr_bins;
    mf_int nr_iters;
    mf_float lambda; 
    mf_float eta;
    mf_int do_nmf;
    mf_int quiet; 
    mf_int copy_data;
};

struct mf_parameter mf_get_default_param();

struct mf_model
{
    mf_int m;
    mf_int n;
    mf_int k;
    mf_float *P;
    mf_float *Q;
};

mf_int mf_save_model(struct mf_model const *model, char const *path);

struct mf_model* mf_load_model(char const *path);

void mf_destroy_model(struct mf_model **model);

struct mf_model* mf_train(
    struct mf_problem const *prob, 
    struct mf_parameter param);

struct mf_model* mf_train_with_validation(
    struct mf_problem const *tr, 
    struct mf_problem const *va, 
    struct mf_parameter param);

mf_float mf_cross_validation(
    struct mf_problem const *prob, 
    mf_int nr_folds, 
    struct mf_parameter param);

mf_float mf_predict(struct mf_model const *model, mf_int p_idx, mf_int q_idx);

#ifdef __cplusplus
} // namespace mf

} // extern "C"
#endif

#endif // _LIBMF_H
