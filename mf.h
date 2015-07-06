#ifndef _LIBMF_H
#define _LIBMF_H

#include <utility>

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

enum {SQ_MF=0, LR_MF=5, SQ_HINGE_MF=6, ROW_BPR=10, COL_BPR=11};
enum {RMSE=0, LOGLOSS=1, ACC=2, ROW_MPR=10, COL_MPR=11, ROW_AUC=12, COL_AUC=13};

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
    mf_float lambda_p1;
    mf_float lambda_p2;
    mf_float lambda_q1;
    mf_float lambda_q2;
    mf_float eta;
    mf_int do_nmf;
    mf_int quiet; 
    mf_int copy_data;
    mf_int solver;
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

mf_problem read_problem(std::string path);

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

mf_float mf_predict(struct mf_model const *model, mf_int u, mf_int v);

mf_double calc_rmse(mf_problem *prob, mf_model *model);

mf_double calc_logloss(mf_problem *prob, mf_model *model);

mf_double calc_accuracy(mf_problem *prob, mf_model *model);

mf_double calc_mpr(mf_problem *prob, mf_model *model, bool transpose);

mf_double calc_auc(mf_problem *prob, mf_model *model, bool transpose);

#ifdef __cplusplus
} // namespace mf

} // extern "C"
#endif

#endif // _LIBMF_H
