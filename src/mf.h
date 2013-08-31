#pragma GCC diagnostic ignored "-Wunused-result" 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <ctime>
#include <cstring>
#include <climits>
#include <cfloat>
#include <random>
#include <numeric>
#include <algorithm>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <pmmintrin.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#define DATAVER 1
#define MODELVER 1

#define EN_SHOW_SCHED false
#define EN_SHOW_GRID false

#define flag fprintf(stderr, "LINE: %d\n", __LINE__)

enum FileType {DATA,MODEL};
void convert(int argc, char **argv);
void train(int argc, char **argv);
void predict(int argc, char **argv);
void view(int argc, char **argv);
void exit_file_error(char *path);
void exit_file_ver(float ver);
struct Clock {
    clock_t begin, end;
    void tic();
    float toc();
};
struct Node {
    int uid, iid; 
    float rate;
};
struct Matrix {
    int nr_us, nr_is;
    long nr_rs;
    float avg;
    Node *M;
    Matrix();
    Matrix(int nr_rs, int nr_us, int nr_is, float avg);
    Matrix(char *path);
    Matrix(char *path, int *map_u, int *map_i);
    void read_meta(FILE *f);
    void read(char *path);
    void write(char *path);
    void sort();
    static bool sort_uid_iid(Node lhs, Node rhs);
    ~Matrix();
};
struct Model {
	int nr_us, nr_is, dim, dim_off, nr_thrs, iter, nr_gubs, nr_gibs, *map_uf, *map_ub, *map_if, *map_ib;
    float *P, *Q, *UB, *IB, lp, lq, lub, lib, glp, glq, glub, glib, gamma, avg;
    bool en_rand_shuffle, en_avg, en_ub, en_ib;
    Model();
    Model(char *src);
    void initialize(Matrix *Tr);
    void read_meta(FILE *f);
    void read(char *path);
    void write(char *path);
    void gen_rand_map();
    void shuffle();
    void inv_shuffle();
    ~Model();
};
float calc_rate(Model *model, Node *r);
float calc_rmse(Model *model, Matrix *R);
