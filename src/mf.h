#pragma GCC diagnostic ignored "-Wunused-result"
#include <memory>
#include <chrono>
#include <vector>

#define flag { printf("LINE: %d\n", __LINE__); fflush(stdout); }

class Timer
{
public:
    Timer();
    void reset();
    void reset(std::string const &msg);
    void tic();
    void tic(std::string const &msg);
    float toc();
    float toc(std::string const &msg);
private:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::milliseconds duration;
};

struct Node
{
    Node() : uid(0), iid(0), rate(0) {}
    int uid, iid;
    float rate;
};

struct Matrix
{
    Matrix() : nr_users(0), nr_items(0), nr_ratings(0), avg(0), R(0) {}
    int nr_users, nr_items;
    long nr_ratings;
    float avg;
    std::vector<Node> R;
};

std::shared_ptr<Matrix> read_matrix_meta(FILE *f);

std::shared_ptr<Matrix> read_matrix_meta(std::string const &path);

std::shared_ptr<Matrix> read_matrix(std::string const &path);

bool write_matrix(Matrix const &M, std::string const &path);

struct Parameter
{
    Parameter() : dim(40), lp(1), lq(1), lub(-1), lib(-1), gamma(0.001) {}
    int dim;
    float lp, lq, lub, lib, gamma;
};

struct Model
{
    Model() : param(), nr_users(0), nr_items(0), avg(0), P(nullptr), Q(nullptr),
              UB(0), IB(0) {}
    Parameter param;
    int nr_users, nr_items;
    float avg;
    float *P, *Q;
    std::vector<float> UB, IB;
    ~Model();
};

std::shared_ptr<Model> read_model_meta(FILE *f);

std::shared_ptr<Model> read_model_meta(std::string const &path);

std::shared_ptr<Model> read_model(std::string const &path);

bool write_model(Model const &model, std::string const &path);

float calc_rate(Model const &model, Node const &r);

float calc_rmse(Model const &model, Matrix const &M);

int get_aligned_dim(int const dim);

int convert(int const argc, char const * const * const argv);

int train(int const argc, char const * const * const argv);

int predict(int const argc, char const * const * const argv);

int view(int const argc, char const * const * const argv);
