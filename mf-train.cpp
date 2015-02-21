#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

#include "mf.h"

using namespace std;
using namespace mf;

struct Option
{
    Option() : param(mf_get_default_param()), nr_folds(1), do_cv(false) {}
    string tr_path, va_path, model_path;
    mf_parameter param;
    mf_int nr_folds;
    bool do_cv;
};

string train_help()
{
    return string(
"usage: mf-train [options] training_set_file [model_file]\n"
"\n"
"options:\n"
"-l <lambda>: set regularization parameter (default 0.1)\n"
"-k <dimensions>: set number of dimensions (default 8)\n"
"-t <iter>: set number of iterations (default 20)\n"
"-r <eta>: set learning rate (default 0.1)\n"
"-s <threads>: set number of threads (default 1)\n"
"-p <path>: set path to the validation set\n"
"-v <fold>: set number of folds for cross validation\n"
"--quiet: quiet mode (no outputs)\n"
"--nmf: perform non-negative matrix factorization\n");
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option option;

    mf_int i;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-l") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify lambda after -l");
            i++;
            option.param.lambda = stof(args[i]);
            if(option.param.lambda < 0)
                throw invalid_argument("regularization parameter should not be smaller than zero");
        }
        else if(args[i].compare("-k") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of factors after -k");
            i++;
            option.param.k = stoi(args[i]);
            if(option.param.k <= 0)
                throw invalid_argument("number of factors should be greater than zero");
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of iterations after -t");
            i++;
            option.param.nr_iters = stoi(args[i]);
            if(option.param.nr_iters <= 0)
                throw invalid_argument("number of iterations should be greater than zero");
        }
        else if(args[i].compare("-r") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify eta after -r");
            i++;
            option.param.eta = stof(args[i]);
            if(option.param.eta <= 0)
                throw invalid_argument("learning rate should be greater than zero");
        }
        else if(args[i].compare("-s") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of threads after -s");
            i++;
            option.param.nr_threads = stoi(args[i]);
            if(option.param.nr_threads <= 0)
                throw invalid_argument("number of threads should be greater than zero");
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;
            option.va_path = string(args[i]);
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of folds after -v");
            i++;
            option.nr_folds = stoi(args[i]);
            if(option.nr_folds <= 1)
                throw invalid_argument("number of folds should be larger than 1");
            option.do_cv = true;
        }
        else if(args[i].compare("--nmf") == 0)
        {
            option.param.do_nmf = true;
        }
        else if(args[i].compare("--quiet") == 0)
        {
            option.param.quiet = true;
        }
        else
        {
            break;
        }
    }

    if(option.nr_folds > 1 && !option.va_path.empty())
        throw invalid_argument("cannot specify -p and -v simultaneously");

    if(i >= argc)
        throw invalid_argument("training data not specified");

    option.tr_path = string(args[i++]);

    if(i < argc)
    {
        option.model_path = string(args[i]);
    }
    else if(i == argc)
    {
        const char *ptr = strrchr(&*option.tr_path.begin(), '/');
        if(!ptr)
            ptr = option.tr_path.c_str();
        else
            ++ptr;
        option.model_path = string(ptr) + ".model";
    }
    else
    {
        throw invalid_argument("invalid argument");
    }

    option.param.copy_data = false;

    return option;
}

mf_problem read_problem(string path)
{
    mf_problem prob;
    prob.m = 0;
    prob.n = 0;
    prob.nnz = 0;
    prob.R = nullptr;

    if(path.empty())
    {
        return prob;
    }

    ifstream f(path);
    if(!f.is_open())
        throw runtime_error("cannot open " + path);
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

    return prob;
}

int main(int argc, char **argv)
{
    Option option;
    try
    {
        option = parse_option(argc, argv);
    }
    catch(invalid_argument &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    mf_problem tr, va;
    try
    {
        tr = read_problem(option.tr_path);
        va = read_problem(option.va_path);
    }
    catch(runtime_error &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    if(option.do_cv)
    {
        mf_cross_validation(&tr, option.nr_folds, option.param);
    }
    else
    {
        mf_model *model = 
            mf_train_with_validation(&tr, &va, option.param);

        // use the following function if you do not have a validation set

        // mf_model model = 
        //     mf_train_with_validation(&tr, option.fpsg_command.c_str());

        mf_int status = mf_save_model(model, option.model_path.c_str());

        if(status != 0)
        {
            cout << "cannot save model to " << option.model_path << endl;

            delete[] tr.R;
            delete[] va.R;
            mf_destroy_model(&model);

            return 1;
        }

        mf_destroy_model(&model);
    }

    delete[] tr.R;
    delete[] va.R;

    return 0;
}
