#include <cstring>
#include <cstdlib>
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
"-l1 <lambda>,<lambda>: set L1-regularization parameters for P and Q (default 0)\n"
"  P and Q share the same lambda if only one lambda is spceified.\n"
"-l2 <lambda>,<lambda>: set L2-regularization parameters of P and Q (default 0.1)\n"
"  P and Q share the same lambda if only one lambda is spceified.\n"
"-x <solver>: specify the type of solver (default 0)\n"
"  for numerical matrix factorization\n"
"\t 0 -- L2-loss\n"
"\t 1 -- L1-loss\n"
"  for binary matrix factorization\n"
"\t 5 -- logistic loss\n"
"\t 6 -- square hinge loss\n"
"\t 7 -- hinge loss\n"
"  for one-class matrix factorization\n"
"\t10 -- row-oriented bayesian personalized ranking\n"
"\t11 -- column-oriented bayesian personalized ranking\n"
"-k <dimensions>: set number of dimensions (default 8)\n"
"-t <iter>: set number of iterations (default 20)\n"
"-r <eta>: set learning rate (default 0.1)\n"
"-s <threads>: set number of threads (default 12)\n"
"-p <path>: set path to the validation set\n"
"-v <fold>: set number of folds for cross validation\n"
"-n <blocks>: set number of blocks in disk-level trainning\n"
"--quiet: quiet mode (no outputs)\n"
"--nmf: perform non-negative matrix factorization\n"
"--disk: train on disk\n");
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
        if(args[i].compare("-l1") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify lambda after -l1");
            i++;

            char *pch = strtok(argv[i], ",");
            if(strtod(pch, NULL) < 0)
                throw invalid_argument("regularization coefficient should be non-negative");
            option.param.lambda_p1 = (mf_float)strtod(pch, NULL);
            option.param.lambda_q1 = (mf_float)strtod(pch, NULL);
            pch = strtok(NULL, ",");
            if(pch != NULL)
            {
                if(strtod(pch, NULL) < 0)
                    throw invalid_argument("regularization coefficient should be non-negative");
                option.param.lambda_q1 = (mf_float)strtod(pch, NULL);
            }
        }
        else if(args[i].compare("-l2") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify lambda after -l2");
            i++;
            
            char *pch = strtok(argv[i], ",");
            if(strtod(pch, NULL) < 0)
                throw invalid_argument("regularization coefficient should be non-negative");
            option.param.lambda_p2 = strtod(pch, NULL);
            option.param.lambda_q2 = strtod(pch, NULL);
            pch = strtok(NULL, ",");
            if(pch != NULL)
            {
                if(strtod(pch, NULL) < 0)
                throw invalid_argument("regularization parameter should not be smaller than zero");
                option.param.lambda_q2 = strtod(pch, NULL);
            }
        }
        else if(args[i].compare("-k") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of factors after -k");
            i++;
            option.param.k = atoi(argv[i]);
            if(option.param.k <= 0)
                throw invalid_argument("number of factors should be greater than zero");
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of iterations after -t");
            i++;
            option.param.nr_iters = atoi(argv[i]);
            if(option.param.nr_iters <= 0)
                throw invalid_argument("number of iterations should be greater than zero");
        }
        else if(args[i].compare("-r") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify eta after -r");
            i++;
            option.param.eta = atof(argv[i]);
            if(option.param.eta <= 0)
                throw invalid_argument("learning rate should be greater than zero");
        }
        else if(args[i].compare("-s") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify number of threads after -s");
            i++;
            option.param.nr_threads = atoi(argv[i]);
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
            option.nr_folds = atoi(argv[i]);
            if(option.nr_folds <= 1)
                throw invalid_argument("number of folds should be larger than 1");
            option.do_cv = true;
        }
        else if(args[i].compare("-x") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify the type of solver after -x");
            i++;
            option.param.solver = atoi(argv[i]);
            if(option.param.solver != P_L2_MFR &&
               option.param.solver != P_L1_MFR &&
               option.param.solver != P_LR_MFC &&
               option.param.solver != P_L2_MFC &&
               option.param.solver != P_L1_MFC &&
               option.param.solver != P_ROW_BPR_MFOC &&
               option.param.solver != P_COL_BPR_MFOC)
                throw invalid_argument("unknown solver type");
        }
        else if(args[i].compare("-n") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of blocks after -n");
            i++;
            option.param.nr_blocks = atoi(argv[i]);
            if(option.param.nr_blocks <= 1)
                throw invalid_argument("number of blocks should be larger than 1");
            option.param.nr_bins = ceil(sqrt(option.param.nr_blocks));
        }
        else if(args[i].compare("--nmf") == 0)
        {
            option.param.do_nmf = true;
        }
        else if(args[i].compare("--quiet") == 0)
        {
            option.param.quiet = true;
        }
        else if(args[i].compare("--disk") == 0)
        {
            option.param.disk = true;
        }
        else
        {
            break;
        }
    }

    if(option.nr_folds > 1 && !option.va_path.empty())
        throw invalid_argument("cannot specify -p and -v simultaneously");

    if(option.nr_folds > 1 && option.param.disk == true)
        throw invalid_argument("cannot specify -v and --disk simultaneously");

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
    if(option.param.disk != true)
    {
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
    }

    if(option.do_cv)
    {
        mf_cross_validation(&tr, option.nr_folds, option.param);
    }
    else
    {
        mf_model *model;
        if(option.param.disk != true)
            model = mf_train_with_validation(&tr, &va, option.param);
        else
            model = mf_train_with_validation_on_disk(option.tr_path.c_str(), option.va_path.c_str(), option.param);

        // use the following function if you do not have a validation set

        // mf_model model = 
        //     mf_train_with_validation(&tr, option.fpsg_command.c_str());

        mf_int status = mf_save_model(model, option.model_path.c_str());

        if(status != 0)
        {
            cout << "cannot save model to " << option.model_path << endl;

            if(option.param.disk != true)
            {
                delete[] tr.R;
                delete[] va.R;
            }
            mf_destroy_model(&model);

            return 1;
        }

        mf_destroy_model(&model);
    }

    if(option.param.disk != true)
    {
        delete[] tr.R;
        delete[] va.R;
    }

    return 0;
}
