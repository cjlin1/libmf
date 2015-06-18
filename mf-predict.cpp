#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mf.h"

using namespace std;
using namespace mf;

struct Option
{
    Option() : eval(RMSE) {}
    string test_path, model_path, output_path;
    mf_int eval;
};

string predict_help()
{
    return string(
"usage: mf-predict [options] test_file model_file [output_file]\n"
"\n"
"options:\n"
"-e <eval>: specify the evaluation criterion (default 0)\n"
"\t 0 -- Root mean square error\n"
"\t 1 -- Logistic error\n"
"\t10 -- Mean percentile rank\n"
"\t11 -- Area under ROC curve\n");
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    mf_int i;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-e") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify evaluation criterion after -e");
            i++;
            option.eval = stoi(args[i]);
            if(option.eval != RMSE && option.eval != LOGLOSS &&
               option.eval != AUC  && option.eval != MPR)
                throw invalid_argument("unknown evaluation criterion");
        }
        else
            break;
    }
    if(i >= argc-1)
        throw invalid_argument("testing data and model file not specified");
    option.test_path = string(args[i++]);
    option.model_path = string(args[i++]);

    if(i < argc)
    {
        option.output_path = string(args[i]);
    }
    else if(i == argc)
    {
        const char *ptr = strrchr(&*option.test_path.begin(), '/');
        if(!ptr)
            ptr = option.test_path.c_str();
        else
            ++ptr;
        option.output_path = string(ptr) + ".out";
    }
    else
    {
        throw invalid_argument("invalid argument");
    }

    return option;
}

void predict(string test_path, string model_path, string output_path, mf_int eval)
{
    mf_problem prob = read_problem(test_path);

    ofstream f_out(output_path);
    if(!f_out.is_open())
        throw runtime_error("cannot open " + output_path);

    mf_model *model = mf_load_model(model_path.c_str()); // use shared_ptr?
    if(model == nullptr)
        throw runtime_error("cannot load model from " + model_path);

    for(mf_int i = 0; i < prob.nnz; i++)
    {
        mf_float r = mf_predict(model, prob.R[i].u, prob.R[i].v);
        f_out << r << endl;
    }

    switch(eval)
    {
        case RMSE:
        {
            auto rmse = calc_rmse(&prob, model);
            cout << fixed << setprecision(4) << "RMSE = " << rmse << endl;
            break;
        }
        case LOGLOSS:
        {
            auto logloss = calc_logloss(&prob, model);
            cout << fixed << setprecision(4) << "LOGLOSS = " << logloss << endl;
            break;
        }
        case AUC:
        {
            auto row_wise_auc = calc_auc(&prob, model, false);
            auto col_wise_auc = calc_auc(&prob, model, true);
            cout << fixed << setprecision(4) <<  "Row-wise AUC = " << row_wise_auc << endl;
            cout << fixed << setprecision(4) <<  "Colmn-wise AUC = " << col_wise_auc << endl;
            break;
        }
        case MPR:
        {
            auto row_wise_mpr = calc_mpr(&prob, model, false);
            auto col_wise_mpr = calc_mpr(&prob, model, true);
            cout << fixed << setprecision(4) <<  "Row-wise MPR = " << row_wise_mpr << endl;
            cout << fixed << setprecision(4) <<  "Column-wise MPR = " << col_wise_mpr << endl;
            break;
        }
        default:
        {
            throw invalid_argument("unknown evaluation criterion");
            break;
        }
    }
    mf_destroy_model(&model);
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

    try
    {
        predict(option.test_path, option.model_path, option.output_path, option.eval);
    }
    catch(runtime_error &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    return 0;
}
