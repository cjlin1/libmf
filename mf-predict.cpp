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
    string test_path, model_path, output_path;
};

string predict_help()
{
    return string(
"usage: mf-predict test_file model_file output_file\n");
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    if(argc != 4)
        throw invalid_argument("invalid argument");

    option.test_path = string(args[1]);
    option.model_path = string(args[2]);
    option.output_path = string(args[3]);

    return option;
}

void predict(string test_path, string model_path, string output_path)
{
    ifstream f_te(test_path);
    if(!f_te.is_open())
        throw runtime_error("cannot open " + test_path);

    ofstream f_out(output_path);
    if(!f_out.is_open())
        throw runtime_error("cannot open " + output_path);

    mf_model *model = mf_load_model(model_path.c_str());
    if(model == nullptr)
        throw runtime_error("cannot load model from " + model_path);

    mf_double loss = 0;
    mf_long l = 0;
    mf_node N;
    while(f_te >> N.u >> N.v >> N.r)
    {
        mf_float r = mf_predict(model, N.u, N.v);
        f_out << r << endl;
        mf_float e = N.r - r;
        loss += e*e;
        ++l;
    }

    cout << "RMSE = " << fixed << setprecision(4) << sqrt(loss/l) << endl;

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
        predict(option.test_path, option.model_path, option.output_path);
    }
    catch(runtime_error &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    return 0;
}
