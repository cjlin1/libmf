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
    Option() : eval(2) {}
	string test_path, model_path, output_path;
	mf_int eval;
};

string predict_help()
{
    return string(
"usage: mf-predict [options] test_file model_file [output_file]\n"
"\n"
"options:\n"
"-e <eval>: specify the evaluation function (default 2)\n"
"	0 -- auc\n"
"	1 -- mpr\n"
"	2 -- rmse\n"
"	3 -- logloss\n");
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
				throw invalid_argument("need to specify evaluation function after -d");
			i++;
			option.eval = stoi(args[i]);
			if(option.eval > 3 || option.eval < 0)
				throw invalid_argument("unknown evaluation function");
		}
		else
		{
			break;
		}
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
			++ptr
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
    /*
	ifstream f_te(test_path);
    if(!f_te.is_open())
        throw runtime_error("cannot open " + test_path);
	*/
	mf_problem prob = read_problem(test_path);

    ofstream f_out(output_path);
    if(!f_out.is_open())
        throw runtime_error("cannot open " + output_path);

    mf_model *model = mf_load_model(model_path.c_str());
    if(model == nullptr)
        throw runtime_error("cannot load model from " + model_path);

    for(mf_int i = 0; i < prob.nnz; i++)
	{
		mf_float r = mf_predict(model, prob.R[i],u, prob.R[i].v);
		f_out << r << endl;
	}
	switch(eval)
	{
		case 0:
			auto row_wise_mpr_auc = calc_mpr_auc(&prob, *model, false);
			auto col_wise_mpr_auc = calc_mpr_auc(&prob, *model, true);
			cout << fixed << setprecision(4) <<  "AUC = (" << row_wise_mpr_auc.second() << ", " << col_wise_mpr_auc.second() << ")" << endl;
			break;
		case 1:
			auto row_wise_mpr_auc = calc_mpr_auc(&prob, *model, false);
			auto col_wise_mpr_auc = calc_mpr_auc(&prob, *model, true);
			cout << fixed << setprecision(4) <<  "AUC = (" << row_wise_mpr_auc.first() << ", " << col_wise_mpr_auc.first() << ")" << endl;
			break;
		case 2:
			auto rmse = calc_rmse(&prob, *model);
			cout << fixed << setprecision(4) << "RMSE = " << rmse << endl;
		case 3:
			auto logloss = calc_logloss(&prob, *model);
			cout << fixed << setprecision(4) << "LOGLOSS = " << logloss << endl;
			break;
		default:
			break;
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
