#include "mf.h"

struct PredictOption {
    char *model_path, *test_path, *dst_path;
	PredictOption(int argc, char **argv);	
    static void exit_predict();
	~PredictOption();
};
PredictOption::PredictOption(int argc, char **argv) {
    if(argc!=5 && argc!=4) exit_predict();
    if(!strcmp(argv[1],"help")) exit_predict();
    model_path=argv[3], test_path=argv[2]; 
	if(argc==5) {
		dst_path = new char[strlen(argv[4])];
		sprintf(dst_path,"%s",argv[4]);
	}
	else {
		char *p = strrchr(argv[2],'/');
		if(p==NULL)
			p = argv[2];
		else
			++p;
		dst_path = new char[strlen(p)+5];
		sprintf(dst_path,"%s.out",p);
	}
}
PredictOption::~PredictOption() { delete [] dst_path; }

void PredictOption::exit_predict() {
    printf(
    "usage: libmf predict binary_test_file model output\n"
	"\n"
	"Predict a test file from a model\n"
    ); exit(1);
}

void predict(Model *model, char *test_path, char *dst_path) {

    Matrix *Te = new Matrix(test_path); FILE *f = fopen(dst_path, "w"); double rmse = 0;

    printf("Predicting..."); fflush(stdout); Clock clock; clock.tic();

    for(int rx=0; rx<Te->nr_rs; rx++) {
        float rate = calc_rate(model,&Te->M[rx]); 
        float e = Te->M[rx].rate - rate;
        fprintf(f,"%f\n",rate);  rmse += e*e;
    }
    printf("done. %.2lf\n",clock.toc()); fflush(stdout);

    printf("RMSE: %.4lf\n",sqrt(rmse/Te->nr_rs));

    delete Te;
}

void predict(int argc, char **argv) {

    PredictOption *option = new PredictOption(argc,argv);

    Model *model = new Model(option->model_path);

    predict(model, option->test_path, option->dst_path);

    delete option; delete model;
}
