#include "mf.h"

struct ViewOption {
    char *src;
	ViewOption(int argc, char **argv);	
    static void exit_view();
};
void ViewOption::exit_view() {
    printf(
    "usage: libmf view file\n"
	"\n"
	"View info in a binary data or model file\n"
    ); exit(1);
}
ViewOption::ViewOption(int argc, char **argv) {
    if(argc!=3) exit_view();
    if(!strcmp(argv[1],"help")) exit_view();
    src = argv[2];
    FILE *f = fopen(src, "rb"); if(!f) exit_file_error(src); fclose(f);
}

void view_data(FILE *f) {
    Matrix *R = new Matrix; fseek(f,0,SEEK_SET);
    R->read_meta(f);
    printf("number of users = %d\n", R->nr_us);
    printf("number of items = %d\n", R->nr_is);
    printf("number of ratings = %ld\n", R->nr_rs);
    printf("rating average = %f\n", R->avg);
}

void view_model(FILE *f) {
    Model *model = new Model; fseek(f,0,SEEK_SET);
    model->read_meta(f);
    printf("dimensions = %d\n", model->dim);
    printf("iterations = %d\n", model->iter);
    printf("lambda p = %f\n", model->lp);
    printf("lambda q = %f\n", model->lq);
    if(model->en_ub) printf("lambda user bias = %f\n", model->lub);
    if(model->en_ib) printf("lambda item bias = %f\n", model->lib);
    printf("gamma = %f\n", model->gamma);
    printf("random shuffle = %d\n", (int)model->en_rand_shuffle);
    printf("use average = %d\n", (int)model->en_avg);
}

void view(int argc, char **argv) {
    ViewOption option(argc,argv);

    FILE *f = fopen(option.src, "rb"); int type;

    fread(&type,sizeof(int),1,f);

    if(type==DATA) view_data(f);
    else if(type==MODEL) view_model(f);
    else fprintf(stderr,"Invalid file type.\n");

    fclose(f);
}
