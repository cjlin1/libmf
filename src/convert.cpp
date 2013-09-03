#include "mf.h"

struct ConvertOption {
    char *src, *dst;
    ConvertOption(int argc, char **argv);	
    static void exit_convert();
    ~ConvertOption();
};
ConvertOption::ConvertOption(int argc, char **argv) {
    if(argc!=3 && argc!=4) exit_convert();

    src = argv[2];
    if(argc==4) {
        dst = new char[strlen(argv[3])+1];
        sprintf(dst,"%s",argv[3]); 
    }
    else {
		char *p = strrchr(argv[2],'/');
		if(p==NULL) p = argv[2];
		else p++;
        dst = new char[strlen(p)+5];
		sprintf(dst,"%s.bin",p);
    }
}
void ConvertOption::exit_convert() {
    printf(
        "usage: libmf convert text_file binary_file\n"
        "\n"
        "Convert a text file to a binary file\n"
    ); 
    exit(1);
}
ConvertOption::~ConvertOption() { delete[] dst; }

void convert(char *src_path, char *dst_path) {
    printf("Converting %s... ", src_path); fflush(stdout);
    Clock clock; clock.tic();

    int uid, iid, nr_us=0, nr_is=0, nr_rs; float rate; double sum = 0;
    FILE *f = fopen(src_path, "r"); if(!f) exit_file_error(src_path);
    std::vector<Node> rs; 

    while(fscanf(f,"%d %d %f\n",&uid,&iid,&rate)!=EOF) {
        if(uid+1>nr_us) nr_us = uid+1; if(iid+1>nr_is) nr_is = iid+1; sum += rate;
        Node r; r.uid=uid, r.iid=iid, r.rate=rate; rs.push_back(r);
    }
    nr_rs = rs.size(); fclose(f);

    Matrix *R = new Matrix(nr_rs,nr_us,nr_is,sum/nr_rs);

    for(auto it=rs.begin(); it!=rs.end(); it++) R->M[it-rs.begin()] = (*it); 

    printf("done. %.2f\n", clock.toc()); fflush(stdout);

    R->write(dst_path);

    delete R;
}

void convert(int argc, char **argv) {
    ConvertOption *option = new ConvertOption(argc,argv);

    convert(option->src,option->dst);

    delete option;
}
