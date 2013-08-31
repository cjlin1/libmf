#include "mf.h"

void exit_file_error(char *path) { fprintf(stderr,"\nError: Invalid file name %s.\n", path); exit(1); }
void exit_file_ver(float current_ver, float file_ver) { 
    fprintf(stderr,"\nError: Inconsistent file version.\n");
    fprintf(stderr,"current version:%.2f    file version:%.2f\n",current_ver,file_ver);
    exit(1); 
}

void Clock::tic() { begin = clock(); }
float Clock::toc() {
    end = clock();
    return (float)(end-begin)/CLOCKS_PER_SEC;
}

Matrix::Matrix() {}
Matrix::Matrix(int nr_rs, int nr_us, int nr_is, float avg) : nr_us(nr_us), nr_is(nr_is), nr_rs(nr_rs), avg(avg) { M = new Node[nr_rs]; }
Matrix::Matrix(char *path) { read(path); }
Matrix::Matrix(char *path, int *map_u, int *map_i) {
    read(path); 
    for(int rx=0; rx<nr_rs; rx++) { M[rx].uid = map_u[M[rx].uid]; M[rx].iid = map_i[M[rx].iid]; }
}
void Matrix::read_meta(FILE *f) {
    int type; float ver;
    fread(&type,sizeof(int),1,f); if(type!=(int)DATA) { fprintf(stderr,"Error: It is not a data file.\n"); exit(1); }
    fread(&ver,sizeof(float),1,f); if(ver!=(float)DATAVER) exit_file_ver(DATAVER,ver);
    fread(this,sizeof(Matrix),1,f);
    this->M = NULL;
}
void Matrix::read(char *path) {
    printf("Reading from %s...",path); fflush(stdout);
    Clock clock; clock.tic();
    FILE *f = fopen(path, "rb"); if(!f) exit_file_error(path);
    read_meta(f);
    M = new Node[nr_rs];
    fread(M,sizeof(Node),nr_rs,f);
    fclose(f);
    printf("done. %.2f\n",clock.toc()); fflush(stdout);
}
void Matrix::write(char *path) {
    printf("Writing %s... ",path); fflush(stdout);
    Clock clock; clock.tic();
    FILE *f = fopen(path,"wb"); if(!f) exit_file_error(path); float ver = (float)DATAVER;int file_type = DATA;
    fwrite(&file_type,sizeof(int),1,f);
    fwrite(&ver,sizeof(float),1,f);
    fwrite(this,sizeof(Matrix),1,f);
    fwrite(M,sizeof(Node),nr_rs,f);
    fclose(f);
    printf("done. %.2f\n",clock.toc()); fflush(stdout);
}
void Matrix::sort() { std::sort(M,M+nr_rs,Matrix::sort_uid_iid); }
bool Matrix::sort_uid_iid(Node lhs, Node rhs) { if(lhs.uid!=rhs.uid) return lhs.uid < rhs.uid; else return lhs.iid < rhs.iid; }
Matrix::~Matrix() { delete [] M; }

Model::Model() {}
Model::Model(char *path) { read(path); }
void Model::initialize(Matrix *Tr) {
    printf("Initializing model..."); fflush(stdout);
    Clock clock; clock.tic();

    nr_us = Tr->nr_us; nr_is = Tr->nr_is; glp=1-gamma*lp; glq=1-gamma*lq; glub=1-gamma*lub; glib=1-gamma*lib; dim_off = dim%4? (dim/4)*4+4 : dim; avg = en_avg? Tr->avg : 0.0;

    srand48(0L);
    P = new float[nr_us*dim_off]; float *p = P;
    for(int px=0; px<nr_us; px++) {
        for(int dx=0; dx<dim; dx++) *(p++)=0.1*drand48();
        for(int dx=dim; dx<dim_off; dx++) *(p++) = 0;
    }

    srand48(0L);
    Q = new float[nr_is*dim_off]; float *q = Q;
    for(int qx=0; qx<nr_is; qx++) {
        for(int dx=0; dx<dim; dx++) *(q++)=0.1*drand48();
        for(int dx=dim; dx<dim_off; dx++) *(q++) = 0;
    }

    if(en_ub) {
        UB = new float[nr_us];
        for(int ubx=0; ubx<nr_us; ubx++) UB[ubx] = 0;
    }

    if(en_ib) {
        IB = new float[nr_is];
        for(int ibx=0; ibx<nr_is; ibx++) IB[ibx] = 0;
    }

    printf("done. %.2f\n", clock.toc()); fflush(stdout);
}
void Model::read_meta(FILE *f) {
    int type; float ver;
    fread(&type,sizeof(int),1,f); if(type!=(int)MODEL) { fprintf(stderr,"Error: It is not a model file.\n"); exit(1); }
    fread(&ver, sizeof(float), 1, f); if(ver!=(float)MODELVER) exit_file_ver(MODELVER,ver);
    fread(this,sizeof(Model),1,f);
    this->P = NULL; this->Q = NULL;
}
void Model::read(char *path) {
    printf("Reading model..."); fflush(stdout);
    Clock clock; clock.tic();
    FILE *f = fopen(path,"rb"); if(!f) exit_file_error(path);
    read_meta(f);
    P = new float[nr_us*dim_off]; Q = new float[nr_is*dim_off];
    fread(P, sizeof(float), nr_us*dim_off, f);
    fread(Q, sizeof(float), nr_is*dim_off, f);
	if(en_ub) {
		UB = new float[nr_us];
		fread(UB, sizeof(float), nr_us, f);
	}
	if(en_ib) {
		IB = new float[nr_is];
		fread(IB, sizeof(float), nr_is, f);
	}
	if(en_rand_shuffle) {
        map_uf = new int[nr_us]; map_ub = new int[nr_us]; map_if = new int[nr_is]; map_ib = new int[nr_is]; 
		fread(map_uf, sizeof(int), nr_us, f);
		fread(map_ub, sizeof(int), nr_us, f);
		fread(map_if, sizeof(int), nr_is, f);
		fread(map_ib, sizeof(int), nr_is, f);
	}
    fclose(f);
    printf("done. %.2f\n",clock.toc()); fflush(stdout);
}
void Model::write(char *path) {
    printf("Writing model..."); fflush(stdout);
    Clock clock; clock.tic();
    FILE *f = fopen(path, "wb"); if(!f) exit_file_error(path); float ver = (float)MODELVER; int file_type = MODEL;
    fwrite(&file_type,sizeof(int),1,f);
    fwrite(&ver,sizeof(float),1,f);
    fwrite(this,sizeof(Model),1,f);
    fwrite(P,sizeof(float),nr_us*dim_off,f);
    fwrite(Q,sizeof(float),nr_is*dim_off,f);
	if(en_ub) fwrite(UB, sizeof(float), nr_us, f);
	if(en_ib) fwrite(IB, sizeof(float), nr_is, f);
	if(en_rand_shuffle) {
		fwrite(map_uf, sizeof(int), nr_us, f);
		fwrite(map_ub, sizeof(int), nr_us, f);
		fwrite(map_if, sizeof(int), nr_is, f);
		fwrite(map_ib, sizeof(int), nr_is, f);
	}
    fclose(f);
    printf("done. %.2f\n", clock.toc()); fflush(stdout);
}
void Model::gen_rand_map() {
    map_uf = new int[nr_us]; map_ub = new int[nr_us]; map_if = new int[nr_is]; map_ib = new int[nr_is];
    for(int ix=0; ix<nr_us; ix++) map_uf[ix] = ix; for(int ix=0; ix<nr_is; ix++) map_if[ix] = ix;
    std::random_shuffle(map_uf,map_uf+nr_us); std::random_shuffle(map_if,map_if+nr_is);
    for(int ix=0; ix<nr_us; ix++) map_ub[map_uf[ix]] = ix; for(int ix=0; ix<nr_is; ix++) map_ib[map_if[ix]] = ix;
}
void Model::shuffle() {
	float *P1 = new float[nr_us*dim_off]; float *Q1 = new float[nr_is*dim_off]; float *UB1 = new float[nr_us]; float *IB1 = new float[nr_is];
	for(int px=0; px<nr_us; px++) std::copy(&P[px*dim_off],&P[px*dim_off+dim_off],&P1[map_uf[px]*dim_off]);
	for(int qx=0; qx<nr_is; qx++) std::copy(&Q[qx*dim_off],&Q[qx*dim_off+dim_off],&Q1[map_if[qx]*dim_off]);
	delete [] P; delete [] Q; P = P1; Q = Q1;
    if(en_ub) {
        for(int px=0; px<nr_us; px++) UB1[map_uf[px]] = UB[px];
        delete [] UB; UB = UB1;
    }
    if(en_ib) {
        for(int qx=0; qx<nr_is; qx++) IB1[map_if[qx]] = IB[qx];
        delete [] IB; IB = IB1;
    }
}
void Model::inv_shuffle() {
	float *P1 = new float[nr_us*dim_off]; float *Q1 = new float[nr_is*dim_off]; float *UB1 = new float[nr_us]; float *IB1 = new float[nr_is];
	for(int px=0; px<nr_us; px++) std::copy(&P[px*dim_off],&P[px*dim_off+dim_off],&P1[map_ub[px]*dim_off]);
	for(int qx=0; qx<nr_is; qx++) std::copy(&Q[qx*dim_off],&Q[qx*dim_off+dim_off],&Q1[map_ib[qx]*dim_off]);
	delete [] P; delete [] Q; P = P1; Q = Q1;
    if(en_ub) {
        for(int px=0; px<nr_us; px++) UB1[map_ub[px]] = UB[px];
        delete [] UB; UB = UB1;
    }
    if(en_ib) {
        for(int qx=0; qx<nr_is; qx++) IB1[map_ib[qx]] = IB[qx];
        delete [] IB; IB = IB1;
    }
}
Model::~Model() { 
    delete [] P; delete [] Q; delete [] map_uf; delete [] map_ub; delete [] map_if; delete [] map_ib; 
    if(en_ub) delete [] UB;
    if(en_ib) delete [] IB;
}

float calc_rate(Model *model, Node *r) { 
    float rate = std::inner_product(&model->P[r->uid*model->dim_off], &model->P[r->uid*model->dim_off]+model->dim, &model->Q[r->iid*model->dim_off], 0.0) + model->avg; 
    if(model->en_ub) rate += model->UB[r->uid];
    if(model->en_ib) rate += model->IB[r->iid];
    return rate;
}
float calc_rmse(Model *model, Matrix *R) {
    double loss=0; float e; 
    for(int rx=0; rx<R->nr_rs; rx++) { e = R->M[rx].rate - calc_rate(model,&R->M[rx]); loss += e*e; }
    return sqrt(loss/R->nr_rs);
}
