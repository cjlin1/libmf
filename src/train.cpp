#include "mf.h"

struct GridMatrix;

struct Monitor {
    int iter, *nr_tr_usrs, *nr_tr_isrs;
    float tr_time;
    bool en_show_tr_rmse, en_show_obj;
    Matrix *Va;
    Model *model;
    Monitor();
    void print_header();
    void show(float iter_time, double loss, float tr_rmse);
    void scan_tr(Matrix *Tr);
    double calc_reg();
    ~Monitor();
};
Monitor::Monitor() : iter(0), tr_time(0.0), Va(NULL), model(NULL) { }  
void Monitor::print_header() {
    char output[1024];
    sprintf(output, "%4s", "iter"); 
    sprintf(output+strlen(output), " %10s", "time"); 
    if(en_show_tr_rmse) sprintf(output+strlen(output), " %10s", "tr_rmse"); 
    if(Va) sprintf(output+strlen(output), " %10s", "va_rmse"); 
    if(en_show_obj) sprintf(output+strlen(output), " %13s %13s %13s", "loss", "reg", "obj"); 
    printf("%s\n", output);
}
void Monitor::show(float iter_time, double loss, float tr_rmse) {
    char output[1024]; tr_time += iter_time;
    sprintf(output, "%-4d %10.2f", iter++, tr_time); 
    if(en_show_tr_rmse) sprintf(output+strlen(output), " %10.3f", tr_rmse); 
    if(Va) sprintf(output+strlen(output), " %10.3f", calc_rmse(model,Va)); 
    if(en_show_obj) {
        double reg = calc_reg();
        sprintf(output+strlen(output), " %13.3e %13.3e %13.3e", loss, reg, loss+reg); 
    }
    printf("%s\n", output); fflush(stdout);  
}
void Monitor::scan_tr(Matrix *Tr) {
    nr_tr_usrs = new int[Tr->nr_us]; nr_tr_isrs = new int[Tr->nr_is];
    for(int ux=0; ux<Tr->nr_us; ux++) nr_tr_usrs[ux] = 0; 
    for(int ix=0; ix<Tr->nr_is; ix++) nr_tr_isrs[ix] = 0;
    for(long rx=0; rx<Tr->nr_rs; rx++) { nr_tr_usrs[Tr->M[rx].uid]++; nr_tr_isrs[Tr->M[rx].iid]++; }
}
double Monitor::calc_reg() {
    double reg_p = 0, reg_q = 0;
    for(int ux=0; ux<model->nr_us; ux++) reg_p += nr_tr_usrs[ux]*std::inner_product(&model->P[ux*model->dim_off],&model->P[ux*model->dim_off]+model->dim,&model->P[ux*model->dim_off],0.0); 
    for(int ix=0; ix<model->nr_is; ix++) reg_q += nr_tr_isrs[ix]*std::inner_product(&model->Q[ix*model->dim_off],&model->Q[ix*model->dim_off]+model->dim,&model->Q[ix*model->dim_off],0.0); 
    return reg_p*model->lp + reg_q*model->lq;
}
Monitor::~Monitor() { delete [] nr_tr_usrs; delete [] nr_tr_isrs; }


struct TrainOption {
	char *tr_path, *va_path, *model_path;
	TrainOption(int argc, char **argv, Model *model, Monitor *monitor);	
    static void exit_train();
	~TrainOption();
};
TrainOption::TrainOption(int argc, char **argv, Model *model, Monitor *monitor) : va_path(NULL) {

    model->dim=40, model->nr_thrs=4, model->iter=40, model->nr_gubs=0, model->nr_gibs=0, model->lp=1, model->lq=1, model->lub=model->lp, model->lib=model->lq, model->gamma=0.001, model->avg=0.0, model->en_rand_shuffle=false, model->en_avg=true, model->en_ub=true, model->en_ib=true;

    monitor->en_show_tr_rmse=false, monitor->en_show_obj=false;
	
    int i;
	for(i=2; i<argc; i++) {
		if(argv[i][0]!='-') break;
		if(i+1>=argc) exit_train();
        if(!strcmp(argv[i], "-k")) {
            model->dim = atoi(argv[++i]);
            if(model->dim<=0) { fprintf(stderr,"dimensions should > 0\n"); exit(1); }
        }
        else if(!strcmp(argv[i], "-t")) {
            model->iter = atoi(argv[++i]);
            if(model->iter<=0) { fprintf(stderr,"iterations should > 0\n"); exit(1); }
        }
        else if(!strcmp(argv[i], "-s")) {
            model->nr_thrs = atoi(argv[++i]);
            if(model->nr_thrs<=0) { fprintf(stderr,"number of threads should > 0\n"); exit(1); }
        }
        else if(!strcmp(argv[i], "-p")) {
            model->lp = atof(argv[++i]);
            if(model->lp<0) { fprintf(stderr,"cost should >= 0\n"); exit(1); }
        }
        else if(!strcmp(argv[i], "-q")) {
            model->lq = atof(argv[++i]);
            if(model->lq<0) { fprintf(stderr,"cost should >= 0\n"); exit(1); }
        }
        else if(!strcmp(argv[i], "-g")) {
            model->gamma = atof(argv[++i]);
            if(model->gamma<=0) { fprintf(stderr,"learning rate should > 0\n"); exit(1); }
        }
        else if(!strcmp(argv[i], "-v")) va_path = argv[++i];
        else if(!strcmp(argv[i], "-blk")) {
            char *p = strtok(argv[++i],"x");
            model->nr_gubs = atoi(p);

            p = strtok(NULL,"x");
            model->nr_gibs = atoi(p);

            if(model->nr_gubs<=0 || model->nr_gibs<=0) { fprintf(stderr,"number of blocks should > 0\n"); exit(1); }
        }
        else if(!strcmp(argv[i], "--rand-shuffle")) model->en_rand_shuffle = true;
        else if(!strcmp(argv[i], "--no-rand-shuffle")) model->en_rand_shuffle = false;
        else if(!strcmp(argv[i], "--tr-rmse")) monitor->en_show_tr_rmse = true;
        else if(!strcmp(argv[i], "--no-tr-rmse")) monitor->en_show_tr_rmse = false;
        else if(!strcmp(argv[i], "--obj")) monitor->en_show_obj = true;
        else if(!strcmp(argv[i], "--no-obj")) monitor->en_show_obj = false;
        else if(!strcmp(argv[i], "--use-avg")) model->en_avg = true;
        else if(!strcmp(argv[i], "--no-use-avg")) model->en_avg = false;
        else if(!strcmp(argv[i], "--user-bias")) model->en_ub = true;
        else if(!strcmp(argv[i], "--no-user-bias")) model->en_ub = false;
        else if(!strcmp(argv[i], "--item-bias")) model->en_ib = true;
        else if(!strcmp(argv[i], "--no-item-bias")) model->en_ib = false;
        else if(!strcmp(argv[i], "-ub")) {
            float lub = atof(argv[++i]);
            if(lub<0) model->en_ub = false;
            else {
                model->en_ub = true;
                model->lub = lub;
            }
        }
        else if(!strcmp(argv[i], "-ib")) {
            float lib = atof(argv[++i]);
            if(lib<0) model->en_ib = false;
            else {
                model->en_ib = true;
                model->lib = lib;
            }
        }
        else { fprintf(stderr,"Invalid option: %s\n", argv[i]); exit_train(); }
	}

    if(model->nr_gubs==0) (model->nr_gubs) = 2*(model->nr_thrs);
    if(model->nr_gibs==0) (model->nr_gibs) = 2*(model->nr_thrs);

	if(i>=argc) exit_train();

	tr_path = argv[i++]; 
	
	if(i<argc) {
		model_path = new char[strlen(argv[i])+1];
		sprintf(model_path,"%s",argv[i]);
	}
	else {
		char *p = strrchr(argv[i-1],'/');
		if(p==NULL)
			p = argv[i-1];
		else
			++p;
        model_path = new char[strlen(p)+7];
		sprintf(model_path,"%s.model",p);
	}

    if(va_path) { FILE *f = fopen(va_path, "rb"); if(!f) exit_file_error(va_path); fclose(f); } //Check if validation set exist.

} 
void TrainOption::exit_train() {
    printf(
    "usage: libmf train [options] binary_train_file model\n"
    "\n"
    "options:\n" 
    "-k <dimensions>: set the number of dimensions (default 40)\n" 
    "-t <iterations>: set the number of iterations (default 40)\n" 
    "-s <number of threads>: set the number of threads (default 4)\n" 
    "-p <cost>: set the regularization cost for P (default 1)\n" 
    "-q <cost>: set the regularization cost for Q (default 1)\n" 
    "-ub <cost>: set the regularization cost for user bias (default 1), set <0 to disable\n"
    "-ib <cost>: set the regularization cost for item bias (default 1), set <0 to disable\n"
    "-g <gamma>: set the learning rate for parallel SGD (default 0.001)\n" 
    "-v <path>: set the path to validation set\n" 
    "-blk <blocks>: set the number of blocks for parallel SGD (default 2s x 2s)\n" 
    "    For example, if you want 3x4 blocks, then use '-blk 3x4'\n" 
    "--rand-shuffle --no-rand-shuffle: enable / disable random suffle (default disabled)\n"
    "    This options should be used when the data is imbalanced.\n"
    "--tr-rmse --no-tr-rmse: enable / disable show rmse on training data (default disabled)\n"
	"    This option shows the estimated RMSE on training data. It also slows down the training procedure.\n"
    "--obj --no-obj: enable / disable show objective value (default disabled)\n"
	"    This option shows the estimated objective value on training data. It also slows down the training procedure.\n"
    "--use-avg --no-use-avg: enable / disable using training data average (default enabled)\n"
    ); 
    exit(1);
}
TrainOption::~TrainOption() { delete [] model_path; }


struct GridMatrix {
    int nr_gubs, nr_gibs;
    long nr_rs;
    Matrix **GMS;
    GridMatrix(Matrix *G, int *map_u, int *map_i, int nr_gubs, int nr_gibs, int nr_thrs);
    static void sort_ratings(Matrix *M, std::mutex *mtx, int *nr_thrs);
    ~GridMatrix();
};
GridMatrix::GridMatrix(Matrix *R, int *map_u, int *map_i, int nr_gubs, int nr_gibs, int nr_thrs) : nr_gubs(nr_gubs), nr_gibs(nr_gibs) {
    printf("Griding..."); fflush(stdout);
    Clock clock; clock.tic(); GMS = new Matrix*[nr_gubs*nr_gibs]; this->nr_rs = R->nr_rs; std::mutex mtx;

    int seg_u = (int)ceil(double(R->nr_us)/nr_gubs), seg_i = (int)ceil(double(R->nr_is)/nr_gibs);

    int r_map[nr_gubs][nr_gibs];
    for(int mx=0; mx<nr_gubs; mx++) for(int nx=0; nx<nr_gibs; nx++) r_map[mx][nx] = 0;
    for(long rx=0; rx<R->nr_rs; rx++) { 
        int new_uid = map_u? map_u[R->M[rx].uid] : R->M[rx].uid; int new_iid = map_i? map_i[R->M[rx].iid] : R->M[rx].iid;
        r_map[new_uid/seg_u][new_iid/seg_i]++; 
    }
    for(int mx=0; mx<nr_gubs; mx++) for(int nx=0; nx<nr_gibs; nx++) GMS[mx*nr_gibs+nx] = new Matrix(r_map[mx][nx],-1,-1,0);

    for(int mx=0; mx<nr_gubs; mx++) for(int nx=0; nx<nr_gibs; nx++) r_map[mx][nx] = 0; // Use r_map as index counter.
    for(long rx=0; rx<R->nr_rs; rx++) {
        int new_uid = map_u? map_u[R->M[rx].uid] : R->M[rx].uid; int new_iid = map_i? map_i[R->M[rx].iid] : R->M[rx].iid;
        int thub = new_uid/seg_u, thib = new_iid/seg_i;
        GMS[thub*nr_gibs+thib]->M[r_map[thub][thib]] = R->M[rx];
        GMS[thub*nr_gibs+thib]->M[r_map[thub][thib]].uid = new_uid;
        GMS[thub*nr_gibs+thib]->M[r_map[thub][thib]++].iid = new_iid;
    }

    if(map_u) {
        int nr_alive_thrs = 0;
        for(int mx=0; mx<nr_gubs*nr_gibs; mx++) {
            while(nr_alive_thrs>=nr_thrs) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
            { std::lock_guard<std::mutex> lock(mtx); nr_alive_thrs++; }
            std::thread worker = std::thread(GridMatrix::sort_ratings, GMS[mx], &mtx, &nr_alive_thrs); worker.detach();
        }
        while(nr_alive_thrs!=0) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
    }

    printf("done. %.2f\n", clock.toc()); fflush(stdout);

    if(EN_SHOW_GRID) {
        printf("\n");
        for(int mx=0; mx<nr_gubs; mx++) {
            for(int nx=0; nx<nr_gibs; nx++) printf("%7ld ", GMS[mx*nr_gibs+nx]->nr_rs);
            printf("\n");
        }
        printf("\n");
    }
}
void GridMatrix::sort_ratings(Matrix *M, std::mutex *mtx, int *nr_thrs) {
    M->sort();
    std::lock_guard<std::mutex> lock(*mtx);
    (*nr_thrs)--;
}
GridMatrix::~GridMatrix() { for(int ix=0; ix<nr_gubs*nr_gibs; ix++) delete GMS[ix]; delete GMS; }

class Scheduler {
    int *nr_jts, *order, nr_gubs, nr_gibs, nr_thrs, total_jobs, nr_paused_thrs;
    bool *blocked_u, *blocked_i, paused, terminated;
    double *losses;
    std::mutex mtx;
    bool all_paused();
public:
    Scheduler(int nr_gubs, int nr_gibs, int nr_thrs);
    int get_job();
    void put_job(int jid, double loss);
    double get_loss();
    int get_total_jobs();
    void pause_sgd();
    void pause();
    void resume();
    void terminate();
    bool is_terminated();
    void show();
    ~Scheduler();
};
Scheduler::Scheduler(int nr_gubs, int nr_gibs, int nr_thrs) : nr_gubs(nr_gubs), nr_gibs(nr_gibs), nr_thrs(nr_thrs), total_jobs(0), nr_paused_thrs(0), paused(false), terminated(false) {
    nr_jts = new int[nr_gubs*nr_gibs]; order = new int[nr_gubs*nr_gibs]; blocked_u = new bool[nr_gubs]; blocked_i = new bool[nr_gibs]; losses = new double[nr_gubs*nr_gibs];
    for(int mx=0; mx<nr_gubs*nr_gibs; mx++) nr_jts[mx]=0, losses[mx]=0, order[mx]=mx; 
    for(int mx=0; mx<nr_gubs; mx++) blocked_u[mx] = false; 
    for(int mx=0; mx<nr_gibs; mx++) blocked_i[mx] = false; 
}
int Scheduler::get_job() {
    int jid=-1, ts=INT_MAX;
    while(true) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            for(int mx=0; mx<nr_gubs*nr_gibs; mx++) {
                int nx = order[mx];
                if(blocked_u[nx/nr_gibs] || blocked_i[nx%nr_gibs]) continue;
                if(nr_jts[nx]<ts) ts=nr_jts[nx], jid=nx;
            }
        }
        if(jid!=-1) break;
        pause();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        blocked_u[jid/nr_gibs]=true, blocked_i[jid%nr_gibs]=true, nr_jts[jid]++;
    }
    return jid;
}
void Scheduler::put_job(int jid, double loss) { std::lock_guard<std::mutex> lock(mtx); blocked_u[jid/nr_gibs]=false; blocked_i[jid%nr_gibs]=false; losses[jid]=loss; total_jobs++; }
double Scheduler::get_loss() {
    double loss = 0;
    for(int ix=0; ix<nr_gubs*nr_gibs; ix++) loss += losses[ix];
    return loss;
}
int Scheduler::get_total_jobs() { return total_jobs; }
void Scheduler::pause_sgd() { 
    {
        std::lock_guard<std::mutex> lock(mtx);
        paused = true; 
    }
    while(!all_paused()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
void Scheduler::pause() { 
    {
        std::lock_guard<std::mutex> lock(mtx);
        if(!paused) return; 
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        nr_paused_thrs++;
    }
    while(paused) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    {
        std::lock_guard<std::mutex> lock(mtx);
        nr_paused_thrs--;
    }
}
bool Scheduler::all_paused() { 
    std::lock_guard<std::mutex> lock(mtx);
    return (nr_paused_thrs==nr_thrs);
}
void Scheduler::resume() { 
    std::lock_guard<std::mutex> lock(mtx); 
    std::random_shuffle(order, order+nr_gubs*nr_gibs);
    paused = false;
}
void Scheduler::terminate() { terminated = true; }
bool Scheduler::is_terminated() { return terminated; }
void Scheduler::show() {
    for(int mx=0; mx<nr_gubs; mx++) {
        for(int nx=0; nx<nr_gibs; nx++) printf("%3d ", nr_jts[mx*nr_gibs+nx]);
        printf("\n");
    }
    printf("\n"); fflush(stdout);
}
Scheduler::~Scheduler() { delete [] nr_jts; delete [] order; delete [] losses; delete [] blocked_u; delete [] blocked_i; }

void sgd(GridMatrix *TrG, Model *model, Scheduler *scheduler, int tid) {

    float *const P=model->P, *const Q=model->Q, *const UB=model->UB, *const IB=model->IB; const int dim=model->dim_off;
    Node *r, *rn; float *p, *pn, *q, *qn, *ub, *ubn, *ib, *ibn; int dx, jid; long mx, nr_rs; double loss; bool en_ub=model->en_ub, en_ib=model->en_ib;
    __m128 XMMglp = _mm_load1_ps(&model->glp), XMMglq = _mm_load1_ps(&model->glq), XMMg = _mm_load1_ps(&model->gamma), XMMavg = _mm_load1_ps(&model->avg);
    __m128d XMMl = _mm_setzero_pd();

    while(true) {
		jid = scheduler->get_job();
		rn = TrG->GMS[jid]->M; nr_rs = TrG->GMS[jid]->nr_rs; pn = P + rn->uid*dim; qn = Q + rn->iid*dim; ubn = UB + rn->uid; ibn = IB + rn->iid;
    	XMMl = _mm_setzero_pd();
		for(mx=0; mx<nr_rs; mx++) {
			r = rn; rn++; __m128 XMMr = _mm_load1_ps(&r->rate), XMMge = _mm_setzero_ps(), XMMub, XMMib;
			p = pn; pn = P + rn->uid*dim;
			q = qn; qn = Q + rn->iid*dim;
			ub = ubn; ubn = UB + rn->uid;
			ib = ibn; ibn = IB + rn->iid;
			for(dx=0; dx<dim-7; dx+=8) {
				__m128 XMMp0 = _mm_load_ps(p+dx);   __m128 XMMq0 = _mm_load_ps(q+dx); 
				__m128 XMMp1 = _mm_load_ps(p+dx+4); __m128 XMMq1 = _mm_load_ps(q+dx+4); 
				XMMp0 = _mm_mul_ps(XMMp0,XMMq0); XMMp1 = _mm_mul_ps(XMMp1,XMMq1);
				XMMge = _mm_add_ps(XMMge, _mm_add_ps(XMMp0,XMMp1)); 
			}
			for(; dx<dim; dx+=4) {
				__m128 XMMp0 = _mm_load_ps(p+dx);   __m128 XMMq0 = _mm_load_ps(q+dx); 
				XMMge = _mm_add_ps(XMMge, _mm_mul_ps(XMMp0,XMMq0)); 
			}
			XMMge = _mm_hadd_ps(XMMge,XMMge); XMMge = _mm_hadd_ps(XMMge,XMMge); 
			if(en_ub) { XMMub = _mm_load1_ps(ub); XMMge = _mm_add_ps(XMMge,XMMub); }
			if(en_ib) { XMMib = _mm_load1_ps(ib); XMMge = _mm_add_ps(XMMge,XMMib); }
			XMMge = _mm_sub_ps(XMMr,_mm_add_ps(XMMge,XMMavg)); XMMl = _mm_add_pd(XMMl,_mm_cvtps_pd(_mm_mul_ps(XMMge,XMMge))); XMMge = _mm_mul_ps(XMMge,XMMg);
			_mm_prefetch(rn,_MM_HINT_T0); _mm_prefetch(rn+7,_MM_HINT_T1); _mm_prefetch(rn+15,_MM_HINT_T2);
			_mm_prefetch(qn,_MM_HINT_T0); _mm_prefetch(Q+(rn+7)->iid*dim,_MM_HINT_T1); _mm_prefetch(Q+(rn+15)->iid*dim,_MM_HINT_T2);
			_mm_prefetch(pn,_MM_HINT_T0); _mm_prefetch(P+(rn+7)->uid*dim,_MM_HINT_T1); _mm_prefetch(P+(rn+15)->uid*dim,_MM_HINT_T2);
			for(dx=0; dx<dim-7; dx+=8) {
				__m128 XMMp0 = _mm_load_ps(p+dx);   __m128 XMMq0 = _mm_load_ps(q+dx);   __m128 XMMt0 = _mm_add_ps(_mm_setzero_ps(),XMMp0);
				__m128 XMMp1 = _mm_load_ps(p+dx+4); __m128 XMMq1 = _mm_load_ps(q+dx+4); __m128 XMMt1 = _mm_add_ps(_mm_setzero_ps(),XMMp1);
				XMMp0 = _mm_add_ps(_mm_mul_ps(XMMge,XMMq0),_mm_mul_ps(XMMglp,XMMp0));
				XMMq0 = _mm_add_ps(_mm_mul_ps(XMMge,XMMt0),_mm_mul_ps(XMMglq,XMMq0));
				XMMp1 = _mm_add_ps(_mm_mul_ps(XMMge,XMMq1),_mm_mul_ps(XMMglp,XMMp1));
				XMMq1 = _mm_add_ps(_mm_mul_ps(XMMge,XMMt1),_mm_mul_ps(XMMglq,XMMq1));
				_mm_store_ps(p+dx,XMMp0); _mm_store_ps(q+dx,XMMq0);
				_mm_store_ps(p+dx+4,XMMp1); _mm_store_ps(q+dx+4,XMMq1);
			}
			for(; dx<dim; dx+=4) {
				__m128 XMMp0 = _mm_load_ps(p+dx);   __m128 XMMq0 = _mm_load_ps(q+dx);   __m128 XMMt0 = _mm_add_ps(_mm_setzero_ps(),XMMp0);
				XMMp0 = _mm_add_ps(_mm_mul_ps(XMMge,XMMq0),_mm_mul_ps(XMMglp,XMMp0));
				XMMq0 = _mm_add_ps(_mm_mul_ps(XMMge,XMMt0),_mm_mul_ps(XMMglq,XMMq0));
				_mm_store_ps(p+dx,XMMp0); _mm_store_ps(q+dx,XMMq0);
			}
			float ge;
			if(en_ub || en_ib) {
				_mm_store1_ps(&ge,XMMge);
				if(en_ub) { *ub = model->glp*(*ub) + ge; }
				if(en_ib) { *ib = model->glq*(*ib) + ge; }
			}
		}
		_mm_store_sd(&loss,XMMl); scheduler->put_job(jid,loss);
		scheduler->pause();
		if(scheduler->is_terminated()) break;
    }
}

void gsgd(GridMatrix *TrG, Model *model, Monitor *monitor) {
	printf("SGD Starts!\n"); fflush(stdout);
    int iter=1; Scheduler *scheduler = new Scheduler(model->nr_gubs,model->nr_gibs,model->nr_thrs); std::vector<std::thread> threads; Clock clock; clock.tic(); 

    for(int tx=0; tx<model->nr_thrs; tx++) threads.push_back(std::thread(sgd,TrG,model,scheduler,tx));

    monitor->print_header();
    while(iter<=model->iter) {
        if(scheduler->get_total_jobs()>=iter*model->nr_gubs*model->nr_gibs) {
            scheduler->pause_sgd();

            float iter_time = clock.toc(); double loss = scheduler->get_loss(); 
            if(EN_SHOW_SCHED) scheduler->show();
            monitor->show(iter_time,loss,sqrt(loss/TrG->nr_rs));
            iter++; clock.tic();

            scheduler->resume();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }     
    scheduler->terminate();

	printf("Waiting for all threads terminate..."); fflush(stdout); clock.tic();
    for(auto it=threads.begin(); it!=threads.end(); it++) it->join(); delete scheduler;   
    printf("done. %.2f\n", clock.toc()); fflush(stdout);
}

void train(int argc, char **argv) {

    Model *model = new Model;

    Monitor *monitor = new Monitor;

    TrainOption *option = new TrainOption(argc, argv, model, monitor);
	
    Matrix *Tr, *Va=NULL; GridMatrix *TrG;

    Tr = new Matrix(option->tr_path);

    model->initialize(Tr);
    if(model->en_rand_shuffle) model->gen_rand_map();

    if(option->va_path) {
        if(model->en_rand_shuffle) Va = new Matrix(option->va_path,model->map_uf,model->map_if);
        else Va = new Matrix(option->va_path);
    }

    if(Va && (Va->nr_us>Tr->nr_us || Va->nr_is>Tr->nr_is)) { fprintf(stderr, "Validation set out of range.\n"); exit(1); }
     
    if(model->en_rand_shuffle) model->shuffle();

    monitor->model = model; monitor->Va = Va; monitor->scan_tr(Tr);

    TrG = new GridMatrix(Tr,model->map_uf,model->map_if,model->nr_gubs,model->nr_gibs,model->nr_thrs);

    delete Tr;

    gsgd(TrG,model,monitor);

    if(model->en_rand_shuffle) model->inv_shuffle();

    model->write(option->model_path);

    delete model; delete monitor; delete option; delete TrG; 
}
