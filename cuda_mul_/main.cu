#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <random>
#include <cmath>
#include <time.h>
#include <functional>
#include <iomanip>
#include <vector>

using namespace std;
typedef long long int64_t;

//random engine
std::default_random_engine rand_gen(time(NULL));
std::uniform_real_distribution<float> unif_distr(0.0, 1.0);
auto float_rand = std::bind(unif_distr, rand_gen);

const int vec_size = 16 * 1024 * 1024;
const int b_vec_size = vec_size * sizeof(float);

class Timer{
private:
	int64_t _t1;
public:
	void start(){ _t1 = __rdtsc();}
	int64_t diff(){ return __rdtsc() - _t1; }
};


ostream& show_dev_info(ostream& s, cudaDeviceProp& dev_prop){
	s.fill('.');
	int col_size = 40;
	s << setw(col_size) << left << "device name: " << dev_prop.name << endl
		<< setw(col_size) << "total glob mem(KB): " << left << dev_prop.totalGlobalMem / 1024 << endl
		<< setw(col_size) << "shared mem per block: " << left << dev_prop.sharedMemPerBlock << endl
		<< setw(col_size) << "register per block: " << left << dev_prop.regsPerBlock << endl
		<< setw(col_size) << "warp size: " << left << dev_prop.warpSize << endl
		<< setw(col_size) << "memory pitch(KB): " << left << dev_prop.memPitch / 1024 << endl
		<< setw(col_size) << "max threads per block: " << left << dev_prop.maxThreadsPerBlock << endl;
	//max threads dimensions: x,y,z
	s << "thread dimensions, x : " << dev_prop.maxThreadsDim[0] << " y : " << dev_prop.maxThreadsDim[1] << " z : " << dev_prop.maxThreadsDim[2] << endl;
	s << "max grid size: x : " << dev_prop.maxGridSize[0] << " y : " << dev_prop.maxGridSize[1] << " z : " << dev_prop.maxGridSize[2] << endl;
	s << "clock rate: " << dev_prop.clockRate << endl;
	s << "compute compability : " << dev_prop.major << "." << dev_prop.minor << endl;
	s << "texture alignment" << dev_prop.textureAlignment << endl;
	s << "device overlap" << dev_prop.deviceOverlap << endl;
	s << "total constant memory(KB):" << dev_prop.totalConstMem / 1024 << endl;
	s << "multiprocessor count " << dev_prop.multiProcessorCount << endl;
	return s;
}

ostream& get_devs_info(ostream& s){
	int n_devs = 0;
	cudaDeviceProp dev_prop;
	cudaGetDeviceCount(&n_devs);
	for (int i = 0; i < n_devs; i++){
		s << "Device " << i << endl;
		cudaGetDeviceProperties(&dev_prop, i);
		show_dev_info(s, dev_prop);
	}
	return s;
}


void randomize_vect(float*& v,int size){
	for (int i = 0; i < size; i++)
		v[i] = float_rand();
}

void cuda_copy_measure(float*& src,float*& dst,int size,cudaMemcpyKind cpyKind,const char* label){
	//create events for syncronization and time measurement
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//begin measurements for host to device copy
	float time;
	cudaEventRecord(start);
	cudaMemcpy(dst, src, size, cpyKind);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "host to device " << label <<" time(ms) : " << time << endl;

	//free resources
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

int main(){
	get_devs_info(cout);

	//allocate host memory
	float* v1_host = new float[vec_size];
	float* v2_host = new float[vec_size];
	float* res_host = new float[vec_size];
	//fill host memory
	randomize_vect(v1_host,vec_size);
	randomize_vect(v2_host,vec_size);
	//allocate device memory
	float* v1_dev = nullptr;
	float* v2_dev = nullptr;
	float* res_dev = nullptr;
	cudaMalloc((void**)&v1_dev, b_vec_size);
	cudaMalloc((void**)&v2_dev, b_vec_size);
	cudaMalloc((void**)&res_dev, b_vec_size);
	//set kernel launch config
	dim3 n_threads = dim3(1024);
	dim3 n_blocks = dim3(vec_size / n_threads.x);
	//begin measurements for host to device copy
	cuda_copy_measure(v1_dev, v1_host, b_vec_size, cudaMemcpyHostToDevice, "v1");
	cuda_copy_measure(v2_dev, v2_host, b_vec_size, cudaMemcpyHostToDevice, "v2");
	//free resources

	cudaFree(v1_dev);
	cudaFree(v2_dev);
	cudaFree(res_dev);

	delete[] v1_host;
	delete[] v2_host;
	delete[] res_host;

	system("pause");
}