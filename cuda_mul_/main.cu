#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <random>
#include <cmath>
#include <time.h>
#include <functional>
#include <iomanip>
#include <vector>
#include <Windows.h>

using namespace std;
typedef long long int64_t;
typedef unsigned long long uint64_t;

//random engine
std::default_random_engine rand_gen(time(NULL));
std::uniform_real_distribution<float> unif_distr(0.0, 1.0);
auto float_rand = std::bind(unif_distr, rand_gen);

const int MB = 2 << 19;
const int vec_size = 16 * MB;
const int b_vec_size = vec_size * sizeof(float);

class Timer{
private:
	uint64_t _t1;
	uint64_t _freq;
public:
	Timer(){
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		_freq = freq.QuadPart;
	}
	void nano_start(){ 
		_t1 = __rdtsc();
	}
	void start(){
		LARGE_INTEGER time_stemp;
		QueryPerformanceCounter(&time_stemp);
		_t1 = time_stemp.QuadPart;
	}
	double nano_time_diff(){ return (double)(__rdtsc() - _t1) / _freq / 1000; }
	double time_diff(){ 
		LARGE_INTEGER time_stemp;
		QueryPerformanceCounter(&time_stemp);
		return (double)(time_stemp.QuadPart - _t1) / _freq;
	}
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


class CudaEventSyncTimer{
	/*cudaDeviceSynchronize(), is that they stall the GPU pipeline.
	For this reason, CUDA offers a relatively light - weight alternative to CPU timers via 
	the CUDA event API.
	*/
private:
	cudaEvent_t _start;
	cudaEvent_t _stop;
public:
	CudaEventSyncTimer(){
		cudaEventCreate(&_start);
		cudaEventCreate(&_stop);
	}
	~CudaEventSyncTimer(){
		cudaEventDestroy(_start);
		cudaEventDestroy(_stop);
	}
	void start(cudaStream_t stream_no = 0){
		cudaEventRecord(_start, stream_no);
	}
	float sync_stop(cudaStream_t stream_no = 0){
		cudaEventRecord(_stop,stream_no);
		cudaEventSynchronize(_stop);
		float time;
		cudaEventElapsedTime(&time, _start, _stop);
		return time;
	}
	float stop(){
		cudaEventRecord(_stop);
		float time;
		cudaEventElapsedTime(&time, _start, _stop);
		return time;
	}
};

__global__ void cuda_mul_kern(float* v1, float* v2, float* res){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	res[idx] = v1[idx] * v2[idx];
}

double calc_bandwidth(int n_bytes, double sec){
	return (double)n_bytes / MB / sec;
}

void write_vec(float* v,int v_size,ostream& s){
	for (int i = 0; i < v_size; i++)
		s << v[i] << " ";
}

void sequential_mul(float* v1, float* v2, float* res, int size){
	for (int i = 0; i < size; i++)
		res[i] = v1[i] * v2[i];
}


int main(){
	//get_devs_info(cout);
	//allocate host memory	
	float* v1_host = new float[vec_size];
	float* v2_host = new float[vec_size];
	float* res_host = new float[vec_size];
	float* res_sequential = new float[vec_size];
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
	//begin measurements for host to device copy
	float millisec;
	CudaEventSyncTimer cuda_timer;
	cuda_timer.start();
	//The data transfers between the host and device using cudaMemcpy() are synchronous 
	cudaMemcpy(v1_dev, v1_host, b_vec_size, cudaMemcpyHostToDevice);
	millisec = cuda_timer.sync_stop();
	cout << "host to device v1 time(ms) : "<< millisec ;
	cout << setw(40) << "bandwidth (MB/s) : " << calc_bandwidth(b_vec_size, millisec / 1000) << endl;
	//cout << "real sec : " << (double)(clock() - t1)/CLOCKS_PER_SEC << endl;
	//cout << "host to device v1 time(s) : " << t.clock_diff() << endl;
	

	//t.start();
	cuda_timer.start();
	cudaMemcpy(v2_dev, v2_host, b_vec_size, cudaMemcpyHostToDevice);
	millisec = cuda_timer.sync_stop();
	cout << "host to device v2 time(ms) : " << millisec;
	cout << setw(40) << "bandwidth (MB/s) : " << calc_bandwidth(b_vec_size, millisec / 1000) << endl;
	//cout << "host to device v2 time(s) : " << t.clock_diff() << endl;


	//set kernel launch config
	dim3 n_threads = dim3(1024);
	cout << "threads per block : " << n_threads.x << endl;
	//dim3 n_blocks = dim3(vec_size / n_threads.x);
	dim3 n_blocks = dim3((vec_size + n_threads.x - 1) / n_threads.x);
	cout << "blocks per grid : " << n_blocks.x << endl;
	
	//measure cuda multiplication
	//t.start();
	
	cuda_timer.start();
	//Kernel launches, on the other hand, are asynchronous.
	cuda_mul_kern <<<n_blocks, n_threads>>>(v1_dev, v2_dev, res_dev);
	//block CPU execution until all previously issued commands on the device have completed
	cudaDeviceSynchronize();
	cout << "kernel time(ms) : " << cuda_timer.sync_stop() << endl;
	//cout << "kernel time(ms) : " << t.clock_diff() << endl;
	
	cuda_timer.start();
	cudaMemcpy(res_host,res_dev,b_vec_size,cudaMemcpyDeviceToHost);
	millisec = cuda_timer.sync_stop();
	cout << "device to host result time(ms) : " << millisec;
	cout << setw(40) << "bandwidth (MB/s) : " << calc_bandwidth(b_vec_size, millisec / 1000) << endl;
	
	//sequential multiplication
	Timer timer;
	double time;
	timer.start();
	sequential_mul(v1_host,v2_host,res_sequential,vec_size);
	time = timer.time_diff();
	cout << "sequential mul time (ms) : " << time * 1000 << endl;

	//write cuda result
	fstream file("cuda_result.txt");
	//file.open("cuda_result.txt", ios::out | ios::trunc);
	write_vec(res_host, vec_size, file);
	file.close();



	//free resources
	cudaFree(v1_dev);
	cudaFree(v2_dev);
	cudaFree(res_dev);

	delete[] v1_host;
	delete[] v2_host;
	delete[] res_host;
	
	system("pause");
}