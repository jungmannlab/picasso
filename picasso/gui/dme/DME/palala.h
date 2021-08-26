/*
PALALA - PArallel LAmbda LAuncher
Jelmer Cnossen - 2018

PALALA implements a parallel_for loop that either uses multiple CPU threads, or runs on CUDA. 
On CUDA, all the data is automatically copied to- and from the GPU. 
To use, CUDA extended lambda's need to be enabled (lambda functions that can be copied through a kernel function call).
This is done by adding "--expt-extended-lambda" to the NVCC command line.

TODO:
- clean up
- Conditional cudaDeviceSynchronize(). Only if there is memory that needs to be copied back to host
- Allow specifying cuda streams, async memory copyInProgress

Example:

	// Copying all elements of src_data to dst_data. Data will be copied to GPU and back if needed.
	std::vector<int> dst_data(10);
	std::vector<int> src_data(10);

	parallel_for(N, PALALA(int idx, int* dst, const int* src) {
		dst[idx] = src[idx];
	}, dst_data, src_data);
*/
#pragma once
#ifndef PALALA_HEADER_INC
#define PALALA_HEADER_INC

#include <cassert>
#include <future>

#ifdef __CUDACC__
#define PALALA_CUDA
#endif

#include "StringUtils.h"

#ifdef PALALA_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaUtils.h"

#define PLL_DEVHOST __device__ __host__ 
#else
#define PLL_DEVHOST
#endif

struct array_with_debugname {
#ifdef _DEBUG
	const char* debugTag;
	void setDebugName(const char *n) { debugTag = n; }
#else
	void setDebugName(const char *n) {}
#endif
};

// POD to pass to kernel
template<typename T>
struct param_array : array_with_debugname {
	T* data;
	size_t size;
	bool isOutputArray = false; // TODO: Make this templated
	param_array(T* d=0, size_t s=0) : data(d), size(s) {}
	PLL_DEVHOST operator T* () const { return data; }
	PLL_DEVHOST T& operator[](int i) const { return data[i]; }
};


// POD to pass to kernel
template<typename T>
struct const_param_array : array_with_debugname {
	const T* data;
	size_t size;
	const_param_array(const T* d, size_t s) : data(d), size(s) {}
	PLL_DEVHOST operator const T* () const { return data; }
	PLL_DEVHOST const T& operator[] (int i) const { return data[i]; }
};


#ifdef PALALA_CUDA

template<typename T>
struct device_param_buf {
	T* h_data;
	size_t size;
	DeviceArray<T> d_data;

	device_param_buf(const device_param_buf& o) : d_data(o.d_data), h_data(o.h_data), size(o.size) {
#ifdef PALALA_DEBUG
		DebugPrintf("device_param_buf&&\n");
#endif
	}

	device_param_buf(device_param_buf&& o) : d_data(std::forward<DeviceArray<T>>(o.d_data)), h_data(o.h_data), size(o.size) {
#ifdef PALALA_DEBUG
		DebugPrintf("device_param_buf&&\n");
#endif
		o.h_data = nullptr;
		o.size = 0;
	}

	device_param_buf(std::vector<T>& v) : h_data(v.data()), size(v.size()), d_data(v) {
#ifdef PALALA_DEBUG
		DebugPrintf("device_param_buf copy constr..\n");
#endif
	}

	// Device array is entirely in device memory, so 
	device_param_buf(DeviceArray<T>& v) : h_data(0), size(v.size()), d_data(v.d) {
		assert(0);
	}


	device_param_buf( param_array<T>& v ) : h_data(v.data), size(v.size), d_data(v.size, v.data) {
#ifdef PALALA_DEBUG
		DebugPrintf("device_param_buf copy constr..\n");
#endif
	}


	~device_param_buf() {
		// copy back to host
		if (size > 0 && h_data)
			d_data.CopyToHost(h_data);
#ifdef PALALA_DEBUG
		DebugPrintf("~device_param_buf: device data: %d\n", d_data.size);
#endif
	}
	param_array<T> kernelParam() { return param_array<T>(d_data.ptr(), d_data.size()); }
};


template<typename T>
struct const_device_param_buf {
	const T* h_data;
	size_t size;
	DeviceArray<T> d_data;

	const_device_param_buf(const const_device_param_buf& o) : d_data(o.d_data), h_data(o.h_data), size(o.size) {
#ifdef PALALA_DEBUG
		DebugPrintf("const_device_param_buf copy constr.\n");
#endif
	}

	const_device_param_buf(const_device_param_buf&& o) : d_data(std::forward<DeviceArray<T>>(o.d_data)), h_data(o.h_data), size(o.size) {
#ifdef PALALA_DEBUG
		DebugPrintf("const_device_param_buf move constr.\n");
#endif
		o.h_data = 0;
		o.size = 0;
	}

	const_device_param_buf(const std::vector<T>& v) : h_data(v.data()), size(v.size()), d_data(v) {
#ifdef PALALA_DEBUG
		DebugPrintf("const_device_param_buf copy constr..\n");
#endif
	}
	const_device_param_buf(const const_param_array<T>& v) : h_data(v.data), size(v.size), d_data(v.size, v.data) {
#ifdef PALALA_DEBUG
		DebugPrintf("const_device_param_buf(const const_param_array&) copy constr..\n");
#endif
	}
	const_device_param_buf(const param_array<T>& v) : h_data(v.data), size(v.size), d_data(v.size, v.data) {
	}

	~const_device_param_buf() {
#ifdef PALALA_DEBUG
		DebugPrintf("~const_device_param_buf: device data: %d\n", d_data.size);
#endif
	}
	const_param_array<T> kernelParam() const { return const_param_array<T>(d_data.ptr(), d_data.size()); }
};
#endif
template<typename T>
struct host_param_buf {
	T* h_data;
	size_t size;

	host_param_buf(const host_param_buf& o) : h_data(o.h_data), size(o.size) {}
	host_param_buf(const param_array<T>& v) : h_data(v.data), size(v.size) {}
	host_param_buf(std::vector<T>& v) {
		h_data = &v[0];
		size = v.size();
}

	~host_param_buf() {}
	param_array<T> asParamArray() const { return param_array<T>(h_data, size); }

	operator T*() const{ return h_data; }
};

template<typename T>
struct const_host_param_buf {
	const T* h_data;
	size_t size;

	const_host_param_buf(const const_host_param_buf& o) : h_data(o.h_data), size(o.size) {}

	const_host_param_buf(const_host_param_buf&& o) : h_data(o.h_data), size(o.size) {
		o.h_data = 0;
		o.size = 0;
	}

	const_host_param_buf(const std::vector<T>& v) : h_data(&v[0]), size(v.size()) {}
	const_host_param_buf(const const_param_array<T>& a) : h_data(a.data), size(a.size) {}
	~const_host_param_buf() {}
	const_param_array<T> asParamArray() const { return const_param_array<T>(h_data, size); }

	operator const T* () const { return h_data; }
};




template<typename T>
typename param_array<T> _make_array(T* ptr, size_t size, const char *dbgname) { 
	auto r = param_array<T>(ptr, size); 
	r.setDebugName(dbgname);
	return r;
}

template<typename T>
typename const_param_array<T> _make_array(const T* ptr, size_t size, const char *dbgname) { 
	auto r = const_param_array<T>(ptr, size);
	r.setDebugName(dbgname);
	return r;
}

template<typename T>
typename const_param_array<T> _const_array(const T* ptr, size_t size, const char *dbgname) { 
	auto r = const_param_array<T>(ptr, size); 
	r.setDebugName(dbgname);
	return r;
}

template<typename T>
typename param_array<T> _out_array(T* ptr, size_t size, const char *dbgname) {
	auto r = param_array<T>(ptr, size); 
	r.isOutputArray = true;
	r.setDebugName(dbgname);
	return r;
}

#define make_array(_Ptr, _Size) _make_array(_Ptr, _Size, #_Ptr)
#define const_array(_Ptr, _Size) _const_array(_Ptr, _Size, #_Ptr)
#define out_array(_Ptr, _Size) _out_array(_Ptr, _Size, #_Ptr)
#define const_vector(_Vec) _const_array((_Vec).data(), (_Vec).size(), #_Vec)

// Default
template<bool cuda, typename T>
struct get_param_arg
{
	typedef T type;
};
template<typename T>
struct pass_to_kernel {
	//	static_assert(std::is_pod<T>::value || std::is_fundamental<T>::value, "Can only pass primitive/POD types or std::vector's");
	typedef T type;
	static T pass(const T& v) { return v; }
};

#ifdef PALALA_CUDA
// std::vector
template<typename T>
struct get_param_arg<true, std::vector<T> >
{
	typedef device_param_buf<T> type;
};
template<typename T>
struct get_param_arg<true, DeviceArray<T> >
{
	typedef device_param_buf<T> type;
};
template<typename T>
struct get_param_arg<true, const std::vector<T> >
{
	typedef const_device_param_buf<T> type;
};

template<typename T>
struct get_param_arg<true, param_array<T> >
{
	typedef device_param_buf<T> type;
};
template<typename T>
struct get_param_arg<true, const_param_array<T> >
{
	typedef const_device_param_buf<T> type;
};
template<typename T>
struct get_param_arg<true, const param_array<T> >
{
	typedef const_device_param_buf<T> type;
};


template<typename T>
struct pass_to_kernel<device_param_buf<T>> {
	typedef param_array<T> type;
	static param_array<T> pass(device_param_buf<T>& v) { return v.kernelParam(); }
};
template<typename T>
struct pass_to_kernel<const_device_param_buf<T>> {
	typedef const_param_array<T> type;
	static const_param_array<T> pass(const const_device_param_buf<T>& v) { return v.kernelParam(); }
};
#endif
template<typename T>
struct get_param_arg<false, std::vector<T> >
{
	typedef host_param_buf<T> type;
};
template<typename T>
struct get_param_arg<false, const std::vector<T> >
{
	typedef const_host_param_buf<T> type;
};
template<typename T>
struct get_param_arg<false, param_array<T> >
{
	typedef host_param_buf<T> type;
};
template<typename T>
struct get_param_arg<false, const_param_array<T> >
{
	typedef const_host_param_buf<T> type;
};

template<typename T>
struct pass_to_kernel<host_param_buf<T>> {
	typedef param_array<T> type;
	static param_array<T> pass(const host_param_buf<T>& v) { return v.asParamArray();}
};

template<typename T>
struct pass_to_kernel<const_host_param_buf<T>> {
	typedef const_param_array<T> type;
	static const_param_array<T> pass(const const_host_param_buf<T>& v) { return v.asParamArray(); }
};
template<typename T>
typename pass_to_kernel<T>::type _pass_to_kernel(T& v) {
	return pass_to_kernel<T>::pass(v);
}
template<typename T>
typename pass_to_kernel<const T>::type _pass_to_kernel(const T& v) {
	return pass_to_kernel<const T>::pass(v);
}

template<typename T>
auto convert_arg(T& v) -> typename get_param_arg<false, T>::type {
	return get_param_arg<false, T>::type(v);
}
template<typename T>
auto convert_arg(T&& v) -> typename get_param_arg<false, T>::type {
	return get_param_arg<false, T>::type(v);
}

template<typename T>
auto convert_arg(const T& v) -> typename get_param_arg<false, const T>::type {
	return get_param_arg<false, const T>::type(v);
}

template<typename Function, typename Tuple, std::size_t ...I>
void call_func(int nx, bool singleThread, Function f, const Tuple& t, std::index_sequence<I...>)
{
	if (singleThread) {
		for (int i = 0; i < nx; i++)
			f(i, (_pass_to_kernel(std::get<I>(t))) ...);
	}
	else {
		std::vector< std::future<void> > futures(nx);

		for (int i = 0; i < nx; i++)
			futures[i] = std::async(f, i, (_pass_to_kernel(std::get<I>(t))) ...);

		for (auto& e : futures)
			e.get();
	}
}

template<typename Function, typename Tuple, std::size_t ...I>
void call_func(int nx, int ny, bool singleThread, Function f, Tuple& t, std::index_sequence<I...>)
{
	if (singleThread) {
		for (int x = 0; x < nx; x++)
			for (int y = 0; y < ny; y++)
				f(x, y, (_pass_to_kernel(std::get<I>(t))) ...);
	}
	else {
		std::vector< std::future<void> > futures(nx*ny);

		for (int x = 0; x < nx; x++)
			for (int y = 0; y < ny; y++) 
				futures[x*ny+y] = std::async(f, x, y, (_pass_to_kernel(std::get<I>(t))) ...);

		for (auto& e : futures)
			e.get();
	}
}

template<typename Function, typename... Args>
void palala_for_cpu(int nx, Function f, Args&&... args) {
	call_func(nx, false, f, std::make_tuple(convert_arg(std::forward<Args>(args))...), std::index_sequence_for<Args...>{});
}

template<typename Function, typename... Args>
void singlethread_for(int nx, Function f, Args&&... args) {
	call_func(nx, true, f, std::make_tuple(convert_arg(std::forward<Args>(args))...), std::index_sequence_for<Args...>{});
}


template<typename Function, typename... Args>
void palala_for_cpu(int nx, int ny, Function f, Args&&... args) {
	call_func(nx, ny, false, f, std::make_tuple(convert_arg(std::forward<Args>(args))...), std::index_sequence_for<Args...>{});
}

template<typename Function, typename... Args>
void singlethread_for(int nx, int ny, Function f, Args&&... args) {
	call_func(nx, ny, true, f, std::make_tuple(convert_arg(std::forward<Args>(args))...), std::index_sequence_for<Args...>{});
}

#ifdef PALALA_CUDA

template <typename Function, typename... Arguments>
__global__ void Kernel1D(int n, Function f, Arguments... args)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	if (x<n) f(x, args...);
}
template <typename Function, typename... Arguments>
__global__ void Kernel2D(int nx, int ny, Function f, Arguments... args)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x < nx && y < ny) f(x, y, args...);
}


template<typename Function, typename Tuple, std::size_t ...I>
void call_kernel(int n, Function f, Tuple& t, std::index_sequence<I...>)
{
	dim3 numThreads(128);
	dim3 numBlocks((n + numThreads.x - 1) / numThreads.x);
	Kernel1D << < numBlocks, numThreads >> > (n, f, _pass_to_kernel(std::get<I>(t))...);
	ThrowIfCUDAError();
	cudaDeviceSynchronize();
}

template<typename Function, typename Tuple, std::size_t ...I>
void call_kernel(int nx, int ny, Function f, Tuple& t, std::index_sequence<I...>)
{
	dim3 numThreads(16, 16);
	dim3 numBlocks((nx + numThreads.x - 1) / numThreads.x, (ny + numThreads.y - 1) / numThreads.y);
	Kernel2D << < numBlocks, numThreads >> > (nx, ny, f, _pass_to_kernel(std::get<I>(t))...);
	ThrowIfCUDAError();
	cudaDeviceSynchronize();
}

template<typename T>
auto cuda_convert_arg(T&& v) -> typename get_param_arg<true, T>::type {
	//DebugPrintf("cuda_convert_arg(&v) with type v=%s -> converted to %s\n", typeid(T).name(), typeid(get_param_arg<true, T>::type).name());
	return get_param_arg<true, T>::type(v);
}
template<typename T>
auto cuda_convert_arg(T& v) -> typename get_param_arg<true, T>::type {
	//DebugPrintf("cuda_convert_arg(&v) with type v=%s -> converted to %s\n", typeid(T).name(), typeid(get_param_arg<true, T>::type).name());
	return get_param_arg<true, T>::type(v);
}
template<typename T>
auto cuda_convert_arg(const T& v) -> typename get_param_arg<true, const T>::type {
	//DebugPrintf("cuda_convert_arg(const &v) with type v=%s -> converted to %s\n", typeid(T).name(), typeid(get_param_arg<true, T>::type).name());
	return get_param_arg<true, const T>::type(v);
	//return get_param_arg<true, const T&>::type(v);
}


template<typename Function, typename... Args>
void palala_for_cuda(int nx, Function f, Args&&... args) {
	auto argtuple = std::make_tuple(cuda_convert_arg(std::forward<Args>(args))...);
	call_kernel(nx, f, argtuple, std::index_sequence_for<Args...>{});
}
template<typename Function, typename... Args>
void palala_for_cuda(int nx, int ny, Function f, Args&&... args) {
	auto argtuple = std::make_tuple(cuda_convert_arg(std::forward<Args>(args))...);
	call_kernel(nx, ny, f, argtuple, std::index_sequence_for<Args...>{});
}

#endif


template <typename Function, typename... Args>
void palala_for(int nx, bool useCuda, Function f, Args&&... args)
{
#ifdef PALALA_CUDA
	if (useCuda)
		palala_for_cuda(nx, f, std::forward<Args>(args)...);
	else {
#ifdef _DEBUG
		singlethread_for(nx, f, std::forward<Args>(args)...);
#else
		palala_for_cpu(nx, f, std::forward<Args>(args)...);
#endif
	}
#else
	if (useCuda)
		throw std::runtime_error("Trying to use CUDA in a non-CUDA build");
	palala_for_cpu(nx, f, std::forward<Args>(args)...);
#endif
}


template <typename Function, typename... Args>
void palala_for(int nx, int ny, bool useCuda, Function f, Args&&... args)
{
#ifdef PALALA_CUDA
	if (useCuda)
		palala_for_cuda(nx, ny, f, std::forward<Args>(args)...);
	else {
#ifdef _DEBUG
		singlethread_for(nx, ny, f, std::forward<Args>(args)...);
#else
		palala_for_cpu(nx, ny, f, std::forward<Args>(args)...);
#endif
	}
#else
	if (useCuda)
		throw std::runtime_error("Trying to use CUDA in a non-CUDA build");
	palala_for_cpu(nx, f, std::forward<Args>(args)...);
#endif
}

#ifdef PALALA_CUDA
#define PLL_FN [=] __device__ __host__
#else
#define PLL_FN [=]
#endif

#endif