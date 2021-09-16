// DeviceArray and PinnedArray classes
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdarg>
#include "StringUtils.h"
#include "CudaMath.h"
#include <stdexcept>


#ifndef PLL_DEVHOST
	#if __CUDACC__
		#define PLL_DEVHOST __device__ __host__
	#else
		#define PLL_DEVHOST
	#endif
#endif

inline void ThrowIfCUDAError(cudaError_t err)
{
	if (err != cudaSuccess) {
		const char* errstr = cudaGetErrorString(err);
		throw std::runtime_error(SPrintf("CUDA error: %s\n" ,errstr).c_str());
	}
}

inline void ThrowIfCUDAError()
{
	cudaError_t err = cudaGetLastError();
	ThrowIfCUDAError(err);
}

#ifdef _DEBUG
inline void CUDAErrorCheck(cudaError_t e) { ThrowIfCUDAError(e); }
#else
inline void CUDAErrorCheck(cudaError_t e) {}
#endif

template<typename T>
class DeviceArray {
public:
	DeviceArray() {
		d = 0;
		n = 0;
	}

	DeviceArray(size_t N) { 
		d = 0;
		n = 0;
		Init(N);
	}
	DeviceArray(DeviceArray&& o) {
		d = o.d;
		n = o.n;
		o.d = 0;
		o.n = 0;
	}
	DeviceArray(size_t N, const T* src) {
		d = 0; n = 0;
		Init(N);
		if (N > 0 && src)
			CUDAErrorCheck(cudaMemcpy(d, src, sizeof(T)*n, cudaMemcpyHostToDevice));
	}
	DeviceArray(const DeviceArray<T>& src) {
		d = 0; n = 0;
		Init(src.n);
		if (n>0)
			CUDAErrorCheck(cudaMemcpy(d, src.d, sizeof(T)*n, cudaMemcpyDeviceToDevice));
	}
	DeviceArray(const std::vector<T>& src) {
		d=0; n=0; 
		Init(src.size());
		if (n>0)
			CUDAErrorCheck(cudaMemcpy(d, &src[0], sizeof(T)*n, cudaMemcpyHostToDevice));
	}
	~DeviceArray(){
		Free();
	}
	void Init(size_t s) {
		if(n != s) {
			Free();
		}
		if (s!=0) {
			CUDAErrorCheck(cudaMalloc(&d, sizeof(T)*s));
			n = s;
		}
	}
	void Free() {
		if (d) {
			CUDAErrorCheck(cudaFree(d));
			d=0;
		}
	}
	operator std::vector<T>() const {
		std::vector<T> dst(n);
		if (n == 0) return dst;
		CUDAErrorCheck(cudaMemcpy(&dst[0], d, sizeof(T)*n, cudaMemcpyDeviceToHost));
		return dst;
	}
	T* begin() { return d; }
	T* end() { return d + n; }
	const T* begin() const { return d; }
	const T* end() const { return d + n; }

	DeviceArray<T>& operator=(const std::vector<T>& src) {
		Init(src.size());
		if(n>0) 
			CUDAErrorCheck(cudaMemcpy(d, &src[0], sizeof(T)*n, cudaMemcpyHostToDevice));
		return *this;
	}
	DeviceArray<T>& operator=(const DeviceArray<T>& src) {
		Init(src.n);
		if(src.n>0)
			CUDAErrorCheck(cudaMemcpy(d, src.d, sizeof(T)*n, cudaMemcpyDeviceToDevice));
		return *this;
	}
	void CopyToHost(T* dst, bool async=false, cudaStream_t s=0) const {
		if (n > 0) {
			if (async)
				CUDAErrorCheck(cudaMemcpyAsync(dst, d, sizeof(T) * n, cudaMemcpyDeviceToHost, s));
			else
				CUDAErrorCheck(cudaMemcpy(dst, d, sizeof(T) * n, cudaMemcpyDeviceToHost));
		}
	}
	void CopyToHost(T* dst, size_t count, bool async = false, cudaStream_t s = 0) const {
		if (count > 0) {
			if (async)
				CUDAErrorCheck(cudaMemcpyAsync(dst, d, sizeof(T) * count, cudaMemcpyDeviceToHost, s));
			else
				CUDAErrorCheck(cudaMemcpy(dst, d, sizeof(T) * count, cudaMemcpyDeviceToHost));
		}
	}
	void CopyToHost(std::vector<T>& dst ,bool async=false, cudaStream_t s=0) const {
		if (dst.size() != n)
			dst.resize(n);
		CopyToHost(&dst[0], async, s);
	}
	void CopyToDevice(const std::vector<T>& src, bool async=false, cudaStream_t s=0) {
		CopyToDevice(&src[0], src.size(), async, s);
	}
	void CopyToDevice(const T* first, size_t n, bool async=false, cudaStream_t s=0) {
		if (this->n < n)
			Init(n);
		if (n > 0) {
			if (async)
				CUDAErrorCheck(cudaMemcpyAsync(d, first, sizeof(T) * n, cudaMemcpyHostToDevice, s));
			else
				CUDAErrorCheck(cudaMemcpy(d, first, sizeof(T) * n, cudaMemcpyHostToDevice));
		}
	}
	void Clear(cudaStream_t s = 0) {
		if (s)
			CUDAErrorCheck(cudaMemsetAsync(d, 0, n * sizeof(T), s));
		else
			CUDAErrorCheck(cudaMemset(d, 0, n * sizeof(T)));
	}
	// debugging util. Be sure to synchronize before
	std::vector<T> ToVector() const {
		std::vector<T> v (n);
		CUDAErrorCheck(cudaMemcpy(&v[0], d, sizeof(T)*n, cudaMemcpyDeviceToHost));
		return v;
	}
	size_t MemSize() const { return n*sizeof(T); }
	size_t size() const { return n; }
	T* ptr() { return d;  }
	const T* ptr() const { return d; }
	T* data() { return d; }
	const T* data() const { return d; }
protected:
	size_t n;
	T* d;
};



template<typename T, int flags=0>
class PinnedArray
{
public:
	PinnedArray() {
		d=0; n=0;
	}
	~PinnedArray() {
		Free();
	}
	PinnedArray(size_t n) {
		d=0; Init(n);
	}
	PinnedArray(PinnedArray&& src) {
		d = src.d;
		n = src.n;
		src.d = 0;
		src.n = 0;
	}
	PinnedArray(const PinnedArray& src) {
		d=0;Init(src.n);
		for(int k=0;k<src.n;k++)
			d[k]=src[k];
	}
	template<typename TOther, int F>
	PinnedArray& operator=(const PinnedArray<TOther, F>& src) {
		if (src.n != n) Init(src.n);
		for(int k=0;k<src.n;k++)
			d[k]=src[k];
		return *this;
	}
	template<typename Iterator>
	PinnedArray(Iterator first, Iterator end) {
		d=0; Init(end-first);
		for (int k = 0; first != end; ++first) {
			d[k++] = *first;
		}
	}
	template<typename T>
	PinnedArray(const DeviceArray<T>& src) {
		d=0; Init(src.size()); src.CopyToHost(d,false);
	}

	void CopyFromDevice(const DeviceArray<T>& src, cudaStream_t stream) {
		src.CopyToHost(d, true, stream);
	}

	void CopyFromDevice(const T* src, size_t size, cudaStream_t stream) {
		CUDAErrorCheck(cudaMemcpyAsync(d, src, sizeof(T)*size, cudaMemcpyDeviceToHost, stream));
	}

	size_t size() const { return n; }
	T* begin() { return d; }
	T* end() { return d+n; }
	const T* begin() const { return d; }
	const T* end() const { return d+n; }
	T* data() { return d; }
	void Free() {
		CUDAErrorCheck(cudaFreeHost(d));
		d=0;n=0;
	}
	void Init(size_t n) {
		if (d) Free();
		this->n = n;
		CUDAErrorCheck(cudaMallocHost(&d, sizeof(T)*n, flags));
	}
	T& operator[](int i) {  return d[i]; }
	const T&operator[](int i) const { return d[i];}
	size_t MemSize() { return n*sizeof(T); }

protected:
	T* d;
	size_t n;
};


template<typename T>
class DeviceImage {
public:
	int width, height;
	int pitch;
	T* data;

	struct ImageIndexer
	{
		T* data;
		int pixelPitch, width;
		PLL_DEVHOST ImageIndexer offset(int x, int y) const { return { data + x + y * pixelPitch, pixelPitch, width }; }
		PLL_DEVHOST int pixelIndex(int x, int y) const { return y * pixelPitch + x; }
		PLL_DEVHOST T& pixel(int x, int y) const { return data[y*pixelPitch + x]; }
		PLL_DEVHOST T& operator()(int x, int y) const { return data[y*pixelPitch + x]; }
	};
	struct ConstImageIndexer
	{
		const T* __restrict__ data;
		int pixelPitch, width;
		PLL_DEVHOST ConstImageIndexer offset(int x, int y) const { return { data + x + y * pixelPitch, pixelPitch, width }; }
		PLL_DEVHOST int pixelIndex(int x, int y) const { return y * pixelPitch + x; }
		PLL_DEVHOST const T& pixel(int x, int y) const { return data[y*pixelPitch + x]; }
		PLL_DEVHOST const T& operator()(int x, int y) const { return data[y*pixelPitch + x]; }
	};

	ImageIndexer GetIndexer()
	{
		return ImageIndexer{ data, int(pitch/sizeof(T)), width };
	}
	ConstImageIndexer GetIndexer() const
 	{
		return ConstImageIndexer{ data, int(pitch / sizeof(T)), width };
	}
	ConstImageIndexer GetConstIndexer() const
	{
		return GetIndexer();
	}

	void Swap(DeviceImage<T>& other) {
		std::swap(other.data, data);
		std::swap(other.width, width);
		std::swap(other.height, height);
	}

	int NumPixels() const {
		return width * height;
	}

	void Init(int2 s) {
		Init(s.x, s.y);
	}

	void Init(int w, int h) {
		if (data) {
			CUDAErrorCheck(cudaFree(data));
		}
		width = w;
		height = h;
		if (NumPixels() > 0) {
			size_t p;
			CUDAErrorCheck(cudaMallocPitch(&data, &p, width*sizeof(T), height));
			pitch = (int)p;
		}
	}

	DeviceImage(int width = 0, int height = 0)
		: pitch(0), data(0), width(width), height(height)
	{
		Init(width, height);
	}

	DeviceImage(int2 size)
		: pitch(0), data(0), width(size.x), height(size.y)
	{
		Init(size);
	}

	int PitchInPixels() const
	{
		return pitch / sizeof(T);
	}

	int2 Size() const { return { width,height }; }

	// Returns a GPU-memory location
	float* Index(int x, int y)
	{
		return &data[PitchInPixels() * y + x];
	}

	DeviceImage(const DeviceImage& src) : data(0)
	{
		Init(src.width, src.height);
		CUDAErrorCheck(cudaMemcpy2D(data, pitch, src.data, src.pitch, width * sizeof(T), height, cudaMemcpyDeviceToDevice));
	}

	DeviceImage(const T* h_data, int w, int h) :
		data(0)
	{
		Init(w, h);
		CopyFromHost(h_data);
	}

	DeviceImage(DeviceImage&& src) :
		data(src.data), width(src.width), height(src.height), pitch(src.pitch)
	{
		src.data = 0;
		src.width = src.height = src.pitch=0;
	}

	~DeviceImage() {
		if (data) {
			CUDAErrorCheck(cudaFree(data));
			data = 0;
		}
	}

	void CopyFromDevice(const T* src, int src_pitch, cudaStream_t stream=0)
	{
		if (stream)
			CUDAErrorCheck(cudaMemcpy2DAsync(data, pitch, src, src_pitch, sizeof(T)*width, height, cudaMemcpyDeviceToDevice, stream));
		else
			CUDAErrorCheck(cudaMemcpy2D(data, pitch, src, src_pitch, sizeof(T)*width, height, cudaMemcpyDeviceToDevice));
	}

	void CopyFromDevice(const T* src, int src_pitch, int dstx, int dsty, int w,int h, cudaStream_t stream = 0)
	{
		if (stream)
			CUDAErrorCheck(cudaMemcpy2DAsync(&data[PitchInPixels()*dsty+dstx], pitch, src, src_pitch, sizeof(T)*w, h, cudaMemcpyDeviceToDevice, stream));
		else
			CUDAErrorCheck(cudaMemcpy2D(&data[PitchInPixels()*dsty + dstx], pitch, src, src_pitch, sizeof(T)*w, h, cudaMemcpyDeviceToDevice));
	}

	void CopyFromHost(const T* host_data, int x, int y, int w, int h, cudaStream_t stream = 0) {
		if (stream)
			CUDAErrorCheck(cudaMemcpy2DAsync(&data[y * PitchInPixels() + x], pitch, host_data, sizeof(T)*w,
				sizeof(T)*w, h, cudaMemcpyHostToDevice, stream));
		else
			CUDAErrorCheck(cudaMemcpy2D(&data[y * PitchInPixels() + x], pitch, host_data, sizeof(T)*w,
				sizeof(T)*w, h, cudaMemcpyHostToDevice));
	}
	void CopyToHost(T* host_data, int x, int y, int w, int h, cudaStream_t stream = 0) const {
		if (stream)
			CUDAErrorCheck(cudaMemcpy2DAsync(host_data, sizeof(T)*w, &data[y * PitchInPixels() + x], pitch,
				sizeof(T)*w, h, cudaMemcpyDeviceToHost, stream));
		else
			CUDAErrorCheck(cudaMemcpy2D(host_data, sizeof(T)*w, &data[y * PitchInPixels() + x], pitch,
				sizeof(T)*w, h, cudaMemcpyDeviceToHost));
	}

	void CopyFromHost(const T* host_data, cudaStream_t stream = 0) {
		if (stream)
			CUDAErrorCheck(cudaMemcpy2DAsync(data, pitch, host_data, sizeof(T)*width, sizeof(T)*width, height, cudaMemcpyHostToDevice, stream));
		else
			CUDAErrorCheck(cudaMemcpy2D(data, pitch, host_data, sizeof(T)*width, sizeof(T)*width, height, cudaMemcpyHostToDevice));
	}

	void CopyToHost(T* host_data, cudaStream_t stream = 0) const {
		if (stream)
			CUDAErrorCheck(cudaMemcpy2DAsync(host_data, sizeof(T)*width, data, pitch, sizeof(T)* width, height, cudaMemcpyDeviceToHost, stream));
		else
			CUDAErrorCheck(cudaMemcpy2D(host_data, sizeof(T)*width, data, pitch, sizeof(T)* width, height, cudaMemcpyDeviceToHost));
	}

	void Clear(cudaStream_t stream = 0) {
		if (stream)
			CUDAErrorCheck(cudaMemset2DAsync(data, pitch, 0, sizeof(T)*width, height, stream));
		else
			CUDAErrorCheck(cudaMemset2D(data, pitch, 0, sizeof(T)*width, height));
	}

	template<typename TOperator>
	void Apply(const DeviceImage<T>& other, TOperator op, cudaStream_t stream) {
		auto this_ = GetIndexer();
		auto other_ = other.GetConstIndexer();
		LaunchKernel(height, width, [=]__device__(int y, int x) {
			this_(x, y) = op(this_(x, y), other_(x, y));
		}, 0, stream);
	}

	std::vector<T> AsVector() const {
		std::vector<T> img(width*height);
		//cudaStreamSynchronize(0);
		cudaDeviceSynchronize();
		CopyToHost(&img[0]);
		return img;
	}
};

template<typename T>
class DeviceImageStack
{
public:
	DeviceImage<T> image;
	int height; // height for a single image in the stack

	DeviceImageStack(int w, int h, int count) : image(w, h*count),height(h) {}
	DeviceImageStack() : height(0) {}
	DeviceImageStack(DeviceImageStack&& s) : image(std::move(s.image)), height(s.height) {
	}

	struct ImageIndexer
	{
		T* data;
		int pixelPitch, width;
		PLL_DEVHOST ImageIndexer offset(int x, int y, int z) const { return { data + x + (y+z*height) * pixelPitch, pixelPitch, width }; }
		PLL_DEVHOST int pixelIndex(int x, int y, int z) const { return (y + z * height) * pixelPitch + x; }
		PLL_DEVHOST T& pixel(int x, int y, int z) const { return data[(y + z * height)*pixelPitch + x]; }
	};
	struct ConstImageIndexer
	{
		const T* __restrict__ data;
		int pixelPitch, width;
		PLL_DEVHOST ConstImageIndexer offset(int x, int y, int z) const { return { data + x + (y + z * height) * pixelPitch, pixelPitch, width }; }
		PLL_DEVHOST int pixelIndex(int x, int y, int z) const { return (y + z * height) * pixelPitch + x; }
		PLL_DEVHOST const T& pixel(int x, int y, int z) const { return data[(y + z * height)*pixelPitch + x]; }
	};

	ImageIndexer GetIndexer() { return image.GetIndexer(); }
	ConstImageIndexer GetIndexer() const { return image.GetConstIndexer(); }
	ConstImageIndexer GetConstIndexer() const { return image.GetConstIndexer(); }
};

template <typename Function>
__global__ void CUDA_LambdaKernel(int n, Function f)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	if (x<n) f(x);
}

template<typename Function>
void LaunchKernel(int n, Function f, int sharedMemorySize=0, cudaStream_t stream=0, int numThreads=256)
{
	dim3 numThreads_(numThreads);
	dim3 numBlocks((n + numThreads - 1) / numThreads);
	CUDA_LambdaKernel << < numBlocks, numThreads_, sharedMemorySize, stream >> > (n, f);
	ThrowIfCUDAError();
}


template <typename Function>
__global__ void CUDA_LambdaKernel2D(int nx, int ny, Function f)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x<nx && y<ny) f(x,y);
}

template<typename Function>
void LaunchKernel(int nx, int ny, Function f, int sharedMemorySize = 0, cudaStream_t stream = 0, dim3 numThreads = { 16,16 })
{
	dim3 numBlocks((nx + numThreads.x - 1) / numThreads.x, (ny + numThreads.y - 1) / numThreads.y);
	CUDA_LambdaKernel2D <<< numBlocks, numThreads, sharedMemorySize, stream >>> (nx, ny, f);

	ThrowIfCUDAError(cudaGetLastError());
}


template <typename Function>
__global__ void CUDA_LambdaKernel3D(int nx, int ny, int nz, Function f)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x<nx && y<ny && z<nz) f(x, y, z);
}

template<typename Function>
void LaunchKernel(int nx, int ny, int nz, Function f, int sharedMemorySize = 0, cudaStream_t stream = 0)
{
	dim3 numThreads(16, 16, 2);
	dim3 numBlocks(
		(nx + numThreads.x - 1) / numThreads.x,
		(ny + numThreads.y - 1) / numThreads.y,
		(nz + numThreads.z - 1) / numThreads.z
	);
	CUDA_LambdaKernel3D <<< numBlocks, numThreads, sharedMemorySize, stream >>> (nx, ny, nz, f);

	ThrowIfCUDAError(cudaGetLastError());
}


template<typename T, bool UseCuda>
class GetVectorType {};

template<typename T>
class GetVectorType<T, true> {
public:
	typedef DeviceArray<T> type;
};

template<typename T>
class GetVectorType<T, false> {
public:
	typedef std::vector<T> type;
};


