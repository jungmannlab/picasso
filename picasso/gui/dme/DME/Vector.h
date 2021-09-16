// N-dimensional vector class that implements all the basic operators
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "palala.h"


// Compile-time square root
//--expt-relaxed-constexpr
constexpr int CompileTimeSqrt(int n, int i = 1) {
	return n == i ? n : (i * i < n ? CompileTimeSqrt(n, i + 1) : i);
}


template<typename T, int Size>
struct Vector {
	enum { K = Size };
	enum { size = Size };
	T elem[Size];
	typedef T TElem;
	typedef T* iterator;
	typedef T value_type;

	PLL_DEVHOST Vector() {
		for (int i = 0; i < Size; i++)
			elem[i] = T{};
	}
	PLL_DEVHOST Vector(const std::initializer_list<T>& l) {
		int i = 0;
		for(const auto& e : l)
			elem[i++] = e;
	}
	Vector(const std::vector<T>& l) {
		if (l.size() != Size)
			throw std::runtime_error(SPrintf("%s(std::vector<>) is given vector with %d elements", typeid(*this).name(), Size).c_str());
		int i = 0;
		for (const auto& e : l)
			elem[i++] = e;
	}

	PLL_DEVHOST static Vector<T, Size> ones() {
		Vector<T, Size> r;
		for (int i = 0; i < Size; i++) r[i] = 1.0f;
		return r;
	}

	template <typename ...Args>
	explicit PLL_DEVHOST constexpr Vector(const Args&... args) : elem{ args... } {
	}

	PLL_DEVHOST Vector(T(&v)[Size]) {
		for (int i = 0; i < Size; i++)
			elem[i] = v[i];
	}
	template<typename T2>
	explicit PLL_DEVHOST Vector(const Vector<T2,Size> &v) {
		for (int i = 0; i < Size; i++)
			elem[i] = (T)v[i];
	}

	PLL_DEVHOST Vector operator-() const {
		Vector v;
		for (int i = 0; i < K; i++)
			v[i] = -elem[i];
		return v;
	}

	template<typename T2>
	PLL_DEVHOST Vector< decltype(T()+T2()), Size > operator+(Vector<T2, Size> v) const {
		Vector< decltype(T() + T2()), Size > r;
		for (int i = 0; i < K; i++)
			r.elem[i] = elem[i] + v[i];
		return r;
	}
	template<typename T2>
	PLL_DEVHOST Vector< decltype(T() / T2()), Size > operator/(Vector<T2, Size> v) const {
		Vector< decltype(T() / T2()), Size > r;
		for (int i = 0; i < K; i++)
			r.elem[i] = elem[i] / v[i];
		return r;
	}
	template<typename T2>
	PLL_DEVHOST Vector< decltype(T() / T2()), Size > operator/(T2 v) const {
		Vector< decltype(T() / T2()), Size > r;
		for (int i = 0; i < K; i++)
			r.elem[i] = elem[i] / v;
		return r;
	}
	template<typename T2>
	PLL_DEVHOST Vector< decltype(T() - T2()), Size > operator-(Vector<T2, Size> v) const {
		Vector< decltype(T() - T2()), Size > r;
		for (int i = 0; i < K; i++)
			r.elem[i] = elem[i] - v[i];
		return r;
	}
	template<typename T2>
	PLL_DEVHOST Vector< decltype(T() * T2()), Size > operator*(Vector<T2, Size> v) const {
		Vector< decltype(T()* T2()), Size > r;
		for (int i = 0; i < K; i++)
			r.elem[i] = elem[i] * v[i];
		return r;
	}
	template<typename T2>
	PLL_DEVHOST Vector< decltype(T() * T2()), Size > operator*(T2 v) const {
		Vector< decltype(T()* T2()), Size > r;
		for (int i = 0; i < K; i++)
			r.elem[i] = elem[i] * v;
		return r;
	}
	/*
	template<typename T2>
	friend PLL_DEVHOST Vector< decltype(T2()* T()), Size > operator*(T2 a, T b) {
		Vector r;
		for (int i = 0; i < K; i++)
			r.elem[i] = a * b.elem[i];
		return r;
	}*/
	template<typename T2>
	PLL_DEVHOST Vector< decltype(T() + T2()), Size > operator+(T2 v) const {
		Vector< decltype(T() + T2()), Size > r;
		for (int i = 0; i < K; i++)
			r.elem[i] = elem[i] + v;
		return r;
	}
	template<typename T2>
	PLL_DEVHOST Vector< decltype(T() + T2()), Size > operator-(T2 v) const {
		Vector< decltype(T() + T2()), Size > r;
		for (int i = 0; i < K; i++)
			r.elem[i] = elem[i] - v;
		return r;
	}

	PLL_DEVHOST T& operator[](int i)
	{
		return elem[i];
	}
	PLL_DEVHOST const T& operator[](int i) const
	{
		return elem[i];
	}
	template<typename T2>
	PLL_DEVHOST Vector& operator+=(Vector<T2, Size> v) {
		for (int i = 0; i < K; i++)
			elem[i] += v[i];
		return *this;
	}
	template<typename T2>
	PLL_DEVHOST Vector& operator-=(Vector<T2, Size> v) {
		for (int i = 0; i < K; i++)
			elem[i] -= v[i];
		return *this;
	}
	template<typename T2>
	PLL_DEVHOST Vector& operator*=(Vector<T2, Size> v) {
		for (int i = 0; i < K; i++)
			elem[i] *= v[i];
		return *this;
	}
	PLL_DEVHOST Vector& operator*=(T v) {
		for (int i = 0; i < K; i++)
			elem[i] *= v;
		return *this;
	}
	template<typename T2>
	PLL_DEVHOST Vector& operator/=(Vector<T2, Size> v) {
		for (int i = 0; i < K; i++)
			elem[i] /= v[i];
		return *this;
	}
	template<typename T2>
	PLL_DEVHOST Vector& operator/=(T2 v) {
		for (int i = 0; i < K; i++)
			elem[i] /= v;
		return *this;
	}

	PLL_DEVHOST T sum() const {
		T s = elem[0];
		for (int i = 1; i < K; i++)
			s += elem[i];
		return s;
	}

	PLL_DEVHOST T prod() const {
		T p = elem[0];
		for (int i = 1; i < K; i++)
			p *= elem[i];
		return p;
	}

	PLL_DEVHOST T sqLength() const {
		T sum{};
		for (int i = 0; i < K; i++)
			sum += elem[i] * elem[i];
		return sum;
	}
	PLL_DEVHOST auto length() -> float const {
		return sqrtf((float)sqLength());
	}

	PLL_DEVHOST T normalize() {
		T len = length();
		*this /= len;
		return len;
	}

	PLL_DEVHOST Vector normalized() const {
		Vector n = *this;
		n.normalize();
		return n;
	}

	PLL_DEVHOST bool hasNan() const {
		for (int i = 0; i < K; i++)
			if (isnan(elem[i])) return true;
		return false;
	}

	PLL_DEVHOST void setInf() {
		for (int i = 0; i < K; i++)
			elem[i] = INFINITY;
	}

	template<int SliceSize>
	PLL_DEVHOST Vector<T, SliceSize> slice(int startIndex=0) const
	{
		Vector<T, SliceSize> r;
		for (int i = 0; i < SliceSize; i++)
			r[i] = elem[startIndex + i];
		return r;
	}

	PLL_DEVHOST Vector<T, CompileTimeSqrt(K)> diagonal() const
	{
		Vector<T, CompileTimeSqrt(K)> r;
		for (int i = 0; i < r.K; i++)
			r[i] = elem[i*(1 + r.K)];
		return r;
	}

	template<typename TFunc>
	PLL_DEVHOST auto apply(TFunc f) const -> Vector<decltype(f(T())), Size> {
		Vector<decltype(f(T())), Size> r;
		for (int i = 0; i < Size; i++)
			r[i] = f(elem[i]);
		return r;
	}

	PLL_DEVHOST Vector<T, Size> abs() const {
		return apply([](T e) {return ::abs(e); });
	}

	PLL_DEVHOST Vector<T, Size> sqrt() const {
		return apply([](T e) {return (T)::sqrt((T)e); });
	}
	PLL_DEVHOST Vector<T, Size> floor() const {
		return apply([](T e) {return ::floor(e); });
	}
	PLL_DEVHOST Vector<T, Size> ceil() const {
		return apply([](T e) {return ::ceil(e); });
	}

	PLL_DEVHOST friend Vector<T, Size>  operator/(T a, const Vector& x) {
		Vector<T,Size> r;
		for (int i = 0; i < Size; i++)
			r.elem[i] = a / x.elem[i];
		return r;
	}

	PLL_DEVHOST Vector<T, 2> xy() const {
		return Vector<T, 2>{elem[0], elem[1]};
	}

	const T* begin() const { return elem; }
	const T* end() const { return elem + K; }

	PLL_DEVHOST T max() const {
		T m = elem[0];
		for (int i = 1; i < Size; i++)
			if (elem[i] > m) m = elem[i];
		return m;
	}
	PLL_DEVHOST T min() const {
		T m = elem[0];
		for (int i = 1; i < Size; i++)
			if (elem[i] < m) m = elem[i];
		return m;
	}
	template<typename T2>
	PLL_DEVHOST Vector<T, Size> maximum(const Vector<T2, Size>& other) {
		Vector<T, Size> r;
		for (int i = 0; i < Size; i++)
			r[i] = other[i] > elem[i] ? other[i] : elem[i];
		return r;
	}
	template<typename T2>
	PLL_DEVHOST Vector<T, Size> minimum(const Vector<T2, Size>& other) {
		Vector<T, Size> r;
		for (int i = 0; i < Size; i++)
			r[i] = other[i] < elem[i] ? other[i] : elem[i];
		return r;
	}
	PLL_DEVHOST Vector<T, Size> reverse() const {
		Vector<T, Size> r;
		for (int i = 0; i < Size; i++)
			r[Size - 1 - i] = elem[i];
		return r;
	}
};

typedef Vector<float, 2> Vector2f;
typedef Vector<float, 3> Vector3f;
typedef Vector<float, 4> Vector4f;
typedef Vector<float, 5> Vector5f;
typedef Vector<float, 6> Vector6f;
typedef Vector<double, 2> Vector2d;
typedef Vector<double, 3> Vector3d;
typedef Vector<double, 4> Vector4d;
typedef Vector<double, 5> Vector5d;
typedef Vector<double, 6> Vector6d;
typedef Vector<int, 2> Int2;
typedef Vector<int, 3> Int3;
typedef Vector<int, 4> Int4;
typedef Vector<int, 5> Int5;
typedef Vector<int, 6> Int6;


typedef Vector<Vector2f, 2> Matrix22f;
typedef Vector<Vector3f, 3> Matrix33f;
typedef Vector<Vector4f, 4> Matrix44f;


template<typename T, int s>
void PrintVector(const Vector<T, s>& v) {
	for (int i = 0; i < s; i++)
		DebugPrintf(" %.3f", v[i]);
	DebugPrintf("\n");
}

template<typename T, int S>
Vector<T, S> ToVector(T(&a)[S]) {
	return Vector<T, S>(a);
}

template<typename T, int S>
Vector<T, S> log(const Vector<T, S>& a) {
	Vector<T, S> r;
	for (int i = 0; i < S; i++)
		r[i] = log(a[i]);
	return r;
}

