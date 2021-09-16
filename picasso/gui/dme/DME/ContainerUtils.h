// Some STL container helper code
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once


template<typename T>
void DeleteAll(T& container)
{
	for (auto& i : container)
		delete i;
}

template<typename T, typename Fn>
auto Transform(const std::vector<T>& v, Fn fn) -> std::vector<decltype(fn(v[0]))>
{
	std::vector<decltype(fn(v[0]))> r;
	r.reserve(v.size());
	for (auto x : v)
		r.push_back(fn(x));
	return r;
}

inline std::vector<int> Range(int a) {
	std::vector<int> r(a);
	for (int i = 0; i < a; i++)
		r[i] = i;
	return r;
}

template<typename T>
std::vector<T> Linspace(T a, T b, int count) {
	std::vector<T> r(count);
	T step = (b - a) / (count-1);
	for (int i = 0; i < count; i++)
		r[i] = a + step * i;
	return r;
}

template<typename T>
T ArraySum(const std::vector<T>& a) {
	T sum = {};
	for (auto& x : a) sum += x;
	return sum;
}

template<typename TDst, typename TSrc>
std::vector<TDst> Cast(const std::vector<TSrc>& a) {
	std::vector<TDst> dst(a.size());
	for (size_t i = 0; i < a.size(); i++) dst[i] = (TDst)a[i];
	return dst;
}
