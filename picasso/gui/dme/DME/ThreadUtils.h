// C++ multithread helper code
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include <mutex>
#include <atomic>
#include <future>

template<typename TAction>
void LockedAction(std::mutex& m, TAction fn) {
	std::lock_guard<std::mutex> l(m);
	fn();
}
template<typename TFunc>
auto LockedFunction(std::mutex& m, TFunc fn) -> decltype(fn()) {
	std::lock_guard<std::mutex> l(m);
	return fn();
}

template<typename Function>
void ParallelFor(int n, Function f)
{
	std::vector< std::future<void> > futures(n);

	for (int i = 0; i < n; i++)
		futures[i] = std::async(std::launch::async, f, i);

	for (auto& e : futures)
		e.get();
}
