// Use Kahan Sum for reduced FPU summing errors
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once
#include "palala.h"

template<typename T>
class KahanSum {
public:
	T sum, c;

	KahanSum(T initial = {}) : sum(initial), c{} {}

	PLL_DEVHOST void operator +=(const T& o) {
		T y = o - c;
		T t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	PLL_DEVHOST T operator()() { return sum; }

	PLL_DEVHOST operator T() { return sum; }
};

