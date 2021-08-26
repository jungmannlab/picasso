// String utils
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include <string>
#include <vector>

/*
* Somewhat portable definition for SNPRINTF, VSNPRINTF, STRCASECMP and STRNCASECMP
*/
#ifdef _MSC_VER
#if _MSC_VER > 1310
#define SNPRINTF _snprintf_s
#define VSNPRINTF _vsnprintf_s
#else
#define SNPRINTF _snprintf
#define VSNPRINTF _vsnprintf
#endif
#define STRCASECMP _stricmp
#define STRNCASECMP _strnicmp
#define ALLOCA(size) _alloca(size) // allocates memory on stack
#else
#define STRCASECMP strcasecmp
#define STRNCASECMP strncasecmp
#define SNPRINTF snprintf
#define VSNPRINTF vsnprintf
#define ALLOCA(size) alloca(size)
#endif

#include "dllmacros.h"

DLL_EXPORT std::string SPrintf(const char *fmt, ...);
CDLL_EXPORT void DebugPrintf(const char*fmt, ...);
CDLL_EXPORT void SetDebugPrintCallback(int(*cb)(const char* msg));
DLL_EXPORT std::vector<std::string> StringSplit(const std::string& str, char sep);

// FindIndexInSplitString("x,y,z", "y", ',') will return 1 (index of Y in comma seperated list)
CDLL_EXPORT int FindIndexInSplitString(const char* str, const char* item, char seperator);

template<typename T>
void printMatrix(T* m, int rows, int cols = -1, const char *fmt = "%.3f") {
	if (cols < 0) cols = rows;
	for (int i = 0; i < rows; i++) {
		if (i == 0)
			DebugPrintf("[[");
		else
			DebugPrintf(" [");
		for (int j = 0; j < cols; j++) {
			DebugPrintf(fmt, m[i*cols + j]);
			if (j < cols - 1)
				DebugPrintf(",");
		}
		if (i == rows - 1)
			DebugPrintf("]]");
		else
			DebugPrintf("],");
		DebugPrintf("\n");
	}
}
template<typename T>
void PrintVector(T* v, int d, const char* fmt = " %.3f") {
	for (int i = 0; i < d; i++)
		DebugPrintf(fmt, v[i]);
	DebugPrintf("\n");
}

