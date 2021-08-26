// String utils
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "StringUtils.h"
#include <cstdarg>
#include <Windows.h>

#include <mutex>

std::mutex printMutex;
int (*debugPrintCallback)(const char *msg)=0;

DLL_EXPORT std::string SPrintf(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);

	va_end(ap);
	return buf;
}

CDLL_EXPORT void DebugPrintf(const char* fmt, ...) {
	std::lock_guard<std::mutex> l(printMutex);

	va_list ap;
	va_start(ap, fmt);
	char buf[1024];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	bool print = true;
	if (debugPrintCallback)
		print = !!debugPrintCallback(buf);

	if (print) {
		OutputDebugString(buf);
		fputs(buf, stdout);
	}
}

CDLL_EXPORT void SetDebugPrintCallback(int(*cb)(const char *msg))
{
	std::lock_guard<std::mutex> l(printMutex);
	debugPrintCallback = cb;
}


DLL_EXPORT std::vector<std::string> StringSplit(const std::string& str, char sep)
{
	std::vector<std::string> r;
	std::string cur;
	for (int x = 0; x<str.size(); x++) {
		if (str[x] == sep) {
			r.push_back(cur);
			cur = "";
		}
		else {
			cur += str[x];
		}
	}
	if (cur.size()>0)
		r.push_back(cur);
	return r;
}

CDLL_EXPORT int FindIndexInSplitString(const char * str, const char * item, char seperator)
{
	// TODO: fix this, this fails when doing complicated labels like FindIndexInSplitString("abc, ab", "ab", ','). 
	// Should return 1 but returns 0

	const char *found = strstr(str, item);
	if (!found) return -1;

	size_t pos = found - str;
	size_t i = 0;
	int n = 0;
	while (i<pos) {
		if (str[i] == seperator)
			n++;
		i++;
	}
	return n;
}

