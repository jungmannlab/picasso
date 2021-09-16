// Macros to define DLL API entry points
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#define DLL_CALLCONV __cdecl
#ifdef DME_EXPORTS
	#define DLL_EXPORT __declspec(dllexport) 
#else
	#define DLL_EXPORT __declspec(dllimport)
#endif

// Support C for matlab imports
#ifdef __cplusplus
#define CDLL_EXPORT extern "C" DLL_EXPORT
#else
#define CDLL_EXPORT DLL_EXPORT
#endif
