// Localization rendering utils
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "DLLMacros.h"
#include "Vector.h"

CDLL_EXPORT void DrawROIs(float* image, int width, int height, const float* rois, int numrois, int roisize, Int2* roiposYX);

