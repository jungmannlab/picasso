// Localization rendering utils
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021

#include "DLLMacros.h"
#include <math.h>


// spotList [ x y sigmaX sigmaY intensity ]
CDLL_EXPORT void Gauss2D_Draw(float * image, int imgw, int imgh, float * spotList, int nspots)
{
	auto squared = [](double x) { return x * x; };

	for (int i = 0; i < nspots; i++) {
		float* spot = &spotList[5 * i];
		double sigmaScale = 4.0f; 
		double hwx = spot[2] * sigmaScale;
		double hwy = spot[3] * sigmaScale;
		int minx = int(spot[0] - hwx), miny = int(spot[1] - hwy);
		int maxx = int(spot[0] + hwx + 1), maxy = int(spot[1] + hwy + 1);
		if (minx < 0) minx = 0;
		if (miny < 0) miny = 0;
		if (maxx > imgw - 1) maxx = imgw - 1;
		if (maxy > imgh - 1) maxy = imgh - 1;

		double _1o2sxs = 1.0f / (sqrt(2.0f) * spot[2]);
		double _1o2sys = 1.0f / (sqrt(2.0f) * spot[3]);
		for (int y = miny; y <= maxy; y++) {
			for (int x = minx; x <= maxx; x++) {
				float& pixel = image[y*imgw + x];
				pixel += spot[4] * exp(-(squared((x - spot[0])*_1o2sxs) + squared((y - spot[1])*_1o2sys))) / (2 * 3.141593f*spot[2] * spot[3]);
			}
		}
	}
}




