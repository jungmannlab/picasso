// Template K-d tree class for Drift estimation using entropy minimization (DME)
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include <vector>
#include <list>
#include <numeric>
#include <algorithm>
#include <array>
#include "Vector.h"

template<typename T, int D>
class KDTree {
public:
	typedef Vector<T, D> Point;

	KDTree(const std::vector<Point>& pts, int maxPointsPerLeaf) {
		std::vector<int> indices(pts.size());
		std::iota(indices.begin(), indices.end(), 0);
		Build(pts, indices, maxPointsPerLeaf);
	}

	KDTree(const std::vector<Point>& pts, const std::vector<int>& indices, int maxPointsPerLeaf) {
		Build(pts, indices, maxPointsPerLeaf);
	}

	std::vector<int> GetPointsInEllipsoid(Point center, Point radius) {
		std::vector<int> list;
		AddPointsInEllipsoidToList(center, radius, list);
		return list;
	}

	void AddPointsInEllipsoidToList(Point center, Point radius, std::vector<int>& dst, int maxPtCount) {
		if (maxPtCount > 0 && dst.size() >= maxPtCount)
			return;

		if (indices.empty()) {
			if (childs[0] && center[axis] - radius[axis] <= divider) 
				childs[0]->AddPointsInEllipsoidToList(center, radius, dst, maxPtCount);

			if (childs[1] && center[axis] + radius[axis] > divider) 
				childs[1]->AddPointsInEllipsoidToList(center, radius, dst, maxPtCount);
		}
		else
		{
			for (int i = 0; i < points.size(); i++) {
				float dist2 = ((points[i] - center) / radius).sqLength();
				if (dist2 <= 1.0f) {
					dst.push_back(indices[i]);
					if (maxPtCount > 0 && dst.size() >= maxPtCount)
						return;
				}
			}
		}
	}

	int GetNodeCount() {
		int nc = 1;
		for (int i = 0; i < 2; i++)
			if (childs[i]) nc += childs[i]->GetNodeCount();
		return nc;
	}


private:

	void MakeLeafNode(const std::vector<Point>& pts, const std::vector<int>& indices)
	{
		this->indices = indices;
		points.resize(indices.size());
		for (int i = 0; i < indices.size(); i++)
			points[i] = pts[indices[i]];
		axis = 0;
		divider = T{};
	}

	void Build(const std::vector<Point>& pts, const std::vector<int>& indices, int maxPointsPerLeaf) {
		if (indices.size() <= maxPointsPerLeaf) 
			MakeLeafNode(pts, indices);
		else {
			auto meanAndVar = ComputeMeanAndVar(pts, indices);
			Point var = meanAndVar[1];

			// select axis with highest variance
			int bestAxis = 0;
			if (D > 1) {
				for (int ax = 1; ax < D; ax++) {
					if (var[ax] > var[bestAxis])
						bestAxis = ax;
				}
			}
			axis = bestAxis;
			divider = meanAndVar[0][axis];

			std::vector<int> subidx[2];
			// split at mean (median would be better but slower)
			for (int i = 0; i < indices.size(); i++) {
				int lstIndex = pts[indices[i]][axis] <= divider ? 0 : 1;
				subidx[lstIndex].push_back(indices[i]);
			}

			// This is to prevent an edge case from happening: 
			// If >maxPointsPerLeaf points have the same value, 
			// the tree would make infinite sub tree nodes.
			if (subidx[0].size() == 0 || subidx[1].size() == 0) {
				MakeLeafNode(pts, indices);
			}
			else {
				for (int i = 0; i < 2; i++) {
					childs[i] = std::make_unique<KDTree>(pts, subidx[i], maxPointsPerLeaf);
				}
			}
		}
	}

	int axis;
	T divider;
	std::vector<int> indices;
	std::vector<Point> points;
	std::unique_ptr<KDTree> childs[2];

	static std::array<Point, 2> ComputeMeanAndVar(const std::vector<Point>& pts, const std::vector<int>& indices) {
		Point sum, sum2;
		for (int i = 0; i < indices.size(); i++) {
			int idx = indices[i];
			sum += pts[idx];
			sum2 += pts[idx] * pts[idx];
		}

		Point variance = (sum2 - (sum * sum) / indices.size()) / indices.size();
		Point mean = sum / indices.size();

		return { { mean,variance } };
	}
};


template<typename T, int D>
class NeighborList {
public:
	typedef Vector<T, D> Point;
	std::vector<int> startIndices, nbCounts;
	std::vector<int> nbIndices;
	int maxNbCount;

	NeighborList() { maxNbCount = 0; }

	// For every point in A, make a list of points in B within searchRange
	// nbIndices will hold indices into ptsB
	// startIndices and nbCounts are both length ptsA.size()
	template<typename TFilterFn>
	void Build(const std::vector<Point>& ptsA, const std::vector<Point>& ptsB, Point searchRange, TFilterFn filter, int neigborCountLimit)
	{
		KDTree<float, D> kdtree(ptsB, 20);
		Build(kdtree, ptsA, searchRange, filter, neigborCountLimit);
	}

	template<typename TFilterFn>
	void Build(KDTree<float, D>& kdtree, const std::vector<Point>& pts, Point searchRange, TFilterFn filter, int neigborCountLimit)
	{
		startIndices.resize(pts.size());
		nbCounts.resize(pts.size());
		std::vector< std::vector<int> > neighborLists(pts.size());

		ParallelFor((int)pts.size(), [&](int i) {
			kdtree.AddPointsInEllipsoidToList(pts[i], searchRange, neighborLists[i], neigborCountLimit);
			});

		int s = 0;
		for (auto& l : neighborLists)
			s += (int)l.size();

		nbIndices.clear();
		nbIndices.reserve(s);

		// Add them all into a single array to move to cuda
		for (int i = 0; i < pts.size(); i++) {
			startIndices[i] = (int)nbIndices.size();
			int n = 0;
			for (int idx : neighborLists[i]) {
				if (filter(i, idx)) {
					nbIndices.push_back(idx);
					n++;
				}
			}
			nbCounts[i] = n;
		}
		maxNbCount = *std::max_element(nbCounts.begin(), nbCounts.end());
		//DebugPrintf("Node count: %d. #neighbors: %d. max(spotNeighborCount): %d\n", kdtree.GetNodeCount(), nbIndices.size(), maxNbCount);
	}
};


// Iterate through all the neighbors of the points in pts, using multi-threading for speedup
// The iteration allows large datasets to be processed without hitting memory issues
// Callback has arguments: (int processedUpto, vector<int> indices, vector<int> startpos, vector<int> counts)
template<typename T, int D, typename Callback>
void IterateThroughNeighbors(KDTree<float, D>& kdtree,
	const std::vector< Vector<T, D> >& pts, Vector<T,D> searchRange, int minBatchSize, int maxNeighborCount, Callback cb)
{
	typedef Vector<T, D> Point;

	std::list< std::vector<int> > nblists;
	int numNeighbors = 0;

	int i = 0;
	int processedUpto = 0;
	while (i < pts.size())
	{
		int batchsize = std::min(50, (int)pts.size() - i);
		std::vector< std::vector<int> > neighborListInBatch(batchsize);
		ParallelFor(batchsize, [&](int j) {
			kdtree.AddPointsInEllipsoidToList(pts[i + j], searchRange, neighborListInBatch[j], maxNeighborCount);
			});
		i += batchsize;

		for (int j = 0; j < batchsize; j++) {
			numNeighbors += (int)neighborListInBatch[j].size();
			nblists.push_back(std::move(neighborListInBatch[j]));
		}

		if (numNeighbors >= minBatchSize || i == pts.size()) {
			// turn nblists into indices,startpos and counts
			int npts = (int)nblists.size();
			std::vector<int> indices, startpos(npts), counts(npts);
			indices.reserve(numNeighbors);

			// Add them all into a single array to move to cuda
			int k = 0;
			for (std::vector<int>& neighbors : nblists) {
				startpos[k] = (int)indices.size();
				counts[k] = (int)neighbors.size();
				indices.insert(indices.end(), neighbors.begin(), neighbors.end());
				k++;
			}
			if (!cb(processedUpto, std::move(indices), std::move(startpos), std::move(counts)))
				break;

			nblists.clear();
			numNeighbors = 0;
			processedUpto = i;
		}
	}
}
