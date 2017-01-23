#ifndef HASHINDEXSUPPORT_H
#define HASHINDEXSUPPORT_H

#include <cstdint>
#include <armadillo>
#include <unordered_map>
#include <algorithm>
#include <sys/types.h>
#include <sstream>
#include <random>
#include <set>
#include <unordered_set>
#include <list>
#include <iostream>

std::vector<uint64_t> queryDist(uint64_t maxDist, uint64_t bitSize) {
    std::vector<uint64_t> results(0);
    if (maxDist == 0) {
        results.push_back(0);
    } else if (maxDist == 1) {
        for (uint64_t i = 0; i < bitSize; ++i) {
            results.push_back(1ull << i);
        }
    } else {
        for (uint64_t i = maxDist-1; i < bitSize; ++i) {
            const auto answers = queryDist(maxDist-1,i);
            for (const auto& a : answers) {
                results.push_back(a + (1ull << i ));
            }
        }
    }
    return results;
};

std::vector<std::vector<uint64_t>> getBallDistances(uint64_t maxDist, uint64_t bitSize) {
    /*
     This function returns a list in expanding distance order of hamming distance codes.
     These can be xor'd with a hash function to produce an expanding order search list.
    */
    std::vector<std::vector<uint64_t>> dists(maxDist);
    for (uint64_t i = 0; i < maxDist; ++i) {
        dists[i] = queryDist(i,bitSize);
        std::sort (dists[i].begin(), dists[i].end());
        if (i > 0) {
            dists[0].insert(dists[0].end(),dists[i].begin(),dists[i].end());
        }
    }
    dists.resize(1);
    return dists;
};

#endif