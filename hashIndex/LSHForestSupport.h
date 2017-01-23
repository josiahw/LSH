#ifndef LSHFORESTSUPPORT_H
#define LSHFORESTSUPPORT_H
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
#include <deque>

#include <iostream>

struct HashNode {
    uint64_t prefix; // the hash code prefix
    uint32_t length; // the prefix length
    int32_t leftChild;
    int32_t rightChild;
    std::vector<uint32_t> children;
};

#endif