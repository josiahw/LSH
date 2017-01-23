#ifndef NULLTRANSFORMER_H
#define NULLTRANSFORMER_H

#include <cstdint>
#include <armadillo>
#include <unordered_map>
#include <algorithm>
#include <sys/types.h>
#include <sstream>
#include <random>
#include <set>
#include <list>
#include <iostream>

class NullTransformer {
    /*
        A pre-LSH low-dimensional projection filter. This allows hash function collections
        to specialise a common pre-filtering step to be done once for hte whole collection.
        TODO: figure out a way to make this nicer, and rename it
    */
    public:

        NullTransformer(void) {};

        NullTransformer(const arma::mat& data, const uint64_t& numDims) {
        }

        arma::mat transform(const arma::mat& d) {
            return d;
        }

        arma::mat transform(const arma::Mat<float>& d) {
            return arma::conv_to<arma::mat>::from(d);
        }

        arma::rowvec transform(const arma::rowvec& d) {
            return d;
        }
};

#endif