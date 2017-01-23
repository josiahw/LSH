#ifndef THRESHOLDTRANSFORMER_H
#define THRESHOLDTRANSFORMER_H

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

class ThresholdTransformer {
    /*
        A pre-LSH low-dimensional projection filter. This allows hash function collections
        to specialise a common pre-filtering step to be done once for hte whole collection.
        TODO: figure out a way to make this nicer, and rename it
    */
    private:
        arma::rowvec means;
    public:

        ThresholdTransformer(void) {};

        ThresholdTransformer(const arma::mat& data, const uint64_t& numDims) {
            means = arma::mean(data,0);
        }

        arma::umat transform(const arma::mat& d) {
            arma::umat result(d.n_rows,d.n_cols);
            for (size_t i = 0; i < d.n_rows; ++i) {
                result.row(i) = d.row(i) > means[i];
            }
            return std::move(result);
        }

        arma::umat transform(const arma::Mat<float>& d) {
            arma::umat result(d.n_rows,d.n_cols);
            for (size_t i = 0; i < d.n_rows; ++i) {
                result.row(i) = d.row(i) > means[i];
            }
            return std::move(result);
        }
};

#endif