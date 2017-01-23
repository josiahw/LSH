#ifndef MEANTRANSFORMER_H
#define MEANTRANSFORMER_H

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

class MeanTransformer {
    /*
        A pre-LSH low-dimensional projection filter. This allows hash function collections
        to specialise a common pre-filtering step to be done once for hte whole collection.
        TODO: figure out a way to make this nicer, and rename it
    */
    private:
        arma::vec means;
    public:

        MeanTransformer(void) {};

        MeanTransformer(const arma::mat& data, const uint64_t& numDims) {
            means = arma::mean(data,1);
        }

        MeanTransformer(const arma::Mat<float>& data, const uint64_t& numDims) {
            means = arma::mean(arma::conv_to<arma::mat>::from(data),1);
        }

        arma::mat transform(const arma::mat& d) {
            arma::mat result(d.n_rows,d.n_cols);
            for (size_t i = 0; i < d.n_cols; ++i) {
                result.col(i) = d.col(i) - means;
            }
            return std::move(result);
        }

        arma::Mat<float> transform(const arma::Mat<float>& d) {
            arma::Mat<float> result(d.n_rows,d.n_cols);
            for (size_t i = 0; i < d.n_cols; ++i) {
                result.col(i) = d.col(i) - arma::conv_to<arma::Col<float>>::from(means);
            }
            return std::move(result);
        }
};

#endif