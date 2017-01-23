#ifndef DOUBLEHADAMARDFUNCS_H
#define DOUBLEHADAMARDFUNCS_H

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


template <class MAT = arma::mat>
MAT HadamardRecursive(const arma::vec& P, const MAT& x, const arma::rowvec& means) {
    // These functions create a reasonably efficient log-time hadamard projection calculator
    MAT r1 = x.t();
    r1.each_row() -= arma::conv_to<MAT>::from(means);
    if (r1.n_cols != P.n_rows) {
        r1.resize(r1.n_rows,P.n_rows);
    }
    MAT r2(r1.n_rows,P.n_rows);
    for (int i = P.n_rows/2; i > 1; i /= 2) {

        #pragma omp parallel for
        for (int j = 0; j < P.n_rows; j += 2*i) {
            r2.cols(j,j+i-1)  = r1.cols(j,j+i-1) + r1.cols(j+i,j+2*i-1);
            r2.cols(j+i,j+2*i-1)  = r1.cols(j,j+i-1) - r1.cols(j+i,j+2*i-1);
        }
        i /= 2;
        if (i == 1) {
            #pragma omp parallel for
            for (int i = 0; i < P.n_rows; i += 2) {
                r1.col(i) = P[i] * r2.col(i) + P[i+1] * r2.col(i+1);
                r1.col(i+1) = P[i] * r2.col(i) - P[i+1] * r2.col(i+1);
            }
            return r1;
        }
        #pragma omp parallel for
        for (int j = 0; j < P.n_rows; j += 2*i) {
            r1.cols(j,j+i-1)  = r2.cols(j,j+i-1) + r2.cols(j+i,j+2*i-1);
            r1.cols(j+i,j+2*i-1)  = r2.cols(j,j+i-1) - r2.cols(j+i,j+2*i-1);
        }

    }
    #pragma omp parallel for
    for (int i = 0; i < P.n_rows; i += 2) {
        r2.col(i) = P[i] * r1.col(i) + P[i+1] * r1.col(i+1);
        r2.col(i+1) = P[i] * r1.col(i) - P[i+1] * r1.col(i+1);
    }
    return r2;
}

template <class MAT = arma::mat>
MAT HadamardRecursive(const arma::vec& P, const MAT& x) {
    MAT r1 = x.t();
    MAT r2(r1.n_rows,P.n_rows);
    for (int i = P.n_rows/2; i > 1; i /= 2) {
        #pragma omp parallel for
        for (int j = 0; j < P.n_rows; j += 2*i) {
            r2.cols(j,j+i-1)  = r1.cols(j,j+i-1) + r1.cols(j+i,j+2*i-1);
            r2.cols(j+i,j+2*i-1)  = r1.cols(j,j+i-1) - r1.cols(j+i,j+2*i-1);
        }
        i /= 2;
        if (i == 1) {
            #pragma omp parallel for
            for (int i = 0; i < P.n_rows; i += 2) {
                r1.col(i) = P[i] * r2.col(i) + P[i+1] * r2.col(i+1);
                r1.col(i+1) = P[i] * r2.col(i) - P[i+1] * r2.col(i+1);
            }
            return r1;
        }
        #pragma omp parallel for
        for (int j = 0; j < P.n_rows; j += 2*i) {
            r1.cols(j,j+i-1)  = r2.cols(j,j+i-1) + r2.cols(j+i,j+2*i-1);
            r1.cols(j+i,j+2*i-1)  = r2.cols(j,j+i-1) - r2.cols(j+i,j+2*i-1);
        }

    }
    #pragma omp parallel for
    for (int i = 0; i < P.n_rows; i += 2) {
        r2.col(i) = P[i] * r1.col(i) + P[i+1] * r1.col(i+1);
        r2.col(i+1) = P[i] * r1.col(i) - P[i+1] * r1.col(i+1);
    }
    return r2;
}

#endif