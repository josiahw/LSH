#ifndef TESTLSH_H
#define TESTLSH_H
#define ARMA_64BIT_WORD
#include <cstdint>
#include <armadillo>
#include <unordered_map>
#include <algorithm>
#include <sys/types.h>
#include <sstream>
#include <random>
#include <set>
#include <list>
#include <sys/time.h>

//hash functions
#include "RandomProjectionHashFunction.h"
#include "RandomSubSamplingHashFunction.h"
#include "ShiftInvariantKernelHashFunction.h"
#include "DHHashFunction.h"
#include "RandomRotationHashFunction.h"

//data transformers
#include "NullTransformer.h"
#include "MeanTransformer.h"
#include "ThresholdTransformer.h"
#include "PCATransformer.h"
#include "DHHashTransformer.h"

//hash indices
#include "StdHashIndex.h"
#include "ResizeableHashIndex.h"
#include "LSHForestHashIndex.h"
#include "LSHForestHashIndexAsync.h"

//hash collection
#include "HashCollection.h"


double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
#include <iostream>

template<class HashGenerator, class DataTransformer, class HashFunction, class HashIndex, class MAT>
class TestLSH {
public:
    const static std::tuple<double, double, double> TestLSHTimings(
        const MAT& data,
        const MAT& queries,
        const arma::Mat<uint32_t>& groundtruth, //this is the ground truth knn for the first m values of data against the rest of data
        const size_t& datadimensionality,
        const size_t& datalength,
        const size_t& querylength,
        const size_t& NumHashFunctions, //number of hash tables in the collection
        const size_t& NumHashBits, //number of bits per function
        const size_t& NumHashCandidates, //for trainable algorithms: number of candidates to optimise over
        const size_t& MaxSearchRadius, //maximum hamming ball / tree depth to explore
        const size_t& MaxQuerySize, //for the whole query
        const size_t& MaxRetrievalThreshold, //for each table
        const size_t& MinRetrievalThreshold, //for each table
        const size_t& KNeighbours) {

        //Step 1: build the hash functions
        double t0 = get_wall_time();
        HashCollection<DataTransformer,
                   HashFunction,
                   HashIndex,
                   MAT
                   > hashindex =
                   HashGenerator::GetHashes(
                            groundtruth,
                            data,
                            NumHashCandidates,
                            NumHashFunctions,
                            MinRetrievalThreshold,
                            MaxSearchRadius,
                            MaxQuerySize,
                            NumHashBits);
        double t1 = get_wall_time();

        //Step 2: build the hash index
        double t2 = get_wall_time();
        hashindex.buildChunkedDB(data);
        double t3 = get_wall_time();

        //Step 3: do the queries
        double t4 = get_wall_time();
        const std::pair<arma::Mat<uint32_t>,MAT> result__ = hashindex.batchQuery(queries,
                                                                            KNeighbours,
                                                                            data);
        double t5 = get_wall_time();

        //return the timings for all 3 parts
        return std::tuple<double,double,double>(t1 - t0, t3 - t2, t5 - t4);
    }


const static std::pair<arma::mat,arma::mat> TestLSHAccuracy(
        const MAT& data,
        const MAT& queries,
        const arma::Mat<uint32_t>& groundtruth, //this is the ground truth knn for the first m values of data against the rest of data
        const size_t& datadimensionality,
        const size_t& datalength,
        const size_t& querylength,
        const size_t& NumHashFunctions, //number of hash tables in the collection
        const size_t& NumHashBits, //number of bits per function
        const size_t& NumHashCandidates, //for trainable algorithms: number of candidates to optimise over
        const size_t& MaxSearchRadius, //maximum hamming ball / tree depth to explore
        const size_t& MaxQuerySize, //for the whole query
        const size_t& MaxRetrievalThreshold, //for each table
        const size_t& MinRetrievalThreshold, //for each table
        const size_t& KNeighbours) {

        //Step 1: build the hash functions
        HashCollection<DataTransformer,
                   HashFunction,
                   HashIndex,
                   MAT
                   > hashindex =
                   HashGenerator::GetHashes(
                            groundtruth,
                            data,
                            NumHashCandidates,
                            NumHashFunctions,
                            MinRetrievalThreshold,
                            MaxSearchRadius,
                            MaxQuerySize,
                            NumHashBits);

        //Step 2: build the hash index
        hashindex.buildChunkedDB(data);

        //Step 3: calculate the performance for each query
        arma::mat recalls(21, queries.n_cols);
        recalls.fill(-1);
        recalls.row(0) *= 0;
        arma::mat precisions(21, queries.n_cols);
        precisions.fill(-1);
        precisions.row(0) *= 0;
        for (size_t i = 0; i < queries.n_cols; ++i) {
            //get the query
            arma::Col<uint32_t> inds = hashindex.testQuery(queries.col(i));
            //std::cout << inds.t();

            //solve for ground-truth
            static auto neighbourCmp = ([](const std::pair<double,size_t>& a,
                                           const std::pair<double,size_t>& b) {
                return std::get<0>(a) < std::get<0>(b);
            });

            std::vector<std::pair<double,size_t>> neighbourStack;
            for (size_t j = 0; j < KNeighbours; ++j) {
                double dist = arma::norm(data.col(j) - queries.col(i));
                neighbourStack.push_back(std::make_pair(dist,j));

            }
            std::make_heap(neighbourStack.begin(),
                               neighbourStack.end(),
                               neighbourCmp);
            for (size_t j = KNeighbours; j < data.n_cols; ++j) {
                double dist = arma::norm(data.col(j) - queries.col(i));
                if (dist < std::get<0>(neighbourStack.front())) {
                    std::pop_heap(neighbourStack.begin(),
                                  neighbourStack.end(),
                                  neighbourCmp);
                    neighbourStack.back() = std::make_pair(dist,j);
                    std::push_heap(neighbourStack.begin(),
                                   neighbourStack.end(),
                                   neighbourCmp);
                }

            }
            //sort the neighbours
            std::sort(neighbourStack.begin(),
                       neighbourStack.end(),
                       neighbourCmp);
            /*for (const auto& n : neighbourStack) {
                std::cout << ", " << n.second;
            }
            std::cout << std::endl << std::endl;*/
            //Calculate statistics: Recall@r, Precision@r
            double found = 0.0;
            for (size_t j = 0; j < inds.n_elem; ++j) {
                for (size_t k = 0; k < neighbourStack.size(); ++k) {
                    if (neighbourStack[k].second == inds[j]) {
                        found += 1.;
                    }
                }
                if (j % (MaxQuerySize / 20) == 0) {
                    recalls(j / (MaxQuerySize / 20), i) = found / KNeighbours;
                    precisions(j / (MaxQuerySize / 20), i) = found / j;
                }
            }

        }

        //return the performance statistics
        return {recalls, precisions};
    }

};

#endif