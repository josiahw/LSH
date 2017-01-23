#ifndef HASHCOLLECTION_H
#define HASHCOLLECTION_H
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
#include "StdHashIndex.h"
#include "ResizeableHashIndex.h"
#include "LSHForestHashIndex.h"
#include "LSHForestHashIndexAsync.h"

#include <iostream>

template<class DataTransformer, class HashFunction, class HashIndex, class MAT>
class HashCollection {
    private:
        DataTransformer downsampler;
        std::vector<HashFunction> hashes;
        std::vector<HashIndex> indices;
        uint64_t querySize, minQuery, hashBits;
    public:
        uint64_t queryDepth;
        HashCollection(const MAT& data,
                       const uint64_t& candidatebits,
                       const uint64_t& querysize,
                       const uint64_t& minquery,
                       const uint64_t& querydepth,
                       const uint64_t& bits) :
                       querySize(querysize),
                       minQuery(minquery),
                       queryDepth(querydepth),
                       hashBits(bits) {
            //make the pca transformer
            downsampler = DataTransformer(
                            data.cols(0,std::min<uint64_t>(30000-1,data.n_rows-1)),
                            candidatebits);
        }

        void addHash(const HashFunction& h) {
            hashes.push_back(h);
        }

        void buildDB(const MAT& data) {
            //XXX: this assumes we can grab the data in one chunk
            auto tf = downsampler.transform(data);
            indices.clear();
            indices.resize(hashes.size());
            //build the hash databases
            #pragma omp parallel for
            for (uint64_t i = 0; i < hashes.size(); ++i) {
                indices[i] = HashIndex(hashes[i].getHash(tf),
                                       minQuery,
                                       queryDepth,
                                       hashBits);
            }
        }

        template<class H = HashIndex>
        typename std::enable_if<std::is_same<H, StdHashIndex>::value, void>::type
          buildChunkedDB(const MAT& data, const uint64_t& chunkSize = 1000000) {
            indices.clear();
            indices.resize(hashes.size());
            //init the hashes
            #pragma omp parallel for
            for (uint64_t i = 0; i < hashes.size(); ++i) {
                indices[i] = HashIndex(data.n_cols,
                                       minQuery,
                                       queryDepth,
                                       hashBits);
            }

            //count bucket sizes
            for (uint64_t i = 0; i < data.n_cols; i += chunkSize) {
                auto tf = downsampler.transform(
                                data.cols(i,std::min<uint64_t>(i+chunkSize-1,data.n_cols-1))
                                );
                //build the hash databases
                #pragma omp parallel for
                for (uint64_t j = 0; j < hashes.size(); ++j) {
                    indices[j].countIndices(hashes[j].getHash(tf));
                }
            }

            for (uint64_t j = 0; j < hashes.size(); ++j) {
                indices[j].prepareIndices();
            }

            //count bucket sizes
            for (uint64_t i = 0; i < data.n_cols; i += chunkSize) {
                auto tf = downsampler.transform(
                                data.cols(i,std::min<uint64_t>(i+chunkSize-1,data.n_cols-1))
                                );

                //build the hash databases
                #pragma omp parallel for
                for (uint64_t j = 0; j < hashes.size(); ++j) {
                    indices[j].loadHashes(hashes[j].getHash(tf));
                }
            }
        }

        template<class H = HashIndex>
        typename std::enable_if<!std::is_same<H, StdHashIndex>::value, void>::type
          buildChunkedDB(const MAT& data, const uint64_t& chunkSize = 1000000) {
            indices.clear();
            indices.resize(hashes.size());
            //init the hashes
            #pragma omp parallel for
            for (uint64_t i = 0; i < hashes.size(); ++i) {
                indices[i] = HashIndex(data.n_cols,
                                       minQuery,
                                       queryDepth,
                                       hashBits);
            }

            //count bucket sizes
            for (uint64_t i = 0; i < data.n_cols; i += chunkSize) {
                auto tf = downsampler.transform(
                                data.cols(i,std::min<uint64_t>(i+chunkSize-1,data.n_cols-1))
                                );
                //build the hash databases
                #pragma omp parallel for
                for (uint64_t j = 0; j < hashes.size(); ++j) {
                    indices[j].countIndices(hashes[j].getHash(tf));
                }
            }

            /*for (uint64_t j = 0; j < hashes.size(); ++j) {
                indices[j].prepareIndices();
            }

            //count bucket sizes
            for (uint64_t i = 0; i < data.n_cols; i += chunkSize) {
                auto tf = downsampler.transform(
                                data.cols(i,std::min<uint64_t>(i+chunkSize-1,data.n_cols-1))
                                );

                //build the hash databases
                #pragma omp parallel for
                for (uint64_t j = 0; j < hashes.size(); ++j) {
                    indices[j].loadHashes(hashes[j].getHash(tf));
                }
            }*/
        }

        template<class H = HashIndex>
        typename std::enable_if<!std::is_same<H, LSHForestHashIndex>::value, arma::Col<uint32_t>>::type testQuery(const MAT& q) {
            /*
            This query method keeps items in the order they were discovered in the LSH collection.
            This allows statistics testing for performance analysis, but isn't necessary for practical use.
            */
            auto tf = downsampler.transform(q);
            arma::Col<uint32_t> results(querySize);
            std::set<uint32_t> uniqueResults;
            for (uint64_t i = 0; i < indices.size(); ++i) {
                indices[i].fillQuery(hashes[i].getHash(tf)[0], results, uniqueResults);
                if (uniqueResults.size() >= results.n_elem) break;
            }
            if (uniqueResults.size() < results.n_elem) {
                results.shed_rows(uniqueResults.size(),results.n_elem-1);
            }
            return results;
        }

        template<class H = HashIndex>
        typename std::enable_if<std::is_same<H, LSHForestHashIndex>::value, arma::Col<uint32_t>>::type testQuery(const MAT& q) {
            /*
            This query method keeps items in the order they were discovered in the LSH collection.
            This allows statistics testing for performance analysis, but isn't necessary for practical use.
            */
            auto tf = downsampler.transform(q);
            arma::Col<uint32_t> results(querySize);
            std::set<uint32_t> uniqueResults;
            std::vector<std::vector<int32_t>> nodevals(indices.size());
            for (uint64_t j = 0; j < queryDepth and uniqueResults.size() <= querySize; ++j) {
                for (uint64_t i = 0; i < indices.size(); ++i) {
                    indices[i].fillQuery(hashes[i].getHash(tf)[0], results, uniqueResults,nodevals[i],j);
                    if (uniqueResults.size() >= results.n_elem) break;
                }
            }
            if (uniqueResults.size() < results.n_elem) {
                results.shed_rows(uniqueResults.size(),results.n_elem-1);
            }
            return results;
        }


        template<class H = HashIndex>
        typename std::enable_if<!std::is_same<H, LSHForestHashIndex>::value, arma::Col<uint32_t>>::type  query(const MAT& q) {
            //this does the transform, hash, and query for a single item
            auto tf = downsampler.transform(q);
            std::set<uint32_t> uniqueResults;
            for (uint64_t i = 0; i < indices.size() and uniqueResults.size() <= querySize; ++i) {
                indices[i].fillQuerySet(hashes[i].getHash(tf)[0], uniqueResults, querySize);
            }
            arma::Col<uint32_t> results(uniqueResults.size());
            uint32_t cnt = 0;
            for (const auto& r : uniqueResults) {
                results[cnt] = r;
                ++cnt;
            }
            return results;
        }

        template<class H = HashIndex>
        typename std::enable_if<std::is_same<H, LSHForestHashIndex>::value, arma::Col<uint32_t>>::type  query(const MAT& q) {
            //this does the transform, hash, and query for a single item
            auto tf = downsampler.transform(q);
            std::set<uint32_t> uniqueResults;
            std::vector<std::vector<int32_t>> nodevals(indices.size());
            for (uint64_t j = 0; j < queryDepth and uniqueResults.size() <= querySize; ++j) {
                for (uint64_t i = 0; i < indices.size() and uniqueResults.size() <= querySize; ++i) {
                    indices[i].fillQuerySet(hashes[i].getHash(tf)[0], uniqueResults,nodevals[i],j,querySize);
                }
            }
            arma::Col<uint32_t> results(uniqueResults.size());
            uint32_t cnt = 0;
            for (const auto& r : uniqueResults) {
                results[cnt] = r;
                ++cnt;
            }
            return results;
        }

        template<class H = HashIndex>
        typename std::enable_if<!std::is_same<H, LSHForestHashIndex>::value, std::pair<arma::Mat<uint32_t>,MAT>>::type
        batchQuery(const MAT& q, const uint64_t& kneighbours, const MAT& baseData) {
            //this does the transform, hash, and query for a single item
            auto tf = downsampler.transform(q);
            arma::Mat<uint32_t> resultinds(q.n_cols,kneighbours);
            MAT resultdists(q.n_cols,kneighbours);
            resultinds.fill(0);
            //resultdists.fill(0.0);
            arma::umat queryHashes(q.n_cols,indices.size());
            #pragma omp parallel for
            for (uint64_t i = 0; i < indices.size(); ++i) {
                queryHashes.col(i) = hashes[i].getHash(tf);
            }
            tf.reset();

            #pragma omp parallel for
            for (uint64_t j = 0; j < q.n_cols; ++j) {
                std::set<uint32_t> uniqueResults;

                for (uint64_t i = 0; i < indices.size() and uniqueResults.size() <= querySize; ++i) {
                    indices[i].fillQuerySet(queryHashes(j,i), uniqueResults);
                }
                if (uniqueResults.size()) {

                    arma::Col<uint32_t> results(uniqueResults.size());
                    MAT dists(1,uniqueResults.size());
                    uint32_t cnt = 0;
                    const MAT qj = q.col(j);
                    for (const auto& r : uniqueResults) {
                        results[cnt] = r;
                        dists[cnt] = arma::norm(baseData.col(r)-qj);
                        ++cnt;
                    }
                    auto cmp = ([&dists](const uint32_t& a, const uint32_t& b) {
                                return dists[a]<dists[b];
                            });

                    std::vector<uint32_t> distanceStack;
                    distanceStack.reserve(kneighbours);
                    cnt = std::min<uint32_t>(kneighbours,results.n_elem);
                    for (uint32_t i = 0; i < cnt; ++i) {
                        distanceStack.emplace_back(i);
                    }
                    if (distanceStack.size() == kneighbours) {
                        std::make_heap(distanceStack.begin(),distanceStack.end(),cmp);
                        float current = dists[distanceStack.front()];
                        for (uint32_t i = distanceStack.size(); i < results.n_elem; ++i) {
                            if (dists[i] < current) {
                                std::pop_heap(distanceStack.begin(),distanceStack.end(),cmp);
                                distanceStack[kneighbours-1] = i;
                                std::push_heap(distanceStack.begin(),distanceStack.end(),cmp);
                                current = dists[distanceStack.front()];
                            }
                        }
                    }
                    const arma::uvec tmpInds = arma::conv_to<arma::uvec>::from(distanceStack);
                    resultinds.submat(j,0,j,distanceStack.size()-1) = results(tmpInds).t();
                    resultdists.submat(j,0,j,distanceStack.size()-1) = dists(tmpInds).t();
                }

            }
            return std::make_pair(resultinds,resultdists);
        }


        template<class H = HashIndex>
        typename std::enable_if<std::is_same<H, LSHForestHashIndex>::value, std::pair<arma::Mat<uint32_t>,MAT>>::type
        batchQuery(const MAT& q, const uint64_t& kneighbours, const MAT& baseData) {
            //this does the transform, hash, and query for a single item
            auto tf = downsampler.transform(q);
            arma::Mat<uint32_t> resultinds(q.n_cols,kneighbours);
            MAT resultdists(q.n_cols,kneighbours);
            resultinds.fill(0);
            resultdists.fill(0.0);
            arma::umat queryHashes(q.n_cols,indices.size());
            for (uint64_t i = 0; i < indices.size(); ++i) {
                queryHashes.col(i) = hashes[i].getHash(tf);
            }
            tf.reset();

            //find all the results and collate them


            //#pragma omp parallel for
            for (uint64_t j = 0; j < q.n_cols; ++j) {
                std::unordered_set<uint32_t> uniqueResults;
                uniqueResults.reserve(querySize);
                std::vector<std::vector<int32_t>> nodevals(indices.size());
                for (uint64_t k = 0; k < queryDepth and uniqueResults.size() <= querySize; ++k) {
                    for (uint64_t i = 0; i < indices.size() and uniqueResults.size() <= querySize; ++i) {
                        indices[i].fillQuerySet(queryHashes(j,i), uniqueResults,nodevals[i],k,querySize);
                    }
                }
                if (uniqueResults.size()) {
                    arma::Col<uint32_t> results(uniqueResults.size());
                    MAT dists(1,uniqueResults.size());
                    uint32_t cnt = 0;
                    for (const auto& r : uniqueResults) {
                        results[cnt] = r;
                        dists[cnt] = arma::norm(baseData.col(r)-q.col(j));
                        ++cnt;
                    }

                    auto cmp = ([&dists](const uint32_t& a, const uint32_t& b) {
                                return dists[a]<dists[b];
                            });

                    std::vector<uint32_t> distanceStack;
                    distanceStack.reserve(kneighbours);
                    cnt = std::min<uint32_t>(kneighbours,results.n_elem);
                    for (uint32_t i = 0; i < cnt; ++i) {
                        distanceStack.emplace_back(i);
                    }

                    std::make_heap(distanceStack.begin(),distanceStack.end(),cmp);
                    float current = dists[distanceStack.front()];
                    for (uint32_t i = distanceStack.size(); i < results.n_elem; ++i) {
                        if (dists[i] < current) {
                            std::pop_heap(distanceStack.begin(),distanceStack.end(),cmp);
                            distanceStack[kneighbours-1] = i;
                            std::push_heap(distanceStack.begin(),distanceStack.end(),cmp);
                            current = dists[distanceStack.front()];
                        }
                    }

                    const arma::uvec tmpInds = arma::conv_to<arma::uvec>::from(distanceStack);
                    resultinds.submat(j,0,j,distanceStack.size()-1) = results(tmpInds).t();
                    resultdists.submat(j,0,j,distanceStack.size()-1) = dists(tmpInds).t();
                }
            }
            return std::make_pair(resultinds,resultdists);
        }

        auto transform(const MAT& data)
        -> decltype(downsampler.transform(data)) {
            //access to the transform so we don't double up when training
            return downsampler.transform(data);
        }

};

#endif
