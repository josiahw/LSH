#ifndef LSH_H
#define LSH_H
#include <iostream>
#include <cstdint>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <armadillo>

template <typename T>
class LSH
{

protected:
private:
    unsigned int totalHashes;
    unsigned int totalBits;
    unsigned int functionBits;
    std::vector<std::vector<size_t>> hashes;
    std::vector<arma::rowvec> originalData;
    arma::rowvec means;
    std::vector<std::unordered_map<uint64_t,size_t>> hashMaps;
    std::vector<size_t> prehash;

    void sortIndices(const arma::rowvec& sortBy, std::vector<size_t>& indices) {
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }

        //get sorted indices
        std::sort(std::begin(indices),std::end(indices),
                  [&sortBy](const size_t& a, const size_t& b) {
                        return sortBy[a]>sortBy[b];
                  });
    }



public:
    LSH(uint32_t numberOfTables, uint32_t bitsPerTable=32) : totalHashes(numberOfTables), totalBits(bitsPerTable), hashMaps(numberOfTables) {
    }
    virtual ~LSH() {}

    void buildHashes(const arma::mat& allData) {
        //per-dimension means
        means = arma::mean(allData);

        //per-dimension deviations
        arma::rowvec variances = arma::var(allData);

        //per-dimension deviations
        //arma::rowvec standardDeviations = arma::sqrt(variances);

        //location for std-sorted indices
        //std::vector<size_t> sortedIndices(allData.n_cols);

        double vSum = arma::sum(variances); //get our variance sums

        //make a sorted by decreasing variance set of indices
        //sortIndices(variances,sortedIndices);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0, vSum);

        hashes.clear();
        std::vector<uint8_t> used(allData.n_cols,0);

        for (size_t i = 0; i < this->totalHashes; ++i) {

            std::vector<size_t> hashN;
            while (hashN.size() < this->totalBits) {
                double distSum = variances[0];
                double target = dist(gen);
                size_t selection = 0;

                while (distSum < target) {
                    ++selection;
                    distSum += variances[selection];
                }

                bool repeated = (used[selection] == 2);
                if (not repeated) {
                    used[selection] = 2;
                    hashN.push_back(selection);
                }
            }
            for(auto j : hashN) {
                used[j] = 1;
            }
            hashes.push_back(std::move(hashN));
        }

        //calculate the candidates to mean-split when we hash.
        for(size_t i = 0; i < used.size(); ++i) {
            if (used[i] == 1) {
                prehash.push_back(i);
            }
        }
    }


    std::vector<uint64_t> getHashes(const arma::rowvec& p) {
        std::vector<uint64_t> tmp(p.n_cols);
        std::vector<uint64_t> result(totalHashes,0);
        for (auto i : prehash) {
            tmp[i] = p[i] > means[i];
        }
        for (size_t i = 0; i < totalHashes; ++i) {
            for (auto j : hashes[i]) {
                result[i] = (result[i] << 1) | tmp[j];
            }
        }
        return result;
    }


    void insert(const arma::rowvec& data) {
        const std::vector<uint64_t> hashVals = getHashes(data);
        for (size_t i = 0; i < this->totalHashes; ++i) {
            hashMaps[i].insert(std::make_pair(hashVals[i],originalData.size()));
        }
        originalData.push_back(data);

    }

    std::vector<size_t> query(const arma::rowvec& p) { //query for all the items in buckets and return a unique list of ID's
        const std::vector<uint64_t> hashVals = getHashes(p);
        std::vector<size_t> values;
        std::set<uint64_t> uniqueValues;

        //Get a list of bucket iterators. We know from the way we construct our table that ID's strictly increase.
        //Therefore we have a sorted list of ID's in each bucket. Sort through for unique ones.

        for (size_t i = 0; i < this->totalHashes; ++i) {
            const size_t bucketID = hashMaps[i].bucket(hashVals[i]);
            for (auto j = hashMaps[i].begin(bucketID); j != hashMaps[i].end(bucketID); ++j) {
                uniqueValues.insert((*j).second);
                //std::cout << (*j).second << " ";
            }
            //std::cout << std::endl;
        }
        values.reserve(uniqueValues.size());
        for (auto j = uniqueValues.begin(); j != uniqueValues.end(); ++j) {
            values.push_back((*j));
        }

        return values;
    }

    std::vector<size_t> queryKNN_Euclidean(const arma::rowvec& p) {
        const std::vector<size_t> neighbours = query(p);
        std::vector<size_t> range(neighbours.size());
        std::vector<size_t> newNeighbours(neighbours.size());
        arma::rowvec dists(neighbours.size());
        for (size_t i = 0; i < neighbours.size(); ++i) {
            range[i] = i;
            const arma::rowvec diff = p-originalData[neighbours[i]];
            dists[i] = -arma::dot(diff,diff);
        }
        sortIndices(dists,range);
        for (size_t i = 0; i < neighbours.size(); ++i) {
            newNeighbours[i] = neighbours[range[i]];
        }
        return newNeighbours;
    }

    void loadDataSet(const arma::mat& allData) { //load the dataset as a whole, this should be the default

        const size_t numBuckets = 65535;

        for (auto& map : hashMaps) {
            // Disable the load factor check on the hash tables
            map.max_load_factor(std::numeric_limits<float>::max());
            // Set the number of buckets
            map.rehash(numBuckets);
        }

        //XXX: this should really only be called on instantiation
        buildHashes(allData);
        originalData.reserve(allData.n_rows);
        for (size_t i = 0; i < allData.n_rows; ++i) {
            insert(allData.row(i));
        }
    }
};

#endif // LSH_H
