#ifndef STDHASHINDEX_H
#define STDHASHINDEX_H

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
#include "HashIndexSupport.h"

class StdHashIndex {
	private:
		arma::Col<uint32_t> indices, binAddresses, bInd;
		std::unordered_map<uint64_t,uint32_t> binLocations;
		uint64_t minRetrieved, cntr;
		std::vector<std::vector<uint64_t>> binDists;

	public:

		StdHashIndex() {
		    binLocations.reserve(800000);
		    };

		StdHashIndex(
			const arma::uvec& hashCodes,
		    const uint64_t queryThreshold=10,
		    const uint64_t queryDepth=3,
		    const uint64_t bitSize = 32
		    ) {

		    //adaptive hash-ball query settings
		    static auto tmp  = getBallDistances(queryDepth,bitSize);
		    binDists = tmp;
		    minRetrieved = queryThreshold;

		    //count up all bins
		    binLocations.reserve(hashCodes.n_elem);
		    binLocations.reserve(2*hashCodes.n_elem);
		    binLocations.max_load_factor(0.5);
		    for (const auto& h : hashCodes) {
		        ++binLocations[h];
		    }

		    //set the bin addresses
		    cntr = 0;
		    uint64_t total = 0;
		    binAddresses.resize(binLocations.size() + 1);

		    for (auto& b : binLocations) {
		        binAddresses[cntr] = total;
		        total += b.second;
		        b.second = cntr;
		        ++cntr;
		    }
		    binAddresses[binLocations.size()] = total;
		    //std::cout << binLocations.size() << std::endl;
		    //insert all the indices
		    arma::Col<uint32_t> bInd = binAddresses;
		    indices.resize(hashCodes.n_elem);
		    cntr = 0;
		    for (const auto& h : hashCodes) {
		        const auto& bl = binLocations[h];
		        indices[bInd[bl]] = cntr;
		        ++bInd[binLocations[bl]];
		        ++cntr;
		    }
		}

		StdHashIndex(const uint64_t& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {

		    //adaptive hash-ball query settings
		    static auto tmp  = getBallDistances(queryDepth,bitSize);
		    binDists = tmp;
		    minRetrieved = queryThreshold;
		    indices.resize(hashCodes);

		}

		//chunked Function 1: countIndices
		void countIndices(const arma::uvec& hashCodes) {
		    //count up all bins
		    for (const auto& h : hashCodes) {
		        ++binLocations[h];
		    }
		}

		//function2: prepareIndices
		void prepareIndices() {
		    //set the bin addresses
		    cntr = 0;
		    uint64_t total = 0;
		    binAddresses.resize(binLocations.size() + 1);
		    for (auto& b : binLocations) {
		        binAddresses[cntr] = total;
		        total += b.second;
		        b.second = cntr;
		        ++cntr;
		    }
		    binAddresses[binLocations.size()] = total;

		    //insert all the indices
		    bInd = binAddresses;
		    cntr = 0;
		}


		    //function3: loadHashes
		void loadHashes(const arma::uvec& hashCodes) {
		    for (const auto& h : hashCodes) {
		        const auto& bl = binLocations[h];
		        indices[bInd[bl]] = cntr;
		        ++bInd[bl];
		        ++cntr;
		    }
		}



		void fillQuery(const uint64_t& hashCode, arma::Col<uint32_t>& results, std::set<uint32_t>& uniqueResults) {
		    //this fills an external query - so we can have a hash collection outside it collecting queries

		    //expand the hash-ball until we meet our minimum query threshold
		    uint64_t cnt = 0;
		    //for (const auto& b : binDists) {
		        for (const uint64_t& v : binDists.front()) {
		            const auto hc = hashCode^v;
					const auto& binLoc = binLocations.find(hc);
					if (binLoc != binLocations.end()) {
						const uint32_t addr = binLoc->second;
						const uint32_t end = binAddresses[addr+1];
						for ( uint32_t i = binAddresses[addr]; i < end; ++i) {
							const uint32_t& index = indices[i];
							if (not uniqueResults.count(index)) {
								results[uniqueResults.size()] = index;
								uniqueResults.insert(index);
								++cnt;
								if (uniqueResults.size() >= results.n_elem) break;
							}
						}
						if (cnt > minRetrieved or uniqueResults.size() >= results.n_elem) break;

					}
		        }
				//if (cnt > minRetrieved or uniqueResults.size() >= results.n_elem) break;
		    //}

		}

		template<class SET = std::set<uint32_t>>
		void fillQuerySet(const uint64_t& hashCode, SET& uniqueResults) {
		    //this fills an external query set, making some locality assumptions to help the boosting process.
		    const uint64_t minRet = uniqueResults.size()+minRetrieved;
		    //expand the hash-ball until we meet our minimum query threshold
		    //for (const auto& b : binDists) {
		        for (const uint64_t& v : binDists.front()) {
		            const auto hc = hashCode^v;
					const auto& binLoc = binLocations.find(hc);
					if (binLoc != binLocations.end()) {
						const uint32_t& addr = binLoc->second;
						const uint32_t& end = binAddresses[addr+1];
						//uniqueResults.insert(indices.begin_row(binAddresses[addr]),
						//                     indices.end_row(end-1));
						for ( uint32_t i = binAddresses[addr]; i < end and uniqueResults.size() < minRet; ++i) {
							uniqueResults.insert(indices[i]);
						}
						if (uniqueResults.size() >= minRet) break;

					}
		        }
				//if (uniqueResults.size() >= minRet) break;
		    //}

		}

		uint64_t size() const {
			return indices.n_elem;
		}

		/* XXX: I spose we should implement this sometime.
		void saveTable(std::string fileName) {
		    ;
		}*/
};

#endif
