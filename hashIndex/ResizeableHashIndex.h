//#include "_HashIndex.h"
//#include "LSHForestIndex.h"
#ifndef RESIZEABLEHASHINDEX_H
#define RESIZEABLEHASHINDEX_H

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

#include "HashIndexSupport.h"


/*std::vector<uint64_t> queryDist(uint64_t maxDist, uint64_t bitSize) {
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
};*/

class ResizeableHashIndex {
	private:
	    static const uint64_t multiplier = 1500450271 ;
	    uint64_t primeFactor = 5915587277 ;
	    uint64_t minRetrieved,
	             cntr,
	             outerMask,
	             innerMask,
	             tableSize,
	             maxSize,
	             maxDist;
	    double innerLoadFactor, outerLoadFactor;
	    std::vector<std::pair<int32_t,uint32_t>> hashTable;
	    std::vector<std::vector<std::pair<uint64_t,uint32_t>>> hashBins;
	    std::vector<std::pair<uint64_t,uint32_t>> emptyBin;
		std::vector<std::vector<uint64_t>> binDists;

	    void checkResizeInner() {
	        //NOTE: this will likely reduce the maxdist
	        if (hashBins.size() > innerLoadFactor*tableSize) {

	            //resize the table
	            const uint64_t oldMask = innerMask;
	            const uint64_t oldSize = hashTable.size();
	            tableSize *= 2;
	            innerMask = tableSize-1;
	            hashTable.resize(tableSize,{-1,0});

	            //go through all bins and move them if necessary
	            for (uint64_t i = 0; i < oldSize; ++i) {
	                if (std::get<0>(hashTable[i]) >= 0) {
	                    const uint64_t startKey = std::get<0>(
	                                            hashBins[std::get<1>(hashTable[i])].front()
	                                            );
	                    //const uint64_t endKey = std::get<0>(
	                    //                        hashBins[std::get<1>(hashTable[i])].back()
	                    //                        )
	                    if ( (startKey & oldMask) != (startKey & innerMask) ) {
	                        //move the whole cell
	                        //can safely ignore adding to populated bins here
	                        uint64_t position = (startKey & oldMask);
	                        uint64_t position2 = (startKey & innerMask);
	                        if (std::get<0>(hashTable[position2]) < 0) {
	                            std::swap(hashTable[position],
	                                      hashTable[position2]);
	                        } else {
	                            while (std::get<0>(hashTable[position2]) >= 0) {
	                                if (std::get<0>(hashTable[position2]) <
	                                    std::get<0>(hashTable[position])) {
	                                    std::swap(hashTable[position],
	                                              hashTable[position2]);
	                                }
	                                ++std::get<0>(hashTable[position]);
	                                ++position2;
	                                position2 &= innerMask;
	                            }
	                        }

	                        //compact existing hashes
	                        while (std::get<0>(hashTable[position2+1]) > 0) {
	                            hashTable[position2] = hashTable[position2+1];
	                            --std::get<0>(hashTable[position2]);
	                            ++position2;
	                            position2 &= innerMask;
	                        }
	                        hashTable[position2] = {-1,0};
	                        --i;
	                    }
	                }
	            }
	        }
	    };

	    void checkResizeOuter() {
	        //NOTE: if not resizing, this will likely increase the maxdist
	        ;
	    };

	    const std::vector<std::pair<uint64_t,uint32_t>>& getBin(const uint64_t& hashCode) {
	        const uint64_t comparison = hashCode & outerMask;
	        uint64_t position = (multiplier*hashCode % primeFactor);
	        for (int64_t i = 0; i <= maxDist; ++i, ++position) {
	            position &= innerMask;
	            const auto& htp = hashTable[position];
	            if (std::get<0>(htp) < i) {
	                break;
	            } else if ((std::get<0>(
	                        hashBins[std::get<1>(htp)].front()
	                        ) & outerMask) == comparison) {
	                return hashBins[std::get<1>(htp)];
	            }
	        }

	        return emptyBin;
	    };

	    void insertElement(const uint64_t& hashCode, const bool& checkResize=true) {
	        const uint64_t comparison = hashCode & outerMask;
	        uint64_t position = (multiplier*hashCode % primeFactor);
	        for (int32_t i = 0; i < tableSize; ++i, ++position) {
	            position &= innerMask;
	            auto& htp = hashTable[position];
	            if (std::get<0>(htp) < i) { //insert a new entry

	                //insert a new hashtable entry
	                std::pair<int32_t,uint32_t> tmp = {i,hashBins.size()};
	                std::swap(htp,tmp);
	                maxDist = maxDist > i ? maxDist : i;

	                //insert a new hash bin
	                hashBins.push_back(
	                    std::vector<std::pair<uint64_t,uint32_t>>(1,{hashCode,cntr})
	                    );
	                ++cntr;

	                //propagate tmp if needed
	                while (std::get<0>(tmp) >= 0) {
	                    ++position;
	                    position &= innerMask;
	                    ++std::get<0>(tmp);
	                    maxDist = maxDist > std::get<0>(tmp) ? maxDist : std::get<0>(tmp);
	                    std::swap(hashTable[position],tmp);
	                }
	                break;

	            } else if ((std::get<0>( //insert into a populated bin
	                        hashBins[
	                                std::get<1>(htp)
	                                ].front()
	                        ) & outerMask) == comparison) {
	                //XXX: this needs to be sorted in the streaming version
	                hashBins[std::get<1>(htp)].push_back({hashCode,cntr});
	                ++cntr;

	                break;

	            }
	        }

	        if (checkResize) {
	            checkResizeInner();
	            checkResizeOuter();
	        }
	    };

	public:

		ResizeableHashIndex() {
		    ;
		    //binLocations.reserve(800000);
		    };

		ResizeableHashIndex(const arma::uvec& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {
		    //NOTE: this is a non-resizing version of the hashtable
		    outerLoadFactor = innerLoadFactor = 0.65;
		    minRetrieved = queryThreshold;

		    cntr = 0;
		    outerMask = (1ull << bitSize)-1;
	        tableSize = 1;
	        while (tableSize * innerLoadFactor < hashCodes.n_elem) {
	            tableSize *= 2;
	        }
	        tableSize = std::min<uint64_t>((1ull << bitSize),tableSize);
	        innerMask = tableSize-1;
	        maxSize = tableSize;
	        maxDist = 0;
		    hashTable.resize(tableSize,{-1,0});
		    hashBins.reserve(hashCodes.n_elem);

		    //adaptive hash-ball query settings
		    static auto tmp  = getBallDistances(queryDepth,bitSize);
		    binDists = tmp;
		    minRetrieved = queryThreshold;

		    for (const auto& h : hashCodes) {
		        insertElement(h,false);
		    }
		    //std::cout << maxDist << ", " << hashBins.size() << ", " << tableSize << std::endl;
		}

		ResizeableHashIndex(const uint64_t& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {

		    //NOTE: this is a non-resizing version of the hashtable
		    outerLoadFactor = innerLoadFactor = 0.65;
		    minRetrieved = queryThreshold;

		    cntr = 0;
		    outerMask = (1ull << bitSize)-1;
	        tableSize = 1;
	        while (tableSize * innerLoadFactor < hashCodes) {
	            tableSize *= 2;
	        }
	        tableSize = std::min<uint64_t>((1ull << bitSize),tableSize);
	        innerMask = tableSize-1;
	        maxSize = tableSize;
	        maxDist = 0;
		    hashTable.resize(tableSize,{-1,0});
		    hashBins.reserve(hashCodes);

		    //adaptive hash-ball query settings
		    static auto tmp  = getBallDistances(queryDepth,bitSize);
		    binDists = tmp;
		    minRetrieved = queryThreshold;

		}

		//chunked Function 1: countIndices
		void countIndices(const arma::uvec& hashCodes) {
		    //count up all bins
		    for (const auto& h : hashCodes) {
		        insertElement(h,false);
		    }
		    //std::cout << hashBins.size() << ", " << maxDist << std::endl;
		}

		void fillQuery(const uint64_t& hashCode, arma::Col<uint32_t>& results, std::set<uint32_t>& uniqueResults) {
		    //this fills an external query set, making some locality assumptions to help the boosting process.
		    uint64_t cnt = 0;
		    const uint64_t maskedHashCode = hashCode & outerMask;
		    const uint64_t minRet = std::min<uint64_t>(uniqueResults.size()+minRetrieved,results.n_elem);
		    //expand the hash-ball until we meet our minimum query threshold
	        for (const uint64_t& v : binDists.front()) {
				for (const auto& h : getBin(maskedHashCode^v)) {
				    const auto& index = std::get<1>(h);
				    if (not uniqueResults.count(index)) {
						results[uniqueResults.size()] = index;
						uniqueResults.insert(index);
						++cnt;
						if (uniqueResults.size() >= results.n_elem) break;
					}
				}
				if (uniqueResults.size() >= minRet) break;
	        }
		}

		template<class SET = std::set<uint32_t>>
		void fillQuerySet(const uint64_t& hashCode, SET& uniqueResults) {
		    //this fills an external query set, making some locality assumptions to help the boosting process.
		    const uint64_t maskedHashCode = hashCode & outerMask;
		    const uint64_t minRet = uniqueResults.size()+minRetrieved;
		    //expand the hash-ball until we meet our minimum query threshold
	        for (const uint64_t& v : binDists.front()) {
				for (const auto& h : getBin(maskedHashCode^v)) {
				    uniqueResults.insert(std::get<1>(h));
				}
				if (uniqueResults.size() >= minRet) break;
	        }

		}

		uint64_t size() const {
			return cntr;
		}

		/* XXX: I spose we should implement this sometime.
		void saveTable(std::string fileName) {
		    ;
		}*/
};


#endif
