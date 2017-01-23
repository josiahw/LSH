#ifndef LSHFORESTHASHINDEX_H
#define LSHFORESTHASHINDEX_H
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
#include "LSHForestSupport.h"

//template<class DataTransformer, class HashFunction>
class LSHForestHashIndex {
	private:
	    uint64_t minRetrieved,
	             nBits,
	             cntr,
	             rootNode;


		std::vector<HashNode> hashTree;

	    int32_t insertElement(const uint64_t& hashCode,
	                           const int32_t& node,
	                           const uint64_t& length = 0) {
	        int32_t newNode = node;
	        //we need to create a new node
	        if (node == -1) {
	            newNode = hashTree.size();
	            hashTree.push_back({hashCode,
                                    64,
                                    -1,
                                    -1,
                                    std::vector<uint32_t>(1,cntr)
                                    });
	            ++cntr;
	            return hashTree.size()-1;
	        //we've hit a child node with no branches
	        } else if (hashTree[node].children.size() > 0) {

	            if (hashCode == hashTree[node].prefix) {
	                //hashTree[node].children.resize(hashTree[node].children.size() + 1);
	                hashTree[node].children.push_back(cntr);
	                ++cntr;
	                return node;
	            } else {
	                newNode = hashTree.size();

	                //find new prefix length
	                uint32_t newLength = length;
	                while (newLength < 63) {
	                    if ( (hashCode >> (64 - newLength)) !=
	                         (hashTree[node].prefix >> (64 - newLength))) {
	                        break;
	                    }
	                    ++newLength;
	                }

	                //add a new child node
	                hashTree.push_back({hashCode,
                                        64,
                                        -1,
                                        -1,
                                        std::vector<uint32_t>(1,cntr)
                                        });
	                ++cntr;

	                //check which side the new child node should be inserted on
	                if ( (hashCode >> (64 - newLength)) & 1 ) {
	                    hashTree.push_back({hashCode,
                                            newLength,
                                            newNode,
                                            node,
                                            std::vector<uint32_t>(0)
                                            });
	                } else {
	                    hashTree.push_back({hashCode,
                                            newLength,
                                            node,
                                            newNode,
                                            std::vector<uint32_t>(0)
                                            });
	                }
	                return hashTree.size()-1;
	            }


	        //we've hit a decision node and need to insert a node above it.
	        } else if (
	                   (hashTree[node].prefix/2ull >> (64 - hashTree[node].length)) !=
	                   (hashCode/2ull >> (64 - hashTree[node].length))
	                   ) {
	            newNode = hashTree.size();

                //find new prefix length
                uint32_t newLength = length;
                while (newLength < hashTree[node].length-1) {
                    if ( (hashCode >> (64 - newLength)) !=
                         (hashTree[node].prefix >> (64 - newLength))) {
                        break;
                    }
                    ++newLength;
                }
                //std::cout << length << ", " << newLength << ", " << hashTree[node].length << ", " << (hashCode >> (64 - hashTree[node].length)) << ", " << (hashTree[node].prefix >> (64 - hashTree[node].length)) << ", " << hashCode << ", " << hashTree[node].prefix << std::endl;

                //add a new child node
                hashTree.push_back({hashCode,
                                    64,
                                    -1,
                                    -1,
                                    std::vector<uint32_t>(1,cntr)
                                    });
                ++cntr;

                if ( (hashCode >> (64 - newLength)) & 1) {
                    hashTree.push_back({hashCode,
                                        newLength,
                                        newNode,
                                        node,
                                        std::vector<uint32_t>(0)
                                        });
                } else {
                    hashTree.push_back({hashCode,
                                        newLength,
                                        node,
                                        newNode,
                                        std::vector<uint32_t>(0)
                                        });
                }
                return hashTree.size()-1;


	        //we've hit a decision node and should descend
	        } else {
	            uint64_t l = hashTree[node].length+1;
	            int32_t n;
	            if ( (hashCode >> (64 - hashTree[node].length)) & 1 ) {
	                n = hashTree[node].leftChild;
	                hashTree[node].leftChild = insertElement(hashCode,
	                                                         n,
	                                                         l);
	            } else {
	                n = hashTree[node].rightChild;
	                hashTree[node].rightChild = insertElement(hashCode,
                                                              n,
                                                              l);
	            }
	            return node;
	        }

	        return newNode;
	    };

	    //collect all values in a subtree
	    void collect(const int32_t& node, arma::Col<uint32_t>& results, std::set<uint32_t>& uniqueResults) {
            if (uniqueResults.size() >= results.n_elem) {
                return;
            }
	        if (hashTree[node].children.size() > 0) {
	            for (const auto& c : hashTree[node].children) {
	                if (not uniqueResults.count(c)) {
						results[uniqueResults.size()] = c;
						uniqueResults.insert(c);
	                    if (uniqueResults.size() >= results.n_elem) {
	                        return;
	                    }
	                }
	            }
	        } else {
	            collect(hashTree[node].leftChild,results,uniqueResults);
	            collect(hashTree[node].rightChild,results,uniqueResults);
	        }
	    }

	    //collect all values in a subtree
	    template<class SET = std::set<uint32_t>>
	    void collect(const int32_t& node, SET& uniqueResults, const uint64_t& maxSize) {
            if (uniqueResults.size() >= maxSize) {
                return;
            }

	        if (hashTree[node].children.size() > 0) {
	            for (const auto& c : hashTree[node].children) {
	                if (not uniqueResults.count(c)) {
						uniqueResults.insert(c);

	                    if (uniqueResults.size() >= maxSize) {
	                        return;
	                    }
	                }
	            }
	        } else {
	            collect(hashTree[node].leftChild,uniqueResults,maxSize);
	            collect(hashTree[node].rightChild,uniqueResults,maxSize);
	        }
	    }

	public:
		uint64_t qDepth;
		LSHForestHashIndex() {
		    ;
		    };

		LSHForestHashIndex(const arma::uvec& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {
		    minRetrieved = queryThreshold;
		    qDepth = queryDepth;
		    nBits = 64-bitSize;
		    cntr = 0;
	        //maxDist = 0;

		    minRetrieved = queryThreshold;

		    hashTree.reserve(std::min<uint64_t>(2*hashCodes.n_elem,2ull<<bitSize));
		    rootNode =  -1;
		    for (const auto& h : hashCodes) {
		        rootNode = insertElement(h,rootNode,nBits);
		    }
		}

		LSHForestHashIndex(const uint64_t& hashCodes,
		          const uint64_t queryThreshold=10,
		          const uint64_t queryDepth=3,
		          const uint64_t bitSize = 32) {

		    minRetrieved = queryThreshold;
		    qDepth = queryDepth;
		    nBits = 64-bitSize;
		    cntr = 0;
	        //maxDist = 0;

		    minRetrieved = queryThreshold;

		    rootNode = -1;
		    hashTree.reserve(std::min<uint64_t>(2*hashCodes,2ull<<bitSize));

		}

		//chunked function for inserting elements
		void countIndices(const arma::uvec& hashCodes) {
		    for (const auto& h : hashCodes) {
		        rootNode = insertElement(h,rootNode,nBits);
		        //std::cout << rootNode << std::endl;
		    }
		}

		void insert(const uint64_t& hashCode) {
		    rootNode = insertElement(hashCode,rootNode,nBits);
		    //std::cout << rootNode << std::endl;
		}

		void fillQuery(const uint64_t& hashCode, arma::Col<uint32_t>& results, std::set<uint32_t>& uniqueResults, std::vector<int32_t>& nodevals, const uint64_t& qDepth) {
		    //1: descend the tree to the closest leaf, saving nodes as we go
		    int32_t counter = 0;
		    if (qDepth == 0) {
		        nodevals.reserve(64);
		        nodevals.push_back(rootNode);
		        while (hashTree[nodevals[counter]].children.size() == 0) {
		            const auto& hc = hashTree[nodevals[counter]];
		            ++counter;
		            if ( hashCode >> (64 - hc.length) & 1) {
		                nodevals.push_back(hc.leftChild);
		            } else {
		                nodevals.push_back(hc.rightChild);
		            }
		        }

		        //2: add immediate neighbours
		        collect(nodevals[counter], results, uniqueResults);
		    } else {
		        counter = nodevals.size()-1;

		        //3: ascend up the tree, adding alternate branches until we hit the return requirement
		        while (counter > 0
		               //and minRet > uniqueResults.size()
		               and 64-hashTree[nodevals[counter]].length < qDepth) {
		            --counter;

		        }
		        if (64-hashTree[nodevals[counter]].length == qDepth and counter < nodevals.size()-1) {
		            (hashTree[nodevals[counter]].leftChild == nodevals[counter+1]) ?
                            collect(hashTree[nodevals[counter]].rightChild, results, uniqueResults) :
                            collect(hashTree[nodevals[counter]].leftChild, results, uniqueResults);
                }
		    }
		}

		template<class SET = std::set<uint32_t>>
		void fillQuerySet(const uint64_t& hashCode, SET& uniqueResults, std::vector<int32_t>& nodevals, const uint64_t& qDepth, const uint64_t& maxSize) {

		    //1: descend the tree to the closest leaf, saving nodes as we go
		    int32_t counter = 0;
		    if (qDepth == 0) {
		        nodevals.reserve(64);
		        nodevals.push_back(rootNode);
		        while (hashTree[nodevals[counter]].children.size() == 0) {
		            const auto& hc = hashTree[nodevals[counter]];


		            //std::cout << hashTree[hc.leftChild].children.size() << ", " << hashTree[hc.rightChild].children.size() << ", " << 64-hc.length << ", " << counter << ", " << hashTree[hc.leftChild].prefix << ", " << hashTree[hc.rightChild].prefix << ", " << nodevals[counter] << std::endl;
		            ++counter;
		            if ( hashCode >> (64 - hc.length) & 1) {
		                nodevals.push_back(hc.leftChild);
		            } else {
		                nodevals.push_back(hc.rightChild);
		            }

		        }
		        //std::cout << hashTree[nodevals[counter]].prefix << ", " << hashCode << ", " << hashTree[nodevals[counter]].children.size() << ", " << hashTree[nodevals[counter]].length << ", " << hashTree[nodevals[counter]].leftChild << ", " << hashTree[nodevals[counter]].rightChild << ", " << nodevals[counter] << std::endl;
		        //2: add immediate neighbours
		        collect(nodevals[counter], uniqueResults,maxSize);

		    } else {
	            counter = nodevals.size()-1;

	            //3: ascend up the tree, adding alternate branches until we hit the return requirement

	            while (counter > 0
	                   and 64-hashTree[nodevals[counter]].length < qDepth) {
	                --counter;
	            }
	            //std::cout << qDepth << ", " << 64-hashTree[nodevals[counter]].length << ", " << counter << ", " << hashTree.size() << ", " << nodevals[counter] << std::endl;
	            if (64-hashTree[nodevals[counter]].length == qDepth and counter < nodevals.size()-1) {
                (hashTree[nodevals[counter]].leftChild == nodevals[counter+1]) ?
                        collect(hashTree[nodevals[counter]].rightChild, uniqueResults,maxSize) :
                        collect(hashTree[nodevals[counter]].leftChild, uniqueResults,maxSize);
                }
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
