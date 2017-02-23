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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <iostream>
#include <sstream>

//hash functions
#include "RandomProjectionHashFunction.h"
#include "RandomSubSamplingHashFunction.h"
#include "ShiftInvariantKernelHashFunction.h"
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

//hash collection constructors
#include "RandomConstructor.h"
#include "RDHF.h"
#include "ARDHF.h"
#include "Boost.h"

//unit test class
#include "TestLSH.h"


#include <iostream>

void HammingRadiusBenchmarkData(const arma::Mat<float>& data, const arma::Mat<float>& dataqueries) {
    const size_t datasize = data.n_cols;
    const size_t querysize = dataqueries.n_cols;
    const size_t datadim = data.n_rows;

    size_t hashFuncs = 100;
    size_t hashBits = 32;
    size_t maxQuerySize = 1000000;
    size_t maxSearchThreshold = 3;
    size_t minRetrievalThreshold = maxQuerySize;
    size_t maxRetrievalThreshold = maxQuerySize;
    size_t knn = 100;

    for (maxSearchThreshold = 1; maxSearchThreshold < 8; ++maxSearchThreshold) {
      //Section 1: test hash indices
      std::cout << "Testing Random Projection Hashing with STDHashIndex" << std::endl;
      std::cout << "Search Radius: " << maxSearchThreshold << std::endl;
      auto results = TestLSH<RandomConstructor<MeanTransformer,
                                              RandomProjectionHashFunction,
                                              StdHashIndex,
                                              HashCollection<MeanTransformer,
                                                             RandomProjectionHashFunction,
                                                             StdHashIndex,
                                                             arma::Mat<float>>,
                                              arma::Mat<float>>,
                             MeanTransformer,
                             RandomProjectionHashFunction,
                             StdHashIndex,
                             arma::Mat<float>
                              >::TestLSHTimings(data,
                                       dataqueries,
                                       arma::Mat<uint32_t>(), //groundtruth not used for random
                                       datadim,
                                       datasize,
                                       querysize,
                                       hashFuncs,
                                       hashBits,
                                       hashFuncs, //candidate hash functs not used for random
                                       maxSearchThreshold,
                                       maxQuerySize,
                                       maxRetrievalThreshold,
                                       minRetrievalThreshold,
                                       knn
                                      );
      auto results2 = TestLSH<RandomConstructor<MeanTransformer,
                                              RandomProjectionHashFunction,
                                              StdHashIndex,
                                              HashCollection<MeanTransformer,
                                                             RandomProjectionHashFunction,
                                                             StdHashIndex,
                                                             arma::Mat<float>>,
                                              arma::Mat<float>>,
                             MeanTransformer,
                             RandomProjectionHashFunction,
                             StdHashIndex,
                             arma::Mat<float>
                              >::TestLSHAccuracy(data,
                                       dataqueries,
                                       arma::Mat<uint32_t>(), //groundtruth not used for random
                                       datadim,
                                       datasize,
                                       querysize,
                                       hashFuncs,
                                       hashBits,
                                       hashFuncs, //candidate hash functs not used for random
                                       maxSearchThreshold,
                                       maxQuerySize,
                                       maxRetrievalThreshold,
                                       minRetrievalThreshold,
                                       knn
                                      );

      std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
      std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
      std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
      std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
      std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;


      std::cout << "Testing Random Projection Hashing with Robin Hood Hash Index" << std::endl;
      std::cout << "Search Radius: " << maxSearchThreshold << std::endl;
      results = TestLSH<RandomConstructor<MeanTransformer,
                                              RandomProjectionHashFunction,
                                              ResizeableHashIndex,
                                              HashCollection<MeanTransformer,
                                                             RandomProjectionHashFunction,
                                                             ResizeableHashIndex,
                                                             arma::Mat<float>>,
                                              arma::Mat<float>>,
                             MeanTransformer,
                             RandomProjectionHashFunction,
                             ResizeableHashIndex,
                             arma::Mat<float>
                              >::TestLSHTimings(data,
                                       dataqueries,
                                       arma::Mat<uint32_t>(), //groundtruth not used for random
                                       datadim,
                                       datasize,
                                       querysize,
                                       hashFuncs,
                                       hashBits,
                                       hashFuncs, //candidate hash functs not used for random
                                       maxSearchThreshold,
                                       maxQuerySize,
                                       maxRetrievalThreshold,
                                       minRetrievalThreshold,
                                       knn
                                      );
      results2 = TestLSH<RandomConstructor<MeanTransformer,
                                              RandomProjectionHashFunction,
                                              ResizeableHashIndex,
                                              HashCollection<MeanTransformer,
                                                             RandomProjectionHashFunction,
                                                             ResizeableHashIndex,
                                                             arma::Mat<float>>,
                                              arma::Mat<float>>,
                             MeanTransformer,
                             RandomProjectionHashFunction,
                             ResizeableHashIndex,
                             arma::Mat<float>
                              >::TestLSHAccuracy(data,
                                       dataqueries,
                                       arma::Mat<uint32_t>(), //groundtruth not used for random
                                       datadim,
                                       datasize,
                                       querysize,
                                       hashFuncs,
                                       hashBits,
                                       hashFuncs, //candidate hash functs not used for random
                                       maxSearchThreshold,
                                       maxQuerySize,
                                       maxRetrievalThreshold,
                                       minRetrievalThreshold,
                                       knn
                                      );

      std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
      std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
      std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
      std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
      std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;
    }
    for (maxSearchThreshold = 1; maxSearchThreshold < 12; ++maxSearchThreshold) {
      std::cout << "Testing Random Projection Hashing with LSHForest" << std::endl;
      std::cout << "Search Radius: " << maxSearchThreshold << std::endl;
      auto results = TestLSH<RandomConstructor<MeanTransformer,
                                              RandomProjectionHashFunction,
                                              LSHForestHashIndex,
                                              HashCollection<MeanTransformer,
                                                             RandomProjectionHashFunction,
                                                             LSHForestHashIndex,
                                                             arma::Mat<float>>,
                                              arma::Mat<float>>,
                             MeanTransformer,
                             RandomProjectionHashFunction,
                             LSHForestHashIndex,
                             arma::Mat<float>
                              >::TestLSHTimings(data,
                                       dataqueries,
                                       arma::Mat<uint32_t>(), //groundtruth not used for random
                                       datadim,
                                       datasize,
                                       querysize,
                                       hashFuncs,
                                       hashBits,
                                       hashFuncs, //candidate hash functs not used for random
                                       maxSearchThreshold,
                                       maxQuerySize,
                                       maxRetrievalThreshold,
                                       minRetrievalThreshold,
                                       knn
                                      );
      auto results2 = TestLSH<RandomConstructor<MeanTransformer,
                                              RandomProjectionHashFunction,
                                              LSHForestHashIndex,
                                              HashCollection<MeanTransformer,
                                                             RandomProjectionHashFunction,
                                                             LSHForestHashIndex,
                                                             arma::Mat<float>>,
                                              arma::Mat<float>>,
                             MeanTransformer,
                             RandomProjectionHashFunction,
                             LSHForestHashIndex,
                             arma::Mat<float>
                              >::TestLSHAccuracy(data,
                                       dataqueries,
                                       arma::Mat<uint32_t>(), //groundtruth not used for random
                                       datadim,
                                       datasize,
                                       querysize,
                                       hashFuncs,
                                       hashBits,
                                       hashFuncs, //candidate hash functs not used for random
                                       maxSearchThreshold,
                                       maxQuerySize,
                                       maxRetrievalThreshold,
                                       minRetrievalThreshold,
                                       knn
                                      );

      std::cout << "Hash functions constructed " << std::get<0>(results) << std::endl;
      std::cout << "Hash Database built in " << std::get<1>(results) << " seconds." << std::endl;
      std::cout << "whole database searched in " << std::get<2>(results) << std::endl;
      std::cout << "Mean recall: " << arma::mean(arma::max(results2.first.t(),1));
      std::cout << "Mean Query Size: " << arma::mean(arma::conv_to<arma::mat>::from(arma::index_max(results2.first.t(),1))) / 20 * maxQuerySize;
  }

}


int main (void) {
  //constant values for the GIST dataset
  size_t width = 960;
  size_t dblength = 1000000;
  size_t querylength = 500;

  //mmap the database and query values
  int dbfd;
  float* dbfmap;
  uint64_t fsize = width * dblength * sizeof(float);
  dbfd = open(std::string("gist_mmapready").c_str(), O_RDWR);
  dbfmap = reinterpret_cast<float*>(mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, dbfd, 0));
  if (dbfmap == MAP_FAILED) {
      close(dbfd);
      perror("Error mmapping the file");
      exit(EXIT_FAILURE);
  }
  arma::Mat<float> database = arma::Mat<float>(dbfmap, width, dblength, false);

  int qfd;
  float* qfmap;
  uint64_t qfsize = width * querylength * sizeof(float);
  qfd = open(std::string("gist_queries_mmapready").c_str(), O_RDWR);
  qfmap = reinterpret_cast<float*>(mmap(NULL, qfsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, qfd, 0));
  if (qfmap == MAP_FAILED) {
      close(qfd);
      perror("Error mmapping the file");
      exit(EXIT_FAILURE);
  }
  arma::Mat<float> queries = arma::Mat<float>(qfmap, width, querylength, false);
  std::cout << "Benchmarking GIST data" << std::endl;
  HammingRadiusBenchmarkData(database, queries);

  munmap(dbfmap, fsize);
  close(dbfd);
  munmap(qfmap, qfsize);
  close(qfd);

  //constant values for the SIFT dataset
  width = 128;
  dblength = 1000000;
  querylength = 500;

  //mmap the database and query values
  fsize = width * dblength * sizeof(float);
  dbfd = open(std::string("sift_mmapready").c_str(), O_RDWR);
  dbfmap = reinterpret_cast<float*>(mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, dbfd, 0));
  if (dbfmap == MAP_FAILED) {
      close(dbfd);
      perror("Error mmapping the file");
      exit(EXIT_FAILURE);
  }
  database = arma::Mat<float>(dbfmap, width, dblength, false);

  qfsize = width * querylength * sizeof(float);
  qfd = open(std::string("sift_queries_mmapready").c_str(), O_RDWR);
  qfmap = reinterpret_cast<float*>(mmap(NULL, qfsize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_NORESERVE, qfd, 0));
  if (qfmap == MAP_FAILED) {
      close(qfd);
      perror("Error mmapping the file");
      exit(EXIT_FAILURE);
  }
  queries = arma::Mat<float>(qfmap, width, querylength, false);
  std::cout << "Benchmarking SIFT data" << std::endl;
  HammingRadiusBenchmarkData(database, queries);

  munmap(dbfmap, fsize);
  close(dbfd);
  munmap(qfmap, qfsize);
  close(qfd);

  return 0;
};
