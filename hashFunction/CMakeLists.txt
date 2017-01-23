add_library(nulltransformer NullTransformer.cpp NullTransformer.h)
add_library(meantransformer MeanTransformer.cpp MeanTransformer.h)
add_library(thresholdtransformer ThresholdTransformer.cpp ThresholdTransformer.h)
add_library(pcatransformer PCATransformer.cpp PCATransformer.h)
add_library(dhhashtransformer DHHashTransformer.cpp DHHashTransformer.h)

target_link_libraries(nulltransformer
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(meantransformer
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(thresholdtransformer
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(pcatransformer
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(dhhashtransformer
                    ${ARMADILLO_LIBRARIES})

add_library(randomprojectionhashfunction RandomProjectionHashFunction.cpp RandomProjectionHashFunction.h)
add_library(randomsubsamplinghashfunction RandomSubSamplingHashFunction.cpp RandomSubSamplingHashFunction.h)
add_library(shiftinvariantkernelhashfunction ShiftInvariantKernelHashFunction.cpp ShiftInvariantKernelHashFunction.h)
add_library(dhhashfunction DHHashFunction.cpp DHHashFunction.h)
add_library(randomrotationhashfunction RandomRotationHashFunction.cpp RandomRotationHashFunction.h)

target_link_libraries(randomprojectionhashfunction
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(randomsubsamplinghashfunction
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(shiftinvariantkernelhashfunction
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(dhhashfunction
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(randomrotationhashfunction
                    ${ARMADILLO_LIBRARIES})