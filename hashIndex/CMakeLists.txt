#add_library(hashindexkgraph HashIndexKgraph.cpp HashIndexKGraph.h)
add_library(lshforesthashindex LSHForestHashIndex.cpp LSHForestHashIndex.h)
add_library(lshforesthashindexasync LSHForestHashIndexAsync.cpp LSHForestHashIndexAsync.h)
add_library(resizeablehashindex ResizeableHashIndex.cpp ResizeableHashIndex.h)
add_library(stdhashindex StdHashIndex.cpp StdHashIndex.h)

#target_link_libraries(hashindexkgraph
#                    ${ARMADILLO_LIBRARIES})
target_link_libraries(lshforesthashindex
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(lshforesthashindexasync
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(resizeablehashindex
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(stdhashindex
                    ${ARMADILLO_LIBRARIES})

add_library(hashcollection HashCollection.cpp HashCollection.h)


target_link_libraries(hashcollection
#                    hashindexkgraph
                    lshforesthashindex
                    lshforesthashindexasync
                    resizeablehashindex
                    stdhashindex
                    ${ARMADILLO_LIBRARIES})