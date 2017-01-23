add_library(testlsh TestLSH.h TestLSH.cpp)


target_link_libraries(testlsh
                    ${ARMADILLO_LIBRARIES})