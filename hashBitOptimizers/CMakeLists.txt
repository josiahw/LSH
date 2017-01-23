add_library(boost Boost.h Boost.cpp)
add_library(rdhf RDHF.h RDHF.cpp)
add_library(ardhf ARDHF.h ARDHF.cpp)
add_library(randomconstructor RandomConstructor.h RandomConstructor.cpp)


target_link_libraries(boost
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(rdhf
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(ardhf
                    ${ARMADILLO_LIBRARIES})
target_link_libraries(randomconstructor
                    ${ARMADILLO_LIBRARIES})