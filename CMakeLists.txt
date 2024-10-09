cmake_minimum_required(VERSION 3.29)
project(homework1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -mtune=native -diag-disable=10441 -static-libstdc++")

find_package(OpenBLAS REQUIRED)
if (NOT OpenBLAS_FOUND)
    message(FATAL_ERROR "OpenBLAS library not found")
endif ()

add_executable(naive sgemm-naive.cpp benchmark.cpp)
add_executable(blas sgemm-blas.cpp benchmark.cpp)
add_executable(blocked sgemm-blocked.cpp benchmark.cpp)
add_executable(blocked-original sgemm-blocked-original.cpp benchmark.cpp)
target_link_libraries(naive PRIVATE ${OpenBLAS_LIBRARIES})
target_link_libraries(blas PRIVATE ${OpenBLAS_LIBRARIES})
target_link_libraries(blocked PRIVATE ${OpenBLAS_LIBRARIES})
target_link_libraries(blocked-original PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(test test.cpp sgemm-blocked.cpp)
target_link_libraries(test PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(testblas test.cpp sgemm-blas.cpp)
target_link_libraries(testblas PRIVATE ${OpenBLAS_LIBRARIES})
