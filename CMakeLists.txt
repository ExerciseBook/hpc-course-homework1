cmake_minimum_required(VERSION 3.29)
project(homework1)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -mtune=native")

# Find OpenBLAS
find_package(OpenBLAS REQUIRED)
if (NOT OpenBLAS_FOUND)
    message(FATAL_ERROR "OpenBLAS library not found")
endif ()
set(OpenBLAS_LIBRARIES OpenBLAS::OpenBLAS)

# Find MKL
#find_package(MKL REQUIRED)
#if (NOT MKL_FOUND)
#    message(FATAL_ERROR "MKL library not found")
#endif()

# Set MKL variables manually
set(MKL_LIBRARIES
    # "/home/software/spack/opt/spack/linux-rhel7-cascadelake/gcc-10.4.0/intel-oneapi-mkl-2022.2.1-mq6qsk6obdy5xj3z5oq7jhej4rc533yy/mkl/2022.2.1/lib/intel64/libmkl_intel_ilp64.so"
    # "/home/software/spack/opt/spack/linux-rhel7-cascadelake/gcc-10.4.0/intel-oneapi-mkl-2022.2.1-mq6qsk6obdy5xj3z5oq7jhej4rc533yy/mkl/2022.2.1/lib/intel64/libmkl_core.so"
    # "/home/software/spack/opt/spack/linux-rhel7-cascadelake/gcc-10.4.0/intel-oneapi-mkl-2022.2.1-mq6qsk6obdy5xj3z5oq7jhej4rc533yy/mkl/2022.2.1/lib/intel64/libmkl_intel_thread.so"
    # "/home/software/spack/opt/spack/linux-rhel7-cascadelake/gcc-10.4.0/intel-oneapi-mkl-2022.2.1-mq6qsk6obdy5xj3z5oq7jhej4rc533yy/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so"
)

add_executable(naive sgemm-naive.cpp benchmark.cpp)
add_executable(blas sgemm-blas.cpp benchmark.cpp)
add_executable(blocked sgemm-blocked.cpp benchmark.cpp)
add_executable(blocked-original sgemm-blocked-original.cpp benchmark.cpp)
target_link_libraries(naive PRIVATE ${OpenBLAS_LIBRARIES} ${MKL_LIBRARIES})
target_link_libraries(blas PRIVATE ${OpenBLAS_LIBRARIES} ${MKL_LIBRARIES})
target_link_libraries(blocked PRIVATE ${OpenBLAS_LIBRARIES} ${MKL_LIBRARIES})
target_link_libraries(blocked-original PRIVATE ${OpenBLAS_LIBRARIES} ${MKL_LIBRARIES})

add_executable(test test.cpp sgemm-blocked.cpp)
target_link_libraries(test PRIVATE ${OpenBLAS_LIBRARIES} ${MKL_LIBRARIES})