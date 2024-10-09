#include <cstdint>
#include <iostream>

auto inline min(auto a, auto b) {
    return a < b ? a : b;
}

void inline print_array(float* a, int length) {
    for (int i = 0; i < length; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

void inline print_matrix(float* matrix, int stride, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i + j * stride] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#define matidx(M, i, j) M[(i) + stride##M * (j)]
#define A(i, j) matidx(A, i, j)
#define B(i, j) matidx(B, i, j)
#define C(i, j) matidx(C, i, j)

void* alignedMalloc(size_t size, size_t align = 4096) {
#if WIN32
    void* ret = _aligned_malloc(size, align);
    if (!ret) {
        throw std::runtime_error("Can not align malloc packed pool!");
    }
#else
    void* ret = nullptr;
    if (posix_memalign(&ret, align, size) != 0) {
        throw std::runtime_error("Can not align malloc packed pool!");
    }
#endif
    return ret;
}

void alignedFree(void* ptr) {
#if WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}