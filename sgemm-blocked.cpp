#include "matrix_config.h"
#include "matrix_helper.h"
#include <immintrin.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char* sgemm_desc = "Simple blocked sgemm.";


void do_block(
        // clang-format off
        int M, int K, int N,
        int strideA, int strideB, int strideC,
        float *A, float *B, float *C,

        float* packed_A
        // clang-format on
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float cij = C(i, j);
            for (int k = 0; k < K; ++k) {
                // 所以 A的 i行k列 被映射到了 packB的 k行i列
                // cij += A[i + strideA* k]* B[k + strideB * j] ;
                cij += packed_A[k + i * BLOCK_K] * B[k + strideB * j];
            }
            C(i, j) = cij;
        }
    }
}

// 原来A的 i行j列 被打包到了 j行i列
void packA(float* dst, int row, int width, int stride, float* src) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < width; j++) {
            dst[width * i + j] = src[stride * j + i];
        }
    }
}

void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {
    // A
    //  M行 K列
    auto strideA = M;

    // B
    //  K行 N列
    auto strideB = K;

    // C
    //  M行 N列
    auto strideC = M;

#if WIN32
    float* packed_A = (float*) _aligned_malloc(BLOCK_M * BLOCK_K * sizeof(float), 4096);
    if (!packed_A) {
        throw std::runtime_error("Can not align malloc packed pool!");
    }
#else
    float* packed_A = nullptr;
    if (posix_memalign((void**) &packed_A, 4096, BLOCK_M * BLOCK_K * sizeof(float)) != 0) {
        throw std::runtime_error("Can not align malloc packed pool!");
    }
#endif
    for (int k = 0; k < K; k += BLOCK_K) {
        int K0 = min(BLOCK_K, K - k);
        for (int m = 0; m < M; m += BLOCK_M) {
            int M0 = min(BLOCK_M, M - m);
            packA(packed_A, BLOCK_M, BLOCK_K, strideA, &A(m, k));
            // print_matrix(&A(k, m), strideA, BLOCK_M, BLOCK_K);
            // print_matrix(packed_A, BLOCK_K, BLOCK_K, BLOCK_M);

            for (int n = 0; n < N; n += BLOCK_N) {
                int N0 = min(BLOCK_N, N - n);
                do_block(M0, K0, N0, strideA, strideB, strideC, &(A(m, k)), &(B(k, n)), &(C(m, n)), packed_A);
            }
        }
    }

#if WIN32
    _aligned_free(packed_A);
#else
    free(packed_A);
#endif
}
