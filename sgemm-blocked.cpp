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

    float* packed_B
        // clang-format on
) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float cij = C(i, j);
            for (int k = 0; k < K; ++k) {
                // cij += A(i, k) * B[k + strideB * j] ;
                cij += A(i, k) * packed_B[j + k * BLOCK_N];
            }
            C(i, j) = cij;
        }
    }
}

// 原来B的 i行j列 被打包到了 j行i列
void packB(float* dst, int row, int width, int stride, float* src) {
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

    float* packed_B;
    if (posix_memalign((void**) &packed_B, 4096, BLOCK_N * BLOCK_K * sizeof(float)) != 0) {
        throw std::runtime_error("Can not align malloc packed pool!");
    }

    // 遍历 B 的每一行
    for (int n = 0; n < N; n += BLOCK_N) {
        for (int k = 0; k < K; k += BLOCK_K) {
            packB(packed_B, BLOCK_K, BLOCK_N, strideB, &B(k, n));
            // print_matrix(&B(k, n), strideB, BLOCK_K, BLOCK_N);
            // print_matrix(packed_B, BLOCK_K, BLOCK_N, BLOCK_K);

            for (int m = 0; m < M; m += BLOCK_M) {
                int M0 = min(BLOCK_M, M - m);
                int K0 = min(BLOCK_K, K - k);
                int N0 = min(BLOCK_N, N - n);

                // 优先遍历 B 来减少转置 B 时的开销
                do_block(M0, K0, N0, strideA, strideB, strideC, &(A(m, k)), &(B(k, n)), &(C(m, n)), packed_B);
            }
        }
    }
}
