#include "matrix_config.h"
#include "matrix_helper.h"
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

const char* sgemm_desc = "Simple blocked sgemm.";

#define UNROLLING_LANE 16

void do_block(
        // clang-format off
        int M, int K, int N,
        int strideA, int strideB, int strideC,
        float *A, float *B, float *C,

        float* packed_A
        // clang-format on
) {
    for (int j = 0; j < N; ++j) {
        int i = 0;

        for (; i + 3 < M; i += 4) {
            // std::cout << "j: " << j << ", i:" << i << std::endl;
            int ip1 = i + 1;
            int ip2 = i + 2;
            int ip3 = i + 3;

            float cij = C(i, j);
            float cip1j = C(ip1, j);
            float cip2j = C(ip2, j);
            float cip3j = C(ip3, j);

            for (int k = 0; k < K; ++k) {
                // 所以 A的 i行k列 被映射到了 packA的 ((i / 4) * K + k) * 4 + i % 4
                auto p = ((i / 4) * K + k) * 4;
                // std::cout << "p:" << ((i / 4) * K + k) * 4 << ", packed_A[p]:" << packed_A[p] << std::endl;
                cij += packed_A[p] * B[k + strideB * j];
                cip1j += packed_A[p + 1] * B[k + strideB * j];
                cip2j += packed_A[p + 2] * B[k + strideB * j];
                cip3j += packed_A[p + 3] * B[k + strideB * j];
            }

            C(i, j) = cij;
            C(ip1, j) = cip1j;
            C(ip2, j) = cip2j;
            C(ip3, j) = cip3j;
        }

        for (; i < M; ++i) {
            float cij = C(i, j);
            for (int k = 0; k < K; ++k) {
                cij += packed_A[ ((i / 4) * K + k) * 4 + i % 4] * B[k + strideB * j];
            }
            C(i, j) = cij;
        }
    }
}

// dst 大小为 BLOCK_M * BLOCK_K 这么大
// src 是一个数组，他存储的内容为 A E I M B F J N C G K O D H L P Q U Y 3 R V Z 4 S W 1 5 T X 2 6
// src 的内容解释为
// A B C D Q R S T
// E F G H U V W X
// I J K L Y Z 1 2
// M N O P 3 4 5 6
//
// dst 的目标内容为
// A E I M B F J N C G K O D H L P Q U Y 3 R V Z 4 S W 1 5 T X 2 6
//
// 可以看到，dst 的意思是把 src 按照 4 行 4 行展开
// 行,列 -> dst 下标
//
// 当 src 的行列不能被4整除时我们这样操作
// src 是一个数组，他存储的内容为 A E I B F J C G K D H L M O Q N P R
// src 的内容解释为
// A B C D M N 0 0
// E F G H O P 0 0
// I J K L Q R 0 0
// 0 0 0 0 0 0 0 0
//
// dst 的目标内容为
// A E I 0 B F J 0 C G K 0 D H L 0 M O Q 0 N P R 0 0 0 0 0 0 0 0 0
//
// 一个混合的例子
// src 是一个数组，他存储的内容为 A E I B F J C G K D H L M O Q N P R
// src 的内容解释为
// A B C D 3 4 0 0
// E F G H 5 6 0 0
// I J K L 7 8 0 0
// M N O P 9 a 0 0
// Q R S T b c 0 0
// U V W X d e 0 0
// Y Z 1 2 f g 0 0
// 0 0 0 0 0 0 0 0
//
// dst 的目标内容为
// A E I M B F J N C G K O D H L P 3 5 7 9 4 6 8 a 0 0 0 0 0 0 0 0 Q U Y Z R V Z 0 S W 1 0 T X 2 0 b d f 0 c e g 0 0 0 0 0 0 0 0 0
void packA(float* dst, int srcStride, int srcHeight, int srcWidth, float* src) {
    for (int i = 0; i < srcHeight; i++) {
        for (int j = 0; j < srcWidth; j++) {
            dst[((i / 4) * srcWidth + j) * 4 + i % 4] = src[srcStride * j + i];
            // dst[srcWidth * i + j] = src[srcStride * j + i];
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
            packA(packed_A, strideA, M0, K0, &A(m, k));
            // print_matrix(&A(k, m), strideA, M0, K0);
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

    // throw std::runtime_error("");
}
