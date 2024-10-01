#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const char* sgemm_desc = "Simple blocked sgemm.";

constexpr size_t BLOCK_ROW = 64;
constexpr size_t BLOCK_COL = 128;

constexpr size_t UNROLL_NUM = 4;

constexpr size_t SIMD_UNROLL = 32;

auto inline min(auto a, auto b) {
    return a < b ? a : b;
}

static void print_array(float* a, int length) {
    for (int i = 0; i < length; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

static void print_matrix(float* matrix, int stride, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i + j * stride] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static void pack_left_A(int K, int stride, float* A, float* packed_A) {
    float* dst = packed_A;
    int k = 0;
    for (; k < K; k++) {
        *(dst + 0) = A[0 + k * stride];
        *(dst + 1) = A[1 + k * stride];
        *(dst + 2) = A[2 + k * stride];
        *(dst + 3) = A[3 + k * stride];
        dst += 4;
    }
}

static void pack_right_B(int K, int stride, float* B, float* packed_B) {
    float* dst = packed_B;
    int k = 0;
    for (; k < K; ++k) {
        *(dst + 0) = B[k + 0 * stride];
        *(dst + 1) = B[k + 1 * stride];
        *(dst + 2) = B[k + 2 * stride];
        *(dst + 3) = B[k + 3 * stride];
        dst += 4;
    }
}
void do_mul_square(int n, int strideA, int strideB, int strideC, float* A, float* B, float* C) {
    float c00 = C[0 + 0 * strideC];
    float c01 = C[0 + 1 * strideC];
    float c02 = C[0 + 2 * strideC];
    float c03 = C[0 + 3 * strideC];
    float c10 = C[1 + 0 * strideC];
    float c11 = C[1 + 1 * strideC];
    float c12 = C[1 + 2 * strideC];
    float c13 = C[1 + 3 * strideC];
    float c20 = C[2 + 0 * strideC];
    float c21 = C[2 + 1 * strideC];
    float c22 = C[2 + 2 * strideC];
    float c23 = C[2 + 3 * strideC];
    float c30 = C[3 + 0 * strideC];
    float c31 = C[3 + 1 * strideC];
    float c32 = C[3 + 2 * strideC];
    float c33 = C[3 + 3 * strideC];

    for (int k = 0; k < n; k++) {
        float a0k = A[0 + k * strideA];
        float a1k = A[1 + k * strideA];
        float a2k = A[2 + k * strideA];
        float a3k = A[3 + k * strideA];

        float b0k = B[0 + k * strideB];
        float b1k = B[1 + k * strideB];
        float b2k = B[2 + k * strideB];
        float b3k = B[3 + k * strideB];

        c00 += a0k * b0k;
        c01 += a0k * b1k;
        c02 += a0k * b2k;
        c03 += a0k * b3k;

        c10 += a1k * b0k;
        c11 += a1k * b1k;
        c12 += a1k * b2k;
        c13 += a1k * b3k;

        c20 += a2k * b0k;
        c21 += a2k * b1k;
        c22 += a2k * b2k;
        c23 += a2k * b3k;

        c30 += a3k * b0k;
        c31 += a3k * b1k;
        c32 += a3k * b2k;
        c33 += a3k * b3k;
    }

    C[0 + 0 * strideC] = c00;
    C[0 + 1 * strideC] = c01;
    C[0 + 2 * strideC] = c02;
    C[0 + 3 * strideC] = c03;
    C[1 + 0 * strideC] = c10;
    C[1 + 1 * strideC] = c11;
    C[1 + 2 * strideC] = c12;
    C[1 + 3 * strideC] = c13;
    C[2 + 0 * strideC] = c20;
    C[2 + 1 * strideC] = c21;
    C[2 + 2 * strideC] = c22;
    C[2 + 3 * strideC] = c23;
    C[3 + 0 * strideC] = c30;
    C[3 + 1 * strideC] = c31;
    C[3 + 2 * strideC] = c32;
    C[3 + 3 * strideC] = c33;
}

int newM, newN, newK;

static void do_block(
        // clang-format off
        int M, int N, int K,
        int strideA, int strideB, int strideC,
        float *A, float *B, float *C,
        float *packed_A, float *packed_B, int should_pack_B
        // clang-format on
) {
    int j;
    for (j = 0; j + UNROLL_NUM <= N; j += UNROLL_NUM) {
        if (should_pack_B) {
            pack_right_B(K, strideB, B + j * strideB, packed_B + j * K);
            // std::cout << "packB" << std::endl;
            // print_matrix(packed_B, newN, newN, BLOCK_COL);
        }
        int i;
        for (i = 0; i + UNROLL_NUM <= M; i += UNROLL_NUM) {
            if (j == 0) {
                pack_left_A(K, strideA, A + i, packed_A + i * K);
                // std::cout << "packA" << std::endl;
                // print_matrix(packed_A, SIMD_UNROLL, SIMD_UNROLL, BLOCK_COL * BLOCK_ROW / SIMD_UNROLL);
            }
            // std::cout << "k: " << K << std::endl;

            do_mul_square(K, UNROLL_NUM, UNROLL_NUM, strideC, packed_A + i * K, packed_B + j * K, C + i + j * strideC);
        }
    }
}

int should_padding(int m, int n, int& new_m_ptr, int& new_n_ptr) {
    new_m_ptr = (m + UNROLL_NUM - 1) / UNROLL_NUM * UNROLL_NUM;
    new_n_ptr = (n + UNROLL_NUM - 1) / UNROLL_NUM * UNROLL_NUM;
    return m != new_m_ptr || n != new_n_ptr;
}

float* get_padding_matrix(int stride, int m, int n, int new_m, int new_n, const float* A) {
    // std::cout
    // << "m: " << m << ", "
    // << "n: " << n << ", "
    // << "new_m: " << new_m << ", "
    // << "new_n: " << new_n << ", "
    // << std::endl;

    float* ret = NULL;
    int err = posix_memalign((void**) &ret, 4096, sizeof(float) * new_m * new_n);
    if (err != 0) {
        throw std::runtime_error("Can not align malloc padding matrix!");
    }
    int j;
    for (j = 0; j < n; ++j) {
        memcpy(ret + j * new_m, A + j * stride, sizeof(float) * m);
        memset(ret + m + j * new_m, 0, sizeof(float) * (new_m - m));
    }
    for (; j < new_n; ++j) {
        memset(ret + j * new_m, 0, sizeof(float) * new_m);
    }
    return ret;
}

void back_padding(int stride, int m, int n, int new_m, int new_n, float* A, float* padding_A) {
    for (int j = 0; j < n; ++j) {
        memcpy(A + j * stride, padding_A + j * new_m, sizeof(float) * m);
    }
}

void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {
    // std::cout
    // << "m: " << M << ", "
    // << "n: " << N << ", "
    // << "k: " << K << ", "
    // << std::endl;

    // print_matrix(A, M, M, K);
    // print_matrix(B, K, K, N);
    // print_matrix(C, M, M, N);
    // print_array(A, M*K);
    // print_array(B, K*N);
    // print_array(C, M*N);

    float *padding_A = A, *padding_B = B, *padding_C = C;

    // int newM, newN, newK;
    int should_pad_A = should_padding(M, K, newM, newK);
    int should_pad_B = should_padding(K, N, newK, newN);
    int should_pad_C = should_padding(M, N, newM, newN);
    float *packed_A, *packed_B;
    // TODO 给他搞成一个 workspace
    int tempRet1 = posix_memalign((void**) &packed_A, 4096, BLOCK_COL * BLOCK_ROW * sizeof(float));
    int tempRet2 = posix_memalign((void**) &packed_B, 4096, BLOCK_COL * newN * sizeof(float));

    if (tempRet1 != 0 || tempRet2 != 0) {
        throw std::runtime_error("Can not align malloc packed pool!");
    }

    if (should_pad_A) {
        // std::cout << "pad A" << std::endl;
        padding_A = get_padding_matrix(M, M, K, newM, newK, A);
    }
    if (should_pad_B) {
        // std::cout << "pad B" << std::endl;
        padding_B = get_padding_matrix(K, K, N, newK, newN, B);
    }
    if (should_pad_C) {
        // std::cout << "pad C" << std::endl;
        padding_C = get_padding_matrix(M, M, N, newM, newN, C);
    }

    // print_matrix(padding_A, newM, M, K);
    // print_matrix(padding_B, newK, K, N);
    // print_matrix(padding_C, newM, M, N);

    // print_matrix(padding_A, newM, newM, newK);
    // print_matrix(padding_B, newK, newK, newN);
    // print_matrix(padding_C, newM, newM, newN);

    // print_array(padding_A, newM * newK);
    // print_array(padding_B, newK * newN);
    // print_array(padding_C, newM * newN);


    for (int k = 0; k < newK; k += BLOCK_COL) {
        int K = min(newK - k, BLOCK_COL);
        for (int i = 0; i < newM; i += BLOCK_ROW) {
            int M = min(newM - i, BLOCK_ROW);
            int N = newN;

            // clang-format off
            do_block(
                    M, N, K,
                    newM, newK, newM,
                    padding_A + i + k * newM,
                    padding_B + k,
                    padding_C + i,

                    packed_A, packed_B, i == 0
            );
            // clang-format on
        }
    }

    free(packed_A);
    free(packed_B);

    if (should_pad_A) {
        free(padding_A);
    }
    if (should_pad_B) {
        free(padding_B);
    }
    if (should_pad_C) {
        back_padding(M, M, N, newM, newN, C, padding_C);
        free(padding_C);
    }
    // throw std::runtime_error("shit");
}
