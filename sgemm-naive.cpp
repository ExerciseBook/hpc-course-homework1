const char* sgemm_desc = "Naive, three-loop sgemm.";

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are MxK, KxN, MxN matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void custom_sgemm(int M, int K, int N, float* A, float* B, float* C) {
    int lda = M;
    int ldb = K;
    int ldc = M;
    /* For each row i of A */
    for (int i = 0; i < M; ++i) /* For each column j of B */
        for (int j = 0; j < N; ++j) {
            /* Compute C(i,j) */
            float cij = C[i + j * ldc];
            for (int k = 0; k < K; k++) cij += A[i + k * lda] * B[k + j * ldb];
            C[i + j * ldc] = cij;
        }
}
