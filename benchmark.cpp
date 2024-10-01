#include <stdio.h>  // For: perror
#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <string.h> // For: memset

#include <float.h> // For: DBL_EPSILON
#include <math.h>  // For: fabs

#ifdef GETTIMEOFDAY
#include <sys/time.h> // For struct timeval, gettimeofday
#else
#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#endif

/* reference_sgemm wraps a call to the BLAS-3 routine sgemm, via the standard FORTRAN interface - hence the reference semantics. */
#define SGEMM sgemm_
extern "C" void SGEMM(char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);

void reference_sgemm(int M, int K, int N, float ALPHA, float* A, float* B, float* C) {
    char TRANSA = 'N';
    char TRANSB = 'N';
    float BETA = 1.;
    int LDA = M;
    int LDB = K;
    int LDC = M;
    SGEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

/* Your function must have the following signature: */
extern const char* sgemm_desc;
extern void custom_sgemm(int, int, int, float*, float*, float*);

double wall_time() {
#ifdef GETTIMEOFDAY
    struct timeval t;
    gettimeofday(&t, NULL);
    return 1. * t.tv_sec + 1.e-6 * t.tv_usec;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1. * t.tv_sec + 1.e-9 * t.tv_nsec;
#endif
}


int randint(int l, int u) {
    int temp;
    srand((unsigned) time(NULL));
    temp = floor(l + (1.0 * rand() / RAND_MAX) * (u - l + 1));
    return temp;
}

void die(const char* message) {
    perror(message);
    exit(EXIT_FAILURE);
}

void fill(float* p, int n) {
    int tt;
    float tmp;
    for (int i = 0; i < n; ++i) {
        tt = rand();
        tmp = (float) tt / (float) (RAND_MAX);
        //printf("%.2lf\n", tmp);
        p[i] = 2 * tmp - 1; // Uniformly distributed over [-1, 1]
    }
}

void absolute_value(float* p, int n) {
    for (int i = 0; i < n; ++i) p[i] = fabs(p[i]);
}

/* The benchmarking program */
int main(int argc, char** argv) {
    printf("Description:\t%s\n\n", sgemm_desc);

    /* Test sizes should highlight performance dips at multiples of certain powers-of-two */
    float initial = randint(1, 10);

    int test_sizes_M[] = {/* Multiples-of-32, +/- 1. for final benchmarking. */
                          31,  32,  33,  63,  64,  65,  95,  96,  97,  127, 128, 129, 159, 160,  161,  191,
                          192, 193, 223, 224, 225, 255, 256, 257, 287, 288, 289, 319, 320, 321,  351,  352,
                          353, 383, 384, 385, 415, 416, 417, 447, 448, 449, 479, 480, 481, 511,  512,  513,
                          543, 544, 545, 575, 576, 577, 607, 608, 609, 639, 640, 641, 671, 672,  673,  703,
                          704, 705, 735, 736, 737, 767, 768, 769, 799, 800, 801, 831, 832, 833,  863,  864,
                          865, 895, 896, 897, 927, 928, 929, 959, 960, 961, 991, 992, 993, 1023, 1024, 1025};
    int test_sizes_K[] = {/* Multiples-of-32, +/- 16. for final benchmarking. */
                          16,  32,  48,  48,  64,  80,  80,  96,  112, 112, 128, 144, 144,  160,  176,  176,
                          192, 208, 208, 224, 240, 240, 256, 272, 272, 288, 304, 304, 320,  336,  336,  352,
                          368, 368, 384, 400, 400, 416, 432, 432, 448, 464, 464, 480, 496,  496,  512,  528,
                          528, 544, 560, 560, 576, 592, 592, 608, 624, 624, 640, 656, 656,  672,  688,  688,
                          704, 720, 720, 736, 752, 752, 768, 784, 784, 800, 816, 816, 832,  848,  848,  864,
                          880, 880, 896, 912, 912, 928, 944, 944, 960, 976, 976, 992, 1008, 1008, 1024, 1040};
    int test_sizes_N[] = {/* Multiples-of-32, +16/32. for final benchmarking. */
                          48,  32,  64,  80,  64,  96,  112, 96,  128, 144, 128,  160, 176,  160,  192,  208,
                          192, 224, 240, 224, 256, 272, 256, 288, 304, 288, 320,  336, 320,  352,  368,  352,
                          384, 400, 384, 416, 432, 416, 448, 464, 448, 480, 496,  480, 512,  528,  512,  544,
                          560, 544, 576, 592, 576, 608, 624, 608, 640, 656, 640,  672, 688,  672,  704,  720,
                          704, 736, 752, 736, 768, 784, 768, 800, 816, 800, 832,  848, 832,  864,  880,  864,
                          896, 912, 896, 928, 944, 928, 960, 976, 960, 992, 1008, 992, 1024, 1040, 1024, 1056};

    /* A representative subset of the first list for initial test. Currently uncommented. */
    // int test_sizes_M[] = {31,   32,   96,   97,  127,  128,  129,  191,  192,  193,  255,  256,  257,  319,  320,  321,  417,  479,  480,  481,  511,  512,  639,  640,  767,  768,  769};
    // int test_sizes_K[] = {16,   32,   96,  112,  112,  128,  144,  176,  192,  208,  240,  256,  272,  304,  320,  336,  432,  464,  480,  496,  496,  512,  624,  640,  752,  768,  784};
    // int test_sizes_N[] = {48,   32,   96,  128,  144,  128,  160,  208,  192,  224,  272,  256,  288,  336,  320,  352,  448,  496,  480,  512,  528,  512,  656,  640,  784,  768,  800};

    int nsizes = sizeof(test_sizes_M) / sizeof(test_sizes_M[0]);

    /* assume last size is also the largest size */
    int nmax_M = test_sizes_M[nsizes - 1];
    int nmax_K = test_sizes_K[nsizes - 1];
    int nmax_N = test_sizes_N[nsizes - 1];

    /* allocate memory for all problems */
    float* buf = NULL;
    buf = (float*) malloc((nmax_M * nmax_K + nmax_K * nmax_N + nmax_M * nmax_N) * sizeof(float));
    if (buf == NULL)
        die("failed to allocate largest problem size");

    /* For each test size */
    for (int isize = 0; isize < nsizes; ++isize) {
        /* Create and fill 3 random matrices A,B,C*/
        int m = test_sizes_M[isize];
        int k = test_sizes_K[isize];
        int n = test_sizes_N[isize];

        float* A = buf + 0;
        float* B = A + nmax_M * nmax_K;
        float* C = B + nmax_K * nmax_N;

        fill(A, m * k);
        fill(B, k * n);
        fill(C, m * n);

        /* Measure performance (in Gflops/s). */

        /* Time a "sufficiently long" sequence of calls to reduce noise */
        double Gflops_s, seconds = -1.0;
        double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
        int n_iterations = 0;
        for (n_iterations = 1; seconds < timeout;) {
            /* Warm-up */
            n_iterations *= 2;

            custom_sgemm(m, k, n, A, B, C);

            /* Benchmark n_iterations runs of custom_sgemm */
            seconds = -wall_time();
            for (int it = 0; it < n_iterations; ++it) custom_sgemm(m, k, n, A, B, C);
            seconds += wall_time();

            /*  compute Mflop/s rate */
            Gflops_s = 2.e-9 * n_iterations * m * k * n / seconds;
        }
        printf("Size: (%dx%d), (%dx%d)\tGflop/s: %.3g (%d iter, %.3f seconds)\n",
               m,
               k,
               k,
               n,
               Gflops_s,
               n_iterations,
               seconds);

        /* Ensure that error does not exceed the theoretical error bound. */

        /* C := A * B, computed with custom_sgemm */
        memset(C, 0, m * n * sizeof(float));
        for (int i = 0; i < m * n; ++i) {
            C[i] = initial;
        }
        custom_sgemm(m, k, n, A, B, C);

        /* Do not explicitly check that A and B were unmodified on custom_sgemm exit
         *  - if they were, the following will most likely detect it:
         * C := C - A * B, computed with reference_sgemm */
        reference_sgemm(m, k, n, -1., A, B, C);

        /* A := |A|, B := |B|, C := |C| */
        absolute_value(A, m * k);
        absolute_value(B, k * n);
        absolute_value(C, m * n);

        /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_sgemm */
        reference_sgemm(m, k, n, -3. * FLT_EPSILON * n, A, B, C);

        /* If any element in C is positive, then something went wrong in custom_sgemm */
        for (int i = 0; i < m * n; ++i)
            if (C[i] > initial)
                die("*** FAILURE *** Error in matrix multiply exceeds componentwise error bounds.\n");
    }

    free(buf);

    return 0;
}
