#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#define N 8000L
#define ND (N*N/100)

typedef struct {
    int i, j, v;
} tmd;

int A[N][N], B[N][N], C1[N][N], C2[N][N];
int iAD[N+1], jBD[N+1];
tmd AD[ND], BD[ND], CD[N*N];

long long Suma;

int cmp_fil(const void *pa, const void *pb) {
    tmd *a = (tmd*)pa;
    tmd *b = (tmd*)pb;
    if (a->i > b->i) return 1;
    else if (a->i < b->i) return -1;
    else return (a->j - b->j);
}

int cmp_col(const void *pa, const void *pb) {
    tmd *a = (tmd*)pa;
    tmd *b = (tmd*)pb;
    if (a->j > b->j) return 1;
    else if (a->j < b->j) return -1;
    else return (a->i - b->i);
}

int main() {
    int i, j, k;
    int neleC = 0;
    int num_threads;

    // Generació de matrius disperses A i B (seqüencial)
    bzero(A, sizeof(int)*(N*N));
    bzero(B, sizeof(int)*(N*N));

    for(k=0;k<ND;k++)
    {
        AD[k].i=rand()%(N-1);
        AD[k].j=rand()%(N-1);
        AD[k].v=rand()%100+1;
        while (A[AD[k].i][AD[k].j]) {
            if(AD[k].i < AD[k].j)
                AD[k].i = (AD[k].i + 1)%N;
            else
                AD[k].j = (AD[k].j + 1)%N;
        }
        A[AD[k].i][AD[k].j] = AD[k].v;
    }
    qsort(AD,ND,sizeof(tmd),cmp_fil);

    for(k=0;k<ND;k++) {
        BD[k].i=rand()%(N-1);
        BD[k].j=rand()%(N-1);
        BD[k].v=rand()%100+1;
        while (B[BD[k].i][BD[k].j]) {
            if(BD[k].i < BD[k].j)
                BD[k].i = (BD[k].i + 1)%N;
            else
                BD[k].j = (BD[k].j + 1)%N;
        }
        B[BD[k].i][BD[k].j] = BD[k].v;
    }

    qsort(BD,ND,sizeof(tmd),cmp_col);

    k = 0;
    for (i = 0; i < N + 1; i++) {
        while (k < ND && i > AD[k].i) k++;
        iAD[i] = k;
    }

    k = 0;
    for (j = 0; j < N + 1; j++) {
        while (k < ND && j > BD[k].j) k++;
        jBD[j] = k;
    }

    Suma = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
        }

        // Inicialització paral·lela de C1 i C2
        #pragma omp for collapse(2) nowait
        for(i=0; i<N; i++)
            for(j=0; j<N; j++)
                C1[i][j] = 0;

        #pragma omp for collapse(2) nowait
        for(i=0; i<N; i++)
            for(j=0; j<N; j++)
                C2[i][j] = 0;

        // Matriu dispersa per matriu
        // Optimitzat amb millor localitat de cache
        #pragma omp for schedule(dynamic, 32) nowait
        for (int r = 0; r < N; r++) {
            for (int ptr = iAD[r]; ptr < iAD[r+1]; ptr++) {
                int col_A = AD[ptr].j;
                int val_A = AD[ptr].v;
                int *C1_row = C1[r];      // Punter a la fila per millor cache locality
                int *B_row = B[col_A];

                for (int c = 0; c < N; c++) {
                    C1_row[c] += val_A * B_row[c];
                }
            }
        }

        // Matriu dispersa per matriu dispersa -> dona matriu Dispersa
        #pragma omp for schedule(dynamic, 8) nowait
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                int sum = 0;
                int ka = iAD[r];
                int kb = jBD[c];

                while (ka < iAD[r+1] && kb < jBD[c+1]) {
                    if (AD[ka].j == BD[kb].i) {
                        sum += AD[ka].v * BD[kb].v;
                        ka++;
                        kb++;
                    } else if (AD[ka].j < BD[kb].i) {
                        ka++;
                    } else {
                        kb++;
                    }
                }

                if (sum != 0) {
                    C2[r][c] = sum;
                }
            }
        }

        // Construcció optimitzada de CD amb buffers locals per evitar atomic
        int *local_counts;
        tmd **local_CD;

        #pragma omp single
        {
            local_counts = (int*)calloc(num_threads, sizeof(int));
            local_CD = (tmd**)malloc(num_threads * sizeof(tmd*));
            for(i=0; i<num_threads; i++) {
                // Reservem memòria suficient per cada thread
                local_CD[i] = (tmd*)malloc((N*N/num_threads + 1000) * sizeof(tmd));
            }
        }

        #pragma omp for collapse(2) nowait
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                if (C2[r][c] != 0) {
                    int tid = omp_get_thread_num();
                    int idx = local_counts[tid]++;
                    local_CD[tid][idx].i = r;
                    local_CD[tid][idx].j = c;
                    local_CD[tid][idx].v = C2[r][c];
                }
            }
        }

        // Fusionar buffers locals en CD global
        #pragma omp single
        {
            neleC = 0;
            for(i=0; i<num_threads; i++) {
                memcpy(&CD[neleC], local_CD[i], local_counts[i]*sizeof(tmd));
                neleC += local_counts[i];
                free(local_CD[i]);
            }
            free(local_CD);
            free(local_counts);
        }

        // Comprovació MD x M -> M i MD x MD -> M
        #pragma omp for collapse(2) nowait
        for (int r = 0; r < N; r++) {
            for(int c = 0; c < N; c++) {
                if (C2[r][c] != C1[r][c]) {
                    printf("Diferencies C1 i C2 pos %d,%d: %d != %d\n", r, c, C1[r][c], C2[r][c]);
                }
            }
        }

        // Comprovacio MD X MD -> M i MD x MD -> MD
        #pragma omp for reduction(+:Suma)
        for(int m = 0; m < neleC; m++) {
            Suma += CD[m].v;
            if (CD[m].v != C1[CD[m].i][CD[m].j]) {
                printf("Diferencies C1 i CD a i:%d,j:%d,v%d, m:%d, vd:%d\n",
                       CD[m].i, CD[m].j, C1[CD[m].i][CD[m].j], m, CD[m].v);
            }
        }
    }

    printf("\nNumero elements de la matriu dispersa C: %d\n", neleC);
    printf("Suma dels elements de C: %lld \n", Suma);

    return 0;
}