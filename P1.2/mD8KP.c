#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#define N 8000L
#define ND (N*N/100)
#define BLOCK_SIZE 16 // Mida del bloc per la memòria cau L1

typedef struct {
    int i,j,v;
} tmd;

int A[N][N],B[N][N],C[N][N],C1[N][N],C2[N][N];
int jBD[N+1];
tmd AD[ND],BD[ND],CD[N*N];

long long Suma;

int cmp_fil(const void *pa, const void *pb)
{
    tmd * a = (tmd*)pa;
    tmd * b = (tmd*)pb;
    if (a->i > b->i) return(1);
    else if (a->i < b->i) return (-1);
    else return (a->j - b->j);
}

int cmp_col(const void *pa, const void *pb)
{
    tmd * a = (tmd*)pa;
    tmd * b = (tmd*)pb;
    if (a->j > b->j) return(1);
    else if (a->j < b->j) return (-1);
    else return (a->i - b->i);
}

int main()
{
    int i,j,k,neleC;
    
    // First-touch
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < N; idx++) {
        memset(A[idx], 0, sizeof(int) * N);
        memset(B[idx], 0, sizeof(int) * N);
        memset(C[idx], 0, sizeof(int) * N);
        memset(C1[idx], 0, sizeof(int) * N);
        memset(C2[idx], 0, sizeof(int) * N);
    }
     
    for(k=0;k<ND;k++) {
        AD[k].i=rand()%(N-1);
        AD[k].j=rand()%(N-1);
        AD[k].v=rand()%100+1;
        while (A[AD[k].i][AD[k].j]) {
            if(AD[k].i < AD[k].j) AD[k].i = (AD[k].i + 1)%N;
            else AD[k].j = (AD[k].j + 1)%N;
        }
        A[AD[k].i][AD[k].j] = AD[k].v;
    }
    qsort(AD,ND,sizeof(tmd),cmp_fil); // ordenat per files

    for(k=0;k<ND;k++) {
        BD[k].i=rand()%(N-1);
        BD[k].j=rand()%(N-1);
        BD[k].v=rand()%100+1;
        while (B[BD[k].i][BD[k].j]) {
            if(BD[k].i < BD[k].j) BD[k].i = (BD[k].i + 1)%N;
            else BD[k].j = (BD[k].j + 1)%N;
        }
        B[BD[k].i][BD[k].j] = BD[k].v;
    }
    qsort(BD,ND,sizeof(tmd),cmp_col); // ordenat per columnes
    
    // calcul dels index de les columnes
    k=0;
    for (j=0; j<N+1; j++) {
      while (k < ND && j>BD[k].j) k++;
      jBD[j] = k;
    }

    neleC = 0;
    
    #pragma omp parallel
    {
        // 1. Reservem la memòria.
        int (*priv_VBcol)[BLOCK_SIZE] = malloc(sizeof(int) * N * BLOCK_SIZE);
        int (*priv_VCcol)[BLOCK_SIZE] = malloc(sizeof(int) * N * BLOCK_SIZE);

        // Control de seguretat per si el sistema operatiu ens denega la memòria
        if (priv_VBcol == NULL || priv_VCcol == NULL) {
            printf("Error: No hi ha memòria suficient per als fils.\n");
            exit(1);
        }

        // --- C1: Matriu dispersa per matriu ---
        #pragma omp for schedule(static) nowait
        for (int ib = 0; ib < N; ib += BLOCK_SIZE) {
            for (int k = 0; k < ND; k++) {
                int rA = AD[k].i;
                int cA = AD[k].j;
                int vA = AD[k].v;
                
                // Eliminat pragma omp simd per evitar fallades d'alineació de maquinari
                for (int i = ib; i < ib + BLOCK_SIZE; i++) {
                    C1[rA][i] += vA * B[cA][i];
                }
            }
        }
                
        // --- C2: Matriu dispersa per matriu dispersa ---
        #pragma omp for schedule(dynamic, 32) nowait
        for (int ib = 0; ib < N; ib += BLOCK_SIZE) {
            memset(priv_VBcol, 0, sizeof(int) * N * BLOCK_SIZE);

            // expandir Columna de B[*][i]
            for (int i = ib; i < ib + BLOCK_SIZE; i++) {
                for (int k = jBD[i]; k < jBD[i+1]; k++) {
                    priv_VBcol[BD[k].i][i - ib] = BD[k].v;
                }
            }
            
            // Calcul de tota una columna de C
            for (int k = 0; k < ND; k++) {
                int rA = AD[k].i;
                int cA = AD[k].j;
                int vA = AD[k].v;
                
                for (int i = ib; i < ib + BLOCK_SIZE; i++) {
                    C2[rA][i] += vA * priv_VBcol[cA][i - ib];
                }
            }
        }
                    
        // --- CD: Matriu dispersa per matriu dispersa -> dona matriu Dispersa ---
        #pragma omp for schedule(dynamic, 8)
        for (int ib = 0; ib < N; ib += BLOCK_SIZE) {
            memset(priv_VBcol, 0, sizeof(int) * N * BLOCK_SIZE);
            memset(priv_VCcol, 0, sizeof(int) * N * BLOCK_SIZE);

            // expandir Columna de B[*][i]
            for (int i = ib; i < ib + BLOCK_SIZE; i++) {
                for (int k = jBD[i]; k < jBD[i+1]; k++) {
                    priv_VBcol[BD[k].i][i - ib] = BD[k].v;
                }
            }
            
            // Calcul de tota una columna de C
            for (int k = 0; k < ND; k++) {
                int rA = AD[k].i;
                int cA = AD[k].j;
                int vA = AD[k].v;
                
                for (int i = ib; i < ib + BLOCK_SIZE; i++) {
                    priv_VCcol[rA][i - ib] += vA * priv_VBcol[cA][i - ib];
                }
            }
            
            // Compressio de C
            for (int i = ib; i < ib + BLOCK_SIZE; i++) {
                for (int j = 0; j < N; j++) {
                    if (priv_VCcol[j][i - ib]) {
                        int pos;
                        #pragma omp atomic capture
                        pos = neleC++;

                        CD[pos].i = j;
                        CD[pos].j = i;
                        CD[pos].v = priv_VCcol[j][i - ib];
                    }
                }
            }
        }

        free(priv_VBcol);
        free(priv_VCcol);
    }

    // Comprovacio MD x M -> M i MD x MD -> M
    for (i=0;i<N;i++)
        for(j=0;j<N;j++)
            if (C2[i][j] != C1[i][j])
                printf("Diferencies C1 i C2 pos %d,%d: %d != %d\n",i,j,C1[i][j],C2[i][j]);

    // Comprovacio MD X MD -> M i MD x MD -> MD
    Suma = 0;
    for(k=0;k<neleC;k++) {
        Suma += CD[k].v;
        if (CD[k].v != C1[CD[k].i][CD[k].j])
            printf("Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",CD[k].i,CD[k].j,C1[CD[k].i][CD[k].j],k,CD[k].v);
    }
     
    printf ("\nNumero elements de la matriu dispersa C %d\n",neleC);   
    printf("Suma dels elements de C %lld \n",Suma);
    exit(0);
}