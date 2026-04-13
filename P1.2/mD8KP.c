#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <strings.h>
#include <assert.h>
#include <omp.h>
#define N 8000L
#define ND N*N/100
typedef struct {
    int i,j,v;
} tmd;
int A[N][N],B[N][N],C[N][N],C1[N][N],C2[N][N];
int jBD[N+1],jAD[N+1],VCcol[N],VBcol[N];

typedef struct { int v; char pad[60]; } nele_padded;
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
    
    bzero(C,sizeof(int)*(N*N));
    bzero(C1,sizeof(int)*(N*N));
    bzero(C2,sizeof(int)*(N*N));
     
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
    qsort(AD,ND,sizeof(tmd),cmp_fil); // ordenat per files
    for(k=0;k<ND;k++)
    {
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
    qsort(BD,ND,sizeof(tmd),cmp_col); // ordenat per columnes
    
    k=0;
    for (j=0; j<N+1; j++)
     {
      while (k < ND && j>BD[k].j) k++;
      jBD[j] = k;
     }

    k=0;
    for (i=0; i<N+1; i++)
     {
      while (k < ND && i>AD[k].i) k++;
      jAD[i] = k;
     }

    #pragma omp parallel for schedule(guided) private(i,k)
    for (int row=0; row<N; row++)
        for (k=jAD[row]; k<jAD[row+1]; k++)
            for (i=0; i<N; i++)
                C1[row][i] += AD[k].v * B[AD[k].j][i];

    #pragma omp parallel private(j,k)
    {
        int VBcol_priv[N];
        for (j=0;j<N;j++) VBcol_priv[j] = 0;

        #pragma omp for schedule(guided)
        for(i=0;i<N;i++)
          {
            // expandir Columna de B[*][i]
            for (k=jBD[i];k<jBD[i+1];k++)
                    VBcol_priv[BD[k].i] = BD[k].v;
            // Calcul de tota una columna de C
            for (k=0;k<ND;k++)
                C2[AD[k].i][i] += AD[k].v * VBcol_priv[AD[k].j];
            // Neteja selectiva: nomes les posicions tocades
            for (k=jBD[i];k<jBD[i+1];k++)
                VBcol_priv[BD[k].i] = 0;
          }
    }

    neleC = 0;

    int nthreads;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    long long chunk = (long long)N*N / nthreads + 1;

    tmd         *CD_local   = (tmd*)malloc((long long)nthreads * chunk * sizeof(tmd));
    nele_padded *nele_local = (nele_padded*)calloc(nthreads, sizeof(nele_padded));

    #pragma omp parallel private(j,k)
    {
        int tid = omp_get_thread_num();
        int VBcol_priv[N];
        int VCcol_priv[N];
        for (j=0;j<N;j++) VBcol_priv[j] = VCcol_priv[j] = 0;
        tmd *my_CD   = CD_local + (long long)tid * chunk;
        int  my_nele = 0;

        #pragma omp for schedule(guided)
        for(i=0;i<N;i++)
          {
            // expandir Columna de B[*][i]
            for (k=jBD[i];k<jBD[i+1];k++)
                    VBcol_priv[BD[k].i] = BD[k].v;
            // Calcul de tota una columna de C
            for (k=0;k<ND;k++)
                VCcol_priv[AD[k].i] += AD[k].v * VBcol_priv[AD[k].j];
            // Neteja selectiva de VBcol i compressio de VCcol
            for (k=jBD[i];k<jBD[i+1];k++)
                VBcol_priv[BD[k].i] = 0;
            for (j=0;j<N;j++)
             {
                if (VCcol_priv[j])
                 {
                    my_CD[my_nele].i = j;
                    my_CD[my_nele].j = i;
                    my_CD[my_nele].v = VCcol_priv[j];
                    VCcol_priv[j] = 0;
                    my_nele++;
                 }
             }
          }
        nele_local[tid].v = my_nele;
    }

    for (int t=0; t<nthreads; t++)
     {
        tmd *my_CD = CD_local + (long long)t * chunk;
        for (int e=0; e<nele_local[t].v; e++)
            CD[neleC++] = my_CD[e];
     }

    free(CD_local);
    free(nele_local);


    // Comprovacio MD x M -> M i MD x MD -> M
    for (i=0;i<N;i++)
        for(j=0;j<N;j++)
            if (C2[i][j] != C1[i][j])
                printf("Diferencies C1 i C2 pos %d,%d: %d != %d\n",i,j,C1[i][j],C2[i][j]);
    // Comprovacio MD X MD -> M i MD x MD -> MD
    Suma = 0;
    for(k=0;k<neleC;k++)
     {
        Suma += CD[k].v;
        if (CD[k].v != C1[CD[k].i][CD[k].j])
            printf("Diferencies C1 i CD a i:%d,j:%d,v%d, k:%d, vd:%d\n",CD[k].i,CD[k].j,C1[CD[k].i][CD[k].j],k,CD[k].v);
     }
     
    printf ("\nNumero elements de la matriu dispersa C %d\n",neleC);   
    printf("Suma dels elements de C %lld \n",Suma);
    exit(0);
}