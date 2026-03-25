#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define N 600000
#define G 200
#define THRESHOLD 50

long V[N];
long R[G];
int A[G];

void kmean(int fN, int fK, long fV[], long fR[], int fA[])
{
    int iter=0;
    long dif;


    long fS[G];
    int fD[N];

    do {
        dif = 0;

        #pragma omp parallel
        {
            int i, j, min;
            long local_dif, temp_dif, t;


            long local_fS[G];
            int local_fA[G];
            for(i=0; i<fK; i++) {
                local_fS[i] = 0;
                local_fA[i] = 0;
            }


            #pragma omp for nowait
            for(i=0; i<fK; i++){
                fS[i] = 0;
                fA[i] = 0;
            }


            #pragma omp for private(j, min, local_dif, temp_dif)
            for (i=0; i<fN; i++){
                min = 0;
                local_dif = abs(fV[i] - fR[0]);

                for (j=1; j<fK; j++){
                    temp_dif = abs(fV[i] - fR[j]);
                    if (temp_dif < local_dif){
                        min = j;
                        local_dif = temp_dif;
                    }
                }
                fD[i] = min;
            }


            #pragma omp for nowait
            for(i=0; i<fN; i++){
                local_fS[fD[i]] += fV[i];
                local_fA[fD[i]] ++;
            }


            #pragma omp critical
            {
                for(i=0; i<fK; i++){
                    fS[i] += local_fS[i];
                    fA[i] += local_fA[i];
                }
            }
            #pragma omp barrier


            #pragma omp for private(t) reduction(+:dif)
            for(i=0; i<fK; i++){
                t = fR[i];
                if (fA[i]) fR[i] = fS[i] / fA[i];
                dif += abs(t - fR[i]);
            }
        }

        iter++;
    } while(dif);

    printf("iter %d\n",iter);
}

void qs(int ii, int fi, long fV[], int fA[]){
    int i, f;
    long pi, pa, vtmp, vta, vfi, vfa;

    pi = fV[ii];
    pa = fA[ii];
    i = ii + 1;
    f = fi;
    vtmp = fV[i];
    vta = fA[i];

    while (i <= f) {
        if (vtmp < pi) {
            fV[i-1] = vtmp;
            fA[i-1] = vta;
            i++;
            vtmp = fV[i];
            vta = fA[i];
        }
        else {
            vfi = fV[f];
            vfa = fA[f];
            fV[f] = vtmp;
            fA[f] = vta;
            f--;
            vtmp = vfi;
            vta = vfa;
        }
    }
    fV[i-1] = pi;
    fA[i-1] = pa;

    if (fi - ii > THRESHOLD) {
        #pragma omp task
        { if (ii < f) qs(ii, f, fV, fA); }

        #pragma omp task
        { if (i < fi) qs(i, fi, fV, fA); }

        #pragma omp taskwait
    }
    else {
        if (ii < f) qs(ii, f, fV, fA);
        if (i < fi) qs(i, fi, fV, fA);
    }
}

int main()
{
    int i;


    for (i=0; i<N; i++) V[i] = (rand()%rand())/N;
    for (i=0; i<G; i++) R[i] = V[i];


    kmean(N,G,V,R,A);


    #pragma omp parallel
    {
        #pragma omp single
        {
            qs(0,G-1,R,A);
        }
    }

    for (i=0; i<G; i++)
        printf("R[%d] : %ld te %d agrupats\n",i,R[i],A[i]);

    return(0);
}