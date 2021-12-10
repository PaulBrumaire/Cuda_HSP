#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define N 20
#define P 20

void MatrixInit(float *M, int n, int p){
    int i;
    for (i=0;i<n*p;i++){
        M[i]= (float)rand()/(RAND_MAX/2)-1;
    }
}

void MatrixPrint(float *M, int n, int p){
    int i;
    for (i=0;i<n*p;i++){
        if(M[i]>0) printf(" ");
        printf("%1.2f ", M[i]);
        if ((i+1+p)%p==0 ){
            printf("\n");
        }
    }
    printf("\n");

}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i;
    for (i=0;i<n*p;i++){
        Mout[i]=M1[i]+M2[i];
    }
}

void MatrixSub(float *M1, float *M2, float *Mout, int n, int p){
    int i;
    for (i=0;i<n*p;i++){
        Mout[i]=M1[i]-M2[i];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n, int p){
    int i,j,z;
    float temp;
    for (i=0;i<n;i++){
        for (z=0; z<p;z++){
            temp = 0;
            for (j=0; j<p;j++){
                temp+=M1[i*p+j]*M2[j*n+z];
                // printf("%f %f -> id =%d\n",M1[i*p+j],M2[j*n+z],i*n+z);
            }
            Mout[i*p+z]= temp;
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n, int p){
    int l = threadIdx.x;
    int c = blockIdx.x;
    float temp = 0;
    for (int i=0; i<p;i++){
        temp+=M1[l*p+i]*M2[i*p+c];
    }
    Mout[l*n+c]= temp;

}



int main(){
    struct timeval start;
    struct timeval stop;
    struct timeval startgpu;
    struct timeval stopgpu;

    float *d_M1, *d_M2, *d_Mout;

    int nb_thread = N;
    int nb_block = P;

    //dim3 thread1(nb_thread,nb_thread);
    //dim3 block1(nb_block,nb_block);
    float* M1 = (float*)malloc(N*P * sizeof(float));
    float* M2 = (float*)malloc(N*P * sizeof(float));
    float* Mout = (float*)malloc(N*P * sizeof(float));
    float* Mout2 = (float*)malloc(N*P * sizeof(float));
    float* MDiff = (float*)malloc(N*P * sizeof(float));

    MatrixInit(M1,N,P);
    MatrixInit(M2,N,P);
    //MatrixInit(Mout,N,P);
    //MatrixPrint(M1,n,p);
    //MatrixPrint(M2,n,p);
    
    //CPU
    gettimeofday(&start,0);
    MatrixMult(M1, M2, Mout, N, P);
    gettimeofday(&stop,0);
    double time_spent = (double)(stop.tv_usec - start.tv_usec)/1000000 + (double)(stop.tv_sec - start.tv_sec) ;
    printf("Duration on CPU = %f s\n",time_spent);
    MatrixPrint(Mout,N,P);

    //GPU
    cudaMalloc((void**)&d_M1, sizeof(float)*N*P);
    cudaMalloc((void**)&d_M2, sizeof(float)*N*P);
    cudaMalloc((void**)&d_Mout, sizeof(float)*N*P);

    cudaMemcpy(d_M1, M1, sizeof(float) * N*P, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * N*P, cudaMemcpyHostToDevice);

    gettimeofday(&startgpu,0);
    cudaMatrixMult<<<nb_block,nb_thread>>>(d_M1, d_M2, d_Mout, N, P);
    gettimeofday(&stopgpu,0);
    cudaMemcpy(Mout2, d_Mout, sizeof(float)*N*P, cudaMemcpyDeviceToHost);
    double time_spentgpu = (double)(stopgpu.tv_usec - startgpu.tv_usec)/1000000 + (double)(stopgpu.tv_sec - startgpu.tv_sec) ;
    printf("Duration on GPU = %f s\n",time_spentgpu);
    MatrixPrint(Mout2,N,P);
    // C'est plus rapide sur le GPU que sur le CPU
    cudaDeviceSynchronize();
    

}