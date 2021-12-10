#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>


void MatrixInit(float *M, int n, int p){
    int i;
    for (i=0;i<n*p;i++){
        M[i]= (float)rand()/(RAND_MAX/2)-1;
    }
}

void MatrixPrint(float *M, int n, int p){
    int i;
    for (i=0;i<n*p;i++){
        printf("%f ", M[i]);
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


__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id<n*p){
        Mout[id]=M1[id]+M2[id];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n, int p){
    int i,j,z;
    float temp;
    for (i=0;i<n;i++){
        for (z=0; z<p;z++){
            temp = 0;
            for (j=0; j<p;j++){
                temp+=M1[i*p+j]*M2[j*n+i*z];
            }
            Mout[i*n+z]= temp;
        }
    }
}



int main(){
    struct timeval start;
    struct timeval stop;
    struct timeval startgpu;
    struct timeval stopgpu;

    float *d_M1, *d_M2, *d_Mout;

    int n = 10000;
    int p = 10000;
    int nb_thread = 1000;
    int nb_block = ((n*p + nb_thread)/nb_thread);
    float* M1 = (float*)malloc(n*p * sizeof(float));
    float* M2 = (float*)malloc(n*p * sizeof(float));
    float* Mout = (float*)malloc(n*p * sizeof(float));
    MatrixInit(M1,n,p);
    MatrixInit(M2,n,p);
    MatrixInit(Mout,n,p);
    //MatrixPrint(M1,n,p);
    
    //CPU
    gettimeofday(&start,0);
    MatrixAdd(M1, M2, Mout, n, p);
    gettimeofday(&stop,0);
    double time_spent = (double)(stop.tv_usec - start.tv_usec)/1000000 + (double)(stop.tv_sec - start.tv_sec) ;
    printf("Duration on CPU = %f s\n",time_spent);
    //MatrixPrint(Mout,n,p);

    //GPU
    cudaMalloc((void**)&d_M1, sizeof(float)*n*p);
    cudaMalloc((void**)&d_M2, sizeof(float)*n*p);
    cudaMalloc((void**)&d_Mout, sizeof(float)*n*p);

    cudaMemcpy(d_M1, M1, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n*p, cudaMemcpyHostToDevice);

    gettimeofday(&startgpu,0);
    cudaMatrixAdd<<<nb_block,nb_thread>>>(d_M1, d_M2, d_Mout, n, p);
    gettimeofday(&stopgpu,0);
    cudaMemcpy(Mout, d_Mout, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
    double time_spentgpu = (double)(stopgpu.tv_usec - startgpu.tv_usec)/1000000 + (double)(stopgpu.tv_sec - startgpu.tv_sec) ;
    printf("Duration on GPU = %f s\n",time_spentgpu);
    // C'est plus rapide sur le GPU que sur le CPU
    cudaDeviceSynchronize();
    
    
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);
}