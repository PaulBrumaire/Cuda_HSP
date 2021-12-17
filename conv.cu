#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define N 32
#define P 32
#define Q 6
#define K 5


void MatrixInit(float *M, int n, int p,int q){
    int i;
    for (i=0;i<n*p*q;i++){
        M[i]= (float)rand()/(RAND_MAX);
    }
}

void MatrixInit2(float *M, int n, int p,int q){
    int i;
    for (i=0;i<n*p*q;i++){
        M[i]= (float)rand()/(RAND_MAX/2)-1;
    }
}

void MatrixInit0(float *M, int n, int p,int q){
    int i;
    for (i=0;i<n*p*q;i++){
        M[i]= (float) 0;
    }
}


void MatrixInitId(float *M, int n, int p,int q){
    int i;
    for (i=0;i<n*p*q;i++){
        M[i]= (float) 0;
    }
    M[(n*p)/2]= (float) 1;
}

void MatrixPrint(float *M, int n, int p, int q){
    int i;
    for (i=0;i<n*p*q;i++){
        if(M[i]>0) printf(" ");
        printf("%1.2f ", M[i]);
        if ((i+1+p)%p==0 ){
            printf("\n");
        }
        if ((i+1)%(n*p)==0 ){
            printf("\n\n");
        }
    }
    printf("\n");
}

__device__ float cudaActivationTanh(float val) {
    // float temp = exp(2*val);
    // printf("%f\n",temp);
    // return (float) (temp-1)/(temp+1);
    return tanhf(val);
}

__global__ void cudaConv2D(float *img, float *kernels, float * out, int n, int p, int q, int k ) {
    // n,p= lignes,col img
    // q = nb kernels
    // k = dim kernel (k*k)

    int m=n-k+1; //dim img out
    //int id = (k-1)/2; //center

    int l = threadIdx.x; //->28 (dim img out)
    int c = threadIdx.y;  //->28 (dim img out)
    
    int d = blockIdx.x;  //->6  (dim nb kernel deconv)
    
    float temp=0;
    int i,j;
    //Calcul du bloc K*K
    for (int ki= 0; ki<k;ki++){
        for (int kj= 0; kj<k;kj++){
            i=l+ki;
            j=c+kj;   
            temp+=img[i*n + j] * kernels[d*k*k + ki*k + kj];
        }
    }

    out[d*m*m + l*m + c] = cudaActivationTanh(temp);//temp;//
}





__global__ void cudaConv3D(float *img, float *kernels, float * out, int n, int p, int q, int k ) {
    // n,p= lignes,col img
    // q = nb kernels
    // k = dim kernel (k*k)

    int m=n-k+1; //dim img out
    //int id = (k-1)/2; //center

    int l = threadIdx.x; //(dim img out)
    int c = threadIdx.y;  //(dim img out)
    
    int d = blockIdx.x;  //(dim nb kernel deconv)
    //int o = blockIdx.y; //dim nb images in

    // printf("/ %d",d);
    
    
    int i,j;
    for (int o=0; o<1; o++){
        //Calcul du bloc K*K =5*5
        float temp=0;
        for (int ki= 0; ki<k;ki++){
            for (int kj= 0; kj<k;kj++){
                i=l+ki;
                j=c+kj;   
                temp+=img[o*n*p +i*n + j] * kernels[d*k*k + ki*k + kj];
            }
        }

        //printf("/%d",o);//*m*m*16 +d*m*m+ l*m + c) ;
        // o=0-5, d=0-15, l=0-9, c=0-9
        out[ o*m*m*q +d*m*m+ l*m + c]= cudaActivationTanh(temp);
    }
}

__global__ void cudaCombine(float *in, float * out, float * id ) {
    int l = threadIdx.x; // (ligne)
    int c = threadIdx.y;  //(colonne)
    
    int g = blockIdx.x; // (dim out profondeur)

    float temp = 0;
    for(int i=0;i<6;i++){
        temp+=in[i*16*10*10 + g*10*10 +l*10 +c]*id[i*16+g];
    }
    out[g*10*10 + l*10 + c ]= temp;
}

__global__ void cudaMeanPool(float *in, float *out, int n, int p, int q) {
    // n,p= lignes,col in
    // q = nb kernels = profondeur

    int m=n/2; //dim out

    int l = threadIdx.x; //->28 (dim out)
    int c = threadIdx.y;  //->28 (dim out)
    int d = blockIdx.x;  //->6  (dim nb kernel)
    
    float temp=0;
    int i,j;
    //Calcul du bloc K*K
    for (int ki= 0; ki<2;ki++){
        for (int kj= 0; kj<2;kj++){
            i=l+ki;
            j=c+kj;    
            temp+=in[d*n*p + i*n + j];
        }
    }

    out[d*m*m + l*m + c]= temp/4;
}




int main(){
    int L=(N-K+1); //dim out conv
    int M=(L/2); //dim out pool

    float *raw_data, *C1_data, *C1_kernel, *S2_data,*C3_dataTemp, *C3_data , *C3_kernel;

    srand(time(NULL));

    raw_data = (float*)malloc(N*P * sizeof(float));
    C1_data = (float*)malloc(Q*L*L * sizeof(float));
    C1_kernel = (float*)malloc(Q*K*K * sizeof(float));
    S2_data = (float*)malloc(Q*M*M * sizeof(float));

    C3_dataTemp = (float*)malloc(96*10*10 * sizeof(float));
    C3_data = (float*)malloc(16*10*10 * sizeof(float));
    C3_kernel = (float*)malloc(16*K*K * sizeof(float));

    float combineId[96] = {
        1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,
        1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,
        1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1,
        0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1,
        0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,
        0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1
    };

    MatrixInit(raw_data,N,P,1);
    MatrixInit0(C1_data,L,L,Q);
    MatrixInit2(C1_kernel,K,K,Q);

    MatrixInit0(S2_data,M,M,Q);

    MatrixInit0(C3_dataTemp,10,10,96);
    MatrixInit0(C3_data,10,10,16);
    MatrixInit2(C3_kernel,K,K,16);

    //MatrixPrint(combineId,6,16,1);


    printf("IMAGE\n");
    MatrixPrint(raw_data,N,P,1);
    printf("KERNEL\n");
    //MatrixPrint(C1_kernel,K,K,1);
    //GPU
    float *d_combine,*d_raw, *d_C1, *d_C1_kernel, *d_S2,*d_C3Temp, *d_C3, *d_C3_kernel;

    //CUDA ARRRAY---------------------------------------------------------------------
    cudaMalloc((void**)&d_raw, sizeof(float)*N*P);
    cudaMalloc((void**)&d_C1, sizeof(float)*Q*L*L);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float)*Q*K*K);
    cudaMalloc((void**)&d_S2, sizeof(float)*Q*M*M);
    cudaMalloc((void**)&d_C3Temp, sizeof(float)*96*10*10);
    cudaMalloc((void**)&d_C3, sizeof(float)*16*10*10);
    cudaMalloc((void**)&d_C3_kernel, sizeof(float)*16*K*K);
    cudaMalloc((void**)&d_combine, sizeof(float)*16*6);

    cudaMemcpy(d_raw, raw_data, sizeof(float) * N*P, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1, C1_data, sizeof(float) * Q*L*L, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * Q*K*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2, S2_data, sizeof(float) * Q*M*M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3Temp, C3_dataTemp, sizeof(float) * 96*10*10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3, C3_data, sizeof(float) * 16*10*10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_kernel, C3_kernel, sizeof(float) * 16*K*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_combine, combineId, sizeof(float) * 16*6, cudaMemcpyHostToDevice);


    //CONV 1---------------------------------------------------------------------
    dim3 nb_thread(L,L);//L=28
    dim3 nb_block(Q);//Q=6

    cudaConv2D<<<nb_block,nb_thread>>>(d_raw, d_C1_kernel, d_C1, N, P, Q, K );
    cudaDeviceSynchronize();

    //MEAN 1---------------------------------------------------------------------
    dim3 nb_thread2(M,M);//M=14
    dim3 nb_block2(Q);//Q=6
    cudaMeanPool<<<nb_block2,nb_thread2>>>(d_C1, d_S2, L, L, Q );
    cudaDeviceSynchronize();

    //CONV 2  14*14*6 -> 10*10*96 -> 10*10*16 ---------------------------------------------------------------------
    dim3 nb_thread3(10,10); //im out
    dim3 nb_block3(16); //nb kernel
    cudaConv3D<<<nb_block3,nb_thread3>>>(d_S2, d_C3_kernel, d_C3Temp, 14, 14, 16, K );


    //COMBINE ---------------------------------------------------------------------------------
    cudaDeviceSynchronize();
    dim3 nb_thread4(10,10);
    dim3 nb_block4(16);
    cudaCombine<<<nb_block4,nb_thread4>>>(d_C3Temp,d_C3, d_combine);
    cudaDeviceSynchronize();


    //ARRAY COPY TO CPU --------------------------------------------------------------------------
    cudaMemcpy(C1_data, d_C1, sizeof(float)* Q*L*L, cudaMemcpyDeviceToHost);
    printf("OUT\n");
    MatrixPrint(C1_data,L,L,1);
    
    cudaMemcpy(S2_data, d_S2, sizeof(float)* Q*M*M, cudaMemcpyDeviceToHost);
    printf("MEAN\n");
    //MatrixPrint(S2_data,M,M,1); //M<L

    cudaMemcpy(C3_dataTemp, d_C3Temp, sizeof(float)* 96*10*10, cudaMemcpyDeviceToHost);
    cudaMemcpy(C3_data, d_C3, sizeof(float)* 16*10*10, cudaMemcpyDeviceToHost);
    
    printf("C3 Temp\n");
    //MatrixPrint(C3_dataTemp,10,10,16); 
    printf("C3\n");
    MatrixPrint(C3_data,10,10,16); //M<L
    
    cudaDeviceSynchronize();
    
    cudaFree(d_raw);
    cudaFree(d_C1);
    cudaFree(d_C1_kernel);

    free(raw_data);
    free(C1_data);
    free(C1_kernel);
}