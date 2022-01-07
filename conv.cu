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
#define WIDTH 28
#define HEIGHT 28

void vector_add(float *in) {
    float temp =0;
    for(int i = 0; i < 10; i++){
        temp += in[i];
    }
    printf("%f \n",temp);
}

void indexMax(float *in) {
    int k = 0;
    float max = in[k];

    for (int i = 0; i < 10; ++i)
    {
        if (in[i] > max)
        {
            max = (float)in[i];
            k = i;
        }
    }
    printf("C'est %d \n",k);
}

void read_file(char* path, float * out){
    FILE *f = fopen(path, "r");

    if (f == NULL)
    {
        printf("Error: could not open file %s", path);
    }
    int i =0;

    while ((fscanf(f,"%f", &out[i])) != EOF){
        i++;
    }
    fclose(f);
}

void readImage(float * data){
    FILE *fptr;
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;

    //Open File
    if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
        printf("Can't open file");
        exit(1);
    }

    //Read File
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    for(int i=2; i<WIDTH+2; i++){
        for(int j=2; j<HEIGHT+2; j++){ 
            fread(&val, sizeof(unsigned char), 1, fptr);  
            data[i*P+j]=(float)val/255;
        }
    }

    
}


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


void ActivationSoftmax(float* input, size_t size) {
	int i;
	double m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < size; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}

	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(input[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < size; ++i) {
		input[i] = exp(input[i] - constant);
	}

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


__global__ void cudaFullyConnected(float *in, float *w, float *out, int n, int p,int q, int activation){
    // n, p= 5 dim in
    // q = 16 profondeur de in
    int l = threadIdx.x; // 120 = taille vecteur sortie
    float temp = 0;
    for (int i=0; i<n*p*q;i++){
        temp+=in[i]*w[i*n*p*q +l];
    }
    // 1 for tanh 2 for softmax
    if (activation ==1) {
        out[l]= cudaActivationTanh(temp); 
    }
    else if (activation ==2) {
        out[l] = temp;
    }

}



int main(){
    int L=(N-K+1); //dim out conv
    int M=(L/2); //dim out pool

    float *raw_data, *C1_data, *C1_kernel, *S2_data,*C3_dataTemp, *C3_data , *C3_kernel, *S4_data, *F5_data, *F6_data, *OUTPUT, *W1, *W2, *W3;

    srand(time(NULL));

    raw_data = (float*)malloc(N*P * sizeof(float));
    C1_data = (float*)malloc(Q*L*L * sizeof(float));
    C1_kernel = (float*)malloc(Q*K*K * sizeof(float));
    S2_data = (float*)malloc(Q*M*M * sizeof(float));

    C3_dataTemp = (float*)malloc(96*10*10 * sizeof(float));
    C3_data = (float*)malloc(16*10*10 * sizeof(float));
    C3_kernel = (float*)malloc(16*6*K*K * sizeof(float));
    S4_data = (float*)malloc(16*5*5 * sizeof(float));
    
    F5_data = (float*)malloc(120 * sizeof(float));
    F6_data = (float*)malloc(84 * sizeof(float));
    OUTPUT = (float*)malloc(10 * sizeof(float));
    W1 = (float*)malloc(120*16*5*5 * sizeof(float));
    W2 = (float*)malloc(84*120 * sizeof(float));
    W3 = (float*)malloc(84*10 * sizeof(float));

    float combineId[96] = {
        1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,
        1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,
        1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1,
        0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1,
        0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1,
        0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1
    };

    MatrixInit0(raw_data,N,P,1);
    readImage(raw_data);
    //MatrixPrint(raw_data,N,P,1);
    MatrixInit0(C1_data,L,L,Q);
    MatrixInit0(C1_kernel,K,K,Q);
    read_file("k1.h",C1_kernel);
    //MatrixPrint(C1_kernel,K,K,Q);

    MatrixInit0(S2_data,M,M,Q);

    MatrixInit0(C3_dataTemp,10,10,96);
    MatrixInit0(C3_data,10,10,16);
    MatrixInit0(C3_kernel,K,K,16*6);
    read_file("k2.h",C3_kernel);

    MatrixInit0(S4_data,5,5,16);

    MatrixInit0(F5_data,120,1,1);
    MatrixInit0(F6_data,84,1,1);
    MatrixInit0(OUTPUT,10,1,1);
    MatrixInit0(W1,400,120,1);
    read_file("w1.h",W1);
    MatrixInit0(W2,120,84,1);
    read_file("w2.h",W2);
    MatrixInit0(W3,84,10,1);
    read_file("w3.h",W3);


    //MatrixPrint(combineId,6,16,1);


    printf("IMAGE\n");
    //MatrixPrint(raw_data,N,P,1);
    printf("KERNEL\n");
    //MatrixPrint(C1_kernel,K,K,1);
    //GPU
    float *d_combine,*d_raw, *d_C1, *d_C1_kernel, *d_S2,*d_C3Temp, *d_C3, *d_C3_kernel, *d_S4, *d_F5, *d_F6, *d_OUTPUT, *d_W1, *d_W2, *d_W3;

    //CUDA ARRRAY---------------------------------------------------------------------
    cudaMalloc((void**)&d_raw, sizeof(float)*N*P);
    cudaMalloc((void**)&d_C1, sizeof(float)*Q*L*L);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float)*Q*K*K);
    cudaMalloc((void**)&d_S2, sizeof(float)*Q*M*M);
    cudaMalloc((void**)&d_C3Temp, sizeof(float)*96*10*10);
    cudaMalloc((void**)&d_C3, sizeof(float)*16*10*10);
    cudaMalloc((void**)&d_C3_kernel, sizeof(float)*16*6*K*K);
    cudaMalloc((void**)&d_combine, sizeof(float)*16*6);
    cudaMalloc((void**)&d_S4, sizeof(float)*16*5*5);
    cudaMalloc((void**)&d_F5, sizeof(float)*120);
    cudaMalloc((void**)&d_F6, sizeof(float)*84);
    cudaMalloc((void**)&d_OUTPUT, sizeof(float)*10);
    cudaMalloc((void**)&d_W1, sizeof(float)*120*400);
    cudaMalloc((void**)&d_W2, sizeof(float)*120*84);
    cudaMalloc((void**)&d_W3, sizeof(float)*84*10);

    cudaMemcpy(d_raw, raw_data, sizeof(float) * N*P, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1, C1_data, sizeof(float) * Q*L*L, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * Q*K*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2, S2_data, sizeof(float) * Q*M*M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3Temp, C3_dataTemp, sizeof(float) * 96*10*10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3, C3_data, sizeof(float) * 16*10*10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3_kernel, C3_kernel, sizeof(float) * 16*6*K*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_combine, combineId, sizeof(float) * 16*6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S4, S4_data, sizeof(float) * 16*5*5, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F5, F5_data, sizeof(float) * 120, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F6, F6_data, sizeof(float) * 84, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OUTPUT, OUTPUT, sizeof(float) * 10, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, sizeof(float) * 120*400, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, sizeof(float) * 120*84, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, W3, sizeof(float) * 10*84, cudaMemcpyHostToDevice);



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
    dim3 nb_block3(96); //nb kernel
    cudaConv3D<<<nb_block3,nb_thread3>>>(d_S2, d_C3_kernel, d_C3Temp, 14, 14, 96, K );


    //COMBINE ---------------------------------------------------------------------------------
    cudaDeviceSynchronize();
    dim3 nb_thread4(10,10);
    dim3 nb_block4(16);
    cudaCombine<<<nb_block4,nb_thread4>>>(d_C3Temp,d_C3, d_combine);
    cudaDeviceSynchronize();

    //MEAN 2---------------------------------------------------------------------
    dim3 nb_thread5(5,5);
    dim3 nb_block5(16);
    cudaMeanPool<<<nb_block5,nb_thread5>>>(d_C3, d_S4, 10, 10, 16 );
    cudaDeviceSynchronize();


    //FC 1---------------------------------------------------------------------
    dim3 nb_thread6(120);
    cudaFullyConnected<<<1,nb_thread6>>>(d_S4, d_W1, d_F5,  5, 5, 16, 1 );
    cudaDeviceSynchronize();

    //FC 2---------------------------------------------------------------------
    dim3 nb_thread7(84);
    cudaFullyConnected<<<1,nb_thread7>>>(d_F5, d_W2, d_F6,  120, 1, 1, 1 );
    cudaDeviceSynchronize();

    //FC 3---------------------------------------------------------------------
    dim3 nb_thread8(10);
    cudaFullyConnected<<<1,nb_thread8>>>(d_F6, d_W3, d_OUTPUT,  84, 1, 1, 2 );
    cudaDeviceSynchronize();



    //ARRAY COPY TO CPU --------------------------------------------------------------------------
    cudaMemcpy(C1_data, d_C1, sizeof(float)* Q*L*L, cudaMemcpyDeviceToHost);
    printf("C1\n");
    //MatrixPrint(C1_data,L,L,1);
    
    cudaMemcpy(S2_data, d_S2, sizeof(float)* Q*M*M, cudaMemcpyDeviceToHost);
    printf("MEAN\n");
    //MatrixPrint(S2_data,M,M,1); //M<L

    cudaMemcpy(C3_dataTemp, d_C3Temp, sizeof(float)* 96*10*10, cudaMemcpyDeviceToHost);
    cudaMemcpy(C3_data, d_C3, sizeof(float)* 16*10*10, cudaMemcpyDeviceToHost);
    
    printf("C3 Temp\n");
    //MatrixPrint(C3_dataTemp,10,10,16); 
    printf("C3\n");
    //MatrixPrint(C3_data,10,10,16); //M<L

    cudaMemcpy(S4_data, d_S4, sizeof(float)* 16*5*5, cudaMemcpyDeviceToHost);
    printf("MEAN2\n");
    //MatrixPrint(S4_data,5,5,2); 

    cudaMemcpy(OUTPUT, d_OUTPUT, sizeof(float)* 10, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    ActivationSoftmax(OUTPUT,10);
    printf("OUTPUT\n");
    MatrixPrint(OUTPUT,10,1,1); 
    
    indexMax(OUTPUT);

    cudaFree(d_raw);
    cudaFree(d_C1);
    cudaFree(d_C1_kernel);

    free(raw_data);
    free(C1_data);
    free(C1_kernel);
}