

#include "include/verfiy.h"
#include "include/conv2d.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define _Float16 float

/*选手自定义的kernel入参结构体*/
struct mykernelParamType
{
    _Float16*   pin;                            //输入数据地址
    _Float16*   pweight;                        //权值数据地址
    _Float16*   pout;                           //输出数据地址
    unsigned int      n;                              //batch szie            
    unsigned int      c;                              //channel number        
    unsigned int      h;                              //数据高                
    unsigned int      w;                              //数据宽                
    unsigned int      k;                              //卷积核数量            
    unsigned int      r;                              //卷积核高              
    unsigned int      s;                              //卷积核宽              
    unsigned int      u;                              //卷积在高方向上的步长  
    unsigned int      v;                              //卷积在宽方向上的步长  
    unsigned int      p;                              //卷积在高方向上的补边  
    unsigned int      q;                              //卷积在宽方向上的补边  
    unsigned int      Oh;                             //卷积在高方向上输出大小    
    unsigned int      Ow;                             //卷积在宽方向上输出大小
};    


extern "C" __global__ void myKernelConv2dGpu(mykernelParamType param){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    
    if(x >= param.Oh*param.Ow || y >= param.k || z >= param.n) return;
    
    int posOh = x/param.Ow;
    int posOw = x%param.Ow;
    int posh_ori = posOh*param.u - param.p;
    int posw_ori = posOw*param.v - param.q;
    
    float sum = 0.0;
    int inOffset = z*param.c*param.h*param.w + posh_ori*param.w + posw_ori;
    int weiOffset = y*param.c*param.r*param.s;
    int inChannelOffset = param.h*param.w;
    int weightChannelOffset = param.r*param.s;
    
    for(int i = 0; i < param.r; i++){
        for(int j = 0; j < param.s; j++){
            int posh_real = posh_ori + i;
            int posw_real = posw_ori + j;            
            if(posh_real>=0 && posw_real>=0 && posw_real<param.w && posh_real<param.h){
                int inOffsetTmp = inOffset;
                int weiOffsetTmp = weiOffset;
                for(int channel = 0; channel<param.c; channel++){
                    sum += (float)(param.pin[inOffsetTmp + i*param.w + j] * param.pweight[weiOffsetTmp + i*param.s + j]);
                    inOffsetTmp += inChannelOffset;
                    weiOffsetTmp += weightChannelOffset;
                }               
            }
        }
    }   

    int outOffset = z*param.k*param.Oh*param.Ow + y*param.Oh*param.Ow + x;
    param.pout[outOffset] = (_Float16)sum;
}






/*选手需要返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param){
    mykernelParamType* pArgs = (mykernelParamType*)param;

    unsigned int n = problem->n;
    unsigned int c = problem->c;
    unsigned int h = problem->h;
    unsigned int w = problem->w;
    unsigned int k = problem->k;
    unsigned int r = problem->r;
    unsigned int s = problem->s;
    unsigned int u = problem->u;
    unsigned int v = problem->v;
    unsigned int p = problem->p;
    unsigned int q = problem->q;

    unsigned int outh = (h - r + 2*p)/u + 1;
    unsigned int outw = (w - s + 2*q)/v + 1;

    kernelInfo->blockx   = (outh*outw + 15)/16;                    //blockx  number
    kernelInfo->blocky   = (k+15)/16;                    //blocky  number
    kernelInfo->blockz   = n;                    //blockz  number
    kernelInfo->threadx  = 16;                   //threadx number per block
    kernelInfo->thready  = 16;                   //thready number per block
    kernelInfo->threadz  = 1;                   //threadz number per block
    kernelInfo->dynmicLdsSize = 0;
    kernelInfo->kernelPtr= (void*)myKernelConv2dGpu;                 //kernel ptr

    pArgs->pin = problem->in;
    pArgs->pweight = problem->weight;
    pArgs->pout = problem->out;
    pArgs->n = n;                              //batch szie              default value 1
    pArgs->c = c;                              //channel number          default value 32
    pArgs->h = h;                              //数据高                  default value 32
    pArgs->w = w;                              //数据宽                  default value 32
    pArgs->k = k;                              //卷积核数量              default value 32
    pArgs->r = r;                              //卷积核高                default value 1
    pArgs->s = s;                              //卷积核宽                default value 1
    pArgs->u = u;                              //卷积在高方向上的步长     default value 1
    pArgs->v = v;                              //卷积在宽方向上的步长     default value 1
    pArgs->p = p;                              //卷积在高方向上的补边     default value 0
    pArgs->q = q;                              //卷积在宽方向上的补边     default value 0
    pArgs->Oh = outh;
    pArgs->Ow = outw;       

    return 0;
}


void cudaExtLaunchKernel(kernelInfo_t* kernelInfo, mykernelParamType* params){
    dim3 groups(kernelInfo->blockx, kernelInfo->blocky, kernelInfo->blockz);
    dim3 threads(kernelInfo->threadx, kernelInfo->thready, kernelInfo->threadz);
    myKernelConv2dGpu<<<groups, threads>>>(*params);
}


int main(int argc, char**argv){
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);
    int h = atoi(argv[3]);
    int w = atoi(argv[4]);
    int k = atoi(argv[5]);
    int r = atoi(argv[6]);
    int s = atoi(argv[7]);
    int u = atoi(argv[8]);
    int v = atoi(argv[9]);
    int p = atoi(argv[10]);
    int q = atoi(argv[11]);

    int outh = (h - r + 2*p)/u + 1;
    int outw = (w - s + 2*q)/v + 1;


    _Float16 *pIn       = (_Float16*)malloc(n*c*h*w*sizeof(_Float16));
    _Float16 *pWeight   = (_Float16*)malloc(k*c*r*s*sizeof(_Float16));
    _Float16 *pOut      = (_Float16*)malloc(n*k*outh*outw*sizeof(_Float16));
    _Float16 *pOut_host = (_Float16*)malloc(n*k*outh*outw*sizeof(_Float16));

    _Float16 *pIn_device,*pWeight_device,*pOut_device;
    cudaMalloc((void**)&pIn_device, n*c*h*w*sizeof(_Float16));
    cudaMalloc((void**)&pWeight_device, k*c*r*s*sizeof(_Float16));
    cudaMalloc((void**)&pOut_device, n*k*outh*outw*sizeof(_Float16));
    
    for(int i = 0; i < n*c*h*w; i++){
        pIn[i] = (rand()%255)/255.0;
    }
    
    for(int i = 0; i < k*c*r*s; i++){
        pWeight[i] = (rand()%255)/255.0;
    }
    
    for(int i = 0; i < n*k*outh*outw; i++){
        pOut[i] = 0.0;
        pOut_host[i] = 0.0;
    }
           
    cudaMemcpy(pIn_device, pIn, n*c*h*w*sizeof(_Float16),cudaMemcpyHostToDevice);
    cudaMemcpy(pWeight_device,pWeight,k*c*r*s*sizeof(_Float16),cudaMemcpyHostToDevice);
    cudaMemcpy(pOut_device,pOut,n*k*outh*outw*sizeof(_Float16),cudaMemcpyHostToDevice);
   
    /********************step 1*****************************/

    problem_t problem;
    kernelInfo_t kernelInfo;

    problem.in        = pIn_device;        
    problem.weight    = pWeight_device;
    problem.out       = pOut_device;             
    problem.n         = n;                             
    problem.c         = c;                             
    problem.h         = h;                             
    problem.w         = w;                             
    problem.k         = k;                             
    problem.r         = r;                             
    problem.s         = s;                             
    problem.u         = u;                             
    problem.v         = v;                             
    problem.p         = p;                             
    problem.q         = q;                               

    /********************************** step 2****************************/
    mykernelParamType* param = (mykernelParamType*)malloc(sizeof(mykernelParamType));
    getkernelInfo(&problem, &kernelInfo, param);
        
    /*******************************warm up and get result************************************/
    cudaExtLaunchKernel(&kernelInfo, param);
    cudaMemcpy(pOut_host, pOut_device,  n*k*outh*outw*sizeof(_Float16), cudaMemcpyDeviceToHost); 


    /*******************************cost time test************************************/
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    float time_elapsed=0.0;
    
    int iternum = 10;
    for(int i=0; i<iternum; i++){
        cudaExtLaunchKernel(&kernelInfo, param); 
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed,start,stop);
    cudaDeviceSynchronize();

    printf("time: %f us\n", time_elapsed*1000/iternum);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
      
    
    printf("output size : %d \n",n*k*outh*outw);
    printf("===================start verfiy===================\n");
    conv2dcpu(pIn, pWeight, pOut, n, c, h, w, k, r, s, u, v, p, q);

    int error=0;
    for(int i=0;i<n*k*outh*outw;i++){
        if((fabs(pOut_host[i] - pOut[i]))/pOut_host[i] > 0.01){
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f\n", i, pOut_host[i], pOut[i]);
            error++;
            break;
        }        
    }

    printf("================finish,error:%d=========================\n",error);

    cudaFree(pIn_device);
    cudaFree(pWeight_device);
    cudaFree(pOut_device);
    
    free(pIn);
    free(pWeight);
    free(pOut);
    free(pOut_host);

    free(param);
    return 0;
}