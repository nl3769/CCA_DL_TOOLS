#include "mex.h"
#include "add_wrapper.hpp"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
///#include<gpu/mxGPUArray.h>


#define GBF_DAS_HIGHRESIMAGE 1
#define GBF_DAS_LOWRESIMAGES 2
#define GBF_DAS_LOWRESIMAGE 3

void verifyMexArguments(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 7) {
        mexErrMsgTxt("Not enough input arguments. Expected 7 arguments");
    }
    int nbFields = mxGetNumberOfFields(prhs[6]);
    if (nbFields != 10) {
        mexErrMsgTxt("Not enough field in the params struct. Expected 10 fields");
    }
}

void beamformLowResolutionImage(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

    // ...Get emitter position
    int field_num = mxGetFieldNumber(prhs[6], "tx_x");
    if (field_num==-1)
    {
        mexErrMsgTxt("tx_x field is not set. Set a field tx_x for transmit element x position in the params struct");
    }
    const mxArray * mxTx_posx = mxGetFieldByNumber(prhs[6], 0, field_num);
    double Tx_pos_x = double(mxGetPr(mxTx_posx)[0]);

    field_num = mxGetFieldNumber(prhs[6], "tx_y");
    if (field_num == -1)
    {
        mexErrMsgTxt("tx_y field is not set. Set a field tx_y for transmit element x position in the params struct");
    }
    const mxArray* mxTx_posy = mxGetFieldByNumber(prhs[6], 0, field_num);
    double Tx_pos_y = double(mxGetPr(mxTx_posy)[0]);

    field_num = mxGetFieldNumber(prhs[6], "tx_z");
    if (field_num == -1)
    {
        mexErrMsgTxt("tx_z field is not set. Set a field tx_z for transmit element x position in the params struct");
    }
    const mxArray* mxTx_posz = mxGetFieldByNumber(prhs[6], 0, field_num);
    double Tx_pos_z = double(mxGetPr(mxTx_posz)[0]);
    //std::cout << "Emitter position : " << Tx_pos_x << " , " << Tx_pos_y << " , " << Tx_pos_z << std::endl;
    // ...Get reciever position
    field_num = mxGetFieldNumber(prhs[6], "rx_x");
    if (field_num == -1)
    {
        mexErrMsgTxt("rx_x field is not set. Set a field rx_x for recieve element x positions in the params struct");
    }
    const mxArray* mxRx_posx = mxGetFieldByNumber(prhs[6], 0, field_num);
    double* Rx_pos_x = mxGetPr(mxRx_posx);
    field_num = mxGetFieldNumber(prhs[6], "rx_y");
    if (field_num == -1)
    {
        mexErrMsgTxt("rx_y field is not set. Set a field rx_y for recieve element x positions in the params struct");
    }
    const mxArray* mxRx_posy = mxGetFieldByNumber(prhs[6], 0, field_num);
    double* Rx_pos_y = mxGetPr(mxRx_posy);
    field_num = mxGetFieldNumber(prhs[6], "rx_z");
    if (field_num == -1)
    {
        mexErrMsgTxt("rx_z field is not set. Set a field rx_z for recieve element x positions in the params struct");
    }
    const mxArray* mxRx_posz = mxGetFieldByNumber(prhs[6], 0, field_num);
    double* Rx_pos_z = mxGetPr(mxRx_posz);
    // set reciever number
    int no_elements = mxGetNumberOfElements(mxRx_posx);
    //std::cout << "Numner of recievers : " << no_elements << std::endl;
    // ...Get c
    field_num = mxGetFieldNumber(prhs[6], "c");
    if (field_num == -1)
    {
        mexErrMsgTxt("c field is not set. Set a field c for the ultrasound speed in the params struct");
    }
    const mxArray* mxUS_c = mxGetFieldByNumber(prhs[6], 0, field_num);
    double Us_c = double(mxGetPr(mxUS_c)[0]);
    //std::cout << "Speed of sound : " << Us_c << std::endl;
    // ...Get fs
    field_num = mxGetFieldNumber(prhs[6], "fs");
    if (field_num == -1)
    {
        mexErrMsgTxt("fs field is not set. Set a field fs for the sampling frequency in the params struct");
    }
    const mxArray* mxfs = mxGetFieldByNumber(prhs[6], 0, field_num);
    double fs = double(mxGetPr(mxfs)[0]);
    //std::cout << "Sampling frequency: " << fs << std::endl;
    // ...Get recieve apodization
    field_num = mxGetFieldNumber(prhs[6], "RxApodization");
    if (field_num == -1)
    {
        mexErrMsgTxt("RxApodization field is not set. Set a field RxApodization for the recieve apodization in the params struct");
    }
    const mxArray* mxApodization = mxGetFieldByNumber(prhs[6], 0, field_num);
    double* Rx_Apodization = mxGetPr(mxApodization);
    //std::cout << "apodization : " << Rx_Apodization[120] << std::endl;
    // ...Get time offset
    field_num = mxGetFieldNumber(prhs[6], "t_offset");
    if (field_num == -1)
    {
        mexErrMsgTxt("t_offset field is not set. Set a field t_offset for the a time offset in the params struct");
    }
    const mxArray* mxOffset = mxGetFieldByNumber(prhs[6], 0, field_num);
    double t_offset = double(mxGetPr(mxOffset)[0]);
    //std::cout << "time offset  : " << t_offset << std::endl;

    // ...Get the rf signals on the host
    double* h_rf_data = mxGetPr(prhs[1]);
    // ...Number of rf samples
    double numberOfSamples = mxGetM(prhs[1]);
    //std::cout << "Number of Rf samples : " << numberOfSamples << std::endl;

    // Get corresponding start time
    double start_time = double(mxGetPr(prhs[2])[0]);
    //std::cout << "Corresponding start time : " << start_time << std::endl;
    //t_offset = start_time;
    // ...Get field points aka pixels
    double* xField = mxGetPr(prhs[3]);
    double* yField = mxGetPr(prhs[4]);
    double* zField = mxGetPr(prhs[5]);
    // Number of lateral pixels in the image 
    double xlen = mxGetN(prhs[3]);
    // Number of axial pixels in the image 
    double zlen = mxGetM(prhs[3]);
    //std::cout << "Number of lateral pixels in the image : " << xlen << std::endl;
    //std::cout << "Number of axial pixels in the image : " << zlen << std::endl;

    // Create the beamformed image array
    mxArray* bfi = mxCreateDoubleMatrix(zlen, xlen, mxREAL);
    // Allocate memory on the GPU
    // the beamformed image
    double* bf_image;
    cudaMalloc((void**)&bf_image, sizeof(double) * xlen * zlen);
    // the recievers position and the recieve apodization
    double* d_rx_pos_x;
    double* d_rx_pos_y;
    double* d_rx_pos_z;
    double* d_rx_apo;
    cudaMalloc((void**)&d_rx_pos_x, sizeof(double) * no_elements);
    cudaMalloc((void**)&d_rx_pos_y, sizeof(double) * no_elements);
    cudaMalloc((void**)&d_rx_pos_z, sizeof(double) * no_elements);
    cudaMalloc((void**)&d_rx_apo, sizeof(double) * no_elements);
    // Copy to the gpu
    cudaMemcpy(d_rx_pos_x, Rx_pos_x, sizeof(double) * no_elements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx_pos_y, Rx_pos_y, sizeof(double) * no_elements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx_pos_z, Rx_pos_z, sizeof(double) * no_elements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx_apo, Rx_Apodization, sizeof(double) * no_elements, cudaMemcpyHostToDevice);
    // The region to beamform
    double* d_xField;
    double* d_yField;
    double* d_zField;
    cudaMalloc((void**)&d_xField, sizeof(double) * xlen * zlen);
    cudaMalloc((void**)&d_yField, sizeof(double) * xlen * zlen);
    cudaMalloc((void**)&d_zField, sizeof(double) * xlen * zlen);
    // Copy to the gpu
    cudaMemcpy(d_xField, xField, sizeof(double) * xlen * zlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_yField, yField, sizeof(double) * xlen * zlen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_zField, zField, sizeof(double) * xlen * zlen, cudaMemcpyHostToDevice);
    // The radio frequency signals
    double* d_rf_data;
    cudaMalloc((void**)&d_rf_data, sizeof(double) * numberOfSamples * no_elements);
    // Copy rf data to the gpu
    cudaMemcpy(d_rf_data, h_rf_data, sizeof(double) * numberOfSamples * no_elements, cudaMemcpyHostToDevice);
    //resetDevice();
    // Call the beamformation kernel
    dasbeamformWithCUDA(bf_image, d_rf_data, start_time,d_xField, d_yField, d_zField, d_rx_pos_x, d_rx_pos_y, d_rx_pos_z, Tx_pos_x, Tx_pos_y, Tx_pos_z, no_elements, numberOfSamples, xlen, zlen, Us_c, fs, d_rx_apo, t_offset);
    //resetDevice();
    // copy beamformed image from gpu to cpu
    double* d_bfi = (double*)mxGetData(bfi);
    cudaMemcpy(d_bfi, bf_image, xlen * zlen * sizeof(double), cudaMemcpyDeviceToHost);

    // Output the beamformed image
    plhs[0] = bfi;

    // Free the GPU Memory
    cudaFree(d_rf_data);
    cudaFree(d_xField);
    cudaFree(d_yField);
    cudaFree(d_zField);
    cudaFree(d_rx_pos_x);
    cudaFree(d_rx_pos_y);
    cudaFree(d_rx_pos_z);
    cudaFree(d_rx_apo);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    verifyMexArguments(nlhs, plhs, nrhs, prhs);

    int id = (int)floor(mxGetScalar(prhs[0]) + 0.5);
    switch (id)
    {
    case GBF_DAS_HIGHRESIMAGE:
        //gbf_das_highresImage(nlhs, plhs, nrhs, prhs); break;
    case GBF_DAS_LOWRESIMAGES:
        //gbf_das_LowresImages(nlhs, plhs, nrhs, prhs); break;
    case GBF_DAS_LOWRESIMAGE:
        beamformLowResolutionImage(nlhs, plhs, nrhs, prhs); break;
    default:
        mexErrMsgTxt("Unkown function id");
        mexEvalString("clear mex");
        break;
    }
    mexEvalString("clear mex");
}


















    































//// Initialization
  //mxInitGPU();
  //// Define variables
  //const mxGPUArray* field;  

  //mxGPUArray* dst;
  //const double* c_field;
  //double* d_dst;
  //int N1, N2;

  //// Check the number of arguments
  //if (nrhs != 4) {
  //    mexErrMsgIdAndTxt("MATLAB:vec_add", "The number of input arguments must be 4.");
  //}
  //if (nlhs != 1) {
  //    mexErrMsgIdAndTxt("MATLAB:vec_add", "The number of output arguments must be 1.");
  //}



  //// ...Get rf data to beamform
  //field = mxGPUCreateFromMxArray(prhs[0]);










  //double x = mxGetScalar(prhs[1]);
  //double y = mxGetScalar(prhs[2]);
  //double z = mxGetScalar(prhs[3]);
  //mexPrintf("x %f", x);
  //mexPrintf("y %f", y);
  //mexPrintf("z %f", z);
  //// Number of dimensions
  //mwSize nDimNum = mxGetNumberOfDimensions(prhs[0]);
  //// dimensions
  //const mwSize* pDims = mxGetDimensions(prhs[0]);
  //int nulpixels = pDims[0];
  //mexPrintf("numpixels %d", nulpixels);
  //// Check the dimension of src vectors
  //N1 = (int)(mxGPUGetNumberOfElements(field));
  //mexPrintf("num elements %d", N1);

  //// Get address of src1 and src2
  //c_field = (const double*)(mxGPUGetDataReadOnly(field));

  //// Allocate memory of the destination variable on device memory
  //mwSize outDim = 1;
  ////mwSize* outDims;
  ////outDims[0] = pDims[0];
  //const mwSize outDims[] = { pDims[0] };
  //mexPrintf("create dst\n");
  //dst = mxGPUCreateGPUArray(outDim,
  //    outDims,
  //    mxDOUBLE_CLASS,
  //    mxREAL,
  //    MX_GPU_DO_NOT_INITIALIZE);
  //mexPrintf("done dst");
  //d_dst = (double*)(mxGPUGetData(dst));

  //// Call kernel function
  ////dim3 block(N1);
  ////dim3 grid((N1 + block.x - 1) / block.x);
  ////vec_add << <grid, block >> > (d_src1, d_src2, k1, k2, d_dst, N1);
  //printf("Starting cuda kernel\n");
  //addWithCUDA(c_field, d_dst, x, y, z, N1, nulpixels);
  //printf("done\n");

  ////plhs[0] = mxCreateNumericArray(nDimNum, pDims, mxDOUBLE_CLASS, mxREAL);
  //// Pass dst to plhs[0]
  //plhs[0] = mxCreateDoubleMatrix(outDims[0], 1, mxREAL);
  //plhs[0] = mxGPUCreateMxArrayOnCPU(dst);

  //// Release memory
  //mxGPUDestroyGPUArray(field);
  ////mxGPUDestroyGPUArray(dst