#include "cuda_runtime.h"
#include "add_wrapper.hpp"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <iostream>


/* -------------------------------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------------------------------- */

__global__ void addKernel(const double *a, double* c,int numpixels,double apo)
{
    const int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < numpixels)
    {
        c[tid] = c[tid] + apo * a[tid];
    }
}

/* -------------------------------------------------------------------------------------------------------------------- */
/* --------------------------------------- DAS LOW RESOLUTION  -------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------------------------------- */

__global__ void das_low_res(double* bf_image,const double* rf_data,double start_time, 
                            const double* xCoords,const double* yCoords,const double* zCoords, 
                            const double* Rx_coords_x,const double* Rx_coords_y, const double* Rx_coords_z , 
                            const double ex, const double ey, const double ez,int no_elements, 
                            int no_samples,int imageW, int imageH, double Us_c,double fs, double* apo,double t_offset)
{
    
    /* 
    bf_image:       array that will contains the beamformed images (imageW * imageH * nb_tx_elements)
    rf_data:        radio frequence data (nb_tx_elements * time_sample * nb_tx_elements)
    start_time:     starting time in  second
    xCoords:
    zCoords:
    Rx_coords_x:
    Rx_coords_z:
    ex:             position of the central element in meter
    ey:             position of the central element in meter
    ez:             position of the central element in meter
    no_elements:    number of piezo. element? 
    no_samples:     number of time samples
    imageW:         width of the low resolution image
    imageH:         height of the low resolution image
    Us_c:           central frequency of the probe
    fs:             sampling frequency
    apo:            apodization window
    t_offset:       times offset (instead of zero padding with fieldII)
    */

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // --- variable declaration
    int ind;          // TODO
    double txDist;    // distance between transmitted element and pixel
    double rcvDist;   // distance between transmitted element and pixel
    double tof;       // time of flight
    double delay;     // starting time for beamforming
    int is1, is2;     // coefficient for linear interpolation
    
    if (col < imageW && row < imageH){
        ind = row * imageW + col;
        xmtDist = sqrt((ex - xCoords[ind]) * (ex - xCoords[ind]) + (ez - zCoords[ind]) * (ez - zCoords[ind])) / Us_c;                                                         // compute distance between the the tx element of current coordinate
        
        // --- loop over received elements
        for (int re = 0; re < no_elements; re++){
            rcvDist= sqrt((Rx_coords_x[re]- xCoords[ind])* (Rx_coords_x[re] - xCoords[ind])+ (Rx_coords_z[re] - zCoords[ind])* (Rx_coords_z[re] - zCoords[ind])) / Us_c;      // compute distance between the the tx element of current coordinate
            tof = (rcvDist + xmtDist);                          // total time of flight
            delay = (tof - start_time + t_offset) * fs + 1;     // check if delay point to an existing value
            if (delay >= 1 && delay <= (no_samples - 1)){
                is1 = (int)floor(delay);
                is2 = is1 - 1;
                bf_image[ind] += apo[re] * (rf_data[no_samples * re + is1] * (is1+1-delay) + rf_data[no_samples * re + is1+1] * (delay-is1));
            }
        }
    }
}

/* -------------------------------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------------------------------- */
void addWithCUDA(const double* a, double* c,int numpixels, double apo){
    int NUM_THREADS = 1024;
    int NUM_BLOCKS = (numpixels + NUM_THREADS - 1) / NUM_THREADS;
    addKernel << <NUM_BLOCKS, NUM_THREADS >> > (a, c, numpixels,apo);
}

void dasbeamformWithCUDA(double* bf_image, const double* rf_data,double start_time, const double* xCoords,const double* yCoords, const double* zCoords, const double* Rx_coords_x,const double* Rx_coords_y, const double* Rx_coords_z, const double ex, const double ey, const double ez, int no_elements, int no_samples, int imageW, int imageH, double Us_c, double fs, double* apo,double t_offset)
{
    int BLOCKDIM_X = 32;
    int BLOCKDIM_Y = 32;
    dim3 dimGrid(((imageW - 1) / BLOCKDIM_X) + 1, ((imageH - 1) / BLOCKDIM_Y) + 1);
    dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y);
    daslowResolutionImage << <dimGrid, dimBlock >> > (bf_image,  rf_data, start_time ,  xCoords,  yCoords,  zCoords,  Rx_coords_x,  Rx_coords_y,  Rx_coords_z,   ex,   ey,   ez,  no_elements,  no_samples,  imageW,  imageH,  Us_c,  fs,apo, t_offset);
    cudaDeviceSynchronize();
}

void resetDevice()
{
    cudaDeviceReset();
}
