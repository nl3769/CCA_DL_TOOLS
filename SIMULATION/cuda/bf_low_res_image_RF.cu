extern "C" { // ---> [kernel]

/* -------------------------------------------------------------------------------------------------------------------- */
/* --------------------------------------- DAS LOW RESOLUTION  -------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------------------------------- */

  __global__ void das_low_res(double *I, const double* RF, const int nb_rx, const int time_sample, const int W, const int H, const double c, const double fs, const double* apod, const int id_tx, const int col_s, const int col_e, const double* tof, const double time_offset)
  {
    /*
    I              -> array that will contains the beamformed images (W * H * nb_tx_elements)
    rf             -> radio fid_rxquence data (nb_tx_elements * time_sample * nb_tx_elements)
    nb_rx          -> number of piezo elements in reception
    time_sample    -> number of time samples
    W              -> width of the low rxsolution image
    H              -> height of the low rxsolution image
    c              -> speed of sound
    fs             -> sampling frequency
    apod           -> apodization window
    id_tx          -> id of tx element
    col_s          -> first col which is reconstructed (to avoid edge issue)
    col_e          -> last col which is reconstructed (to avoid edge issue)
    */

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // --- variables declaration
    int idx;            // index of the image
    double delay;       // starting time for beamforming
    int is1;            // coefficient for interpolation
    int id_apod_rx;     // id to get apodization in id_rxception
    int id_apod_tx;     // id to get apodization in emission
    double tof_;        // time of flight
    double apod_tx;
    double apod_rx;

    // if (col < W  && row < H){
    if (col < W && col >= col_s && col <= col_e  && row < H){
        idx = col * H + row;
        id_apod_tx = H * W * id_tx + idx;

        // loop over rx elements
        for (int id_rx = 0; id_rx < nb_rx; id_rx++){
            id_apod_rx = H * W * id_rx + idx;
            tof_ = tof[id_apod_tx] + tof[id_apod_rx];
            delay = (tof_+ time_offset) * fs  + 1;
            if(delay >= 1 && delay <= (time_sample-1))
            {
              is1 = (int)floor(delay);
              //I[idx] += apod[id_apod_rx] * apod[id_apod_tx] * ( RF[time_sample * id_rx + is1] * (is1 + 1 - delay) + RF[time_sample * id_rx + is1 + 1] * (delay - is1) );
              apod_rx = apod[time_sample * id_rx + is1] * (is1 + 1 - delay) + apod[time_sample * id_rx + is1 + 1] * (delay - is1); 
              apod_tx = apod[time_sample * id_rx + is1] * (is1 + 1 - delay) + apod[time_sample * id_rx + is1 + 1] * (delay - is1); 
;
              I[idx] += apod_tx * apod_rx * ( RF[time_sample * id_rx + is1] * (is1 + 1 - delay) + RF[time_sample * id_rx + is1 + 1] * (delay - is1) );
            }
        }
    }
  }
}
