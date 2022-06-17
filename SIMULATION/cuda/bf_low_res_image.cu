extern "C" { // ---> [kernel]

/* -------------------------------------------------------------------------------------------------------------------- */
/* --------------------------------------- DAS LOW RESOLUTION  -------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------------------------------- */

  __global__ void das_low_res(double* I, const double* rf, const int nb_rx, const int n_rcv, const int W, const int H, const double c, const double fs, const double* apod, const int id_tx, const int col_s, const int col_e, const double* tof)
  {
    /*
    I              -> array that will contains the beamformed images (W * H * nb_tx_elements)
    rf             -> radio fid_rxquence data (nb_tx_elements * n_rcv * nb_tx_elements)
    xCoords        -> coordinate of the image in meter
    zCoords        -> coordinates of the image in meter
    ex             -> position of tx element in x direction
    ez             -> position of tx element in z direction
    nb_rx          -> number of piezo elements in reception
    n_rcv          -> number of time samples
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
    int idx;       // index of the image
    double delay;       // starting time for beamforming
    int is1;            // coefficient for interpolation
    int id_apod_rx;     // id to get apodization in id_rxception
    int id_apod_tx;     // id to get apodization in emission
    double tof_tx;
    double tof_rx;
    double tof_;

    if (col < W  && row < H){
    // if (col < W && col > (col_s-1) && col < col_e  && row < H){
        idx = col * H + row;
        id_apod_tx = H * W * id_tx + idx;
        tof_tx = tof[id_apod_tx];

        // --- loop over rx elements
        for (int id_rx = 0; id_rx < nb_rx; id_rx++){
        //for (int id_rx = 96; id_rx < 97; id_rx++){
            id_apod_rx = H * W * id_rx + idx;
            tof_rx = tof[id_apod_rx];
            tof_ = (tof_rx + tof_tx);
            delay = tof_ * fs + 1;
            if(delay >= 1 && delay <= (n_rcv-1)){
              is1 = (int)floor(delay);
              //I[idx] += ( rf[n_rcv * id_rx + is1] * (is1 + 1 - delay) + rf[n_rcv * id_rx + is1 + 1] * (delay - is1) );
              I[idx] += apod[id_apod_rx] * apod[id_apod_tx] * ( rf[n_rcv * id_rx + is1] * (is1 + 1 - delay) + rf[n_rcv * id_rx + is1 + 1] * (delay - is1) );
              // I[idx] += apod[id_apod_rx] * apod[id_apod_tx];
            }
        }
    }
  }
}
