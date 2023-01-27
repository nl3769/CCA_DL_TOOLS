extern "C" { // ---> [kernel]

/* -------------------------------------------------------------------------------------------------------------------- */
/* --------------------------------------- DAS LOW RESOLUTION  -------------------------------------------------------- */
/* ------------------------------------------ LAINE NOLANN ------------------------------------------------------------ */
/* -------------------------------------------------------------------------------------------------------------------- */

  __global__ void bf_low_res_images(const double* Iiq, const double* Riq, const double* pos_el_x, const double* pos_el_z, const int nb_el, const int nb_sample, const int W, const int H, const double c, const double fs, const double* apod, const double* x_img, const double* z_img, const double* t_offset, double* I, double* R)
  {

    // --- variables
    int idx;            // index of the image
    double delay;       // exact time (to read RF signals)
    int id_t;           // time index used for interpolation
    double tof;         // time of flight
    int id_apod_tx;     // tx apodization
    int id_apod_rx;     // rx apodization
    double rx_dst;      // pts to rx element in meter
    double tx_dst;      // pts to tx element in meter
    double apod_tx;     // apodization in emission
    double apod_rx;     // apodization in reception
    double val_sample;  // value of the beamformed pts

    int col = blockIdx.x * blockDim.x + threadIdx.x;    // column in the grid image
    int row = blockIdx.y * blockDim.y + threadIdx.y;    // row in the grid image
    int id_tx = blockIdx.z * blockDim.z + threadIdx.z;  // id_tx element

    if (col < W  && row < H && id_tx < nb_el)
    {
        idx = col * H + row;
        id_apod_tx = W * H * id_tx + idx;
        tx_dst = sqrt( (pos_el_x[id_tx] - x_img[col])*(pos_el_x[id_tx] - x_img[col]) + (pos_el_z[id_tx] - z_img[row])*(pos_el_z[id_tx] - z_img[row]) );
        // --- loop over rx elements
        for (int id_rx = 0; id_rx < nb_el; id_rx++)
        {
            rx_dst = sqrt( (pos_el_x[id_rx] - x_img[col])*(pos_el_x[id_rx] - x_img[col]) + (pos_el_z[id_rx] - z_img[row])*(pos_el_z[id_rx] - z_img[row]) );
            tof = (rx_dst + tx_dst)/c  + t_offset[id_tx];
            delay = tof * fs + 1;
            if(delay >= 1 && delay <= (nb_sample-1))
            {
              id_t = (int)floor(delay);
              id_apod_rx = W * H * id_rx + idx;
              apod_tx = apod[id_apod_tx];
              apod_rx = apod[id_apod_rx];
              // --- beamform imaginary part
              val_sample = Iiq[nb_el * nb_sample * id_tx + nb_sample * id_rx + id_t] * (id_t + 1 - delay) + Iiq[nb_el * nb_sample * id_tx + nb_sample * id_rx + id_t + 1] * (delay - id_t);
              I[H * W * id_tx + idx] += apod_tx * apod_rx * val_sample;
              // --- beamform real part
              val_sample = Riq[nb_el * nb_sample * id_tx + nb_sample * id_rx + id_t] * (id_t + 1 - delay) + Riq[nb_el * nb_sample * id_tx + nb_sample * id_rx + id_t + 1] * (delay - id_t);
              R[H * W * id_tx + idx] += apod_tx * apod_rx * val_sample;
            }
        }
    }
  }
}
