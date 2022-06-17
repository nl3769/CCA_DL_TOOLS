#ifndef ADD_WRAPPER_HPP
#define ADD_WRAPPER_HPP

void addWithCUDA(const double*, double*, int,double);

void dasbeamformWithCUDA(double* bf_image, const double* rf_data,double start_time, const double* xCoords,const double* yCoords, const double* zCoords, 
	const double* Rx_coords_x,const double* Rx_coords_y, const double* Rx_coords_z, const double ex, const double ey,
	const double ez, int no_elements, int no_samples, int imageW, int imageH, double Us_c, double fs,double* apo,double t_offset);

void resetDevice();

#endif
