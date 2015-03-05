#ifndef MT_CPU_H
#define MT_CPU_H


#include <cmath>
#include <stdio.h>
#include <string.h>


//#include "err_code.h"


#include "mt_defines.h"

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include <iostream>
#include <vector>
#include "util.hpp"
//#include <err_code.h>


void mt_cpu(	int		n_step,				// полное количество шагов по времени
				int 	load_coords,		//

				float 	x_in[][N_d],
				float 	y_in[][N_d],
				float 	t_in[][N_d],

				float 	x_out[][N_d],
				float 	y_out[][N_d],
				float 	t_out[][N_d]
			);




unsigned getDeviceList(std::vector<cl::Device>& devices);
void getDeviceName(cl::Device& device, std::string& name);

#endif //MT_CPU_H
