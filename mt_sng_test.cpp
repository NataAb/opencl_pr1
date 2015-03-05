#include "mt_cpu.h"
#include "err_code.h"
#include "device_picker.hpp"




#define COMPARE


#define COMP_LIMIT		0.01f

//#define ONLINECOMPLE1



// ai a?aiy aaaaaa TOTAL_STEPS ii?ii aaeaou iaiuoa
#define TOTAL_STEPS             10000       // iieiia eiee?anoai oaaia ii a?aiaie

#define STEPS_TO_WRITE          1               // ?a?ac yoi cia?aiea oaaia n?aaieaaai eii?aeiaou e auaiaei a oaee

void calculate_constants();
void init_coords(float x[][N_d], float y[][N_d], float t[][N_d]);
void init_coords(std::vector<float> & x, std::vector<float> & y, std::vector<float> & t);
int compare_results(float x_1[][N_d], float y_1[][N_d], float t_1[][N_d],
float x_2[][N_d], float y_2[][N_d], float t_2[][N_d] );
int compare_results(float x_1[][N_d], float y_1[][N_d], float t_1[][N_d],
std::vector<float>& ar1, std::vector<float>& ar2, std::vector<float>& ar3 ); 
int compare_results(std::vector<float>& ar1, std::vector<float>& ar2, std::vector<float>& ar3,
std::vector<float>& ar4, std::vector<float>& ar5, std::vector<float>& ar6 );								  

void print_coords(FILE *f_p, float x[][N_d], float y[][N_d], float t[][N_d]);
void print_coords(FILE* fp, std::vector<float>& ar1, std::vector<float>& ar2, std::vector<float>& ar3);

void init_cl(cl::Context &context, cl::CommandQueue &queue, cl::Program &program);


float x_1[13][N_d];
float y_1[13][N_d];
float t_1[13][N_d];

int size = N_d*13;
std::vector<float> h_x_inout(size); 
std::vector<float> h_y_inout(size); 
std::vector<float> h_t_inout(size);


cl::Buffer d_x_in, d_y_in, d_t_in, d_x_out, d_y_out, d_t_out, d_x_inout, d_y_inout, d_t_inout;


double start_time;      // Starting time
double run_time;        // Timing data
util::Timer timer;      // Timer



int main() {

	//calculate_constants();


	FILE *f_u, *f_p, *f_cl1;
	int i,j;

	int error = 0;

	f_p = fopen ("MT_coords_CPU.txt","w");
	f_cl1 = fopen("MT_coords_CPU_CL.txt","w");

	if (f_p==NULL) {

		printf("Error opening file!\n");
		return -1;
	}



	


	// get golden results
	init_coords(x_1,y_1,t_1);



	/////////////////////////////////////////////////////////opencl///////////////////////////////////////////////////////

	try{

		cl::Context context;
		cl::CommandQueue queue;
		cl::Program program;
		
		init_coords(h_x_inout, h_y_inout, h_t_inout);

		init_cl(context, queue, program);

		
		
		#ifndef COMPARE
		std::cout<<"cpu  started "<<std::endl;
		start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
		for(int k=0; k<(TOTAL_STEPS/STEPS_TO_WRITE); k++) {
			mt_cpu(STEPS_TO_WRITE,1,x_1,y_1,t_1,x_1,y_1, t_1);
			print_coords(f_p, x_1, y_1, t_1);
		}
		run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
		std::cout<<"cpu time is "<<run_time<<std::endl;
		#endif



		
		std::cout<<"TEST  started "<<std::endl;

		#ifdef ONLINECOMPLE1
		program = cl::Program(context, util::loadProgram("./prog1.cl"), true);
		#else
		cl_uint deviceIndex = 0;

		// Get list of devices
		std::vector<cl::Device> devices;
		unsigned numDevices = getDeviceList(devices);
		cl::Device device = devices[deviceIndex];
		std::vector<cl::Device> chosen_device;
		chosen_device.push_back(device);
		FILE *fp = fopen("./prog1.bc", "rb");
		if (fp == NULL)
		{
			printf("Error opening cl compiled file");
			return -1;
		}

		// Determine the size of the binary
		size_t binarySize;
		fseek(fp, 0, SEEK_END);
		binarySize = ftell(fp);
		rewind(fp);

		unsigned char *programBinary = new unsigned char[binarySize];
		fread(programBinary, 1, binarySize, fp);
		fclose(fp);
		std::vector< std::pair<const void*, ::size_t> > vec;
		vec.push_back(std::pair<const void*, ::size_t>(programBinary, binarySize));
		program = cl::Program(context, chosen_device, vec);
		delete [] programBinary;
		clBuildProgram(program(), 0,  NULL, NULL, NULL, NULL);
		
		cl:: Kernel t(program, "cl_calc", NULL);
		cl_kernel_work_group_info workGroupSizeUsed;
		clGetKernelWorkGroupInfo (t(),device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void*)&workGroupSizeUsed, NULL);
		std::cout<<"allowed workgroup size is "<<workGroupSizeUsed<<std::endl;
		
		#endif
		cl::make_kernel<int,cl::Buffer, cl::Buffer, cl::Buffer> cl_calc(program, "cl_calc");

		cl::EnqueueArgs temp(queue, cl::NDRange(16) , cl::NDRange(16));	 	

		d_x_inout = cl::Buffer(context, h_x_inout.begin(), h_x_inout.end(), false);
		d_y_inout = cl::Buffer(context, h_y_inout.begin(), h_y_inout.end(), false);        
		d_t_inout = cl::Buffer(context, h_t_inout.begin(), h_t_inout.end(), false);

		
		
		std::cout<<"16  started "<<std::endl;
		
		#ifndef COMPARE
		start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;		
		#endif
		

		
		for(int k=0; k<(TOTAL_STEPS/STEPS_TO_WRITE); k++) {
			#ifdef COMPARE 
			mt_cpu(STEPS_TO_WRITE,1,x_1,y_1,t_1,x_1,y_1, t_1);
			#endif	
			cl_calc(temp,STEPS_TO_WRITE, d_x_inout, d_y_inout, d_t_inout);

			cl::copy(queue, d_x_inout, h_x_inout.begin(), h_x_inout.end());
			cl::copy(queue, d_y_inout, h_y_inout.begin(), h_y_inout.end());
			cl::copy(queue, d_t_inout, h_t_inout.begin(), h_t_inout.end());			
			print_coords(f_cl1, h_x_inout, h_y_inout, h_t_inout );
			#ifdef COMPARE 	
			print_coords(f_p, x_1, y_1, t_1);	
			error = compare_results(x_1,y_1,t_1,h_x_inout,h_y_inout,h_t_inout);
			if (error)
			{printf("16 Compare results failed at step = %d, errors = %d\n", k, error); break;}
			#endif	
		}
		#ifndef COMPARE
		run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;             
		std::cout<<"cpu cl 16 cores time is "<<run_time<<std::endl;
		#else
		if (!error)
		printf("16 Test OK!\n");
		#endif
		



		

		fclose(f_p);
		fclose(f_cl1);

	}
	catch (cl::Error err) {
		std::cout << "Exception\n";
		std::cerr
		<< "ERROR: "
		<< err.what()
		<< "("
		<< err_code(err.err())
		<< ")"
		<< std::endl;
	}       



	return 0;
}




void print_coords(FILE *f_p, float x[][N_d], float y[][N_d], float t[][N_d])
{
	int i,j;

	for (i=0; i<13; i++) {
		for (j=0; j<N_d; j++)
		fprintf(f_p,"%.2f\t  ", x[i][j]);

		for (j=0; j<N_d; j++)
		fprintf(f_p,"%.2f\t  ", y[i][j]);

		for (j=0; j<N_d; j++)
		fprintf(f_p,"%.2f\t  ", t[i][j]);
	}

	fprintf(f_p,"\n");

}

void print_coords(FILE* fp, std::vector<float>& ar1, std::vector<float>& ar2, std::vector<float>& ar3){
	for (int i=0; i<13; i++) {
		for (int j=0; j<N_d; j++)
		fprintf(fp,"%.2f\t  ", ar1[i*N_d + j]);

		for (int j=0; j<N_d; j++)
		fprintf(fp,"%.2f\t  ", ar2[i*N_d + j]);

		for (int j=0; j<N_d; j++)
		fprintf(fp,"%.2f\t  ", ar3[i*N_d + j]);
	}
	
	fprintf(fp,"\n");

}






void init_coords(float x[][N_d], float y[][N_d], float t[][N_d])
{
	int i,j;


	// caaaiea y eii?aeiaou aey ie?iae nie?aee
	for (i=0; i<13; i++)
	y[i][0] = 2.0f*6/13*(i+1);


	// caaaiea y eii?aeiao aey inoaeuiuo iieaeoe ai iieiaeiu auniou o?oai?ee
	for (j=1; j<N_d-4; j++)
	for (i=0; i<13; i++)
	y[i][j] = y[i][j-1] + 2.0f*Rad;


	// caaaiea x e teta eii?aeiao oae ?oiau aue oeeeia? ai iieiaeiu auniou o?oai?ee
	for (j=0; j<N_d-5; j++)
	for (i=0; i<13; i++)  {

		x[i][j] = 0.0;
		t[i][j] = 0.0;
	}


	//
	for (i=0; i<13; i++)  {

		x[i][N_d-5] = 0.6;
		t[i][N_d-5] = 0.2;

	}


	for (j=N_d-4; j<N_d; j++)
	for (i=0; i<13; i++)  {

		x[i][j] = x[i][j-1] + 2*Rad*sinf(t[i][j-1]);
		y[i][j] = y[i][j-1] + 2*Rad*cosf(t[i][j-1]);
		t[i][j] = t[i][j-1];

	}

}


void init_coords(std::vector<float> & x, std::vector<float> & y, std::vector<float> & t)
{
	float x_temp[13][N_d];
	float y_temp[13][N_d];
	float t_temp[13][N_d];
	for (int i =0; i < 13; i++){
		for (int j = 0; j <N_d; j++){
			x_temp[i][j]=x[N_d*i+j];
			y_temp[i][j]=y[N_d*i+j];
			t_temp[i][j]=t[N_d*i+j];
		}
	} 
	init_coords(x_temp, y_temp,t_temp);

	for (int i =0; i < 13; i++){
		for (int j = 0; j <N_d; j++){
			x[N_d*i+j]=x_temp[i][j];
			y[N_d*i+j]=y_temp[i][j];
			t[N_d*i+j]=t_temp[i][j];
		}
	}      

}






int compare_results(float x_1[][N_d], float y_1[][N_d], float t_1[][N_d],
float x_2[][N_d], float y_2[][N_d], float t_2[][N_d] )
{

	int error = 0;
	for(int i = 0; i<13; i++)
	for(int j = 0; j<N_d; j++) {

		if (fabs(x_1[i][j] - x_2[i][j]) > COMP_LIMIT)
		error++;
		if (fabs(y_1[i][j] - y_2[i][j]) > COMP_LIMIT)
		error++;
		if (fabs(t_1[i][j] - t_2[i][j]) > COMP_LIMIT)
		error++;

	}


	return error;
}
int compare_results(float x_1[][N_d], float y_1[][N_d], float t_1[][N_d],
std::vector<float>& ar1, std::vector<float>& ar2, std::vector<float>& ar3 )
{

	int error = 0;
	for(int i = 0; i<13; i++)
	for(int j = 0; j<N_d; j++) {

		if (fabs(x_1[i][j] - ar1[N_d*i+j]) > COMP_LIMIT)
		{error++; std::cout<<"i is "<<i<<" j is "<<j<<" x_1 is "<<x_1[i][j]<<" ar1 "<<ar1[N_d*i+j]<<std::endl;}
		if (fabs(y_1[i][j] - ar2[N_d*i+j]) > COMP_LIMIT)
		{error++; std::cout<<"i is "<<i<<" j is "<<j<<" y_1 is "<<y_1[i][j]<<" ar2 "<<ar2[N_d*i+j]<<std::endl;}
		if (fabs(t_1[i][j] - ar3[N_d*i+j]) > COMP_LIMIT)
		{error++; std::cout<<"i is "<<i<<" j is "<<j<<" t_1 is "<<t_1[i][j]<<" ar3 "<<ar3[N_d*i+j]<<std::endl;}

	}


	return error;
}


int compare_results(std::vector<float>& ar1, std::vector<float>& ar2, std::vector<float>& ar3,
std::vector<float>& ar4, std::vector<float>& ar5, std::vector<float>& ar6 )
{

	int error = 0;
	for(int i = 0; i<13; i++)
	for(int j = 0; j<N_d; j++) {

		if (fabs(ar1[N_d*i+j] - ar4[N_d*i+j]) > 0.001f)
		error++;
		if (fabs(ar2[N_d*i+j] - ar5[N_d*i+j]) > 0.001f)
		error++;
		if (fabs(ar3[N_d*i+j] - ar5[N_d*i+j]) > 0.001f)
		error++;

	}


	return error;
}



// ooieoey, eioi?ay eniieuciaaeanu aey au?eneaiey ianneaia eiinoaio a mt_defines.h
void calculate_constants()
{

	int num_PF, i;
	float gamma = 0;

	float Ax_1[13];
	float Ax_2[13];
	float Ax_3[13];
	float A_Bx_4[13];
	float Ay_1[13];
	float Ay_2[13];
	float Ay_3[13];

	float A_By_4[13];
	float Az_1, Az_2;

	float Bx_1[13];
	float Bx_2[13];
	float Bx_3[13];
	float By_1[13];
	float By_2[13];
	float By_3[13];
	float Bz_1, Bz_2;


	float C_1_r = rad_mon*sin(psi_r)*cos(fi_r);
	float C_2_r = rad_mon*sin(psi_r)*sin(fi_r);
	float C_3_r = rad_mon*cos(psi_r);


	float C_1_L = rad_mon*sin(psi_l)*cos(fi_l);
	float C_2_L = rad_mon*sin(psi_l)*sin(fi_l);
	float C_3_L = rad_mon*cos(psi_l);


	Az_1 = C_1_r;
	Az_2 = C_3_r;

	Bz_1 = C_1_L;
	Bz_2 = C_3_L;


	for (num_PF=1; num_PF<=13; num_PF++) {
		if( num_PF == 1) gamma = (2*pi/13*(num_PF-23));
		else if ( num_PF == 2)  gamma = (2*pi/13*(num_PF-25));
		else if ( num_PF == 3)  gamma = (2*pi/13*(num_PF-1));
		else if ( num_PF == 4)  gamma = (2*pi/13*(num_PF-3));
		else if ( num_PF == 5)  gamma = (2*pi/13*(num_PF-5));
		else if ( num_PF == 6)  gamma = (2*pi/13*(num_PF-7));
		else if ( num_PF == 7)  gamma = (2*pi/13*(num_PF-9));
		else if ( num_PF == 8)  gamma = (2*pi/13*(num_PF-11));
		else if ( num_PF == 9)  gamma = (2*pi/13*(num_PF-13));
		else if ( num_PF == 10) gamma = (2*pi/13*(num_PF-15));
		else if ( num_PF == 11) gamma = (2*pi/13*(num_PF-17));
		else if ( num_PF == 12) gamma = (2*pi/13*(num_PF-19));
		else if ( num_PF == 13) gamma = (2*pi/13*(num_PF-21));


		i = num_PF - 1;

		Ax_1[i] = C_1_r*cos(gamma);
		Ax_2[i] = C_2_r*sin(gamma);
		Ax_3[i] = C_3_r*cos(gamma);
		A_Bx_4[i] = sin(2*pi/13*(num_PF-1));

		Ay_1[i] = C_1_r*sin(gamma);
		Ay_2[i] = C_3_r*sin(gamma);
		Ay_3[i] = C_2_r*cos(gamma);
		A_By_4[i] = cos(2*pi/13*(num_PF-1));

		Bx_1[i] = C_1_L*cos(gamma);
		Bx_2[i] = C_2_L*sin(gamma);
		Bx_3[i] = C_3_L*cos(gamma);

		By_1[i] = C_1_L*sin(gamma);
		By_2[i] = C_3_L*sin(gamma);
		By_3[i] = C_2_L*cos(gamma);


	}


	printf("done!\n");






}


void init_cl(cl::Context &context, cl::CommandQueue &queue, cl::Program &program){
	cl_uint deviceIndex = 0;

	// Get list of devices
	std::vector<cl::Device> devices;
	unsigned numDevices = getDeviceList(devices);
	cl::Device device = devices[deviceIndex];

	//   std::string name;
	//     getDeviceName(device, name);
	//       std::cout << "\nUsing OpenCL device: " << name << "\n";

	std::vector<cl::Device> chosen_device;
	chosen_device.push_back(device);
	context= cl::Context(chosen_device);
	queue = cl::CommandQueue(context, device);

	
}



