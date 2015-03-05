// ioc64 -cmd=build -input=prog2.cl -device=cpu -ir=prog2.bc to create executable
#define N_d 16
#define viscPF       (4.3e-7*2) 
#define viscPF_teta  (2.3e-6*2) 
#define B_Koeff 174.0f 
#define dt          (2e-10) 
#define dt_viscPF_teta          4.34782596e-5f
#define dt_viscPF                       0.000232558144f 
#define R_MT 8.128f
#define A_Koeff 53.0f
#define b_lat   12.9f
#define A_long_D 90.0f
#define b_long_D 7.9f
#define A_long_T 90.0f
#define b_long_T 7.9f
#define ro0       0.12f
#define ro0_long  0.12f
#define inv_ro0_long  8.3333f
#define c_lat  0.10f
#define d_lat  0.25f
#define C_Koeff 300.0f
#define Rad       2.0f
#define inv_ro0                         8.3333f
#define clat_dlat_ro0           6.6666f
#define clong_dlong_ro0         6.6666f
#define         d_lat_ro0                       33.3333f
#define         d_long_ro0                      33.3333f
#define pi 3.141592653f
#define fi_r            1.3291395f
#define psi_r           1.801785f
#define fi_l            1.0856f
#define psi_l           -1.339725f
#define rad_mon          2.0f
#define teta0_D 0.2f 
#define teta0_T 0.0f

void calc_grad_c(       __private int,          __private int,  __private int , __private int , __private char ,        __private char, __local  float*,        __local float*, __local  float*,        
__local         float*, __local         float*, __local         float*, __local         float*,__local  float*, __local         float*, __local         float*, __local         float*, __local         float*, __local         float*, __local         float*, __local         float*);


__kernel void cl_calc( 
const int          niters,                                                                   
__global float*    x_inout,                          
__global float*    y_inout,
__global float*    t_inout

) {
	int k,p;
	int i = get_global_id(0);
	int j;
	//int j = get_global_id(1);
	char pos;
	int i2;
	int j2;
	char type = 0;
	float f_x, f_y, f_t;


	__local float x_loc[13*(N_d+3)];
	__local float y_loc[13*(N_d+3)];
	__local float t_loc[13*(N_d+3)];
	__local      float     lat_l_x[13*(N_d+1)];
	__local      float     lat_l_y[13*(N_d+1)];
	__local  float     lat_l_t[13*(N_d+1)];
	__local      float     lat_r_x[13*(N_d+3)];
	__local      float     lat_r_y[13*(N_d+3)];
	__local  float     lat_r_t[13*(N_d+3)];
	__local      float     long_u_x[13*(N_d+1)];
	__local      float     long_u_y[13*(N_d+1)];
	__local      float     long_u_t[13*(N_d+1)];
	__local      float     long_d_x[13*(N_d+1)];
	__local      float     long_d_y[13*(N_d+1)];
	__local      float     long_d_t[13*(N_d+1)];




	if ((i < 13)){
		for (j = 0; j< N_d; j++){
			x_loc[i*(N_d+3) + j] = x_inout[i*N_d + j];
			y_loc[i*(N_d+3) + j] = y_inout[i*N_d + j];
			t_loc[i*(N_d+3) + j] = t_inout[i*N_d + j];
		}
	}       



	for (k=0; k<niters; k++){
		barrier( CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);
		if ((i < 13)){
			for (j = 0; j< N_d; j++){

				pos =convert_char(j % 2);
				i2 = (i==12)? 0 : (i+1);
				j2 = (i==12)? (j+3) : j;

				calc_grad_c(i, j, i2, j2, type,  pos,

				x_loc,  y_loc, t_loc,  
				
				lat_l_x, lat_l_y, lat_l_t,
				lat_r_x,lat_r_y, lat_r_t,

				long_u_x, long_u_y, long_u_t,
				long_d_x, long_d_y, long_d_t );


			}
		}

		barrier( CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);
		
		if ((i < 13)){
			for (j = 1; j< N_d; j++){
				f_x = lat_l_x[i*(N_d+1) + j] + lat_r_x[i*(N_d+3) + j] + long_u_x[i*(N_d+1) + j] + long_d_x[i*(N_d+1) + j];

				f_y = lat_l_y[i*(N_d+1) + j] + lat_r_y[i*(N_d+3) + j] + long_u_y[i*(N_d+1) + j] + long_d_y[i*(N_d+1) + j];

				f_t = lat_l_t[i*(N_d+1) + j] + lat_r_t[i*(N_d+3) + j] + long_u_t[i*(N_d+1) + j] + long_d_t[i*(N_d+1) + j];


				x_loc[i*(N_d+3) + j] -= dt_viscPF * f_x;
				y_loc[i*(N_d+3) + j]   -= dt_viscPF * f_y;
				t_loc[i*(N_d+3) + j]   -= dt_viscPF_teta * f_t;

			}

		}
	}

	barrier( CLK_LOCAL_MEM_FENCE| CLK_GLOBAL_MEM_FENCE);
	if ((i < 13)){
		for (j = 1; j< N_d; j++){
			x_inout[i*N_d + j] = x_loc[i*(N_d+3) + j];
			y_inout[i*N_d + j] = y_loc[i*(N_d+3) + j];
			t_inout[i*N_d + j] = t_loc[i*(N_d+3) + j];
		}
	}       


  
}   


void calc_grad_c(       __private       int i1,                 // i index i?aaie iieaeoeu
__private int j1,                       // j index i?aaie iieaeoeu

__private int i2,                       // i index eaaie iieaeoeu
__private int j2,                       // j index eaaie iieaeoeu
__private char type,            // dimer type: 0 - 'D', 1 - 'T'
__private char pos,             // monomer position in dimer: 0 - bottom, 1 - top

__local  float* x_in,
__local  float* y_in,
__local  float* t_in,

__local      float*     lat_l_x,
__local      float*     lat_l_y,
__local    float*     lat_l_t,
__local      float*     lat_r_x,
__local      float*     lat_r_y,
__local    float*     lat_r_t,
__local      float*     long_u_x,
__local      float*     long_u_y,
__local      float*     long_u_t,
__local      float*     long_d_x,
__local      float*     long_d_y,
__local      float*     long_d_t

)


{
	
	__constant float Ax_1[13] = {-0.165214628f, 0.0561592989f, 0.264667839f, 0.412544012f, 0.465911359f, 0.412544012f, 0.264667839f, 0.0561594106f, -0.165214419f, -0.348739684f, -0.452372819f, -0.452372819f, -0.348739684f};
	__constant float Ax_2[13] = {1.76747036f, 1.87652779f, 1.5556947f, 0.878470898f, 0.0f, -0.878470898f, -1.5556947f, -1.87652767f, -1.76747072f, -1.25350749f, -0.452380866f, 0.452380747f, 1.25350738f};
	__constant float Ax_3[13] = {0.162366703f, -0.0551912338f, -0.26010555f, -0.405432671f, -0.45788008f, -0.405432671f, -0.26010555f, -0.0551913455f, 0.162366495f, 0.342728168f, 0.444574922f, 0.444574922f, 0.342728198f};
	__constant float A_Bx_4[13] = {0.0f, 0.46472317f, 0.822983861f, 0.992708862f, 0.935016215f, 0.663122654f, 0.239315659f, -0.239315659f, -0.663122654f, -0.935016215f, -0.992708862f, -0.822983861f, -0.46472317f};
	__constant float Ay_1[13] = {0.435634613f, 0.462514341f, 0.383437514f, 0.216519803f, 0.0f, -0.216519803f, -0.383437514f, -0.462514341f, -0.435634702f, -0.308956355f, -0.111499891f, 0.111499861f, 0.308956355f};
	__constant float Ay_2[13] = {-0.428125232f, -0.454541624f, -0.376827925f, -0.212787479f, -0.0f, 0.212787479f, 0.376827925f, 0.454541624f, 0.428125322f, 0.30363065f, 0.109577879f, -0.10957785f, -0.30363062f};
	__constant float Ay_3[13] = {-0.670314014f, 0.227851257f, 1.07381856f, 1.67378652f, 1.89031017f, 1.67378652f, 1.07381856f, 0.227851719f, -0.67031312f, -1.41491747f, -1.83538115f, -1.83538127f, -1.41491759f};
	__constant float A_By_4[13] = {1.0f, 0.885456026f, 0.568064749f, 0.120536678f, -0.3546049f, -0.748510778f, -0.970941842f, -0.970941842f, -0.748510778f, -0.3546049f, 0.120536678f, 0.568064749f, 0.885456026f};
	__constant float Az_1 = 0.465911359f;
	__constant float Az_2 = -0.45788008f;
	__constant float Bx_1[13] = {0.321971923f, -0.109443799f, -0.515787303f, -0.80396992f, -0.907972693f, -0.80396992f, -0.515787303f, -0.109444022f, 0.321971506f, 0.679627359f, 0.881588638f, 0.881588697f, 0.679627359f};
	__constant float Bx_2[13] = {-1.61023343f, -1.70958889f, -1.41729772f, -0.800320745f, -0.0f, 0.800320745f, 1.41729772f, 1.70958877f, 1.61023378f, 1.14199352f, 0.412136346f, -0.412136227f, -1.1419934f};
	__constant float Bx_3[13] = {-0.16242376f, 0.0552106313f, 0.260196954f, 0.405575156f, 0.458040982f, 0.405575156f, 0.260196954f, 0.0552107431f, -0.162423551f, -0.342848599f, -0.444731146f, -0.444731146f, -0.342848629f};
	__constant float By_1[13] = {-0.848969102f, -0.901352584f, -0.747246861f, -0.421955943f, -0.0f, 0.421955943f, 0.747246861f, 0.901352525f, 0.848969221f, 0.602097273f, 0.2172921f, -0.217292041f, -0.602097213f};
	__constant float By_2[13] = {0.428275675f, 0.454701364f, 0.376960337f, 0.212862253f, 0.0f, -0.212862253f, -0.376960337f, -0.454701334f, -0.428275764f, -0.303737342f, -0.109616384f, 0.109616362f, 0.303737313f};
	__constant float By_3[13] = {0.610681832f, -0.207581252f, -0.978290021f, -1.52488387f, -1.7221452f, -1.52488387f, -0.978290021f, -0.207581669f, 0.610681057f, 1.28904426f, 1.67210281f, 1.67210281f, 1.28904426f};
	__constant float Bz_1 = -0.907972693f;
	__constant float Bz_2 = 0.458040982f;
	

	float x_1 =  x_in[i1*(N_d+3) + j1];
	float y_1 = y_in[i1*(N_d+3) + j1];
	float teta_1 = t_in[i1*(N_d+3) + j1];
	float x_2 = x_in[i2*(N_d+3) + j2];              
	float y_2 = y_in[i2*(N_d+3) + j2];
	float teta_2 =t_in[i2*(N_d+3) + j2] ;
	float x_3 = x_in[i1*(N_d+3) + j1+1];            
	float y_3 = y_in[i1*(N_d+3) + j1+1];
	float teta_3 = t_in[i1*(N_d+3) + j1+1];
	float cos_t_A = cos(teta_2);
	float sin_t_A = sin(teta_2);
	float cos_t_B = cos(teta_1);
	float sin_t_B = sin(teta_1);
	float cos_t_1 = cos_t_B;
	float sin_t_1 = sin_t_B;
	float cos_t_3 = cos(teta_3);
	float sin_t_3 = sin(teta_3);
	float Ax_left = Ax_1[i2]*cos_t_A + Ax_3[i2]*sin_t_A - Ax_2[i2] +
	(x_2 + R_MT) * A_Bx_4[i2];
	float Ay_left = Ay_1[i2]*cos_t_A + Ay_2[i2]*sin_t_A + Ay_3[i2] +
	(x_2 + R_MT) * A_By_4[i2];
	float Az_left = -Az_1*sin_t_A + Az_2*cos_t_A + y_2;
	float Bx_right = Bx_1[i1]*cos_t_B + Bx_3[i1]*sin_t_B - Bx_2[i1] +
	(x_1 + R_MT) * A_Bx_4[i1];
	float By_right = By_1[i1]*cos_t_B + By_2[i1]*sin_t_B + By_3[i1] +
	(x_1 + R_MT) * A_By_4[i1];
	float Bz_right = -Bz_1*sin_t_B + Bz_2*cos_t_B + y_1;
	float Dx = Ax_left - Bx_right;
	float Dy = Ay_left - By_right;
	float Dz = Az_left - Bz_right;
	float dist = sqrt(( pow(Dx, 2) + pow(Dy, 2) + pow(Dz, 2) ));
	if (dist <=1e-7 ){
		dist = 1e-5;
	}
	float inv_dist = 1.0f/dist;
	float drdAx = Dx * inv_dist;
	float drdAy = Dy * inv_dist;
	float drdAz = Dz * inv_dist;
	float drdBx = -drdAx;
	float drdBy = -drdAy;
	float drdBz = -drdAz;

	float dA_X_dteta = -sin_t_A*Ax_1[i2] + cos_t_A*Ax_3[i2];
	float dA_Y_dteta = -sin_t_A*Ay_1[i2] + cos_t_A*Ay_2[i2];
	float dA_Z_dteta = -cos_t_A*Az_1 - sin_t_A*Az_2;

	float drdx_A = drdAx*A_Bx_4[i2] + drdAy*A_By_4[i2];
	float drdy_A = drdAz;
	float drdteta_A = drdAx*dA_X_dteta + drdAy*dA_Y_dteta + drdAz*dA_Z_dteta;

	//================================================
	float dB_X_dteta = -sin_t_B*Bx_1[i1] + cos_t_B*Bx_3[i1];
	float dB_Y_dteta = -sin_t_B*By_1[i1] + cos_t_B*By_2[i1];
	float dB_Z_dteta = -cos_t_B*Bz_1 - sin_t_B*Bz_2;

	float drdx_B = drdBx*A_Bx_4[i1] + drdBy*A_By_4[i1];
	float drdy_B = drdBz;
	float drdteta_B = drdBx*dB_X_dteta + drdBy*dB_Y_dteta + drdBz*dB_Z_dteta;


	float Grad_U_tmp = (b_lat* dist *exp(-dist*inv_ro0)*(2.0f - dist*inv_ro0) +
	dist* clat_dlat_ro0 * exp( - (dist*dist) * d_lat_ro0 )  ) * A_Koeff;

	if ((i1==12)&&(j1>=(N_d-3))) {

		lat_r_x[i2*(N_d+3) + j2] = 0.0f;
		lat_r_y[i2*(N_d+3) + j2] = 0.0f;
		lat_r_t[i2*(N_d+3) + j2] = 0.0f;

		lat_l_x[i1*(N_d+1) + j1] = 0.0f;
		lat_l_y[i1*(N_d+1)+ j1] = 0.0f;
		lat_l_t[i1*(N_d+1) + j1] = 0.0f;

	} else {

		lat_r_x[i2*(N_d+3) + j2] = Grad_U_tmp * drdx_A;
		lat_r_y[i2*(N_d+3)+ j2] = Grad_U_tmp * drdy_A;
		lat_r_t[i2*(N_d+3) + j2] = Grad_U_tmp * drdteta_A;

		lat_l_x[i1*(N_d+1) + j1] = Grad_U_tmp * drdx_B;
		lat_l_y[i1*(N_d+1) + j1] = Grad_U_tmp * drdy_B;
		lat_l_t[i1*(N_d+1) + j1] = Grad_U_tmp * drdteta_B;

	}



	//      [nd]    -       mol3
	//      [nd-1]  -       mol1


	// longitudinal gradient

	float r_long_x = (x_3 - x_1) - Rad*(sin_t_1 + sin_t_3);
	float r_long_y = (y_3 - y_1) - Rad*(cos_t_1 + cos_t_3);
	float r_long = sqrt( r_long_x*r_long_x + r_long_y*r_long_y);

	if (r_long <=1e-15 ){
		r_long = 1e-7;
	}

	float drdx_long = - r_long_x/r_long;
	float drdy_long = - r_long_y/r_long;

	float dUdr_C;

	if (pos==0) {           // bottom monomer (interaction inside dimer)
		dUdr_C = C_Koeff*r_long;
	} else {                        // top monomer (interaction with upper dimer)

		float tmp1 = r_long *  exp(-r_long*inv_ro0_long)*(2 - r_long*inv_ro0_long);
		float tmp2      = r_long * clong_dlong_ro0 * exp(-(r_long*r_long) * d_long_ro0 );

		if (type==0)    // dimer type 'D'
		dUdr_C = (tmp1*b_long_D + tmp2) * A_long_D;
		else                    // dimer type 'T'
		dUdr_C = (tmp1*b_long_T + tmp2) * A_long_T;
	}



	float Grad_tmp_x = drdx_long * dUdr_C;
	float Grad_tmp_y = drdy_long * dUdr_C;

	float GradU_C_teta_1 = -dUdr_C*( drdx_long*(-Rad*cos_t_1) + drdy_long*(Rad*sin_t_1));
	float GradU_C_teta_3 =  dUdr_C*(-drdx_long*(-Rad*cos_t_3) - drdy_long*(Rad*sin_t_3));

	float Grad_tmp;
	if (type==0)            // dimer type 'D'
	Grad_tmp = B_Koeff*(teta_3 - teta_1 - teta0_D);
	else                            // dimer type 'T'
	Grad_tmp = B_Koeff*(teta_3 - teta_1 - teta0_T);

	// iiiaiye ooo ciae - ana ca?aaioaei!
	float GradU_B_teta_1 = - Grad_tmp;
	float GradU_B_teta_3 = + Grad_tmp;


	if (j1 == (N_d-1)) {

		long_u_x[i1*(N_d+1)+j1]                 = 0.0f;
		long_u_y[i1*(N_d+1)+j1]                 = 0.0f;
		long_u_t[i1*(N_d+1)+j1] = 0.0f;

		long_d_x[i1*(N_d+1)+j1+1]               = 0.0f;
		long_d_y[i1*(N_d+1)+j1+1]               = 0.0f;
		long_d_t[i1*(N_d+1)+j1+1]       = 0.0f;

	} else {

		long_u_x[i1*(N_d+1)+j1]                 = Grad_tmp_x;
		long_u_y[i1*(N_d+1)+j1]                 = Grad_tmp_y;
		long_u_t[i1*(N_d+1)+j1] = GradU_C_teta_1 + GradU_B_teta_1;

		long_d_x[i1*(N_d+1)+j1+1]               = - Grad_tmp_x;
		long_d_y[i1*(N_d+1)+j1+1]               = - Grad_tmp_y;
		long_d_t[i1*(N_d+1)+j1+1]       = GradU_C_teta_3 + GradU_B_teta_3;


	}



}