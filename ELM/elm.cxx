/*
 * 1D Braginksii model for plasma density, energy and momentum with
 * convolution conductivity using trapezium rule integration
 */

#include <bout/physicsmodel.hxx>
#include <bout/constants.hxx>
#include <derivs.hxx>
#include <field_factory.hxx>
#include <interpolation.hxx>

#include <future>
#include <vector>
#include <math.h>

#include "/usr/local/include/xtensor/xarray.hpp"
#include "/usr/local/include/xtensor/xio.hpp"
#include "/usr/local/include/xtensor/xnpy.hpp"

BoutReal TrapeziumIntegrate(Field3D f, int i1, int i2, BoutReal dx){
  BoutReal result = 0.5*(f(0,i1,0) + f(0,i2,0));

  if (i1 == i2){
    return 0.0;
  }

  else if (i1 > i2){
    for (int i = i2+1; i < i1; ++i){
      result += f(0,i,0);
  }
  result *= -dx;
  }

  else {
    for (int i = i1+1; i < i2; ++i){
      result += f(0,i,0);
  }
  result *= dx;
  }

  return result;
}

BoutReal TrapeziumIntegrate(std::vector<BoutReal> f, int i1, int i2, BoutReal dx){
  BoutReal result = 0.;

  if (i1 > i2){
    result = 0.5*(f[i1] + f[i2]);
    for (int i = i2+1; i < i1; ++i){
      result += f[i];
    }
    result *= -dx;
  }

  else if (i2 > i1) {
    result = 0.5*(f[i1] + f[i2]);
    for (int i = i1+1; i < i2; ++i){
      result += f[i];
    }
    result *= dx;
  }

  return result;
}

typedef int HEAT_TYPE;
enum{SPITZER_HARM=0, LIMTED=1, CONVOLUTION=2};

typedef int PULSE;
enum{NONE=0, STEP=1, ELM=2};

class SOLNL : public PhysicsModel {
private:

  // Evolving variables
  Field3D T, n, v;

  // Derived quantities
  Field3D p, q, p_dyn, qSH, qFS, v_centre;

  Field3D A,B,C, gamma;

  // Source terms
  Field3D ypos, Sn_bg, Sn_pl, Sn_elm, S_n, S_u, Su;

  // Constants
  BoutReal kappa_0, q_in, length;
  BoutReal m_i = SI::Mp;
  BoutReal m_e = SI::Me;

  // Plasma parameters
  Field3D logLambda, lambda_0, lambda;

  // staggered n & T
  Field3D n_stag, T_stag;

  //Normalisations
  BoutReal n_t, c_st, T_t;

  // Convolution kernel
  Field3D w = 0.0;

  // Time derivatives
  Field3D ddt_T, ddt_n, ddt_v;

  // Number of non boundary grid points
  int N = mesh->yend-mesh->ystart+1;

  HEAT_TYPE heat_type;
  PULSE pulse;

protected:
  // This is called once at the start
  int init(bool restarting) override {

    Options *options = Options::getRoot()->getSection("SHConduction");

    OPTION(options, kappa_0, 1.0);
    OPTION(options, q_in, 1.0);
    OPTION(options, T_t, 1.0);
    OPTION(options, n_t, 1.0);
    OPTION(options, heat_type, 0);
	OPTION(options, pulse, 0);

    c_st = sqrt(2*SI::qe*T_t/m_i);

    OPTION(Options::getRoot()->getSection("mesh"), length, 1.0);

    FieldFactory f(mesh);
    Sn_bg = f.create3D("n:Sn_bg");
	Sn_pl = f.create3D("n:Sn_pl");
	Sn_elm = f.create3D("n:Sn_elm");

    v.setLocation(CELL_YLOW); // Stagger
    qSH.setLocation(CELL_YLOW); // Stagger
    q.setLocation(CELL_YLOW); // Stagger
	p.mergeYupYdown();

    ypos = f.create3D("mesh:ypos");
    ypos.applyBoundary("lower_target", "free_o3");
    ypos.applyBoundary("upper_target", "free_o3");
    ypos.mergeYupYdown();

	// Target heat condition
	q_in = -5.5 * T_t * n_t * c_st * SI::qe;
	Su = -2*q_in/length;
	//Su = -q_in * exp(-pow((ypos-0.5*length),2)/200)/sqrt(M_PI*200);

    SOLVE_FOR3(T, n, v);

    SAVE_REPEAT5(q, qSH, qFS, p, p_dyn);
    SAVE_REPEAT3(ddt_n, ddt_T, ddt_v)
    SAVE_REPEAT3(v_centre, S_n, S_u);
	SAVE_REPEAT2(T_stag, n_stag);
    SAVE_REPEAT3(lambda, logLambda, lambda_0);

    SAVE_ONCE4(kappa_0, q_in, T_t, length);
    SAVE_ONCE5(Sn_bg, Sn_pl, n_t, c_st, Su);
	SAVE_ONCE(Sn_elm);
    SAVE_ONCE3(ypos, heat_type, pulse);
	SAVE_ONCE(N);

    return 0;
  }

  xt::xarray<double> FieldToArray (const Field3D F) const{
     xt::xarray<double> X = xt::zeros<double>({mesh->LocalNy});
     for (int k = 0; k < mesh->LocalNy; ++k) {
       X(k) =  F(0,k,0);
     }
   return X;
 }

  /* Weird stuff happens in the Field3D constructor
     that breaks the threaded integrals so just pull
     the 1D data we need into a vector */
  std::vector<BoutReal> field_to_vector(Field3D F) const{
    std::vector<BoutReal> arr(mesh->LocalNy);
    for (int j = 0; j < mesh->LocalNy; ++j) {arr[j] = F(0,j,0);}
    return arr;
  }


  /* Avoid having shared copies of the PhysicsModel
     between threads by pulling the important stuff
     into a closure */
  std::function<double(int,int)>  make_kernel(CELL_LOC loc) const{

    std::vector<BoutReal> n_arr = field_to_vector(interp_to(n*n_t, loc));
    std::vector<BoutReal> T_arr = field_to_vector(interp_to(T, loc));
	std::vector<BoutReal> lambda_arr = field_to_vector(interp_to(lambda, loc));

    return [=](int i1, int i2){
    return exp(-abs(TrapeziumIntegrate(n_arr, i2, i1, length/N)) / (lambda_arr[i2]*n_arr[i2])) / (2*lambda_arr[i2]);};
  }

  Field3D heat_convolution(Field3D qSH, CELL_LOC loc) const{
    /*
      Calculate the convolution at each grid point
    */

    // Lots of closure magic to make this thread safe
    std::function<double(int,int)> kernel = make_kernel(qSH.getLocation());
    std::vector<BoutReal> q_arr = field_to_vector(qSH);

	xt::xarray<double> X = xt::zeros<double>({N+1,N+1}) - 1;

    auto parallel_core = [&](int j) {
      std::vector<BoutReal> F(N+1, 0);
      for (int k = mesh->ystart; k < mesh->yend+2; ++k) {
          F[k-mesh->ystart] = q_arr[k] * kernel(j, k);
		  X(j-mesh->ystart, k-mesh->ystart) = kernel(j, k);
      }
      return TrapeziumIntegrate(F, 0, N, length/N);
    };

    // Launch the futures
    Field3D heat = 0.0;
    heat.setLocation(CELL_YLOW);

    std::vector<std::future<BoutReal>> futures(N+1);
    for (int j = mesh->ystart; j < mesh->yend+2; ++j) {
      futures[j- mesh->ystart] = std::async(std::launch::async, std::bind(parallel_core, j));
    }

    // Collect the futures
    for (int j = mesh->ystart; j < mesh->yend+2; ++j) {
      heat(0,j,0) = futures[j-mesh->ystart].get();
    }

	//xt::dump_npy("/Users/HannahWhitham/Project/MSciProject/ELM/qSH.npy", FieldToArray(qSH));
	//xt::dump_npy("/Users/HannahWhitham/Project/MSciProject/ELM/q.npy", FieldToArray(heat));
	xt::dump_npy("/Users/HannahWhitham/Project/MSciProject/ELM/kernel.npy", X);

    // These extrapolated boundaries are technically invalid,
    // but necessary to use interp_to to shift ylow <-> ycenter
    // maybe the sheath BC needs to be applied here????
    //heat.applyBoundary("upper_target", "free_o3");
    //heat.applyBoundary("lower_target", "free_o3");
    //heat.mergeYupYdown();

    return interp_to(heat, loc);
  }

  int rhs(BoutReal t) override {
    mesh->communicate(n,v,T);

    n(0,0,0) = (3*n(0,1,0) - n(0,2,0))/2.;
    n(0,N+3,0) = (3*n(0,N+2,0) - n(0,N+1,0))/2.;
    n = floor(n, 1e-4);

    v(0,0,0) = (3*v(0,1,0) - v(0,2,0)) / 2.;
    v(0,N+3,0) = (3*v(0,N+2,0) - v(0,N+1,0)) / 2.;

    // This cell doesn't get used, as we take only C2 derivatives
    // But the extrapolation can sometimes make it negative
    // which causes a problem with the T^2.5 term.
    //T(0,0,0) = T_t; T(0,N+3,0) = T_t;
    T = floor(T, 0.);

    // Need to calculate the value of q one cell into the right boundary
    // because this cell is ON the boundary now as a result of being staggered
    // so we can use the T BC to get a well defined value here
    // this is then used later to calculate (DDY(q))_{N-1}
    qSH = DDY(T, CELL_YLOW, DIFF_C2);
    qSH(0,N+2,0) = (T(0,N+2,0)-T(0,N+1,0))/mesh->coordinates()->dy(0,0,0);
    qSH *= -kappa_0 * pow(interp_to(T, CELL_YLOW), 2.5);

    // For tidiness, doesn't actually affect anything
    qSH(0,0,0) = 0; qSH(0,1,0) = 0;  qSH(0,N+3,0) = 0;

	// Plasma parameter functions
	logLambda = 15.2 - 0.5*log(n * (n_t/1e20)) + log((T/1000));
	lambda_0 = (2.5e17  * T * T) / (n * n_t * logLambda);
	lambda = 32 * sqrt(2) * lambda_0;

    // Free streaming heat flow
    qFS = 0.03 * interp_to(n, CELL_YLOW) * n_t * interp_to(T, CELL_YLOW) * SI::qe * pow((2*interp_to(T, CELL_YLOW)*SI::qe/m_e),1.5);

    switch(heat_type){
        case SPITZER_HARM :
            q = qSH;
        break;

        case LIMTED :
            q = ((qSH * qFS) / (qSH + qFS));
        break;

        case CONVOLUTION :
			q = -heat_convolution(qSH, CELL_YLOW);
        break;
    }

	// Sheath heat condition DO NOT TOUCH THEM THEY ARE CORRECT!!!!!
	T_stag = interp_to(T, CELL_YLOW);
	n_stag = interp_to(n, CELL_YLOW);
	BoutReal qt_low = -5.5 * T_stag(0,2,0) * n_stag(0,2,0)*  n_t * sqrt(2*SI::qe*T_stag(0,2,0)/m_i) * SI::qe;
	BoutReal qt_upr = 5.5 * T_stag(0,N+2,0) * n_stag(0,N+2,0) * n_t * sqrt(2*SI::qe*T_stag(0,N+2,0)/m_i) * SI::qe;
	q.applyBoundary("lower_target", "dirichlet(" + std::to_string(qt_low) + ")");
	q.applyBoundary("upper_target", "dirichlet(" + std::to_string(qt_upr) + ")");

	switch(pulse){
		case NONE :
			S_n = Sn_bg;
			S_u = Su;
		break;

		case STEP :
			if (t >= 6.0e-3 && t<= 6.0e-3 + 1.0e-6) {
				S_n = Sn_bg + 10*Sn_pl; // particle pulse for 1 microsec
			}
			else {
				S_n = Sn_bg;
			}
			S_u = Su;
		break;

		case ELM :
			if (t >= 4.0e-3 && t <= 4.0e-3 + 2e-4) {
				S_n = (0.5*Sn_elm)+Sn_bg;
				S_u = 3.3e7*Sn_elm;
			}
			else {
				S_n = Sn_bg;
				S_u = Su;
			}
		break;
	}

    // Fluid pressure
    p = 2*(n_t*n)*SI::qe*T;
    p_dyn = m_i * (n_t*n) * (c_st*v) * (c_st*v);

    // Fluid equations
    ddt(n) = c_st * (S_n - FDDY(v, n, CELL_CENTRE));
    ddt(v) = (-DDY(p, CELL_YLOW)) / (m_i*n*n_t*c_st) - c_st * (2 * VDDY(v, v, CELL_YLOW)  +  v * (VDDY(v, n, CELL_YLOW)/n));
    n.applyTDerivBoundary();

    ddt(T) = (1 / (3 * n_t * n * SI::qe)) * (S_u - DDY(q, CELL_CENTRE, DIFF_C2) + VDDY(v,p, CELL_CENTRE)) + (T/n) * ddt(n);

    v_centre=interp_to(v, CELL_CENTRE);

    ddt_T = ddt(T);
    ddt_n = ddt(n);
    ddt_v = ddt(v);

    return 0;
  }
};

BOUTMAIN(SOLNL);
