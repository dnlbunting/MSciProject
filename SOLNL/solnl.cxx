
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

#include "/usr/local/include/xtensor/xarray.hpp"
#include "/usr/local/include/xtensor/xio.hpp"
#include "/usr/local/include/xtensor/xnpy.hpp"

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

  Field3D T_stag, n_stag;

  // Derived quantities
  Field3D p, q, p_dyn, qSH, qFS, v_centre, v_th;//, kappa_0;
  BoutReal qt_upr, qt_low;
  Field3D col, qL, qR;

  Field3D A,B,C, gamma;

  // Source terms
  Field3D ypos, Sn_bg, Sn_pl, Sn_elm, S_n, S_u, Su;
  BoutReal Liz_l, Liz_u, L_iz, L_sn;

  // Constants
  BoutReal q_in, T_t, length, kappa_0;
  BoutReal m_i = SI::Mp;
  BoutReal m_e = SI::Me;
  BoutReal Su0, Sn0, gam;
  BoutReal alpha, beta;

  // Plasma parameters
  Field3D logLambda, lambda_0, lambda;

  //Normalisations
  BoutReal n_t, c_st;

  // Time derivatives
  Field3D ddt_T, ddt_n, ddt_v;

  // Number of non boundary grid points
  int N = mesh->yend-mesh->ystart+1;

  HEAT_TYPE heat_type;
  PULSE pulse;
  BoutReal cap;
  //xt::xarray<double> X;
  bool knorm;
  bool convection;
  bool save_kernel;
  bool delay;

protected:
  // This is called once at the start
  int init(bool restarting) override {

    Options *options = Options::getRoot()->getSection("SHConduction");

    OPTION(options, q_in, 1.0);
    OPTION(options, T_t, 1.0);
    OPTION(options, n_t, 1.0);
    OPTION(options, Su0, 1.0);
    OPTION(options, Sn0, 1.0);
	OPTION(options, gam, 1.0);
	OPTION(options, beta, 1.0);
    OPTION(options, heat_type, 0);
	OPTION(options, pulse, 0);
    OPTION(options, knorm, false);
	OPTION(options, convection, false);
	OPTION(options, save_kernel, false);
	OPTION(options, delay, false);
	OPTION(options, L_iz, 1.0);
	OPTION(options, L_sn, 1.0);
	OPTION(options, alpha, 1.0);
	OPTION(options, cap, 1.0);

    c_st = sqrt(5*SI::qe*T_t/(3*m_i));

    OPTION(Options::getRoot()->getSection("mesh"), length, 1.0);

	  // Target heat condition
	  q_in = -gam * T_t * n_t * c_st * SI::qe;
	  Su = -2*q_in/length;

    FieldFactory f(mesh);
    Sn_elm = f.create3D("n:Sn_elm");
	Sn_pl = f.create3D("n:Sn_pl");

    v.setLocation(CELL_YLOW); // Stagger
	v_th.setLocation(CELL_YLOW); // Stagger
    qSH.setLocation(CELL_YLOW); // Stagger
    q.setLocation(CELL_YLOW); // Stagger
	qFS.setLocation(CELL_YLOW); // Stagger
	//kappa_0.setLocation(CELL_YLOW); // Stagger
	lambda.setLocation(CELL_YLOW); // Stagger
	logLambda.setLocation(CELL_YLOW); // Stagger
	lambda_0.setLocation(CELL_YLOW); // Stagger

	col.setLocation(CELL_YLOW);
	qL.setLocation(CELL_YLOW);
	qR.setLocation(CELL_YLOW);

    ypos = f.create3D("mesh:ypos");
    ypos.applyBoundary("lower_target", "free_o3");
    ypos.applyBoundary("upper_target", "free_o3");
    ypos.mergeYupYdown();

    SOLVE_FOR3(T, n, v);

    SAVE_REPEAT5(q, qSH, qFS, p, p_dyn);
    SAVE_REPEAT3(ddt_n, ddt_T, ddt_v);
    SAVE_REPEAT3(v_centre, v_th, Sn_bg);
    SAVE_REPEAT3(lambda,lambda_0, logLambda);
	SAVE_REPEAT3(col, qL, qR);
    SAVE_ONCE4(kappa_0, q_in, T_t, length);
    SAVE_ONCE4(S_n, n_t, c_st, S_u);
    SAVE_ONCE2(ypos, heat_type);
	SAVE_ONCE3(alpha, beta, gam);

    return 0;
  }

  //int outputMonitor(BoutReal simtime, int iter, int NOUT) override{
  //	if (save_kernel){
  //  	if (iter == -1){
  //  	  X = xt::zeros<double>({NOUT+1, N+1 ,N+1}) * nan("");
  //  	}
  //
  //  	std::function<double(int,int)> kernel = make_kernel(qSH.getLocation());
  //  	std::vector<BoutReal> q_arr = field_to_vector(qSH);
  //  	for (int j = mesh->ystart; j < mesh->yend+2; ++j) {
  //  	  for (int k = mesh->ystart; k < mesh->yend+2; ++k) {
  //			  X(iter+1, j-mesh->ystart, k-mesh->ystart) = kernel(j, k);
  //			  std::vector<BoutReal> G(N+1, 1);
  //			  G[k-mesh->ystart] = kernel(j, k);
  //			  if (knorm){
  //			      X=X/TrapeziumIntegrate(G, 0, N, length/N);
  //			  }
  //
  //  	  }
  //  	}
  //
  //  	if (iter % 1000 == 0){
  //  	     xt::dump_npy("./kernel"+std::to_string(knorm)+std::to_string(alpha)+".npy", X);
  //  	}
  //	}
  //
  //  return 0;
  //}

  /* Weird stuff happens in the Field3D constructor
     that breaks the threaded integrals so just pull
     the 1D data we need into a vector */
  std::vector<BoutReal> field_to_vector(Field3D F) const{
    std::vector<BoutReal> arr(mesh->LocalNy);
    for (int j = 0; j < mesh->LocalNy; ++j) {arr[j] = F(0,j,0);}
    return arr;
  }
  xt::xarray<double> FieldToArray (const Field3D F) const{
     xt::xarray<double> X = xt::zeros<double>({mesh->LocalNy});
     for (int k = 0; k < mesh->LocalNy; ++k) {
       X(k) =  F(0,k,0);
     }
   return X;
 }

  /* Avoid having shared copies of the PhysicsModel
     between threads by pulling the important stuff
     into a closure */
  std::function<double(int,int)>  make_kernel(CELL_LOC loc) const{

      std::vector<BoutReal> n_arr = field_to_vector(n_stag);
      std::vector<BoutReal> T_arr = field_to_vector(T_stag);
      std::vector<BoutReal> lambda_arr = field_to_vector(lambda);

      if (knorm) {
          return [=](int i1, int i2){
            return exp(-abs(TrapeziumIntegrate(n_arr, i2, i1, length/N)) / (lambda_arr[i2]*n_arr[i2]));
          };
      }
      else{
        return [=](int i1, int i2){
          return exp(-abs(TrapeziumIntegrate(n_arr, i2, i1, length/N)) / (lambda_arr[i2]*n_arr[i2]))/(2*lambda_arr[i2]);
        };
      }
    }

    Field3D heat_convolution(Field3D qSH, CELL_LOC loc) const{
      /*
        Calculate the convolution at each grid point
      */

      // Lots of closure magic to make this thread safe
      std::function<double(int,int)> kernel = make_kernel(qSH.getLocation());
      std::vector<BoutReal> q_arr = field_to_vector(qSH);

	  xt::xarray<double> X = xt::zeros<double>({N+1,N+1}) - 1;
	  xt::xarray<double> Y = xt::zeros<double>({N+1});
      auto parallel_core = [&](int j) {
        std::vector<BoutReal> F(N+1, 0);
        std::vector<BoutReal> G(N+1, 1);
        for (int k = mesh->ystart; k < mesh->yend+2; ++k) {
            F[k-mesh->ystart] = q_arr[k] * kernel(j, k);
            X(j-mesh->ystart, k-mesh->ystart) = kernel(j, k);
            if (knorm){G[k-mesh->ystart] = kernel(j, k);}
		}

        BoutReal ret = TrapeziumIntegrate(F, 0, N, length/N);
        if (knorm) {ret = ret/TrapeziumIntegrate(G, 0, N, length/N);}

	    if (save_kernel){
	    	Y(j-mesh->ystart)=TrapeziumIntegrate(G, 0, N, length/N);
	    	xt::dump_npy("/Users/HannahWhitham/Documents/University/Y4/Project/MSciProject/SOLNL/kernel"+std::to_string(knorm)+std::to_string(alpha)+".npy", X);
	    	xt::dump_npy("/Users/HannahWhitham/Documents/University/Y4/Project/MSciProject/SOLNL/norm"+std::to_string(knorm)+std::to_string(alpha)+".npy", Y);
	    }
        return ret;
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

      return interp_to(heat, loc);
    }



  int rhs(BoutReal t) override {
    mesh->communicate(n,v,T);

    n = floor(n, 0);
    T = floor(T, 1);

    T_stag = floor(interp_to(T, CELL_YLOW), 1);
    n_stag = floor(interp_to(n, CELL_YLOW), 0);
	v_centre = interp_to(v, CELL_CENTRE);

	v_th = sqrt(2*T_stag*SI::qe/m_e);

    // Apply sheath sound speed boundary
    BoutReal c_su = sqrt(5*SI::qe*T_stag(0,2,0)/(3*m_i));
    BoutReal c_sl = -sqrt(5*SI::qe*T_stag(0,N+2,0)/(3*m_i));

    v.applyBoundary("lower_target", "dirichlet_o2(" + std::to_string(c_sl/c_st) + ")");
    v.applyBoundary("upper_target", "dirichlet_o2(" + std::to_string(c_su/c_st) + ")");

	Liz_l = 3e19/(n_stag(0,2,0)*n_t);
	Liz_u = 3e19/(n_stag(0,N+2,0)*n_t);

	Sn_bg = exp(-ypos/Liz_l)/Liz_l + exp(-(length-ypos)/Liz_u)/Liz_u;
	//Sn_bg = exp(-ypos/L_iz)/L_iz + exp(-(length-ypos)/L_iz)/L_iz;

	// Plasma parameter functions
	logLambda = 15.2 - 0.5*log(n_stag * (n_t/1e20)) + log(T_stag/1000);
	lambda_0 = (2.5e17/n_t) * T_stag * T_stag / (n_stag * logLambda);
	lambda = floor(alpha * sqrt(2) * lambda_0, 0.5);

	for (int j = mesh->ystart; j < mesh->yend+2; ++j) {
		if (lambda(0,j,0) >= cap){
			lambda(0,j,0) = cap;
		}
	}

	switch(pulse){
			case NONE :
				S_n = Sn_bg;
				S_u = Su;
			break;

			case STEP :
				if (t >= 1.0e-3 && t<= 1.0e-3 + 1.0e-6) {
					S_n = Sn_bg + 10*Sn_pl; // particle pulse for 1 microsec
				}
				else {
					S_n = Sn_bg;
				}
				S_u = Su;
			break;

			case ELM :
				if (t >= 1.0e-2 && t <= 1.0e-2 + 2e-4) {
					S_n = (Sn0*Sn_elm)+Sn_bg;
					S_u = Su0*Sn_elm+Su;
				}
				else {
					S_n = Sn_bg;
					S_u = Su;
				}
			break;
	}

	kappa_0 = 2000;//30692/logLambda;

    // Need to calculate the value of q one cell into the right boundary
    // because this cell is ON the boundary now as a result of being staggered
    // so we can use the T BC to get a well defined value here
    // this is then used later to calculate (DDY(q))_{N-1}
    qSH = DDY(T, CELL_YLOW, DIFF_C2);
    qSH(0,N+2,0) = (T(0,N+2,0)-T(0,N+1,0))/mesh->coordinates()->dy(0,0,0);
    qSH *= -kappa_0 * pow(T_stag, 2.5);

    // For tidiness, doesn't actually affect anything
    //qSH(0,0,0) = nan(""); qSH(0,1,0) = nan("");  qSH(0,N+3,0) = nan("");
    //lambda(0,0,0) = nan(""); lambda(0,1,0) = nan("");  lambda(0,N+3,0) = nan("");

    // Free streaming heat flow
    qFS =  beta * n_stag * n_t * T_stag * SI::qe * v_th;
	//qFS(0,0,0) = nan(""); qFS(0,1,0) = nan("");  qFS(0,N+3,0) = nan("");

	qt_low = -gam * T_stag(0,2,0) * n_stag(0,2,0)*  n_t * sqrt(2*SI::qe*T_stag(0,2,0)/m_i) * SI::qe;
    qt_upr = gam * T_stag(0,N+2,0) * n_stag(0,N+2,0) * n_t * sqrt(2*SI::qe*T_stag(0,N+2,0)/m_i) * SI::qe;

	switch(heat_type){
        case SPITZER_HARM :
            q = qSH;
        break;

        case LIMTED :
			q = (qSH * qFS) / (abs(qSH) + qFS);
        break;

        case CONVOLUTION :
            q = heat_convolution(qSH, CELL_YLOW);
        break;
    }

	Field3D q_conv = 2.5*SI::qe*n_t*c_st*v*n_stag*T_stag;
    //Convection term
	if (convection) {
    	q += q_conv;
	}

	if (heat_type != SPITZER_HARM) {

		col = length/lambda_0;
		qL = (qt_low - q(0,2,0) - exp(-col)   * (qt_upr-q(0,N+2,0)) ) / (1-exp(-2*col));
		qR = (qt_upr - q(0,N+2,0) - exp(-col) * (qt_low-q(0,2,0))   ) / (1-exp(-2*col));

		// -0.25 offset to staggered grid
		q = q + qL * exp(-(ypos-0.25)/lambda_0) + qR * exp((ypos-0.25-length)/lambda_0);
	}

	q.applyBoundary("lower_target", "dirichlet_o2(" + std::to_string(qt_low) + ")");
	q.applyBoundary("upper_target", "dirichlet_o2(" + std::to_string(qt_upr) + ")");

	//if (delay) {
	//	if (t<=4e-3){
    //		q.applyBoundary("lower_target", "dirichlet_o2(" + std::to_string(qt_low) + ")");
    //		q.applyBoundary("upper_target", "dirichlet_o2(" + std::to_string(qt_upr) + ")");
	//	}
	//	else {
	//		q.applyBoundary("lower_target", "free_o2");
	//		q.applyBoundary("upper_target", "free_o2");
	//	}
	//}
	//else {
	//	q.applyBoundary("lower_target", "dirichlet_o2(" + std::to_string(qt_low) + ")");
	//	q.applyBoundary("upper_target", "dirichlet_o2(" + std::to_string(qt_upr) + ")");
	//}

    // Fluid pressure
    p = 2*(n_t*n)*SI::qe*T;
    p.mergeYupYdown();
    p_dyn = m_i * (n_t*n) * (c_st*v) * (c_st*v);

	//T(0,1,0) = (3*T(0,2,0)-T(0,3,0))/2;
	//T(0,0,0) = (3*T(0,1,0)-T(0,2,0))/2;
	//T(0,N+2,0) = (3*T(0,N+1,0)-T(0,N,0))/2;
	//T(0,N+3,0) = (3*T(0,N+2,0)-T(0,N+1,0))/2;

    // Fluid equations
    ddt(n) = c_st * (S_n - FDDY(v, n, CELL_CENTRE));
    ddt(v) = -DDY(p, CELL_YLOW)/(m_i*n_stag*n_t*c_st) - c_st*(2 * VDDY(v, v, CELL_YLOW)  +  v*(VDDY(v, n, CELL_YLOW)/n_stag));
    ddt(T) = (1 / (3 * n_t*n * SI::qe)) * ( S_u - DDY(q, CELL_CENTRE, DIFF_C2) + VDDY(v, p, CELL_CENTRE)) + (T/n) * ddt(n);

    ddt_T = ddt(T);
    ddt_n = ddt(n);
    ddt_v = ddt(v);

    return 0;
  }
};

BOUTMAIN(SOLNL);