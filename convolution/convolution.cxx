/*
 * 1D Braginksii model for plasma density, energy and momentum with
 * Spitzer-Harm conductivity driven with constant heat boundary conditions
 */

#include <bout/physicsmodel.hxx>
#include <bout/constants.hxx>
#include <derivs.hxx>
#include <field_factory.hxx>
#include <functional>
#include <math.h>

#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnpy.hpp"

typedef std::function<double(double)>  SplineFunc;

class gsl_function_pp : public gsl_function
{
   public:
   gsl_function_pp(SplineFunc const& func) : _func(func){
    function=&gsl_function_pp::invoke;
    params=this;
   }
   private:
   SplineFunc _func;
   static double invoke(double x, void *params) {
    return static_cast<gsl_function_pp*>(params)->_func(x);
   }
};



class Convolution : public PhysicsModel {
private:

  // Evolving variables
  Field3D T, n, v;

  // Time derivatives
  Field3D ddt_T, ddt_n, ddt_v;

  // Derived quantities
  Field3D p, q, p_dyn, qSH;

  // Source terms
  Field3D S_n ;

  BoutReal kappa_0, q_in, T_t, length;
  BoutReal m_i = SI::Mp;

  //Normalisations
  BoutReal n_t, c_st;

  // Convolution kernel
  Field3D w = 0.0;

  // Interpolation splines and accelerators for n and T
  int N = mesh->yend-mesh->ystart+1;
  gsl_interp_accel *Tacc = gsl_interp_accel_alloc ();
  gsl_interp_accel *nacc = gsl_interp_accel_alloc ();
  gsl_interp_accel *qSHacc = gsl_interp_accel_alloc ();
  gsl_spline *_Tspline = gsl_spline_alloc(gsl_interp_cspline, N);
  gsl_spline *_nspline = gsl_spline_alloc(gsl_interp_cspline, N);
  gsl_spline *_qSHspline = gsl_spline_alloc(gsl_interp_cspline, N);

  SplineFunc Tspline,nspline,qSHspline;

  //Integration workspace
  gsl_integration_workspace * integration_workspace = gsl_integration_workspace_alloc(10000);


protected:

  int init(bool restarting) override {

    Options *options = Options::getRoot()->getSection("SHConduction");

    OPTION(options, kappa_0, 1.0);
    OPTION(options, q_in, 1.0);
    OPTION(options, T_t, 1.0);
    OPTION(options, n_t, 1.0);

    c_st = sqrt(2*SI::qe*T_t/m_i);

    OPTION(Options::getRoot()->getSection("mesh"), length, 1.0);

    FieldFactory f(mesh);
    S_n = f.create3D("n:S_n");

    SOLVE_FOR3(T, n, v);
    SAVE_REPEAT4(ddt_T, ddt_n, ddt_v, qSH);
    SAVE_REPEAT3(q, p, p_dyn);
    SAVE_ONCE4(kappa_0, q_in, T_t, length);
    SAVE_ONCE3(S_n, n_t, c_st);

    v.setLocation(CELL_YLOW); // Stagger

    return 0;
  }

  xt::xarray<double> FieldToArray (Field3D F){
    xt::xarray<double> X = xt::zeros<double>({N});
    for (int k = mesh->ystart; k < mesh->yend+1; ++k) {
      X(k-mesh->ystart) =  F(0,k,0);
    }
    return X;
  }

  xt::xarray<double> SplineToArray (SplineFunc F){
    xt::xarray<double> X = xt::zeros<double>({N});
    for (int k = mesh->ystart; k < mesh->yend+1; ++k) {
      double x = mesh->GlobalY(k) * length;
      X(k-mesh->ystart) =  F(x);
    }
    return X;
  }

  /*
   Interpolates a Field3D f using the underlying mesh
   and wraps it in a lambda
  */
  SplineFunc GSLInterpolate(Field3D f, gsl_spline* spline, gsl_interp_accel* acc){
    double x[N], y[N];
    /* Copy this into a new array, BOUT doesn't guarantee
       that the underlying field data is contiguous.
       Obviously this also breaks if nprocs > 1 */

    for (int j = mesh->ystart; j < mesh->yend+1; ++j) {
      y[j-mesh->ystart] = f(0,j,0);
      x[j-mesh->ystart] = mesh->GlobalY(j) * length;
    }
    gsl_spline_init(spline, x, y, N);
    gsl_interp_accel_reset(nacc);

    SplineFunc  F = [=](double x){return gsl_spline_eval (spline, x, acc);};
    return F;
  }

  /*
    Uses interpolation spline lambda to integrate
    the underlying field from x1 to x2
  */
  BoutReal GSLIntegrate(SplineFunc spline, BoutReal x1, BoutReal x2,  bool handle = false ){

    double result, error;
    int errno;
    gsl_function_pp Fp(spline);
    gsl_function *F = static_cast<gsl_function*>(&Fp);

    if (handle){gsl_set_error_handler_off();}
    errno = gsl_integration_qags(F, x1, x2, 1e-6, 1e-7, 1000, integration_workspace, &result, &error);
    if (handle){gsl_set_error_handler (NULL);}

    if (errno != 0){
      output << "Integration Error! " << errno << " result = " <<result << " error = " << error << endl;
    }
    return result;
  }

  BoutReal TrapziumIntegrate(SplineFunc spline, BoutReal x1, BoutReal x2, int N){
    BoutReal result = 0.5*(spline(x1) + spline(x2));
    BoutReal dx = (x2 - x1)/N;

    for (int i = 1; i < N; ++i){
     result += spline(x1 + i*dx);
    }
    result *= dx;
    return result;
  }

  BoutReal kernel(BoutReal x1, BoutReal x2){

    BoutReal Z = 1;
    BoutReal a = 32;
    BoutReal w, lambda_0, lambda, logLambda;

    logLambda = 15.2 - 0.5*log(nspline(x2) * (n_t/1e20)) + log(Tspline(x2)/1000);
    lambda_0 = (2.5e17/n_t) * Tspline(x2) / (nspline(x2) * logLambda);
    lambda = a * sqrt(Z+1)*lambda_0;

    BoutReal n_integral = TrapziumIntegrate(nspline, x2, x1, 100); //GSLIntegrate(nspline, x2, x1, true);

    w = exp(-abs(n_integral)/(lambda*nspline((x2))))/(2*lambda);
    return w;
  }

  Field3D heat_convolution(SplineFunc qSH){

    xt::xarray<double> X = xt::zeros<double>({100, 100}) - 1;
    Field3D heat = 0.0;
    BoutReal xstart = mesh->GlobalY(mesh->ystart) * length;
    BoutReal xend = mesh->GlobalY(mesh->yend) * length;

    for (int j = mesh->ystart; j < mesh->yend+1; ++j) {
      double x = mesh->GlobalY(j) * length;
      double result, error;

      SplineFunc F = [&](double x1){ return qSH(x1) * kernel(x, x1);};

      /*for (int k = mesh->ystart; k < mesh->yend+1; ++k) {
          double x1 = mesh->GlobalY(k) * length;
          X(j-mesh->ystart,k-mesh->ystart) =  F(x1);
      }*/

      //result = GSLIntegrate([&](double x1){ return qSH(x1) * kernel(x, x1);}, xstart, xend, true);
      result = TrapziumIntegrate(F, xstart, xend, 100);
      heat(0,j,0) = result;
    }

    //xt::dump_npy("/Users/Daniel/Documents/Imperial/MSciProject/test.npy", X);
    return heat;
  }


  int rhs(BoutReal t) override {
    mesh->communicate(n,v,T);

    qSH = -kappa_0 * pow(T, 2.5) * DDY(T);

    // Set up interpolating splines
    Tspline = GSLInterpolate(T, _Tspline, Tacc);
    nspline = GSLInterpolate(n, _nspline, nacc);
    qSHspline = GSLInterpolate(qSH, _qSHspline, qSHacc);

    p = 2*(n_t*n)*SI::qe*T;
    p_dyn = m_i * (n_t*n) * (c_st*v) * (c_st*v);
    p.mergeYupYdown();

    q = heat_convolution(qSHspline); // Convolution
    if (t > 5e-3){
      q.applyBoundary("upper_target", "dirichlet_o4(" + std::to_string(q_in) + ")"); // Fix upstream end to be a constant heat flow
    }
    else {
      q.applyBoundary("upper_target", "dirichlet_o4(0)"); // Fix upstream end to be a constant heat flow
    }

    q.applyBoundary("lower_target", "free_o3"); // Allow heat at target to equilibrate

      /*xt::dump_npy("/Users/Daniel/Documents/Imperial/MSciProject/q.npy", FieldToArray(q));
      xt::dump_npy("/Users/Daniel/Documents/Imperial/MSciProject/qSH.npy", FieldToArray(qSH));
      exit(1);
    }*/

    ddt(n) = c_st * (S_n - FDDY(v, n));
    ddt(v) = -c_st * VDDY(v, v) - DDY(p)/(m_i*n*n_t*c_st);
    n.applyTDerivBoundary();
    ddt(T) = (1 / (3 * n_t*n * SI::qe)) * (-DDY(q) + VDDY(v, p)) + (T/n) * ddt(n);

    ddt_T = ddt(T);
    ddt_n = ddt(n);
    ddt_v = ddt(v);

    return 0;
  }
};

BOUTMAIN(Convolution);
