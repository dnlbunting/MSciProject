
/*
 * 1D Braginksii model for plasma density, energy and momentum with
 * Spitzer-Harm conductivity driven with constant heat boundary conditions
 */

#include <bout/physicsmodel.hxx>
#include <bout/constants.hxx>
#include <derivs.hxx>
#include <field_factory.hxx>

#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

#include <interpolation.hxx>


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

class Doubled : public PhysicsModel {
private:

  // Evolving variables
  Field3D T, n, v;

  // Time derivatives
  Field3D ddt_T, ddt_n, ddt_v;

  // Derived quantities
  Field3D p, q, p_dyn, qSH;

  // Source terms
  Field3D S_n, S_u, ypos;

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
  gsl_spline *_Tspline = gsl_spline_alloc(gsl_interp_cspline, N+5);
  gsl_spline *_nspline = gsl_spline_alloc(gsl_interp_cspline, N+5);
  gsl_spline *_qSHspline = gsl_spline_alloc(gsl_interp_cspline, N+5);

  SplineFunc Tspline,nspline,qSHspline;

  //Integration workspace
  gsl_integration_workspace * integration_workspace = gsl_integration_workspace_alloc(10000);

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

protected:

  // This is called once at the start
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
    S_u = f.create3D("T:S_u");

    v.setLocation(CELL_YLOW); // Stagger
    qSH.setLocation(CELL_YLOW); // Stagger
    q.setLocation(CELL_YLOW); // Stagger

    ypos = f.create3D("mesh:ypos");
    ypos.applyBoundary("lower_target", "free_o3");
    ypos.applyBoundary("upper_target", "free_o3");
    ypos.mergeYupYdown();


    SOLVE_FOR3(T, n, v);
    SAVE_REPEAT3(ddt_T, ddt_n, ddt_v);
    SAVE_REPEAT4(q, qSH, p, p_dyn);
    SAVE_ONCE4(kappa_0, q_in, T_t, length);
    SAVE_ONCE4(S_n, n_t, c_st, S_u);
    SAVE_ONCE(ypos);

    return 0;
  }

    /*
   Interpolates a Field3D f using the underlying mesh
   and wraps it in a lambda
  */
  SplineFunc GSLInterpolate(Field3D f, gsl_spline* spline, gsl_interp_accel* acc){
    double x[N+5], y[N+5];
    double dy = length/N;
    /* Copy this into a new array, BOUT doesn't guarantee
       that the underlying field data is contiguous.
       Obviously this also breaks if nprocs > 1 */

    // Interpolate the x axis to be in the same location
    // the input field
    Field3D ypos_m = interp_to(ypos, f.getLocation());

    //Extrapolate the field into the boundary region
    ypos_m.applyBoundary("upper_target", "free_o3");
    ypos_m.applyBoundary("lower_target", "free_o3");
    f.applyBoundary("upper_target", "free_o3");
    f.applyBoundary("lower_target", "free_o3");

    if (f.getLocation() == CELL_YLOW){
      for (int j = mesh->ystart-2; j < mesh->yend+3; ++j) {
        y[j] = f(0,j,0);
        x[j] = ypos_m(0,j,0);
        //output << j << "= " << x[j-mesh->ystart] << ", " << y[j-mesh->ystart] << endl;
      }
      // Pad final element
      x[N+4] = x[N+3] + 0.5*dy;
      y[N+4] = y[N+3];
    }

    else if (f.getLocation() == CELL_CENTRE){
      for (int j = mesh->ystart-2; j < mesh->yend+3; ++j) {
        y[j+1] = f(0,j,0);
        x[j+1] = ypos_m(0,j,0);
        //output << j << "= " << x[j-mesh->ystart+1] << ", " << y[j-mesh->ystart+1] << endl;
      }
      // Pad first element
      x[0] = x[1] - 0.5*dy;
      y[0] = y[1];
    }

    gsl_spline_init(spline, x, y, N+5);
    gsl_interp_accel_reset(acc);

    SplineFunc  F = [=](double x){return gsl_spline_eval(spline, x, acc);};
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
    errno = gsl_integration_qags(F, x1, x2, 1e-5, 1e-5, 1000, integration_workspace, &result, &error);
    if (handle){gsl_set_error_handler (NULL);}

    if (errno != 0){
      output << "Integration Error! " << errno << "x1 = " << x1 << ", x2 = " << x2 << " result = " <<result << " error = " << error << endl;
      xt::dump_npy("/Users/Daniel/Documents/Imperial/MSciProject/doubled/kernel.npy", SplineToArray(spline));
      exit(1);
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

    BoutReal n_integral = TrapziumIntegrate(nspline, x2, x1, 200);
    w = exp(-abs(n_integral)/(lambda*nspline(x2)))/(2*lambda);

    return w;
  }

  Field3D heat_convolution(SplineFunc qSH, CELL_LOC loc){
    Field3D heat = 0.0;
    heat.setLocation(loc);

    xt::xarray<double> X = xt::zeros<double>({200, 200}) - 1;
    xt::dump_npy("/Users/Daniel/Documents/Imperial/MSciProject/doubled/qSH.npy", SplineToArray(qSH));

    Field3D ypos_m = interp_to(ypos, loc);
    BoutReal xstart = ypos_m(0,mesh->ystart,0);
    BoutReal xend = ypos_m(0,mesh->yend,0);

    for (int j = mesh->ystart; j < mesh->yend+1; ++j) {
      double x = ypos_m(0,j,0);
      double result, error;
      SplineFunc F = [&](double x1){ return qSH(x1) * kernel(x, x1);};

      for (int k = mesh->ystart; k < mesh->yend+1; ++k) {
          X(j-mesh->ystart, k-mesh->ystart) =  F(ypos_m(0,k,0));
      }

      //result = GSLIntegrate([&](double x1){ return qSH(x1) * kernel(x, x1);}, xstart, xend, true);
      result = TrapziumIntegrate(F, xstart, xend, 200);
      heat(0,j,0) = result;
    }

    xt::dump_npy("/Users/Daniel/Documents/Imperial/MSciProject/doubled/X.npy", X);
    xt::dump_npy("/Users/Daniel/Documents/Imperial/MSciProject/doubled/q.npy", FieldToArray(heat));
    exit(0);
    return heat;
  }


  int rhs(BoutReal t) override {
    mesh->communicate(n,v,T);

    // Create constant T heat bath BCs for T
    T.applyBoundary("upper_target", "dirichlet(" + std::to_string(T_t) + ")"); // Fix upstream end to be a constant heat flow
    T.applyBoundary("upper_target", "dirichlet(" + std::to_string(T_t) + ")"); // Fix upstream end to be a constant heat flow
    T(0,0,0) = T_t; T(0,1,0) = T_t;  T(0,2,0) = T_t;
    T(0,202,0) = T_t; T(0,203,0) = T_t;  T(0,201,0) = T_t;

    Field3D T_low = interp_to(T, CELL_YLOW);
    qSH = -kappa_0 * pow(T_low, 2.5) * DDY(T, CELL_YLOW);

    // Set up interpolating splines
    Tspline = GSLInterpolate(T, _Tspline, Tacc);
    nspline = GSLInterpolate(n, _nspline, nacc);
    qSHspline = GSLInterpolate(qSH, _qSHspline, qSHacc);

    p = 2*(n_t*n)*SI::qe*T;
    p.mergeYupYdown();
    p_dyn = m_i * (n_t*n) * (c_st*v) * (c_st*v);


    if (t > 0.005){
        q = heat_convolution(qSHspline, CELL_YLOW); // Convolution
    }
    else
      {q = qSH;}


    ddt(n) = c_st * (S_n - FDDY(v, n, CELL_CENTRE));
    ddt(v) = -c_st * VDDY(v, v, CELL_YLOW) - DDY(p, CELL_YLOW)/(m_i*interp_to(n,CELL_YLOW)*n_t*c_st);
    n.applyTDerivBoundary();
    ddt(T) = (1 / (3 * n_t*n * SI::qe)) * (S_u - DDY(q, CELL_CENTRE) + VDDY(v,p, CELL_CENTRE)) + (T/n) * ddt(n);

    ddt_T = ddt(T);
    ddt_n = ddt(n);
    ddt_v = ddt(v);

    return 0;
  }
};

BOUTMAIN(Doubled);
