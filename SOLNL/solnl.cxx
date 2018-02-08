
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

class SOLNL : public PhysicsModel {
private:

  // Evolving variables
  Field3D T, n, v;

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

  // Number of non boundary grid points
  int N = mesh->yend-mesh->ystart+1;

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
    SAVE_REPEAT4(q, qSH, p, p_dyn);
    SAVE_ONCE4(kappa_0, q_in, T_t, length);
    SAVE_ONCE4(S_n, n_t, c_st, S_u);
    SAVE_ONCE(ypos);

    return 0;
  }

  BoutReal kernel(int i1, int i2){

    BoutReal Z = 1;
    BoutReal a = 32;
    BoutReal w, lambda_0, lambda, logLambda;

    logLambda = 15.2 - 0.5*log(n(0,i2,0) * (n_t/1e20)) + log(T(0,i2,0)/1000);
    lambda_0 = (2.5e17/n_t) * T(0,i2,0) / (n(0,i2,0) * logLambda);
    lambda = a * sqrt(Z+1)*lambda_0;

    BoutReal n_integral = TrapeziumIntegrate(n, i2, i1, length/N);
    w = exp(-abs(n_integral)/(lambda*n(0,i2,0)))/(2*lambda);

    return w;
  }

  Field3D heat_convolution(Field3D qSH, CELL_LOC loc){
    /*
      Calculate the convolution at each grid point
    */
    Field3D heat = 0.0;
    std::vector<std::future<BoutReal>> futures(mesh->yend+1);

    for (int j = mesh->ystart; j < mesh->yend+1; ++j) {

      std::vector<BoutReal> F(mesh->yend+1);
      for (int k = mesh->ystart; k < mesh->yend+1; ++k) {
          F[k] = qSH(0,j,0) * kernel(j, k);
      }
      futures[j] = std::async( [=]{return TrapeziumIntegrate(F, mesh->ystart, mesh->yend, length/N);} );
    }
    for (int j = mesh->ystart; j < mesh->yend+1; ++j) {
      heat(0,j,0) = futures[j].get();
    }

    // These extrapolated boundaries are technically invalid,
    // but necessary to use interp_to to shift ylow <-> ycenter
    // maybe the sheath BC needs to be applied here????
    heat.applyBoundary("upper_target", "free_o3");
    heat.applyBoundary("lower_target", "free_o3");
    heat.mergeYupYdown();

    return interp_to(heat, loc);
  }


  int rhs(BoutReal t) override {
    mesh->communicate(n,v,T);

    // Create constant T heat bath BCs for T
    T.applyBoundary("upper_target", "dirichlet(" + std::to_string(T_t) + ")");
    T.applyBoundary("upper_target", "dirichlet(" + std::to_string(T_t) + ")");
    T(0,0,0) = T_t; T(0,1,0) = T_t;  T(0,2,0) = T_t;
    T(0,202,0) = T_t; T(0,203,0) = T_t;  T(0,201,0) = T_t;

    qSH = -kappa_0 * DDY(T, CELL_YLOW) * pow(T, 2.5);
    // These extrapolated boundaries are technically invalid,
    // but necessary to use interp_to to shift ylow <-> ycenter
    qSH.applyBoundary("upper_target", "free_o3");
    qSH.applyBoundary("lower_target", "free_o3");

    // Fluid pressure
    p = 2*(n_t*n)*SI::qe*T;
    p.mergeYupYdown();
    p_dyn = m_i * (n_t*n) * (c_st*v) * (c_st*v);

    q = heat_convolution(qSH, CELL_YLOW);

    // Fluid equations
    ddt(n) = c_st * (S_n - FDDY(v, n, CELL_CENTRE));
    ddt(v) = -c_st * VDDY(v, v, CELL_YLOW) - DDY(p, CELL_YLOW)/(m_i*interp_to(n,CELL_YLOW)*n_t*c_st);
    n.applyTDerivBoundary();
    ddt(T) = (1 / (3 * n_t*n * SI::qe)) * (S_u - DDY(q, CELL_CENTRE) + VDDY(v,p, CELL_CENTRE)) + (T/n) * ddt(n);

    return 0;
  }
};

BOUTMAIN(SOLNL);
