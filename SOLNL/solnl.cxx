
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
  Field3D p, q, p_dyn, qSH, v_centre;

  Field3D A,B,C;

  // Source terms
  Field3D S_n, S_u, ypos;

  BoutReal kappa_0, q_in, T_t, length;
  BoutReal m_i = SI::Mp;

  //Normalisations
  BoutReal n_t, c_st;

  // Convolution kernel
  Field3D w = 0.0;

  // Time derivatives
  Field3D ddt_T, ddt_n, ddt_v;

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
    SAVE_REPEAT3(ddt_n, ddt_T, ddt_v)
    SAVE_REPEAT4(A, B, C, v_centre)

    SAVE_ONCE4(kappa_0, q_in, T_t, length);
    SAVE_ONCE4(S_n, n_t, c_st, S_u);
    SAVE_ONCE(ypos);

    return 0;
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
  std::function<double(int,int)>  make_kernel() const{
    BoutReal Z = 1;
    BoutReal a = 32;

    std::vector<BoutReal> n_arr = field_to_vector(n);
    std::vector<BoutReal> T_arr = field_to_vector(T);

    return [=](int i1, int i2){
    BoutReal lambda_0, lambda, logLambda;
    logLambda = 15.2 - 0.5*log(n_arr[i2] * (n_t/1e20)) + log(T_arr[i2]/1000);
    lambda_0 = (2.5e17/n_t) * T_arr[i2] / (n_arr[i2] * logLambda);
    lambda = a * sqrt(Z+1)*lambda_0;
    return exp(-abs(TrapeziumIntegrate(n_arr, i2, i1, length/N))/(lambda*n_arr[i2]))/(2*lambda);};
  }

  Field3D heat_convolution(Field3D qSH, CELL_LOC loc) const{
    /*
      Calculate the convolution at each grid point
    */

    // Lots of closure magic to make this thread safe
    std::function<double(int,int)> kernel = make_kernel();
    std::vector<BoutReal> q_arr = field_to_vector(qSH);

    auto parallel_core = [=](int j) {
      std::vector<BoutReal> F(mesh->yend+1);
      for (int k = mesh->ystart; k < mesh->yend+1; ++k) {
          F[k] = q_arr[j] * kernel(j, k);
      }
      return TrapeziumIntegrate(F, mesh->ystart, mesh->yend, length/N);
    };


    // Launch the futures
    Field3D heat = 0.0;
    std::vector<std::future<BoutReal>> futures(mesh->yend+1);
    for (int j = mesh->ystart; j < mesh->yend+1; ++j) {
      futures[j] = std::async(std::launch::async, std::bind(parallel_core, j));
    }

    // Collect the futures
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
    T(0,0,0) = T_t; T(0,1,0) = T_t;
    T(0,N+2,0) = T_t; T(0,N+3,0) = T_t;

    // Need to calculate the value of q one cell into the right boundary
    // because this cell is ON the boundary now as a result of being staggered
    // so we can use the T BC to get a well defined value here
    // this is then used later to calculate (DDY(q))_{N-1}
    qSH = DDY(T, CELL_YLOW, DIFF_C2);
    qSH(0,N+2,0) = (T(0,N+2,0)-T(0,N+1,0))/mesh->coordinates()->dy(0,0,0);
    qSH *= -kappa_0 * interp_to(pow(T, 2.5), CELL_YLOW);

    // For tidiness, doesn't actually affect anything
    qSH(0,0,0) = 0; qSH(0,1,0) = 0;  qSH(0,N+3,0) = 0;

    q = qSH;

    // Fluid pressure
    p = 2*(n_t*n)*SI::qe*T;
    p_dyn = m_i * (n_t*n) * (c_st*v) * (c_st*v);

    // Fluid equations
    ddt(n) =  c_st * (S_n - FDDY(v, n, CELL_CENTRE));
    ddt(v) = (-DDY(p, CELL_YLOW))/(m_i*n*n_t*c_st) - c_st*(2 * VDDY(v, v, CELL_YLOW)  +  v*(VDDY(v, n, CELL_YLOW)/n));
    n.applyTDerivBoundary();
    ddt(T) = (1 / (3 * n_t*n * SI::qe)) * ( S_u - DDY(q, CELL_CENTRE, DIFF_C2) ); //+ VDDY(v,p, CELL_CENTRE)) + (T/n) * ddt(n);

    A=VDDY(v, n, CELL_CENTRE);
    B=0;C=0;v_centre=0;

    ddt_T = ddt(T);
    ddt_n = ddt(n);
    ddt_v = ddt(v);

    return 0;
  }
};

BOUTMAIN(SOLNL);
