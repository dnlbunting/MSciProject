/*
 * 1D Braginksii model for plasma density, energy and momentum with
 * Spitzer-Harm conductivity driven with constant heat boundary conditions
 */

#include <bout/physicsmodel.hxx>
#include <bout/constants.hxx>
#include <derivs.hxx>
#include <field_factory.hxx>

class Braginksii : public PhysicsModel {
private:

  // Evolving variables
  Field3D T, n, v;

  // Time derivatives
  Field3D ddt_T, ddt_n, ddt_v;

  // Derived quantities
  Field3D p, q, p_dyn;

  // Source terms
  Field3D S_n ;

  BoutReal kappa_0, q_in, T_t, length;
  BoutReal m_i = SI::Mp;

  //Normalisations
  BoutReal n_t, c_st;

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

    SOLVE_FOR3(T, n, v);
    SAVE_REPEAT3(ddt_T, ddt_n, ddt_v);
    SAVE_REPEAT3(q, p, p_dyn);
    SAVE_ONCE4(kappa_0, q_in, T_t, length);
    SAVE_ONCE3(S_n, n_t, c_st);

    v.setLocation(CELL_YLOW); // Stagger

    return 0;
  }

  int rhs(BoutReal t) override {
    mesh->communicate(n,v,T);

    p = 2*(n_t*n)*SI::qe*T;
    p_dyn = m_i * (n_t*n) * (c_st*v) * (c_st*v);

    q = 2.5*p*c_st*v - kappa_0 * pow(T, 2.5) * DDY(T); // Spitzer-Haram conductivity
    q.applyBoundary("upper_target", "dirichlet_o4(" + std::to_string(q_in) + ")"); // Fix upstream end to be a constant heat flow
    q.applyBoundary("lower_target", "free_o3"); // Allow heat at target to equilibrate

    ddt(n) = c_st * (S_n - FDDY(v, n));
    ddt(v) = -c_st * VDDY(v, v) - DDY(p)/(m_i*n*n_t*c_st);
    n.applyTDerivBoundary();
    ddt(T) = (1 / (3 * n_t*n * SI::qe)) * (-DDY(q) + v * DDY(p)) + (T/n) * ddt(n);

    ddt_T = ddt(T);
    ddt_n = ddt(n);
    ddt_v = ddt(v);

    return 0;
  }
};

BOUTMAIN(Braginksii);
