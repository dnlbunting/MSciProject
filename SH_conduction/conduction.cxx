/*
 * 1D temperature conduction problem modified
 * to use Spitzer-Harm conductivity and to be
 * driven with constant heat boundary conditions
 */

#include <bout/physicsmodel.hxx>
#include <bout/constants.hxx>
#include <derivs.hxx>

class SHConduction : public PhysicsModel {
private:

  Field3D T, q;
  BoutReal kappa_0, q_in, T_t, length;

  Field3D n = 1e-19; // Use constant density everywhere

protected:

  // This is called once at the start
  int init(bool restarting) override {

    // Save the simulation parameters with the results
    Options *options = Options::getRoot()->getSection("SHConduction");

    OPTION(options, kappa_0, 1.0);
    OPTION(options, q_in, 1.0);
    OPTION(options, T_t, 1.0);
    OPTION(Options::getRoot()->getSection("mesh"), length, 1.0);

    SOLVE_FOR(T);
    SAVE_REPEAT(q);
    SAVE_ONCE4(kappa_0, q_in, T_t, length);

    return 0;
  }

  int rhs(BoutReal t) override {

    // Want to apply boundary conditions to q that are consistent with the boundary conditions
    // applied to T through the configuration file

    // For a Dirichlet BC on q this means inverting the q definition to find a condition
    // on T and DY(T) at the boundary -> Trivial for Newton/SH but hard in general ie convolution

    // Need to set the T boundaries to free_o3 which just evolves them to 3rd order
    // How can this be reconciled with the need to fix the target temperature???


    q = -kappa_0 * pow(T, 2.5) * DDY(T); // Spitzer-Haram conductivity

    q.applyBoundary("upper_target", "dirichlet_o4(" + std::to_string(q_in) + ")"); // Fix upstream end to be a constant heat flow
    q.applyBoundary("lower_target", "free_o3"); // Allow heat at target to equilibrate

    ddt(T) = -(2./3.) * (n / SI::qe) * DDY(q); // Evolve T

    return 0;
  }
};

BOUTMAIN(SHConduction);
