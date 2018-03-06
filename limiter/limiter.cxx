/*
 * 1D Braginksii model for plasma density, energy and momentum with
 * free streaming limited heat flow
 */

#include <bout/physicsmodel.hxx>
#include <bout/constants.hxx>
#include <derivs.hxx>
#include <field_factory.hxx>
#include <interpolation.hxx>

class Doubled : public PhysicsModel {
private:

    // Evolving variables
    Field3D T, n, v;

    // Derived quantities
    Field3D p, q, p_dyn, qSH, qFS, v_centre;//, Sn_centre;

	Field3D A, B, C;

    // Source terms
    Field3D S_n, S_u, ypos;

    BoutReal kappa_0, q_in, T_t, length;
    BoutReal m_i = SI::Mp;
	BoutReal m_e = SI::Me;

    //Normalisations
    BoutReal n_t, c_st;

    // Time derivatives
    Field3D ddt_T, ddt_n, ddt_v, ddy_T, ddy_nv;

    // Number of non boundary grid points
    int N = mesh->yend-mesh->ystart+1;

protected:

  // This is called once at the start
  int init(bool restarting) override {

    Options *options = Options::getRoot()->getSection("SHConduction");

    OPTION(options, kappa_0, 1.0);
    OPTION(options, T_t, 1.0);
    OPTION(options, n_t, 1.0);

    c_st = sqrt(2*SI::qe*T_t/m_i);

    OPTION(Options::getRoot()->getSection("mesh"), length, 1.0);

	q_in = -7.5 * T_t * n_t * c_st * SI::qe;
	S_u = -2*q_in/length;

    FieldFactory f(mesh);
    S_n = f.create3D("n:S_n");
    //S_u = f.create3D("T:S_u");

    v.setLocation(CELL_YLOW);   // Stagger
    qSH.setLocation(CELL_YLOW); // Stagger
    q.setLocation(CELL_YLOW);   // Stagger
	//S_n.setLocation(CELL_YLOW); // Stagger

    ypos = f.create3D("mesh:ypos");
    ypos.applyBoundary("lower_target", "free_o3");
    ypos.applyBoundary("upper_target", "free_o3");
    ypos.mergeYupYdown();

    SOLVE_FOR3(T, n, v);
    SAVE_REPEAT2(p, p_dyn);
	SAVE_REPEAT4(q, qSH, qFS, v_centre);
	SAVE_REPEAT3(A, B, C);
	SAVE_REPEAT3(ddt_T, ddt_n, ddt_v);
	SAVE_REPEAT2(ddy_T, ddy_nv)
    SAVE_ONCE4(kappa_0, q_in, T_t, length);
	SAVE_ONCE(S_n);//, Sn_centre);
    SAVE_ONCE3(n_t, c_st, S_u);
    SAVE_ONCE(ypos);

    return 0;
  }

  int rhs(BoutReal t) override {
    mesh->communicate(n,v,T);

    // This cell doesn't get used, as we take only C2 derivatives
    // But the extrapolation can sometimes make it negative
    // which causes a problem with the T^2.5 term.
    T(0,0,0) = T_t; T(0,N+3,0) = T_t;

    // Need to calculate the value of q one cell into the right boundary
    // because this cell is ON the boundary now as a result of being staggered
    // so we can use the T BC to get a well defined value here
    // this is then used later to calculate (DDY(q))_{N-1}
    qSH = DDY(T, CELL_YLOW, DIFF_C2);
    qSH(0,N+2,0) = (T(0,N+2,0)-T(0,N+1,0))/mesh->coordinates()->dy(0,0,0);
    qSH *= -kappa_0 * interp_to(pow(T, 2.5), CELL_YLOW);

    // For tidiness, doesn't actually affect anything
    qSH(0,0,0) = 0; qSH(0,1,0) = 0;  qSH(0,N+3,0) = 0;

	qFS = 0.03 * n * T * SI::qe * (T/m_e);       // Free streaming heat flow
    q = ((qSH * qFS) / (qSH + qFS));             // limited heatflow

    // Fluid pressure
    p = 2*(n_t*n)*SI::qe*T;
    p_dyn = m_i * (n_t*n) * (c_st*v) * (c_st*v);

	//p.mergeYupYdown();

	v_centre = interp_to(v, CELL_CENTRE);

    // Fluid equations
    ddt(n) =  c_st * (S_n - FDDY(v, n, CELL_CENTRE));
    ddt(v) = (-DDY(p, CELL_YLOW))/(m_i*n*n_t*c_st) - c_st*(2 * VDDY(v, v, CELL_YLOW)  +  v*(VDDY(v, n, CELL_YLOW)/n));
    n.applyTDerivBoundary();
    ddt(T) = (1 / (3 * n_t*n * SI::qe)) * (S_u - DDY(q, CELL_CENTRE, DIFF_C2)); //+ VDDY(v,p, CELL_CENTRE)) + (T/n) * ddt(n);

	A =  (1 / (3 * n_t*n * SI::qe)) * (S_u - DDY(q, CELL_CENTRE, DIFF_C2) );
	B =  0;//VDDY(v_centre, p, CELL_CENTRE);
	C =  (T/n) * ddt(n);

	ddy_nv = FDDY(v, n, CELL_CENTRE);
	ddy_T = DDY(T, CELL_CENTRE);
    ddt_T = ddt(T);
    ddt_n = ddt(n);
    ddt_v = ddt(v);

    return 0;
  }
};

BOUTMAIN(Doubled);