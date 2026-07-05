/* Author: Giuseppe Orlando, 2026. */

// @sect{Include files}
#include <filesystem>
namespace fs = std::filesystem;

// We start by including all the necessary deal.II header files
//
#include <deal.II/base/parallel.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/timer.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/distributed/solution_transfer.h>

/*--- Include headers related to the problem of interest ---*/
#include "include/ic_bc/ic_3D_nonhydrostatic_hill.h"

#include "include/ic_bc/Rayleigh_damping.h"
#include "include/mapping/mapping.h"

#include "euler_operator.h"
#include "turbulent_operator.h"

using namespace Atmospheric_Flow;
using namespace Turbulent_Diffusivity;

// @sect{The <code>EulerSolver</code> class}

// Now for the main class of the program. It implements the solver for the
// Euler equations using the discretization previously implemented.
//
template<unsigned dim>
class EulerSolver {
public:
  using Vec    = LinearAlgebra::distributed::Vector<double>;
  using Number = Vec::value_type;

  EulerSolver(const RunTimeParameters::Data_Storage& data,
              const TimeStepping::RungeKutta<Number>& explicit_RK,
              const TimeStepping::RungeKutta<Number>& implicit_RK); /*--- Class constructor ---*/

  void run(const bool verbose = false,
           const unsigned output_interval = 10,
           const std::string& n_files = "",
           const std::string& dt_save_ = "");
  /*--- The run function which actually runs the simulation ---*/

protected:
  DeclException2(ExcInvalidTimeStep,
                 Number,
                 Number,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  // Auxiliary variables for some physical parameters
  const Number t0; /*--- Initial time auxiliary variable ----*/
  const Number T;  /*--- Final time auxiliary variable ----*/

  // Auxiliary variables for some numerical parameters
  Number dt;          /*--- Time step auxiliary variable ---*/
  Number CFL;         /*--- Courant number auxiliary variable (not necessarily used) ---*/
  bool   dt_from_CFL; /*--- Fix whether the time step is fixed from CFL or from input ---*/

  unsigned n_stages;   /*--- Number of IMEX stages ---*/
  unsigned IMEX_stage; /*--- Flag to check at which current stage of the IMEX we are ---*/

  // Auxiliary variables for linear solvers parameters
  unsigned max_its;        /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
  Number   atol_iterative; /*--- Auxiliary variable for the absolute tolerance of linear solvers ---*/
  Number   rtol_iterative; /*--- Auxiliary variable for the relative tolerance of linear solvers ---*/

  // Domain discretization
  parallel::distributed::Triangulation<dim> triangulation; /*--- The variable which stores the mesh ---*/

  // Finite element spaces for all the variables
  FESystem<dim> fe_density;
  FESystem<dim> fe_velocity;
  FESystem<dim> fe_pressure;

  // Degrees of freedom handlers for all the variables
  DoFHandler<dim> dof_handler_density;
  DoFHandler<dim> dof_handler_velocity;
  DoFHandler<dim> dof_handler_pressure;

  // Auxiliary mapping for curved boundary
  MappingQ<dim> mapping;

  // Auxiliary quadratures for all the variables
  QGaussLobatto<dim> quadrature_density;
  QGaussLobatto<dim> quadrature_velocity;
  QGaussLobatto<dim> quadrature_pressure;

  // Variables for the density
  std::vector<Vec> rho_s;
  Vec rhs_rho;

  // Variables for the velocity
  std::vector<Vec> u_s;
  Vec u_fixed;
  Vec rhs_u;

  // Variables for the pressure
  std::vector<Vec> pres_s;
  Vec pres_fixed;
  Vec dpres_fixed;
  Vec rhs_pres;

  // Background fields
  Vec rho_bar;
  Vec u_bar;
  Vec pres_bar;

  // Variables for the potential temperature
  std::vector<Vec> theta_s;
  Vec rhs_theta;

  // Damping layers functions for all the unknowns
  Vec dt_tau_rho;
  Vec dt_tau_u;
  Vec dt_tau_pres;
  Vec dt_tau_rho_aux;
  Vec dt_tau_u_aux;
  Vec dt_tau_pres_aux;

  Vec dt_tau_rho_right;
  Vec dt_tau_u_right;
  Vec dt_tau_pres_right;
  Vec dt_tau_rho_aux_right;
  Vec dt_tau_u_aux_right;
  Vec dt_tau_pres_aux_right;

  Vec dt_tau_rho_left;
  Vec dt_tau_u_left;
  Vec dt_tau_pres_left;
  Vec dt_tau_rho_aux_left;
  Vec dt_tau_u_aux_left;
  Vec dt_tau_pres_aux_left;

  Vec dt_tau_rho_right_y;
  Vec dt_tau_u_right_y;
  Vec dt_tau_pres_right_y;
  Vec dt_tau_rho_aux_right_y;
  Vec dt_tau_u_aux_right_y;
  Vec dt_tau_pres_aux_right_y;

  Vec dt_tau_rho_left_y;
  Vec dt_tau_u_left_y;
  Vec dt_tau_pres_left_y;
  Vec dt_tau_rho_aux_left_y;
  Vec dt_tau_u_aux_left_y;
  Vec dt_tau_pres_aux_left_y;

  // Auxiliary structures for the matrix-free
  std::shared_ptr<MatrixFree<dim, Number>> matrix_free_storage;

  std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Auxiliary container for the matrix-free ---*/

  std::vector<const AffineConstraints<Number>*> constraints; /*--- Auxiliary container for the matrix-free ---*/
  AffineConstraints<Number> constraints_velocity,
                            constraints_pressure,
                            constraints_density;

  std::vector<QGauss<1>> quadratures; /*--- Auxiliary container for the quadrature in matrix-free ---*/

  // Manifold (mapping) data structures
  GalChenMapping::PushForward<dim, Number> push_forward;
  GalChenMapping::PullBack<dim, Number>    pull_back;
  FunctionManifold<dim, dim, dim>          manifold;

  // Functions to set the initial conditions
  ICBC::Density<dim, Number>  rho_init;
  ICBC::Velocity<dim, Number> u_init;
  ICBC::Pressure<dim, Number> pres_init;

  // Functions for the Rayleigh damping profile
  RayleighDamping::Rayleigh<dim, 1, Number>       dt_tau;
  RayleighDamping::Rayleigh_Aux<dim, 1, Number>   dt_tau_aux;
  RayleighDamping::Rayleigh<dim, dim, Number>     dt_tau_vel;
  RayleighDamping::Rayleigh_Aux<dim, dim, Number> dt_tau_vel_aux;

  RayleighDamping::Rayleigh_Right<dim, 1, Number>       dt_tau_right;
  RayleighDamping::Rayleigh_Aux_Right<dim, 1, Number>   dt_tau_aux_right;
  RayleighDamping::Rayleigh_Right<dim, dim, Number>     dt_tau_vel_right;
  RayleighDamping::Rayleigh_Aux_Right<dim, dim, Number> dt_tau_vel_aux_right;

  RayleighDamping::Rayleigh_Left<dim, 1, Number>       dt_tau_left;
  RayleighDamping::Rayleigh_Aux_Left<dim, 1, Number>   dt_tau_aux_left;
  RayleighDamping::Rayleigh_Left<dim, dim, Number>     dt_tau_vel_left;
  RayleighDamping::Rayleigh_Aux_Left<dim, dim, Number> dt_tau_vel_aux_left;

  RayleighDamping::Rayleigh_RightY<dim, 1, Number>       dt_tau_right_y;
  RayleighDamping::Rayleigh_Aux_RightY<dim, 1, Number>   dt_tau_aux_right_y;
  RayleighDamping::Rayleigh_RightY<dim, dim, Number>     dt_tau_vel_right_y;
  RayleighDamping::Rayleigh_Aux_RightY<dim, dim, Number> dt_tau_vel_aux_right_y;

  RayleighDamping::Rayleigh_LeftY<dim, 1, Number>       dt_tau_left_y;
  RayleighDamping::Rayleigh_Aux_LeftY<dim, 1, Number>   dt_tau_aux_left_y;
  RayleighDamping::Rayleigh_LeftY<dim, dim, Number>     dt_tau_vel_left_y;
  RayleighDamping::Rayleigh_Aux_LeftY<dim, dim, Number> dt_tau_vel_aux_left_y;

  // Now we declare a bunch of variables for output
  fs::path saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/

  ConditionalOStream pcout;

  std::ofstream      time_out;
  ConditionalOStream ptime_out;
  TimerOutput        time_table;

  // Parameters related to restart
  bool     restart,
           save_for_restart;
  unsigned step_restart;
  Number   time_restart;
  bool     as_initial_conditions;

  // Auxiliary routines to perform the spatial discretization
  void create_triangulation(const RunTimeParameters::Data_Storage& data); /*--- Function to create the grid ---*/

  void setup_dofs(); /*--- Function to set the dofs ---*/

  void initialize(); /*--- Function to initialize the fields ---*/

  // Auxiliary routine to save results
  void output_results(const unsigned step); /*--- Function to save the results ---*/

  // Auxiliary routines for post-processing
  Number get_max_velocity() const; /*--- Get maximum velocity to compute the Courant number ---*/

  Number get_min_density() const; /*--- Get minimum density ---*/

  Number get_max_density() const; /*--- Get maximum density ---*/

  Number compute_max_celerity() const; /*--- Compute maximum celerity for acoustic Courant number ---*/

  std::array<Number, dim> compute_max_Cu_per_direction() const; /*--- Get maximum Courant numbers along all the directions ---*/

  std::array<Number, dim> compute_max_C_per_direction() const; /*--- Get maximum acoustic Courant numbers along all the directions ---*/

private:
  using MatrixType = EULEROperator<dim,
                                   EquationData::degree_u,
                                   EquationData::degree_rho,
                                   EquationData::degree_p,
                                   EquationData::quadrature_degree,
                                   EquationData::quadrature_degree + GalChenMapping::extra_quadrature_degree,
                                   Vec>;
  MatrixType euler_matrix;

  using TurbulentType = TurbulentOperator<dim,
                                          EquationData::degree_u,
                                          EquationData::degree_p,
                                          EquationData::quadrature_degree,
                                          EquationData::quadrature_degree + GalChenMapping::extra_quadrature_degree,
                                          Vec>;
  TurbulentType turbulent_matrix;

  Number atol_fixed_point; /*--- Auxiliary variable for the absolute tolerance of fixed point loop ---*/
  Number rtol_fixed_point; /*--- Auxiliary variable for the relative tolerance of fixed point loop ---*/

  Vector<Number> Linfty_error_per_cell_pres; /*--- Auxiliary variable for the end of the fixed point loop ---*/

  Vec rhs_momentum,
      rhs_u_precomputed,
      rhs_pres_precomputed,
      extra_rhs_u; /*--- Auxiliary vectors for the Schur complement ---*/

  Number Ma;     /*--- Mach number for post-processing ---*/
  Number inv_Ma; /*--- Inverse Mach number for post-processing ---*/
  Number gamma;  /*--- gamma (i.e. Cp/Cv) ---*/
  Number Gamma;  /*--- (gamma - 1)/gamma ---*/

  Number h_min; /*--- Minimum cell diameter ---*/

  // Auxiliary routines to numerically solve the problem
  void update_density(); /*--- Function to update the density ---*/

  void pressure_fixed_point(); /*--- Function to compute the pressure in the fixed point loop ---*/

  void update_velocity(); /*--- Function to compute the velocity in the fixed point loop ---*/

  void update_pressure(); /*--- Function to compute the pressure for the weighting step of the IMEX ---*/

  void precompute_rhs_pressure(); /*--- Auxiliary function to compute the rhs of the pressure equation ---*/

  unsigned perform_fixed_point_loop(); /*--- Auxiliary function for the fixed point loop ---*/

  void diffusion_step(); /*--- Solve the velocity equation for the turbulent part ---*/

  void temperature_step(); /*--- Solve the potential temperature equation for the turbulent part ---*/
};


//////////////////////////////////////////////////////////////
/*---- START WITH CLASS CONSTRUCTOR ---*/
/////////////////////////////////////////////////////////////

// @sect{ <code>EulerSolver::EulerSolver</code> }

// In the constructor, we just read all the data from the
// <code>Data_Storage</code> object that is passed as an argument, verify that
// the data we read are reasonable and, finally, create the triangulation and
// load the initial data.
//
template<unsigned dim>
EulerSolver<dim>::EulerSolver(const RunTimeParameters::Data_Storage& data,
                              const TimeStepping::RungeKutta<Number>& explicit_RK,
                              const TimeStepping::RungeKutta<Number>& implicit_RK):
  /*--- Time integration ---*/
  t0(data.initial_time),
  T(data.final_time),
  dt(data.dt),
  n_stages(explicit_RK.get_n_stages()),
  IMEX_stage(2),
  /*--- Linear solvers ---*/
  max_its(data.max_iterations),
  atol_iterative(data.atol_iterative),
  rtol_iterative(data.rtol_iterative),
  /*--- Space discretization ---*/
  triangulation(MPI_COMM_WORLD,
                parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe_density(FE_DGQ<dim>(EquationData::degree_rho), 1),
  fe_velocity(FE_DGQ<dim>(EquationData::degree_u), dim),
  fe_pressure(FE_DGQ<dim>(EquationData::degree_p), 1),
  dof_handler_density(triangulation),
  dof_handler_velocity(triangulation),
  dof_handler_pressure(triangulation),
  mapping(GalChenMapping::degree_mapping),
  quadrature_density(EquationData::degree_rho + 1),
  quadrature_velocity(EquationData::degree_u + 1),
  quadrature_pressure(EquationData::degree_p + 1),
  rho_s(n_stages + 1),
  u_s(n_stages + 1),
  pres_s(n_stages),
  theta_s(n_stages),
  dof_handlers(EquationData::n_vars),
  constraints(EquationData::n_vars),
  /*--- Domain ---*/
  push_forward(data.z_max, data.h, data.xc, data.yc, data.ac, data.L_ref),
  pull_back(data.z_max, data.h, data.xc, data.yc, data.ac, data.L_ref),
  manifold(push_forward, pull_back),
  /*--- Initial condition ---*/
  rho_init(data.p_bar, data.T_bar, data.rho_ref, data.L_ref,
           data.N, data.initial_time),
  u_init(data.u_bar, data.u_ref, data.initial_time),
  pres_init(data.p_bar, data.T_bar, data.p_ref, data.L_ref,
            data.N, data.initial_time),
  /*--- Boundary condition (Rayleigh damping) ---*/
  dt_tau(data.z_start, data.z_max, data.lambda_z, data.L_ref),
  dt_tau_aux(data.z_start, data.z_max, data.lambda_z, data.L_ref),
  dt_tau_vel(data.z_start, data.z_max, data.lambda_z, data.L_ref),
  dt_tau_vel_aux(data.z_start, data.z_max, data.lambda_z, data.L_ref),
  dt_tau_right(data.x_start_right, data.x_max, data.lambda_x_right, data.L_ref),
  dt_tau_aux_right(data.x_start_right, data.x_max, data.lambda_x_right, data.L_ref),
  dt_tau_vel_right(data.x_start_right, data.x_max, data.lambda_x_right, data.L_ref),
  dt_tau_vel_aux_right(data.x_start_right, data.x_max, data.lambda_x_right, data.L_ref),
  dt_tau_left(data.x_start_left, data.x_min, data.lambda_x_left, data.L_ref),
  dt_tau_aux_left(data.x_start_left, data.x_min, data.lambda_x_left, data.L_ref),
  dt_tau_vel_left(data.x_start_left, data.x_min, data.lambda_x_left, data.L_ref),
  dt_tau_vel_aux_left(data.x_start_left, data.x_min, data.lambda_x_left, data.L_ref),
  dt_tau_right_y(data.y_start_right, data.y_max, data.lambda_y_right, data.L_ref),
  dt_tau_aux_right_y(data.y_start_right, data.y_max, data.lambda_y_right, data.L_ref),
  dt_tau_vel_right_y(data.y_start_right, data.y_max, data.lambda_y_right, data.L_ref),
  dt_tau_vel_aux_right_y(data.y_start_right, data.y_max, data.lambda_y_right, data.L_ref),
  dt_tau_left_y(data.y_start_left, data.y_min, data.lambda_y_left, data.L_ref),
  dt_tau_aux_left_y(data.y_start_left, data.y_min, data.lambda_y_left, data.L_ref),
  dt_tau_vel_left_y(data.y_start_left, data.y_min, data.lambda_y_left, data.L_ref),
  dt_tau_vel_aux_left_y(data.y_start_left, data.y_min, data.lambda_y_left, data.L_ref),
  /*--- Output ---*/
  saving_dir(data.dir),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  restart(data.restart),
  save_for_restart(data.save_for_restart),
  step_restart(data.step_restart),
  time_restart(data.time_restart),
  as_initial_conditions(data.as_initial_conditions),
  /*--- Auxiliary and fixed-point loop ---*/
  euler_matrix(data, explicit_RK, implicit_RK),
  turbulent_matrix(data, implicit_RK),
  atol_fixed_point(data.atol_fixed_point),
  rtol_fixed_point(data.rtol_fixed_point),
  Ma(euler_matrix.get_Mach()), inv_Ma(static_cast<Number>(1.0)/Ma),
  gamma(EquationData::Cp_Cv),
  Gamma((gamma - static_cast<Number>(1.0))/gamma)
  {
    /*--- Check time step coherence ---*/
    if(data.CFL.empty()) {
      dt = static_cast<Number>(data.dt);
      AssertThrow(!((dt <= static_cast<Number>(0.0)) || (dt > T)),
                  ExcInvalidTimeStep(dt, T));

      dt_from_CFL = false;
    }
    else {
      CFL = static_cast<Number>(std::stod(data.CFL));

      dt_from_CFL = true;
    }

    /*--- Check non-dimensional parameters coherence ---*/
    if(std::abs(Ma*Ma -
                static_cast<Number>(data.rho_ref)*
                static_cast<Number>(data.u_ref)*static_cast<Number>(data.u_ref)/
                static_cast<Number>(data.p_ref)) > static_cast<Number>(1e-10)) {
      pcout << "WARNING: The non-dimensional Mach number in the parameter file is not coherent "
                "with the reference values declared (and theoretically used for initial conditions and computational domain)."
                "The simulation will go on with the Mach number read in the parameter file, but you may want to double check!" << std::endl;
    }
    if(std::abs(euler_matrix.get_Froude()*euler_matrix.get_Froude() -
                static_cast<Number>(data.u_ref)*static_cast<Number>(data.u_ref)/
                (static_cast<Number>(EquationData::g)*static_cast<Number>(data.L_ref))) > static_cast<Number>(1e-10)) {
      pcout << "WARNING: The non-dimensional Froude number in the parameter file is not coherent "
                "with the reference values declared (and theoretically used for initial conditions and computational domain)."
                "The simulation will go on with the Froude number read in the parameter file, but you may want to double check!" << std::endl;
    }

    /*--- Create saving directory if needed and related output stream ---*/
    if(!fs::exists(saving_dir)) {
      fs::create_directory(saving_dir);
    }
    time_out = std::ofstream(saving_dir.string() + "/time_analysis_" +
                             Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat");

    /*--- Initialize structures ---*/
    matrix_free_storage = std::make_shared<MatrixFree<dim, Number>>();

    /*--- Clear the containers for safety ---*/
    constraints_velocity.clear();
    constraints_velocity.close();
    constraints_pressure.clear();
    constraints_pressure.close();
    constraints_density.clear();
    constraints_density.close();

    quadratures.clear();

    /*--- Call initializing routines ---*/
    create_triangulation(data);
    setup_dofs();
    initialize();
  }


//////////////////////////////////////////////////////////////
/*---- FOCUS NOW ON INITIALIAZING AUXILIARY ROUTINES ---*/
/////////////////////////////////////////////////////////////

// @sect{<code>EulerSolver::create_triangulation</code>}

// The method that creates the triangulation.
//
template<unsigned dim>
void EulerSolver<dim>::create_triangulation(const RunTimeParameters::Data_Storage& data) {
  TimerOutput::Scope t(time_table, "Create triangulation");

  Point<dim, Number> lower_left;
  lower_left[0] = static_cast<Number>(data.x_min);
  lower_left[1] = static_cast<Number>(data.y_min);
  lower_left[2] = static_cast<Number>(data.z_min);
  Point<dim, Number> upper_right;
  upper_right[0] = static_cast<Number>(data.x_max)/
                   static_cast<Number>(data.L_ref);
  upper_right[1] = static_cast<Number>(data.y_max)/
                   static_cast<Number>(data.L_ref);
  upper_right[2] = static_cast<Number>(data.z_max)/
                   static_cast<Number>(data.L_ref);

  GridGenerator::subdivided_hyper_rectangle(triangulation, {data.n_elements_x, data.n_elements_y, data.n_elements_z},
                                            lower_left, upper_right, true);

  /*--- Consider periodic conditions along the horizontal direction ---*/
  std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>> periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 2, 3, 1, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  /*--- Build the proper triangulation ---*/
  if(restart) {
    triangulation.load(saving_dir.string() + "/solution_ser-" + Utilities::int_to_string(step_restart, 5));
  }
  else {
    triangulation.refine_global(data.n_global_refines);
  }

  /*--- Apply the mapping to build the physical domain ---*/
  GridTools::transform([this](const Point<dim, Number>& chart_point) {
                                return manifold.push_forward(chart_point);
                              },
                              triangulation);

  triangulation.set_all_manifold_ids_on_boundary(2*(dim - 1), 111);
  triangulation.set_manifold(111, manifold);
}

// After creating the triangulation, it creates the mesh dependent
// data, i.e. it distributes degrees of freedom, and
// initializes the matrices and vectors that we will use.
//
template<unsigned dim>
void EulerSolver<dim>::setup_dofs() {
  TimerOutput::Scope t(time_table, "Setup dofs");

  pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;
  h_min = GridTools::minimal_cell_diameter(triangulation, mapping)/std::sqrt(dim);
  pcout << "h_min = " << h_min << std::endl;

  /*--- Set degrees of freedom ---*/
  dof_handler_velocity.distribute_dofs(fe_velocity);
  dof_handler_pressure.distribute_dofs(fe_pressure);
  dof_handler_density.distribute_dofs(fe_density);

  pcout << "dim (V_h) = " << dof_handler_velocity.n_dofs()
        << std::endl
        << "dim (Q_h) = " << dof_handler_pressure.n_dofs()
        << std::endl
        << "dim (X_h) = " << dof_handler_density.n_dofs()
        << std::endl
        << "Ma        = " << Ma
        << std::endl
        << "Fr        = " << euler_matrix.get_Froude()
        << std::endl
        << std::endl;

  /*--- Set additional data to check which variables neeed to be updated ---*/
  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mapping_update_flags                = (update_values | update_gradients |
                                                         update_quadrature_points | update_JxW_values);
  additional_data.mapping_update_flags_inner_faces    = (update_values | update_normal_vectors |
                                                         update_quadrature_points | update_JxW_values);
  additional_data.mapping_update_flags_boundary_faces = (update_values | update_normal_vectors |
                                                         update_quadrature_points | update_JxW_values);
  additional_data.tasks_parallel_scheme               = MatrixFree<dim, Number>::AdditionalData::none;

  /*--- Set the container with the dof handlers ---*/
  dof_handlers[EquationData::U_INDEX_DOF]   = &dof_handler_velocity;
  dof_handlers[EquationData::P_INDEX_DOF]   = &dof_handler_pressure;
  dof_handlers[EquationData::RHO_INDEX_DOF] = &dof_handler_density;

  /*--- Set the container with the constraints. Each entry is empty (no Dirichlet and weak imposition in general)
        and this is necessary only for compatibilty reasons ---*/
  constraints[EquationData::U_INDEX_DOF]   = &constraints_velocity;
  constraints[EquationData::P_INDEX_DOF]   = &constraints_pressure;
  constraints[EquationData::RHO_INDEX_DOF] = &constraints_density;

  /*--- Set the quadrature formula to compute the integrals for assembling bilinear and linear forms ---*/
  quadratures.push_back(QGauss<1>(EquationData::quadrature_degree));
  quadratures.push_back(QGauss<1>(EquationData::quadrature_degree + GalChenMapping::extra_quadrature_degree));
  quadratures.push_back(QGauss<1>(EquationData::degree_u + 1));
  quadratures.push_back(QGauss<1>(EquationData::degree_rho + 1));
  quadratures.push_back(QGauss<1>(EquationData::degree_p + 1));

  /*--- Initialize the matrix-free structure with DofHandlers, Constraints, Quadratures and AdditionalData ---*/
  matrix_free_storage->reinit(mapping, dof_handlers, constraints, quadratures, additional_data);

  /*--- Initialize the variables related to the velocity ---*/
  for(std::size_t idx = 0; idx < u_s.size(); ++idx) {
    matrix_free_storage->initialize_dof_vector(u_s[idx], EquationData::U_INDEX_DOF);
  }
  matrix_free_storage->initialize_dof_vector(u_fixed, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rhs_u, EquationData::U_INDEX_DOF);

  /*--- Initialize the variables related to the pressure ---*/
  for(std::size_t idx = 0; idx < pres_s.size(); ++idx) {
    matrix_free_storage->initialize_dof_vector(pres_s[idx], EquationData::P_INDEX_DOF);
  }
  matrix_free_storage->initialize_dof_vector(pres_fixed, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dpres_fixed, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rhs_pres, EquationData::P_INDEX_DOF);

  /*--- Initialize the variables related to the density ---*/
  for(std::size_t idx = 0; idx < rho_s.size(); ++idx) {
    matrix_free_storage->initialize_dof_vector(rho_s[idx], EquationData::RHO_INDEX_DOF);
  }
  matrix_free_storage->initialize_dof_vector(rhs_rho, EquationData::RHO_INDEX_DOF);

  /*--- Initialize the variables related to the potential temperature ---*/
  for(std::size_t idx = 0; idx < theta_s.size(); ++idx) {
    matrix_free_storage->initialize_dof_vector(theta_s[idx], EquationData::THETA_INDEX_DOF);
  }
  matrix_free_storage->initialize_dof_vector(rhs_theta, EquationData::THETA_INDEX_DOF);

  /*--- Initialize the auxiliary variable for the Schur complement ---*/
  matrix_free_storage->initialize_dof_vector(rhs_momentum, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rhs_u_precomputed, EquationData::U_INDEX_DOF);
  rhs_u_precomputed = 0;
  matrix_free_storage->initialize_dof_vector(rhs_pres_precomputed, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(extra_rhs_u, EquationData::U_INDEX_DOF);

  /*--- Initialize the variables related to the damping layers ---*/
  matrix_free_storage->initialize_dof_vector(dt_tau_u, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_u_aux, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel, dt_tau_u);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau, dt_tau_pres);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau, dt_tau_rho);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_aux, dt_tau_u_aux);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_aux, dt_tau_pres_aux);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_aux, dt_tau_rho_aux);

  matrix_free_storage->initialize_dof_vector(dt_tau_u_right, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_right, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_right, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_u_aux_right, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux_right, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux_right, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_right, dt_tau_u_right);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_right, dt_tau_pres_right);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_right, dt_tau_rho_right);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_aux_right, dt_tau_u_aux_right);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_aux_right, dt_tau_pres_aux_right);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_aux_right, dt_tau_rho_aux_right);

  matrix_free_storage->initialize_dof_vector(dt_tau_u_left, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_left, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_left, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_u_aux_left, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux_left, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux_left, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_left, dt_tau_u_left);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_left, dt_tau_pres_left);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_left, dt_tau_rho_left);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_aux_left, dt_tau_u_aux_left);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_aux_left, dt_tau_pres_aux_left);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_aux_left, dt_tau_rho_aux_left);

  matrix_free_storage->initialize_dof_vector(dt_tau_u_right_y, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_right_y, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_right_y, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_u_aux_right_y, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux_right_y, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux_right_y, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_right_y, dt_tau_u_right_y);
  VectorTools::interpolate(dof_handler_pressure, dt_tau_right_y, dt_tau_pres_right_y);
  VectorTools::interpolate(dof_handler_density, dt_tau_right_y, dt_tau_rho_right_y);
  VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_aux_right_y, dt_tau_u_aux_right_y);
  VectorTools::interpolate(dof_handler_pressure, dt_tau_aux_right_y, dt_tau_pres_aux_right_y);
  VectorTools::interpolate(dof_handler_density, dt_tau_aux_right_y, dt_tau_rho_aux_right_y);

  matrix_free_storage->initialize_dof_vector(dt_tau_u_left_y, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_left_y, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_left_y, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_u_aux_left_y, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux_left_y, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux_left_y, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_left_y, dt_tau_u_left_y);
  VectorTools::interpolate(dof_handler_pressure, dt_tau_left_y, dt_tau_pres_left_y);
  VectorTools::interpolate(dof_handler_density, dt_tau_left_y, dt_tau_rho_left_y);
  VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_aux_left_y, dt_tau_u_aux_left_y);
  VectorTools::interpolate(dof_handler_pressure, dt_tau_aux_left_y, dt_tau_pres_aux_left_y);
  VectorTools::interpolate(dof_handler_density, dt_tau_aux_left_y, dt_tau_rho_aux_left_y);

  matrix_free_storage->initialize_dof_vector(u_bar, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(pres_bar, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rho_bar, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(mapping, dof_handler_velocity, u_init, u_bar);
  VectorTools::interpolate(mapping, dof_handler_pressure, pres_init, pres_bar);
  VectorTools::interpolate(mapping, dof_handler_density, rho_init, rho_bar);

  dt_tau_u.scale(u_bar);
  dt_tau_pres.scale(pres_bar);
  dt_tau_rho.scale(rho_bar);

  dt_tau_u_right.scale(u_bar);
  dt_tau_pres_right.scale(pres_bar);
  dt_tau_rho_right.scale(rho_bar);

  dt_tau_u_left.scale(u_bar);
  dt_tau_pres_left.scale(pres_bar);
  dt_tau_rho_left.scale(rho_bar);

  dt_tau_u_right_y.scale(u_bar);
  dt_tau_pres_right_y.scale(pres_bar);
  dt_tau_rho_right_y.scale(rho_bar);

  dt_tau_u_left_y.scale(u_bar);
  dt_tau_pres_left_y.scale(pres_bar);
  dt_tau_rho_left_y.scale(rho_bar);

  /*--- Initialize the auxiliary variable to check the error and stop the fixed point loop ---*/
  Vector<Number> error_per_cell_tmp(triangulation.n_active_cells());
  Linfty_error_per_cell_pres.reinit(error_per_cell_tmp);
}

// @sect{ <code>EulerSolver::initialize</code> }

// This method loads the initial data
//
template<unsigned dim>
void EulerSolver<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize state");

  /*--- Initialize the fields in case of restart ---*/
  if(restart) {
    parallel::distributed::SolutionTransfer<dim, Vec>
    solution_transfer_density(dof_handler_density);
    parallel::distributed::SolutionTransfer<dim, Vec>
    solution_transfer_velocity(dof_handler_velocity);
    parallel::distributed::SolutionTransfer<dim, Vec>
    solution_transfer_pressure(dof_handler_pressure);

    rho_s.front().zero_out_ghost_values();
    u_s.front().zero_out_ghost_values();
    pres_s.front().zero_out_ghost_values();

    solution_transfer_density.deserialize(rho_s.front());
    solution_transfer_velocity.deserialize(u_s.front());
    solution_transfer_pressure.deserialize(pres_s.front());
  }
  /*--- Initialize the fields ---*/
  else {
    VectorTools::interpolate(mapping, dof_handler_density, rho_init, rho_s.front());
    VectorTools::interpolate(mapping, dof_handler_velocity, u_init, u_s.front());
    VectorTools::interpolate(mapping, dof_handler_pressure, pres_init, pres_s.front());
  }

  /*--- Initilize also the potential temperature so as to obtain a proper saving of initial state ---*/
  for(const auto& cell: dof_handler_pressure.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      std::vector<types::global_dof_index> dof_indices(fe_pressure.dofs_per_cell);
      cell->get_dof_indices(dof_indices);
      for(unsigned idx = 0; idx < dof_indices.size(); ++idx) {
        const auto pres = pres_s.front()(dof_indices[idx]);
        const auto T    = pres/rho_s.front()(dof_indices[idx]);
        const auto Pi   = std::pow(pres, Gamma);
        theta_s.front()(dof_indices[idx]) = T/Pi;
      }
    }
  }
}


//////////////////////////////////////////////////////////////
/*---- AUXILIARY ROUTINES TO NUMERICALLY SOLVE THE CONSERVATION LAWS ---*/
/////////////////////////////////////////////////////////////

// @sect{<code>EulerSolver::update_density</code>}

// This implements the update of the density
//
template<unsigned dim>
void EulerSolver<dim>::update_density() {
  TimerOutput::Scope t(time_table, "Update density");

  /*--- Set the proper dof index to specify that we are dealing with continuity equation ---*/
  const std::vector<unsigned> index_dof_handler = {EquationData::RHO_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::RHO_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  std::vector<Vec> vectors_rhs_density_equation;
  for(unsigned idx_s = 0; idx_s < IMEX_stage - 1; ++idx_s) {
    vectors_rhs_density_equation.push_back(rho_s[idx_s]);
    vectors_rhs_density_equation.push_back(u_s[idx_s]);
  }

  euler_matrix.vmult_rhs_density(rhs_rho, vectors_rhs_density_equation);

  /*--- Solve the system for the density ---*/
  euler_matrix.vmult(rho_s[IMEX_stage - 1], rhs_rho);
}

// @sect{<code>EulerSolver::pressure_fixed_point</code>}

// Auxiliary routine to compute the rhs of the pressure equation
// (contribution that does not change during the fixed point loop)
//
template<unsigned dim>
void EulerSolver<dim>::precompute_rhs_pressure() {
  /*--- Set the proper dof index ---*/
  const std::vector<unsigned> index_dof_handler = {EquationData::U_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::U_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  std::vector<Vec> vectors_rhs_momentum_equation;
  for(unsigned idx_s = 0; idx_s < IMEX_stage - 1; ++idx_s) {
    vectors_rhs_momentum_equation.push_back(rho_s[idx_s]);
    vectors_rhs_momentum_equation.push_back(u_s[idx_s]);
    vectors_rhs_momentum_equation.push_back(pres_s[idx_s]);
  }
  vectors_rhs_momentum_equation.push_back(rho_s[IMEX_stage - 1]);

  euler_matrix.vmult_rhs_momentum(rhs_momentum, vectors_rhs_momentum_equation);

  /*--- Solve to compute first contribution to rhs --*/
  euler_matrix.vmult(rhs_u_precomputed, rhs_momentum);
}

// This implements a step of the fixed point procedure for the computation of the pressure
//
template<unsigned dim>
void EulerSolver<dim>::pressure_fixed_point() {
  TimerOutput::Scope t(time_table, "Fixed point pressure");

  /*--- Set the proper dof index ---*/
  const std::vector<unsigned> index_dof_handler = {EquationData::P_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::P_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  std::vector<Vec> vectors_rhs_energy_equation;
  for(unsigned idx_s = 0; idx_s < IMEX_stage - 1; ++idx_s) {
    vectors_rhs_energy_equation.push_back(rho_s[idx_s]);
    vectors_rhs_energy_equation.push_back(u_s[idx_s]);
    vectors_rhs_energy_equation.push_back(pres_s[idx_s]);
  }
  vectors_rhs_energy_equation.push_back(rho_s[IMEX_stage - 1]);
  vectors_rhs_energy_equation.push_back(u_fixed);
  vectors_rhs_energy_equation.push_back(pres_fixed);

  euler_matrix.vmult_rhs_energy(rhs_pres, vectors_rhs_energy_equation);

  // Perform matrix-vector multiplication with enthalpy matrix (which changes over time)
  euler_matrix.set_pres_fixed(pres_fixed); // Set the current pressure for the fixed point loop to the operator
  euler_matrix.vmult_enthalpy(rhs_pres_precomputed, rhs_u_precomputed);

  // Conclude computation of rhs for pressure fixed point
  rhs_pres.add(static_cast<Number>(-1.0), rhs_pres_precomputed);

  /*--- Jacobi preconditioner for this system ---*/
  PreconditionJacobi<MatrixType> preconditioner_Jacobi;
  euler_matrix.compute_diagonal();
  preconditioner_Jacobi.initialize(euler_matrix);

  /*--- Solve the linear system for the pressure---*/
  SolverControl solver_control(max_its, atol_iterative + rtol_iterative*rhs_pres.l2_norm(), false, true);
  SolverGMRES<Vec> gmres(solver_control);

  gmres.solve(euler_matrix, pres_fixed, rhs_pres, preconditioner_Jacobi);
}

// Auxiliary routine for the fixed point loop
//
template<unsigned dim>
unsigned EulerSolver<dim>::perform_fixed_point_loop() {
  /*--- Compute the contribution to the rhs that never changes ---*/
  precompute_rhs_pressure();

  /*--- Perform the fixed point loop ---*/
  unsigned iter;
  for(iter = 0; iter < 100; ++iter) {
    dpres_fixed.equ(static_cast<Number>(1.0), pres_fixed),
    pressure_fixed_point();
    update_velocity();

    /*--- Compute the relative error for the pressure ---*/
    VectorTools::integrate_difference(dof_handler_pressure, pres_fixed, Functions::ZeroFunction<dim, Number>(),
                                      Linfty_error_per_cell_pres, quadrature_pressure, VectorTools::Linfty_norm);
    const auto den = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm);
    dpres_fixed.add(static_cast<Number>(-1.0), pres_fixed);
    VectorTools::integrate_difference(dof_handler_pressure, dpres_fixed, Functions::ZeroFunction<dim, Number>(),
                                      Linfty_error_per_cell_pres, quadrature_pressure, VectorTools::Linfty_norm);
    const auto error = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm);
    if(error < atol_fixed_point + rtol_fixed_point*den)
      break; /*--- The fixed point loop is stopped whenever the relative error in infinity norm is below the specified tolerance ---*/
  }

  return iter;
}

// @sect{<code>EulerSolver::update_velocity</code>}

// This implements the velocity update in the fixed point procedure
// and in the final update stage of IMEX scheme
//
template<unsigned dim>
void EulerSolver<dim>::update_velocity() {
  TimerOutput::Scope t(time_table, "Update velocity");

  /*--- Set the proper dof index ---*/
  const std::vector<unsigned> index_dof_handler = {EquationData::U_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::U_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  if(IMEX_stage <= n_stages) {
    rhs_u.equ(static_cast<Number>(1.0), rhs_momentum);
    euler_matrix.vmult_pressure(extra_rhs_u, pres_fixed);
    rhs_u.add(static_cast<Number>(-1.0), extra_rhs_u);
  }
  else {
    std::vector<Vec> vectors_rhs_momentum_equation;
    for(unsigned idx_s = 0; idx_s < IMEX_stage - 1; ++idx_s) {
      vectors_rhs_momentum_equation.push_back(rho_s[idx_s]);
      vectors_rhs_momentum_equation.push_back(u_s[idx_s]);
      vectors_rhs_momentum_equation.push_back(pres_s[idx_s]);
    }

    euler_matrix.vmult_rhs_momentum(rhs_u, vectors_rhs_momentum_equation);
  }

  /*--- Solve the system for the velocity ---*/
  if(IMEX_stage <= n_stages) {
    euler_matrix.vmult(u_fixed, rhs_u);
  }
  else {
    euler_matrix.vmult(u_s.back(), rhs_u);
  }
}

// @sect{<code>EulerSolver::update_pressure</code>}

// This implements the update of the pressure for the hyperbolic part
//
template<unsigned dim>
void EulerSolver<dim>::update_pressure() {
  TimerOutput::Scope t(time_table, "Update pressure");

  /*--- Set the proper dof index ---*/
  const std::vector<unsigned> index_dof_handler = {EquationData::P_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::P_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  std::vector<Vec> vectors_rhs_energy_equation;
  for(unsigned idx_s = 0; idx_s < IMEX_stage - 1; ++idx_s) {
    vectors_rhs_energy_equation.push_back(rho_s[idx_s]);
    vectors_rhs_energy_equation.push_back(u_s[idx_s]);
    vectors_rhs_energy_equation.push_back(pres_s[idx_s]);
  }
  vectors_rhs_energy_equation.push_back(rho_s.back());
  vectors_rhs_energy_equation.push_back(u_s.back());

  euler_matrix.vmult_rhs_energy(rhs_pres, vectors_rhs_energy_equation);

  /*--- Solve the system for the pressure ---*/
  euler_matrix.vmult(pres_s.front(), rhs_pres);
}

// @sect{<code>EulerSolver::diffusion_step</code>}

// This implements the solution of the velocity equation for the turbulent operator
//
template<unsigned dim>
void EulerSolver<dim>::diffusion_step() {
  TimerOutput::Scope t(time_table, "Diffusion step");

  /*--- Set the proper dof index ---*/
  const std::vector<unsigned> index_dof_handler = {EquationData::U_INDEX_DOF};
  turbulent_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  turbulent_matrix.set_NS_stage(EquationData::U_INDEX_SYSTEM_TURB);

  /*--- Compute the rhs ---*/
  std::vector<Vec> vectors_rhs_velocity_equation;
  for(unsigned idx_s = 0; idx_s < IMEX_stage - 1; ++idx_s) {
    vectors_rhs_velocity_equation.push_back(u_s[idx_s]);
    vectors_rhs_velocity_equation.push_back(theta_s[idx_s]);
  }
  turbulent_matrix.vmult_rhs_velocity(rhs_u, vectors_rhs_velocity_equation);

  /*--- Jacobi preconditioner for this system ---*/
  PreconditionJacobi<TurbulentType> preconditioner_Jacobi;
  turbulent_matrix.compute_diagonal();
  preconditioner_Jacobi.initialize(turbulent_matrix);

  /*--- Solve the linear system for the velocity ---*/
  SolverControl solver_control(max_its, atol_iterative + rtol_iterative*rhs_u.l2_norm(), false, true);
  SolverGMRES<Vec> gmres(solver_control);

  u_s[IMEX_stage - 1].equ(static_cast<Number>(1.0), u_s[IMEX_stage - 2]);
  gmres.solve(turbulent_matrix, u_s[IMEX_stage - 1], rhs_u, preconditioner_Jacobi);
}

// @sect{<code>EulerSolver::temperature_step</code>}

// This implements the solution of the temperature equation for the turbulent operator
//
template<unsigned dim>
void EulerSolver<dim>::temperature_step() {
  TimerOutput::Scope t(time_table, "Temperature step");

  /*--- Set the proper dof index ---*/
  const std::vector<unsigned int> index_dof_handler = {EquationData::THETA_INDEX_DOF};
  turbulent_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  turbulent_matrix.set_NS_stage(EquationData::THETA_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  std::vector<Vec> vectors_rhs_theta_equation;
  for(unsigned idx_s = 0; idx_s < IMEX_stage - 1; ++idx_s) {
    vectors_rhs_theta_equation.push_back(u_s[idx_s]);
    vectors_rhs_theta_equation.push_back(theta_s[idx_s]);
  }
  turbulent_matrix.vmult_rhs_temperature(rhs_theta, vectors_rhs_theta_equation);

  /*--- Jacobi preconditioner for this system ---*/
  PreconditionJacobi<TurbulentType> preconditioner_Jacobi;
  turbulent_matrix.compute_diagonal();
  preconditioner_Jacobi.initialize(turbulent_matrix);

  /*--- Solve the linear system for the potential temperature ---*/
  SolverControl solver_control(max_its, atol_iterative + rtol_iterative*rhs_theta.l2_norm(), false, true);
  SolverGMRES<Vec> gmres(solver_control);

  theta_s[IMEX_stage - 1].equ(static_cast<Number>(1.0), theta_s[IMEX_stage - 2]);
  gmres.solve(turbulent_matrix, theta_s[IMEX_stage - 1], rhs_theta, preconditioner_Jacobi);
}


//////////////////////////////////////////////////////////////
/*---- AUXILIARY ROUTINES FOR POST-PROCESSING ---*/
/////////////////////////////////////////////////////////////

// @sect{ <code>EulerSolver::output_results</code> }

// This method plots the current solution.
//
template<unsigned dim>
void EulerSolver<dim>::output_results(const unsigned step) {
  TimerOutput::Scope t(time_table, "Output results");

  /*--- Save the fields ---*/
  DataOut<dim> data_out;

  rho_s.front().update_ghost_values();
  data_out.add_data_vector(dof_handler_density, rho_s.front(), "rho", {DataComponentInterpretation::component_is_scalar});
  std::vector<std::string> velocity_names(dim, "u");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
  u_s.front().update_ghost_values();
  data_out.add_data_vector(dof_handler_velocity, u_s.front(), velocity_names, component_interpretation_velocity);
  pres_s.front().update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, pres_s.front(), "p", {DataComponentInterpretation::component_is_scalar});

  theta_s.front().update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, theta_s.front(), "theta", {DataComponentInterpretation::component_is_scalar});

  /*--- Save background state ---*/
  rho_bar.update_ghost_values();
  data_out.add_data_vector(dof_handler_density, rho_bar, "rho_bar", {DataComponentInterpretation::component_is_scalar});
  u_bar.update_ghost_values();
  std::fill(velocity_names.begin(), velocity_names.end(), "u_bar");
  data_out.add_data_vector(dof_handler_velocity, u_bar, velocity_names, component_interpretation_velocity);
  pres_bar.update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, pres_bar, "p_bar", {DataComponentInterpretation::component_is_scalar});

  data_out.build_patches(mapping, EquationData::degree_u, DataOut<dim>::curved_inner_cells);

  DataOutBase::DataOutFilterFlags flags(false, true);
  DataOutBase::DataOutFilter      data_filter(flags);
  data_out.write_filtered_data(data_filter);
  std::string output = saving_dir.string() + "/solution-" + Utilities::int_to_string(step, 5) + ".h5";
  data_out.write_hdf5_parallel(data_filter, output, MPI_COMM_WORLD);
  std::vector<XDMFEntry> xdmf_entries;
  auto new_xdmf_entry = data_out.create_xdmf_entry(data_filter, output, step, MPI_COMM_WORLD);
  xdmf_entries.push_back(new_xdmf_entry);
  output = saving_dir.string() + "/solution-" + Utilities::int_to_string(step, 5) + ".xdmf";
  data_out.write_xdmf_file(xdmf_entries, output, MPI_COMM_WORLD);
  output = saving_dir.string() + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  /*--- Save high order mapping ---*/
  output = saving_dir.string() + "/solution_high_order-" + Utilities::int_to_string(step, 5) + ".vtu";
  DataOutBase::VtkFlags flags_high_order;
  flags_high_order.write_higher_order_cells = true;
  data_out.set_flags(flags_high_order);
  data_out.build_patches(mapping, GalChenMapping::degree_mapping, DataOut<dim>::curved_inner_cells);
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  /*--- Serialization ---*/
  if(save_for_restart) {
    parallel::distributed::SolutionTransfer<dim, Vec>
    solution_transfer_density(dof_handler_density);
    parallel::distributed::SolutionTransfer<dim, Vec>
    solution_transfer_velocity(dof_handler_velocity);
    parallel::distributed::SolutionTransfer<dim, Vec>
    solution_transfer_pressure(dof_handler_pressure);

    rho_s.front().update_ghost_values();
    u_s.front().update_ghost_values();
    pres_s.front().update_ghost_values();

    solution_transfer_density.prepare_for_serialization(rho_s.front());
    solution_transfer_velocity.prepare_for_serialization(u_s.front());
    solution_transfer_pressure.prepare_for_serialization(pres_s.front());

    triangulation.save(saving_dir.string() + "/solution_ser-" + Utilities::int_to_string(step, 5));
  }

  /*--- Call this function to be sure to be able to write again on these fields ---*/
  rho_s.front().zero_out_ghost_values();
  u_s.front().zero_out_ghost_values();
  pres_s.front().zero_out_ghost_values();
  theta_s.front().zero_out_ghost_values();
}


// The following function is used in determining the maximum velocity
// in order to compute the CFL
//
template<unsigned dim>
typename EulerSolver<dim>::Number EulerSolver<dim>::get_max_velocity() const {
  const unsigned n_q_points = quadrature_velocity.size();
  FEValues<dim> fe_values_velocity(mapping, fe_velocity, quadrature_velocity, update_values);
  std::vector<Vector<Number>> velocity_values(n_q_points, Vector<Number>(dim));

  auto max_local_velocity = static_cast<Number>(0.0);

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_velocity.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values_velocity.reinit(cell);

      fe_values_velocity.get_function_values(u_s.front(), velocity_values);

      for(unsigned q = 0; q < n_q_points; q++) {
        max_local_velocity = std::max(max_local_velocity, velocity_values[q].l2_norm());
      }
    }
  }

  return Utilities::MPI::max(max_local_velocity, MPI_COMM_WORLD);
}

// The following function is used in determining the minimum density
//
template<unsigned dim>
typename EulerSolver<dim>::Number EulerSolver<dim>::get_min_density() const {
  const unsigned n_q_points = quadrature_density.size();
  FEValues<dim> fe_values(mapping, fe_density, quadrature_density, update_values);
  std::vector<Number> solution_values(n_q_points);

  auto min_local_density = std::numeric_limits<Number>::max();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_density.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values.get_function_values(rho_s[IMEX_stage - 1], solution_values);

      for(unsigned q = 0; q < n_q_points; ++q) {
        min_local_density = std::min(min_local_density, solution_values[q]);
      }
    }
  }

  return Utilities::MPI::min(min_local_density, MPI_COMM_WORLD);
}

// The following function is used in determining the maximum density
//
template<unsigned dim>
typename EulerSolver<dim>::Number EulerSolver<dim>::get_max_density() const {
  return rho_s[IMEX_stage - 1].linfty_norm();
}

// The following function is used in determining the maximum celerity
//
template<unsigned dim>
typename EulerSolver<dim>::Number EulerSolver<dim>::compute_max_celerity() const {
  const unsigned n_q_points = quadrature_pressure.size();
  FEValues<dim> fe_values(mapping, fe_pressure, quadrature_pressure, update_values);
  std::vector<Number> solution_values_pressure(n_q_points),
                      solution_values_density(n_q_points);

  auto max_local_celerity = std::numeric_limits<Number>::min();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_pressure.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values.get_function_values(pres_s.front(), solution_values_pressure);
      fe_values.get_function_values(rho_s.front(), solution_values_density);

      for(unsigned q = 0; q < n_q_points; ++q) {
        max_local_celerity = std::max(max_local_celerity,
                                      std::sqrt(gamma*(solution_values_pressure[q]/solution_values_density[q])));
      }
    }
  }

  return Utilities::MPI::max(max_local_celerity, MPI_COMM_WORLD);
}

// The following function is used in determining the maximum advective Courant numbers along the directions
//
template<unsigned dim>
std::array<typename EulerSolver<dim>::Number, dim>
EulerSolver<dim>::compute_max_Cu_per_direction() const {
  const unsigned n_q_points = quadrature_velocity.size();
  FEValues<dim> fe_values(mapping, fe_velocity, quadrature_velocity, update_values);
  std::vector<Vector<Number>> solution_values_velocity(n_q_points, Vector<Number>(dim));

  std::array<Number, dim> res;
  std::fill(res.begin(), res.end(), std::numeric_limits<Number>::min());

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_velocity.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values.get_function_values(u_s.front(), solution_values_velocity);

      for(unsigned q = 0; q < n_q_points; ++q) {
        for(unsigned d = 0; d < dim; ++d) {
          res[d] = std::max(res[d],
                            EquationData::degree_u*(std::abs(solution_values_velocity[q](d))*dt/
                            static_cast<Number>(cell->extent_in_direction(d))));
        }
      }
    }
  }

  for(unsigned d = 0; d < dim; ++d) {
    res[d] = Utilities::MPI::max(res[d], MPI_COMM_WORLD);
  }

  return res;
}

// The following function is used in determining the maximum Courant number along the directions
//
template<unsigned dim>
std::array<typename EulerSolver<dim>::Number, dim>
EulerSolver<dim>::compute_max_C_per_direction() const {
  const unsigned n_q_points = quadrature_pressure.size();
  FEValues<dim> fe_values(mapping, fe_pressure, quadrature_pressure, update_values);
  std::vector<Number> solution_values_pressure(n_q_points),
                      solution_values_density(n_q_points);

  std::array<Number, dim> res;
  std::fill(res.begin(), res.end(), std::numeric_limits<Number>::min());

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_pressure.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values.get_function_values(pres_s.front(), solution_values_pressure);
      fe_values.get_function_values(rho_s.front(), solution_values_density);

      for(unsigned q = 0; q < n_q_points; ++q) {
        auto local_celerity = std::sqrt(gamma*(solution_values_pressure[q]/solution_values_density[q]));
        for(unsigned d = 0; d < dim; ++d) {
          res[d] = std::max(res[d],
                            inv_Ma*EquationData::degree_u*(local_celerity*dt/
                            static_cast<Number>(cell->extent_in_direction(d))));
        }
      }
    }
  }

  for(unsigned d = 0; d < dim; ++d) {
    res[d] = Utilities::MPI::max(res[d], MPI_COMM_WORLD);
  }

  return res;
}


//////////////////////////////////////////////////////////////
/*---- ROUTINE THAT EFFECTIVELY PERFORMS THE TIME LOOP ---*/
/////////////////////////////////////////////////////////////

// @sect{ <code>EulerSolver::run</code> }

// This is the time marching function, which starting at <code>t0</code>
// advances in time using the projection method with time step <code>dt</code>
// until <code>T</code>.
//
// Its second parameter, <code>verbose</code> indicates whether the function
// should output information what it is doing at any given moment:
// we use the ConditionalOStream class to do that for us.
//
template<unsigned dim>
void EulerSolver<dim>::run(const bool verbose,
                           const unsigned output_interval,
                           const std::string& n_files_,
                           const std::string& dt_save_) {
  ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  /*--- Initialize and save initial state ---*/
  auto time  = t0;
  unsigned n = 0;

  if(restart && !as_initial_conditions) {
    n    = step_restart;
    time = time_restart;
  }
  else {
    output_results(n);
  }

  /*--- Set some potential parameters for time saving ---*/
  if(!n_files_.empty() && !dt_save_.empty()) {
    std::cerr << "Both number of files and time-interval saving not empty. Pick one!" << std::endl;
    exit(1);
  }
  Number   dt_save; // After how much time save each file (potentially unsued)
  unsigned n_saved = 0; // Number of files saved (potentially unused)
  unsigned n_files = 0; // Number of files to be saved (potentially unused)
  if(!n_files_.empty()) {
    n_files = std::stoi(n_files_);
    dt_save = T/static_cast<Number>(n_files);
  }
  else if(!dt_save_.empty()) {
    dt_save = static_cast<Number>(std::stod(dt_save_));
  }

  /*--- Time loop ---*/
  auto tot_fixed_point_iters = static_cast<Number>(0.0);
  if(dt_from_CFL) {
    dt = CFL*h_min/(get_max_velocity()*EquationData::degree_u);
    euler_matrix.set_dt(dt);
  }
  while(std::abs(T - time) > static_cast<Number>(1e-10)) {
    if(!dt_save_.empty()) {
      if(time + dt > (n_saved + 1)*dt_save) {
        auto dt_tmp = (n_saved + 1)*dt_save - time;
        euler_matrix.set_dt(dt_tmp);
        time += dt_tmp;
      }
      else {
        time += dt;
      }
    }
    else {
      time += dt;
    }
    n++;
    pcout << "Step = " << n << " Time = " << time << std::endl;

    /*--- Internal stage of IMEX scheme (type II) ---*/
    for(IMEX_stage = 2; IMEX_stage <= n_stages; ++IMEX_stage) {
      euler_matrix.set_IMEX_stage(IMEX_stage);

      verbose_cout << "  Update density stage " << IMEX_stage << std::endl;
      update_density();
      pcout << "Minimum density " << get_min_density() << std::endl;
      pcout << "Maximum density " << get_max_density() << std::endl;

      verbose_cout << "  Fixed point pressure stage " << IMEX_stage << std::endl;
      // Set the current density to the operator
      euler_matrix.set_rho_for_fixed(rho_s[IMEX_stage - 1]);
      pres_fixed.equ(static_cast<Number>(1.0), pres_s[IMEX_stage - 2]);
      u_fixed.equ(static_cast<Number>(1.0), u_s[IMEX_stage - 2]);
      const auto iter = perform_fixed_point_loop();
      tot_fixed_point_iters += (iter + 1);
      // Assign the fields after the fixed point loop
      pres_s[IMEX_stage - 1].equ(static_cast<Number>(1.0), pres_fixed);
      u_s[IMEX_stage - 1].equ(static_cast<Number>(1.0), u_fixed);
    }

    /*--- Final stage of RK scheme to update ---*/
    IMEX_stage = n_stages + 1;
    euler_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update density" << std::endl;
    update_density();
    pcout << "Minimum density " << get_min_density() << std::endl;
    pcout << "Maximum density " << get_max_density() << std::endl;

    verbose_cout << "  Update velocity" << std::endl;
    // Set the current density to the operator
    euler_matrix.set_rho_for_fixed(rho_s.back());
    update_velocity();

    verbose_cout << "  Update pressure" << std::endl;
    update_pressure();

    /*--- Update before turbulent operator. Compute also potential temperataure ---*/
    rho_s.front().equ(static_cast<Number>(1.0), rho_s.back());
    u_s.front().equ(static_cast<Number>(1.0), u_s.back());
    for(const auto& cell: dof_handler_pressure.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        std::vector<types::global_dof_index> dof_indices(fe_pressure.dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
          const auto pres = pres_s.front()(dof_indices[idx]);
          const auto T    = pres/rho_s.front()(dof_indices[idx]);
          const auto Pi   = std::pow(pres, Gamma);
          theta_s.front()(dof_indices[idx]) = T/Pi;
        }
      }
    }

    /*--- Stages of stiffly-acurate ESDIRK scheme (type II) ---*/
    for(IMEX_stage = 2; IMEX_stage <= n_stages; ++IMEX_stage) {
      turbulent_matrix.set_IMEX_stage(IMEX_stage);

      verbose_cout << "  Update velocity stage " << IMEX_stage << " turbulent" << std::endl;
      turbulent_matrix.set_u_curr(u_s[IMEX_stage - 2]);
      turbulent_matrix.set_theta_curr(theta_s[IMEX_stage - 2]);
      diffusion_step();

      verbose_cout << "  Update potential temperature stage " << IMEX_stage << " turbulent" << std::endl;
      temperature_step();
    }

    /*--- Update before applying damping layer ---*/
    u_s.front().equ(static_cast<Number>(1.0), u_s[IMEX_stage - 1]);
    for(const auto& cell: dof_handler_pressure.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        std::vector<types::global_dof_index> dof_indices(fe_pressure.dofs_per_cell);
        cell->get_dof_indices(dof_indices);
        for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
          pres_s.front()(dof_indices[idx]) = std::pow(rho_s.front()(dof_indices[idx])*
                                                      theta_s.back()(dof_indices[idx]),
                                                      gamma);
        }
      }
    }

    /*--- Apply the damping layer for the vertical component ---*/
    rho_s.front().add(static_cast<Number>(1.0), dt_tau_rho);
    rho_s.front().scale(dt_tau_rho_aux);
    u_s.front().add(static_cast<Number>(1.0), dt_tau_u);
    u_s.front().scale(dt_tau_u_aux);
    pres_s.front().add(static_cast<Number>(1.0), dt_tau_pres);
    pres_s.front().scale(dt_tau_pres_aux);

    /*--- Apply the damping layer for the right lateral part ---*/
    rho_s.front().add(static_cast<Number>(1.0), dt_tau_rho_right);
    rho_s.front().scale(dt_tau_rho_aux_right);
    u_s.front().add(static_cast<Number>(1.0), dt_tau_u_right);
    u_s.front().scale(dt_tau_u_aux_right);
    pres_s.front().add(static_cast<Number>(1.0), dt_tau_pres_right);
    pres_s.front().scale(dt_tau_pres_aux_right);

    /*--- Apply the damping layer for the left lateral part ---*/
    rho_s.front().add(static_cast<Number>(1.0), dt_tau_rho_left);
    rho_s.front().scale(dt_tau_rho_aux_left);
    u_s.front().add(static_cast<Number>(1.0), dt_tau_u_left);
    u_s.front().scale(dt_tau_u_aux_left);
    pres_s.front().add(static_cast<Number>(1.0), dt_tau_pres_left);
    pres_s.front().scale(dt_tau_pres_aux_left);

    /*--- Apply the damping layer for the right y lateral part ---*/
    rho_s.front().add(static_cast<Number>(1.0), dt_tau_rho_right_y);
    rho_s.front().scale(dt_tau_rho_aux_right_y);
    u_s.front().add(static_cast<Number>(1.0), dt_tau_u_right_y);
    u_s.front().scale(dt_tau_u_aux_right_y);
    pres_s.front().add(static_cast<Number>(1.0), dt_tau_pres_right_y);
    pres_s.front().scale(dt_tau_pres_aux_right_y);

    /*--- Apply the damping layer for the left y lateral part ---*/
    rho_s.front().add(static_cast<Number>(1.0), dt_tau_rho_left_y);
    rho_s.front().scale(dt_tau_rho_aux_left_y);
    u_s.front().add(static_cast<Number>(1.0), dt_tau_u_left_y);
    u_s.front().scale(dt_tau_u_aux_left_y);
    pres_s.front().add(static_cast<Number>(1.0), dt_tau_pres_left_y);
    pres_s.front().scale(dt_tau_pres_aux_left_y);

    /*--- Compute auxiliary post-processing data ---*/
    const auto max_celerity = inv_Ma*compute_max_celerity();
    pcout<< "Maximum celerity = " << max_celerity << std::endl;
    pcout << "CFL_c = " << EquationData::degree_u*(max_celerity*dt/h_min) << std::endl;
    const auto max_C_x_y_z = compute_max_C_per_direction();
    pcout << "CFL_c_x = " << max_C_x_y_z[0] << std::endl;
    pcout << "CFL_c_y = " << max_C_x_y_z[1] << std::endl;
    pcout << "CFL_c_z = " << max_C_x_y_z[2] << std::endl;
    const auto max_velocity = get_max_velocity();
    pcout<< "Maximum velocity = " << max_velocity << std::endl;
    pcout << "CFL_u = " << EquationData::degree_u*(max_velocity*dt/h_min) << std::endl;
    const auto max_Cu_x_y_z = compute_max_Cu_per_direction();
    pcout << "CFL_u_x = " << max_Cu_x_y_z[0] << std::endl;
    pcout << "CFL_u_y = " << max_Cu_x_y_z[1] << std::endl;
    pcout << "CFL_u_z = " << max_Cu_x_y_z[2] << std::endl;

    /*--- Recompute time step if needed ---*/
    if(dt_from_CFL) {
      dt = CFL*h_min/(max_velocity*EquationData::degree_u);
      euler_matrix.set_dt(dt);
    }

    /*--- Save the results ---*/
    if(n_files_.empty() && dt_save_.empty()) {
      if(n % output_interval == 0) {
        verbose_cout << "Plotting solution" << std::endl;
        output_results(n);
      }
    }
    else if(!dt_save_.empty()) {
      if(std::abs(time - (n_saved + 1)*dt_save) < static_cast<Number>(1e-10)) {
        verbose_cout << "Plotting solution" << std::endl;
        output_results(n);
        ++n_saved;
        euler_matrix.set_dt(dt);
      }
    }
    else {
      if(time >= static_cast<Number>(n_saved + 1)*dt_save) {
        verbose_cout << "Plotting solution" << std::endl;
        output_results(n);
        ++n_saved;
      }
    }

    if(time + dt > T && T - time > static_cast<Number>(1e-10)) {
      /*--- Recompute and reset the time if needed towards the end of the simulation to stop at the proper final time ---*/
      dt = T - time;
      euler_matrix.set_dt(dt);
    }
  }

  /*--- Save the final results if not previously done ---*/
  pcout << "Average fixed point iterations: "
        << static_cast<Number>(tot_fixed_point_iters)/((n_stages - 1)*n)
        << std::endl;
  if(n_files_.empty() && dt_save_.empty()) {
    if(n % output_interval != 0) {
      verbose_cout << "Plotting final solution" << std::endl;
      output_results(n);
    }
  }
  else if(!dt_save_.empty()) {
    if(time < (n_saved + 1)*dt_save) {
      verbose_cout << "Plotting final solution" << std::endl;
      output_results(n);
    }
  }
  else {
    if(n_saved < n_files) {
      verbose_cout << "Plotting final solution" << std::endl;
      output_results(n);
    }
  }
}


//////////////////////////////////////////////////////////////
/*---- MAIN FUNCTION ---*/
/////////////////////////////////////////////////////////////

// @sect{ The main function }

// The main function is quite standard. We just need to declare the EulerSolver
// instance and let the simulation run. Include also a help message
//
void print_help(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n\n"
            << "Options:\n"
            << "  -p, --param FILE     Parameter file to read\n"
            << "  -h, --help           Show this help message\n\n"
            << "Default parameter file: parameter-file.prm\n";
}

int main(int argc, char *argv[]) {
  try {
    /*--- Read the parameters ---*/
    std::string parameter_file = "parameter-file.prm";
    for(int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      if(arg == "-h" || arg == "--help") {
        print_help(argv[0]);
        return 0;
      }
      else if(arg == "-p" || arg == "--param") {
        if(i + 1 >= argc) {
          std::cerr << "Error: missing argument after " << arg << "\n";
          return 1;
        }
        parameter_file = argv[++i];
      }
      else {
        std::cerr << "Unknown option: " << arg << "\n";
        print_help(argv[0]);
        return 1;
      }
    }

    RunTimeParameters::Data_Storage data;
    data.read_data(parameter_file);

    /*-- Initialize console and output ---*/
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    /*--- Initilize Butcher tableaux. Now the declaration of the coefficients (hard-coded for the moment)
          is totally relegated here, so this is the only place where we need to modify (much cleaner). ---*/
    const unsigned n_stages = 3;
    using Number = typename EulerSolver<3>::Number;

    std::vector<std::vector<Number>> a(n_stages, std::vector<Number>(n_stages));
    std::vector<Number> b(n_stages);
    std::fill(a.begin(), a.end(),
              std::vector<Number>(n_stages, static_cast<Number>(0.0)));
    const auto chi = static_cast<Number>(2.0) - static_cast<Number>(std::sqrt(2.0));
    a[1][0] = chi;
    a[2][0] = static_cast<Number>(0.5);
    a[2][1] = static_cast<Number>(0.5);
    b[0]    = static_cast<Number>(0.5) - static_cast<Number>(0.25)*chi;
    b[1]    = static_cast<Number>(0.5) - static_cast<Number>(0.25)*chi;
    b[2]    = static_cast<Number>(0.5)*chi;
    TimeStepping::RungeKutta<Number> explicit_RK(a, b);

    std::vector<std::vector<Number>> a_tilde(n_stages, std::vector<Number>(n_stages));
    std::fill(a_tilde.begin(), a_tilde.end(),
              std::vector<Number>(n_stages, static_cast<Number>(0.0)));
    a_tilde[1][0] = static_cast<Number>(0.5)*chi;
    a_tilde[1][1] = static_cast<Number>(0.5)*chi;
    a_tilde[2][0] = static_cast<Number>(0.5) - static_cast<Number>(0.25)*chi;
    a_tilde[2][1] = static_cast<Number>(0.5) - static_cast<Number>(0.25)*chi;
    a_tilde[2][2] = static_cast<Number>(0.5)*chi;
    std::vector<Number> b_tilde = b;
    TimeStepping::RungeKutta<Number> implicit_RK(a_tilde, b_tilde);

    /*-- Run the simulation ---*/
    EulerSolver<3> test(data, explicit_RK, implicit_RK);
    test.run(data.verbose, data.output_interval, data.n_files, data.dt_save);

    if(curr_rank == 0) {
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;
    }

    return 0;

  }
  catch(std::exception& exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

}
