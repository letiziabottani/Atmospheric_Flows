/*--- Author: Giuseppe Orlando, 2026. ---*/
#pragma once

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones
//
#include <deal.II/base/parameter_handler.h>

#include <fstream>

// @sect{Run-time parameters}
//
// Since our method has several parameters that can be fine-tuned we put them
// into an external file, so that they can be determined at run-time.
//
namespace RunTimeParameters {
  using namespace dealii;

  class Data_Storage {
  public:
    Data_Storage(); /*--- Class constructor ---*/

    void read_data(const std::string& filename); /*--- The function that actually reads the parameters ---*/

    /*--- Start with physical parameters ---*/
    double initial_time; /*--- Variable to set the initial time (default equal to 0) ---*/
    double final_time;   /*--- Variable to set the final time ---*/

    double x_min; /*--- Left-end domain x direction ---*/
    double x_max; /*--- Right-end domain x direction ---*/
    double y_min; /*--- Left-end domain y direction ---*/
    double y_max; /*--- Right-end domain y direction ---*/
    double z_min; /*--- Left-end domain z direction ---*/
    double z_max; /*--- Right-end domain z direction ---*/

    // The present code is meant to work using non-dimensional variables and using
    // the non-dimensional equations described in Orlando et al., JCP, 2022.
    // If one wishes to consider a dimensional version, it is sufficient
    // to set the Mach numer equal to 1 and the Froude number equal to 1/sqrt(g),
    // where g is, as usual, the acceleration of gravity.
    //
    double Mach;   /*--- The Mach number ---*/
    double Froude; /*--- The Froude number ---*/
    double Rossby; /*--- The Rossby number ---*/

    double L_ref;   /*--- Reference length ---*/
    double u_ref;   /*--- Reference velocity ---*/
    double p_ref;   /*--- Reference pressure ---*/
    double T_ref;   /*--- Reference temperature ---*/
    double rho_ref; /*--- Reference density ---*/

    double lapse_rate; /*--- Lapse rate for the background temperature profile ---*/
    double b;          /*--- Parameter related to the width of the jet in the vertical direction ---*/
    double Omega;      /*--- Rotation rate ---*/
    double phi0;      /*--- Latitude for the f-plane approximation ---*/

    double up; /*--- Amplitude of the perturbation ---*/
    double Lp; /*--- Width of the perturbation ---*/
    double xc; /*--- x-coordinate of the perturbation ---*/
    double yc; /*--- y-coordinate of the perturbation ---*/

    double N; /*--- Buoyancy frequency ---*/

    double u_bar;   /*--- Reference background velocity ---*/
    double p_bar;   /*--- Reference background pressure ---*/
    double T_bar;   /*--- Reference background temperature ---*/
    double rho_bar; /*--- Reference background density ---*/

    /*--- Numerical parameters ---*/
    unsigned degree_u;   /*--- Polynomial degree for the velocity (not used so far) ---*/
    unsigned degree_rho; /*--- Polynomial degree for the density (not used so far) ---*/
    unsigned degree_p;   /*--- Polynomial degree for the pressure (not used so far) ---*/

    double dt;       /*--- The time-step ---*/
    std::string CFL; /*--- The Courant number (declare as string so as to verify if empty or not) ---*/

    double z_start;  /*--- Start of Rayleigh damping for top boundary ---*/
    double lambda_z; /*--- Intensity of Rayleigh damping for top boundary ---*/

    double x_start_left;  /*--- Start of Rayleigh damping for left boundary ---*/
    double lambda_x_left; /*--- Intensity of Rayleigh damping for left boundary ---*/

    double x_start_right;  /*--- Start of Rayleigh damping for right boundary ---*/
    double lambda_x_right; /*--- Intensity of Rayleigh damping for right boundary ---*/

    double y_start_left;  /*--- Start of Rayleigh damping for y left boundary ---*/
    double lambda_y_left; /*--- Intensity of Rayleigh damping for y left boundary ---*/

    double y_start_right;  /*--- Start of Rayleigh damping for y right boundary ---*/
    double lambda_y_right; /*--- Intensity of Rayleigh damping for y right boundary ---*/

    double atol_fixed_point; /*--- Absolute tolerance for the fixed point loop ---*/
    double rtol_fixed_point; /*--- Relative tolerance for the fixed point loop ---*/

    double l_mixing; /*--- Mixing length (in the case of turbulent simulations) ---*/

    /*--- Mesh parameters ---*/
    unsigned n_elements_x;     /*--- Number of (initial) elements along x direction ---*/
    unsigned n_elements_y;     /*--- Number of (initial) elements along y direction ---*/
    unsigned n_elements_z;     /*--- Number of (initial) elements along z direction ---*/
    unsigned n_global_refines; /*--- Number of global refinements for the initial (coarse) mesh ---*/

    unsigned degree_mapping; /*--- Degree of mapping for curved boundary (not used so far) ---*/

    unsigned max_loc_refinements;   /*--- Maximum number of refinements allowed ---*/
    unsigned min_loc_refinements;   /*--- Minimum number of refinements allowed ---*/
    unsigned refinement_iterations; /*--- How often performing mesh adaptation ---*/

    /*--- Parameters related to the linear solver ---*/
    unsigned max_iterations; /*--- Maximum number of iterations for the linear solver ---*/
    double   atol_iterative; /*--- Absolute tolerance for the linear solver ---*/
    double   rtol_iterative; /*--- Relative tolerance for the linear solver ---*/

    /*--- Parameters related to the output ---*/
    bool        verbose;         /*--- Choose if being verboe or not ---*/
    unsigned    output_interval; /*--- Set how often save the fields ---*/
    std::string n_files;         /*--- Set how often save the fields through the number of output files (potentially unused) ---*/
    std::string dt_save;         /*--- Set after how much time perfoming the save (potentially unused) ---*/

    std::string dir; /*--- Directory where the data are saved. This has to be created before launching the code
                           and we assume it is a subfolder of the folder with the executable and the parameter file.
                           This behaviour can be easily changed giving, e.g., the absolute path ---*/

    /*--- Auxiliary parameters related to restart ---*/
    bool     restart;
    bool     save_for_restart;
    unsigned step_restart;
    double   time_restart;
    bool     as_initial_conditions;

  protected:
    ParameterHandler prm; /*--- Auxiliary variable which handles the parameters ---*/
  };

  // In the constructor of this class we declare all the parameters.
  // We employ the 'enter_subsection' to divide into categories and
  // the 'declare_entry' to declare a certain parameter to be setted.
  //
  Data_Storage::Data_Storage(): initial_time(0.0),
                                final_time(1.0),
                                x_min(0.0),
                                x_max(1.0),
                                y_min(0.0),
                                y_max(1.0),
                                z_min(0.0),
                                z_max(1.0),
                                Mach(1.0),
                                Froude(0.319275428407050),
                                Rossby(1.0),
                                L_ref(1.0),
                                u_ref(1.0),
                                p_ref(1.0),
                                T_ref(1.0),
                                rho_ref(1.0),
                                N(0.01),
                                lapse_rate(0.0065),
                                b(1.0),
                                Omega(7.2921e-5),
                                phi0(45.0),
                                up(1.0),
                                Lp(600000.0),
                                xc(2000000.0),
                                yc(2500000.0),
                                u_bar(1.0),
                                p_bar(1.0),
                                T_bar(1.0),
                                rho_bar(1.0),
                                degree_u(1),
                                degree_rho(1),
                                degree_p(1),
                                dt(5e-4),
                                CFL(""),
                                z_start(1.0),
                                lambda_z(1.0),
                                x_start_left(0.0),
                                lambda_x_left(1.0),
                                x_start_right(1.0),
                                lambda_x_right(1.0),
                                y_start_left(0.0),
                                lambda_y_left(1.0),
                                y_start_right(1.0),
                                lambda_y_right(1.0),
                                atol_fixed_point(1e-12),
                                rtol_fixed_point(1e-10),
                                l_mixing(1.0),
                                n_elements_x(1),
                                n_elements_y(1),
                                n_elements_z(1),
                                n_global_refines(0),
                                degree_mapping(1),
                                max_loc_refinements(0),
                                min_loc_refinements(0),
                                refinement_iterations(0),
                                max_iterations(1000),
                                atol_iterative(1e-14),
                                rtol_iterative(1e-12),
                                verbose(true),
                                output_interval(15),
                                n_files(""),
                                dt_save(""),
                                dir(""),
                                restart(false),
                                save_for_restart(false),
                                step_restart(0),
                                time_restart(0.0),
                                as_initial_conditions(false) {
    /*--- Start declaring entries for the physical parameters ---*/
    prm.enter_subsection("Physical data");
    {
      prm.declare_entry("initial_time",
                        "0.0",
                        Patterns::Double(0.0),
                        "The initial time of the simulation.");
      prm.declare_entry("final_time",
                        "1.0",
                        Patterns::Double(0.0),
                        "The final time of the simulation.");

      prm.declare_entry("x_min",
                        "0.0",
                        Patterns::Double(0.0),
                        "The left-end of the domain along x-direction.");
      prm.declare_entry("x_max",
                        "1.0",
                        Patterns::Double(0.0),
                        "The right-end of the domain along x-direction.");
      prm.declare_entry("y_min",
                        "0.0",
                        Patterns::Double(0.0),
                        "The left-end of the domain along y-direction.");
      prm.declare_entry("y_max",
                        "1.0",
                        Patterns::Double(0.0),
                        "The right-end of the domain along y-direction.");
      prm.declare_entry("z_min",
                        "0.0",
                        Patterns::Double(0.0),
                        "The left-end of the domain along z-direction.");
      prm.declare_entry("z_max",
                        "1.0",
                        Patterns::Double(0.0),
                        "The right-end of the domain along z-direction.");

      prm.declare_entry("Mach",
                        "1.0",
                        Patterns::Double(0.0),
                        " The Mach number.");
      prm.declare_entry("Froude",
                        "0.319275428407050",
                        Patterns::Double(0.0),
                        "The Froude number.");
      prm.declare_entry("Rossby",
                        "100.0",
                        Patterns::Double(0.0),
                        "The Rossby number.");

      prm.declare_entry("L_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference length.");
      prm.declare_entry("u_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference velocity.");
      prm.declare_entry("p_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference pressure.");
      prm.declare_entry("T_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference temperature.");
      prm.declare_entry("rho_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference density.");

     /* prm.declare_entry("h",
                        "1.0",
                        Patterns::Double(0.0),
                        "The hill height.");
      prm.declare_entry("xc",
                        "1.0",
                        Patterns::Double(0.0),
                        "The x-Center of the hill.");
      prm.declare_entry("yc",
                        "1.0",
                        Patterns::Double(0.0),
                        "The y-Center of the hill.");
      prm.declare_entry("ac",
                        "1.0",
                        Patterns::Double(0.0),
                        "The width of the hill.");*/

      prm.declare_entry("N",
                        "0.01",
                        Patterns::Double(0.0),
                        "Buoyancy frequency.");

                        prm.declare_entry("lapse_rate",
                        "0.0065",
                        Patterns::Double(0.0),
                        "Lapse rate for the background temperature profile.");
      prm.declare_entry("b",
                        "1.0",
                        Patterns::Double(0.0),
                        "Parameter related to the width of the jet in the vertical direction.");
      prm.declare_entry("Omega",
                        "7.2921e-5",
                        Patterns::Double(0.0),
                        "Rotation rate.");
      prm.declare_entry("phi0",
                        "45.0",
                        Patterns::Double(0.0),
                        "Latitude for the f-plane approximation."); 
      prm.declare_entry("up",
                        "1.0",
                        Patterns::Double(0.0),
                        "Amplitude of the perturbation.");
      prm.declare_entry("Lp",
                        "600000.0",
                        Patterns::Double(0.0),
                        "Width of the perturbation.");
      prm.declare_entry("xc",
                        "2000000.0",
                        Patterns::Double(0.0),
                        "x-coordinate of the perturbation.");
      prm.declare_entry("yc",
                        "2500000.0",
                        Patterns::Double(0.0),
                        "y-coordinate of the perturbation.");
      prm.declare_entry("u_bar",
                        "1.0",
                        Patterns::Double(0.0),
                        "The (horizontal) background.");
      prm.declare_entry("p_bar",
                        "1.0",
                        Patterns::Double(0.0),
                        "The background pressure (at z = 0).");
      prm.declare_entry("T_bar",
                        "1.0",
                        Patterns::Double(0.0),
                        "The background temperature (at z = 0).");
      prm.declare_entry("rho_bar",
                        "1.0",
                        Patterns::Double(0.0),
                        "The background density (at z = 0).");
    }
    prm.leave_subsection();

    /*--- Focus now on some numerical parameters ---*/
    prm.enter_subsection("Numerical data");
    {
      prm.declare_entry("degree_u",
                        "1",
                        Patterns::Integer(0, 15),
                        "Polynomial degree for the velocity.");
      prm.declare_entry("degree_rho",
                        "1",
                        Patterns::Integer(0, 15),
                        "Polynomial degree for the density.");
      prm.declare_entry("degree_p",
                        "1",
                        Patterns::Integer(0, 15),
                        "Polynomial degree for the pressure.");

      prm.declare_entry("dt",
                        "5e-4",
                        Patterns::Double(0.0),
                        "The time step size.");
      prm.declare_entry("CFL", "");

      prm.declare_entry("z_start",
                        "1.0",
                        Patterns::Double(0.0),
                        "Start of Rayleigh damping for top boundary.");
      prm.declare_entry("lambda_z",
                        "1.0",
                        Patterns::Double(0.0),
                        "Intensity of Rayleigh damping for top boundary.");
      prm.declare_entry("x_start_left",
                        "0.0",
                        Patterns::Double(0.0),
                        "Start of Rayleigh damping for left boundary.");
      prm.declare_entry("lambda_x_left",
                        "1.0",
                        Patterns::Double(0.0),
                        "Intensity of Rayleigh damping for left boundary.");
      prm.declare_entry("x_start_right",
                        "1.0",
                        Patterns::Double(0.0),
                        "Start of Rayleigh damping for right boundary.");
      prm.declare_entry("lambda_x_right",
                        "1.0",
                        Patterns::Double(0.0),
                        "Intensity of Rayleigh damping for right boundary.");
      prm.declare_entry("y_start_left",
                        "0.0",
                        Patterns::Double(0.0),
                        "Start of Rayleigh damping for y left boundary.");
      prm.declare_entry("lambda_y_left",
                        "1.0",
                        Patterns::Double(0.0),
                        "Intensity of Rayleigh damping for y left boundary.");
      prm.declare_entry("y_start_right",
                        "1.0",
                        Patterns::Double(0.0),
                        "Start of Rayleigh damping for y right boundary.");
      prm.declare_entry("lambda_y_right",
                        "1.0",
                        Patterns::Double(0.0),
                        "Intensity of Rayleigh damping for y right boundary.");

      prm.declare_entry("atol_fixed_point",
                        "1e-12",
                        Patterns::Double(0.0),
                        "Absolute tolerance for the fixed point loop.");
      prm.declare_entry("rtol_fixed_point",
                        "1e-10",
                        Patterns::Double(0.0),
                        "Relative tolerance for the fixed point loop.");

      prm.declare_entry("l_mixing",
                        "1.0",
                        Patterns::Double(0.0),
                        "Mixing length in the case of turbulent simulations.");
    }
    prm.leave_subsection();

    /*--- Focus now on some mesh parameters ---*/
    prm.enter_subsection("Mesh parameters");
    {
      prm.declare_entry("n_elements_x",
                        "1",
                        Patterns::Integer(0, 100000000),
                        "The number of (initial) elements along x direction.");
      prm.declare_entry("n_elements_y",
                        "1",
                        Patterns::Integer(0, 100000000),
                        "The number of (initial) elements along y direction.");
      prm.declare_entry("n_elements_z",
                        "1",
                        Patterns::Integer(0, 100000000),
                        "The number of (initial) elements along z direction.");
      prm.declare_entry("n_of_refines",
                        "3",
                        Patterns::Integer(0, 15),
                        "The number of global refinements we want for the mesh.");

      prm.declare_entry("degree_mapping",
                        "1",
                         Patterns::Integer(1, 15),
                         "Polynomial degree mapping curved boundary.");

      prm.declare_entry("max_loc_refinements",
                        "4",
                         Patterns::Integer(1, 10),
                         " The number of maximum local refinements in case of adaptive mesh.");
      prm.declare_entry("min_loc_refinements",
                        "2",
                         Patterns::Integer(0, 10),
                         " The number of minimum local refinements in case of adaptive mesh.");
      prm.declare_entry("refinement_iterations",
                        "0",
                         Patterns::Integer(0, 100000000),
                         "How ofter performing mesh adaptation if desired.");
    }
    prm.leave_subsection();

    /*--- Focus now on the data of the linear solvers ---*/
    prm.enter_subsection("Data linear solvers");
    {
      prm.declare_entry("max_iterations",
                        "1000",
                        Patterns::Integer(1, 30000),
                        "The maximal number of iterations GMRES must make.");
      prm.declare_entry("atol_iterative",
                        "1e-14",
                        Patterns::Double(0.0),
                        "Absolute tolerance for the linear solver.");
      prm.declare_entry("rtol_iterative",
                        "1e-12",
                        Patterns::Double(0.0),
                        "Relative tolerance for the linear solver.");
    }
    prm.leave_subsection();

    /*--- Focus now on the restart parameters ---*/
    prm.enter_subsection("Restart data");
    {
      prm.declare_entry("time_restart",
                        "5e-4",
                        Patterns::Double(0.0),
                        "The time of restart.");
      prm.declare_entry("step_restart",
                        "0",
                         Patterns::Integer(0, 100000000),
                         "The step at which restart occurs.");
      prm.declare_entry("restart",
                        "false",
                        Patterns::Bool(),
                        "This indicates whether we are in presence of a "
                        "restart or not.");
      prm.declare_entry("save_for_restart",
                        "false",
                        Patterns::Bool(),
                        "This indicates whether we want to save for possible "
                        "restart or not.");
      prm.declare_entry("as_initial_conditions",
                        "false",
                        Patterns::Bool(),
                        "This indicates whether restart is used as initial condition "
                        "or to continue the simulation.");
    }
    prm.leave_subsection();

    /*--- Output related parameters ---*/
    prm.enter_subsection("Output data");
    {
      prm.declare_entry("verbose",
                        "true",
                        Patterns::Bool(),
                        "This indicates whether the output of the solution "
                        "process should be verbose.");

      prm.declare_entry("output_interval",
                        "1",
                        Patterns::Integer(1),
                        "This indicates between how many time steps we print "
                        "the solution.");
      prm.declare_entry("n_files", "");
      prm.declare_entry("dt_save", "");

      prm.declare_entry("saving directory", "SimTest");
    }
    prm.leave_subsection();
  }

  // Function to read all declared parameters in the constructor
  //
  void Data_Storage::read_data(const std::string& filename) {
    std::ifstream file(filename);
    AssertThrow(file, ExcFileNotOpen(filename));

    prm.parse_input(file);

    /*--- Start with physical related parameters ---*/
    prm.enter_subsection("Physical data");
    {
      initial_time = prm.get_double("initial_time");
      final_time   = prm.get_double("final_time");

      x_min = prm.get_double("x_min");
      x_max = prm.get_double("x_max");
      y_min = prm.get_double("y_min");
      y_max = prm.get_double("y_max");
      z_min = prm.get_double("z_min");
      z_max = prm.get_double("z_max");

      Mach   = prm.get_double("Mach");
      Froude = prm.get_double("Froude");
      Rossby = prm.get_double("Rossby");

      L_ref   = prm.get_double("L_ref");
      u_ref   = prm.get_double("u_ref");
      p_ref   = prm.get_double("p_ref");
      T_ref   = prm.get_double("T_ref");
      rho_ref = prm.get_double("rho_ref");
/*
      h  = prm.get_double("h");
      xc = prm.get_double("xc");
      yc = prm.get_double("yc");
      ac = prm.get_double("ac");
*/
      N = prm.get_double("N");

      u_bar   = prm.get_double("u_bar");
      p_bar   = prm.get_double("p_bar");
      T_bar   = prm.get_double("T_bar");
      rho_bar = prm.get_double("rho_bar");
    }
    prm.leave_subsection();

    /*--- Focus now on some numerical parameters ---*/
    prm.enter_subsection("Numerical data");
    {
      degree_u   = prm.get_integer("degree_u");
      degree_rho = prm.get_integer("degree_rho");
      degree_p   = prm.get_integer("degree_p");

      dt  = prm.get_double("dt");
      CFL = prm.get("CFL");

      z_start        = prm.get_double("z_start");
      lambda_z       = prm.get_double("lambda_z");
      x_start_left   = prm.get_double("x_start_left");
      lambda_x_left  = prm.get_double("lambda_x_left");
      x_start_right  = prm.get_double("x_start_right");
      lambda_x_right = prm.get_double("lambda_x_right");
      y_start_left   = prm.get_double("y_start_left");
      lambda_y_left  = prm.get_double("lambda_y_left");
      y_start_right  = prm.get_double("y_start_right");
      lambda_y_right = prm.get_double("lambda_y_right");

      atol_fixed_point = prm.get_double("atol_fixed_point");
      rtol_fixed_point = prm.get_double("rtol_fixed_point");

      l_mixing = prm.get_double("l_mixing");
    }
    prm.leave_subsection();

    /*--- Focus now on some mesh parameters ---*/
    prm.enter_subsection("Mesh parameters");
    {
      n_elements_x     = prm.get_integer("n_elements_x");
      n_elements_y     = prm.get_integer("n_elements_y");
      n_elements_z     = prm.get_integer("n_elements_z");
      n_global_refines = prm.get_integer("n_of_refines");

      degree_mapping = prm.get_integer("degree_mapping");

      max_loc_refinements   = prm.get_integer("max_loc_refinements");
      min_loc_refinements   = prm.get_integer("min_loc_refinements");
      refinement_iterations = prm.get_integer("refinement_iterations");
    }
    prm.leave_subsection();

    /*--- Focus now on the data of the linear solvers ---*/
    prm.enter_subsection("Data linear solvers");
    {
      max_iterations = prm.get_integer("max_iterations");
      atol_iterative = prm.get_double("atol_iterative");
      rtol_iterative = prm.get_double("rtol_iterative");
    }
    prm.leave_subsection();

    /*--- Read parameters related to restart ---*/
    prm.enter_subsection("Restart data");
    {
      time_restart          = prm.get_double("time_restart");
      step_restart          = prm.get_integer("step_restart");
      restart               = prm.get_bool("restart");
      save_for_restart      = prm.get_bool("save_for_restart");
      as_initial_conditions = prm.get_bool("as_initial_conditions");
    }
    prm.leave_subsection();

    /*--- Output related data ---*/
    prm.enter_subsection("Output data");
    {
      verbose = prm.get_bool("verbose");

      output_interval = prm.get_integer("output_interval");
      n_files         = prm.get("n_files");
      dt_save         = prm.get("dt_save");

      dir = prm.get("saving directory");
    }
  }

} // namespace RunTimeParameters