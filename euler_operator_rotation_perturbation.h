/* Author: Giuseppe Orlando, 2025. */
#pragma once

// @sect{Include files}

// We start by including all the necessary deal.II header files
//
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/meshworker/mesh_loop.h>

/*--- Include headers related to the problem of interest ---*/
#include "include/io/runtime_parameters.h"
#include "include/equation_data.h"
#include "include/space_discretization/numerical_flux/Rusanov_flux.h"
#include "include/time_integrator/runge_kutta.h"

// This is the class that implements the discretization
//
namespace Atmospheric_Flow {
  using namespace dealii;

  // @sect{ <code>EULEROperator::EULEROperator</code> }
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  class EULEROperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    EULEROperator(); /*--- Default constructor ---*/

    EULEROperator(const RunTimeParameters::Data_Storage& data,
                  const TimeStepping::RungeKutta<Number>& explicit_RK,
                  const TimeStepping::RungeKutta<Number>& implicit_RK); /*--- Constructor with some input related data ---*/

    template<typename T>
    inline DEAL_II_ALWAYS_INLINE
    void set_dt(const T time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                          in case of modifications of the time step. ---*/

    inline DEAL_II_ALWAYS_INLINE
    Number get_Mach() const; /*--- Getter of the Mach number. This is useful for debugging purpose. ---*/

    inline DEAL_II_ALWAYS_INLINE
    Number get_Froude() const; /*--- Getter of the Froude number. This is useful for debugging purpose. ---*/

    inline DEAL_II_ALWAYS_INLINE
    Number get_Rossby() const; /*--- Getter of the Rossby number. This is useful for debugging purpose. ---*/

    inline DEAL_II_ALWAYS_INLINE
    void set_IMEX_stage(const unsigned stage); /*--- Setter of the IMEX stage. ---*/

    inline DEAL_II_ALWAYS_INLINE
    void set_Euler_stage(const unsigned stage); /*--- Setter of the equation currently under solution. ---*/

    inline DEAL_II_ALWAYS_INLINE
    unsigned get_Euler_stage() const; /*--- Getter of the equation currently under solution. ---*/

    void set_rho_for_fixed(const Vec& src); /*--- Setter of the current density. This is for the assembling of the bilinear forms
                                                  where only one source vector can be passed in input. ---*/

    void set_pres_fixed(const Vec& src); /*--- Setter of the current pressure. This is for the assembling of the bilinear forms
                                               where only one source vector can be passed in input. ---*/

    void vmult_rhs_density(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                               of the continuity equation. ---*/

    void vmult_rhs_momentum(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                                 of the momentum equation. ---*/

    void vmult_rhs_energy(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                               of the energy equation. ---*/

    void vmult_pressure(Vec& dst, const Vec& src) const; /*--- Action of matrix 'B'. ---*/

    void vmult_enthalpy(Vec& dst, const Vec& src) const; /*--- Action of matrix 'C'. ---*/

    virtual void compute_diagonal() override; /*--- Overriden function to compute the diagonal. ---*/

    void set_background_velocity(const Vec &u_background) {
      u_bar = u_background;
    }                                                         /*--- Setter for the background velocity field. ---*/


  protected:
    /*--- Define typedef for sake of readability and convenience ----*/
    using FEEvaluation_rho  = FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number>;
    using FEEvaluation_u    = FEEvaluation<dim, fe_degree_u, n_q_points_1d, 3, Number>;
    using FEEvaluation_pres = FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>;

    using FEFaceEvaluation_rho  = FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number>;
    using FEFaceEvaluation_u    = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, 3, Number>;
    using FEFaceEvaluation_pres = FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>;

    using FEFaceEvaluation_rho_boundary  = FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_boundary, 1, Number>;
    using FEFaceEvaluation_u_boundary    = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_boundary, 3, Number>;
    using FEFaceEvaluation_pres_boundary = FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_boundary, 1, Number>;

    Number Ma; /*--- Mach number. ---*/
    Number Fr; /*--- Froude number. ---*/
    Number Ro; /*--- Rossby number. ---*/

    Number dt; /*--- Time step. ---*/

    /*--- The following variables follow the classical Butcher tableaux notation ---*/
    std::vector<std::vector<Number>> a;
    std::vector<std::vector<Number>> a_tilde;

    std::vector<Number> b;
    std::vector<Number> b_tilde;

    unsigned n_stages; /*--- Number of stages ---*/

    unsigned IMEX_stage;          /*--- Flag for the IMEX stage ---*/
    mutable unsigned Euler_stage; /*--- Flag for the equation actually considered ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    Vec rho_for_fixed,
        pres_fixed; /*--- Auxiliary vectors for fixed point loop ---*/

    Vec u_bar; /*--- Background velocity field ---*/

    /*--- Auxiliary function for the numerical flux ---*/
    NumericalFlux::RusanovFluxEuler<3, VectorizedArray<Number>> num_flux;

    Number Ma2;          /*--- Squared Mach number ---*/
    Number inv_Ma2;      /*--- Inverse of squared Mach number ---*/
    Number inv_Fr2;      /*--- Inverse of squared Froude number ---*/
    Number Ma2_ov_Fr2;   /*--- Mach squared over Froude squared ---*/
    Number gamma_m1;     /*--- gamma - 1 (gamma ratio specific heats) ---*/
    Number inv_gamma_m1; /*--- Inverse gamma - 1 ---*/
    Number inv_Gamma;    /*--- gamma/(gamma - 1) ---*/
    Number inv_Ro;       /*--- Inverse of Rossby number ---*/

    Tensor<1, 3, VectorizedArray<Number>> e_k; /*--- Unit normal vector along vertical direction ---*/
    Tensor<1, 3, Tensor<1, dim, VectorizedArray<Number>>> identity; /*--- Identity tensor ---*/
    Tensor<1, 3, VectorizedArray<Number>> tmp_diagonal_velocity; /*--- Auxiliary vector to compute the diagonal of the velocity matrix ----*/

    /*--- Assembler functions for the rhs related to the continuity equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_density(const MatrixFree<dim, Number>&       data,
                                        Vec&                                 dst,
                                        const std::vector<Vec>&              src,
                                        const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_rhs_face_term_density(const MatrixFree<dim, Number>&       data,
                                        Vec&                                 dst,
                                        const std::vector<Vec>&              src,
                                        const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_rhs_boundary_term_density(const MatrixFree<dim, Number>&       data,
                                            Vec&                                 dst,
                                            const std::vector<Vec>&              src,
                                            const std::pair<unsigned, unsigned>& face_range) const {}
                                               /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler function related to the bilinear form of the continuity equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_density(const MatrixFree<dim, Number>&       data,
                                    Vec&                                 dst,
                                    const Vec&                           src,
                                    const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the rhs related to the momentum equation. ---*/
    void assemble_rhs_cell_term_momentum(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const std::vector<Vec>&              src,
                                         const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_rhs_face_term_momentum(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const std::vector<Vec>&              src,
                                         const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_rhs_boundary_term_momentum(const MatrixFree<dim, Number>&       data,
                                             Vec&                                 dst,
                                             const std::vector<Vec>&              src,
                                             const std::pair<unsigned, unsigned>& face_range) const;

    /*--- Assembler function for the 'A' matrix. ---*/
    void assemble_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the 'B' matrix. ---*/
    void assemble_cell_term_pressure(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_face_term_pressure(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_boundary_term_pressure(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const Vec&                           src,
                                         const std::pair<unsigned, unsigned>& face_range) const;

    /*--- Assembler functions for the rhs of the energy equation. ---*/
    void assemble_rhs_cell_term_energy(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const std::vector<Vec>&              src,
                                       const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_rhs_face_term_energy(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const std::vector<Vec>&              src,
                                       const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_rhs_boundary_term_energy(const MatrixFree<dim, Number>&       data,
                                           Vec&                                 dst,
                                           const std::vector<Vec>&              src,
                                           const std::pair<unsigned, unsigned>& face_range) const {}
                                           /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler function for the 'D' matrix. ---*/
    void assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                            Vec&                                 dst,
                                            const Vec&                           src,
                                            const std::pair<unsigned, unsigned>& cell_range) const;

    void assemble_inverse_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                                    Vec&                                 dst,
                                                    const Vec&                           src,
                                                    const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler function for the 'C' matrix. ---*/
    void assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_face_term_enthalpy(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_boundary_term_enthalpy(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const Vec&                           src,
                                         const std::pair<unsigned, unsigned>& face_range) const {}
                                         /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler functions for the diagonal part of the matrix for the continuity equation. For compatibilty conditions,
          also face and boundary contributions have to be defined, even though they are empty. ---*/
    void assemble_diagonal_cell_term_density(const MatrixFree<dim, Number>&       data,
                                             Vec&                                 dst,
                                             const unsigned&                      src,
                                             const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the diagonal part of 'A' matrix. ---*/
    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                              Vec&                                 dst,
                                              const unsigned&                      src,
                                              const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the diagonal part of the ellptic operator associated to the Schur complement for the pressure. ---*/
    void assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&       data,
                                              Vec&                                 dst,
                                              const unsigned&                      src,
                                              const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the diagonal part of 'D' matrix. ---*/
    void assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                                     Vec&                                 dst,
                                                     const unsigned&                      src,
                                                     const std::pair<unsigned, unsigned>& cell_range) const;
  };


  //////////////////////////////////////////////////////////////
  /*---- START WITH CLASS CONSTRUCTORS ---*/
  /////////////////////////////////////////////////////////////

  // Default constructor
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  EULEROperator<dim,
                fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary,
                Vec>::
  EULEROperator():
    MatrixFreeOperators::Base<dim, Vec>(), Ma(), Fr(), Ro(), dt(),
    IMEX_stage(1), Euler_stage(1), num_flux(),
    Ma2(Ma*Ma), inv_Ma2(static_cast<Number>(1.0)/Ma2), inv_Fr2(static_cast<Number>(1.0)/(Fr*Fr)), Ma2_ov_Fr2(Ma2*inv_Fr2),
    gamma_m1(static_cast<Number>(EquationData::Cp_Cv) - static_cast<Number>(1.0)),
    inv_gamma_m1(static_cast<Number>(1.0)/gamma_m1),
    inv_Gamma(static_cast<Number>(EquationData::Cp_Cv)*inv_gamma_m1), inv_Ro()
    {
      /*--- We create auxiliary vectors and tensors that will never change
            independently on the stage, so we declare it once and for all. ---*/
      for(unsigned d = 0; d < 2; ++d) {
        e_k[d]                   = make_vectorized_array<Number>(0.0);
        tmp_diagonal_velocity[d] = make_vectorized_array<Number>(1.0);
      }
      identity[0][0] = make_vectorized_array<Number>(1.0);
      if constexpr(dim == 2) {
        identity[2][1] = make_vectorized_array<Number>(1.0);
      }
      else {
        identity[1][1] = make_vectorized_array<Number>(1.0);
        identity[2][2] = make_vectorized_array<Number>(1.0);
      }
      e_k[2]                   = make_vectorized_array<Number>(1.0);
      tmp_diagonal_velocity[2] = make_vectorized_array<Number>(1.0);
    }


    
  // Constructor with runtime parameters storage
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  EULEROperator<dim,
                fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary,
                Vec>::
  EULEROperator(const RunTimeParameters::Data_Storage& data,
                const TimeStepping::RungeKutta<Number>& explicit_RK,
                const TimeStepping::RungeKutta<Number>& implicit_RK):
    MatrixFreeOperators::Base<dim, Vec>(),
    Ma(static_cast<Number>(data.Mach)),
    Fr(static_cast<Number>(data.Froude)),
    Ro(static_cast<Number>(data.Rossby)),
    dt(static_cast<Number>(data.dt)),
    n_stages(explicit_RK.get_n_stages()),
    IMEX_stage(1), Euler_stage(1), num_flux(Ma),
    Ma2(Ma*Ma), inv_Ma2(static_cast<Number>(1.0)/Ma2),
    inv_Fr2(static_cast<Number>(1.0)/(Fr*Fr)), Ma2_ov_Fr2(Ma2*inv_Fr2),
    gamma_m1(static_cast<Number>(EquationData::Cp_Cv) - static_cast<Number>(1.0)),
    inv_gamma_m1(static_cast<Number>(1.0)/gamma_m1),
    inv_Gamma(static_cast<Number>(EquationData::Cp_Cv)*inv_gamma_m1),
    inv_Ro(static_cast<Number>(1.0)/Ro)
    {
      /*--- We create auxiliary vectors and tensors that will never change
            independently on the stage, so we declare it once and for all. ---*/
      for(unsigned d = 0; d < 2; ++d) {
        e_k[d]                   = make_vectorized_array<Number>(0.0);
        tmp_diagonal_velocity[d] = make_vectorized_array<Number>(1.0);
      }
      identity[0][0] = make_vectorized_array<Number>(1.0);
      if constexpr(dim == 2) {
        identity[2][1] = make_vectorized_array<Number>(1.0);
      }
      else {
        identity[1][1] = make_vectorized_array<Number>(1.0);
        identity[2][2] = make_vectorized_array<Number>(1.0);
      }
      e_k[2]                   = make_vectorized_array<Number>(1.0);
      tmp_diagonal_velocity[2] = make_vectorized_array<Number>(1.0);

      /*--- Initialize the RK coefficients ---*/
      explicit_RK.get_coefficients(a, b);
      implicit_RK.get_coefficients(a_tilde, b_tilde);

      Assert(n_stages == implicit_RK.get_n_stages(), ExcInternalError());
    }


  //////////////////////////////////////////////////////////////
  /*---- FOCUS NOW ON SOME AUXILIARY GETTERS AND SETTERS ---*/
  /////////////////////////////////////////////////////////////

  // Setter of time-step
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  template<typename T>
  inline DEAL_II_ALWAYS_INLINE
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_dt(const T time_step) {
    dt = static_cast<Number>(time_step);
  }

  // Getter of Mach number
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  typename EULEROperator<dim,
                         fe_degree_u, fe_degree_rho, fe_degree_p,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::Number
  EULEROperator<dim,
                fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary,
                Vec>::
  get_Mach() const {
    return Ma;
  }

  // Getter of Froude number
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  typename EULEROperator<dim,
                         fe_degree_u, fe_degree_rho, fe_degree_p,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::Number
  EULEROperator<dim,
                fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary,
                Vec>::
  get_Froude() const {
    return Fr;
  }

  // Getter of Rossby number
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  typename EULEROperator<dim,
                         fe_degree_u, fe_degree_rho, fe_degree_p,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::Number
  EULEROperator<dim,
                fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary,
                Vec>::
  get_Rossby() const {
    return Ro;
  }

  // Setter of IMEX stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_IMEX_stage(const unsigned stage) {
    AssertIndexRange(stage, n_stages + 2);
    Assert(stage > 0, ExcInternalError());

    IMEX_stage = stage;
  }

  // Setter of Euler stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_Euler_stage(const unsigned stage) {
    AssertIndexRange(stage, EquationData::n_vars + 1);
    Assert(stage > 0, ExcInternalError());

    Euler_stage = stage;
  }

  // Getter of Euler stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  unsigned EULEROperator<dim,
                         fe_degree_u, fe_degree_rho, fe_degree_p,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  get_Euler_stage() const {
    return Euler_stage;
  }


  // Setter of density for fixed point
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_rho_for_fixed(const Vec& src) {
    rho_for_fixed = src;
    rho_for_fixed.update_ghost_values();
  }

  // Setter of pressure for fixed point
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_pres_fixed(const Vec& src) {
    pres_fixed = src;
    pres_fixed.update_ghost_values();
  }


  //////////////////////////////////////////////////////////////
  /*---- ASSEMBLING LINEAR AND BILINEAR FORMS FOR THE CONTINUITY EQUATION ---*/
  /////////////////////////////////////////////////////////////

  // Assemble rhs cell term for the density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_cell_term_density(const MatrixFree<dim, Number>&       data,
                                 Vec&                                 dst,
                                 const std::vector<Vec>&              src,
                                 const std::pair<unsigned, unsigned>& cell_range) const {
    
    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the old density and
      the old velocity. 'phi' will be used only to 'submit' the result.
      The second argument specifies which dof handler has to be used. ---*/
      FEEvaluation_rho              phi(data, EquationData::RHO_INDEX_DOF);
      std::vector<FEEvaluation_rho> phi_rho_prime(IMEX_stage - 1, FEEvaluation_rho(data, EquationData::RHO_INDEX_DOF));
      std::vector<FEEvaluation_u>   phi_u_prime(IMEX_stage - 1, FEEvaluation_u(data, EquationData::U_INDEX_DOF));

      FEEvaluation_rho phi_rho_bar(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u   phi_u_bar(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
        it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
        call to be coherent). All these considerations are valid also for the other assembler functions. ---*/
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          phi_rho_prime[s - 1].reinit(cell);
          phi_rho_prime[s - 1].gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
          phi_u_prime[s - 1].reinit(cell);
          phi_u_prime[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
        }

        phi_rho_bar.reinit(cell);
        phi_rho_bar.gather_evaluate(src[2*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_u_bar.reinit(cell);
        phi_u_bar.gather_evaluate(src[2*(IMEX_stage - 1) + 1], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the density fluctuation at the previous step (always needed) ---*/
          const auto& rho_prime_old = phi_rho_prime.front().get_value(q);

          /*--- Compute background quantities ---*/
          const auto& rho_bar = phi_rho_bar.get_value(q);
          const auto& u_bar   = phi_u_bar.get_value(q);
          Tensor<1, dim, VectorizedArray<Number>> u_bar_tmp;
          if constexpr(dim == 2) {
            u_bar_tmp[0] = u_bar[0];
            u_bar_tmp[1] = u_bar[2];
          }

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& rho_prime_s = phi_rho_prime[s - 1].get_value(q);
            const auto& u_prime_s   = phi_u_prime[s - 1].get_value(q);

            if constexpr(dim == 2) {
              Tensor<1, dim, VectorizedArray<Number>> u_prime_s_tmp;
              u_prime_s_tmp[0] = u_prime_s[0];
              u_prime_s_tmp[1] = u_prime_s[2];
              flux += a[IMEX_stage - 1][s - 1]*dt*
                      ((rho_bar + rho_prime_s)*(u_bar_tmp + u_prime_s_tmp));
              flux -= a[IMEX_stage - 1][s - 1]*dt*rho_bar*u_bar_tmp;
            }
            else {
              flux += a[IMEX_stage - 1][s - 1]*dt*
                      ((rho_bar + rho_prime_s)*(u_bar + u_prime_s));
              flux -= a[IMEX_stage - 1][s - 1]*dt*rho_bar*u_bar;
            }
          }

          phi.submit_value(rho_prime_old, q);
          /*--- submit_value is used for quantities to be tested against test functions ---*/
          phi.submit_gradient(flux, q);
          /*--- submit_gradient is used for quantities to be tested against gradient of test functions ---*/
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        /*--- 'integrate_scatter' is the responsible of distributing into dst.
              The flag parameter specifies if we are testing against the test function and/or its gradient ---*/
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_rho              phi(data, EquationData::RHO_INDEX_DOF);
      std::vector<FEEvaluation_rho> phi_rho_prime(IMEX_stage - 1, FEEvaluation_rho(data, EquationData::RHO_INDEX_DOF));
      std::vector<FEEvaluation_u>   phi_u_prime(IMEX_stage - 1, FEEvaluation_u(data, EquationData::U_INDEX_DOF));

      FEEvaluation_rho phi_rho_bar(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u   phi_u_bar(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          phi_rho_prime[s - 1].reinit(cell);
          phi_rho_prime[s - 1].gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
          phi_u_prime[s - 1].reinit(cell);
          phi_u_prime[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
        }

        phi_rho_bar.reinit(cell);
        phi_rho_bar.gather_evaluate(src[2*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_u_bar.reinit(cell);
        phi_u_bar.gather_evaluate(src[2*(IMEX_stage - 1) + 1], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the density fluctuation at the previous step (always needed) ---*/
          const auto& rho_prime_old = phi_rho_prime.front().get_value(q);

          /*--- Compute background quantities ---*/
          const auto& rho_bar = phi_rho_bar.get_value(q);
          const auto& u_bar   = phi_u_bar.get_value(q);
          Tensor<1, dim, VectorizedArray<Number>> u_bar_tmp;
          if constexpr(dim == 2) {
            u_bar_tmp[0] = u_bar[0];
            u_bar_tmp[1] = u_bar[2];
          }

          /*--- Compute the quantities at the previous stages for the flux ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& rho_prime_s = phi_rho_prime[s - 1].get_value(q);
            const auto& u_prime_s   = phi_u_prime[s - 1].get_value(q);

            if constexpr(dim == 2) {
              Tensor<1, dim, VectorizedArray<Number>> u_prime_s_tmp;
              u_prime_s_tmp[0] = u_prime_s[0];
              u_prime_s_tmp[1] = u_prime_s[2];
              flux += b[s - 1]*dt*
                      ((rho_bar + rho_prime_s)*(u_bar_tmp + u_prime_s_tmp));
              flux -= b[s - 1]*dt*rho_bar*u_bar_tmp;
            }
            else {
              flux += b[s - 1]*dt*
                      ((rho_bar + rho_prime_s)*(u_bar + u_prime_s));
              flux -= b[s - 1]*dt*rho_bar*u_bar;
            }
          }

          phi.submit_value(rho_prime_old, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term for the density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_face_term_density(const MatrixFree<dim, Number>&       data,
                                 Vec&                                 dst,
                                 const std::vector<Vec>&              src,
                                 const std::pair<unsigned, unsigned>& face_range) const {
                                  
    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities.
            'true' means that we are reading the information from 'inside', whereas 'false' from 'outside' ---*/
      FEFaceEvaluation_rho phi_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_p(data, false, EquationData::RHO_INDEX_DOF),
                           phi_rho_prime_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_prime_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_prime_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_prime_p(data, false, EquationData::U_INDEX_DOF);

      FEFaceEvaluation_rho phi_rho_bar_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_bar_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_bar_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_bar_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_prime_m.reinit(face);
        phi_rho_prime_p.reinit(face);
        phi_u_prime_m.reinit(face);
        phi_u_prime_p.reinit(face);

        phi_rho_bar_m.reinit(face);
        phi_rho_bar_m.gather_evaluate(src[2*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_rho_bar_p.reinit(face);
        phi_rho_bar_p.gather_evaluate(src[2*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_u_bar_m.reinit(face);
        phi_u_bar_m.gather_evaluate(src[2*(IMEX_stage - 1) + 1], EvaluationFlags::values);
        phi_u_bar_p.reinit(face);
        phi_u_bar_p.gather_evaluate(src[2*(IMEX_stage - 1) + 1], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.normal_vector(q); /*--- Notice that the unit normal vector is the same from
                                                              'both sides'. ---*/
          Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
          if constexpr(dim == 2) {
            n_minus_tmp[0] = n_minus[0];
            n_minus_tmp[2] = n_minus[1];
          }
          else {
            n_minus_tmp = n_minus;
          }

          /*--- First, focus on background quantities ---*/
          const auto& rho_bar_m = phi_rho_bar_m.get_value(q);
          const auto& rho_bar_p = phi_rho_bar_p.get_value(q);
          const auto& u_bar_m   = phi_u_bar_m.get_value(q);
          const auto& u_bar_p   = phi_u_bar_p.get_value(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux_num = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> flux_bar = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> flux_full = make_vectorized_array<Number>(0.0);
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            /*--- Retrieve the useful fields ---*/
            phi_rho_prime_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_rho_prime_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_u_prime_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
            phi_u_prime_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);

            const auto& rho_prime_s_m = phi_rho_prime_m.get_value(q);
            const auto& rho_prime_s_p = phi_rho_prime_p.get_value(q);
            const auto& u_prime_s_m   = phi_u_prime_m.get_value(q);
            const auto& u_prime_s_p   = phi_u_prime_p.get_value(q);
            const auto& rho_bar_s_m = phi_rho_bar_m.get_value(q);
            const auto& rho_bar_s_p = phi_rho_bar_p.get_value(q);
            const auto& u_bar_s_m   = phi_u_bar_m.get_value(q);
            const auto& u_bar_s_p   = phi_u_bar_p.get_value(q);

            /*--- Compute the numerical flux ---*/
            flux_full += a[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_continuity(rho_bar_m + rho_prime_s_m,
                                                           u_bar_m + u_prime_s_m,
                                                           rho_bar_p + rho_prime_s_p,
                                                           u_bar_p + u_prime_s_p,
                                                           n_minus_tmp);
          flux_bar += a[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_continuity(rho_bar_s_m,
                                                           u_bar_s_m,
                                                           rho_bar_s_p,
                                                           u_bar_s_p,
                                                           n_minus_tmp);
          }
          flux_num = flux_full - flux_bar;
          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_rho phi_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_p(data, false, EquationData::RHO_INDEX_DOF),
                           phi_rho_prime_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_prime_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_prime_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_prime_p(data, false, EquationData::U_INDEX_DOF);

      FEFaceEvaluation_rho phi_rho_bar_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_bar_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_bar_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_bar_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_prime_m.reinit(face);
        phi_rho_prime_p.reinit(face);
        phi_u_prime_m.reinit(face);
        phi_u_prime_p.reinit(face);

        phi_rho_bar_m.reinit(face);
        phi_rho_bar_m.gather_evaluate(src[2*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_rho_bar_p.reinit(face);
        phi_rho_bar_p.gather_evaluate(src[2*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_u_bar_m.reinit(face);
        phi_u_bar_m.gather_evaluate(src[2*(IMEX_stage - 1) + 1], EvaluationFlags::values);
        phi_u_bar_p.reinit(face);
        phi_u_bar_p.gather_evaluate(src[2*(IMEX_stage - 1) + 1], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.normal_vector(q);
          Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
          if constexpr(dim == 2) {
            n_minus_tmp[0] = n_minus[0];
            n_minus_tmp[2] = n_minus[1];
          }
          else {
            n_minus_tmp = n_minus;
          }

          /*--- First, focus on background quantities ---*/
          const auto& rho_bar_m = phi_rho_bar_m.get_value(q);
          const auto& rho_bar_p = phi_rho_bar_p.get_value(q);
          const auto& u_bar_m   = phi_u_bar_m.get_value(q);
          const auto& u_bar_p   = phi_u_bar_p.get_value(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux_num = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> flux_bar = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> flux_full = make_vectorized_array<Number>(0.0);
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            /*--- Retrieve the useful fields ---*/
            phi_rho_prime_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_rho_prime_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values);
            phi_u_prime_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);
            phi_u_prime_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values);

            const auto& rho_prime_s_m = phi_rho_prime_m.get_value(q);
            const auto& rho_prime_s_p = phi_rho_prime_p.get_value(q);
            const auto& u_prime_s_m   = phi_u_prime_m.get_value(q);
            const auto& u_prime_s_p   = phi_u_prime_p.get_value(q);
            const auto& rho_bar_s_m = phi_rho_bar_m.get_value(q);
            const auto& rho_bar_s_p = phi_rho_bar_p.get_value(q);
            const auto& u_bar_s_m   = phi_u_bar_m.get_value(q);
            const auto& u_bar_s_p   = phi_u_bar_p.get_value(q);

            /*--- Compute the numerical_flux ---*/
            flux_full += b[s - 1]*dt*
                        num_flux.numerical_flux_continuity(rho_bar_m + rho_prime_s_m,
                                                           u_bar_m + u_prime_s_m,
                                                           rho_bar_p + rho_prime_s_p,
                                                           u_bar_p + u_prime_s_p,
                                                           n_minus_tmp);

          flux_bar += b[s - 1]*dt*
                        num_flux.numerical_flux_continuity(rho_bar_s_m,
                                                           u_bar_s_m,
                                                           rho_bar_s_p,
                                                           u_bar_s_p,
                                                           n_minus_tmp);

          }
          flux_num=flux_full - flux_bar;
          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_rhs_density(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_density,
                     &EULEROperator::assemble_rhs_face_term_density,
                     &EULEROperator::assemble_rhs_boundary_term_density,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble cell term for the density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_density(const MatrixFree<dim, Number>&       data,
                             Vec&                                 dst,
                             const Vec&                           src,
                             const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, fe_degree_rho + 1, 1, Number> phi(data, EquationData::RHO_INDEX_DOF, 3);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree_rho, 1, Number> inverse(phi);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);

      inverse.apply(phi.begin_dof_values(),
                    phi.begin_dof_values());

      phi.set_dof_values(dst);
    }
  }


  //////////////////////////////////////////////////////////////
  /*---- ASSEMBLING LINEAR AND BILINEAR FORMS FOR THE MOMENTUM EQUATION ---*/
  /////////////////////////////////////////////////////////////

  // Assemble rhs cell term of the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_cell_term_momentum(const MatrixFree<dim, Number>&       data,
                                  Vec&                                 dst,
                                  const std::vector<Vec>&              src,
                                  const std::pair<unsigned, unsigned>& cell_range) const {

                                    return; 
    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_u                 phi(data, EquationData::U_INDEX_DOF);
      std::vector<FEEvaluation_u>    phi_u_prime(IMEX_stage, FEEvaluation_u(data, EquationData::U_INDEX_DOF));
      std::vector<FEEvaluation_pres> phi_pres_prime(IMEX_stage - 1, FEEvaluation_pres(data, EquationData::P_INDEX_DOF));
      std::vector<FEEvaluation_rho>  phi_rho_prime(IMEX_stage, FEEvaluation_rho(data, EquationData::RHO_INDEX_DOF));

      FEEvaluation_rho phi_rho_bar(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u   phi_u_bar(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          phi_rho_prime[s - 1].reinit(cell);
          phi_rho_prime[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_u_prime[s - 1].reinit(cell);
          phi_u_prime[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_pres_prime[s - 1].reinit(cell);
          phi_pres_prime[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
        }
        phi_rho_prime.back().reinit(cell);
        phi_rho_prime.back().gather_evaluate(src[3*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_u_prime.back().reinit(cell);
        phi_u_prime.back().gather_evaluate(src[3*(IMEX_stage - 1) + 1], EvaluationFlags::values);

        phi_rho_bar.reinit(cell);
        phi_rho_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);
        phi_u_bar.reinit(cell);
        phi_u_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the density and the velocity fluctuations at the previous step (always necessary).
                Notice that this is ok because of ESDIRK method. ---*/
          const auto& rho_prime_old = phi_rho_prime.front().get_value(q);
          const auto& u_prime_old   = phi_u_prime.front().get_value(q);

          /*--- Compute background quantities ---*/
          const auto& rho_bar = phi_rho_bar.get_value(q);
          const auto& u_bar   = phi_u_bar.get_value(q);

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, 3, Tensor<1, dim, VectorizedArray<Number>>> flux;
          Tensor<1, 3, VectorizedArray<Number>> gravity_term;
          Tensor<1, 3, VectorizedArray<Number>> rotation_term;
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& u_prime_s          = phi_u_prime[s - 1].get_value(q);
            Tensor<2, 3, VectorizedArray<Number>>  tensor_product_u_s = outer_product(u_bar + u_prime_s, u_bar + u_prime_s);
            Tensor<2, 3, VectorizedArray<Number>>  tensor_product_u_s_bar=outer_product(u_bar, u_bar);

            const auto& p_prime_s_times_identity = phi_pres_prime[s - 1].get_value(q)*identity;

            const auto& rho_prime_s = phi_rho_prime[s - 1].get_value(q);

            if constexpr(dim == 2) {
              Tensor<1, 3, Tensor<1, dim, VectorizedArray<Number>>> tensor_product_u_s_tmp;
              Tensor<1, 3, Tensor<1, dim, VectorizedArray<Number>>> tensor_product_u_s_bar_tmp;
              for(unsigned i = 0; i < 3; ++i) {
                tensor_product_u_s_tmp[i][0] = tensor_product_u_s[i][0];
                tensor_product_u_s_tmp[i][1] = tensor_product_u_s[i][2];
                tensor_product_u_s_bar_tmp[i][0] = tensor_product_u_s_bar[i][0];
                tensor_product_u_s_bar_tmp[i][1] = tensor_product_u_s_bar[i][2];
              }

              flux += a[IMEX_stage - 1][s - 1]*dt*((rho_bar + rho_prime_s)*tensor_product_u_s_tmp);
              flux -= a[IMEX_stage - 1][s - 1]*dt*(rho_bar*tensor_product_u_s_bar_tmp);
              flux += a_tilde[IMEX_stage - 1][s - 1]*dt*(inv_Ma2*p_prime_s_times_identity);
            }
            else {
              flux += a[IMEX_stage - 1][s - 1]*dt*((rho_bar + rho_prime_s)*tensor_product_u_s );
              flux -= a[IMEX_stage - 1][s - 1]*dt*(rho_bar*tensor_product_u_s_bar);
              flux += a_tilde[IMEX_stage - 1][s - 1]*dt*(inv_Ma2*p_prime_s_times_identity);
            }

            gravity_term += a_tilde[IMEX_stage - 1][s - 1]*dt*
                            (inv_Fr2*rho_prime_s*e_k);

            const auto& k_cross_u_prime_s = cross_product_3d(e_k, u_prime_s);
            const auto& k_cross_u_bar     = cross_product_3d(e_k, u_bar);
            rotation_term += a_tilde[IMEX_stage - 1][s - 1]*dt*
                             (inv_Ro*((rho_bar + rho_prime_s)*k_cross_u_prime_s+rho_prime_s*k_cross_u_bar));
          }

          /*--- Add last contribution of the gravity and rotation term (implicit treatment) ---*/
          const auto& rho_prime_s = phi_rho_prime.back().get_value(q);
          gravity_term += a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*(inv_Fr2*rho_prime_s*e_k);

          const auto& u_prime_fixed_s         = phi_u_prime.back().get_value(q);
          const auto& k_cross_u_prime_fixed_s = cross_product_3d(e_k, u_prime_fixed_s);
          const auto& k_cross_u_bar             = cross_product_3d(e_k, u_bar);
          rotation_term += a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                           (inv_Ro*((rho_bar + rho_prime_s)*k_cross_u_prime_fixed_s+rho_prime_s*k_cross_u_bar));

          phi.submit_value((rho_bar + rho_prime_old)*u_prime_old + (rho_prime_old - rho_prime_s)*u_bar - gravity_term -0.0*rotation_term, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_u                 phi(data, EquationData::U_INDEX_DOF);
      std::vector<FEEvaluation_u>    phi_u_prime(IMEX_stage - 1, FEEvaluation_u(data, EquationData::U_INDEX_DOF));
      std::vector<FEEvaluation_pres> phi_pres_prime(IMEX_stage - 1, FEEvaluation_pres(data, EquationData::P_INDEX_DOF));
      std::vector<FEEvaluation_rho>  phi_rho_prime(IMEX_stage, FEEvaluation_rho(data, EquationData::RHO_INDEX_DOF));

      FEEvaluation_rho phi_rho_bar(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u   phi_u_bar(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          phi_rho_prime[s - 1].reinit(cell);
          phi_rho_prime[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_u_prime[s - 1].reinit(cell);
          phi_u_prime[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_pres_prime[s - 1].reinit(cell);
          phi_pres_prime[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
        }
        phi_rho_prime.back().reinit(cell);
        phi_rho_prime.back().gather_evaluate(src[3*(IMEX_stage - 1)], EvaluationFlags::values);

        phi_rho_bar.reinit(cell);
        phi_rho_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 1], EvaluationFlags::values);
        phi_u_bar.reinit(cell);
        phi_u_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the density and the velocity fluctuations at the previous step (always necessary).
                Notice that this is ok because of ESDIRK method. ---*/
          const auto& rho_prime_old = phi_rho_prime.front().get_value(q);
          const auto& u_prime_old   = phi_u_prime.front().get_value(q);

          /*--- Compute background quantities ---*/
          const auto& rho_bar = phi_rho_bar.get_value(q);
          const auto& u_bar   = phi_u_bar.get_value(q);

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, 3, Tensor<1, dim, VectorizedArray<Number>>> flux;
          Tensor<1, 3, VectorizedArray<Number>> gravity_term;
          Tensor<1, 3, VectorizedArray<Number>> rotation_term;
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& u_prime_s          = phi_u_prime[s - 1].get_value(q);
            Tensor<2, 3, VectorizedArray<Number>> tensor_product_u_s = outer_product(u_bar + u_prime_s, u_bar + u_prime_s);
            Tensor<2, 3, VectorizedArray<Number>> tensor_product_u_s_bar=outer_product(u_bar, u_bar);
            const auto& p_prime_s_times_identity = phi_pres_prime[s - 1].get_value(q)*identity;

            const auto& rho_prime_s = phi_rho_prime[s - 1].get_value(q);

            if constexpr(dim == 2) {
              Tensor<1, 3, Tensor<1, dim, VectorizedArray<Number>>> tensor_product_u_s_tmp;
              Tensor<1, 3, Tensor<1, dim, VectorizedArray<Number>>> tensor_product_u_s_bar_tmp;
              for(unsigned i = 0; i < 3; ++i) {
                tensor_product_u_s_tmp[i][0] = tensor_product_u_s[i][0];
                tensor_product_u_s_tmp[i][1] = tensor_product_u_s[i][2];
                tensor_product_u_s_bar_tmp[i][0] = tensor_product_u_s_bar[i][0];
                tensor_product_u_s_bar_tmp[i][1] = tensor_product_u_s_bar[i][2];
              }

              flux += b[s - 1]*dt*((rho_bar + rho_prime_s)*tensor_product_u_s_tmp);
              flux -= b[s - 1]*dt*(rho_bar*tensor_product_u_s_bar_tmp);
              flux += b_tilde[s - 1]*dt*(inv_Ma2*p_prime_s_times_identity);
            }
            else {
              flux += b[s - 1]*dt*((rho_bar + rho_prime_s)*tensor_product_u_s);
              flux -= b[s - 1]*dt*(rho_bar*tensor_product_u_s_bar);
              flux += b_tilde[s - 1]*dt*(inv_Ma2*p_prime_s_times_identity);
            }

            gravity_term += b_tilde[s - 1]*dt*
                            (inv_Fr2*rho_prime_s*e_k);

            const auto& k_cross_u_prime_s = cross_product_3d(e_k, u_prime_s);
            const auto& k_cross_u_bar     = cross_product_3d(e_k, u_bar);
            rotation_term += b_tilde[s - 1]*dt*
                             (inv_Ro*((rho_bar + rho_prime_s)*k_cross_u_prime_s+rho_prime_s*k_cross_u_bar));
          }

          const auto& rho_prime_curr = phi_rho_prime.back().get_value(q);

          phi.submit_value((rho_bar + rho_prime_old)*u_prime_old + (rho_prime_old - rho_prime_curr)*u_bar - 0.0*gravity_term - 0.0*rotation_term, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term of the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_face_term_momentum(const MatrixFree<dim, Number>&       data,
                                  Vec&                                 dst,
                                  const std::vector<Vec>&              src,
                                  const std::pair<unsigned, unsigned>& face_range) const {
                                    return;
    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u    phi_m(data, true, EquationData::U_INDEX_DOF),
                            phi_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_prime_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_prime_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_pres phi_pres_prime_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_prime_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_prime_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_prime_p(data, false, EquationData::RHO_INDEX_DOF);

      FEFaceEvaluation_rho phi_rho_bar_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_bar_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_bar_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_bar_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_prime_m.reinit(face);
        phi_rho_prime_p.reinit(face);
        phi_u_prime_m.reinit(face);
        phi_u_prime_p.reinit(face);
        phi_pres_prime_m.reinit(face);
        phi_pres_prime_p.reinit(face);

        phi_rho_bar_m.reinit(face);
        phi_rho_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);
        phi_rho_bar_p.reinit(face);
        phi_rho_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);
        phi_u_bar_m.reinit(face);
        phi_u_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);
        phi_u_bar_p.reinit(face);
        phi_u_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.normal_vector(q);
          Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
          if constexpr(dim == 2) {
            n_minus_tmp[0] = n_minus[0];
            n_minus_tmp[2] = n_minus[1];
          }
          else {
            n_minus_tmp = n_minus;
          }

          /*--- First, focus on background quantities ---*/
          const auto& rho_bar_m = phi_rho_bar_m.get_value(q);
          const auto& rho_bar_p = phi_rho_bar_p.get_value(q);
          const auto& u_bar_m   = phi_u_bar_m.get_value(q);
          const auto& u_bar_p   = phi_u_bar_p.get_value(q);

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, 3, VectorizedArray<Number>> flux_total, flux_bar;
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            /*--- Retrieve the useful fields ---*/
            phi_rho_prime_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_rho_prime_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_u_prime_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_u_prime_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_pres_prime_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_pres_prime_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& rho_prime_s_m  = phi_rho_prime_m.get_value(q);
            const auto& rho_prime_s_p  = phi_rho_prime_p.get_value(q);
            const auto& u_prime_s_m    = phi_u_prime_m.get_value(q);
            const auto& u_prime_s_p    = phi_u_prime_p.get_value(q);
            const auto& pres_prime_s_m = phi_pres_prime_m.get_value(q);
            const auto& pres_prime_s_p = phi_pres_prime_p.get_value(q);

            /*--- Compute the numerical flux ---*/
            flux_total += a[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_momentum_explicit(rho_bar_m + rho_prime_s_m,
                                                                  u_bar_m + u_prime_s_m,
                                                                  rho_bar_p + rho_prime_s_p,
                                                                  u_bar_p + u_prime_s_p,
                                                                  n_minus_tmp)
                      + a_tilde[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_momentum_implicit(pres_prime_s_m,
                                                                  pres_prime_s_p,
                                                                  n_minus_tmp);

            flux_bar += a[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_momentum_explicit(rho_bar_m,
                                                                  u_bar_m,
                                                                  rho_bar_p,
                                                                  u_bar_p,
                                                                  n_minus_tmp);
          }
          const auto& flux_num= flux_total - flux_bar;
          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u    phi_m(data, true, EquationData::U_INDEX_DOF),
                            phi_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_prime_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_prime_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_pres phi_pres_prime_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_prime_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_prime_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_prime_p(data, false, EquationData::RHO_INDEX_DOF);

      FEFaceEvaluation_rho phi_rho_bar_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_bar_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_bar_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_bar_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_prime_m.reinit(face);
        phi_rho_prime_p.reinit(face);
        phi_u_prime_m.reinit(face);
        phi_u_prime_p.reinit(face);
        phi_pres_prime_m.reinit(face);
        phi_pres_prime_p.reinit(face);

        phi_rho_bar_m.reinit(face);
        phi_rho_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 1], EvaluationFlags::values);
        phi_rho_bar_p.reinit(face);
        phi_rho_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 1], EvaluationFlags::values);
        phi_u_bar_m.reinit(face);
        phi_u_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);
        phi_u_bar_p.reinit(face);
        phi_u_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.normal_vector(q);
          Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
          if constexpr(dim == 2) {
            n_minus_tmp[0] = n_minus[0];
            n_minus_tmp[2] = n_minus[1];
          }
          else {
            n_minus_tmp = n_minus;
          }

          /*--- First, focus on background quantities ---*/
          const auto& rho_bar_m = phi_rho_bar_m.get_value(q);
          const auto& rho_bar_p = phi_rho_bar_p.get_value(q);
          const auto& u_bar_m   = phi_u_bar_m.get_value(q);
          const auto& u_bar_p   = phi_u_bar_p.get_value(q);

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, 3, VectorizedArray<Number>> flux_total, flux_bar;
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            /*--- Retrieve the useful fields ---*/
            phi_rho_prime_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_rho_prime_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_u_prime_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_u_prime_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_pres_prime_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_pres_prime_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& rho_prime_s_m  = phi_rho_prime_m.get_value(q);
            const auto& rho_prime_s_p  = phi_rho_prime_p.get_value(q);
            const auto& u_prime_s_m    = phi_u_prime_m.get_value(q);
            const auto& u_prime_s_p    = phi_u_prime_p.get_value(q);
            const auto& pres_prime_s_m = phi_pres_prime_m.get_value(q);
            const auto& pres_prime_s_p = phi_pres_prime_p.get_value(q);

            /*--- Compute the numerical flux ---*/
            flux_total += b[s - 1]*dt*
                        num_flux.numerical_flux_momentum_explicit(rho_bar_m + rho_prime_s_m,
                                                                  u_bar_m + u_prime_s_m,
                                                                  rho_bar_p + rho_prime_s_p,
                                                                  u_bar_p + u_prime_s_p,
                                                                  n_minus_tmp)
                      + b_tilde[s - 1]*dt*
                        num_flux.numerical_flux_momentum_implicit(pres_prime_s_m,
                                                                  pres_prime_s_p,
                                                                  n_minus_tmp);
            flux_bar += b[s - 1]*dt*
                        num_flux.numerical_flux_momentum_explicit(rho_bar_m,
                                                                  u_bar_m,
                                                                  rho_bar_p,
                                                                  u_bar_p,
                                                                  n_minus_tmp);
          }
          const auto& flux_num= flux_total - flux_bar;
          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Assemble rhs boundary term of the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_boundary_term_momentum(const MatrixFree<dim, Number>&       data,
                                      Vec&                                 dst,
                                      const std::vector<Vec>&              src,
                                      const std::pair<unsigned, unsigned>& face_range) const {
                                        return;
    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u_boundary    phi(data, true, EquationData::U_INDEX_DOF, 1);
      FEFaceEvaluation_pres_boundary phi_pres_prime(data, true, EquationData::P_INDEX_DOF, 1);

      /*--- Loop over all boundary faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_pres_prime.reinit(face);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          const auto& n_minus = phi.normal_vector(q);
          Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
          if constexpr(dim == 2) {
            n_minus_tmp[0] = n_minus[0];
            n_minus_tmp[2] = n_minus[1];
          }
          else {
            n_minus_tmp = n_minus;
          }

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, 3, VectorizedArray<Number>> flux_num;
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            /*--- Retrieve the useful fields ---*/
            phi_pres_prime.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& pres_prime_s   = phi_pres_prime.get_value(q);
            const auto& pres_prime_s_D = pres_prime_s;

            /*--- Compute the numerical flux ---*/
            flux_num += a_tilde[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_momentum_implicit(pres_prime_s,
                                                                  pres_prime_s_D,
                                                                  n_minus_tmp);
          }

          phi.submit_value(-flux_num, q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u_boundary    phi(data, true, EquationData::U_INDEX_DOF, 1);
      FEFaceEvaluation_pres_boundary phi_pres_prime(data, true, EquationData::P_INDEX_DOF, 1);

      /*--- Loop over all boundary faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_pres_prime.reinit(face);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          const auto& n_minus = phi.normal_vector(q);
          Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
          if constexpr(dim == 2) {
            n_minus_tmp[0] = n_minus[0];
            n_minus_tmp[2] = n_minus[1];
          }
          else {
            n_minus_tmp = n_minus;
          }

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, 3, VectorizedArray<Number>> flux_num;
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            /*--- Retrieve the useful fields ---*/
            phi_pres_prime.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& pres_prime_s   = phi_pres_prime.get_value(q);
            const auto& pres_prime_s_D = pres_prime_s;

            /*--- Compute the numerical flux ---*/
            flux_num += b_tilde[s - 1]*dt*
                        num_flux.numerical_flux_momentum_implicit(pres_prime_s,
                                                                  pres_prime_s_D,
                                                                  n_minus_tmp);
          }

          phi.submit_value(-flux_num, q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_rhs_momentum(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_momentum,
                     &EULEROperator::assemble_rhs_face_term_momentum,
                     &EULEROperator::assemble_rhs_boundary_term_momentum,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble cell term for the velocity update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities.
          Since here we have just one 'src' vector, but we also need to deal with the current density,
          we employ the auxiliary vector 'rho_for_fixed' where we setted this information ---*/
    FEEvaluation<dim, fe_degree_u, fe_degree_u + 1, 3, Number>   phi(data, EquationData::U_INDEX_DOF, 2);
    FEEvaluation<dim, fe_degree_rho, fe_degree_u + 1, 1, Number> phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF, 2);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree_u, 3, Number> inverse(phi);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);
      phi.read_dof_values(src);

      AlignedVector<VectorizedArray<Number>> inverse_jxw(phi.n_q_points);
      inverse.fill_inverse_JxW_values(inverse_jxw);

      /*--- Loop over all quadrature points to fill the inverse of the coefficient ---*/
      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        inverse_jxw[q] *= 1.0/phi_rho_for_fixed.get_value(q);
      }

      inverse.apply(inverse_jxw, 3,
                    phi.begin_dof_values(),
                    phi.begin_dof_values());

      phi.set_dof_values(dst);
    }
  }

  // Assemble cell term for the pressure
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read quantities. This operator we are going to implement
          represents a rectangular matrix (we start from the pressure FE space and we end up with the velocity FE space).
          This is the reason of the distinction between 'phi' and 'phi_src'. ---*/
    FEEvaluation_u    phi(data, EquationData::U_INDEX_DOF);
    FEEvaluation_pres phi_src(data, EquationData::P_INDEX_DOF);

    /*--- Loop over all cells. ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(cell);

      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        phi.submit_gradient(-a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*(inv_Ma2*phi_src.get_value(q))*identity, q);
      }

      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the pressure
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& face_range) const {
    FEFaceEvaluation_u    phi_m(data, true, EquationData::U_INDEX_DOF),
                          phi_p(data, false, EquationData::U_INDEX_DOF);
    FEFaceEvaluation_pres phi_src_m(data, true, EquationData::P_INDEX_DOF),
                          phi_src_p(data, false, EquationData::P_INDEX_DOF);

    /*--- Loop over all internal faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, EvaluationFlags::values);
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, EvaluationFlags::values);

      phi_m.reinit(face);
      phi_p.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
        const auto& n_minus = phi_m.normal_vector(q);
        Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
        if constexpr(dim == 2) {
          n_minus_tmp[0] = n_minus[0];
          n_minus_tmp[2] = n_minus[1];
        }
        else {
          n_minus_tmp = n_minus;
        }

        const auto& avg_term = 0.5*(phi_src_m.get_value(q) +
                                    phi_src_p.get_value(q));

        const auto& flux_num = a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                               (inv_Ma2*avg_term*n_minus_tmp);

        phi_m.submit_value(flux_num, q);
        phi_p.submit_value(-flux_num, q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Assemble boundary term for the pressure
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned, unsigned>& face_range) const {
    FEFaceEvaluation_u_boundary    phi(data, true, EquationData::U_INDEX_DOF, 1);
    FEFaceEvaluation_pres_boundary phi_src(data, true, EquationData::P_INDEX_DOF, 1);

    /*--- Loop over all boundary faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_src.reinit(face);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        const auto& n_minus = phi.normal_vector(q);
        Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
        if constexpr(dim == 2) {
          n_minus_tmp[0] = n_minus[0];
          n_minus_tmp[2] = n_minus[1];
        }
        else {
          n_minus_tmp = n_minus;
        }

        const auto& pres_fixed_D = phi_src.get_value(q);

        const auto& avg_term     = 0.5*(phi_src.get_value(q) +
                                        pres_fixed_D);

        phi.submit_value(a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                         (inv_Ma2*avg_term*n_minus_tmp), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  //////////////////////////////////////////////////////////////
  /*---- ASSEMBLING LINEAR AND BILINEAR FORMS FOR THE ENERGY EQUATION ---*/
  /////////////////////////////////////////////////////////////

  // Assemble rhs cell term of the energy equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_cell_term_energy(const MatrixFree<dim, Number>&       data,
                                Vec&                                 dst,
                                const std::vector<Vec>&              src,
                                const std::pair<unsigned, unsigned>& cell_range) const {
                                  
    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_pres              phi(data, EquationData::P_INDEX_DOF);
      std::vector<FEEvaluation_pres> phi_pres_prime(IMEX_stage - 1, FEEvaluation_pres(data, EquationData::P_INDEX_DOF));
      std::vector<FEEvaluation_u>    phi_u_prime(IMEX_stage, FEEvaluation_u(data, EquationData::U_INDEX_DOF));
      std::vector<FEEvaluation_rho>  phi_rho_prime(IMEX_stage, FEEvaluation_rho(data, EquationData::RHO_INDEX_DOF));

      FEEvaluation_rho  phi_rho_bar(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u    phi_u_bar(data, EquationData::U_INDEX_DOF);
      FEEvaluation_pres phi_pres_bar(data, EquationData::P_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          phi_rho_prime[s - 1].reinit(cell);
          phi_rho_prime[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_u_prime[s - 1].reinit(cell);
          phi_u_prime[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_pres_prime[s - 1].reinit(cell);
          phi_pres_prime[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
        }
        phi_rho_prime.back().reinit(cell);
        phi_rho_prime.back().gather_evaluate(src[3*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_u_prime.back().reinit(cell);
        phi_u_prime.back().gather_evaluate(src[3*(IMEX_stage - 1) + 1], EvaluationFlags::values);

        phi_rho_bar.reinit(cell);
        phi_rho_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);
        phi_u_bar.reinit(cell);
        phi_u_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 4], EvaluationFlags::values);
        phi_pres_bar.reinit(cell);
        phi_pres_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 5], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step (always necessary).
                Notice that this is ok because of ESDIRK method. ---*/
          const auto& rho_prime_old  = phi_rho_prime.front().get_value(q);
          const auto& u_prime_old    = phi_u_prime.front().get_value(q);
          const auto& pres_prime_old = phi_pres_prime.front().get_value(q);

          /*--- Compute background quantities ---*/
          const auto& rho_bar  = phi_rho_bar.get_value(q);
          const auto& u_bar    = phi_u_bar.get_value(q);
          const auto& pres_bar = phi_pres_bar.get_value(q);
          Tensor<1, dim, VectorizedArray<Number>> u_bar_tmp;
          if constexpr(dim == 2) {
            u_bar_tmp[0] = u_bar[0];
            u_bar_tmp[1] = u_bar[2];
          }

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          VectorizedArray<Number> gravity_term = make_vectorized_array<Number>(0.0);
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& rho_prime_s  = phi_rho_prime[s - 1].get_value(q);
            const auto& u_prime_s    = phi_u_prime[s - 1].get_value(q);
            const auto& pres_prime_s = phi_pres_prime[s - 1].get_value(q);

            if constexpr(dim == 2) {
              Tensor<1, dim, VectorizedArray<Number>> u_prime_s_tmp;
              u_prime_s_tmp[0] = u_prime_s[0];
              u_prime_s_tmp[1] = u_prime_s[2];

              flux += a[IMEX_stage - 1][s - 1]*dt*
                      ((rho_bar + rho_prime_s)*(0.5*Ma2*scalar_product(u_bar + u_prime_s, u_bar + u_prime_s))*(u_bar_tmp + u_prime_s_tmp));
                    //+ a_tilde[IMEX_stage - 1][s - 1]*dt*(inv_Gamma*((pres_bar + pres_prime_s)*(u_bar_tmp + u_prime_s_tmp)));
              flux -= a[IMEX_stage - 1][s - 1]*dt*((rho_bar)*(0.5*Ma2*scalar_product(u_bar, u_bar))*(u_bar_tmp));
                    //+ a_tilde[IMEX_stage - 1][s - 1]*dt*(inv_Gamma*((pres_bar)*(u_bar_tmp)));
            }
            else {
              flux += a[IMEX_stage - 1][s - 1]*dt*
                      ((rho_bar + rho_prime_s)*(0.5*Ma2*scalar_product(u_bar + u_prime_s, u_bar + u_prime_s))*(u_bar + u_prime_s))
                    + a_tilde[IMEX_stage - 1][s - 1]*dt*
                      (inv_Gamma*((pres_bar + pres_prime_s)*(u_bar + u_prime_s)));
              flux -= (a[IMEX_stage - 1][s - 1]*dt*
                      ((rho_bar)*(0.5*Ma2*scalar_product(u_bar, u_bar))*(u_bar))
                    + a_tilde[IMEX_stage - 1][s - 1]*dt*
                      (inv_Gamma*((pres_bar)*(u_bar))));
            }

            gravity_term += a_tilde[IMEX_stage - 1][s - 1]*dt*
                            (Ma2_ov_Fr2*(rho_bar + rho_prime_s)*(u_bar[2] + u_prime_s[2]));
            gravity_term -= a_tilde[IMEX_stage - 1][s - 1]*dt*
                            (Ma2_ov_Fr2*(rho_bar)*(u_bar[2]));
          }

          /*--- We assign to the rhs the contribution due to kinetic energy in the fixed point loop.
                Add last contribution of the gravity term (implicit treatment) ---*/
          const auto& rho_prime_for_fixed_s = phi_rho_prime.back().get_value(q);
          const auto& u_prime_fixed_s       = phi_u_prime.back().get_value(q);
          gravity_term += a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                          (Ma2_ov_Fr2*(rho_bar + rho_prime_for_fixed_s)*(u_bar[2] + u_prime_fixed_s[2]));
          gravity_term -= a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                          (Ma2_ov_Fr2*(rho_bar)*(u_bar[2]));
          phi.submit_value(//inv_gamma_m1*pres_prime_old +
                           (rho_bar + rho_prime_old)*(0.5*Ma2*scalar_product(u_bar + u_prime_old, u_bar + u_prime_old)) -
                           (rho_bar + rho_prime_for_fixed_s)*(0.5*Ma2*scalar_product(u_bar + u_prime_fixed_s, u_bar + u_prime_fixed_s)) -0.0*gravity_term, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_pres              phi(data, EquationData::P_INDEX_DOF);
      std::vector<FEEvaluation_pres> phi_pres_prime(IMEX_stage - 1, FEEvaluation_pres(data, EquationData::P_INDEX_DOF));
      std::vector<FEEvaluation_u>    phi_u_prime(IMEX_stage, FEEvaluation_u(data, EquationData::U_INDEX_DOF));
      std::vector<FEEvaluation_rho>  phi_rho_prime(IMEX_stage, FEEvaluation_rho(data, EquationData::RHO_INDEX_DOF));

      FEEvaluation_rho  phi_rho_bar(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u    phi_u_bar(data, EquationData::U_INDEX_DOF);
      FEEvaluation_pres phi_pres_bar(data, EquationData::P_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          phi_rho_prime[s - 1].reinit(cell);
          phi_rho_prime[s - 1].gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
          phi_u_prime[s - 1].reinit(cell);
          phi_u_prime[s - 1].gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
          phi_pres_prime[s - 1].reinit(cell);
          phi_pres_prime[s - 1].gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
        }
        phi_rho_prime.back().reinit(cell);
        phi_rho_prime.back().gather_evaluate(src[3*(IMEX_stage - 1)], EvaluationFlags::values);
        phi_u_prime.back().reinit(cell);
        phi_u_prime.back().gather_evaluate(src[3*(IMEX_stage - 1) + 1], EvaluationFlags::values);

        phi_rho_bar.reinit(cell);
        phi_rho_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);
        phi_u_bar.reinit(cell);
        phi_u_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);
        phi_pres_bar.reinit(cell);
        phi_pres_bar.gather_evaluate(src[3*(IMEX_stage - 1) + 4], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step (always necessary).
                Notice that this is ok because of ESDIRK method. ---*/
          const auto& rho_prime_old  = phi_rho_prime.front().get_value(q);
          const auto& u_prime_old    = phi_u_prime.front().get_value(q);
          const auto& pres_prime_old = phi_pres_prime.front().get_value(q);

          /*--- Compute background quantities ---*/
          const auto& rho_bar  = phi_rho_bar.get_value(q);
          const auto& u_bar    = phi_u_bar.get_value(q);
          const auto& pres_bar = phi_pres_bar.get_value(q);
          Tensor<1, dim, VectorizedArray<Number>> u_bar_tmp;
          if constexpr(dim == 2) {
            u_bar_tmp[0] = u_bar[0];
            u_bar_tmp[1] = u_bar[2];
          }

          /*--- Compute the quantities at the previous stages ---*/
          Tensor<1, dim, VectorizedArray<Number>> flux;
          VectorizedArray<Number> gravity_term = make_vectorized_array<Number>(0.0);
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            const auto& rho_prime_s  = phi_rho_prime[s - 1].get_value(q);
            const auto& u_prime_s    = phi_u_prime[s - 1].get_value(q);
            const auto& pres_prime_s = phi_pres_prime[s - 1].get_value(q);

            if constexpr(dim == 2) {
              Tensor<1, dim, VectorizedArray<Number>> u_prime_s_tmp;
              u_prime_s_tmp[0] = u_prime_s[0];
              u_prime_s_tmp[1] = u_prime_s[2];

              flux += b[s - 1]*dt*((rho_bar + rho_prime_s)*(0.5*Ma2*scalar_product(u_bar + u_prime_s, u_bar + u_prime_s))*(u_bar_tmp + u_prime_s_tmp));
                   // + b_tilde[s - 1]*dt*(inv_Gamma*((pres_bar + pres_prime_s)*(u_bar_tmp + u_prime_s_tmp)));
              flux -= b[s - 1]*dt*((rho_bar)*(0.5*Ma2*scalar_product(u_bar, u_bar))*(u_bar_tmp));
                    //+ b_tilde[s - 1]*dt*(inv_Gamma*((pres_bar)*(u_bar_tmp))));
            }
            else {
              flux += b[s - 1]*dt*
          ((rho_bar + rho_prime_s) *
           (0.5*Ma2 *
            scalar_product(u_bar + u_prime_s,
                           u_bar + u_prime_s)) *
           (u_bar + u_prime_s))
        + b_tilde[s - 1]*dt*
          (inv_Gamma *
           ((pres_bar + pres_prime_s) *
            (u_bar + u_prime_s)));
            flux -= (b[s - 1]*dt*
                      ((rho_bar)*(0.5*Ma2*scalar_product(u_bar, u_bar))*(u_bar))
                    + b_tilde[s - 1]*dt*
                      (inv_Gamma*((pres_bar)*(u_bar))));
            }

            gravity_term += b_tilde[s - 1]*dt*
                            (Ma2_ov_Fr2*(rho_bar + rho_prime_s)*(u_bar[2] + u_prime_s[2]));
            gravity_term -= b_tilde[s - 1]*dt*
                            (Ma2_ov_Fr2*(rho_bar)*(u_bar[2]));
          }

          /*--- We assign to the rhs the contribution due to the (already updated) kinetic energy ---*/
          const auto& rho_prime_curr = phi_rho_prime.back().get_value(q);
          const auto& u_prime_curr   = phi_u_prime.back().get_value(q);

          phi.submit_value(//inv_gamma_m1*pres_prime_old +
                           (rho_bar + rho_prime_old)*(0.5*Ma2*scalar_product(u_bar + u_prime_old, u_bar + u_prime_old)) -
                           (rho_bar + rho_prime_curr)*(0.5*Ma2*scalar_product(u_bar + u_prime_curr, u_bar + u_prime_curr)) - 0.0*gravity_term, q);
          phi.submit_gradient(flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term of the energy equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_face_term_energy(const MatrixFree<dim, Number>&       data,
                                Vec&                                 dst,
                                const std::vector<Vec>&              src,
                                const std::pair<unsigned, unsigned>& face_range) const {
                                
    /*--- Intermediate stages ---*/
    if(IMEX_stage <= n_stages) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_pres phi_m(data, true, EquationData::P_INDEX_DOF),
                            phi_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_prime_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_prime_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_u    phi_u_prime_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_prime_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_prime_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_prime_p(data, false, EquationData::RHO_INDEX_DOF);

      FEFaceEvaluation_rho  phi_rho_bar_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_bar_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u    phi_u_bar_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_bar_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_pres phi_pres_bar_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_bar_p(data, false, EquationData::P_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_prime_m.reinit(face);
        phi_rho_prime_p.reinit(face);
        phi_u_prime_m.reinit(face);
        phi_u_prime_p.reinit(face);
        phi_pres_prime_m.reinit(face);
        phi_pres_prime_p.reinit(face);

        phi_rho_bar_m.reinit(face);
        phi_rho_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);
        phi_rho_bar_p.reinit(face);
        phi_rho_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);
        phi_u_bar_m.reinit(face);
        phi_u_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 4], EvaluationFlags::values);
        phi_u_bar_p.reinit(face);
        phi_u_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 4], EvaluationFlags::values);
        phi_pres_bar_m.reinit(face);
        phi_pres_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 5], EvaluationFlags::values);
        phi_pres_bar_p.reinit(face);
        phi_pres_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 5], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.normal_vector(q);
          Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
          if constexpr(dim == 2) {
            n_minus_tmp[0] = n_minus[0];
            n_minus_tmp[2] = n_minus[1];
          }
          else {
            n_minus_tmp = n_minus;
          }

          /*--- First, focus on background quantities ---*/
          const auto& rho_bar_m  = phi_rho_bar_m.get_value(q);
          const auto& rho_bar_p  = phi_rho_bar_p.get_value(q);
          const auto& u_bar_m    = phi_u_bar_m.get_value(q);
          const auto& u_bar_p    = phi_u_bar_p.get_value(q);
          const auto& pres_bar_m = phi_pres_bar_m.get_value(q);
          const auto& pres_bar_p = phi_pres_bar_p.get_value(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux_num = make_vectorized_array<Number>(0.0);
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            /*--- Retrieve the useful fields ---*/
            phi_rho_prime_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_rho_prime_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_u_prime_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_u_prime_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_pres_prime_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_pres_prime_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& rho_prime_s_m  = phi_rho_prime_m.get_value(q);
            const auto& rho_prime_s_p  = phi_rho_prime_p.get_value(q);
            const auto& u_prime_s_m    = phi_u_prime_m.get_value(q);
            const auto& u_prime_s_p    = phi_u_prime_p.get_value(q);
            const auto& pres_prime_s_m = phi_pres_prime_m.get_value(q);
            const auto& pres_prime_s_p = phi_pres_prime_p.get_value(q);

            /*--- Compute the numerical flux ---*/
            flux_num += a[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_energy_explicit(rho_bar_m + rho_prime_s_m,
                                                                u_bar_m + u_prime_s_m,
                                                                rho_bar_p + rho_prime_s_p,
                                                                u_bar_p + u_prime_s_p,
                                                                n_minus_tmp);/*
                      + a_tilde[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_energy_implicit(u_bar_m + u_prime_s_m,
                                                                pres_bar_m + pres_prime_s_m,
                                                                u_bar_p + u_prime_s_p,
                                                                pres_bar_p + pres_prime_s_p,
                                                                n_minus_tmp);*/
            flux_num -= a[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_energy_explicit(rho_bar_m,
                                                                u_bar_m,
                                                                rho_bar_p,
                                                                u_bar_p,
                                                                n_minus_tmp);/*
                      + a_tilde[IMEX_stage - 1][s - 1]*dt*
                        num_flux.numerical_flux_energy_implicit(u_bar_m,
                                                                pres_bar_m,
                                                                u_bar_p,
                                                                pres_bar_p,
                                                                n_minus_tmp));*/
          }

          /*--- Compute the contribution at the current stage ---*/
          phi_u_prime_m.gather_evaluate(src[3*(IMEX_stage - 1) + 1], EvaluationFlags::values);
          phi_u_prime_p.gather_evaluate(src[3*(IMEX_stage - 1) + 1], EvaluationFlags::values);
          phi_pres_prime_m.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);
          phi_pres_prime_p.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);

          const auto& u_prime_fixed_s_m    = phi_u_prime_m.get_value(q);
          const auto& u_prime_fixed_s_p    = phi_u_prime_p.get_value(q);
          const auto& pres_prime_fixed_s_m = phi_pres_prime_m.get_value(q);
          const auto& pres_prime_fixed_s_p = phi_pres_prime_p.get_value(q);

          /*--- Compute the stabilization term ---*/
          const auto& lambda_fixed_s = num_flux.compute_lambda(u_bar_m + u_prime_fixed_s_m,
                                                               u_bar_p + u_prime_fixed_s_p,
                                                               n_minus_tmp);
          const auto& jump_rho_e_prime_fixed_s = inv_gamma_m1*(pres_prime_fixed_s_m - pres_prime_fixed_s_p);

          flux_num +=0.0* a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                      (0.5*lambda_fixed_s*jump_rho_e_prime_fixed_s);

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    /*--- Final update ---*/
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_pres phi_m(data, true, EquationData::P_INDEX_DOF),
                            phi_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_prime_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_prime_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_u    phi_u_prime_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_prime_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_prime_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_prime_p(data, false, EquationData::RHO_INDEX_DOF);

      FEFaceEvaluation_rho  phi_rho_bar_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_bar_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u    phi_u_bar_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_bar_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_pres phi_pres_bar_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_bar_p(data, false, EquationData::P_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_prime_m.reinit(face);
        phi_rho_prime_p.reinit(face);
        phi_u_prime_m.reinit(face);
        phi_u_prime_p.reinit(face);
        phi_pres_prime_m.reinit(face);
        phi_pres_prime_p.reinit(face);

        phi_rho_bar_m.reinit(face);
        phi_rho_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);
        phi_rho_bar_p.reinit(face);
        phi_rho_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 2], EvaluationFlags::values);
        phi_u_bar_m.reinit(face);
        phi_u_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);
        phi_u_bar_p.reinit(face);
        phi_u_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 3], EvaluationFlags::values);
        phi_pres_bar_m.reinit(face);
        phi_pres_bar_m.gather_evaluate(src[3*(IMEX_stage - 1) + 4], EvaluationFlags::values);
        phi_pres_bar_p.reinit(face);
        phi_pres_bar_p.gather_evaluate(src[3*(IMEX_stage - 1) + 4], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.normal_vector(q);
          Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
          if constexpr(dim == 2) {
            n_minus_tmp[0] = n_minus[0];
            n_minus_tmp[2] = n_minus[1];
          }
          else {
            n_minus_tmp = n_minus;
          }

          /*--- First, focus on background quantities ---*/
          const auto& rho_bar_m  = phi_rho_bar_m.get_value(q);
          const auto& rho_bar_p  = phi_rho_bar_p.get_value(q);
          const auto& u_bar_m    = phi_u_bar_m.get_value(q);
          const auto& u_bar_p    = phi_u_bar_p.get_value(q);
          const auto& pres_bar_m = phi_pres_bar_m.get_value(q);
          const auto& pres_bar_p = phi_pres_bar_p.get_value(q);

          /*--- Compute the quantities at the previous stages ---*/
          VectorizedArray<Number> flux_num = make_vectorized_array<Number>(0.0);
          for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
            /*--- Retrieve the useful fields ---*/
            phi_rho_prime_m.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_rho_prime_p.gather_evaluate(src[3*(s-1)], EvaluationFlags::values);
            phi_u_prime_m.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_u_prime_p.gather_evaluate(src[3*(s-1) + 1], EvaluationFlags::values);
            phi_pres_prime_m.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);
            phi_pres_prime_p.gather_evaluate(src[3*(s-1) + 2], EvaluationFlags::values);

            const auto& rho_prime_s_m  = phi_rho_prime_m.get_value(q);
            const auto& rho_prime_s_p  = phi_rho_prime_p.get_value(q);
            const auto& u_prime_s_m    = phi_u_prime_m.get_value(q);
            const auto& u_prime_s_p    = phi_u_prime_p.get_value(q);
            const auto& pres_prime_s_m = phi_pres_prime_m.get_value(q);
            const auto& pres_prime_s_p = phi_pres_prime_p.get_value(q);

            /*--- Compute the numerical flux ---*/
            flux_num += b[s - 1]*dt*
                        num_flux.numerical_flux_energy_explicit(rho_bar_m + rho_prime_s_m,
                                                                u_bar_m + u_prime_s_m,
                                                                rho_bar_p + rho_prime_s_p,
                                                                u_bar_p + u_prime_s_p,
                                                                n_minus_tmp);/*
                      + b_tilde[s - 1]*dt*
                        num_flux.numerical_flux_energy_implicit(u_bar_m + u_prime_s_m,
                                                                pres_bar_m + pres_prime_s_m,
                                                                u_bar_p + u_prime_s_p,
                                                                pres_bar_p + pres_prime_s_p,
                                                                n_minus_tmp);*/
            flux_num -= b[s - 1]*dt*
                        num_flux.numerical_flux_energy_explicit(rho_bar_m,  
                                                                u_bar_m,
                                                                rho_bar_p,
                                                                u_bar_p,
                                                                n_minus_tmp);/*
                      + b_tilde[s - 1]*dt*
                        num_flux.numerical_flux_energy_implicit(u_bar_m,
                                                                pres_bar_m,
                                                                u_bar_p,
                                                                pres_bar_p,
                                                                n_minus_tmp));*/
          }

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the energy equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_rhs_energy(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_energy,
                     &EULEROperator::assemble_rhs_face_term_energy,
                     &EULEROperator::assemble_rhs_boundary_term_energy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble cell term for the contribution due to internal energy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_inverse_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                             Vec&                                 dst,
                                             const Vec&                           src,
                                             const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, fe_degree_p + 1, 1, Number> phi(data, EquationData::P_INDEX_DOF, 4);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree_p, 1, Number> inverse(phi);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);

      AlignedVector<VectorizedArray<Number>> inverse_jxw(phi.n_q_points);
      inverse.fill_inverse_JxW_values(inverse_jxw);

      /*--- Loop over all quadrature points to fill the inverse of the coefficient ---*/
      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        inverse_jxw[q] *= gamma_m1;
      }

      inverse.apply(inverse_jxw, 1,
                    phi.begin_dof_values(),
                    phi.begin_dof_values());

      phi.set_dof_values(dst);
    }
  }

  // Assemble cell term for the contribution due to internal energy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, fe_degree_p + 1, 1, Number> phi(data, EquationData::P_INDEX_DOF, 4);

    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        /*--- For an ideal gas the part associated to the internal energy for a pressure based
              is just a modification of the mass matrix ---*/
        phi.submit_value(inv_gamma_m1*phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Assemble cell term for the contribution due to enthalpy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities.
          Since here we have just one 'src' vector, but we also need to deal with the current pressure
          in the fixed point loop, we employ the auxiliary vector 'pres_fixed' where we setted this information ---*/
    FEEvaluation_pres phi(data, EquationData::P_INDEX_DOF),
                      phi_pres_fixed(data, EquationData::P_INDEX_DOF);
    FEEvaluation_u    phi_src(data, EquationData::U_INDEX_DOF);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres_fixed.reinit(cell);
      phi_pres_fixed.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- loop over all quadrature points ---*/
      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        const auto& pres_fixed = phi_pres_fixed.get_value(q);

        const auto& u_fixed = phi_src.get_value(q);
        if constexpr(dim == 2) {
          Tensor<1, dim, VectorizedArray<Number>> u_fixed_tmp;
          u_fixed_tmp[0] = u_fixed[0];
          u_fixed_tmp[1] = u_fixed[2];

          phi.submit_gradient(-a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                              (inv_Gamma*(pres_fixed*u_fixed_tmp)), q);
        }
        else {
          phi.submit_gradient(-a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                              (inv_Gamma*(pres_fixed*u_fixed)), q);
        }
      }

      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the contribution due to enthalpy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_face_term_enthalpy(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& face_range) const {
    FEFaceEvaluation_pres phi_m(data, true, EquationData::P_INDEX_DOF),
                          phi_p(data, false, EquationData::P_INDEX_DOF),
                          phi_pres_fixed_m(data, true, EquationData::P_INDEX_DOF),
                          phi_pres_fixed_p(data, false, EquationData::P_INDEX_DOF);
    FEFaceEvaluation_u    phi_src_m(data, true, EquationData::U_INDEX_DOF),
                          phi_src_p(data, false, EquationData::U_INDEX_DOF);

    /*--- Loop over all faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_pres_fixed_m.reinit(face);
      phi_pres_fixed_m.gather_evaluate(pres_fixed, EvaluationFlags::values);
      phi_pres_fixed_p.reinit(face);
      phi_pres_fixed_p.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, EvaluationFlags::values);
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, EvaluationFlags::values);

      phi_m.reinit(face);
      phi_p.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
        const auto& n_minus = phi_m.normal_vector(q);
        Tensor<1, 3, VectorizedArray<Number>> n_minus_tmp;
        if constexpr(dim == 2) {
          n_minus_tmp[0] = n_minus[0];
          n_minus_tmp[2] = n_minus[1];
        }
        else {
          n_minus_tmp = n_minus;
        }

        const auto& pres_fixed_m      = phi_pres_fixed_m.get_value(q);
        const auto& pres_fixed_p      = phi_pres_fixed_p.get_value(q);

        const auto& avg_flux_enthalpy = 0.5*inv_Gamma*
                                        (pres_fixed_m*phi_src_m.get_value(q) +
                                         pres_fixed_p*phi_src_p.get_value(q));

        const auto& flux_num          = a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                                        scalar_product(avg_flux_enthalpy, n_minus_tmp);

        phi_m.submit_value(flux_num, q);
        phi_p.submit_value(-flux_num, q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  //////////////////////////////////////////////////////////////
  /*---- APPLICATION OF THE DIFFERENT LINEAR OPERATORS ---*/
  /////////////////////////////////////////////////////////////

  // Put together all previous steps
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(Euler_stage, EquationData::n_vars + 1);
    Assert(Euler_stage > 0, ExcInternalError());

    if(Euler_stage == EquationData::RHO_INDEX_SYSTEM) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_density,
                            this, dst, src, false);
    }
    else if(Euler_stage == EquationData::P_INDEX_SYSTEM) {
      if(IMEX_stage <= n_stages) {
        this->data->cell_loop(&EULEROperator::assemble_cell_term_internal_energy,
                              this, dst, src, false);

        /*--- Implementation of the Schur complement operations ---*/
        Vec tmp_1;
        this->data->initialize_dof_vector(tmp_1, EquationData::U_INDEX_DOF);
        this->vmult_pressure(tmp_1, src);

        Euler_stage = EquationData::U_INDEX_SYSTEM;
        const std::vector<unsigned> index_dof_handler_reinit = {EquationData::U_INDEX_DOF};
        auto* tmp_matrix = const_cast<EULEROperator*>(this);
        Vec tmp_2;
        this->data->initialize_dof_vector(tmp_2, EquationData::U_INDEX_DOF);
        tmp_matrix->initialize(tmp_matrix->get_matrix_free(), index_dof_handler_reinit, index_dof_handler_reinit);
        this->vmult(tmp_2, tmp_1);

        Vec tmp_3;
        this->data->initialize_dof_vector(tmp_3, EquationData::P_INDEX_DOF);
        this->vmult_enthalpy(tmp_3, tmp_2);

        dst.add(static_cast<Number>(-1.0), tmp_3);
        Euler_stage = EquationData::P_INDEX_SYSTEM;
        const std::vector<unsigned> index_dof_handler = {EquationData::P_INDEX_DOF};
        tmp_matrix->initialize(tmp_matrix->get_matrix_free(), index_dof_handler, index_dof_handler);
        tmp_matrix->compute_diagonal();
      }
      else {
        this->data->cell_loop(&EULEROperator::assemble_inverse_cell_term_internal_energy,
                              this, dst, src, false);
      }
    }
    else if(Euler_stage == EquationData::U_INDEX_SYSTEM) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_velocity,
                            this, dst, src, false);
    }
    else {
      Assert(false, ExcInternalError());
    }
  }


  // Application of pressure matrix
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_pressure(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&EULEROperator::assemble_cell_term_pressure,
                     &EULEROperator::assemble_face_term_pressure,
                     &EULEROperator::assemble_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }


  // Application of enthalpy matrix
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_enthalpy(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&EULEROperator::assemble_cell_term_enthalpy,
                     &EULEROperator::assemble_face_term_enthalpy,
                     &EULEROperator::assemble_boundary_term_enthalpy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }


  //////////////////////////////////////////////////////////////
  /*---- COMPUTE DIAGONALS---*/
  /////////////////////////////////////////////////////////////

  // Assemble diagonal cell term for the density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_diagonal_cell_term_density(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const unsigned&                          ,
                                      const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, fe_degree_rho + 1, 1, Number> phi(data, EquationData::RHO_INDEX_DOF, 2);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the velocity update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const unsigned&                      ,
                                       const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation_u   phi(data, EquationData::U_INDEX_DOF);
    FEEvaluation_rho phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF);

    /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
          a vector which is 1 for the node of interest and 0 elsewhere. This is what 'tmp_diagonal_velocity' does.
          Moreover, since here we have just one 'src' vector, but we also need to deal with the current density,
          we employ the auxiliary vector 'rho_for_fixed' where we setted this information ---*/
    AlignedVector<Tensor<1, 3, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(Tensor<1, 3, VectorizedArray<Number>>(), j);
        }
        phi.submit_dof_value(tmp_diagonal_velocity, i);
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi_rho_for_fixed.get_value(q)*phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the pressure updated with Schur complement
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const unsigned&                      ,
                                       const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation_pres phi(data, EquationData::P_INDEX_DOF),
                      phi_pres_fixed(data, EquationData::P_INDEX_DOF);
    FEEvaluation_rho  phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres_fixed.reinit(cell);
      phi_pres_fixed.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          const auto& pres_fixed    = phi_pres_fixed.get_value(q);

          const auto& rho_for_fixed = phi_rho_for_fixed.get_value(q);

          phi.submit_value(inv_gamma_m1*phi.get_value(q), q);
          phi.submit_gradient((a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt)*(a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt)*inv_Ma2*
                              (inv_Gamma*(pres_fixed/rho_for_fixed)*phi.get_gradient(q)), q);
        }

        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the contribution due to internal energy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                              Vec&                                 dst,
                                              const unsigned&                      ,
                                              const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, fe_degree_p + 1, 1, Number> phi(data, EquationData::P_INDEX_DOF, 4);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(inv_gamma_m1*phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Compute diagonal of various steps
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  compute_diagonal() {
    AssertIndexRange(Euler_stage, EquationData::n_vars + 1);
    Assert(Euler_stage > 0, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned dummy = 0;

    if(Euler_stage == EquationData::RHO_INDEX_SYSTEM) {
      this->data->initialize_dof_vector(inverse_diagonal, EquationData::RHO_INDEX_DOF);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_density,
                            this, inverse_diagonal, dummy, false);
    }
    else if(Euler_stage == EquationData::P_INDEX_SYSTEM) {
      this->data->initialize_dof_vector(inverse_diagonal, EquationData::P_INDEX_DOF);

      if(IMEX_stage <= n_stages) {
        this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_pressure,
                              this, inverse_diagonal, dummy, false);
      }
      else {
        this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_internal_energy,
                              this, inverse_diagonal, dummy, false);
      }
    }
    else if(Euler_stage == EquationData::U_INDEX_SYSTEM) {
      this->data->initialize_dof_vector(inverse_diagonal, EquationData::U_INDEX_DOF);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_velocity,
                            this, inverse_diagonal, dummy, false);
    }
    else {
      Assert(false, ExcInternalError());
    }

    /*--- For the preconditioner, we actually need the inverse of the diagonal ---*/
    for(unsigned i = 0; i < inverse_diagonal.locally_owned_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != static_cast<Number>(0.0),
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = static_cast<Number>(1.0)/inverse_diagonal.local_element(i);
    }
  }
}
 
