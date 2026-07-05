/* Author: Giuseppe Orlando, 2026. */
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
#include "include/time_integrator/runge_kutta.h"

// This is the class that implements the discretization of the viscous operator
//
namespace Turbulent_Diffusivity {
  using namespace dealii;

  // @sect{ <code>TurbulentOperator::TurbulentOperator</code> }
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  class TurbulentOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    TurbulentOperator(); /*--- Default constructor ---*/

    TurbulentOperator(const RunTimeParameters::Data_Storage& data,
                      const TimeStepping::RungeKutta<Number>& implicit_RK); /*--- Constructor with some input related data ---*/

    template<typename T>
    inline DEAL_II_ALWAYS_INLINE
    void set_dt(const T time_step); /*--- Setter of the time-step. This is useful in case of modifications of the time step. ---*/

    inline DEAL_II_ALWAYS_INLINE
    void set_IMEX_stage(const unsigned stage); /*--- Setter of the IMEX stage. ---*/

    inline DEAL_II_ALWAYS_INLINE
    void set_NS_stage(const unsigned stage); /*--- Setter of the equation currently under solution. ---*/

    void set_u_curr(const Vec& src); /*--- Setter of the current velocity. This is for the assembling of the bilinear forms
                                           where only one source vector can be passed in input to linearize diffusion coefficient. ---*/

    void set_theta_curr(const Vec& src); /*--- Setter of the current potential temperature. This is for the assembling of the bilinear forms
                                               where only one source vector can be passed in input to linearize diffusion coefficient. ---*/

    void vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs for the velocity. ---*/

    void vmult_rhs_temperature(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs for the temperature. ---*/

    virtual void compute_diagonal() override; /*--- Compute the diagonal for several preconditioners ---*/

  protected:
    /*--- Define typedef for sake of readability and convenience ----*/
    using FEEvaluation_u     = FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number>;
    using FEEvaluation_theta = FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>;

    using FEFaceEvaluation_u     = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number>;
    using FEFaceEvaluation_theta = FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>;

    using FEFaceEvaluation_u_boundary     = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_boundary, dim, Number>;
    using FEFaceEvaluation_theta_boundary = FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_boundary, 1, Number>;

    Number dt; /*--- Time step. ---*/

    std::vector<std::vector<Number>> a_tilde; /*--- Classical Butcher tableau notation.
                                                    We assume a stiffly-accurate implicit scheme ----*/

    unsigned n_stages; /*--- Number of stages ---*/

    unsigned IMEX_stage; /*--- Flag for the IMEX stage ---*/
    unsigned NS_stage;   /*--- Flag for the equation actually solved ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    /*--- Parameters related to IP ---*/
    Number C_T = static_cast<Number>(1.0)*(fe_degree_T + 1)*(fe_degree_T + 1);
    Number C_u = static_cast<Number>(1.0)*(fe_degree_u + 1)*(fe_degree_u + 1);

    Vec u_curr,
        theta_curr;

    Number inv_Fr2;   /*--- Inverse of squared Froude number ---*/
    Number l2_mixing; /*--- Square of mixing length ---*/

    Tensor<1, dim, VectorizedArray<Number>> tmp_diagonal_velocity; /*--- Auxiliary vector to compute the diagonal of the velocity matrix ----*/

    /*--- Assembler functions for the rhs related to the velocity equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const std::vector<Vec>&              src,
                                         const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const std::vector<Vec>&              src,
                                         const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&       data,
                                             Vec&                                 dst,
                                             const std::vector<Vec>&              src,
                                             const std::pair<unsigned, unsigned>& face_range) const {}

    /*--- Assembler functions related to the bilinear form of the velocity equation. ---*/
    void assemble_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_face_term_velocity(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_boundary_term_velocity(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const Vec&                           src,
                                         const std::pair<unsigned, unsigned>& face_range) const {}

    /*--- Assembler functions for the rhs related to the potential temperature equation. ---*/
    void assemble_rhs_cell_term_temperature(const MatrixFree<dim, Number>&       data,
                                            Vec&                                 dst,
                                            const std::vector<Vec>&              src,
                                            const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_rhs_face_term_temperature(const MatrixFree<dim, Number>&       data,
                                            Vec&                                 dst,
                                            const std::vector<Vec>&              src,
                                            const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_rhs_boundary_term_temperature(const MatrixFree<dim, Number>&       data,
                                                Vec&                                 dst,
                                                const std::vector<Vec>&              src,
                                                const std::pair<unsigned, unsigned>& face_range) const {}

    /*--- Assembler functions for the potential temperature equation. ---*/
    void assemble_cell_term_temperature(const MatrixFree<dim, Number>&       data,
                                        Vec&                                 dst,
                                        const Vec&                           src,
                                        const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_face_term_temperature(const MatrixFree<dim, Number>&       data,
                                        Vec&                                 dst,
                                        const Vec&                           src,
                                        const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_boundary_term_temperature(const MatrixFree<dim, Number>&       data,
                                            Vec&                                 dst,
                                            const Vec&                           src,
                                            const std::pair<unsigned, unsigned>& face_range) const {}

    /*--- Assembler functions for the diagonal part of the matrix for the velocity equation. ---*/
    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                              Vec&                                 dst,
                                              const unsigned&                      src,
                                              const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&       data,
                                              Vec&                                 dst,
                                              const unsigned&                      src,
                                              const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&       data,
                                                  Vec&                                 dst,
                                                  const unsigned&                      src,
                                                  const std::pair<unsigned, unsigned>& face_range) const {}

    /*--- Assembler functions for the diagonal part of the matrix for the potential temperature equation. ---*/
    void assemble_diagonal_cell_term_temperature(const MatrixFree<dim, Number>&       data,
                                                 Vec&                                 dst,
                                                 const unsigned&                      src,
                                                 const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_diagonal_face_term_temperature(const MatrixFree<dim, Number>&               data,
                                                 Vec&                                         dst,
                                                 const unsigned&                          src,
                                                 const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_diagonal_boundary_term_temperature(const MatrixFree<dim, Number>&               data,
                                                     Vec&                                         dst,
                                                     const unsigned&                          src,
                                                     const std::pair<unsigned, unsigned>& face_range) const {}
  };


  //////////////////////////////////////////////////////////////
  /*---- START WITH CLASS CONSTRUCTORS ---*/
  /////////////////////////////////////////////////////////////

  // Default constructor
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  TurbulentOperator<dim,
                    fe_degree_T, fe_degree_u,
                    n_q_points_1d, n_q_points_1d_boundary, Vec>::
  TurbulentOperator():
    MatrixFreeOperators::Base<dim, Vec>(),
    dt(), n_stages(), IMEX_stage(1), NS_stage(1),
    inv_Fr2(), l2_mixing()
    {
      /*--- We create an auxiliary vector that will never change
            independently on the stage, so we declare it once and for all. ---*/
      for(unsigned d = 0; d < dim; ++d) {
        tmp_diagonal_velocity[d] = make_vectorized_array<Number>(1.0);
      }
    }

  // Constructor with runtime parameters storage
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  TurbulentOperator<dim,
                    fe_degree_T, fe_degree_u,
                    n_q_points_1d, n_q_points_1d_boundary,
                    Vec>::
  TurbulentOperator(const RunTimeParameters::Data_Storage& data,
                    const TimeStepping::RungeKutta<Number>& implicit_RK):
    MatrixFreeOperators::Base<dim, Vec>(),
    dt(data.dt),
    n_stages(implicit_RK.get_n_stages()), IMEX_stage(1), NS_stage(1),
    inv_Fr2(static_cast<Number>(1.0)/
            (static_cast<Number>(data.Froude)*static_cast<Number>(data.Froude))),
    l2_mixing(static_cast<Number>(data.l_mixing)*static_cast<Number>(data.l_mixing))
    {
      implicit_RK.get_coefficients(a_tilde);

      /*--- We create an auxiliary vector that will never change
            independently on the stage, so we declare it once and for all. ---*/
      for(unsigned d = 0; d < dim; ++d) {
        tmp_diagonal_velocity[d] = make_vectorized_array<Number>(1.0);
      }
    }


  //////////////////////////////////////////////////////////////
  /*---- FOCUS NOW ON SOME AUXILIARY SETTERS ---*/
  /////////////////////////////////////////////////////////////

  // Setter of time-step
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  template<typename T>
  inline DEAL_II_ALWAYS_INLINE
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  set_dt(const T time_step) {
    dt = static_cast<Number>(time_step);
  }

  // Setter of IMEX stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  set_IMEX_stage(const unsigned stage) {
    AssertIndexRange(stage, n_stages + 1);
    Assert(stage > 0, ExcInternalError());

    IMEX_stage = stage;
  }

  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  set_NS_stage(const unsigned stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());

    NS_stage = stage;
  }

  // Setter of current velocity
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  set_u_curr(const Vec& src) {
    u_curr = src;
    u_curr.update_ghost_values();
  }

  // Setter of current potential temperature
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  set_theta_curr(const Vec& src) {
    theta_curr = src;
    theta_curr.update_ghost_values();
  }


  //////////////////////////////////////////////////////////////
  /*---- ASSEMBLING LINEAR AND BILINEAR FORMS FOR THE MOMENTUM (VELOCITY) EQUATION ---*/
  /////////////////////////////////////////////////////////////

  // Assemble rhs cell term for the velocity equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                  Vec&                                 dst,
                                  const std::vector<Vec>&              src,
                                  const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEEvaluation_u                  phi(data, EquationData::U_INDEX_DOF);
    std::vector<FEEvaluation_u>     phi_u(IMEX_stage - 1, FEEvaluation_u(data, EquationData::U_INDEX_DOF));
    std::vector<FEEvaluation_theta> phi_theta(IMEX_stage - 1, FEEvaluation_theta(data, EquationData::THETA_INDEX_DOF));

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
        phi_u[s - 1].reinit(cell);
        phi_u[s - 1].gather_evaluate(src[2*(s-1)], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta[s - 1].reinit(cell);
        phi_theta[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::gradients);
      }

      phi.reinit(cell);

      /*--- Loop over all quadrature points. ---*/
      for(const unsigned q : phi.quadrature_point_indices()) {
        /*--- Compute the velocity after hyperbolic operator (always needed).
              Notice this is ok because of ESDIRK method ---*/
        const auto& u_curr = phi_u.front().get_value(q);

        /*--- Compute contribution of the flux at each stage ---*/
        Tensor<2, dim, VectorizedArray<Number>> diff_flux;
        diff_flux = 0;
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          const auto& grad_u_s     = phi_u[s - 1].get_gradient(q);
          const auto& grad_theta_s = phi_theta[s - 1].get_gradient(q);

          const auto& mod_squared_grad_uz_s = grad_u_s[0][dim - 1]*grad_u_s[0][dim - 1];
          const auto& Ri_s                  = inv_Fr2*grad_theta_s[dim - 1]/mod_squared_grad_uz_s;

          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_s[idx] > static_cast<Number>(0.0)) {
              b[idx]    = static_cast<Number>(5.0);
              beta[idx] = static_cast<Number>(-2.0);
            }
            else {
              b[idx]    = static_cast<Number>(20.0);
              beta[idx] = static_cast<Number>(0.5);
            }
          }
          const auto& kappa_s = l2_mixing*std::sqrt(mod_squared_grad_uz_s)*
                                std::pow(static_cast<Number>(1.0) + b*std::abs(Ri_s), beta);

          Tensor<2, dim, VectorizedArray<Number>> diff_flux_s;
          diff_flux_s = 0;
          diff_flux_s[0][dim - 1] = kappa_s*grad_u_s[0][dim - 1];

          diff_flux += a_tilde[IMEX_stage - 1][s - 1]*dt*diff_flux_s;
        }

        phi.submit_value(u_curr, q);
        phi.submit_gradient(-diff_flux, q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }

  // Assemble rhs face term for the velocity equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&       data,
                                  Vec&                                 dst,
                                  const std::vector<Vec>&              src,
                                  const std::pair<unsigned, unsigned>& face_range) const {
    /*--- We start by declaring suitable quantities to read the available quantities ---*/
    FEFaceEvaluation_u     phi_m(data, true, EquationData::U_INDEX_DOF),
                           phi_p(data, false, EquationData::U_INDEX_DOF),
                           phi_u_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_p(data, false, EquationData::U_INDEX_DOF);
    FEFaceEvaluation_theta phi_theta_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_theta_p(data, false, EquationData::THETA_INDEX_DOF);

    /*--- Loop over all internal faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_u_m.reinit(face);
      phi_u_p.reinit(face);
      phi_theta_m.reinit(face);
      phi_theta_p.reinit(face);

      phi_m.reinit(face);
      phi_p.reinit(face);

      const auto coef_jump = C_u*(static_cast<Number>(0.5)*
                                  (std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
                                   std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]))); /*--- Jump constant for IP ---*/

      /*--- Loop over all quadrature points ---*/
      for(const unsigned q : phi_m.quadrature_point_indices()) {
        const auto& n_minus = phi_m.get_normal_vector(q);

        /*--- Compute the quantities at the previous stages ---*/
        Tensor<1, dim, VectorizedArray<Number>> IP_flux_num;
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          /*--- Retrieve the useful fields ---*/
          phi_u_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::values | EvaluationFlags::gradients);
          phi_u_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::values | EvaluationFlags::gradients);
          phi_theta_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::gradients);
          phi_theta_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::gradients);

          const auto& grad_u_s_m     = phi_u_m.get_gradient(q);
          const auto& grad_u_s_p     = phi_u_p.get_gradient(q);
          const auto& grad_theta_s_m = phi_theta_m.get_gradient(q);
          const auto& grad_theta_s_p = phi_theta_p.get_gradient(q);

          const auto& mod_squared_grad_uz_s_m = grad_u_s_m[0][dim - 1]*grad_u_s_m[0][dim - 1];
          const auto& mod_squared_grad_uz_s_p = grad_u_s_p[0][dim - 1]*grad_u_s_p[0][dim - 1];
          const auto& Ri_s_m                  = inv_Fr2*grad_theta_s_m[dim - 1]/mod_squared_grad_uz_s_m;
          const auto& Ri_s_p                  = inv_Fr2*grad_theta_s_p[dim - 1]/mod_squared_grad_uz_s_p;

          VectorizedArray<Number> b_m, b_p;
          VectorizedArray<Number> beta_m, beta_p;
          for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_s_m[idx] > static_cast<Number>(0.0)) {
              b_m[idx]    = static_cast<Number>(5.0);
              beta_m[idx] = static_cast<Number>(-2.0);
            }
            else {
              b_m[idx]    = static_cast<Number>(20.0);
              beta_m[idx] = static_cast<Number>(0.5);
            }

            if(Ri_s_p[idx] > static_cast<Number>(0.0)) {
              b_p[idx]    = static_cast<Number>(5.0);
              beta_p[idx] = static_cast<Number>(-2.0);
            }
            else {
              b_p[idx]    = static_cast<Number>(20.0);
              beta_p[idx] = static_cast<Number>(0.5);
            }
          }
          const auto& kappa_s_m = l2_mixing*std::sqrt(mod_squared_grad_uz_s_m)*
                                  std::pow(static_cast<Number>(1.0) + b_m*std::abs(Ri_s_m), beta_m);
          const auto& kappa_s_p = l2_mixing*std::sqrt(mod_squared_grad_uz_s_p)*
                                  std::pow(static_cast<Number>(1.0) + b_p*std::abs(Ri_s_p), beta_p);

          Tensor<2, dim, VectorizedArray<Number>> avg_diff_flux_s;
          avg_diff_flux_s = 0;
          avg_diff_flux_s[0][dim - 1] = static_cast<Number>(0.5)*
                                        (kappa_s_m*grad_u_s_m[0][dim - 1] +
                                         kappa_s_p*grad_u_s_p[0][dim - 1]);

          /*--- Consider also jump penalization contribution ---*/
          const auto& u_s_m       = phi_u_m.get_value(q);
          const auto& u_s_p       = phi_u_p.get_value(q);
          const auto& avg_kappa_s = static_cast<Number>(2.0)/
                                    (static_cast<Number>(1.0)/kappa_s_m +
                                     static_cast<Number>(1.0)/kappa_s_p);
          auto jump_u_s           = u_s_m - u_s_p;
          jump_u_s[dim - 1]       = make_vectorized_array<Number>(0.0);

          /*--- Compute the numerical flux ---*/
          IP_flux_num += a_tilde[IMEX_stage - 1][s - 1]*dt*
                         (avg_diff_flux_s*n_minus -
                          coef_jump*avg_kappa_s*jump_u_s);
        }

        phi_m.submit_value(IP_flux_num, q);
        phi_p.submit_value(-IP_flux_num, q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Put together all the previous steps for the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&TurbulentOperator::assemble_rhs_cell_term_velocity,
                     &TurbulentOperator::assemble_rhs_face_term_velocity,
                     &TurbulentOperator::assemble_rhs_boundary_term_velocity,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }

  // Assemble cell term for the velocity equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We start declaring suitable instances to read the available quantities ---*/
    FEEvaluation_u     phi(data, EquationData::U_INDEX_DOF),
                       phi_u_curr(data, EquationData::U_INDEX_DOF);
    FEEvaluation_theta phi_theta_curr(data, EquationData::THETA_INDEX_DOF);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr.reinit(cell);
      phi_theta_curr.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      /*--- Loop over all quadrature points ---*/
      for(const unsigned q : phi.quadrature_point_indices()) {
        const auto& u = phi.get_value(q);

        /*--- Compute contribution at current stage ---*/
        const auto& grad_u_curr     = phi_u_curr.get_gradient(q);
        const auto& grad_theta_curr = phi_theta_curr.get_gradient(q);

        const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
        const auto& Ri_curr                  = inv_Fr2*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
        VectorizedArray<Number> b;
        VectorizedArray<Number> beta;
        for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
          if(Ri_curr[idx] > static_cast<Number>(0.0)) {
            b[idx]    = 5.0;
            beta[idx] = -2.0;
          }
          else {
            b[idx]    = 20.0;
            beta[idx] = 0.5;
          }
        }
        const auto& kappa_curr = l2_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                 std::pow(1.0 + b*std::abs(Ri_curr), beta);

        Tensor<2, dim, VectorizedArray<Number>> diff_flux;
        diff_flux = 0;
        const auto& grad_u    = phi.get_gradient(q);
        diff_flux[0][dim - 1] = kappa_curr*grad_u[0][dim - 1];

        phi.submit_value(phi.get_value(q), q);
        phi.submit_gradient(a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*diff_flux, q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the velocity equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_face_term_velocity(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& face_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEFaceEvaluation_u     phi_m(data, true, EquationData::U_INDEX_DOF),
                           phi_p(data, false, EquationData::U_INDEX_DOF),
                           phi_u_curr_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_curr_p(data, false, EquationData::U_INDEX_DOF);
    FEFaceEvaluation_theta phi_theta_curr_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_theta_curr_p(data, false, EquationData::THETA_INDEX_DOF);

    /*--- Loop over all internal faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_u_curr_m.reinit(face);
      phi_u_curr_m.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_u_curr_p.reinit(face);
      phi_u_curr_p.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr_m.reinit(face);
      phi_theta_curr_m.gather_evaluate(theta_curr, EvaluationFlags::gradients);
      phi_theta_curr_p.reinit(face);
      phi_theta_curr_p.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi_m.reinit(face);
      phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      const auto coef_jump = C_u*(static_cast<Number>(0.5)*
                                  (std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
                                   std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]))); /*--- Jump constant for IP ---*/

      /*--- Loop over all quadrature points ---*/
      for(const unsigned q : phi_m.quadrature_point_indices()) {
        const auto& n_minus = phi_m.get_normal_vector(q);

        /*--- Compute contribution at current stage ---*/
        const auto& grad_u_curr_m     = phi_u_curr_m.get_gradient(q);
        const auto& grad_u_curr_p     = phi_u_curr_p.get_gradient(q);
        const auto& grad_theta_curr_m = phi_theta_curr_m.get_gradient(q);
        const auto& grad_theta_curr_p = phi_theta_curr_p.get_gradient(q);

        const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
        const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
        const auto& Ri_curr_m                  = inv_Fr2*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
        const auto& Ri_curr_p                  = inv_Fr2*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
        VectorizedArray<Number> b_m, b_p;
        VectorizedArray<Number> beta_m, beta_p;
        for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
          if(Ri_curr_m[idx] > static_cast<Number>(0.0)) {
            b_m[idx]    = static_cast<Number>(5.0);
            beta_m[idx] = static_cast<Number>(-2.0);
          }
          else {
            b_m[idx]    = static_cast<Number>(20.0);
            beta_m[idx] = static_cast<Number>(0.5);
          }

          if(Ri_curr_p[idx] > static_cast<Number>(0.0)) {
            b_p[idx]    = static_cast<Number>(5.0);
            beta_p[idx] = static_cast<Number>(-2.0);
          }
          else {
            b_p[idx]    = static_cast<Number>(20.0);
            beta_p[idx] = static_cast<Number>(0.5);
          }
        }
        const auto& kappa_curr_m = l2_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                   std::pow(static_cast<Number>(1.0) + b_m*std::abs(Ri_curr_m), beta_m);
        const auto& kappa_curr_p = l2_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                   std::pow(static_cast<Number>(1.0) + b_p*std::abs(Ri_curr_p), beta_p);

        Tensor<2, dim, VectorizedArray<Number>> avg_diff_flux;
        avg_diff_flux = 0;
        const auto& grad_u_m      = phi_m.get_gradient(q);
        const auto& grad_u_p      = phi_p.get_gradient(q);
        avg_diff_flux[0][dim - 1] = static_cast<Number>(0.5)*
                                    (kappa_curr_m*grad_u_m[0][dim - 1] +
                                     kappa_curr_p*grad_u_p[0][dim - 1]);

        /*--- Consider jump penalization ---*/
        const auto& u_m            = phi_m.get_value(q);
        const auto& u_p            = phi_p.get_value(q);
        const auto& avg_kappa_curr = static_cast<Number>(2.0)/
                                     (static_cast<Number>(1.0)/kappa_curr_m +
                                      static_cast<Number>(1.0)/kappa_curr_p);
        auto jump_u                = u_m - u_p;
        jump_u[dim - 1]            = make_vectorized_array<Number>(0.0);

        /*-- Compute the numerical flux ---*/
        const auto& IP_flux_num = a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                                  (-avg_diff_flux*n_minus +
                                    coef_jump*avg_kappa_curr*jump_u);

        phi_m.submit_value(IP_flux_num, q);
        phi_p.submit_value(-IP_flux_num, q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  //////////////////////////////////////////////////////////////
  /*---- ASSEMBLING LINEAR AND BILINEAR FORMS FOR THE ENERGY (POTENTIAL TEMPERATURE) EQUATION ---*/
  /////////////////////////////////////////////////////////////

  // Assemble rhs cell term for the the potential temperature equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_rhs_cell_term_temperature(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const std::vector<Vec>&              src,
                                     const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEEvaluation_theta              phi(data, EquationData::THETA_INDEX_DOF);
    std::vector<FEEvaluation_u>     phi_u(IMEX_stage - 1, FEEvaluation_u(data, EquationData::U_INDEX_DOF));
    std::vector<FEEvaluation_theta> phi_theta(IMEX_stage - 1, FEEvaluation_theta(data, EquationData::THETA_INDEX_DOF));

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
        phi_u[s - 1].reinit(cell);
        phi_u[s - 1].gather_evaluate(src[2*(s-1)], EvaluationFlags::gradients);
        phi_theta[s - 1].reinit(cell);
        phi_theta[s - 1].gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values | EvaluationFlags::gradients);
      }

      phi.reinit(cell);

      /*--- Loop over all quadrature points ---*/
      for(const unsigned q : phi.quadrature_point_indices()) {
        /*--- Compute the potential temperature after hyperbolic operator (always needed).
              Notice this is ok because of ESDIRK method ---*/
        const auto& theta_curr = phi_theta.front().get_value(q);

        /*--- Compute the contribution at the previous stages ---*/
        Tensor<1, dim, VectorizedArray<Number>> diff_flux;
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          const auto& grad_u_s     = phi_u[s - 1].get_gradient(q);
          const auto& grad_theta_s = phi_theta[s - 1].get_gradient(q);

          const auto& mod_squared_grad_uz_s = grad_u_s[0][dim - 1]*grad_u_s[0][dim - 1];
          const auto& Ri_s                  = inv_Fr2*grad_theta_s[dim - 1]/mod_squared_grad_uz_s;

          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_s[idx] > static_cast<Number>(0.0)) {
              b[idx]    = static_cast<Number>(5.0);
              beta[idx] = static_cast<Number>(-2.0);
            }
            else {
              b[idx]    = static_cast<Number>(20.0);
              beta[idx] = static_cast<Number>(0.5);
            }
          }
          const auto& kappa_s = l2_mixing*std::sqrt(mod_squared_grad_uz_s)*
                                std::pow(static_cast<Number>(1.0) + b*std::abs(Ri_s), beta);

          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_s;
          diff_tensor_s = 0;
          diff_tensor_s[dim - 1][dim - 1] = kappa_s;

          diff_flux += a_tilde[IMEX_stage - 1][s - 1]*dt*(diff_tensor_s*grad_theta_s);
        }

        phi.submit_value(theta_curr, q);
        phi.submit_gradient(-diff_flux, q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }

  // Assemble rhs face term for the temperature equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_rhs_face_term_temperature(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const std::vector<Vec>&              src,
                                     const std::pair<unsigned, unsigned>& face_range) const {
    /*--- We start by declaring suitable quantities to read the available quantities ---*/
    FEFaceEvaluation_theta phi_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_p(data, false, EquationData::THETA_INDEX_DOF),
                           phi_theta_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_theta_p(data, false, EquationData::THETA_INDEX_DOF);
    FEFaceEvaluation_u     phi_u_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_p(data, false, EquationData::U_INDEX_DOF);

    /*--- Loop over all internal faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_u_m.reinit(face);
      phi_u_p.reinit(face);
      phi_theta_m.reinit(face);
      phi_theta_p.reinit(face);

      phi_m.reinit(face);
      phi_p.reinit(face);

      const auto coef_jump = C_T*(static_cast<Number>(0.5)*
                                  (std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
                                   std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]))); /*--- Jump constant for IP ---*/

      /*--- Loop over all quadrature points ---*/
      for(const unsigned q : phi_m.quadrature_point_indices()) {
        const auto& n_minus = phi_m.get_normal_vector(q);

        /*--- Compute the quantities at the previous stages ---*/
        VectorizedArray<Number> IP_flux_num = make_vectorized_array<Number>(0.0);
        for(unsigned s = 1; s <= IMEX_stage - 1; ++s) {
          /*--- Retrieve the useful fields ---*/
          phi_u_m.gather_evaluate(src[2*(s-1)], EvaluationFlags::gradients);
          phi_u_p.gather_evaluate(src[2*(s-1)], EvaluationFlags::gradients);
          phi_theta_m.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values | EvaluationFlags::gradients);
          phi_theta_p.gather_evaluate(src[2*(s-1) + 1], EvaluationFlags::values | EvaluationFlags::gradients);

          const auto& grad_u_s_m     = phi_u_m.get_gradient(q);
          const auto& grad_u_s_p     = phi_u_p.get_gradient(q);
          const auto& grad_theta_s_m = phi_theta_m.get_gradient(q);
          const auto& grad_theta_s_p = phi_theta_p.get_gradient(q);

          const auto& mod_squared_grad_uz_s_m = grad_u_s_m[0][dim - 1]*grad_u_s_m[0][dim - 1];
          const auto& mod_squared_grad_uz_s_p = grad_u_s_p[0][dim - 1]*grad_u_s_p[0][dim - 1];
          const auto& Ri_s_m                  = inv_Fr2*grad_theta_s_m[dim - 1]/mod_squared_grad_uz_s_m;
          const auto& Ri_s_p                  = inv_Fr2*grad_theta_s_p[dim - 1]/mod_squared_grad_uz_s_p;

          VectorizedArray<Number> b_m, b_p;
          VectorizedArray<Number> beta_m, beta_p;
          for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_s_m[idx] > static_cast<Number>(0.0)) {
              b_m[idx]    = static_cast<Number>(5.0);
              beta_m[idx] = static_cast<Number>(-2.0);
            }
            else {
              b_m[idx]    = static_cast<Number>(20.0);
              beta_m[idx] = static_cast<Number>(0.5);
            }

            if(Ri_s_p[idx] > static_cast<Number>(0.0)) {
              b_p[idx]    = static_cast<Number>(5.0);
              beta_p[idx] = static_cast<Number>(-2.0);
            }
            else {
              b_p[idx]    = static_cast<Number>(20.0);
              beta_p[idx] = static_cast<Number>(0.5);
            }
          }
          const auto& kappa_s_m = l2_mixing*std::sqrt(mod_squared_grad_uz_s_m)*
                                  std::pow(static_cast<Number>(1.0) + b_m*std::abs(Ri_s_m), beta_m);
          const auto& kappa_s_p = l2_mixing*std::sqrt(mod_squared_grad_uz_s_p)*
                                  std::pow(static_cast<Number>(1.0) + b_p*std::abs(Ri_s_p), beta_p);

          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_s_m,
                                                  diff_tensor_s_p;
          diff_tensor_s_m = 0;
          diff_tensor_s_m[dim - 1][dim - 1] = kappa_s_m;
          diff_tensor_s_p = 0;
          diff_tensor_s_p[dim - 1][dim - 1] = kappa_s_p;
          const auto& avg_diff_flux_s = static_cast<Number>(0.5)*
                                        (diff_tensor_s_m*grad_theta_s_m +
                                         diff_tensor_s_p*grad_theta_s_p);

          /*--- Consider also jump penalization contribution ---*/
          const auto& theta_s_m    = phi_theta_m.get_value(q);
          const auto& theta_s_p    = phi_theta_p.get_value(q);
          const auto& avg_kappa_s  = static_cast<Number>(2.0)/
                                     (static_cast<Number>(1.0)/kappa_s_m +
                                      static_cast<Number>(1.0)/kappa_s_p);
          const auto& jump_theta_s = theta_s_m - theta_s_p;

          /*--- Compute the numerical flux ---*/
          IP_flux_num += a_tilde[IMEX_stage - 1][s - 1]*dt*
                         (scalar_product(avg_diff_flux_s, n_minus) -
                          coef_jump*avg_kappa_s*jump_theta_s);
        }

        phi_m.submit_value(IP_flux_num, q);
        phi_p.submit_value(-IP_flux_num, q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Put together all the previous steps for the temperature equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  vmult_rhs_temperature(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&TurbulentOperator::assemble_rhs_cell_term_temperature,
                     &TurbulentOperator::assemble_rhs_face_term_temperature,
                     &TurbulentOperator::assemble_rhs_boundary_term_temperature,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }

  // Assemble cell term for the temperature equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_cell_term_temperature(const MatrixFree<dim, Number>&       data,
                                 Vec&                                 dst,
                                 const Vec&                           src,
                                 const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEEvaluation_theta phi(data, EquationData::THETA_INDEX_DOF),
                       phi_theta_curr(data, EquationData::THETA_INDEX_DOF);
    FEEvaluation_u     phi_u_curr(data, EquationData::U_INDEX_DOF);

    /*--- Loop over all cells. ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr.reinit(cell);
      phi_theta_curr.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      /*--- Loop over all quadrature points. ---*/
      for(const unsigned q : phi.quadrature_point_indices()) {
        const auto& grad_u_curr     = phi_u_curr.get_gradient(q);
        const auto& grad_theta_curr = phi_theta_curr.get_gradient(q);

        const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
        const auto& Ri_curr                  = inv_Fr2*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
        VectorizedArray<Number> b;
        VectorizedArray<Number> beta;
        for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
          if(Ri_curr[idx] > static_cast<Number>(0.0)) {
            b[idx]    = static_cast<Number>(5.0);
            beta[idx] = static_cast<Number>(-2.0);
          }
          else {
            b[idx]    = static_cast<Number>(20.0);
            beta[idx] = static_cast<Number>(0.5);
          }
        }
        const auto& kappa_curr = l2_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                 std::pow(static_cast<Number>(1.0) + b*std::abs(Ri_curr), beta);
        Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr;
        diff_tensor_curr = 0;
        diff_tensor_curr[dim - 1][dim - 1] = kappa_curr;

        phi.submit_value(phi.get_value(q), q);
        phi.submit_gradient(a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                            (diff_tensor_curr*phi.get_gradient(q)), q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the temperature equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_face_term_temperature(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const Vec&                                   src,
                                 const std::pair<unsigned, unsigned>& face_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEFaceEvaluation_theta phi_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_p(data, false, EquationData::THETA_INDEX_DOF),
                           phi_theta_curr_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_theta_curr_p(data, false, EquationData::THETA_INDEX_DOF);
    FEFaceEvaluation_u     phi_u_curr_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_curr_p(data, false, EquationData::U_INDEX_DOF);

    /*--- Loop over all internal faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_u_curr_m.reinit(face);
      phi_u_curr_m.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_u_curr_p.reinit(face);
      phi_u_curr_p.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr_m.reinit(face);
      phi_theta_curr_m.gather_evaluate(theta_curr, EvaluationFlags::gradients);
      phi_theta_curr_p.reinit(face);
      phi_theta_curr_p.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi_m.reinit(face);
      phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      const auto coef_jump = C_T*(static_cast<Number>(0.5)*
                                  (std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
                                   std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]))); /*--- Jump cosntant for IP ---*/

      /*--- Loop over all quadrature points ---*/
      for(const unsigned q : phi_m.quadrature_point_indices()) {
        const auto& n_minus = phi_m.get_normal_vector(q);

        /*--- Compute contribution at current stage ---*/
        const auto& grad_u_curr_m     = phi_u_curr_m.get_gradient(q);
        const auto& grad_u_curr_p     = phi_u_curr_p.get_gradient(q);
        const auto& grad_theta_curr_m = phi_theta_curr_m.get_gradient(q);
        const auto& grad_theta_curr_p = phi_theta_curr_p.get_gradient(q);

        const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
        const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
        const auto& Ri_curr_m                  = inv_Fr2*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
        const auto& Ri_curr_p                  = inv_Fr2*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
        VectorizedArray<Number> b_m, b_p;
        VectorizedArray<Number> beta_m, beta_p;
        for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
          if(Ri_curr_m[idx] > static_cast<Number>(0.0)) {
            b_m[idx]    = static_cast<Number>(5.0);
            beta_m[idx] = static_cast<Number>(-2.0);
          }
          else {
            b_m[idx]    = static_cast<Number>(20.0);
            beta_m[idx] = static_cast<Number>(0.5);
          }

          if(Ri_curr_p[idx] > static_cast<Number>(0.0)) {
            b_p[idx]    = static_cast<Number>(5.0);
            beta_p[idx] = static_cast<Number>(-2.0);
          }
          else {
            b_p[idx]    = static_cast<Number>(20.0);
            beta_p[idx] = static_cast<Number>(0.5);
          }
        }
        const auto& kappa_curr_m = l2_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                   std::pow(static_cast<Number>(1.0) + b_m*std::abs(Ri_curr_m), beta_m);
        const auto& kappa_curr_p = l2_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                   std::pow(static_cast<Number>(1.0) + b_p*std::abs(Ri_curr_p), beta_p);

        Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr_m,
                                                diff_tensor_curr_p;
        diff_tensor_curr_m = 0;
        diff_tensor_curr_m[dim - 1][dim - 1] = kappa_curr_m;
        diff_tensor_curr_p = 0;
        diff_tensor_curr_p[dim - 1][dim - 1] = kappa_curr_p;
        const auto& avg_diff_flux = static_cast<Number>(0.5)*
                                    (diff_tensor_curr_m*phi_m.get_gradient(q) +
                                     diff_tensor_curr_p*phi_p.get_gradient(q));

        /*--- Consider also IP term ---*/
        const auto& theta_m        = phi_m.get_value(q);
        const auto& theta_p        = phi_p.get_value(q);
        const auto& avg_kappa_curr = static_cast<Number>(2.0)/
                                     (static_cast<Number>(1.0)/kappa_curr_m +
                                      static_cast<Number>(1.0)/kappa_curr_p);
        const auto& jump_theta     = theta_m - theta_p;

        /*--- Compute the numerical flux ---*/
        const auto& IP_flux_num = a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                                  (-scalar_product(avg_diff_flux, n_minus) +
                                   coef_jump*avg_kappa_curr*jump_theta);

        phi_m.submit_value(IP_flux_num, q);
        phi_p.submit_value(-IP_flux_num, q);
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
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(NS_stage, 3);
    Assert(NS_stage > 0, ExcInternalError());

    if(NS_stage == EquationData::U_INDEX_SYSTEM_TURB) {
      this->data->loop(&TurbulentOperator::assemble_cell_term_velocity,
                       &TurbulentOperator::assemble_face_term_velocity,
                       &TurbulentOperator::assemble_boundary_term_velocity,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == EquationData::THETA_INDEX_SYSTEM) {
      this->data->loop(&TurbulentOperator::assemble_cell_term_temperature,
                       &TurbulentOperator::assemble_face_term_temperature,
                       &TurbulentOperator::assemble_boundary_term_temperature,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else {
      Assert(false, ExcInternalError());
    }
  }


  //////////////////////////////////////////////////////////////
  /*---- COMPUTE DIAGONALS---*/
  /////////////////////////////////////////////////////////////

  // Assemble diagonal cell term for the velocity equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const unsigned&                      ,
                                       const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEEvaluation_u     phi(data, EquationData::U_INDEX_DOF),
                       phi_u_curr(data, EquationData::U_INDEX_DOF);
    FEEvaluation_theta phi_theta_curr(data, EquationData::THETA_INDEX_DOF);

    /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
          a vector which is 1 for the node of interest and 0 elsewhere. This is what 'tmp_diagonal_velocity' does. ---*/
    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr.reinit(cell);
      phi_theta_curr.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
        }
        phi.submit_dof_value(tmp_diagonal_velocity, i);
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points. ---*/
        for(const unsigned q : phi.quadrature_point_indices()) {
          const auto& grad_u_curr     = phi_u_curr.get_gradient(q);
          const auto& grad_theta_curr = phi_theta_curr.get_gradient(q);

          const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
          const auto& Ri_curr                  = inv_Fr2*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr[idx] > static_cast<Number>(0.0)) {
              b[idx]    = static_cast<Number>(5.0);
              beta[idx] = static_cast<Number>(-2.0);
            }
            else {
              b[idx]    = static_cast<Number>(20.0);
              beta[idx] = static_cast<Number>(0.5);
            }
          }
          const auto& kappa_curr = l2_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                   std::pow(1.0 + b*std::abs(Ri_curr), beta);

          Tensor<2, dim, VectorizedArray<Number>> diff_flux;
          diff_flux = 0;
          const auto& grad_u    = phi.get_gradient(q);
          diff_flux[0][dim - 1] = kappa_curr*grad_u[0][dim - 1];

          phi.submit_value(phi.get_value(q);, q);
          phi.submit_gradient(a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*diff_flux, q);
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

  // Assemble diagonal face term for the velocity equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const unsigned&                      ,
                                       const std::pair<unsigned, unsigned>& face_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEFaceEvaluation_u     phi_m(data, true, EquationData::U_INDEX_DOF),
                           phi_p(data, false, EquationData::U_INDEX_DOF),
                           phi_u_curr_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_curr_p(data, false, EquationData::U_INDEX_DOF);
    FEFaceEvaluation_theta phi_theta_curr_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_theta_curr_p(data, false, EquationData::THETA_INDEX_DOF);

    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_m(phi_m.dofs_per_component),
                                                           diagonal_p(phi_p.dofs_per_component);

    /*--- Loop over all internal faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_u_curr_m.reinit(face);
      phi_u_curr_m.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_u_curr_p.reinit(face);
      phi_u_curr_p.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr_m.reinit(face);
      phi_theta_curr_m.gather_evaluate(theta_curr, EvaluationFlags::gradients);
      phi_theta_curr_p.reinit(face);
      phi_theta_curr_p.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi_m.reinit(face);
      phi_p.reinit(face);

      const auto coef_jump = C_u*(static_cast<Number>(0.5)*
                                  (std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
                                   std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]))); /*--- Jump constant for IP ---*/

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi_m.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi_m.dofs_per_component; ++j) {
          phi_m.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi_p.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
        }
        phi_m.submit_dof_value(tmp_diagonal_velocity, i);
        phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_p.submit_dof_value(tmp_diagonal_velocity, i);
        phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points ---*/
        for(const unsigned q : phi_m.quadrature_point_indices()) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute contribution at current stage ---*/
          const auto& grad_u_curr_m     = phi_u_curr_m.get_gradient(q);
          const auto& grad_u_curr_p     = phi_u_curr_p.get_gradient(q);
          const auto& grad_theta_curr_m = phi_theta_curr_m.get_gradient(q);
          const auto& grad_theta_curr_p = phi_theta_curr_p.get_gradient(q);

          const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
          const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
          const auto& Ri_curr_m                  = inv_Fr2*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
          const auto& Ri_curr_p                  = inv_Fr2*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
          VectorizedArray<Number> b_m, b_p;
          VectorizedArray<Number> beta_m, beta_p;
          for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr_m[idx] > static_cast<Number>(0.0)) {
              b_m[idx]    = static_cast<Number>(5.0);
              beta_m[idx] = static_cast<Number>(-2.0);
            }
            else {
              b_m[idx]    = static_cast<Number>(20.0);
              beta_m[idx] = static_cast<Number>(0.5);
            }

            if(Ri_curr_p[idx] > static_cast<Number>(0.0)) {
              b_p[idx]    = static_cast<Number>(5.0);
              beta_p[idx] = static_cast<Number>(-2.0);
            }
            else {
              b_p[idx]    = static_cast<Number>(20.0);
              beta_p[idx] = static_cast<Number>(0.5);
            }
          }
          const auto& kappa_curr_m = l2_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                     std::pow(static_cast<Number>(1.0) + b_m*std::abs(Ri_curr_m), beta_m);
          const auto& kappa_curr_p = l2_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                     std::pow(static_cast<Number>(1.0) + b_p*std::abs(Ri_curr_p), beta_p);

          Tensor<2, dim, VectorizedArray<Number>> avg_diff_flux;
          avg_diff_flux = 0;
          const auto& grad_u_m      = phi_m.get_gradient(q);
          const auto& grad_u_p      = phi_p.get_gradient(q);
          avg_diff_flux[0][dim - 1] = static_cast<Number>(0.5)*
                                      (kappa_curr_m*grad_u_m[0][dim - 1] +
                                       kappa_curr_p*grad_u_p[0][dim - 1]);

          /*--- Consider also IP term ---*/
          const auto& u_m            = phi_m.get_value(q);
          const auto& u_p            = phi_p.get_value(q);
          const auto& avg_kappa_curr = static_cast<Number>(2.0)/
                                       (static_cast<Number>(1.0)/kappa_curr_m +
                                        static_cast<Number>(1.0)/kappa_curr_p);
          auto jump_u                = u_m - u_p;
          jump_u[dim - 1]            = make_vectorized_array<Number>(0.0);

          /*--- Compute the numerical flux ---*/
          const auto& IP_flux_num = a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                                    (-avg_diff_flux*n_minus +
                                      coef_jump*avg_kappa_curr*jump_u);

          phi_m.submit_value(IP_flux_num, q);
          phi_p.submit_value(-IP_flux_num, q);
        }

        phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_m[i] = phi_m.get_dof_value(i);
        phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_p[i] = phi_p.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi_m.dofs_per_component; ++i) {
        phi_m.submit_dof_value(diagonal_m[i], i);
        phi_p.submit_dof_value(diagonal_p[i], i);
      }
      phi_m.distribute_local_to_global(dst);
      phi_p.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the temperature equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  assemble_diagonal_cell_term_temperature(const MatrixFree<dim, Number>&       data,
                                          Vec&                                 dst,
                                          const unsigned&                      ,
                                          const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We start by decalring suitable instances to read the available quantities ---*/
    FEEvaluation_theta phi(data, EquationData::THETA_INDEX_DOF),
                       phi_theta_curr(data, EquationData::THETA_INDEX_DOF);
    FEEvaluation_u     phi_u_curr(data, EquationData::U_INDEX_DOF);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr.reinit(cell);
      phi_theta_curr.gather_evaluate(theta_curr, EvaluationFlags::gradients);

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
        for(const unsigned q : phi.quadrature_point_indices()) {
          const auto& grad_u_curr     = phi_u_curr.get_gradient(q);
          const auto& grad_theta_curr = phi_theta_curr.get_gradient(q);

          const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
          const auto& Ri_curr                  = inv_Fr2*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr[idx] > 0.0) {
              b[idx]    = static_cast<Number>(5.0);
              beta[idx] = static_cast<Number>(-2.0);
            }
            else {
              b[idx]    = static_cast<Number>(20.0);
              beta[idx] = static_cast<Number>(0.5);
            }
          }
          const auto& kappa_curr = l2_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                   std::pow(static_cast<Number>(1.0) + b*std::abs(Ri_curr), beta);

          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr;
          diff_tensor_curr = 0;
          diff_tensor_curr[dim - 1][dim - 1] = kappa_curr;

          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                              (diff_tensor_curr*phi.get_gradient(q)), q);
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


  // Assemble diagonal face term for the temperature equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_face_term_temperature(const MatrixFree<dim, Number>&       data,
                                          Vec&                                 dst,
                                          const unsigned&                      ,
                                          const std::pair<unsigned, unsigned>& face_range) const {
    /*--- We start by decalring suitable instances to read the available quantities ---*/
    FEFaceEvaluation_theta phi_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_p(data, false, EquationData::THETA_INDEX_DOF),
                           phi_theta_curr_m(data, true, EquationData::THETA_INDEX_DOF),
                           phi_theta_curr_p(data, false, EquationData::THETA_INDEX_DOF);
    FEFaceEvaluation_u     phi_u_curr_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_curr_p(data, false, EquationData::U_INDEX_DOF);

    AlignedVector<VectorizedArray<Number>> diagonal_m(phi_m.dofs_per_component),
                                           diagonal_p(phi_p.dofs_per_component);

    /*--- Loop over all face ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_u_curr_m.reinit(face);
      phi_u_curr_m.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_u_curr_p.reinit(face);
      phi_u_curr_p.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr_m.reinit(face);
      phi_theta_curr_m.gather_evaluate(theta_curr, EvaluationFlags::gradients);
      phi_theta_curr_p.reinit(face);
      phi_theta_curr_p.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi_m.reinit(face);
      phi_p.reinit(face);

      const auto coef_jump = C_T*(static_cast<Number>(0.5)*
                                  (std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]) +
                                   std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]))); /*--- Jump constant for IP ---*/

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi_m.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi_m.dofs_per_component; ++j) {
          phi_m.submit_dof_value(VectorizedArray<Number>(), j);
          phi_p.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi_m.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_p.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points ---*/
        for(const unsigned q : phi_m.quadrature_point_indices()) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute contribution at current stage ---*/
          const auto& grad_u_curr_m              = phi_u_curr_m.get_gradient(q);
          const auto& grad_u_curr_p              = phi_u_curr_p.get_gradient(q);
          const auto& grad_theta_curr_m          = phi_theta_curr_m.get_gradient(q);
          const auto& grad_theta_curr_p          = phi_theta_curr_p.get_gradient(q);

          const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
          const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
          const auto& Ri_curr_m                  = inv_Fr2*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
          const auto& Ri_curr_p                  = inv_Fr2*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
          VectorizedArray<Number> b_m, b_p;
          VectorizedArray<Number> beta_m, beta_p;
          for(unsigned idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr_m[idx] > static_cast<Number>(0.0)) {
              b_m[idx]    = static_cast<Number>(5.0);
              beta_m[idx] = static_cast<Number>(-2.0);
            }
            else {
              b_m[idx]    = static_cast<Number>(20.0);
              beta_m[idx] = static_cast<Number>(0.5);
            }

            if(Ri_curr_p[idx] > static_cast<Number>(0.0)) {
              b_p[idx]    = static_cast<Number>(5.0);
              beta_p[idx] = static_cast<Number>(-2.0);
            }
            else {
              b_p[idx]    = static_cast<Number>(20.0);
              beta_p[idx] = static_cast<Number>(0.5);
            }
          }
          const auto& kappa_curr_m = l2_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                     std::pow(static_cast<Number>(1.0) + b_m*std::abs(Ri_curr_m), beta_m);
          const auto& kappa_curr_p = l2_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                     std::pow(static_cast<Number>(1.0) + b_p*std::abs(Ri_curr_p), beta_p);

          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr_m,
                                                  diff_tensor_curr_p;
          diff_tensor_curr_m = 0;
          diff_tensor_curr_m[dim - 1][dim - 1] = kappa_curr_m;
          diff_tensor_curr_p = 0;
          diff_tensor_curr_p[dim - 1][dim - 1] = kappa_curr_p;
          const auto& avg_diff_flux = static_cast<Number>(0.5)*
                                      (diff_tensor_curr_m*phi_m.get_gradient(q) +
                                       diff_tensor_curr_p*phi_p.get_gradient(q));

          /*--- Consider also IP term ---*/
          const auto& theta_m        = phi_m.get_value(q);
          const auto& theta_p        = phi_p.get_value(q);
          const auto& avg_kappa_curr = static_cast<Number>(2.0)/
                                       (static_cast<Number>(1.0)/kappa_curr_m +
                                        static_cast<Number>(1.0)/kappa_curr_p);
          const auto& jump_theta     = theta_m - theta_p;

          /*--- Compute the numerical flux ---*/
          const auto& IP_flux_num = a_tilde[IMEX_stage - 1][IMEX_stage - 1]*dt*
                                    (-scalar_product(avg_diff_flux, n_minus) +
                                     coef_jump*avg_kappa_curr*jump_theta);

          phi_m.submit_value(IP_flux_num, q);
          phi_p.submit_value(-IP_flux_num, q);
        }

        phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_m[i] = phi_m.get_dof_value(i);
        phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_p[i] = phi_p.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi_m.dofs_per_component; ++i) {
        phi_m.submit_dof_value(diagonal_m[i], i);
        phi_p.submit_dof_value(diagonal_p[i], i);
      }
      phi_m.distribute_local_to_global(dst);
      phi_p.distribute_local_to_global(dst);
    }
  }


  // Compute diagonal of various steps
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_T,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void TurbulentOperator<dim,
                         fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  compute_diagonal() {
    AssertIndexRange(NS_stage, 3);
    Assert(NS_stage > 0, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned dummy = 0;

    if(NS_stage == EquationData::U_INDEX_SYSTEM_TURB) {
      this->data->initialize_dof_vector(inverse_diagonal, EquationData::U_INDEX_DOF);

      this->data->loop(&TurbulentOperator::assemble_diagonal_cell_term_velocity,
                       &TurbulentOperator::assemble_diagonal_face_term_velocity,
                       &TurbulentOperator::assemble_diagonal_boundary_term_velocity,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == EquationData::THETA_INDEX_SYSTEM) {
      this->data->initialize_dof_vector(inverse_diagonal, EquationData::THETA_INDEX_DOF);

      this->data->loop(&TurbulentOperator::assemble_diagonal_cell_term_temperature,
                       &TurbulentOperator::assemble_diagonal_face_term_temperature,
                       &TurbulentOperator::assemble_diagonal_boundary_term_temperature,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
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

} // End of namespace
