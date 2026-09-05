/*--- Author: Giuseppe Orlando, 2025. ---*/
#pragma once

// Implement my own ceil function as constexpr
// (available starting in std from C++23, too strong requirement)
//
template<typename T = double>
constexpr int my_ceil(const T num) {
  return (static_cast<T>(static_cast<int>(num)) == num) ?
          static_cast<int>(num) :
          static_cast<int>(num) + ((num > static_cast<T>(0.0)) ? 1 : 0);
}

// @sect{Equation data}

// In this namespace, we declare some constant global parameters related to both
// the physics and the numerical discretization
//
namespace EquationData {
  static const unsigned n_vars = 3; /*--- Number of variables for which we solve a linear system ---*/

  /*--- Define auxiliary indices related to the dof handlers order and to linear systems under consideration ---*/
  static const unsigned RHO_INDEX_SYSTEM = 1;
  static const unsigned P_INDEX_SYSTEM   = 2;
  static const unsigned U_INDEX_SYSTEM   = 3;

  static const unsigned U_INDEX_SYSTEM_TURB = 1;
  static const unsigned THETA_INDEX_SYSTEM  = 2;

  static const unsigned U_INDEX_DOF   = 0;
  static const unsigned P_INDEX_DOF   = 1;
  static const unsigned RHO_INDEX_DOF = 2;

  static const unsigned THETA_INDEX_DOF = P_INDEX_DOF;

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned degree_p   = 4;
  static const unsigned degree_rho = 4;
  static const unsigned degree_u   = 4;

  static const unsigned quadrature_degree = 2*degree_u + 1; /*--- Accuracy for quadrature formula ---*/

  /*--- Physical parameters ---*/
  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

  static const double g = 9.80616; /*--- Acceleration of gravity ---*/
} // namespace EquationData
