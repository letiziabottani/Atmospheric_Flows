/*--- Author: Giuseppe Orlando, 2026. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones
//
#include <deal.II/base/function.h>

#include "../equation_data.h"

#include <cmath>

// @sect{Initial conditions}

// In this namespace, we declare the initial background conditions.
// Some parameters could be read at run-time, but this would be very
// configuration dependent and the parameter file would become unreadable
//
namespace ICBC {
  using namespace dealii;

  /**
   * We declare now the class that describes the initial condition for the velocity.
   */
  template<unsigned dim, typename T = double>
  class Velocity: public Function<dim, T> {
  public:
    Velocity(const T u_bar_, const T u_ref_,
             const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evaluation for each component ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Vector evaluation of the velocity ---*/

  private:
    T u_bar; /*--- Background velocity ---*/
    T u_ref; /*--- Reference velocity (for non-dimensional variables) ---*/
  };

  // Constructor which relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Velocity<dim, T>::Velocity(const T u_bar_, const T u_ref_,
                             const T initial_time):
    Function<dim, T>(dim, initial_time),
    u_bar(u_bar_), u_ref(u_ref_) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<unsigned dim, typename T>
  T Velocity<dim, T>::value(const Point<dim, T>& p,
                            const unsigned       component) const {
    AssertIndexRange(component, dim);

    if(component == 0) {
      return u_bar/u_ref;
    }
    else {
      return static_cast<T>(0.0);
    }
  }

  // Put together for a vector evalutation of the velocity.
  //
  template<unsigned dim, typename T>
  void Velocity<dim, T>::vector_value(const Point<dim, T>& p,
                                      Vector<T>&           values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We do the same for the pressure.
   */
  template<unsigned dim, typename T = double>
  class Pressure: public Function<dim, T> {
  public:
    Pressure(const T p_bar_, const T T_bar_,
             const T p_ref_, const T L_ref_,
             const T N_,
             const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evalution of the pressure ---*/

  private:
    T p_bar; /*--- Background pressure ---*/
    T T_bar; /*--- Background temeprature ---*/

    T p_ref; /*--- Reference pressure (for non-dimensional variables) ---*/
    T L_ref; /*--- Reference length (for non-dimensional variables) ---*/

    T N; /*--- Buoyancy frequency ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Pressure<dim, T>::Pressure(const T p_bar_, const T T_bar_,
                             const T p_ref_, const T L_ref_,
                             const T N_,
                             const T initial_time):
    Function<dim, T>(1, initial_time),
    p_bar(p_bar_), T_bar(T_bar_), p_ref(p_ref_), L_ref(L_ref_), N(N_) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<unsigned dim, typename T>
  T Pressure<dim, T>::value(const Point<dim, T>& p,
                            const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const auto Gamma  = (static_cast<T>(EquationData::Cp_Cv) - static_cast<T>(1.0))/
                        static_cast<T>(EquationData::Cp_Cv);

    const auto pi_bar = static_cast<T>(1.0)
                      - static_cast<T>(EquationData::g)*static_cast<T>(EquationData::g)/(N*N)*
                        Gamma/(static_cast<T>(EquationData::R)*T_bar)*
                        (static_cast<T>(1.0) - std::exp(-N*N/static_cast<T>(EquationData::g)*p[2]*L_ref));

    return std::pow(pi_bar, static_cast<T>(1.0)/Gamma);//(p_bar/p_ref)*
  }


  /**
   * We do the same for the density.
   */
  template<unsigned dim, typename T = double>
  class Density: public Function<dim, T> {
  public:
    Density(const T p_bar_, const T T_bar_,
            const T rho_ref_, const T L_ref_,
            const T N_,
            const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evaluation of the density ---*/
  private:
    T p_bar; /*--- Background pressure ---*/
    T T_bar; /*--- Background temeprature ---*/

    T rho_ref; /*--- Reference density (for non-dimensional variables) ---*/
    T L_ref;   /*--- Reference length (for non-dimensional variables) ---*/

    T N; /*--- Buoyancy frequency ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Density<dim, T>::Density(const T p_bar_, const T T_bar_,
                           const T rho_ref_, const T L_ref_,
                           const T N_,
                           const T initial_time):
    Function<dim, T>(1, initial_time),
    p_bar(p_bar_), T_bar(T_bar_), rho_ref(rho_ref_), L_ref(L_ref_), N(N_) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<unsigned dim, typename T>
  T Density<dim, T>::value(const Point<dim, T>& p,
                           const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const auto Gamma  = (static_cast<T>(EquationData::Cp_Cv) - static_cast<T>(1.0))/
                        static_cast<T>(EquationData::Cp_Cv);

    const auto pi_bar = static_cast<T>(1.0)
                      - static_cast<T>(EquationData::g)*static_cast<T>(EquationData::g)/(N*N)*
                        Gamma/(static_cast<T>(EquationData::R)*T_bar)*
                        (static_cast<T>(1.0) - std::exp(-N*N/static_cast<T>(EquationData::g)*p[2]*L_ref));

    const auto theta_bar = T_bar*std::exp(N*N/static_cast<T>(EquationData::g)*p[2]*L_ref);

    const auto rho_bar = p_bar/(static_cast<T>(EquationData::R)*T_bar);

    return std::pow(pi_bar, static_cast<T>(1.0)/
                                            (static_cast<T>(EquationData::Cp_Cv) - static_cast<T>(1.0))); //(rho_bar/rho_ref)*T_bar/theta_bar*
  }

} // namespace EquationData
