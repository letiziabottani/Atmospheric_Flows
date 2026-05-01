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
    Function<dim, T>(3, initial_time),
    u_bar(u_bar_), u_ref(u_ref_) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<unsigned dim, typename T>
  T Velocity<dim, T>::value(const Point<dim, T>& p,
                            const unsigned       component) const {
    AssertIndexRange(component, 3);

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
    Assert(values.size() == 3, ExcDimensionMismatch(values.size(), 3));

    for(unsigned i = 0; i < 3; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We do the same for the pressure.
   */
  template<unsigned dim, typename T = double>
  class Pressure: public Function<dim, T> {
  public:
    Pressure(const T p_ref_, const T L_ref_,
             const T T_ref_,
             const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evalution of the pressure ---*/

  private:
    
    T p_ref; /*--- Reference pressure (for non-dimensional variables) ---*/
    T L_ref; /*--- Reference length (for non-dimensional variables) ---*/
    T T_ref; /*--- Reference temperature (for non-dimensional variables) ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Pressure<dim, T>::Pressure(const T p_ref_, const T L_ref_,
                             const T T_ref_,
                             const T initial_time):
    Function<dim, T>(1, initial_time),
    p_ref(p_ref_), L_ref(L_ref_), T_ref(T_ref_) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<unsigned dim, typename T>
  T Pressure<dim, T>::value(const Point<dim, T>& p,
                            const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, 1);

    static const double delta= EquationData::g/(EquationData::R*T_ref); /*--- scale-height parameter ---*/
    const auto p0 = p_ref *std::exp(-delta * p[1]* L_ref);
    return p0;
  }


  /**
   * We do the same for the density.
   */
  template<unsigned dim, typename T = double>
  class Density: public Function<dim, T> {
  public:
    Density(const T rho_ref_, const T L_ref_,
            const T T_ref_,
            const T H_, const T xc_, const T ac_,
            const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evaluation of the density ---*/
  private:
    T rho_ref; /*--- Reference density (for non-dimensional variables) ---*/
    T L_ref;   /*--- Reference length (for non-dimensional variables) ---*/
    T T_ref;   /*--- Reference temperature (for non-dimensional variables) ---*/

    T H; /*--- Height of the perturbation ---*/
    T xc; /*--- Center of the perturbation ---*/
    T ac; /*--- amplitude of the perturbation ---*/
    
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Density<dim, T>::Density(const T rho_ref_, const T L_ref_,
                           const T T_ref_,
                           const T H_, const T xc_, const T ac_,
                           const T initial_time):
    Function<dim, T>(1, initial_time),
    rho_ref(rho_ref_), L_ref(L_ref_), T_ref(T_ref_),H(H_),xc(xc_),ac(ac_) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<unsigned dim, typename T>
  T Density<dim, T>::value(const Point<dim, T>& p,
                           const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, 1);

    static const double delta= EquationData::g/(EquationData::R*T_ref); /*--- scale-height parameter ---*/
    
    const auto rho0 = rho_ref * std::exp(-delta * p[1]* L_ref);

    const auto r  = (p[0]* L_ref - xc) /ac;

    const auto Tb = static_cast<T>(0.01) * std::exp(-r * r) *std::sin(numbers::PI * p[1] * L_ref / H); 
      
    const auto rho_b0 = - rho_ref *Tb / T_ref;
    
    const auto rho_prime =std::exp(-static_cast<T>(0.5) * delta * p[1] * L_ref) *rho_b0;

    return rho0+rho_prime;
  }

  /**
   * We do the same for the bacground density.
   */
  template<unsigned dim, typename T = double>
  class Density_Bar: public Function<dim, T> {
  public:
    Density_Bar(const T rho_ref_, const T L_ref_,
            const T T_ref_,
            const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evaluation of the density ---*/
  private:
    T rho_ref; /*--- Reference density (for non-dimensional variables) ---*/
    T L_ref;   /*--- Reference length (for non-dimensional variables) ---*/
    T T_ref;   /*--- Reference temperature (for non-dimensional variables) ---*/
    
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Density_Bar<dim, T>::Density_Bar(const T rho_ref_, const T L_ref_,
                           const T T_ref_,
                           const T initial_time):
    Function<dim, T>(1, initial_time),
    rho_ref(rho_ref_), L_ref(L_ref_), T_ref(T_ref_) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<unsigned dim, typename T>
  T Density_Bar<dim, T>::value(const Point<dim, T>& p,
                           const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, 1);

    static const double delta= EquationData::g/(EquationData::R*T_ref); /*--- scale-height parameter ---*/
    
    const auto rho0 = rho_ref * std::exp(-delta * p[1]* L_ref);

    return rho0;
  }

} // namespace EquationData
