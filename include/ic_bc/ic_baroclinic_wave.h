/*--- Baroclinic wave ICs: f-plane, dimensional variables. ---*/

#pragma once

#include <deal.II/base/function.h>

#include "../equation_data.h"

#include <cmath>
#include <algorithm>

namespace ICBC
{
  using namespace dealii;

  /*
   * Auxiliary functions
   */
/*
   // Coriolis parameter
  template<typename T>
  T f0(const T Omega,
       const T phi0)
  {
    return static_cast<T>(2.0) * Omega * std::sin(phi0);
  }

  // Horizontal geopotential perturbation
  template<typename T>
  T phi_prime(const T y,
              const T u_bar,
              const T Ly,
              const T Omega,
              const T phi0)
  {
    const T pi = static_cast<T>(numbers::PI);

    return u_bar/2 * f0(Omega, phi0) *
           (y - Ly/static_cast<T>(2.0)
            - Ly/(static_cast<T>(2.0)*pi)
              * std::sin(static_cast<T>(2.0)*pi*y/Ly));
  }

  // Temperature initial condition (geostrophic wind + thermal wind balance) (equation 9 in the paper)
  template<typename T>
  T temperature(const T y,
                const T eta,
                const T T_bar,
                const T u_bar,
                const T Ly,
                const T lapse_rate,
                const T b,
                const T Omega,
                const T phi0)
  {
    const T R = static_cast<T>(EquationData::R);
    const T g = static_cast<T>(EquationData::g);

    const T log_eta = std::log(eta);

    const T T_mean =
      T_bar * std::pow(eta, R*lapse_rate/g);

    const T T_pert =
      phi_prime(y, u_bar, Ly, Omega, phi0)/R *
      (static_cast<T>(2.0)/(b*b)*log_eta*log_eta
       - static_cast<T>(1.0)) *
      std::exp(-(log_eta/b)*(log_eta/b));

    return T_mean + T_pert;
  }

 // Geopotential mean (equation 7  in the paper ) 
  template<typename T>
  T geopotential_mean(const T eta,
                      const T T_bar,
                      const T lapse_rate)
  {
    const T R = static_cast<T>(EquationData::R);
    const T g = static_cast<T>(EquationData::g);

    return T_bar*g/lapse_rate *
           (static_cast<T>(1.0)
            - std::pow(eta, R*lapse_rate/g));
  }

  // Geopotential (equation 6 in the paper)
  template<typename T>
  T geopotential(const T y,
                 const T eta,
                 const T T_bar,
                 const T u_bar,
                 const T Ly,
                 const T lapse_rate,
                 const T b,
                 const T Omega,
                 const T phi0)
  {
    const T log_eta = std::log(eta);

    return geopotential_mean(eta, T_bar, lapse_rate)
         + phi_prime(y, u_bar, Ly, Omega, phi0)
           * log_eta
           * std::exp(-(log_eta/b)*(log_eta/b));
  }


  template<typename T>
  T eta_from_z(const T y,
               const T z,
               const T T_bar,
               const T u_bar,
               const T Ly,
               const T lapse_rate,
               const T b,
               const T Omega,
               const T phi0)
  {
    const T R = static_cast<T>(EquationData::R);
    const T g = static_cast<T>(EquationData::g);

    T eta = std::exp(-g*z/(R*static_cast<T>(260.0))); // initial guess for Newton method, this corresponds to a hydrostatic isothermal initial pressure profile with T = 260 K (taken in the paper as the log-pressure height)

    for(unsigned int iter = 0; iter < 30; ++iter)
    {
      eta = std::min(std::max(eta, T(1.0e-10)), T(1.0));

      const T F = geopotential(y, eta, T_bar, u_bar, Ly, lapse_rate, b, Omega, phi0) - g*z;

      const T dF = -R * temperature(y, eta, T_bar, u_bar, Ly, lapse_rate, b, Omega, phi0)/ eta;

      const T delta_eta = -F/dF;

      eta += delta_eta;

      if(std::abs(delta_eta) < T(1.0e-12))
        break;
    }

    return std::min(std::max(eta, T(1.0e-10)), T(1.0));
  }

// zonal velocity (equation 1 in the paper) 
  template<typename T>
  T zonal_velocity(const T y,
                   const T eta,
                   const T u_bar,
                   const T L_ref,
                   const T Ly,
                   const T b)
  {
    const T pi = static_cast<T>(numbers::PI);

    const T log_eta = std::log(eta);

    return -u_bar;
           //* std::pow(std::sin(pi*y/Ly), static_cast<T>(2.0))* log_eta* std::exp(-(log_eta/b)*(log_eta/b));
  }


  template<typename T>
  T zonal_velocity_perturbation(const T x,
                                const T y,
                                const T up,
                                const T Lp,
                                const T xc,
                                const T yc)
  {
    const T dx = x - xc;
    const T dy = y - yc;

    return up * std::exp(-(dx*dx + dy*dy)/(Lp*Lp));
  }

  /*
   * Velocity initial condition
   */

  template<unsigned dim, typename T = double>
  class Velocity : public Function<dim,T>
  {
  public:
    Velocity(const T p_bar_,
             const T T_bar_,
             const T u_bar_,
             const T Ly_,
             const T L_ref_,
             const T lapse_rate_,
             const T b_,
             const T Omega_,
             const T phi0_,
             const T up_,
             const T Lp_,
             const T xc_,
             const T yc_,
             const T initial_time = static_cast<T>(0.0));

    virtual T value(const Point<dim,T>& p,
                    const unsigned component = 0) const override;

    virtual void vector_value(const Point<dim,T>& p,
                              Vector<T>& values) const override;

  private:
    T p_bar;
    T T_bar;
    T u_bar;
    T Ly;
    T L_ref;
    T lapse_rate;
    T b;
    T Omega;
    T phi0;
    T up;
    T Lp;
    T xc;
    T yc;
  };


  template<unsigned dim, typename T>
  Velocity<dim,T>::Velocity(const T p_bar_,
                            const T T_bar_,
                            const T u_bar_,
                            const T Ly_,
                            const T L_ref_,
                            const T lapse_rate_,
                            const T b_,
                            const T Omega_,
                            const T phi0_,
                            const T up_,
                            const T Lp_,
                            const T xc_,
                            const T yc_,
                            const T initial_time)
    : Function<dim,T>(3, initial_time),
      p_bar(p_bar_),
      T_bar(T_bar_),
      u_bar(u_bar_),
      Ly(Ly_),
      L_ref(L_ref_),
      lapse_rate(lapse_rate_),
      b(b_),
      Omega(Omega_),
      phi0(phi0_),
      up(up_),
      Lp(Lp_),
      xc(xc_),
      yc(yc_)
  {}


  template<unsigned dim, typename T>
  T Velocity<dim,T>::value(const Point<dim,T>& p,
                           const unsigned component) const
  {
    AssertIndexRange(component, dim);
    /*    const T x = p[0]*L_ref;
    const T y = p[1]*L_ref;
    const T z = p[2]*L_ref;

    const T eta =
      eta_from_z(y, z, T_bar, u_bar, Ly,
                 lapse_rate, b, Omega, phi0);
*/
    if(component == 0)
      {
        //const T u_base= zonal_velocity(y, eta, u_bar,L_ref,  Ly, b);
        //const T u_pert = zonal_velocity_perturbation(x, y, up, Lp, xc, yc);
        return u_bar;//u_base;// + u_pert;
      }
    else
    return static_cast<T>(0.0);
  }


  template<unsigned dim, typename T>
  void Velocity<dim,T>::vector_value(const Point<dim,T>& p,
                                     Vector<T>& values) const
  {
    Assert(values.size() == 3, ExcDimensionMismatch(values.size(), 3));

    for(unsigned int d = 0; d < 3; ++d)
      values[d] = value(p,d);
  }

  /*
   * Background Velocity initial condition
   */

  template<unsigned dim, typename T = double>
  class BackgroundVelocity : public Function<dim,T>
  {
  public:
    BackgroundVelocity(const T p_bar_,
                       const T T_bar_,
                       const T u_bar_,
                      const T Ly_,
                      const T L_ref_,
                      const T lapse_rate_,
                      const T b_,
                      const T Omega_,
                      const T phi0_,
                      const T up_,
                      const T Lp_,
                      const T xc_,
                      const T yc_,
                      const T initial_time = static_cast<T>(0.0));

    virtual T value(const Point<dim,T>& p,
                    const unsigned component = 0) const override;

    virtual void vector_value(const Point<dim,T>& p,
                              Vector<T>& values) const override;

  private:
    T p_bar;
    T T_bar;
    T u_bar;
    T Ly;
    T L_ref;
    T lapse_rate;
    T b;
    T Omega;
    T phi0;
    T up;
    T Lp;
    T xc;
    T yc;
  };


  template<unsigned dim, typename T>
  BackgroundVelocity<dim,T>::BackgroundVelocity(const T p_bar_,
                            const T T_bar_,
                            const T u_bar_,
                            const T Ly_,
                            const T L_ref_,
                            const T lapse_rate_,
                            const T b_,
                            const T Omega_,
                            const T phi0_,
                            const T up_,
                            const T Lp_,
                            const T xc_,
                            const T yc_,
                            const T initial_time)
    : Function<dim,T>(3, initial_time),
      p_bar(p_bar_),
      T_bar(T_bar_),
      u_bar(u_bar_),
      Ly(Ly_),
      L_ref(L_ref_),
      lapse_rate(lapse_rate_),
      b(b_),
      Omega(Omega_),
      phi0(phi0_),
      up(up_),
      Lp(Lp_),
      xc(xc_),
      yc(yc_)
  {}


  template<unsigned dim, typename T>
  T BackgroundVelocity<dim,T>::value(const Point<dim,T>& p,
                           const unsigned component) const
  {
    /*
    AssertIndexRange(component, dim);
    /*
    const T x = p[0]*L_ref;
    const T y = p[1]*L_ref;
    const T z = p[2]*L_ref;

    const T eta =
      eta_from_z(y, z, T_bar, u_bar, Ly,
                 lapse_rate, b, Omega, phi0);
*/
    if(component == 0)
      {
        //const T u_base= zonal_velocity(y, eta, u_bar,L_ref,  Ly, b);
        //const T u_pert = zonal_velocity_perturbation(x, y, up, Lp, xc, yc);
        return u_bar;//u_base;
    }
    else
    return static_cast<T>(0.0);
  }


  template<unsigned dim, typename T>
  void BackgroundVelocity<dim,T>::vector_value(const Point<dim,T>& p,
                                     Vector<T>& values) const
  {
    Assert(values.size() == 3, ExcDimensionMismatch(values.size(), 3));

    for(unsigned int d = 0; d < 3; ++d)
      values[d] = value(p,d);
  }


  /*
   * Pressure initial condition
   */

  template<unsigned dim, typename T = double>
  class Pressure : public Function<dim,T>
  {
  public:
    Pressure(const T p_bar_,
             const T T_bar_,
             const T u_bar_,
             const T L_ref_,
             const T Ly_,
             const T lapse_rate_,
             const T b_,
             const T Omega_,
             const T phi0_,
             const T initial_time = static_cast<T>(0.0));

    virtual T value(const Point<dim,T>& p,
                    const unsigned component = 0) const override;

  private:
    T p_bar;
    T T_bar;
    T u_bar;
    T L_ref;
    T Ly;
    T lapse_rate;
    T b;
    T Omega;
    T phi0;
  };


  template<unsigned dim, typename T>
  Pressure<dim,T>::Pressure(const T p_bar_,
                            const T T_bar_,
                            const T u_bar_,
                            const T L_ref_,
                            const T Ly_,
                            const T lapse_rate_,
                            const T b_,
                            const T Omega_,
                            const T phi0_,
                            const T initial_time)
    : Function<dim,T>(1, initial_time),
      p_bar(p_bar_),
      T_bar(T_bar_),
      u_bar(u_bar_),
      L_ref(L_ref_),
      Ly(Ly_),
      lapse_rate(lapse_rate_),
      b(b_),
      Omega(Omega_),
      phi0(phi0_)
  {}


  template<unsigned dim, typename T>
  T Pressure<dim,T>::value(const Point<dim,T>& p,
                           const unsigned component) const
  {
    /*
    (void)component;
    AssertIndexRange(component, 1);

    const T y = p[1]*L_ref;
    const T z = p[2]*L_ref;

    const T eta =
      eta_from_z(y, z, T_bar, u_bar, Ly,
                 lapse_rate, b, Omega, phi0);

    return p_bar* eta;*/

    const T z = p[1]*L_ref;
    const T R = static_cast<T>(EquationData::R);
    const T g = static_cast<T>(EquationData::g);

    return  p_bar*std::exp(-g*z/(R*T_bar));
  }


  /*
   * Density initial condition
   */

  template<unsigned dim, typename T = double>
  class Density : public Function<dim,T>
  {
  public:
    Density(const T p_bar_,
            const T T_bar_,
            const T u_bar_,
            const T Ly_,
            const T L_ref_,
            const T lapse_rate_,
            const T b_,
            const T Omega_,
            const T phi0_,
            const T initial_time = static_cast<T>(0.0));

    virtual T value(const Point<dim,T>& p,
                    const unsigned component = 0) const override;

  private:
    T p_bar;
    T T_bar;
    T u_bar;
    T Ly;
    T L_ref;
    T lapse_rate;
    T b;
    T Omega;
    T phi0;
  };


  template<unsigned dim, typename T>
  Density<dim,T>::Density(const T p_bar_,
                          const T T_bar_,
                          const T u_bar_,
                          const T Ly_,
                          const T L_ref_,
                          const T lapse_rate_,
                          const T b_,
                          const T Omega_,
                          const T phi0_,
                          const T initial_time)
    : Function<dim,T>(1, initial_time),
      p_bar(p_bar_),
      T_bar(T_bar_),
      u_bar(u_bar_),
      Ly(Ly_),
      L_ref(L_ref_),
      lapse_rate(lapse_rate_),
      b(b_),
      Omega(Omega_),
      phi0(phi0_)
  {}


  template<unsigned dim, typename T>
  T Density<dim,T>::value(const Point<dim,T>& p,
                          const unsigned component) const
  {
    /*
    (void)component;
    AssertIndexRange(component, 1);

    const T R = static_cast<T>(EquationData::R);

    const T y = p[1]*L_ref;
    const T z = p[2]*L_ref;

    const T eta =
      eta_from_z(y, z, T_bar, u_bar, Ly,
                 lapse_rate, b, Omega, phi0);

    const T p_val = p_bar * eta;

    const T T_val =
      temperature(y, eta, T_bar, u_bar, Ly,
                  lapse_rate, b, Omega, phi0);

    //return p_val/(R*T_val);
    return p_bar/(R*T_val);
*/
  const T z = p[1]*L_ref;
  const T R = static_cast<T>(EquationData::R);
  const T g = static_cast<T>(EquationData::g);

  return p_bar*std::exp(-g*z/(R*T_bar))/(R*T_bar) ;
  }

} // namespace ICBC