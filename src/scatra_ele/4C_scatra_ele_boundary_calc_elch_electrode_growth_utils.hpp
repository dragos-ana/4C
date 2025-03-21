// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_BOUNDARY_CALC_ELCH_ELECTRODE_GROWTH_UTILS_HPP
#define FOUR_C_SCATRA_ELE_BOUNDARY_CALC_ELCH_ELECTRODE_GROWTH_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode_utils.hpp"

FOUR_C_NAMESPACE_OPEN


// forward declaration

namespace Discret::Elements
{
  class ScaTraEleParameterStd;
  class ScaTraEleParameterBoundary;
}  // namespace Discret::Elements


namespace Discret::Elements
{
  /*!
   * @brief Calculate the exchange mass flux density of the growth kinetics
   *
   * @param[in] kr                  charge transfer constant
   * @param[in] alpha_a             symmetry coefficient of anodic intercalation reaction
   * @param[in] c_el                electrolyte-side concentration
   * @param[in] kinetic_model       kinetic model of scatra-scatra interface condition
   * @param[in] s2i_condition_type  scatra-scatra interface condition type
   * @return exchange mass flux density of growth reaction
   */
  double calculate_growth_exchange_mass_flux_density(double kr, double alpha_a, double c_el,
      int kinetic_model, const Core::Conditions::ConditionType& s2i_condition_type);

  /*!
   * \brief compute the mass flux density of growth kinetics via local Newton-Raphson iteration
   *
   * @param[in] j0           exchange mass flux density
   * @param[in] frt          factor F/(RT)
   * @param[in] pot_ed       electrode-side electric potential
   * @param[in] pot_el       electrolyte-side electric potential
   * @param[in] epd          half-cell open-circuit potential
   * @param[in] resistance   scatra-scatra interface layer resistance
   * @param[in] thickness    scatra-scatra interface layer thickness
   * @param[in] faraday          faraday constant
   * @param[in] scatraparameterstd       scatra ele std parameter class
   * @param[in] scatraeleparamsboundary  scatra ele boundary parameter class
   * @return Butler-Volmer mass flux density
   */
  double calculate_growth_mass_flux_density(double j0, double frt, double pot_ed, double pot_el,
      double epd, double resistance, double thickness, double faraday,
      const Discret::Elements::ScaTraEleParameterStd* const scatraparameterstd,
      const Discret::Elements::ScaTraEleParameterBoundary* const scatraeleparamsboundary);

  /*!
   * \brief calculate the linearizations associated with Butler-Volmer mass flux density leading
   * to interface growth w.r.t. the electrochemical primary variables
   *
   * @param[in] j0               exchange mass flux density
   * @param[in] frt              factor F/(RT)
   * @param[in] epdderiv         derivative of equilibrium electric potential difference w.r.t.
   *                             concentration at electrode surface
   * @param[in] eta              electrode-electrolyte overpotential at integration point
   * @param[in] resistance       ohmic resistance on the interface
   * @param[in] regfac           regularization factor for scatra-scatra interface growth
   * @param[in] emasterphiint    state variables on master-side integration points
   * @param[in] eslavephiint     state variables on slave-side integration points
   * @param[in] cmax             saturation value of intercalated lithium concentration from
   *                             electrode material
   * @param[in] scatraeleparamsboundary  scatra ele boundary parameter class
   * @param[out] dj_dc_slave     linearization of Butler-Volmer mass flux density w.r.t.
   *                             concentration on slave-side
   * @param[out] dj_dc_master    linearization of Butler-Volmer mass flux density w.r.t.
   *                             concentration on master-side
   * @param[out] dj_dpot_slave   linearization of Butler-Volmer mass flux density w.r.t.
   *                             electric potential on slave-side
   * @param[out] dj_dpot_master  linearization of Butler-Volmer mass flux density w.r.t.
   *                             electric potential on master-side
   */
  void calculate_s2_i_growth_elch_linearizations(double j0, double frt, double epdderiv, double eta,
      double resistance, double regfac, double emasterphiint, double eslavephiint, double cmax,
      const Discret::Elements::ScaTraEleParameterBoundary* const scatraeleparamsboundary,
      double& dj_dc_slave, double& dj_dc_master, double& dj_dpot_slave, double& dj_dpot_master);

  /*!
   * \brief calculate linearizations associated with Butler-Volmer mass flux density w.r.t. the
   * interface film thickness
   *
   * @param[in] j0            exchange mass flux density
   * @param[in] j             interface mass flux density
   * @param[in] frt           factor F/(RT)
   * @param[in] resistivity   resistivity associated to the material of the interface film
   * @param[in] resistance    ohmic resistance of the interface
   * @param[in] regfac        regularization factor for scatra-scatra interface growth
   * @param[in] regfacderiv   derivative of the regularization factor for scatra-scatra
   *                          interface growth w.r.t. the interface film thickness
   * @param[in] expterm1      first exponential term of Butler-Volmer equation
   * @param[in] expterm2      second exponential term of Butler-Volmer equation
   * @param[in] scatraeleparamsboundary   scatra ele boundary parameter class
   * @return  linearization of Butler-Volmer mass flux density w.r.t. interface film thickness
   */
  double calculate_s2_i_elch_growth_linearizations(double j0, double j, double frt,
      double resistivity, double resistance, double regfac, double regfacderiv, double expterm1,
      double expterm2,
      const Discret::Elements::ScaTraEleParameterBoundary* const scatraeleparamsboundary);

  /*!
   * \brief compute regularization factor for lithium stripping
   *
   * @param[in] thickness    scatra-scatra interface layer thickness
   * @param[in] eta          electrode-electrolyte overpotential at integration point
   * @param[in] scatraeleparamsboundary   scatra ele boundary parameter class
   * @return  return the regularization factor if regularization is applied
   */
  double get_regularization_factor(const double thickness, const double eta,
      const Discret::Elements::ScaTraEleParameterBoundary* const scatraeleparamsboundary);


  /*!
   * \brief compute derivative of regularization factor for lithium stripping w.r.t. thickness
   * of plated lithium
   *
   * @param[in] thickness     scatra-scatra interface layer thickness
   * @param[in] eta           electrode-electrolyte overpotential at integration point
   * @param[in] scatraeleparamsboundary   scatra ele boundary parameter class
   * @return  return the derivative of the regularization factor w.r.t. thickness of deposited
   *          material if regularization is applied
   */
  double get_regularization_factor_derivative(const double thickness, const double eta,
      const Discret::Elements::ScaTraEleParameterBoundary* const scatraeleparamsboundary);

}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif
