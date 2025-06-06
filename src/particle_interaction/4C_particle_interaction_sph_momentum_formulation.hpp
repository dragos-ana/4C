// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_PARTICLE_INTERACTION_SPH_MOMENTUM_FORMULATION_HPP
#define FOUR_C_PARTICLE_INTERACTION_SPH_MOMENTUM_FORMULATION_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class SPHMomentumFormulationBase
  {
   public:
    //! constructor
    explicit SPHMomentumFormulationBase();

    //! virtual destructor
    virtual ~SPHMomentumFormulationBase() = default;

    //! init momentum formulation handler
    virtual void init();

    //! setup momentum formulation handler
    virtual void setup();

    //! evaluate specific coefficient
    virtual void specific_coefficient(const double* dens_i, const double* dens_j,
        const double* mass_i, const double* mass_j, const double& dWdrij, const double& dWdrji,
        double* speccoeff_ij, double* speccoeff_ji) const = 0;

    //! evaluate pressure gradient
    virtual void pressure_gradient(const double* dens_i, const double* dens_j,
        const double* press_i, const double* press_j, const double& speccoeff_ij,
        const double& speccoeff_ji, const double* e_ij, double* acc_i, double* acc_j) const = 0;

    //! evaluate shear forces
    virtual void shear_forces(const double* dens_i, const double* dens_j, const double* vel_i,
        const double* vel_j, const double& kernelfac, const double& visc_i, const double& visc_j,
        const double& bulk_visc_i, const double& bulk_visc_j, const double& abs_rij,
        const double& speccoeff_ij, const double& speccoeff_ji, const double* e_ij, double* acc_i,
        double* acc_j) const = 0;

    //! evaluate background pressure (standard formulation)
    virtual void standard_background_pressure(const double* dens_i, const double* dens_j,
        const double& bg_press_i, const double& bg_press_j, const double& speccoeff_ij,
        const double& speccoeff_ji, const double* e_ij, double* mod_acc_i,
        double* mod_acc_j) const = 0;

    //! evaluate background pressure  (generalized formulation)
    virtual void generalized_background_pressure(const double* dens_i, const double* dens_j,
        const double* mass_i, const double* mass_j, const double& mod_bg_press_i,
        const double& mod_bg_press_j, const double& mod_dWdrij, const double& mod_dWdrji,
        const double* e_ij, double* mod_acc_i, double* mod_acc_j) const = 0;

    //! evaluate modified velocity contribution
    virtual void modified_velocity_contribution(const double* dens_i, const double* dens_j,
        const double* vel_i, const double* vel_j, const double* mod_vel_i, const double* mod_vel_j,
        const double& speccoeff_ij, const double& speccoeff_ji, const double* e_ij, double* acc_i,
        double* acc_j) const = 0;
  };

  class SPHMomentumFormulationMonaghan final : public SPHMomentumFormulationBase
  {
   public:
    //! constructor
    explicit SPHMomentumFormulationMonaghan();

    //! evaluate specific coefficient
    void specific_coefficient(const double* dens_i, const double* dens_j, const double* mass_i,
        const double* mass_j, const double& dWdrij, const double& dWdrji, double* speccoeff_ij,
        double* speccoeff_ji) const override;

    //! evaluate pressure gradient
    void pressure_gradient(const double* dens_i, const double* dens_j, const double* press_i,
        const double* press_j, const double& speccoeff_ij, const double& speccoeff_ji,
        const double* e_ij, double* acc_i, double* acc_j) const override;

    //! evaluate shear forces
    void shear_forces(const double* dens_i, const double* dens_j, const double* vel_i,
        const double* vel_j, const double& kernelfac, const double& visc_i, const double& visc_j,
        const double& bulk_visc_i, const double& bulk_visc_j, const double& abs_rij,
        const double& speccoeff_ij, const double& speccoeff_ji, const double* e_ij, double* acc_i,
        double* acc_j) const override;

    //! evaluate background pressure (standard formulation)
    void standard_background_pressure(const double* dens_i, const double* dens_j,
        const double& bg_press_i, const double& bg_press_j, const double& speccoeff_ij,
        const double& speccoeff_ji, const double* e_ij, double* mod_acc_i,
        double* mod_acc_j) const override;

    //! evaluate background pressure (generalized formulation)
    void generalized_background_pressure(const double* dens_i, const double* dens_j,
        const double* mass_i, const double* mass_j, const double& mod_bg_press_i,
        const double& mod_bg_press_j, const double& mod_dWdrij, const double& mod_dWdrji,
        const double* e_ij, double* mod_acc_i, double* mod_acc_j) const override;

    //! evaluate modified velocity contribution
    void modified_velocity_contribution(const double* dens_i, const double* dens_j,
        const double* vel_i, const double* vel_j, const double* mod_vel_i, const double* mod_vel_j,
        const double& speccoeff_ij, const double& speccoeff_ji, const double* e_ij, double* acc_i,
        double* acc_j) const override;
  };

  class SPHMomentumFormulationAdami final : public SPHMomentumFormulationBase
  {
   public:
    //! constructor
    explicit SPHMomentumFormulationAdami();

    //! evaluate specific coefficient
    void specific_coefficient(const double* dens_i, const double* dens_j, const double* mass_i,
        const double* mass_j, const double& dWdrij, const double& dWdrji, double* speccoeff_ij,
        double* speccoeff_ji) const override;

    //! evaluate pressure gradient
    void pressure_gradient(const double* dens_i, const double* dens_j, const double* press_i,
        const double* press_j, const double& speccoeff_ij, const double& speccoeff_ji,
        const double* e_ij, double* acc_i, double* acc_j) const override;

    //! evaluate shear forces
    void shear_forces(const double* dens_i, const double* dens_j, const double* vel_i,
        const double* vel_j, const double& kernelfac, const double& visc_i, const double& visc_j,
        const double& bulk_visc_i, const double& bulk_visc_j, const double& abs_rij,
        const double& speccoeff_ij, const double& speccoeff_ji, const double* e_ij, double* acc_i,
        double* acc_j) const override;

    //! evaluate background pressure (standard formulation)
    void standard_background_pressure(const double* dens_i, const double* dens_j,
        const double& bg_press_i, const double& bg_press_j, const double& speccoeff_ij,
        const double& speccoeff_ji, const double* e_ij, double* mod_acc_i,
        double* mod_acc_j) const override;

    //! evaluate background pressure (generalized formulation)
    void generalized_background_pressure(const double* dens_i, const double* dens_j,
        const double* mass_i, const double* mass_j, const double& mod_bg_press_i,
        const double& mod_bg_press_j, const double& mod_dWdrij, const double& mod_dWdrji,
        const double* e_ij, double* mod_acc_i, double* mod_acc_j) const override;

    //! evaluate modified velocity contribution
    void modified_velocity_contribution(const double* dens_i, const double* dens_j,
        const double* vel_i, const double* vel_j, const double* mod_vel_i, const double* mod_vel_j,
        const double& speccoeff_ij, const double& speccoeff_ji, const double* e_ij, double* acc_i,
        double* acc_j) const override;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
