// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_TURBULENCE_HIT_SCALAR_FORCING_HPP
#define FOUR_C_SCATRA_TURBULENCE_HIT_SCALAR_FORCING_HPP

#include "4C_config.hpp"

#include "4C_inpar_fluid.hpp"
#include "4C_inpar_scatra.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace ScaTra
{
  // forward declarations
  class ScaTraTimIntImpl;

  class HomoIsoTurbScalarForcing
  {
   public:
    //! constructor
    HomoIsoTurbScalarForcing(ScaTraTimIntImpl* timeint);

    //! initialize with initial spectrum
    void set_initial_spectrum(Inpar::ScaTra::InitialField init_field_type);

    //! turn on forcing
    void activate_forcing(const bool activate);

    //! calculate power input
    void calculate_forcing(const int step);

    //! get forcing
    void update_forcing(const int step);

    //! time update of energy spectrum
    void time_update_forcing();

   protected:
    //! sort criterium for double values up to a tolerance of 10-9
    class LineSortCriterion
    {
     public:
      bool operator()(const double& p1, const double& p2) const { return (p1 < p2 - 1E-9); }

     protected:
     private:
    };

   private:
    //! type of forcing
    Inpar::FLUID::ForcingType forcing_type_;

    //! scatra discretization
    std::shared_ptr<Core::FE::Discretization> discret_;

    //! state vector of volume force to be computed
    std::shared_ptr<Core::LinAlg::Vector<double>> forcing_;

    //! state vectors used to compute forcing
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> phiaf_;

    //! threshold wave number for forcing
    //! i.e., forcing is applied to wave numbers <= threshold wave number
    double threshold_wavenumber_;

    //! identify gen-alpha time integration
    bool is_genalpha_;

    //! number of resolved mode
    int nummodes_;

    //! vector of coordinates in one spatial direction (same for the other two directions)
    std::shared_ptr<std::vector<double>> coordinates_;

    //! vector of wave numbers
    std::shared_ptr<std::vector<double>> wavenumbers_;

    //! vector scalar variance spectrum (sum over k=const) at time n
    std::shared_ptr<std::vector<double>> scalarvariancespectrum_n_;

    //! vector scalar variance spectrum  (sum over k=const) at time n+1/n+af
    std::shared_ptr<std::vector<double>> scalarvariancespectrum_np_;

    //! time step length
    double dt_;

    //! flag to activate forcing
    bool activate_;

    //! linear compensation factor
    std::shared_ptr<Core::LinAlg::SerialDenseVector> force_fac_;

    //! interpolation function
    static double interpolate(
        const double& x, const double& x_1, const double& x_2, const double& y_1, const double& y_2)
    {
      const double value = y_1 + (y_2 - y_1) / (x_2 - x_1) * (x - x_1);
      return value;
    }
  };

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
