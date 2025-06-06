// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_ELCH_DIFFCOND_MULTISCALE_HPP
#define FOUR_C_SCATRA_ELE_CALC_ELCH_DIFFCOND_MULTISCALE_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc_elch_diffcond.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    // forward declaration
    class ScaTraEleDiffManagerElchDiffCondMultiScale;

    // implementation of class ScaTraEleCalcElchDiffCondMultiScale
    template <Core::FE::CellType distype, int probdim = Core::FE::dim<distype>>
    class ScaTraEleCalcElchDiffCondMultiScale : public ScaTraEleCalcElchDiffCond<distype, probdim>
    {
     public:
      //! singleton access method
      static ScaTraEleCalcElchDiffCondMultiScale<distype, probdim>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);



     private:
      //! abbreviation
      using my = ScaTraEleCalc<distype, probdim>;
      using mydiffcond = ScaTraEleCalcElchDiffCond<distype, probdim>;
      using my::nen_;
      using my::nsd_;
      using my::nsd_ele_;

      //! constructor
      ScaTraEleCalcElchDiffCondMultiScale(
          const int numdofpernode, const int numscal, const std::string& disname);

      //! macro-scale matrix and vector contributions arising from macro-micro coupling in
      //! multi-scale simulations
      void calc_mat_and_rhs_multi_scale(const Core::Elements::Element* const ele,  //!< element
          Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
          Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
          const int k,                            //!< species index
          const int iquad,                        //!< Gauss point index
          const double timefacfac,  //!< domain integration factor times time integration factor
          const double rhsfac       //!< domain integration factor times time integration factor for
                                    //!< right-hand side vector
          ) override;

      //! calculate electrode state of charge and C rate
      void calculate_electrode_soc_and_c_rate(
          const Core::Elements::Element* const& ele,       //!< the element we are dealing with
          const Core::FE::Discretization& discretization,  //!< discretization
          Core::Elements::LocationArray& la,               //!< location array
          Core::LinAlg::SerialDenseVector&
              scalars  //!< result vector for scalar integrals to be computed
          ) final;

      void calculate_mean_electrode_concentration(const Core::Elements::Element* const& ele,
          const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
          Core::LinAlg::SerialDenseVector& conc) override;

      void calculate_scalars(const Core::Elements::Element* ele,
          Core::LinAlg::SerialDenseVector& scalars, bool inverting, bool calc_grad_phi) override;

      //! get diffusion manager
      std::shared_ptr<ScaTraEleDiffManagerElchDiffCondMultiScale> diff_manager() const
      {
        return std::static_pointer_cast<ScaTraEleDiffManagerElchDiffCondMultiScale>(
            my::diffmanager_);
      };

      //! evaluate action
      int evaluate_action(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, const ScaTra::Action& action,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

      //! compute element matrix and element right-hand side vector
      void sysmat(Core::Elements::Element* ele,       //!< element
          Core::LinAlg::SerialDenseMatrix& emat,      //!< element matrix
          Core::LinAlg::SerialDenseVector& erhs,      //!< element right-hand side vector
          Core::LinAlg::SerialDenseVector& subgrdiff  //!< subgrid diffusivity vector
          ) override;
    };  // class implementation


    // implementation of class ScaTraEleDiffManagerElchDiffCondMultiScale
    class ScaTraEleDiffManagerElchDiffCondMultiScale : public ScaTraEleDiffManagerElchDiffCond
    {
     public:
      //! constructor
      ScaTraEleDiffManagerElchDiffCondMultiScale(int numscal)
          : ScaTraEleDiffManagerElchDiffCond(numscal) {};

      //! Output of transport parameter (to screen)
      void output_transport_params(const int numscal) override
      {
        // call base class routine
        ScaTraEleDiffManagerElchDiffCond::output_transport_params(numscal);
      };
    };
  }  // namespace Elements
}  // namespace Discret
FOUR_C_NAMESPACE_CLOSE

#endif
