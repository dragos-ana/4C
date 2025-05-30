// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_ELE_CALC_LS_HPP
#define FOUR_C_SCATRA_ELE_CALC_LS_HPP

#include "4C_config.hpp"

#include "4C_scatra_ele_calc.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    template <Core::FE::CellType distype>
    class ScaTraEleCalcLS : public ScaTraEleCalc<distype>
    {
     private:
      //! private constructor for singletons
      ScaTraEleCalcLS(const int numdofpernode, const int numscal, const std::string& disname);

      using my = ScaTraEleCalc<distype>;
      using my::nen_;
      using my::nsd_;

     public:
      /// Singleton access method
      static ScaTraEleCalcLS<distype>* instance(
          const int numdofpernode, const int numscal, const std::string& disname);

      int evaluate_action(Core::Elements::Element* ele, Teuchos::ParameterList& params,
          Core::FE::Discretization& discretization, const ScaTra::Action& action,
          Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
          Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseVector& elevec2,
          Core::LinAlg::SerialDenseVector& elevec3) override;

     private:
      // calculate error compared to analytical solution
      void cal_error_compared_to_analyt_solution(const Core::Elements::Element* ele,
          const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephizero,
          Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector& errors);

      // smoothed heaviside function
      void smooth_heaviside_function(const double charelelength, const double phi, double& smoothH);
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
