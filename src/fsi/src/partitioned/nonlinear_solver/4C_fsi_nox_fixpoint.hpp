// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_NOX_FIXPOINT_HPP
#define FOUR_C_FSI_NOX_FIXPOINT_HPP

#include "4C_config.hpp"

#include <NOX_Direction_Generic.H>  // base class
#include <NOX_Direction_UserDefinedFactory.H>
#include <NOX_GlobalData.H>
#include <NOX_Utils.H>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace FSI
  {
    /*! \brief Calculates the fix point direction.

      Calculates the direction
      \f[
      d = f(x)
      \f]

      This is the old (established, trusted, whatever you want) way to
      solve the nonlinear FSI interface equations. The residuum \f$ f(x)
      \f$ is
      \f[
      f(x) = g = d_{\Gamma,i+1} - d_{\Gamma,i}
      \f]
      in the FSI context.

      To be used with relaxation (line search).

    <B>Parameters</B>

    None.

    */
    class FixPoint : public ::NOX::Direction::Generic
    {
     public:
      //! Constructor
      FixPoint(const Teuchos::RCP<::NOX::Utils>& utils, Teuchos::ParameterList& params);


      // derived
      bool reset(
          const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params) override;

      // derived
      bool compute(::NOX::Abstract::Vector& dir, ::NOX::Abstract::Group& group,
          const ::NOX::Solver::Generic& solver) override;

      // derived
      bool compute(::NOX::Abstract::Vector& dir, ::NOX::Abstract::Group& group,
          const ::NOX::Solver::LineSearchBased& solver) override;

     private:
      //! Print error message and throw error
      void throw_error(const std::string& functionName, const std::string& errorMsg);

     private:
      //! Printing Utils
      Teuchos::RCP<::NOX::Utils> utils_;
    };

    /// simple factory that creates fix point direction object
    class FixPointFactory : public ::NOX::Direction::UserDefinedFactory
    {
     public:
      Teuchos::RCP<::NOX::Direction::Generic> buildDirection(
          const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params) const override
      {
        return Teuchos::make_rcp<FixPoint>(gd->getUtils(), params);
      }
    };

  }  // namespace FSI
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
