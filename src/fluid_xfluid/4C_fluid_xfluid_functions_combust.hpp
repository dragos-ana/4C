/*----------------------------------------------------------------------*/
/*! \file

\brief Managing and evaluating of spatial functions for combustion and two-phase flow problems

\level 2

*----------------------------------------------------------------------*/

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#ifndef FOUR_C_FLUID_XFLUID_FUNCTIONS_COMBUST_HPP
#define FOUR_C_FLUID_XFLUID_FUNCTIONS_COMBUST_HPP

FOUR_C_NAMESPACE_OPEN

namespace CORE::UTILS
{
  class FunctionManager;
}

namespace DRT
{
  class Discretization;

  // namespace UTILS

  namespace UTILS
  {
    /// add valid combustion-specific function lines
    void AddValidCombustFunctions(CORE::UTILS::FunctionManager& function_manager);

    /// special implementation for a level set test function "Zalesak's disk"
    class ZalesaksDiskFunction : public CORE::UTILS::FunctionOfSpaceTime
    {
     public:
      double Evaluate(const double* x, double t, std::size_t component) const override;

      [[nodiscard]] std::size_t NumberComponents() const override
      {
        FOUR_C_THROW("Number of components not defined for ZalesaksDiskFunction.");
      };
    };

    /// special implementation two-phase flow test case
    class CollapsingWaterColumnFunction : public CORE::UTILS::FunctionOfSpaceTime
    {
     public:
      double Evaluate(const double* x, double t, std::size_t component) const override;

      [[nodiscard]] std::size_t NumberComponents() const override
      {
        FOUR_C_THROW("Number of components not defined for CollapsingWaterColumnFunction.");
      };
    };
  }  // namespace UTILS
}  // namespace DRT

FOUR_C_NAMESPACE_CLOSE

#endif