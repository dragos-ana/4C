// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_utils_local_integration.hpp"

#include <Sacado.hpp>
#include <Sacado_Fad_DFad.hpp>

FOUR_C_NAMESPACE_OPEN

namespace
{
  using FADdouble = Sacado::Fad::DFad<double>;

  template <int coeff0, int coeff1, int coeff2>
  constexpr auto QUADRATIC_FUNCTION = [](auto x)
  { return std::make_tuple(x, coeff2 * std::pow(x, 2) + coeff1 * x + coeff0); };

  template <int coeff0, int coeff1>
  constexpr auto LINEAR_FUNCTION = [](auto x) { return std::make_tuple(x, coeff1 * x + coeff0); };

  TEST(CoreUtilsLocalIntegrationTest, integrate_simpson_step)
  {
    constexpr auto f1 = QUADRATIC_FUNCTION<1, 2, 1>;
    auto value1 = Core::Utils::integrate_simpson_step(
        0.5, std::get<1>(f1(0.0)), std::get<1>(f1(0.5)), std::get<1>(f1(1.0)));
    EXPECT_NEAR(value1, 7.0 / 3.0, 1e-8);

    constexpr auto f2 = QUADRATIC_FUNCTION<2, -2, 5>;
    auto value2 = Core::Utils::integrate_simpson_step(
        0.5, std::get<1>(f2(0.0)), std::get<1>(f2(0.5)), std::get<1>(f2(1.0)));
    EXPECT_NEAR(value2, 8.0 / 3.0, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateSimpsonEquidistant)
  {
    constexpr auto f1 = QUADRATIC_FUNCTION<1, 2, 1>;
    auto value1 = Core::Utils::integrate_simpson_step(f1(0.0), f1(0.5), f1(1.0));
    EXPECT_NEAR(value1, 7.0 / 3.0, 1e-8);

    constexpr auto f2 = QUADRATIC_FUNCTION<2, -2, 5>;
    auto value2 = Core::Utils::integrate_simpson_step(f2(0.0), f2(0.5), f2(1.0));
    EXPECT_NEAR(value2, 8.0 / 3.0, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateSimpsonNonEquidistant)
  {
    constexpr auto f1 = QUADRATIC_FUNCTION<1, 2, 1>;
    auto value1 = Core::Utils::integrate_simpson_step(f1(0.0), f1(0.9), f1(1.0));
    EXPECT_NEAR(value1, 7.0 / 3.0, 1e-8);

    constexpr auto f2 = QUADRATIC_FUNCTION<2, -2, 5>;
    auto value2 = Core::Utils::integrate_simpson_step(f2(0.0), f2(0.9), f2(1.0));
    EXPECT_NEAR(value2, 8.0 / 3.0, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateSimpsonBCEquidistant)
  {
    constexpr auto f1 = QUADRATIC_FUNCTION<1, 2, 1>;
    auto value1 = Core::Utils::integrate_simpson_step_bc(f1(0.0), f1(0.5), f1(1.0));
    EXPECT_NEAR(value1, 37.0 / 24.0, 1e-8);

    constexpr auto f2 = QUADRATIC_FUNCTION<2, -2, 5>;
    auto value2 = Core::Utils::integrate_simpson_step_bc(f2(0.0), f2(0.5), f2(1.0));
    EXPECT_NEAR(value2, 41.0 / 24.0, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateSimpsonBCNonEquidistant)
  {
    constexpr auto f1 = QUADRATIC_FUNCTION<1, 2, 1>;
    auto value1 = Core::Utils::integrate_simpson_step_bc(f1(0.0), f1(0.9), f1(1.0));
    EXPECT_NEAR(value1, 1141.0 / 3000.0, 1e-8);

    constexpr auto f2 = QUADRATIC_FUNCTION<2, -2, 5>;
    auto value2 = Core::Utils::integrate_simpson_step_bc(f2(0.0), f2(0.9), f2(1.0));
    EXPECT_NEAR(value2, 277.0 / 600.0, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateSimpsonBCEquidistantWithDerivative)
  {
    constexpr auto f1 = QUADRATIC_FUNCTION<2, -2, 5>;
    auto [value, derivative] =
        Core::Utils::integrate_simpson_step_bc_and_return_derivative_c(f1(0.0), f1(0.5), f1(1.0));
    EXPECT_NEAR(0.20833333333333333, derivative, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateSimpsonBCNonEquidistantWithDerivative)
  {
    constexpr auto f1 = QUADRATIC_FUNCTION<2, -2, 5>;
    auto [value, derivative] =
        Core::Utils::integrate_simpson_step_bc_and_return_derivative_c(f1(0.0), f1(0.9), f1(1.0));
    EXPECT_NEAR(0.048333333333333318, derivative, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateSimpsonBCNonEquidistantWithDerivativeWithFad)
  {
    auto function_compute_derivative = [](double x) -> std::tuple<double, FADdouble>
    {
      if (x == 1.2)
      {
        return {x, FADdouble(1, 0, x)};
      }

      return {x, x};
    };
    FADdouble fad_derivative =
        Core::Utils::integrate_simpson_step_bc(function_compute_derivative(-0.1),
            function_compute_derivative(0.2), function_compute_derivative(1.2));


    constexpr auto f1 = QUADRATIC_FUNCTION<1, 2, 1>;

    auto [value, derivative] =
        Core::Utils::integrate_simpson_step_bc_and_return_derivative_c(f1(-0.1), f1(0.2), f1(1.2));

    EXPECT_NEAR(value, 223.0 / 75.0, 1e-8);
    EXPECT_NEAR(fad_derivative.dx(0), derivative, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateTrapezoidal)
  {
    constexpr auto f = LINEAR_FUNCTION<1, 2>;
    auto value = Core::Utils::integrate_trapezoidal_step(f(0.0), f(1.0));
    EXPECT_NEAR(2, value, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateTrapezoidalDerivative)
  {
    constexpr auto f = LINEAR_FUNCTION<1, 2>;
    auto [value, derivative] =
        Core::Utils::integrate_trapezoidal_step_and_return_derivative_b(f(0.0), f(1.0));
    EXPECT_NEAR(2, value, 1e-8);
    EXPECT_NEAR(0.5, derivative, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateTrapezoidalDerivativeWithFad)
  {
    auto function_compute_derivative = [](double x) -> std::tuple<double, FADdouble>
    {
      if (x == 1.2)
      {
        return {x, FADdouble(1, 0, x)};
      }

      return {x, x};
    };
    FADdouble fad_derivative = Core::Utils::integrate_trapezoidal_step(
        function_compute_derivative(-0.1), function_compute_derivative(1.2));
    constexpr auto f = LINEAR_FUNCTION<1, 2>;
    auto [value, derivative] =
        Core::Utils::integrate_trapezoidal_step_and_return_derivative_b(f(-0.1), f(1.2));

    EXPECT_NEAR(value, 2.73, 1e-8);
    EXPECT_NEAR(fad_derivative.dx(0), derivative, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateIntegrateSimpsonTrapezoidal2Items)
  {
    std::array<double, 2> times = {0.0, 1.0};
    auto value = Core::Utils::integrate_simpson_trapezoidal(times, LINEAR_FUNCTION<1, 2>);
    EXPECT_NEAR(2, value, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateIntegrateSimpsonTrapezoidalEvenItems)
  {
    std::array times = {0.0, 0.1, 0.3, 0.5, 0.88, 1.0};
    auto value = Core::Utils::integrate_simpson_trapezoidal(times, QUADRATIC_FUNCTION<1, 2, 1>);
    EXPECT_NEAR(7.0 / 3.0, value, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateIntegrateSimpsonTrapezoidalOddItems)
  {
    std::array times = {0.0, 0.1, 0.3, 0.5, 0.88, 0.89, 1.0};
    auto value = Core::Utils::integrate_simpson_trapezoidal(times, QUADRATIC_FUNCTION<1, 2, 1>);
    EXPECT_NEAR(7.0 / 3.0, value, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateIntegrateSimpsonTrapezoidalOneItem)
  {
    std::array times = {0.0};
    auto value = Core::Utils::integrate_simpson_trapezoidal(times, QUADRATIC_FUNCTION<1, 2, 1>);
    EXPECT_NEAR(0, value, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateIntegrateSimpsonTrapezoidalZeroItem)
  {
    auto function = [](double x) -> std::tuple<double, double>
    { return {x, std::pow(x, 2) + 2 * x + 1}; };
    std::array<double, 0> times = {};
    auto value = Core::Utils::integrate_simpson_trapezoidal(times, function);
    EXPECT_NEAR(0, value, 1e-8);
  }

  TEST(CoreUtilsLocalIntegrationTest, IntegrateIntegrateSimpsonTrapezoidalMultipleIntervals)
  {
    auto function = [](double x) -> std::tuple<double, double>
    {
      if (x <= 1.0) return QUADRATIC_FUNCTION<1, 2, 1>(x);
      return QUADRATIC_FUNCTION<1, -2, 5>(x);
    };
    std::array times = {0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0};
    auto value = Core::Utils::integrate_simpson_trapezoidal(times, function);
    EXPECT_NEAR((7.0 + 29.0) / 3.0, value, 1e-8);
  }
}  // namespace
FOUR_C_NAMESPACE_CLOSE
