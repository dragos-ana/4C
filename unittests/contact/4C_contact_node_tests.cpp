// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_contact_friction_node.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

namespace
{
  using namespace FourC;

  TEST(FrictionNodeTest, TangentialTractionProjectedConsistently)
  {
    const std::array<double, 3> coordinates{0.0, 0.0, 0.0};
    const std::vector<int> dofs{10, 11, 12};

    CONTACT::FriNode friction_node(1, coordinates, 0, dofs, true, false, false);
    friction_node.initialize_data_container();

    FOUR_C_THROW("Throw");
  }

}  // namespace
