// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_combintersection.hpp"
#include "4C_cut_levelsetintersection.hpp"
#include "4C_cut_meshintersection.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_side.hpp"
#include "4C_cut_tetmeshintersection.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cut_test_utils.hpp"

void test_generated_622320()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826088694;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68559);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = -0.1620434782608695945;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68575);
    tri3_xyze(0, 2) = 1.4945652173913044347;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11259);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.017333333333333332538;
    nids.push_back(69310);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.1620434782608695945;
    tri3_xyze(2, 1) = -0.017333333333333325599;
    nids.push_back(69326);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.016000000000000000333;
    nids.push_back(-11469);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.1620434782608695945;
    tri3_xyze(2, 0) = -0.017333333333333325599;
    nids.push_back(69326);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.1620434782608695945;
    tri3_xyze(2, 1) = -0.014666666666666669863;
    nids.push_back(69325);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.016000000000000000333;
    nids.push_back(-11469);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.1620434782608695945;
    tri3_xyze(2, 0) = -0.014666666666666669863;
    nids.push_back(69325);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.17291304347826083609;
    tri3_xyze(2, 1) = -0.014666666666666668128;
    nids.push_back(69309);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.016000000000000000333;
    nids.push_back(-11469);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.017333333333333332538;
    nids.push_back(69310);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(69311);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = -0.1674782608695652153;
    tri3_xyze(2, 2) = -0.018666666666666664742;
    nids.push_back(-11470);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(69311);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.16204347826086956674;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(69327);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = -0.1674782608695652153;
    tri3_xyze(2, 2) = -0.018666666666666664742;
    nids.push_back(-11470);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.16204347826086956674;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(69327);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.1620434782608695945;
    tri3_xyze(2, 1) = -0.017333333333333325599;
    nids.push_back(69326);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = -0.1674782608695652153;
    tri3_xyze(2, 2) = -0.018666666666666664742;
    nids.push_back(-11470);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.1620434782608695945;
    tri3_xyze(2, 0) = -0.017333333333333325599;
    nids.push_back(69326);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.017333333333333332538;
    nids.push_back(69310);
    tri3_xyze(0, 2) = 1.5;
    tri3_xyze(1, 2) = -0.1674782608695652153;
    tri3_xyze(2, 2) = -0.018666666666666664742;
    nids.push_back(-11470);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4673913043478261642;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67055);
    tri3_xyze(0, 1) = 1.4782608695652172948;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(67807);
    tri3_xyze(0, 2) = 1.4728260869565215074;
    tri3_xyze(1, 2) = -0.1783478260869565124;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11013);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67807);
    tri3_xyze(0, 1) = 1.4782608695652172948;
    tri3_xyze(1, 1) = -0.1837826086956521332;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(67791);
    tri3_xyze(0, 2) = 1.4728260869565215074;
    tri3_xyze(1, 2) = -0.1783478260869565124;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11013);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = -0.1620434782608695945;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67823);
    tri3_xyze(0, 1) = 1.4782608695652172948;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(67807);
    tri3_xyze(0, 2) = 1.4728260869565215074;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11015);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67807);
    tri3_xyze(0, 1) = 1.4673913043478261642;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(67055);
    tri3_xyze(0, 2) = 1.4728260869565215074;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11015);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = -0.1837826086956521332;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67791);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = -0.1837826086956521332;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68543);
    tri3_xyze(0, 2) = 1.4836956521739128601;
    tri3_xyze(1, 2) = -0.18921739130434778176;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11133);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = -0.1837826086956521332;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68543);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = -0.19465217391304345806;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68527);
    tri3_xyze(0, 2) = 1.4836956521739128601;
    tri3_xyze(1, 2) = -0.18921739130434778176;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11133);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = -0.1837826086956521332;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67791);
    tri3_xyze(0, 1) = 1.4782608695652172948;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(67807);
    tri3_xyze(0, 2) = 1.4836956521739128601;
    tri3_xyze(1, 2) = -0.17834782608695648465;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11135);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67807);
    tri3_xyze(0, 1) = 1.4891304347826088694;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68559);
    tri3_xyze(0, 2) = 1.4836956521739128601;
    tri3_xyze(1, 2) = -0.17834782608695648465;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11135);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826088694;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68559);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = -0.1837826086956521332;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68543);
    tri3_xyze(0, 2) = 1.4836956521739128601;
    tri3_xyze(1, 2) = -0.17834782608695648465;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11135);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = -0.1837826086956521332;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68543);
    tri3_xyze(0, 1) = 1.4782608695652172948;
    tri3_xyze(1, 1) = -0.1837826086956521332;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(67791);
    tri3_xyze(0, 2) = 1.4836956521739128601;
    tri3_xyze(1, 2) = -0.17834782608695648465;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11135);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67807);
    tri3_xyze(0, 1) = 1.4782608695652172948;
    tri3_xyze(1, 1) = -0.1620434782608695945;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(67823);
    tri3_xyze(0, 2) = 1.4836956521739130821;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11137);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4782608695652172948;
    tri3_xyze(1, 0) = -0.1620434782608695945;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(67823);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = -0.1620434782608695945;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68575);
    tri3_xyze(0, 2) = 1.4836956521739130821;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11137);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = -0.1620434782608695945;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68575);
    tri3_xyze(0, 1) = 1.4891304347826088694;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68559);
    tri3_xyze(0, 2) = 1.4836956521739130821;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11137);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826088694;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68559);
    tri3_xyze(0, 1) = 1.4782608695652172948;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(67807);
    tri3_xyze(0, 2) = 1.4836956521739130821;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11137);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = -0.19465217391304345806;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68527);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = -0.1837826086956521332;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68543);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = -0.18921739130434780951;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11255);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = -0.1837826086956521332;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68543);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.18378260869565216096;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(69295);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = -0.18921739130434780951;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11255);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.18378260869565216096;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(69295);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.19465217391304345806;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(69279);
    tri3_xyze(0, 2) = 1.4945652173913042127;
    tri3_xyze(1, 2) = -0.18921739130434780951;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11255);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = -0.1837826086956521332;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68543);
    tri3_xyze(0, 1) = 1.4891304347826088694;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68559);
    tri3_xyze(0, 2) = 1.4945652173913044347;
    tri3_xyze(1, 2) = -0.17834782608695648465;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11257);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826088694;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68559);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(69311);
    tri3_xyze(0, 2) = 1.4945652173913044347;
    tri3_xyze(1, 2) = -0.17834782608695648465;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11257);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(69311);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.18378260869565216096;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(69295);
    tri3_xyze(0, 2) = 1.4945652173913044347;
    tri3_xyze(1, 2) = -0.17834782608695648465;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11257);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.18378260869565216096;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(69295);
    tri3_xyze(0, 1) = 1.4891304347826086474;
    tri3_xyze(1, 1) = -0.1837826086956521332;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68543);
    tri3_xyze(0, 2) = 1.4945652173913044347;
    tri3_xyze(1, 2) = -0.17834782608695648465;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11257);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.4891304347826086474;
    tri3_xyze(1, 0) = -0.1620434782608695945;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(68575);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.16204347826086956674;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(69327);
    tri3_xyze(0, 2) = 1.4945652173913044347;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11259);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.16204347826086956674;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(69327);
    tri3_xyze(0, 1) = 1.5;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(69311);
    tri3_xyze(0, 2) = 1.4945652173913044347;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11259);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 1.5;
    tri3_xyze(1, 0) = -0.17291304347826086385;
    tri3_xyze(2, 0) = -0.020000000000000000416;
    nids.push_back(69311);
    tri3_xyze(0, 1) = 1.4891304347826088694;
    tri3_xyze(1, 1) = -0.17291304347826086385;
    tri3_xyze(2, 1) = -0.020000000000000000416;
    nids.push_back(68559);
    tri3_xyze(0, 2) = 1.4945652173913044347;
    tri3_xyze(1, 2) = -0.16747826086956524305;
    tri3_xyze(2, 2) = -0.020000000000000000416;
    nids.push_back(-11259);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888837989634799985;
    hex8_xyze(1, 0) = -0.17777693745167399975;
    hex8_xyze(2, 0) = -0.0055550085050327501629;
    nids.push_back(1653793);
    hex8_xyze(0, 1) = 1.4888838267851700614;
    hex8_xyze(1, 1) = -0.17777695280934899258;
    hex8_xyze(2, 1) = -0.016665064572966199058;
    nids.push_back(1653794);
    hex8_xyze(0, 2) = 1.4888841561094099397;
    hex8_xyze(1, 2) = -0.16666619296931300953;
    hex8_xyze(2, 2) = -0.016665065045985600484;
    nids.push_back(1653797);
    hex8_xyze(0, 3) = 1.4888841103540999544;
    hex8_xyze(1, 3) = -0.16666617600415301048;
    hex8_xyze(2, 3) = -0.005555010668345490045;
    nids.push_back(1653796);
    hex8_xyze(0, 4) = 1.4999993064943799581;
    hex8_xyze(1, 4) = -0.17777652881863101331;
    hex8_xyze(2, 4) = -0.0055550171282643398887;
    nids.push_back(1653802);
    hex8_xyze(0, 5) = 1.4999994621857399846;
    hex8_xyze(1, 5) = -0.17777654050145599851;
    hex8_xyze(2, 5) = -0.016665059052471998396;
    nids.push_back(1653803);
    hex8_xyze(0, 6) = 1.5000000240384498973;
    hex8_xyze(1, 6) = -0.16666591955666298919;
    hex8_xyze(2, 6) = -0.016665059299914599528;
    nids.push_back(1653806);
    hex8_xyze(0, 7) = 1.4999998115573600632;
    hex8_xyze(1, 7) = -0.16666590305784700909;
    hex8_xyze(2, 7) = -0.0055550208030329603637;
    nids.push_back(1653805);

    intersection.add_element(622295, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4777661106417701031;
    hex8_xyze(1, 0) = -0.17777726712363900452;
    hex8_xyze(2, 0) = -0.016665083666418001318;
    nids.push_back(1653785);
    hex8_xyze(0, 1) = 1.4777660478293299651;
    hex8_xyze(1, 1) = -0.17777728059035699526;
    hex8_xyze(2, 1) = -0.027775305529144801026;
    nids.push_back(1653810);
    hex8_xyze(0, 2) = 1.4777662284390200576;
    hex8_xyze(1, 2) = -0.1666663972204379951;
    hex8_xyze(2, 2) = -0.027775304035099299127;
    nids.push_back(1653813);
    hex8_xyze(0, 3) = 1.4777662917684499799;
    hex8_xyze(1, 3) = -0.16666638549249698786;
    hex8_xyze(2, 3) = -0.016665084007053600906;
    nids.push_back(1653788);
    hex8_xyze(0, 4) = 1.4888838267851700614;
    hex8_xyze(1, 4) = -0.17777695280934899258;
    hex8_xyze(2, 4) = -0.016665064572966199058;
    nids.push_back(1653794);
    hex8_xyze(0, 5) = 1.4888839412908099202;
    hex8_xyze(1, 5) = -0.17777696543540999485;
    hex8_xyze(2, 5) = -0.027775223178406700797;
    nids.push_back(1653819);
    hex8_xyze(0, 6) = 1.4888842535221900043;
    hex8_xyze(1, 6) = -0.16666620405004300975;
    hex8_xyze(2, 6) = -0.027775221752330700453;
    nids.push_back(1653822);
    hex8_xyze(0, 7) = 1.4888841561094099397;
    hex8_xyze(1, 7) = -0.16666619296931300953;
    hex8_xyze(2, 7) = -0.016665065045985600484;
    nids.push_back(1653797);

    intersection.add_element(622311, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888834234834800174;
    hex8_xyze(1, 0) = -0.18888764031666099852;
    hex8_xyze(2, 0) = -0.016665064588537399637;
    nids.push_back(1653791);
    hex8_xyze(0, 1) = 1.4888835378021600953;
    hex8_xyze(1, 1) = -0.18888765026904699718;
    hex8_xyze(2, 1) = -0.02777522336858210053;
    nids.push_back(1653816);
    hex8_xyze(0, 2) = 1.4888839412908099202;
    hex8_xyze(1, 2) = -0.17777696543540999485;
    hex8_xyze(2, 2) = -0.027775223178406700797;
    nids.push_back(1653819);
    hex8_xyze(0, 3) = 1.4888838267851700614;
    hex8_xyze(1, 3) = -0.17777695280934899258;
    hex8_xyze(2, 3) = -0.016665064572966199058;
    nids.push_back(1653794);
    hex8_xyze(0, 4) = 1.4999989494998000605;
    hex8_xyze(1, 4) = -0.18888708273383600367;
    hex8_xyze(2, 4) = -0.016665059765062800734;
    nids.push_back(1653800);
    hex8_xyze(0, 5) = 1.4999992964859600875;
    hex8_xyze(1, 5) = -0.18888708808247900439;
    hex8_xyze(2, 5) = -0.027775159007911699727;
    nids.push_back(1653825);
    hex8_xyze(0, 6) = 1.4999998089317700956;
    hex8_xyze(1, 6) = -0.17777655412602100249;
    hex8_xyze(2, 6) = -0.02777515755660750138;
    nids.push_back(1653828);
    hex8_xyze(0, 7) = 1.4999994621857399846;
    hex8_xyze(1, 7) = -0.17777654050145599851;
    hex8_xyze(2, 7) = -0.016665059052471998396;
    nids.push_back(1653803);

    intersection.add_element(622317, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888838267851700614;
    hex8_xyze(1, 0) = -0.17777695280934899258;
    hex8_xyze(2, 0) = -0.016665064572966199058;
    nids.push_back(1653794);
    hex8_xyze(0, 1) = 1.4888839412908099202;
    hex8_xyze(1, 1) = -0.17777696543540999485;
    hex8_xyze(2, 1) = -0.027775223178406700797;
    nids.push_back(1653819);
    hex8_xyze(0, 2) = 1.4888842535221900043;
    hex8_xyze(1, 2) = -0.16666620405004300975;
    hex8_xyze(2, 2) = -0.027775221752330700453;
    nids.push_back(1653822);
    hex8_xyze(0, 3) = 1.4888841561094099397;
    hex8_xyze(1, 3) = -0.16666619296931300953;
    hex8_xyze(2, 3) = -0.016665065045985600484;
    nids.push_back(1653797);
    hex8_xyze(0, 4) = 1.4999994621857399846;
    hex8_xyze(1, 4) = -0.17777654050145599851;
    hex8_xyze(2, 4) = -0.016665059052471998396;
    nids.push_back(1653803);
    hex8_xyze(0, 5) = 1.4999998089317700956;
    hex8_xyze(1, 5) = -0.17777655412602100249;
    hex8_xyze(2, 5) = -0.02777515755660750138;
    nids.push_back(1653828);
    hex8_xyze(0, 6) = 1.5000003147648799384;
    hex8_xyze(1, 6) = -0.16666592794941101352;
    hex8_xyze(2, 6) = -0.027775153874800999343;
    nids.push_back(1653831);
    hex8_xyze(0, 7) = 1.5000000240384498973;
    hex8_xyze(1, 7) = -0.16666591955666298919;
    hex8_xyze(2, 7) = -0.016665059299914599528;
    nids.push_back(1653806);

    intersection.add_element(622320, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 1.4888839412908099202;
    hex8_xyze(1, 0) = -0.17777696543540999485;
    hex8_xyze(2, 0) = -0.027775223178406700797;
    nids.push_back(1653819);
    hex8_xyze(0, 1) = 1.4888841799043899261;
    hex8_xyze(1, 1) = -0.17777696861243399984;
    hex8_xyze(2, 1) = -0.038885514565783700636;
    nids.push_back(1653820);
    hex8_xyze(0, 2) = 1.4888844925463400326;
    hex8_xyze(1, 2) = -0.16666620689003999733;
    hex8_xyze(2, 2) = -0.03888551682983770047;
    nids.push_back(1653823);
    hex8_xyze(0, 3) = 1.4888842535221900043;
    hex8_xyze(1, 3) = -0.16666620405004300975;
    hex8_xyze(2, 3) = -0.027775221752330700453;
    nids.push_back(1653822);
    hex8_xyze(0, 4) = 1.4999998089317700956;
    hex8_xyze(1, 4) = -0.17777655412602100249;
    hex8_xyze(2, 4) = -0.02777515755660750138;
    nids.push_back(1653828);
    hex8_xyze(0, 5) = 1.5000003958535998994;
    hex8_xyze(1, 5) = -0.17777655433521999395;
    hex8_xyze(2, 5) = -0.038885343066824000491;
    nids.push_back(1653829);
    hex8_xyze(0, 6) = 1.5000009009827699469;
    hex8_xyze(1, 6) = -0.16666592683110401096;
    hex8_xyze(2, 6) = -0.038885345436736697133;
    nids.push_back(1653832);
    hex8_xyze(0, 7) = 1.5000003147648799384;
    hex8_xyze(1, 7) = -0.16666592794941101352;
    hex8_xyze(2, 7) = -0.027775153874800999343;
    nids.push_back(1653831);

    intersection.add_element(622321, nids, hex8_xyze, Core::FE::CellType::hex8);
  }

  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
