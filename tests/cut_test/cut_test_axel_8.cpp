// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

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

void test_axel8()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.899542;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.635736;
    tri3_xyze(0, 1) = 0.884221;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.574284;
    tri3_xyze(0, 2) = 0.912837;
    tri3_xyze(1, 2) = 0.39778;
    tri3_xyze(2, 2) = 0.599785;
    nids.clear();
    nids.push_back(283);
    nids.push_back(284);
    nids.push_back(285);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.884221;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.574284;
    tri3_xyze(0, 1) = 0.926131;
    tri3_xyze(1, 1) = 0.374621;
    tri3_xyze(2, 1) = 0.563834;
    tri3_xyze(0, 2) = 0.912837;
    tri3_xyze(1, 2) = 0.39778;
    tri3_xyze(2, 2) = 0.599785;
    nids.clear();
    nids.push_back(284);
    nids.push_back(268);
    nids.push_back(285);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.926131;
    tri3_xyze(1, 0) = 0.374621;
    tri3_xyze(2, 0) = 0.563834;
    tri3_xyze(0, 1) = 0.884221;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.574284;
    tri3_xyze(0, 2) = 0.897515;
    tri3_xyze(1, 2) = 0.39778;
    tri3_xyze(2, 2) = 0.538333;
    nids.clear();
    nids.push_back(268);
    nids.push_back(284);
    nids.push_back(287);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.884221;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.574284;
    tri3_xyze(0, 1) = 0.868899;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.512832;
    tri3_xyze(0, 2) = 0.897515;
    tri3_xyze(1, 2) = 0.39778;
    tri3_xyze(2, 2) = 0.538333;
    nids.clear();
    nids.push_back(284);
    nids.push_back(286);
    nids.push_back(287);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.868899;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.512832;
    tri3_xyze(0, 1) = 0.910809;
    tri3_xyze(1, 1) = 0.374621;
    tri3_xyze(2, 1) = 0.502382;
    tri3_xyze(0, 2) = 0.897515;
    tri3_xyze(1, 2) = 0.39778;
    tri3_xyze(2, 2) = 0.538333;
    nids.clear();
    nids.push_back(286);
    nids.push_back(274);
    nids.push_back(287);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.910809;
    tri3_xyze(1, 0) = 0.374621;
    tri3_xyze(2, 0) = 0.502382;
    tri3_xyze(0, 1) = 0.868899;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.512832;
    tri3_xyze(0, 2) = 0.882193;
    tri3_xyze(1, 2) = 0.39778;
    tri3_xyze(2, 2) = 0.476881;
    nids.clear();
    nids.push_back(274);
    nids.push_back(286);
    nids.push_back(296);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.868899;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.512832;
    tri3_xyze(0, 1) = 0.853577;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.451379;
    tri3_xyze(0, 2) = 0.882193;
    tri3_xyze(1, 2) = 0.39778;
    tri3_xyze(2, 2) = 0.476881;
    nids.clear();
    nids.push_back(286);
    nids.push_back(295);
    nids.push_back(296);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.884221;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.574284;
    tri3_xyze(0, 1) = 0.899542;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.635736;
    tri3_xyze(0, 2) = 0.870926;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.610234;
    nids.clear();
    nids.push_back(284);
    nids.push_back(283);
    nids.push_back(306);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.84231;
    tri3_xyze(1, 0) = 0.467259;
    tri3_xyze(2, 0) = 0.584733;
    tri3_xyze(0, 1) = 0.884221;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.574284;
    tri3_xyze(0, 2) = 0.870926;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.610234;
    nids.clear();
    nids.push_back(305);
    nids.push_back(284);
    nids.push_back(306);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.868899;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.512832;
    tri3_xyze(0, 1) = 0.884221;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.574284;
    tri3_xyze(0, 2) = 0.855605;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.548782;
    nids.clear();
    nids.push_back(286);
    nids.push_back(284);
    nids.push_back(308);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.884221;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.574284;
    tri3_xyze(0, 1) = 0.84231;
    tri3_xyze(1, 1) = 0.467259;
    tri3_xyze(2, 1) = 0.584733;
    tri3_xyze(0, 2) = 0.855605;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.548782;
    nids.clear();
    nids.push_back(284);
    nids.push_back(305);
    nids.push_back(308);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.84231;
    tri3_xyze(1, 0) = 0.467259;
    tri3_xyze(2, 0) = 0.584733;
    tri3_xyze(0, 1) = 0.826989;
    tri3_xyze(1, 1) = 0.467259;
    tri3_xyze(2, 1) = 0.523281;
    tri3_xyze(0, 2) = 0.855605;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.548782;
    nids.clear();
    nids.push_back(305);
    nids.push_back(307);
    nids.push_back(308);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.826989;
    tri3_xyze(1, 0) = 0.467259;
    tri3_xyze(2, 0) = 0.523281;
    tri3_xyze(0, 1) = 0.868899;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.512832;
    tri3_xyze(0, 2) = 0.855605;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.548782;
    nids.clear();
    nids.push_back(307);
    nids.push_back(286);
    nids.push_back(308);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.853577;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.451379;
    tri3_xyze(0, 1) = 0.868899;
    tri3_xyze(1, 1) = 0.42094;
    tri3_xyze(2, 1) = 0.512832;
    tri3_xyze(0, 2) = 0.840283;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.48733;
    nids.clear();
    nids.push_back(295);
    nids.push_back(286);
    nids.push_back(312);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.868899;
    tri3_xyze(1, 0) = 0.42094;
    tri3_xyze(2, 0) = 0.512832;
    tri3_xyze(0, 1) = 0.826989;
    tri3_xyze(1, 1) = 0.467259;
    tri3_xyze(2, 1) = 0.523281;
    tri3_xyze(0, 2) = 0.840283;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.48733;
    nids.clear();
    nids.push_back(286);
    nids.push_back(307);
    nids.push_back(312);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.826989;
    tri3_xyze(1, 0) = 0.467259;
    tri3_xyze(2, 0) = 0.523281;
    tri3_xyze(0, 1) = 0.811667;
    tri3_xyze(1, 1) = 0.467259;
    tri3_xyze(2, 1) = 0.461829;
    tri3_xyze(0, 2) = 0.840283;
    tri3_xyze(1, 2) = 0.444099;
    tri3_xyze(2, 2) = 0.48733;
    nids.clear();
    nids.push_back(307);
    nids.push_back(311);
    nids.push_back(312);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.826989;
    tri3_xyze(1, 0) = 0.467259;
    tri3_xyze(2, 0) = 0.523281;
    tri3_xyze(0, 1) = 0.84231;
    tri3_xyze(1, 1) = 0.467259;
    tri3_xyze(2, 1) = 0.584733;
    tri3_xyze(0, 2) = 0.824247;
    tri3_xyze(1, 2) = 0.495935;
    tri3_xyze(2, 2) = 0.556601;
    nids.clear();
    nids.push_back(307);
    nids.push_back(305);
    nids.push_back(319);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    tri3_xyze(0, 0) = 0.84231;
    tri3_xyze(1, 0) = 0.467259;
    tri3_xyze(2, 0) = 0.584733;
    tri3_xyze(0, 1) = 0.821504;
    tri3_xyze(1, 1) = 0.524611;
    tri3_xyze(2, 1) = 0.58992;
    tri3_xyze(0, 2) = 0.824247;
    tri3_xyze(1, 2) = 0.495935;
    tri3_xyze(2, 2) = 0.556601;
    nids.clear();
    nids.push_back(305);
    nids.push_back(315);
    nids.push_back(319);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

  hex8_xyze(0, 0) = 0.833333;
  hex8_xyze(1, 0) = 0.416667;
  hex8_xyze(2, 0) = 0.583333;
  hex8_xyze(0, 1) = 0.833333;
  hex8_xyze(1, 1) = 0.416667;
  hex8_xyze(2, 1) = 0.5;
  hex8_xyze(0, 2) = 0.833333;
  hex8_xyze(1, 2) = 0.5;
  hex8_xyze(2, 2) = 0.5;
  hex8_xyze(0, 3) = 0.833333;
  hex8_xyze(1, 3) = 0.5;
  hex8_xyze(2, 3) = 0.583333;
  hex8_xyze(0, 4) = 0.916667;
  hex8_xyze(1, 4) = 0.416667;
  hex8_xyze(2, 4) = 0.583333;
  hex8_xyze(0, 5) = 0.916667;
  hex8_xyze(1, 5) = 0.416667;
  hex8_xyze(2, 5) = 0.5;
  hex8_xyze(0, 6) = 0.916667;
  hex8_xyze(1, 6) = 0.5;
  hex8_xyze(2, 6) = 0.5;
  hex8_xyze(0, 7) = 0.916667;
  hex8_xyze(1, 7) = 0.5;
  hex8_xyze(2, 7) = 0.583333;

  nids.clear();
  for (int i = 0; i < 8; ++i) nids.push_back(i);

  intersection.add_element(1, nids, hex8_xyze, Core::FE::CellType::hex8);
  intersection.cut_test_cut(true, Cut::VCellGaussPts_DirectDivergence);
}
