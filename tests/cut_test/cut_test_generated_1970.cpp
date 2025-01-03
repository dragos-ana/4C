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

void test_generated_1970()
{
  Cut::MeshIntersection intersection;
  intersection.get_options().init_for_cuttests();  // use full cln
  std::vector<int> nids;

  int sidecount = 0;
  std::vector<double> lsvs(8);
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = -9.23291e-14;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = -1.49392e-13;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2785);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-123);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = -1.49392e-13;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2785);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2765);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-123);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0154125;
    tri3_xyze(1, 0) = -0.0575201;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2699);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-72);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 1) = 1.00001e-12;
    tri3_xyze(1, 1) = -0.05;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2661);
    tri3_xyze(0, 2) = 0.00708835;
    tri3_xyze(1, 2) = -0.0538414;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-72);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2719);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-82);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 1) = 0.012941;
    tri3_xyze(1, 1) = -0.0482963;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-82);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.012941;
    tri3_xyze(1, 0) = -0.0482963;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2681);
    tri3_xyze(0, 1) = 0.0154125;
    tri3_xyze(1, 1) = -0.0575201;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2699);
    tri3_xyze(0, 2) = 0.020782;
    tri3_xyze(1, 2) = -0.0501722;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-82);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = -0.121634;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2713);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2715);
    tri3_xyze(0, 2) = 0.0485458;
    tri3_xyze(1, 2) = -0.1172;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-88);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = -0.0999834;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2715);
    tri3_xyze(0, 1) = 0.0298809;
    tri3_xyze(1, 1) = -0.111517;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2695);
    tri3_xyze(0, 2) = 0.0485458;
    tri3_xyze(1, 2) = -0.1172;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-88);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = -0.0999834;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2715);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2717);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-89);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0298809;
    tri3_xyze(1, 0) = -0.111517;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2695);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2715);
    tri3_xyze(0, 2) = 0.037941;
    tri3_xyze(1, 2) = -0.0915976;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-89);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-92);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 1) = 0.025;
    tri3_xyze(1, 1) = -0.0433013;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-92);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.025;
    tri3_xyze(1, 0) = -0.0433013;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2701);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2719);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-92);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0297746;
    tri3_xyze(1, 0) = -0.0515711;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2719);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 2) = 0.0330594;
    tri3_xyze(1, 2) = -0.0430838;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-92);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2725);
    tri3_xyze(0, 2) = 0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-93);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.106066;
    tri3_xyze(1, 0) = -0.106066;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-97);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = -0.121634;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2713);
    tri3_xyze(0, 2) = 0.0876513;
    tri3_xyze(1, 2) = -0.114229;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-97);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 1) = 0.0816361;
    tri3_xyze(1, 1) = -0.0816361;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2735);
    tri3_xyze(0, 2) = 0.0772252;
    tri3_xyze(1, 2) = -0.100642;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-98);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816361;
    tri3_xyze(1, 0) = -0.0816361;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2735);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2715);
    tri3_xyze(0, 2) = 0.0772252;
    tri3_xyze(1, 2) = -0.100642;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-98);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = -0.0999834;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2715);
    tri3_xyze(0, 1) = 0.0702254;
    tri3_xyze(1, 1) = -0.121634;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2713);
    tri3_xyze(0, 2) = 0.0772252;
    tri3_xyze(1, 2) = -0.100642;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-98);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0702254;
    tri3_xyze(1, 0) = -0.121634;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2713);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 2) = 0.0772252;
    tri3_xyze(1, 2) = -0.100642;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-98);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816361;
    tri3_xyze(1, 0) = -0.0816361;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2735);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2737);
    tri3_xyze(0, 2) = 0.0603553;
    tri3_xyze(1, 2) = -0.0786566;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-99);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2737);
    tri3_xyze(0, 1) = 0.0422746;
    tri3_xyze(1, 1) = -0.0732217;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2717);
    tri3_xyze(0, 2) = 0.0603553;
    tri3_xyze(1, 2) = -0.0786566;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-99);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2717);
    tri3_xyze(0, 1) = 0.0577254;
    tri3_xyze(1, 1) = -0.0999834;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2715);
    tri3_xyze(0, 2) = 0.0603553;
    tri3_xyze(1, 2) = -0.0786566;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-99);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0577254;
    tri3_xyze(1, 0) = -0.0999834;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2715);
    tri3_xyze(0, 1) = 0.0816361;
    tri3_xyze(1, 1) = -0.0816361;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2735);
    tri3_xyze(0, 2) = 0.0603553;
    tri3_xyze(1, 2) = -0.0786566;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-99);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2737);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 2) = 0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-100);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 1) = 0.0297746;
    tri3_xyze(1, 1) = -0.0515711;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2719);
    tri3_xyze(0, 2) = 0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-100);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0422746;
    tri3_xyze(1, 0) = -0.0732217;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2717);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2737);
    tri3_xyze(0, 2) = 0.0434855;
    tri3_xyze(1, 2) = -0.0566714;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-100);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-101);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 1) = 0.0353553;
    tri3_xyze(1, 1) = -0.0353553;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0353553;
    tri3_xyze(1, 0) = -0.0353553;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2721);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = -0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-102);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2745);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2745);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2725);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2725);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2722);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-103);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2745);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2747);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-104);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2725);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2745);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-104);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.121634;
    tri3_xyze(1, 0) = -0.0702254;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2753);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 1) = 0.106066;
    tri3_xyze(1, 1) = -0.106066;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2731);
    tri3_xyze(0, 2) = 0.114229;
    tri3_xyze(1, 2) = -0.0876513;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-107);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.121634;
    tri3_xyze(1, 0) = -0.0702254;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2753);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2755);
    tri3_xyze(0, 2) = 0.100642;
    tri3_xyze(1, 2) = -0.0772252;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-108);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2755);
    tri3_xyze(0, 1) = 0.0816361;
    tri3_xyze(1, 1) = -0.0816361;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2735);
    tri3_xyze(0, 2) = 0.100642;
    tri3_xyze(1, 2) = -0.0772252;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-108);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816361;
    tri3_xyze(1, 0) = -0.0816361;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2735);
    tri3_xyze(0, 1) = 0.0993137;
    tri3_xyze(1, 1) = -0.0993137;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 2) = 0.100642;
    tri3_xyze(1, 2) = -0.0772252;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-108);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0993137;
    tri3_xyze(1, 0) = -0.0993137;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2733);
    tri3_xyze(0, 1) = 0.121634;
    tri3_xyze(1, 1) = -0.0702254;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2753);
    tri3_xyze(0, 2) = 0.100642;
    tri3_xyze(1, 2) = -0.0772252;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-108);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2755);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2757);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-109);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2757);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2737);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-109);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2737);
    tri3_xyze(0, 1) = 0.0816361;
    tri3_xyze(1, 1) = -0.0816361;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2735);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-109);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816361;
    tri3_xyze(1, 0) = -0.0816361;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2735);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2755);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = -0.0603553;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-109);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2757);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-110);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = -0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-110);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = -0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2739);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = -0.0597853;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2737);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-110);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = -0.0597853;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2737);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2757);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = -0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-110);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-111);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-111);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-111);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-111);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-112);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = -0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-112);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = -0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2741);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-112);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = -0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-112);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2765);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-113);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2765);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2745);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-113);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2745);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-113);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2742);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-113);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2765);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2767);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-114);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2767);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2747);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-114);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2747);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2745);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-114);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2745);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2765);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-114);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.144889;
    tri3_xyze(1, 0) = -0.0388229;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2771);
    tri3_xyze(0, 1) = 0.135665;
    tri3_xyze(1, 1) = -0.0363514;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2773);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-117);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.135665;
    tri3_xyze(1, 0) = -0.0363514;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2773);
    tri3_xyze(0, 1) = 0.121634;
    tri3_xyze(1, 1) = -0.0702254;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2753);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-117);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.129904;
    tri3_xyze(1, 0) = -0.075;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2751);
    tri3_xyze(0, 1) = 0.144889;
    tri3_xyze(1, 1) = -0.0388229;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2771);
    tri3_xyze(0, 2) = 0.133023;
    tri3_xyze(1, 2) = -0.0550999;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-117);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.135665;
    tri3_xyze(1, 0) = -0.0363514;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2773);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2775);
    tri3_xyze(0, 2) = 0.1172;
    tri3_xyze(1, 2) = -0.0485458;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-118);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2775);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2755);
    tri3_xyze(0, 2) = 0.1172;
    tri3_xyze(1, 2) = -0.0485458;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-118);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2755);
    tri3_xyze(0, 1) = 0.121634;
    tri3_xyze(1, 1) = -0.0702254;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2753);
    tri3_xyze(0, 2) = 0.1172;
    tri3_xyze(1, 2) = -0.0485458;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-118);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.121634;
    tri3_xyze(1, 0) = -0.0702254;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2753);
    tri3_xyze(0, 1) = 0.135665;
    tri3_xyze(1, 1) = -0.0363514;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2773);
    tri3_xyze(0, 2) = 0.1172;
    tri3_xyze(1, 2) = -0.0485458;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-118);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2775);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2777);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-119);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2777);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2757);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-119);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2757);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = -0.0577254;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2755);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-119);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = -0.0577254;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2755);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2775);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = -0.037941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-119);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2777);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-120);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = -0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-120);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = -0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2759);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = -0.0422746;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2757);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-120);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = -0.0422746;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2757);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2777);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = -0.0273361;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-120);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = -9.23291e-14;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-121);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = -9.23291e-14;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-121);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-121);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-121);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = 9.23291e-14;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-122);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = -0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-122);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = -0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2761);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-122);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = 9.23291e-14;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = -0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-122);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2765);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-123);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2762);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = -9.23291e-14;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-123);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = -1.49392e-13;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2785);
    tri3_xyze(0, 1) = 0.115451;
    tri3_xyze(1, 1) = -1.49392e-13;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2787);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-124);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.115451;
    tri3_xyze(1, 0) = -1.49392e-13;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2787);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2767);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-124);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2767);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2765);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-124);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2765);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = -1.49392e-13;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2785);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-124);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.15;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2791);
    tri3_xyze(0, 1) = 0.140451;
    tri3_xyze(1, 1) = 9.23291e-14;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2793);
    tri3_xyze(0, 2) = 0.142751;
    tri3_xyze(1, 2) = -0.0187936;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-127);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.140451;
    tri3_xyze(1, 0) = 9.23291e-14;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2793);
    tri3_xyze(0, 1) = 0.135665;
    tri3_xyze(1, 1) = -0.0363514;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2773);
    tri3_xyze(0, 2) = 0.142751;
    tri3_xyze(1, 2) = -0.0187936;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-127);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.135665;
    tri3_xyze(1, 0) = -0.0363514;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2773);
    tri3_xyze(0, 1) = 0.144889;
    tri3_xyze(1, 1) = -0.0388229;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2771);
    tri3_xyze(0, 2) = 0.142751;
    tri3_xyze(1, 2) = -0.0187936;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-127);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.144889;
    tri3_xyze(1, 0) = -0.0388229;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2771);
    tri3_xyze(0, 1) = 0.15;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2791);
    tri3_xyze(0, 2) = 0.142751;
    tri3_xyze(1, 2) = -0.0187936;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-127);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.140451;
    tri3_xyze(1, 0) = 9.23291e-14;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2793);
    tri3_xyze(0, 1) = 0.115451;
    tri3_xyze(1, 1) = 1.49392e-13;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2795);
    tri3_xyze(0, 2) = 0.125771;
    tri3_xyze(1, 2) = -0.0165581;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-128);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.115451;
    tri3_xyze(1, 0) = 1.49392e-13;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2795);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2775);
    tri3_xyze(0, 2) = 0.125771;
    tri3_xyze(1, 2) = -0.0165581;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-128);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2775);
    tri3_xyze(0, 1) = 0.135665;
    tri3_xyze(1, 1) = -0.0363514;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2773);
    tri3_xyze(0, 2) = 0.125771;
    tri3_xyze(1, 2) = -0.0165581;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-128);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.135665;
    tri3_xyze(1, 0) = -0.0363514;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2773);
    tri3_xyze(0, 1) = 0.140451;
    tri3_xyze(1, 1) = 9.23291e-14;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2793);
    tri3_xyze(0, 2) = 0.125771;
    tri3_xyze(1, 2) = -0.0165581;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-128);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.115451;
    tri3_xyze(1, 0) = 1.49392e-13;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2795);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = 1.49392e-13;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2797);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-129);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = 1.49392e-13;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2797);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2777);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-129);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2777);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = -0.0298809;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2775);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-129);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = -0.0298809;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2775);
    tri3_xyze(0, 1) = 0.115451;
    tri3_xyze(1, 1) = 1.49392e-13;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2795);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = -0.012941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-129);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = 1.49392e-13;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2797);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = 9.23291e-14;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-130);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = 9.23291e-14;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = -0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-130);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = -0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2779);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = -0.0218829;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2777);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-130);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = -0.0218829;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2777);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = 1.49392e-13;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2797);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = -0.00932385;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-130);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(3002);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = -9.23291e-14;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-131);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = -9.23291e-14;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-131);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3001);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.785305;
    nids.push_back(-131);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(3019);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3001);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-132);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3001);
    tri3_xyze(0, 1) = 0.05;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-132);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.05;
    tri3_xyze(1, 0) = 0;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2781);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = 9.23291e-14;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-132);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = 9.23291e-14;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(3019);
    tri3_xyze(0, 2) = 0.0538414;
    tri3_xyze(1, 2) = 0.00708835;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-132);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = 0.0218829;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(3005);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = -1.49392e-13;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2785);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-133);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = -1.49392e-13;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2785);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = -9.23291e-14;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-133);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = -9.23291e-14;
    tri3_xyze(2, 0) = 0.770611;
    nids.push_back(2782);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.770611;
    nids.push_back(3002);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.761529;
    nids.push_back(-133);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.115451;
    tri3_xyze(1, 0) = -1.49392e-13;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2787);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = -1.49392e-13;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(2785);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-134);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = -1.49392e-13;
    tri3_xyze(2, 0) = 0.752447;
    nids.push_back(2785);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = 0.0218829;
    tri3_xyze(2, 1) = 0.752447;
    nids.push_back(3005);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.752447;
    nids.push_back(-134);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.135665;
    tri3_xyze(1, 0) = 0.0363514;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(3013);
    tri3_xyze(0, 1) = 0.140451;
    tri3_xyze(1, 1) = 9.23291e-14;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2793);
    tri3_xyze(0, 2) = 0.142751;
    tri3_xyze(1, 2) = 0.0187936;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-137);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.140451;
    tri3_xyze(1, 0) = 9.23291e-14;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2793);
    tri3_xyze(0, 1) = 0.15;
    tri3_xyze(1, 1) = 0;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2791);
    tri3_xyze(0, 2) = 0.142751;
    tri3_xyze(1, 2) = 0.0187936;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-137);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = 0.0298809;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(3015);
    tri3_xyze(0, 1) = 0.115451;
    tri3_xyze(1, 1) = 1.49392e-13;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2795);
    tri3_xyze(0, 2) = 0.125771;
    tri3_xyze(1, 2) = 0.0165581;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-138);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.115451;
    tri3_xyze(1, 0) = 1.49392e-13;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2795);
    tri3_xyze(0, 1) = 0.140451;
    tri3_xyze(1, 1) = 9.23291e-14;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2793);
    tri3_xyze(0, 2) = 0.125771;
    tri3_xyze(1, 2) = 0.0165581;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-138);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.140451;
    tri3_xyze(1, 0) = 9.23291e-14;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2793);
    tri3_xyze(0, 1) = 0.135665;
    tri3_xyze(1, 1) = 0.0363514;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(3013);
    tri3_xyze(0, 2) = 0.125771;
    tri3_xyze(1, 2) = 0.0165581;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-138);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = 0.0298809;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(3015);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = 0.0218829;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(3017);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-139);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = 0.0218829;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(3017);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = 1.49392e-13;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2797);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-139);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = 1.49392e-13;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2797);
    tri3_xyze(0, 1) = 0.115451;
    tri3_xyze(1, 1) = 1.49392e-13;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2795);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-139);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.115451;
    tri3_xyze(1, 0) = 1.49392e-13;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2795);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = 0.0298809;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(3015);
    tri3_xyze(0, 2) = 0.0982963;
    tri3_xyze(1, 2) = 0.012941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-139);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = 0.0218829;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(3017);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(3019);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-140);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(3019);
    tri3_xyze(0, 1) = 0.0595492;
    tri3_xyze(1, 1) = 9.23291e-14;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-140);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0595492;
    tri3_xyze(1, 0) = 9.23291e-14;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2799);
    tri3_xyze(0, 1) = 0.0845492;
    tri3_xyze(1, 1) = 1.49392e-13;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2797);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-140);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0845492;
    tri3_xyze(1, 0) = 1.49392e-13;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2797);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = 0.0218829;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(3017);
    tri3_xyze(0, 2) = 0.0708216;
    tri3_xyze(1, 2) = 0.00932385;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-140);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2979);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = 0.0597853;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2977);
    tri3_xyze(0, 2) = 0.0434855;
    tri3_xyze(1, 2) = 0.0566714;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-220);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2981);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2999);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-222);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2999);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2979);
    tri3_xyze(0, 2) = 0.0430838;
    tri3_xyze(1, 2) = 0.0330594;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-222);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = 0.0597853;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2977);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = 0.0422746;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2997);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = 0.0603553;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-229);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = 0.0422746;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2997);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = 0.0577254;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2995);
    tri3_xyze(0, 2) = 0.0786566;
    tri3_xyze(1, 2) = 0.0603553;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-229);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0597853;
    tri3_xyze(1, 0) = 0.0597853;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2977);
    tri3_xyze(0, 1) = 0.0421076;
    tri3_xyze(1, 1) = 0.0421076;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2979);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = 0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-230);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0421076;
    tri3_xyze(1, 0) = 0.0421076;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2979);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2999);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = 0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-230);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2999);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = 0.0422746;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2997);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = 0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-230);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = 0.0422746;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2997);
    tri3_xyze(0, 1) = 0.0597853;
    tri3_xyze(1, 1) = 0.0597853;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2977);
    tri3_xyze(0, 2) = 0.0566714;
    tri3_xyze(1, 2) = 0.0434855;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-230);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2999);
    tri3_xyze(0, 1) = 0.0433013;
    tri3_xyze(1, 1) = 0.025;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(2981);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-232);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0433013;
    tri3_xyze(1, 0) = 0.025;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(2981);
    tri3_xyze(0, 1) = 0.0482963;
    tri3_xyze(1, 1) = 0.012941;
    tri3_xyze(2, 1) = 0.8;
    nids.push_back(3001);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-232);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0482963;
    tri3_xyze(1, 0) = 0.012941;
    tri3_xyze(2, 0) = 0.8;
    nids.push_back(3001);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(3019);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-232);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(3019);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2999);
    tri3_xyze(0, 2) = 0.0501722;
    tri3_xyze(1, 2) = 0.020782;
    tri3_xyze(2, 2) = 0.814695;
    nids.push_back(-232);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0999834;
    tri3_xyze(1, 0) = 0.0577254;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2995);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = 0.0422746;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2997);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = 0.037941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-239);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = 0.0422746;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2997);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = 0.0218829;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(3017);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = 0.037941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-239);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = 0.0218829;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(3017);
    tri3_xyze(0, 1) = 0.111517;
    tri3_xyze(1, 1) = 0.0298809;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(3015);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = 0.037941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-239);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.111517;
    tri3_xyze(1, 0) = 0.0298809;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(3015);
    tri3_xyze(0, 1) = 0.0999834;
    tri3_xyze(1, 1) = 0.0577254;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2995);
    tri3_xyze(0, 2) = 0.0915976;
    tri3_xyze(1, 2) = 0.037941;
    tri3_xyze(2, 2) = 0.847553;
    nids.push_back(-239);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0732217;
    tri3_xyze(1, 0) = 0.0422746;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(2997);
    tri3_xyze(0, 1) = 0.0515711;
    tri3_xyze(1, 1) = 0.0297746;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(2999);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = 0.0273361;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-240);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0515711;
    tri3_xyze(1, 0) = 0.0297746;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(2999);
    tri3_xyze(0, 1) = 0.0575201;
    tri3_xyze(1, 1) = 0.0154125;
    tri3_xyze(2, 1) = 0.829389;
    nids.push_back(3019);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = 0.0273361;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-240);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0575201;
    tri3_xyze(1, 0) = 0.0154125;
    tri3_xyze(2, 0) = 0.829389;
    nids.push_back(3019);
    tri3_xyze(0, 1) = 0.0816682;
    tri3_xyze(1, 1) = 0.0218829;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(3017);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = 0.0273361;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-240);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }
  {
    Core::LinAlg::SerialDenseMatrix tri3_xyze(3, 3);

    nids.clear();
    tri3_xyze(0, 0) = 0.0816682;
    tri3_xyze(1, 0) = 0.0218829;
    tri3_xyze(2, 0) = 0.847553;
    nids.push_back(3017);
    tri3_xyze(0, 1) = 0.0732217;
    tri3_xyze(1, 1) = 0.0422746;
    tri3_xyze(2, 1) = 0.847553;
    nids.push_back(2997);
    tri3_xyze(0, 2) = 0.0659953;
    tri3_xyze(1, 2) = 0.0273361;
    tri3_xyze(2, 2) = 0.838471;
    nids.push_back(-240);
    intersection.add_cut_side(++sidecount, nids, tri3_xyze, Core::FE::CellType::tri3);
  }



  {
    Core::LinAlg::SerialDenseMatrix hex8_xyze(3, 8);

    nids.clear();
    hex8_xyze(0, 0) = 0.1;
    hex8_xyze(1, 0) = -0.05;
    hex8_xyze(2, 0) = 0.8;
    nids.push_back(1973);
    hex8_xyze(0, 1) = 0.1;
    hex8_xyze(1, 1) = 2.08167e-18;
    hex8_xyze(2, 1) = 0.8;
    nids.push_back(1974);
    hex8_xyze(0, 2) = 0.05;
    hex8_xyze(1, 2) = -2.08167e-18;
    hex8_xyze(2, 2) = 0.8;
    nids.push_back(1985);
    hex8_xyze(0, 3) = 0.05;
    hex8_xyze(1, 3) = -0.05;
    hex8_xyze(2, 3) = 0.8;
    nids.push_back(1984);
    hex8_xyze(0, 4) = 0.1;
    hex8_xyze(1, 4) = -0.05;
    hex8_xyze(2, 4) = 0.85;
    nids.push_back(2094);
    hex8_xyze(0, 5) = 0.1;
    hex8_xyze(1, 5) = 2.42861e-18;
    hex8_xyze(2, 5) = 0.85;
    nids.push_back(2095);
    hex8_xyze(0, 6) = 0.05;
    hex8_xyze(1, 6) = -2.42861e-18;
    hex8_xyze(2, 6) = 0.85;
    nids.push_back(2106);
    hex8_xyze(0, 7) = 0.05;
    hex8_xyze(1, 7) = -0.05;
    hex8_xyze(2, 7) = 0.85;
    nids.push_back(2105);

    intersection.add_element(1970, nids, hex8_xyze, Core::FE::CellType::hex8);
  }



  intersection.cut_test_cut(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation);
  intersection.cut_finalize(
      true, Cut::VCellGaussPts_DirectDivergence, Cut::BCellGaussPts_Tessellation, false, true);
}
