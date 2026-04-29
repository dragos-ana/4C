// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_singleton_owner.hpp"



namespace
{
  using namespace FourC;

  namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

  class InelasticDefgradFactorsServiceTest : public ::testing::Test
  {
   protected:
    static constexpr int problemid_ = 0;

    void SetUp() override {}


    Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;
  };



  /// tests the plastic predictor construction within the predictor interpolator
  TEST_F(InelasticDefgradFactorsServiceTest, TestPredictorInterpolatorPlasticPred)
  {
    // construct predictor interpolator with a single Gauss point
    ViscoplastUtils::PredictorInterpolator pred_interpolator{};
    const unsigned int gp = 0;


    // setup adaptive estimate interpolation parameters
    ViscoplastUtils::AdaptiveEstimateInterpolationHardeningParams hardening_params{
        .method = ViscoplastUtils::AdaptiveEstimateInterpolationHardeningMethod::use_previous,
        .allow_integration_failure = false,
        .failure_rel_yield_stress_deviation = 1.0e-16,
        .max_iter_integration = 1,
        .tol_integration = 1.0e-16,
    };
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationParams
        aei_params{.starting_point_type =
                       ViscoplastUtils::AdaptiveEstimateInterpolationStartingPointType::user_set,
            .user_set_starting_point = 0.0,
            .plastic_pred_elastic_stretch_eigenval_type =
                ViscoplastUtils::PlasticPredictorElasticStretchEigenvalType::scale_unit,
            .plastic_pred_elastic_stretch_eigenvect_type = ViscoplastUtils::
                PlasticPredictorElasticStretchEigenvectType::from_elastic_predictor,
            .plastic_pred_elastic_rotation_type =
                ViscoplastUtils::PlasticPredictorElasticRotationType::from_elastic_predictor,
            .max_num_plastic_pred_construct_iters = 100,
            .max_relative_yield_stress_deviation = 1.0e-3,
            .max_num_estimate_interp_iters = 100,
            .min_interp_interval = 1.0e-15,
            .interval_scanning_param = 0.5,
            .max_num_reestimations = 0,
            .min_reestimation_interval = 1.0e-5,
            .hardening_params = hardening_params};


    // setup deformation tensors
    Core::LinAlg::Matrix<3, 3> defgrad{Core::LinAlg::Initialization::zero};
    defgrad(0, 0) = 2.0;
    defgrad(1, 1) = 1.0;
    defgrad(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> last_inv_inelastic_defgrad{Core::LinAlg::Initialization::zero};
    last_inv_inelastic_defgrad(0, 0) = 1.0;
    last_inv_inelastic_defgrad(1, 1) = 1.0;
    last_inv_inelastic_defgrad(2, 2) = 1.0;
    ViscoplastUtils::AdaptiveEstimateInterpolationDeformationTensors aei_deftensors(
        defgrad, last_inv_inelastic_defgrad);
    FOUR_C_EXPECT_NEAR(aei_deftensors.elastic_predictor_elastic_defgrad, defgrad, 1.e-15);

    // construct preliminary plastic predictor
    pred_interpolator.construct_prelim_plastic_pred(gp, aei_deftensors, aei_params);

    // verify whether both predictors are initialized consistently
    Core::LinAlg::Matrix<3, 3> prelim_plastic_pred_ref{last_inv_inelastic_defgrad};
    prelim_plastic_pred_ref.scale(std::pow(defgrad.determinant(), 1.0 / 3.0));
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 0.0),
        aei_deftensors.elastic_predictor_elastic_defgrad, 1.0e-15);
    FOUR_C_EXPECT_NEAR(
        pred_interpolator.interpolate_elastic_defgrad(gp, 1.0), prelim_plastic_pred_ref, 1.0e-15);
  }

}  // namespace
