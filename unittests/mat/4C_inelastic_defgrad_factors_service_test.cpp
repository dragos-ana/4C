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



  /// tests the plastic predictor construction (and the interpolation procedures associated with it)
  /// within the predictor interpolator for the adaptive estimate interpolation
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
        .max_iter_integration = 0,
        .tol_integration = 1.0e-16,
    };
    ViscoplastUtils::AdaptiveEstimateInterpolationParams aei_params{
        .starting_point_type =
            ViscoplastUtils::AdaptiveEstimateInterpolationStartingPointType::user_set,
        .user_set_starting_point = 0.0,
        .plastic_pred_elastic_stretch_eigenval_type =
            ViscoplastUtils::PlasticPredictorElasticStretchEigenvalType::scale_unit,
        .plastic_pred_elastic_stretch_eigenvect_type =
            ViscoplastUtils::PlasticPredictorElasticStretchEigenvectType::from_elastic_predictor,
        .plastic_pred_elastic_rotation_type =
            ViscoplastUtils::PlasticPredictorElasticRotationType::from_elastic_predictor,
        .max_num_plastic_pred_construct_iters = 0,
        .max_relative_yield_stress_deviation = 0.0,
        .max_num_estimate_interp_iters = 0,
        .min_interp_interval = 0.0,
        .interval_scanning_param = 0.0,
        .max_num_reestimations = 0,
        .min_reestimation_interval = 0.0,
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

    // now set the plastic predictor at the interpolation location 0.5 between the elastic and the
    // preliminary plastic predictors
    Core::LinAlg::Matrix<3, 3> plastic_pred_ref{Core::LinAlg::Initialization::zero};
    plastic_pred_ref(0, 0) = 1.5874010519681996;
    plastic_pred_ref(1, 1) = 1.122462048309373;
    plastic_pred_ref(2, 2) = 1.122462048309373;
    pred_interpolator.set_plastic_predictor_after_construction_algo(gp, 0.5);
    FOUR_C_EXPECT_NEAR(
        pred_interpolator.interpolate_elastic_defgrad(gp, 1.0), plastic_pred_ref, 1.0e-8);
  }


  /// tests the bookkeeping of the adaptive estimate interpolation manager
  TEST_F(InelasticDefgradFactorsServiceTest, TestAdaptiveEstimateInterpolationManagerBookkeeping)
  {
    // consider a single Gauss point
    const unsigned int gp = 0;

    // setup adaptive estimate interpolation parameters
    ViscoplastUtils::AdaptiveEstimateInterpolationHardeningParams hardening_params{
        .method = ViscoplastUtils::AdaptiveEstimateInterpolationHardeningMethod::use_previous,
        .allow_integration_failure = false,
        .failure_rel_yield_stress_deviation = 1.0e-16,
        .max_iter_integration = 0,
        .tol_integration = 1.0e-16,
    };
    ViscoplastUtils::AdaptiveEstimateInterpolationParams aei_params{
        .starting_point_type =
            ViscoplastUtils::AdaptiveEstimateInterpolationStartingPointType::user_set,
        .user_set_starting_point = 0.5,
        .plastic_pred_elastic_stretch_eigenval_type =
            ViscoplastUtils::PlasticPredictorElasticStretchEigenvalType::scale_unit,
        .plastic_pred_elastic_stretch_eigenvect_type =
            ViscoplastUtils::PlasticPredictorElasticStretchEigenvectType::from_elastic_predictor,
        .plastic_pred_elastic_rotation_type =
            ViscoplastUtils::PlasticPredictorElasticRotationType::from_elastic_predictor,
        .max_num_plastic_pred_construct_iters = 1,
        .max_relative_yield_stress_deviation = 0.0,
        .max_num_estimate_interp_iters = 1,
        .min_interp_interval = 0.6,
        .interval_scanning_param = 0.5,
        .max_num_reestimations = 1,
        .min_reestimation_interval = 1.0e-5,
        .hardening_params = hardening_params};


    // setup manager
    ViscoplastUtils::AdaptiveEstimateInterpolationManager aei_manager(aei_params);


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

    // test bookkeeping for plastic predictor construction
    aei_manager.reset_and_construct_plastic_pred(gp, aei_deftensors);
    EXPECT_TRUE(aei_manager.is_plastic_pred_construct_possible(gp));  // 0 iterations -> true
    aei_manager.increment_num_plastic_pred_construct_iters();
    EXPECT_TRUE(aei_manager.is_plastic_pred_construct_possible(gp));  // 1 iterations -> true
    aei_manager.increment_num_plastic_pred_construct_iters();
    EXPECT_FALSE(aei_manager.is_plastic_pred_construct_possible(gp));  // 2 iterations -> false

    // test bookkeeping for estimate interpolation
    // 1. number of iterations
    aei_manager.reset_and_construct_plastic_pred(gp, aei_deftensors);

    EXPECT_TRUE(aei_manager.is_estimate_interp_possible(gp));  // 0 iterations -> true
    aei_manager.increment_num_estimate_interp_iters();
    EXPECT_TRUE(aei_manager.is_estimate_interp_possible(gp));  // 1 iterations -> true
    aei_manager.increment_num_estimate_interp_iters();
    EXPECT_FALSE(aei_manager.is_estimate_interp_possible(gp));  // 2 iterations -> false

    // 2. interval length (and interval points after adaptation)
    aei_manager.reset_and_construct_plastic_pred(gp, aei_deftensors);

    EXPECT_EQ(aei_manager.lower_interp_bound(gp), 0.0);
    EXPECT_EQ(aei_manager.current_interp_point(gp), 0.5);  // user-set starting point
    EXPECT_EQ(aei_manager.upper_interp_bound(gp), 1.0);
    EXPECT_TRUE(aei_manager.is_estimate_interp_possible(gp));  // interval length = 1.0

    aei_manager.adapt_interpolation_interval_and_point(
        gp, ViscoplastUtils::ErrorType::overflow_error);
    EXPECT_EQ(aei_manager.lower_interp_bound(gp), 0.5);
    EXPECT_EQ(aei_manager.current_interp_point(gp), 0.75);
    EXPECT_EQ(aei_manager.upper_interp_bound(gp), 1.0);
    EXPECT_FALSE(aei_manager.is_estimate_interp_possible(
        gp));  // interval length = 0.5 < 0.6 (set min. parameter value)

    // 2. ... same test but with under_yield_surface error
    aei_manager.reset_and_construct_plastic_pred(gp, aei_deftensors);
    aei_manager.adapt_interpolation_interval_and_point(
        gp, ViscoplastUtils::ErrorType::under_yield_surface);
    EXPECT_EQ(aei_manager.lower_interp_bound(gp), 0.0);
    EXPECT_EQ(aei_manager.current_interp_point(gp), 0.25);
    EXPECT_EQ(aei_manager.upper_interp_bound(gp), 0.5);
    EXPECT_FALSE(aei_manager.is_estimate_interp_possible(
        gp));  // interval length = 0.5 < 0.6 (set min. parameter value)

    // test bookkeeping for re-estimations
    // 1. using the number of re-estimations
    aei_manager.reset_and_construct_plastic_pred(gp, aei_deftensors);
    EXPECT_TRUE(aei_manager.is_reestimation_possible(gp));  // 0 re-estimations -> true
    aei_manager.increment_num_reestimations();
    EXPECT_TRUE(aei_manager.is_reestimation_possible(gp));  // 1 re-estimation -> true
    aei_manager.increment_num_reestimations();
    EXPECT_FALSE(aei_manager.is_reestimation_possible(gp));  // 2 iterations -> false

    // 2. using the re-estimation disabling function
    aei_manager.reset_and_construct_plastic_pred(gp, aei_deftensors);
    EXPECT_TRUE(aei_manager.is_reestimation_possible(gp));  // 0 re-estimations -> true
    aei_manager.disable_further_reestimations();
    EXPECT_FALSE(aei_manager.is_reestimation_possible(gp));  // re-estimations disabled
  }


}  // namespace
