// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

//
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_fem_general_largerotations.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <numbers>
#include <optional>


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



  /// tests the plastic predictor construction, and the interpolation procedures associated with it,
  /// within the predictor interpolator used for the adaptive estimate interpolation
  TEST_F(InelasticDefgradFactorsServiceTest, TestPredictorInterpolatorPlasticPred)
  {
    // construct predictor interpolator with a single Gauss point
    ViscoplastUtils::PredictorInterpolator pred_interpolator{};
    const unsigned int gp = 0;


    // setup adaptive estimate interpolation parameters
    ViscoplastUtils::AdaptiveEstimateInterpolationHardeningParams hardening_params{
        .method = ViscoplastUtils::AdaptiveEstimateInterpolationHardeningMethod::use_previous,
        .allow_integration_failure = false,
        .failure_relative_yield_stress_deviation = 0.0,
        .max_iter_integration = 0,
        .tol_integration = 0.0,
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


    // auxiliaries
    Core::LinAlg::Matrix<3, 3> unit_3x3{Core::LinAlg::Initialization::zero};
    unit_3x3(0, 0) = unit_3x3(1, 1) = unit_3x3(2, 2) = 1.0;

    // setup previous inelastic defgrad: unit tensor
    Core::LinAlg::Matrix<3, 3> last_inv_inelastic_defgrad{unit_3x3};


    // setup deformation tensor components to be used subsequently
    Core::LinAlg::Matrix<3, 3> lambda{
        Core::LinAlg::Initialization::zero};  // eigenvalue matrix \f$ \mathbf{Lambda} \f$
    Core::LinAlg::Matrix<3, 3> Q{
        Core::LinAlg::Initialization::zero};  // eigenvector rotation \f$ \mathbf{Q} \f$
    Core::LinAlg::Matrix<3, 3> R{
        Core::LinAlg::Initialization::zero};  // rotation \f$ \mathbf{R} \f$
    Core::LinAlg::Matrix<3, 3> ref_rotation{
        Core::LinAlg::Initialization::zero};  // reference rotation to test for: either \f$
                                              // \mathbf{Q}_{\mathrm{ref}} \f$ or \f$
                                              // \mathbf{R}_{\mathrm{ref}} \f$
    Core::LinAlg::Matrix<3, 3> defgrad{
        Core::LinAlg::Initialization::zero};  // full deformation gradient \f$ \mathbf{F} =
                                              // \mathbf{R} \mathbf{Q}^T \mathbf{\Lambda}
                                              // \mathbf{Q}\f$
    Core::LinAlg::Matrix<3, 3> ref_defgrad{
        Core::LinAlg::Initialization::zero};  // reference: full deformation gradient to test for
                                              // \f$ \mathbf{F}_{\text{ref}} \f$

    auto get_rotation_matrix_from_rot_angle_around_z = [](const double angle)
    {
      Core::LinAlg::Matrix<4, 1> rot_quat{Core::LinAlg::Initialization::zero};
      rot_quat(2) = std::sin(0.5 * angle);
      rot_quat(3) = std::cos(0.5 * angle);
      Core::LinAlg::Matrix<3, 3> rot_matrix{Core::LinAlg::Initialization::zero};
      Core::LargeRotations::quaterniontotriad(rot_quat, rot_matrix);
      return rot_matrix;
    };
    auto compute_full_defgrad = [](const Core::LinAlg::Matrix<3, 3>& R,
                                    const Core::LinAlg::Matrix<3, 3>& Q,
                                    const Core::LinAlg::Matrix<3, 3>& lambda)
    {
      Core::LinAlg::Matrix<3, 3> LQ{Core::LinAlg::Initialization::zero};
      LQ.multiply(1.0, lambda, Q, 0.0);
      Core::LinAlg::Matrix<3, 3> QTLQ{Core::LinAlg::Initialization::zero};
      QTLQ.multiply_tn(1.0, Q, LQ, 0.0);
      Core::LinAlg::Matrix<3, 3> defgrad{Core::LinAlg::Initialization::zero};
      defgrad.multiply(1.0, R, QTLQ, 0.0);
      return defgrad;
    };



    // setup eigenvalues of the deformation gradient to be used within all subsequent tests, and
    // already scale them for the plastic predictor
    lambda.scale(0.0);
    lambda(0, 0) = 2.0;
    lambda(1, 1) = 1.0;
    lambda(2, 2) = 1.0;
    Core::LinAlg::Matrix<3, 3> scaled_unit{unit_3x3};
    scaled_unit.scale(std::pow(lambda.determinant(), 1.0 / 3.0));

    // --> first: test the construction and interpolation procedure for a diagonal deformation
    // gradient

    // setup deformation gradient
    Q = get_rotation_matrix_from_rot_angle_around_z(0.0);
    FOUR_C_EXPECT_NEAR(Q, unit_3x3, 1.0e-15);
    R = get_rotation_matrix_from_rot_angle_around_z(0.0);
    FOUR_C_EXPECT_NEAR(R, unit_3x3, 1.0e-15);
    defgrad = compute_full_defgrad(R, Q, lambda);
    FOUR_C_EXPECT_NEAR(defgrad, lambda, 1.0e-15);

    // check the elastic predictor
    ViscoplastUtils::AdaptiveEstimateInterpolationDeformationTensors aei_deftensors(
        defgrad, last_inv_inelastic_defgrad);
    FOUR_C_EXPECT_NEAR(aei_deftensors.elastic_predictor_elastic_defgrad, defgrad, 1.0e-15);

    // construct preliminary plastic predictor
    pred_interpolator.construct_prelim_plastic_pred(gp, aei_deftensors, aei_params);

    // verify whether both predictors are initialized consistently
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 0.0),
        aei_deftensors.elastic_predictor_elastic_defgrad, 1.0e-15);
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 1.0),
        compute_full_defgrad(R, Q, scaled_unit), 1.0e-15);

    // now set the plastic predictor at the interpolation location 0.5 between the elastic and the
    // preliminary plastic predictors
    Core::LinAlg::Matrix<3, 3> lambda_plastic_pred_ref{Core::LinAlg::Initialization::zero};
    lambda_plastic_pred_ref(0, 0) = 1.5874010519681996;
    lambda_plastic_pred_ref(1, 1) = 1.122462048309373;
    lambda_plastic_pred_ref(2, 2) = 1.122462048309373;
    pred_interpolator.set_plastic_predictor_after_construction_algo(gp, 0.5);
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 1.0),
        compute_full_defgrad(R, Q, lambda_plastic_pred_ref), 1.0e-8);

    // --> repeat the procedure above with a deformation gradient additionally containing an
    // eigenvector rotation of 45deg about the z-axis

    // setup deformation gradient
    const double angle_Q = std::numbers::pi / 4.0;
    Q = get_rotation_matrix_from_rot_angle_around_z(angle_Q);
    ref_rotation.scale(0.0);
    ref_rotation(0, 0) = ref_rotation(1, 1) = ref_rotation(1, 0) = 0.5 * std::numbers::sqrt2;
    ref_rotation(0, 1) = -0.5 * std::numbers::sqrt2;
    ref_rotation(2, 2) = 1.0;
    FOUR_C_EXPECT_NEAR(Q, ref_rotation, 1.0e-15);
    R = get_rotation_matrix_from_rot_angle_around_z(0.0);
    FOUR_C_EXPECT_NEAR(R, unit_3x3, 1.0e-15);
    defgrad = compute_full_defgrad(R, Q, lambda);
    ref_defgrad.scale(0.0);
    ref_defgrad(0, 0) = ref_defgrad(1, 1) = 1.5;
    ref_defgrad(0, 1) = ref_defgrad(1, 0) = -0.5;
    ref_defgrad(2, 2) = 1.0;
    FOUR_C_EXPECT_NEAR(defgrad, ref_defgrad, 1.0e-15);

    // check elastic predictor
    aei_deftensors = ViscoplastUtils::AdaptiveEstimateInterpolationDeformationTensors(
        defgrad, last_inv_inelastic_defgrad);
    FOUR_C_EXPECT_NEAR(aei_deftensors.elastic_predictor_elastic_defgrad, defgrad, 1.0e-15);

    // construct preliminary plastic predictor
    pred_interpolator.construct_prelim_plastic_pred(gp, aei_deftensors, aei_params);

    // verify whether both predictors are initialized consistently
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 0.0),
        aei_deftensors.elastic_predictor_elastic_defgrad, 1.0e-15);
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 1.0),
        compute_full_defgrad(R, Q, scaled_unit), 1.0e-15);

    // now set the plastic predictor at the interpolation location 0.5 between the elastic and the
    // preliminary plastic predictors
    pred_interpolator.set_plastic_predictor_after_construction_algo(gp, 0.5);
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 1.0),
        compute_full_defgrad(R, Q, lambda_plastic_pred_ref), 1.0e-8);


    // --> finally, repeat the procedure above with a deformation gradient additionally containing
    // an eigenvector rotation AND a rotation of 45deg about the z-axis

    // setup deformation gradient
    FOUR_C_EXPECT_NEAR(Q, ref_rotation, 1.0e-15);  // Q stays the same as above
    R = get_rotation_matrix_from_rot_angle_around_z(angle_Q);
    FOUR_C_EXPECT_NEAR(R, ref_rotation, 1.0e-15);  // Q stays the same as above
    defgrad = compute_full_defgrad(R, Q, lambda);
    ref_defgrad.scale(0.0);
    ref_defgrad(0, 0) = std::numbers::sqrt2;
    ref_defgrad(0, 1) = -std::numbers::sqrt2;
    ref_defgrad(1, 0) = ref_defgrad(1, 1) = 0.5 * std::numbers::sqrt2;
    ref_defgrad(2, 2) = 1.0;
    FOUR_C_EXPECT_NEAR(defgrad, ref_defgrad, 1.0e-15);

    // check elastic predictor
    aei_deftensors = ViscoplastUtils::AdaptiveEstimateInterpolationDeformationTensors(
        defgrad, last_inv_inelastic_defgrad);
    FOUR_C_EXPECT_NEAR(aei_deftensors.elastic_predictor_elastic_defgrad, defgrad, 1.0e-15);

    // construct preliminary plastic predictor
    pred_interpolator.construct_prelim_plastic_pred(gp, aei_deftensors, aei_params);

    // verify whether both predictors are initialized consistently
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 0.0),
        aei_deftensors.elastic_predictor_elastic_defgrad, 1.0e-15);
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 1.0),
        compute_full_defgrad(R, Q, scaled_unit), 1.0e-15);

    // now set the plastic predictor at the interpolation location 0.5 between the elastic and the
    // preliminary plastic predictors
    pred_interpolator.set_plastic_predictor_after_construction_algo(gp, 0.5);
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 1.0),
        compute_full_defgrad(R, Q, lambda_plastic_pred_ref), 1.0e-8);
  }


  /// tests the bookkeeping of iterations / intervals / re-estimations within the adaptive estimate
  /// interpolation manager
  TEST_F(InelasticDefgradFactorsServiceTest, TestAdaptiveEstimateInterpolationManagerBookkeeping)
  {
    // consider a single Gauss point
    const unsigned int gp = 0;

    // setup adaptive estimate interpolation parameters
    ViscoplastUtils::AdaptiveEstimateInterpolationHardeningParams hardening_params{
        .method = ViscoplastUtils::AdaptiveEstimateInterpolationHardeningMethod::use_previous,
        .allow_integration_failure = false,
        .failure_relative_yield_stress_deviation = 0.0,
        .max_iter_integration = 0,
        .tol_integration = 0.0,
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
    aei_manager.reset_and_construct_prelim_plastic_pred(gp, aei_deftensors);
    EXPECT_TRUE(aei_manager.is_plastic_pred_construct_possible(gp));  // 0 iterations -> true
    aei_manager.increment_num_plastic_pred_construct_iters();
    EXPECT_TRUE(aei_manager.is_plastic_pred_construct_possible(gp));  // 1 iterations -> true
    aei_manager.increment_num_plastic_pred_construct_iters();
    EXPECT_FALSE(aei_manager.is_plastic_pred_construct_possible(gp));  // 2 iterations -> false

    // test bookkeeping for estimate interpolation
    // 1. number of iterations
    aei_manager.reset_and_construct_prelim_plastic_pred(gp, aei_deftensors);

    EXPECT_TRUE(aei_manager.is_estimate_interp_possible(gp));  // 0 iterations -> true
    aei_manager.increment_num_estimate_interp_iters();
    EXPECT_TRUE(aei_manager.is_estimate_interp_possible(gp));  // 1 iterations -> true
    aei_manager.increment_num_estimate_interp_iters();
    EXPECT_FALSE(aei_manager.is_estimate_interp_possible(gp));  // 2 iterations -> false

    // 2. interval length (and interval points after adaptation)
    aei_manager.reset_and_construct_prelim_plastic_pred(gp, aei_deftensors);

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
    aei_manager.reset_and_construct_prelim_plastic_pred(gp, aei_deftensors);
    aei_manager.adapt_interpolation_interval_and_point(
        gp, ViscoplastUtils::ErrorType::under_yield_surface);
    EXPECT_EQ(aei_manager.lower_interp_bound(gp), 0.0);
    EXPECT_EQ(aei_manager.current_interp_point(gp), 0.25);
    EXPECT_EQ(aei_manager.upper_interp_bound(gp), 0.5);
    EXPECT_FALSE(aei_manager.is_estimate_interp_possible(
        gp));  // interval length = 0.5 < 0.6 (set min. parameter value)

    // test bookkeeping for re-estimations
    // 1. using the number of re-estimations
    aei_manager.reset_and_construct_prelim_plastic_pred(gp, aei_deftensors);
    EXPECT_TRUE(aei_manager.is_reestimation_possible(gp));  // 0 re-estimations -> true
    aei_manager.increment_num_reestimations();
    EXPECT_TRUE(aei_manager.is_reestimation_possible(gp));  // 1 re-estimation -> true
    aei_manager.increment_num_reestimations();
    EXPECT_FALSE(aei_manager.is_reestimation_possible(gp));  // 2 iterations -> false

    // 2. using the re-estimation disabling function
    aei_manager.reset_and_construct_prelim_plastic_pred(gp, aei_deftensors);
    EXPECT_TRUE(aei_manager.is_reestimation_possible(gp));  // 0 re-estimations -> true
    aei_manager.disable_further_reestimations();
    EXPECT_FALSE(aei_manager.is_reestimation_possible(gp));  // re-estimations disabled
  }

  /// tests the plastic predictor construction and interpolation at different locations within the
  /// adaptive estimate interpolation manager
  TEST_F(InelasticDefgradFactorsServiceTest, TestAdaptiveEstimateInterpolationManagerInterpolation)
  {
    // consider a single Gauss point
    const unsigned int gp = 0;

    // setup adaptive estimate interpolation parameters
    ViscoplastUtils::AdaptiveEstimateInterpolationHardeningParams hardening_params{
        .method = ViscoplastUtils::AdaptiveEstimateInterpolationHardeningMethod::use_previous,
        .allow_integration_failure = false,
        .failure_relative_yield_stress_deviation = 0.0,
        .max_iter_integration = 0,
        .tol_integration = 0.0,
    };
    ViscoplastUtils::AdaptiveEstimateInterpolationParams aei_params{
        .starting_point_type =
            ViscoplastUtils::AdaptiveEstimateInterpolationStartingPointType::user_set,
        .user_set_starting_point = 0.1,
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
        .interval_scanning_param = 0.5,
        .max_num_reestimations = 0,
        .min_reestimation_interval = 0.0,
        .hardening_params = hardening_params};


    // setup manager
    ViscoplastUtils::AdaptiveEstimateInterpolationManager aei_manager(aei_params);

    // setup deformation tensors (diagonal deformation gradient)
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


    // construct preliminary plastic predictor, and verify endpoints
    aei_manager.reset_and_construct_prelim_plastic_pred(gp, aei_deftensors);
    aei_manager.set_current_interp_point(gp, ViscoplastUtils::AdaptiveEstimateInterpolationManager::
                                                 CurrentInterpPointPreset::elastic_predictor);
    FOUR_C_EXPECT_NEAR(
        aei_manager.interpolate_inverse_inelastic_defgrad(gp, aei_deftensors.inv_defgrad),
        last_inv_inelastic_defgrad, 1.0e-15);

    aei_manager.set_current_interp_point(gp, ViscoplastUtils::AdaptiveEstimateInterpolationManager::
                                                 CurrentInterpPointPreset::plastic_predictor);
    Core::LinAlg::Matrix<3, 3> elastic_defgrad_plastic_pred{Core::LinAlg::Initialization::zero};
    elastic_defgrad_plastic_pred(0, 0) = elastic_defgrad_plastic_pred(1, 1) =
        elastic_defgrad_plastic_pred(2, 2) = std::pow(defgrad.determinant(), 1.0 / 3.0);
    Core::LinAlg::Matrix<3, 3> inv_inelastic_defgrad_plastic_pred_ref{
        Core::LinAlg::Initialization::zero};
    inv_inelastic_defgrad_plastic_pred_ref.multiply(
        1.0, aei_deftensors.inv_defgrad, elastic_defgrad_plastic_pred, 0.0);
    FOUR_C_EXPECT_NEAR(
        aei_manager.interpolate_inverse_inelastic_defgrad(gp, aei_deftensors.inv_defgrad),
        inv_inelastic_defgrad_plastic_pred_ref,
        1.0e-15);  // check using the saved current interpolation point
    FOUR_C_EXPECT_NEAR(
        aei_manager.get_inverse_inelastic_defgrad_plastic_pred(gp, aei_deftensors.inv_defgrad),
        inv_inelastic_defgrad_plastic_pred_ref,
        1.0e-15);  // check using the dedicated plastic predictor recovery method



    // construct the plastic predictor between the elastic predictor and the preliminary plastic
    // predictor (here: exactly in the middle based on the set interval scanning parameter \f$
    // k_{\text{scan}} = 0.5 \f$), and repeat the checks
    aei_manager.set_current_interp_point(gp,
        ViscoplastUtils::AdaptiveEstimateInterpolationManager::CurrentInterpPointPreset::
            standard);  // right in the middle of the elastic predictor and the preliminary plastic
                        // predictor
    aei_manager.set_plastic_predictor_after_construction_algo(gp);

    aei_manager.set_current_interp_point(gp, ViscoplastUtils::AdaptiveEstimateInterpolationManager::
                                                 CurrentInterpPointPreset::elastic_predictor);
    FOUR_C_EXPECT_NEAR(
        aei_manager.interpolate_inverse_inelastic_defgrad(gp, aei_deftensors.inv_defgrad),
        last_inv_inelastic_defgrad, 1.0e-15);

    aei_manager.set_current_interp_point(gp, ViscoplastUtils::AdaptiveEstimateInterpolationManager::
                                                 CurrentInterpPointPreset::plastic_predictor);
    elastic_defgrad_plastic_pred.scale(0.0);
    elastic_defgrad_plastic_pred(0, 0) = 1.5874010519681996;
    elastic_defgrad_plastic_pred(1, 1) = 1.122462048309373;
    elastic_defgrad_plastic_pred(2, 2) = 1.122462048309373;
    inv_inelastic_defgrad_plastic_pred_ref.multiply(
        1.0, aei_deftensors.inv_defgrad, elastic_defgrad_plastic_pred, 0.0);
    FOUR_C_EXPECT_NEAR(
        aei_manager.interpolate_inverse_inelastic_defgrad(gp, aei_deftensors.inv_defgrad),
        inv_inelastic_defgrad_plastic_pred_ref,
        1.0e-15);  // check using the saved current interpolation point
    FOUR_C_EXPECT_NEAR(
        aei_manager.get_inverse_inelastic_defgrad_plastic_pred(gp, aei_deftensors.inv_defgrad),
        inv_inelastic_defgrad_plastic_pred_ref,
        1.0e-15);  // check using the dedicated plastic predictor recovery method


    // --> test further setter options for the current interpolation point
    Core::LinAlg::Matrix<3, 3> interp_elastic_defgrad_ref{Core::LinAlg::Initialization::zero};
    Core::LinAlg::Matrix<3, 3> interp_inv_inelastic_defgrad_ref{Core::LinAlg::Initialization::zero};

    // standard preset: specified with the interval scanning parameter internally, based on the
    // current interpolation bounds (here: \f$ \xi =  0.5 \f$, pristine bounds \f$ \xi_{\text{E}} =
    // 0.0 \f$ and \f$ \xi_{\text{P}} = 1.0 \f$)
    aei_manager.set_current_interp_point(gp,
        ViscoplastUtils::AdaptiveEstimateInterpolationManager::CurrentInterpPointPreset::standard);
    elastic_defgrad_plastic_pred(0, 0) = 1.7817974362806785;
    elastic_defgrad_plastic_pred(1, 1) = 1.0594630943592953;
    elastic_defgrad_plastic_pred(2, 2) = 1.0594630943592953;
    inv_inelastic_defgrad_plastic_pred_ref.multiply(
        1.0, aei_deftensors.inv_defgrad, elastic_defgrad_plastic_pred, 0.0);
    FOUR_C_EXPECT_NEAR(
        aei_manager.interpolate_inverse_inelastic_defgrad(gp, aei_deftensors.inv_defgrad),
        inv_inelastic_defgrad_plastic_pred_ref,
        1.0e-15);  // check using the saved current interpolation point

    // user-set starting point (here: \f$ \xi = 0.1 \f$)
    aei_manager.set_starting_point(gp, std::nullopt);
    aei_manager.set_current_interp_point(gp, ViscoplastUtils::AdaptiveEstimateInterpolationManager::
                                                 CurrentInterpPointPreset::starting_point);
    elastic_defgrad_plastic_pred(0, 0) = 1.9543199368684918;
    elastic_defgrad_plastic_pred(1, 1) = 1.0116194403019225;
    elastic_defgrad_plastic_pred(2, 2) = 1.0116194403019225;
    inv_inelastic_defgrad_plastic_pred_ref.multiply(
        1.0, aei_deftensors.inv_defgrad, elastic_defgrad_plastic_pred, 0.0);
    FOUR_C_EXPECT_NEAR(
        aei_manager.interpolate_inverse_inelastic_defgrad(gp, aei_deftensors.inv_defgrad),
        inv_inelastic_defgrad_plastic_pred_ref,
        1.0e-15);  // check using the saved current interpolation point

    // intermediate point between the lower bound (here \f$ \xi_{\text{E}} = 0.0 \f$), and the
    // current interpolation point (here (here \f$ \xi = 0.1 \f$) because of the user-set starting
    // point) --> here: \f$ \xi_{\text{I}} = 0.05 \f$
    aei_manager.set_current_interp_point(gp, ViscoplastUtils::AdaptiveEstimateInterpolationManager::
                                                 CurrentInterpPointPreset::intermediate_point);
    elastic_defgrad_plastic_pred(0, 0) = 1.9770280407057923;
    elastic_defgrad_plastic_pred(1, 1) = 1.0057929410678534;
    elastic_defgrad_plastic_pred(2, 2) = 1.0057929410678534;
    inv_inelastic_defgrad_plastic_pred_ref.multiply(
        1.0, aei_deftensors.inv_defgrad, elastic_defgrad_plastic_pred, 0.0);
    FOUR_C_EXPECT_NEAR(
        aei_manager.interpolate_inverse_inelastic_defgrad(gp, aei_deftensors.inv_defgrad),
        inv_inelastic_defgrad_plastic_pred_ref,
        1.0e-15);  // check using the saved current interpolation point


    // we now set the lower bound as the current interpolation point, i.e., \f$ \xi_{\text{E}} =
    // \xi_{\text{I}} = 0.05 \f$; and redo the standard preset interpolation, \f$ \xi = 0.5 \left(
    // 0.05 + 1.0 \right) = 0.525 \f$
    aei_manager.set_lower_interp_bound_to_current_interp_point(gp);
    aei_manager.set_current_interp_point(gp,
        ViscoplastUtils::AdaptiveEstimateInterpolationManager::CurrentInterpPointPreset::standard);
    elastic_defgrad_plastic_pred(0, 0) = 1.7715350382047212;
    elastic_defgrad_plastic_pred(1, 1) = 1.0625273666151527;
    elastic_defgrad_plastic_pred(2, 2) = 1.0625273666151527;
    inv_inelastic_defgrad_plastic_pred_ref.multiply(
        1.0, aei_deftensors.inv_defgrad, elastic_defgrad_plastic_pred, 0.0);
    FOUR_C_EXPECT_NEAR(
        aei_manager.interpolate_inverse_inelastic_defgrad(gp, aei_deftensors.inv_defgrad),
        inv_inelastic_defgrad_plastic_pred_ref,
        1.0e-15);  // check using the saved current interpolation point
  }


  /// tests the update routine for the starting point of the adaptive estimate interpolation manager
  TEST_F(InelasticDefgradFactorsServiceTest, TestAdaptiveEstimateInterpolationManagerStartingPoints)
  {
    // consider a single Gauss point
    const unsigned int gp = 0;

    // function: setup plastic predictor for a given aei manager, and set its current interpolation
    // point <- starting point
    auto construct_plastic_predictor_and_set_starting_point =
        [](ViscoplastUtils::AdaptiveEstimateInterpolationManager& input_aei_manager)
    {
      // setup deformation tensors (diagonal deformation gradient)
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

      // reset all interpolation points (also sets the starting point) and construct plastic
      // predictor
      input_aei_manager.reset_and_construct_prelim_plastic_pred(gp, aei_deftensors);
    };

    // setup dummy hardening parameters
    ViscoplastUtils::AdaptiveEstimateInterpolationHardeningParams hardening_params{
        .method = ViscoplastUtils::AdaptiveEstimateInterpolationHardeningMethod::use_previous,
        .allow_integration_failure = false,
        .failure_relative_yield_stress_deviation = 0.0,
        .max_iter_integration = 0,
        .tol_integration = 0.0,
    };

    // set adaptive estimate interpolation manager with user-set starting point
    ViscoplastUtils::AdaptiveEstimateInterpolationParams aei_params_user_set{
        .starting_point_type =
            ViscoplastUtils::AdaptiveEstimateInterpolationStartingPointType::user_set,
        .user_set_starting_point = 0.1,
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
        .interval_scanning_param = 0.5,
        .max_num_reestimations = 0,
        .min_reestimation_interval = 0.0,
        .hardening_params = hardening_params};
    ViscoplastUtils::AdaptiveEstimateInterpolationManager aei_manager_user_set(aei_params_user_set);
    construct_plastic_predictor_and_set_starting_point(aei_manager_user_set);
    EXPECT_EQ(aei_manager_user_set.current_interp_point(gp), 0.1);  // user set starting point
    // setting up the starting point should not be possible; setting it is handled internally!
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(aei_manager_user_set.set_starting_point(gp, 1.0),
        Core::Exception,
        "The starting point should not be set by value for the starting point type");
    aei_manager_user_set.set_starting_point(gp, std::nullopt);  // starting point: 0.1
    EXPECT_EQ(aei_manager_user_set.starting_point(gp), 0.1);


    // set adaptive estimate interpolation manager with starting point type: last interpolation
    // point
    ViscoplastUtils::AdaptiveEstimateInterpolationParams aei_params_last{
        .starting_point_type = ViscoplastUtils::AdaptiveEstimateInterpolationStartingPointType::
            last_interpolation_point,
        .user_set_starting_point = 0.1,
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
        .interval_scanning_param = 0.5,
        .max_num_reestimations = 0,
        .min_reestimation_interval = 0.0,
        .hardening_params = hardening_params};
    ViscoplastUtils::AdaptiveEstimateInterpolationManager aei_manager_last(aei_params_last);
    construct_plastic_predictor_and_set_starting_point(aei_manager_last);
    EXPECT_EQ(aei_manager_last.current_interp_point(gp),
        0.5);  // upon initialization, there is no "last" interpolation point yet!
    // setting up the starting point should not be possible; setting it is handled internally!
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(aei_manager_last.set_starting_point(gp, 1.0), Core::Exception,
        "The starting point should not be set by value for the starting point type");
    aei_manager_last.set_starting_point(gp, std::nullopt);  // starting point: 0.5
    construct_plastic_predictor_and_set_starting_point(aei_manager_last);
    EXPECT_EQ(aei_manager_last.starting_point(gp), 0.5);
    EXPECT_EQ(aei_manager_last.current_interp_point(gp), 0.5);

    // set adaptive estimate interpolation manager with a starting point based on the optimal
    // equivalent stress
    ViscoplastUtils::AdaptiveEstimateInterpolationParams aei_params_optimal_equiv_stress{
        .starting_point_type =
            ViscoplastUtils::AdaptiveEstimateInterpolationStartingPointType::optimal_equiv_stress,
        .user_set_starting_point = 0.1,
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
        .interval_scanning_param = 0.5,
        .max_num_reestimations = 0,
        .min_reestimation_interval = 0.0,
        .hardening_params = hardening_params};
    ViscoplastUtils::AdaptiveEstimateInterpolationManager aei_manager_optimal_equiv_stress(
        aei_params_optimal_equiv_stress);
    construct_plastic_predictor_and_set_starting_point(aei_manager_optimal_equiv_stress);
    EXPECT_EQ(aei_manager_optimal_equiv_stress.starting_point(gp), 0.5);
    EXPECT_EQ(aei_manager_optimal_equiv_stress.current_interp_point(gp), 0.5);
    // value must be provided
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        aei_manager_optimal_equiv_stress.set_starting_point(gp, std::nullopt), Core::Exception,
        "No value has been provided for the starting point value for the starting point");
    // value must not exceed bounds
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(aei_manager_optimal_equiv_stress.set_starting_point(gp, -1.0),
        Core::Exception, "Interpolation is restricted to the interval [0.0, 1.0]!");
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(aei_manager_optimal_equiv_stress.set_starting_point(gp, 2.0),
        Core::Exception, "Interpolation is restricted to the interval [0.0, 1.0]!");
    // set starting point by value, and see whether this has also translated to the current
    // interpolation point
    aei_manager_optimal_equiv_stress.set_starting_point(gp, 0.75);
    construct_plastic_predictor_and_set_starting_point(aei_manager_optimal_equiv_stress);
    EXPECT_EQ(aei_manager_optimal_equiv_stress.starting_point(gp), 0.75);
    EXPECT_EQ(aei_manager_optimal_equiv_stress.current_interp_point(gp), 0.75);
  }



}  // namespace
