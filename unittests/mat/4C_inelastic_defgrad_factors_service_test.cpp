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

  /// tests the LocalIntegrationDeformationTensors of
  /// InelasticDefgradTransvIsotropElastViscoplast
  TEST_F(InelasticDefgradFactorsServiceTest, TestLocalIntegrationDeformationTensors)
  {
    // setup input
    Core::LinAlg::Matrix<3, 3> defgrad{Core::LinAlg::Initialization::zero};
    defgrad(0, 0) = 0.2513819028974873;
    defgrad(0, 1) = 0.957511195526664;
    defgrad(0, 2) = 0.8703229224151933;
    defgrad(1, 0) = 0.675673714544612;
    defgrad(1, 1) = 0.040444301498430923;
    defgrad(1, 2) = 0.10298502801901921;
    defgrad(2, 0) = 0.20079631315327318;
    defgrad(2, 1) = 0.6901106554801166;
    defgrad(2, 2) = 0.1769124998126297;


    Core::LinAlg::Matrix<3, 3> last_iFin{Core::LinAlg::Initialization::zero};
    last_iFin(0, 0) = 0.35530729350748047;
    last_iFin(0, 1) = 0.5829896953147952;
    last_iFin(0, 2) = 0.9336918888091672;
    last_iFin(1, 0) = 0.3099852313939162;
    last_iFin(1, 1) = 0.7243285059889488;
    last_iFin(1, 2) = 0.43375156919140767;
    last_iFin(2, 0) = 0.41454463288433163;
    last_iFin(2, 1) = 0.6433884759079006;
    last_iFin(2, 2) = 0.23890433987101256;


    // reference tensors based on input
    Core::LinAlg::Matrix<3, 3> inv_defgrad_ref{Core::LinAlg::Initialization::zero};
    inv_defgrad_ref(0, 0) = -0.22190619146746082;
    inv_defgrad_ref(0, 1) = 1.4971400487822994;
    inv_defgrad_ref(0, 2) = 0.2201485775679747;
    inv_defgrad_ref(1, 0) = -0.34321290611467375;
    inv_defgrad_ref(1, 1) = -0.4523291882409134;
    inv_defgrad_ref(1, 2) = 1.951751255286342;
    inv_defgrad_ref(2, 0) = 1.590689346533634;
    inv_defgrad_ref(2, 1) = 0.06521297552376205;
    inv_defgrad_ref(2, 2) = -2.2108633434926004;


    Core::LinAlg::Matrix<3, 3> right_cg_ref{Core::LinAlg::Initialization::zero};
    right_cg_ref(0, 0) = 0.5600469890068229;
    right_cg_ref(0, 1) = 0.40659981309094395;
    right_cg_ref(0, 2) = 0.3238910865092303;
    right_cg_ref(1, 0) = 0.40659981309094395;
    right_cg_ref(1, 1) = 1.3947161478897936;
    right_cg_ref(1, 2) = 0.9595983006673772;
    right_cg_ref(2, 0) = 0.3238910865092303;
    right_cg_ref(2, 1) = 0.9595983006673772;
    right_cg_ref(2, 2) = 0.7993659378673543;


    Core::LinAlg::Matrix<3, 3> elastic_predictor_inverse_plastic_defgrad_ref{
        Core::LinAlg::Initialization::zero};
    elastic_predictor_inverse_plastic_defgrad_ref(0, 0) = 0.35530729350748047;
    elastic_predictor_inverse_plastic_defgrad_ref(0, 1) = 0.5829896953147952;
    elastic_predictor_inverse_plastic_defgrad_ref(0, 2) = 0.9336918888091672;
    elastic_predictor_inverse_plastic_defgrad_ref(1, 0) = 0.3099852313939162;
    elastic_predictor_inverse_plastic_defgrad_ref(1, 1) = 0.7243285059889488;
    elastic_predictor_inverse_plastic_defgrad_ref(1, 2) = 0.43375156919140767;
    elastic_predictor_inverse_plastic_defgrad_ref(2, 0) = 0.41454463288433163;
    elastic_predictor_inverse_plastic_defgrad_ref(2, 1) = 0.6433884759079006;
    elastic_predictor_inverse_plastic_defgrad_ref(2, 2) = 0.23890433987101256;


    Core::LinAlg::Matrix<3, 3> elastic_predictor_elastic_defgrad_ref{
        Core::LinAlg::Initialization::zero};
    elastic_predictor_elastic_defgrad_ref(0, 0) = 0.7469198494262896;
    elastic_predictor_elastic_defgrad_ref(0, 1) = 1.4000614513018017;
    elastic_predictor_elastic_defgrad_ref(0, 2) = 0.8579591505610411;
    elastic_predictor_elastic_defgrad_ref(1, 0) = 0.2953008256002754;
    elastic_predictor_elastic_defgrad_ref(1, 1) = 0.48946515367319354;
    elastic_predictor_elastic_defgrad_ref(1, 2) = 0.6730174161271413;
    elastic_predictor_elastic_defgrad_ref(2, 0) = 0.3586066330866571;
    elastic_predictor_elastic_defgrad_ref(2, 1) = 0.7307524651000326;
    elastic_predictor_elastic_defgrad_ref(2, 2) = 0.5290836326068751;

    // initialize LocalIntegrationDeformationTensors and perform checks for the saved quantities
    ViscoplastUtils::LocalIntegrationDeformationTensors deftensors(defgrad, last_iFin);
    FOUR_C_EXPECT_NEAR(deftensors.defgrad, defgrad, 1.0e-15);
    FOUR_C_EXPECT_NEAR(deftensors.inv_defgrad, inv_defgrad_ref, 1.0e-15);
    FOUR_C_EXPECT_NEAR(deftensors.right_cg, right_cg_ref, 1.0e-15);
    FOUR_C_EXPECT_NEAR(deftensors.elastic_predictor_elastic_defgrad,
        elastic_predictor_elastic_defgrad_ref, 1.0e-15);
    FOUR_C_EXPECT_NEAR(deftensors.elastic_predictor_inverse_plastic_defgrad,
        elastic_predictor_inverse_plastic_defgrad_ref, 1.0e-15);
  }


  /// tests the bookkeeping of iterations within the LocalNewtonManager of
  /// InelasticDefgradTransvIsotropElastViscoplast
  TEST_F(InelasticDefgradFactorsServiceTest, TestLocalNewtonManagerIterBookkeeping)
  {
    auto local_newton_params =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams{
            .res_tol = 1.0e-8,
            .incr_tol = 1.0e-8,
            .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
            .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop,
            .max_iter = 5,
            .max_exceedance_fact_res_tol = 1.0e1,
            .max_exceedance_fact_incr_tol = 1.0e1,

        };
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager local_newton_manager(
        local_newton_params);

    EXPECT_EQ(local_newton_manager.iter(), 1);
    EXPECT_EQ(local_newton_manager.curr_num_iters().size(), 1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);

    local_newton_manager.resize(3);
    EXPECT_EQ(local_newton_manager.curr_num_iters().size(), 3);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[2], 0);

    Core::LinAlg::Matrix<10, 1> one_10x1{Core::LinAlg::Initialization::zero};
    for (unsigned int i = 0; i < 10; ++i) one_10x1(i) = 1.0;

    local_newton_manager.init_local_newton(one_10x1, true);
    local_newton_manager.increment_solution_vector_and_iter(one_10x1);
    local_newton_manager.increment_solution_vector_and_iter(one_10x1);
    local_newton_manager.increment_solution_vector_and_iter(one_10x1);
    EXPECT_EQ(local_newton_manager.iter(), 4);
    local_newton_manager.update_after_local_newton(1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 4);

    local_newton_manager.init_local_newton(one_10x1, false);
    local_newton_manager.increment_solution_vector_and_iter(one_10x1);
    EXPECT_EQ(local_newton_manager.iter(), 5);
    local_newton_manager.update_after_local_newton(1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 9);

    local_newton_manager.reset_curr_num_iters(0);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[0], 0);
    local_newton_manager.reset_curr_num_iters(1);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[1], 0);
    local_newton_manager.reset_curr_num_iters(2);
    EXPECT_EQ(local_newton_manager.curr_num_iters()[2], 0);


    // test whether the maximum number of iterations was exceeded
    EXPECT_FALSE(local_newton_manager.is_max_iter_exceeded());
    local_newton_manager.increment_solution_vector_and_iter(one_10x1);
    EXPECT_TRUE(local_newton_manager.is_max_iter_exceeded());
  }


  /// tests the basic functionality of the LocalNewtonManager (initialization, incrementation,
  /// convergence and "stuckness" verification) used within
  /// InelasticDefgradTransvIsotropElastViscoplast
  TEST_F(InelasticDefgradFactorsServiceTest, TestLocalNewtonManagerBasicFunctionality)
  {
    auto local_newton_params =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams{
            .res_tol = 1.0e-8,
            .incr_tol = 1.0e-8,
            .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
            .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop,
            .max_iter = 100,
            .max_exceedance_fact_res_tol = 0.0,
            .max_exceedance_fact_incr_tol = 0.0,
        };
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager local_newton_manager(
        local_newton_params);

    // auxiliaries
    Core::LinAlg::Matrix<10, 1> one_10x1{Core::LinAlg::Initialization::zero};
    for (unsigned int i = 0; i < 10; ++i) one_10x1(i) = 1.0;


    // --> test initialization with and without iteration counter reset

    // with iteration counter reset
    local_newton_manager.init_local_newton(one_10x1, true);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), one_10x1, 1.0e-15);
    EXPECT_EQ(local_newton_manager.iter(), 1);

    // without iteration counter reset
    local_newton_manager.increment_solution_vector_and_iter(
        one_10x1);  // increment the iteration counter
    local_newton_manager.init_local_newton(one_10x1, false);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), one_10x1, 1.0e-15);
    EXPECT_EQ(local_newton_manager.iter(), 2);


    // --> test workflow within the Local Newton: increment the solution vector (save the
    // increment), then set the residual norm, and perform the convergence check
    Core::LinAlg::Matrix<10, 1> vector_under_tol{
        Core::LinAlg::Initialization::zero};  // the 2-norm of this vector is smaller than the set
                                              // value for residual and increment tolerance
    vector_under_tol(0) = 1.0e-9;
    Core::LinAlg::Matrix<10, 1> vector_over_tol{Core::LinAlg::Initialization::zero};
    vector_over_tol(0) = 1.0e-7;  // the 2-norm of this vector is smaller than the set value for
                                  // residual and increment tolerance

    Core::LinAlg::Matrix<10, 1> updated_sol_ref(
        Core::LinAlg::Initialization::zero);  // reference: updated solution vector, used for the
                                              // solution vector checks

    // try out increment and residual vector exceeding the tolerance: no convergence!
    local_newton_manager.init_local_newton(one_10x1, true);
    local_newton_manager.increment_solution_vector_and_iter(vector_over_tol);
    updated_sol_ref.update(1.0, one_10x1, 1.0, vector_over_tol, 0.0);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm,
        vector_over_tol(0) / updated_sol_ref.norm2());
    EXPECT_EQ(local_newton_manager.iter(), 2);
    local_newton_manager.set_residual_norm(vector_over_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_over_tol(0));
    EXPECT_FALSE(local_newton_manager.is_local_newton_converged());

    // try out increment exceeding the tolerance, and residual vector under the tolerance: no
    // convergence!
    local_newton_manager.increment_solution_vector_and_iter(vector_over_tol);
    updated_sol_ref.update(1.0, vector_over_tol, 1.0);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm,
        vector_over_tol(0) / updated_sol_ref.norm2());
    EXPECT_EQ(local_newton_manager.iter(), 3);
    local_newton_manager.set_residual_norm(vector_under_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_under_tol(0));
    EXPECT_FALSE(local_newton_manager.is_local_newton_converged());

    // now the other way around: no convergence!
    local_newton_manager.increment_solution_vector_and_iter(vector_under_tol);
    updated_sol_ref.update(1.0, vector_under_tol, 1.0);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm,
        vector_under_tol(0) / updated_sol_ref.norm2());
    EXPECT_EQ(local_newton_manager.iter(), 4);
    local_newton_manager.set_residual_norm(vector_over_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_over_tol(0));
    EXPECT_FALSE(local_newton_manager.is_local_newton_converged());

    // now, both increment and residual are under the tolerance: convergence!
    local_newton_manager.increment_solution_vector_and_iter(vector_under_tol);
    updated_sol_ref.update(1.0, vector_under_tol, 1.0);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm,
        vector_under_tol(0) / updated_sol_ref.norm2());
    EXPECT_EQ(local_newton_manager.iter(), 5);
    local_newton_manager.set_residual_norm(vector_under_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_under_tol(0));
    EXPECT_TRUE(local_newton_manager.is_local_newton_converged());

    // --> test whether the Local Newton becomes stuck: increment is exactly 0.0, but the residual
    // is still over the set tolerance
    EXPECT_FALSE(local_newton_manager.is_local_newton_stuck());  // for the previous settings, the
                                                                 // Local Newton should not be stuck
    Core::LinAlg::Matrix<10, 1> zero_10x1{Core::LinAlg::Initialization::zero};
    local_newton_manager.increment_solution_vector_and_iter(zero_10x1);
    FOUR_C_EXPECT_NEAR(local_newton_manager.sol(), updated_sol_ref, 1.0e-15);
    EXPECT_EQ(local_newton_manager.convergence_quantities().increment_norm, 0.0);
    EXPECT_EQ(local_newton_manager.iter(), 6);
    local_newton_manager.set_residual_norm(vector_over_tol);
    EXPECT_EQ(local_newton_manager.convergence_quantities().residual_norm, vector_over_tol(0));
    EXPECT_TRUE(local_newton_manager.is_local_newton_stuck());
  }


  /// tests the convergence of the LocalNewtonManager (for various settings) used within
  /// InelasticDefgradTransvIsotropElastViscoplast
  TEST_F(InelasticDefgradFactorsServiceTest, TestLocalNewtonManagerConvergenceVerification)
  {
    // framework for setting up multiple LocalNewtonManager objects with varied parameters
    auto local_newton_base_params =
        Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonParams{
            .res_tol = 1.0e-8,
            .incr_tol = 1.0e-10,
            .conv_check = ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio,
            .diver_cont = ViscoplastUtils::LocalNewtonDiverCont::stop,
            .max_iter = 100,
            .max_exceedance_fact_res_tol = 0.0,
            .max_exceedance_fact_incr_tol = 0.0,
        };
    auto set_up_local_newton_manager = [local_newton_base_params](
                                           const ViscoplastUtils::LocalNewtonConvCheck conv_check)
    {
      Core::LinAlg::Matrix<10, 1> one_10x1{Core::LinAlg::Initialization::zero};
      for (unsigned int i = 0; i < 10; ++i) one_10x1(i) = 1.0;

      auto manager = ViscoplastUtils::LocalNewtonManager({
          .res_tol = local_newton_base_params.res_tol,
          .incr_tol = local_newton_base_params.incr_tol,
          .conv_check = conv_check,  // override
          .diver_cont = local_newton_base_params.diver_cont,
          .max_iter = local_newton_base_params.max_iter,
          .max_exceedance_fact_res_tol = local_newton_base_params.max_exceedance_fact_res_tol,
          .max_exceedance_fact_incr_tol = local_newton_base_params.max_exceedance_fact_incr_tol,
      });
      manager.init_local_newton(one_10x1, true);

      return manager;
    };

    // setup several Local Newton managers
    ViscoplastUtils::LocalNewtonManager manager_res_and_incr = set_up_local_newton_manager(
        ViscoplastUtils::LocalNewtonConvCheck::residual_and_increment_ratio);
    ViscoplastUtils::LocalNewtonManager manager_res =
        set_up_local_newton_manager(ViscoplastUtils::LocalNewtonConvCheck::residual);
    ViscoplastUtils::LocalNewtonManager manager_incr =
        set_up_local_newton_manager(ViscoplastUtils::LocalNewtonConvCheck::increment_ratio);

    // setup vectors (residual / increment) to be used for convergence checks
    auto vector_from_tol = [](const double first_value)
    {
      Core::LinAlg::Matrix<10, 1> out{
          Core::LinAlg::Initialization::zero};  // the 2-norm of this vector is smaller than the set
                                                // value for residual and increment tolerance
      out(0) = first_value;

      return out;
    };

    // setup numerical values smaller than, or exceeding the set tolerances
    const double exceeds_incr_tol{1.0e-9};
    const double exceeds_res_tol{1.0e-7};
    const double smaller_than_incr_tol{1.0e-10};
    const double smaller_than_res_tol{1.0e-9};


    // try out residual and increment exceeding the set tolerances
    manager_res_and_incr.increment_solution_vector_and_iter(vector_from_tol(exceeds_incr_tol));
    manager_res.increment_solution_vector_and_iter(vector_from_tol(exceeds_incr_tol));
    manager_incr.increment_solution_vector_and_iter(vector_from_tol(exceeds_incr_tol));

    manager_res_and_incr.set_residual_norm(vector_from_tol(exceeds_res_tol));
    manager_res.set_residual_norm(vector_from_tol(exceeds_res_tol));
    manager_incr.set_residual_norm(vector_from_tol(exceeds_res_tol));

    EXPECT_FALSE(manager_res_and_incr.is_local_newton_converged());
    EXPECT_FALSE(manager_res.is_local_newton_converged());
    EXPECT_FALSE(manager_incr.is_local_newton_converged());

    // try out residual smaller than the set tolerance, with increment exceeding its set tolerance
    manager_res_and_incr.increment_solution_vector_and_iter(vector_from_tol(exceeds_incr_tol));
    manager_res.increment_solution_vector_and_iter(vector_from_tol(exceeds_incr_tol));
    manager_incr.increment_solution_vector_and_iter(vector_from_tol(exceeds_incr_tol));

    manager_res_and_incr.set_residual_norm(vector_from_tol(smaller_than_res_tol));
    manager_res.set_residual_norm(vector_from_tol(smaller_than_res_tol));
    manager_incr.set_residual_norm(vector_from_tol(smaller_than_res_tol));

    EXPECT_FALSE(manager_res_and_incr.is_local_newton_converged());
    EXPECT_TRUE(manager_res.is_local_newton_converged());
    EXPECT_FALSE(manager_incr.is_local_newton_converged());

    // try out residual exceeding its set tolerance, with increment smaller than its set tolerance
    manager_res_and_incr.increment_solution_vector_and_iter(vector_from_tol(smaller_than_incr_tol));
    manager_res.increment_solution_vector_and_iter(vector_from_tol(smaller_than_incr_tol));
    manager_incr.increment_solution_vector_and_iter(vector_from_tol(smaller_than_incr_tol));

    manager_res_and_incr.set_residual_norm(vector_from_tol(exceeds_res_tol));
    manager_res.set_residual_norm(vector_from_tol(exceeds_res_tol));
    manager_incr.set_residual_norm(vector_from_tol(exceeds_res_tol));

    EXPECT_FALSE(manager_res_and_incr.is_local_newton_converged());
    EXPECT_FALSE(manager_res.is_local_newton_converged());
    EXPECT_TRUE(manager_incr.is_local_newton_converged());
  }
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
        .precondition_elastic_pred = false,
        .tol_precondition_elastic_pred = 0.0,
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
    lambda.clear();
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
    ViscoplastUtils::LocalIntegrationDeformationTensors aei_deftensors(
        defgrad, last_inv_inelastic_defgrad);
    FOUR_C_EXPECT_NEAR(aei_deftensors.elastic_predictor_elastic_defgrad, defgrad, 1.0e-15);

    // construct preliminary plastic predictor
    pred_interpolator.construct_prelim_plastic_pred(
        gp, aei_deftensors.elastic_predictor_elastic_defgrad, aei_params);

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
    ref_rotation.clear();
    ref_rotation(0, 0) = ref_rotation(1, 1) = ref_rotation(1, 0) = 0.5 * std::numbers::sqrt2;
    ref_rotation(0, 1) = -0.5 * std::numbers::sqrt2;
    ref_rotation(2, 2) = 1.0;
    FOUR_C_EXPECT_NEAR(Q, ref_rotation, 1.0e-15);
    R = get_rotation_matrix_from_rot_angle_around_z(0.0);
    FOUR_C_EXPECT_NEAR(R, unit_3x3, 1.0e-15);
    defgrad = compute_full_defgrad(R, Q, lambda);
    ref_defgrad.clear();
    ref_defgrad(0, 0) = ref_defgrad(1, 1) = 1.5;
    ref_defgrad(0, 1) = ref_defgrad(1, 0) = -0.5;
    ref_defgrad(2, 2) = 1.0;
    FOUR_C_EXPECT_NEAR(defgrad, ref_defgrad, 1.0e-15);

    // check elastic predictor
    aei_deftensors =
        ViscoplastUtils::LocalIntegrationDeformationTensors(defgrad, last_inv_inelastic_defgrad);
    FOUR_C_EXPECT_NEAR(aei_deftensors.elastic_predictor_elastic_defgrad, defgrad, 1.0e-15);

    // construct preliminary plastic predictor
    pred_interpolator.construct_prelim_plastic_pred(
        gp, aei_deftensors.elastic_predictor_elastic_defgrad, aei_params);

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
    ref_defgrad.clear();
    ref_defgrad(0, 0) = std::numbers::sqrt2;
    ref_defgrad(0, 1) = -std::numbers::sqrt2;
    ref_defgrad(1, 0) = ref_defgrad(1, 1) = 0.5 * std::numbers::sqrt2;
    ref_defgrad(2, 2) = 1.0;
    FOUR_C_EXPECT_NEAR(defgrad, ref_defgrad, 1.0e-15);

    // check elastic predictor
    aei_deftensors =
        ViscoplastUtils::LocalIntegrationDeformationTensors(defgrad, last_inv_inelastic_defgrad);
    FOUR_C_EXPECT_NEAR(aei_deftensors.elastic_predictor_elastic_defgrad, defgrad, 1.0e-15);

    // construct preliminary plastic predictor
    pred_interpolator.construct_prelim_plastic_pred(
        gp, aei_deftensors.elastic_predictor_elastic_defgrad, aei_params);

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


  /// tests the preconditioning procedure for the elastic deformation gradient within the predictor
  /// interpolator used for the adaptive estimate interpolation
  TEST_F(InelasticDefgradFactorsServiceTest, TestPredictorInterpolatorPreconditioning)
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
    ViscoplastUtils::AdaptiveEstimateInterpolationParams aei_params_no_precondition{
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
        .precondition_elastic_pred = false,
        .tol_precondition_elastic_pred = 0.0,
        .hardening_params = hardening_params};
    ViscoplastUtils::AdaptiveEstimateInterpolationParams aei_params_precondition{
        .starting_point_type = aei_params_no_precondition.starting_point_type,
        .user_set_starting_point = aei_params_no_precondition.user_set_starting_point,
        .plastic_pred_elastic_stretch_eigenval_type =
            aei_params_no_precondition.plastic_pred_elastic_stretch_eigenval_type,
        .plastic_pred_elastic_stretch_eigenvect_type =
            aei_params_no_precondition.plastic_pred_elastic_stretch_eigenvect_type,
        .plastic_pred_elastic_rotation_type =
            aei_params_no_precondition.plastic_pred_elastic_rotation_type,
        .max_num_plastic_pred_construct_iters =
            aei_params_no_precondition.max_num_plastic_pred_construct_iters,
        .max_relative_yield_stress_deviation =
            aei_params_no_precondition.max_relative_yield_stress_deviation,
        .max_num_estimate_interp_iters = aei_params_no_precondition.max_num_estimate_interp_iters,
        .min_interp_interval = aei_params_no_precondition.min_interp_interval,
        .interval_scanning_param = aei_params_no_precondition.interval_scanning_param,
        .max_num_reestimations = aei_params_no_precondition.max_num_reestimations,
        .min_reestimation_interval = aei_params_no_precondition.min_reestimation_interval,
        .precondition_elastic_pred = true,
        .tol_precondition_elastic_pred = 1.0e-13,
        .hardening_params = aei_params_no_precondition.hardening_params};

    // auxiliaries
    Core::LinAlg::Matrix<3, 3> unit_3x3{Core::LinAlg::Initialization::zero};
    unit_3x3(0, 0) = unit_3x3(1, 1) = unit_3x3(2, 2) = 1.0;

    // setup previous inelastic defgrad, and deformation gradient
    Core::LinAlg::Matrix<3, 3> last_inv_inelastic_defgrad{unit_3x3};
    Core::LinAlg::Matrix<3, 3> defgrad{Core::LinAlg::Initialization::zero};
    defgrad(0, 0) = 2.0;
    defgrad(1, 1) = defgrad(2, 2) = 1.0;
    defgrad(0, 1) = defgrad(1, 0) = 1.0e-14;


    // check the elastic predictor
    ViscoplastUtils::LocalIntegrationDeformationTensors aei_deftensors(
        defgrad, last_inv_inelastic_defgrad);
    FOUR_C_EXPECT_NEAR(aei_deftensors.elastic_predictor_elastic_defgrad, defgrad, 1.0e-15);

    // construct preliminary plastic predictor and verify interpolated matrix at point 0.0 (=elastic
    // predictor)
    pred_interpolator.construct_prelim_plastic_pred(
        gp, aei_deftensors.elastic_predictor_elastic_defgrad, aei_params_no_precondition);
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 0.0),
        aei_deftensors.elastic_predictor_elastic_defgrad, 1.0e-15);
    Core::LinAlg::Matrix<3, 3> preconditioned_elastic_defgrad_elastic_predictor{
        aei_deftensors.elastic_predictor_elastic_defgrad};

    pred_interpolator.construct_prelim_plastic_pred(
        gp, aei_deftensors.elastic_predictor_elastic_defgrad, aei_params_precondition);
    preconditioned_elastic_defgrad_elastic_predictor(0, 1) =
        preconditioned_elastic_defgrad_elastic_predictor(1, 0) = 0.0;
    FOUR_C_EXPECT_NEAR(pred_interpolator.interpolate_elastic_defgrad(gp, 0.0),
        preconditioned_elastic_defgrad_elastic_predictor, 1.0e-15);
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
        .precondition_elastic_pred = false,
        .tol_precondition_elastic_pred = 0.0,
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
    ViscoplastUtils::LocalIntegrationDeformationTensors aei_deftensors(
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
        .precondition_elastic_pred = true,
        .tol_precondition_elastic_pred = 1.0e-13,
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
    ViscoplastUtils::LocalIntegrationDeformationTensors aei_deftensors(
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
    elastic_defgrad_plastic_pred.clear();
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
      ViscoplastUtils::LocalIntegrationDeformationTensors aei_deftensors(
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
        .precondition_elastic_pred = false,
        .tol_precondition_elastic_pred = 0.0,
        .hardening_params = hardening_params};
    ViscoplastUtils::AdaptiveEstimateInterpolationManager aei_manager_user_set(aei_params_user_set);
    construct_plastic_predictor_and_set_starting_point(aei_manager_user_set);
    EXPECT_EQ(aei_manager_user_set.current_interp_point(gp), 0.1);  // user set starting point
    // setting up the starting point should not be possible; setting it is handled internally!
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        aei_manager_user_set.set_starting_point(
            gp, ViscoplastUtils::OptimalEquivStressStartingPointInput{.equiv_stress_solution = 0.0,
                    .equiv_stress_elast_pred = 0.0,
                    .equiv_stress_plast_pred = 0.0}),
        Core::Exception,
        "The starting point should not be set using the optimal equivalent stress input");
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
        .precondition_elastic_pred = false,
        .tol_precondition_elastic_pred = 0.0,
        .hardening_params = hardening_params};
    ViscoplastUtils::AdaptiveEstimateInterpolationManager aei_manager_last(aei_params_last);
    construct_plastic_predictor_and_set_starting_point(aei_manager_last);
    EXPECT_EQ(aei_manager_last.current_interp_point(gp),
        0.5);  // upon initialization, there is no "last" interpolation point yet!
    // setting up the starting point should not be possible; setting it is handled internally!
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        aei_manager_last.set_starting_point(
            gp, ViscoplastUtils::OptimalEquivStressStartingPointInput{.equiv_stress_solution = 0.0,
                    .equiv_stress_elast_pred = 0.0,
                    .equiv_stress_plast_pred = 0.0}),
        Core::Exception,
        "The starting point should not be set using the optimal equivalent stress input");
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
        .precondition_elastic_pred = false,
        .tol_precondition_elastic_pred = 0.0,
        .hardening_params = hardening_params};
    ViscoplastUtils::AdaptiveEstimateInterpolationManager aei_manager_optimal_equiv_stress(
        aei_params_optimal_equiv_stress);
    construct_plastic_predictor_and_set_starting_point(aei_manager_optimal_equiv_stress);
    EXPECT_EQ(aei_manager_optimal_equiv_stress.starting_point(gp), 0.5);
    EXPECT_EQ(aei_manager_optimal_equiv_stress.current_interp_point(gp), 0.5);
    // value must be provided
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        aei_manager_optimal_equiv_stress.set_starting_point(gp, std::nullopt), Core::Exception,
        "No input has been provided for calculating the optimal interpolation point based on the "
        "equivalent stress!");
    // set starting point by value, and see whether this has also translated to the current
    // interpolation point
    aei_manager_optimal_equiv_stress.set_starting_point(
        gp, ViscoplastUtils::OptimalEquivStressStartingPointInput{.equiv_stress_solution = 2.5,
                .equiv_stress_elast_pred = 10.0,
                .equiv_stress_plast_pred = 0.0});
    construct_plastic_predictor_and_set_starting_point(aei_manager_optimal_equiv_stress);
    EXPECT_EQ(aei_manager_optimal_equiv_stress.starting_point(gp), 0.75);
    EXPECT_EQ(aei_manager_optimal_equiv_stress.current_interp_point(gp), 0.75);
  }


}  // namespace
