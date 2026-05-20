// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_mat_inelastic_defgrad_factors_service.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_linalg_utils_quaternion_interpolation.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <vector>


            FOUR_C_NAMESPACE_OPEN

    using namespace Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;
namespace AEI =
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolation;


namespace
{

  // map error status to double for csv output
  double error_status_to_double(const ErrorType err_status)
  {
    switch (err_status)
    {
      case (ErrorType::no_errors):
        return 0.0;
      case (ErrorType::overflow_error):
        return 10.0;
      case (ErrorType::under_yield_surface):
        return 2.0;
      case (ErrorType::no_convergence_local_newton):
        return 3.0;
      case (ErrorType::failed_solution_linear_system_lnl):
        return 4.0;
      default:
        FOUR_C_THROW("Error status {} not yet enabled!", err_status);
    }
  }


  // elastic and plastic predictor locations
  constexpr double ELASTIC_PREDICTOR_LOCATION = 0.0;
  constexpr double PLASTIC_PREDICTOR_LOCATION = 1.0;
  const std::vector ELASTIC_AND_PLASTIC_PREDICTOR_LOCATIONS{
      Core::LinAlg::diagonal_matrix<1>(ELASTIC_PREDICTOR_LOCATION),
      Core::LinAlg::diagonal_matrix<1>(PLASTIC_PREDICTOR_LOCATION),
  };

  const auto UNIT_QUATERNION =
      make_matrix(Core::LinAlg::Tensor<double, 4, 1>({0.0, 0.0, 0.0, 1.0}));

  // creates the eigenvalue interpolator used for the Adaptive Estimate Interpolation
  Core::LinAlg::ScalarInterpolator<1> create_eigenvalue_interpolator()
  {
    Core::LinAlg::ScalarInterpolationType interp_type =
        Core::LinAlg::ScalarInterpolationType::logarithmic_weighted_average;
    Core::LinAlg::ScalarInterpolationWeightingFunction weight_func =
        Core::LinAlg::ScalarInterpolationWeightingFunction::inverse_distance;
    Core::LinAlg::ScalarInterpolationParams interp_params;

    return {interp_type, weight_func, interp_params};
  }
  const auto EIGENVAL_INTERPOLATOR_AEI = create_eigenvalue_interpolator();

  // given the diagonal eigenvalue tensors of the elastic deformation gradients associated with the
  // elastic and plastic predictors and an interpolation location between them, interpolate the
  // eigenvalues using the defined eigenvalue interpolator
  std::vector<double> interpolate_eigenvalues(
      const Core::LinAlg::Matrix<3, 3>& eigenvalues_elastic_predictor,
      const Core::LinAlg::Matrix<3, 3>& eigenvalues_plastic_predictor,
      const Core::LinAlg::Matrix<1, 1>& interpolation_location_1x1_matrix)
  {
    const std::vector<std::vector<double>> scalar_interp_eigenval = {
        {eigenvalues_elastic_predictor(0, 0), eigenvalues_elastic_predictor(1, 1),
            eigenvalues_elastic_predictor(2, 2)},
        {eigenvalues_plastic_predictor(0, 0), eigenvalues_plastic_predictor(1, 1),
            eigenvalues_plastic_predictor(2, 2)}};

    return EIGENVAL_INTERPOLATOR_AEI.get_interpolated_scalar(scalar_interp_eigenval,
        ELASTIC_AND_PLASTIC_PREDICTOR_LOCATIONS, interpolation_location_1x1_matrix);
  }


  // Adaptive Estimate Interpolation: compute the elastic deformation gradient, using the
  // interpolated eigenvalues and the relative rotation contributions (quaternions) with respect to
  // the elastic deformation gradient within the elastic predictor
  Core::LinAlg::Matrix<3, 3> compute_elastic_defgrad_wrt_elastic_predictor(
      const std::vector<double>& interpolated_eigenvalues,
      const Core::LinAlg::Matrix<3, 3>& eigenvector_rotation_elastic_predictor,
      const Core::LinAlg::Matrix<4, 1>& interpolated_rel_eigenvector_rotation_quaternion,
      const Core::LinAlg::Matrix<3, 3>& rotation_elastic_predictor,
      const Core::LinAlg::Matrix<4, 1>& interpolated_rel_rotation_quaternion)
  {
    Core::LinAlg::Matrix<3, 3> out{Core::LinAlg::Initialization::zero};

    // construct diagonal interpolated eigenvalue matrix
    Core::LinAlg::Matrix<3, 3> eigenval_matrix{Core::LinAlg::Initialization::zero};
    for (unsigned int i = 0; i < 3; ++i)
    {
      eigenval_matrix(i, i) = interpolated_eigenvalues[i];
    }

    // construct interpolated eigenvector matrix from its relative contribution with respect to the
    // eigenvector matrix from the elastic predictor:
    // \f$ \boldsymbol{Q}_{\boldsymbol{F}_{\mathrm{e,interp}}} =
    // \boldsymbol{Q}_{\boldsymbol{F}_{\mathrm{e}}^{(\mathrm{E})}}
    // \boldsymbol{Q}_{\boldsymbol{F}_{\mathrm{e,interp}},\mathrm{rel}} \f$
    // (the relative quaternion is transformed into an equivalent relative rotation matrix)
    Core::LinAlg::Matrix<3, 3> rel_interp_eigenvect_matrix{Core::LinAlg::Initialization::zero};
    Core::LargeRotations::quaterniontotriad(
        interpolated_rel_eigenvector_rotation_quaternion, rel_interp_eigenvect_matrix);
    Core::LinAlg::Matrix<3, 3> interp_eigenvect_matrix{Core::LinAlg::Initialization::zero};
    interp_eigenvect_matrix.multiply_nn(
        1.0, eigenvector_rotation_elastic_predictor, rel_interp_eigenvect_matrix, 0.0);


    // construct interpolated rotation matrix from its relative contribution with respect to the
    // rotation matrix from the elastic predictor:
    // \f$ \boldsymbol{R}_{\boldsymbol{F}_{\mathrm{e,interp}}} =
    // \boldsymbol{R}_{\boldsymbol{F}_{\mathrm{e}}^{(\mathrm{E})}}
    // \boldsymbol{R}_{\boldsymbol{F}_{\mathrm{e,interp}},\mathrm{rel}} \f$
    // (the relative quaternion is transformed into an equivalent relative rotation matrix)
    Core::LinAlg::Matrix<3, 3> rel_interp_rot_matrix{Core::LinAlg::Initialization::zero};
    Core::LargeRotations::quaterniontotriad(
        interpolated_rel_rotation_quaternion, rel_interp_rot_matrix);
    Core::LinAlg::Matrix<3, 3> interp_rot_matrix{Core::LinAlg::Initialization::zero};
    interp_rot_matrix.multiply_nn(1.0, rotation_elastic_predictor, rel_interp_rot_matrix, 0.0);


    // multiply contributions to construct the final tensor
    Core::LinAlg::Matrix<3, 3> LQ{Core::LinAlg::Initialization::zero};
    LQ.multiply(1.0, eigenval_matrix, interp_eigenvect_matrix, 0.0);
    Core::LinAlg::Matrix<3, 3> QTLQ{Core::LinAlg::Initialization::zero};
    QTLQ.multiply_tn(1.0, interp_eigenvect_matrix, LQ, 0.0);
    out.multiply(1.0, interp_rot_matrix, QTLQ, 0.0);

    return out;
  }


  // precondition matrix: absolute values smaller than a set tolerance are set to 0.0
  Core::LinAlg::Matrix<3, 3> precondition_matrix(
      const Core::LinAlg::Matrix<3, 3>& matrix, const double tol)
  {
    Core::LinAlg::Matrix<3, 3> out_matrix{matrix};

    for (unsigned i = 0; i < 3; ++i)
    {
      for (unsigned j = 0; j < 3; ++j)
      {
        if (std::abs(matrix(i, j)) < tol)
        {
          out_matrix(i, j) = 0.0;
        }
      }
    }
    return out_matrix;
  }


  //! calculates the starting point for the Adaptive Estimate Interpolation based on the equivalent
  //! stress of the previous solution between its both predictors
  double calculate_equiv_stress_starting_point(
      const AEI::InputEquivStressStartingPoint& input_equiv_stress_starting_point)
  {
    // set to elastic predictor if the stress of the elastic predictor is numerically 0.0 ->
    // this is theoretically
    // possible for viscoplastic laws without yield surfaces, which may exhibit plastic flow
    // even in this case; however, the determination of the starting point requires dividing
    // over this stress value, which will not be possible in this specific case.
    // Same goes for the case where the elastic predictor and the plastic predictor are
    // associated with effectively the same stress value (e.g., during stress relaxation) -> set
    // starting point as elastic predictor in these particular cases
    if (input_equiv_stress_starting_point.equiv_stress_elast_pred <= 1.0e-12 ||
        std::abs(input_equiv_stress_starting_point.equiv_stress_plast_pred -
                 input_equiv_stress_starting_point.equiv_stress_elast_pred) /
                input_equiv_stress_starting_point.equiv_stress_elast_pred <
            1.0e-8)
    {
      return ELASTIC_PREDICTOR_LOCATION;
    }


    // compute starting point based on the equivalent stress: we clamp between the elastic and
    // plastic predictors because in some special cases such as stress relaxation, the starting
    // point may be slightly out of this interval (machine precision)
    return std::clamp((input_equiv_stress_starting_point.equiv_stress_solution -
                          input_equiv_stress_starting_point.equiv_stress_elast_pred) /
                          (input_equiv_stress_starting_point.equiv_stress_plast_pred -
                              input_equiv_stress_starting_point.equiv_stress_elast_pred),
        ELASTIC_PREDICTOR_LOCATION, PLASTIC_PREDICTOR_LOCATION);
  }

  /// initialize csv timestep data writer (the csv writer can write at a single Gauss point, or over
  /// all Gauss points -> is_overall_table = true in the latter case)
  void init_timestep_data_csv_writer(const bool use_adaptive_estimate_interpolation,
      Core::IO::RuntimeCsvWriter& csv_writer, const bool is_overall_table = false)
  {
    csv_writer.register_data_vector("Eval. iterations (global)", 1, 16);
    csv_writer.register_data_vector("Total iterations (global)", 1, 16);
    csv_writer.register_data_vector("Eval. iterations (LNL)", 1, 16);
    csv_writer.register_data_vector("Total iterations (LNL)", 1, 16);
    csv_writer.register_data_vector("Eval. time (CU)", 1, 16);
    csv_writer.register_data_vector("Total time (CU)", 1, 16);
    if (use_adaptive_estimate_interpolation)
    {
      csv_writer.register_data_vector("Eval. PPC iters (AEI)", 1, 16);
      csv_writer.register_data_vector("Total PPC iters (AEI)", 1, 16);
      csv_writer.register_data_vector("Eval. interp. iters (AEI)", 1, 16);
      csv_writer.register_data_vector("Total interp. iters (AEI)", 1, 16);
      csv_writer.register_data_vector("Eval. re-estimations (AEI)", 1, 16);
      csv_writer.register_data_vector("Total re-estimations (AEI)", 1, 16);
      csv_writer.register_data_vector("Eval. start. point det. time (AEI)", 1, 16);
      csv_writer.register_data_vector("Total start. point det. time (AEI)", 1, 16);
      if (!is_overall_table)
      {
        csv_writer.register_data_vector("Interp. point equiv. stress history (AEI)", 1, 16);
        csv_writer.register_data_vector("Starting point (AEI)", 1, 16);
      }
    }
  }


  /// special struct for timestep data containing data for the adaptive estimate interpolation (csv)
  struct CsvWritingTimestepAEIData
  {
    //! adaptive estimate interpolation: number of plastic predictor construction iterations
    //! accumulated in the current timestep
    const unsigned int num_plastic_pred_construct_iters;

    //! adaptive estimate interpolation: number of plastic predictor construction iterations
    //! accumulated over all timesteps
    const unsigned int total_num_plastic_pred_construct_iters;

    //! adaptive estimate interpolation: number of estimate interpolation iterations
    //! accumulated in the current timestep
    const unsigned int num_interp_iters;

    //! adaptive estimate interpolation: number of estimate interpolation iterations
    //! accumulated over all timesteps
    const unsigned int total_num_interp_iters;

    //! adaptive estimate interpolation: number of re-estimations
    //! accumulated in the current timestep
    const unsigned int num_reestimations;

    //! adaptive estimate interpolation: number of re-estimations
    //! accumulated over all timesteps
    const unsigned int total_num_reestimations;

    //! adaptive estimate interpolation: interpolation point leading to the equivalent stress of the
    //! solution in the current timestep
    const double interp_point_equiv_stress_history;

    //! adaptive estimate interpolation: starting point in the current timestep
    const double starting_point;

    //! adaptive estimate interpolation: computation time for determining the starting point in the
    //! next timestep, accumulated over the current timestep
    const double starting_point_determination_time;

    //! adaptive estimate interpolation: computation time for determining the starting point in the
    //! next timestep, accumulated over all timesteps
    const double total_starting_point_determination_time;
  };


  /// struct used for timestep data writing (csv)
  struct CsvWritingTimestepData
  {
    //! number of global iterations accumulated in the current timestep
    unsigned int num_global_iters;

    //! number of global iterations accumulated over all timesteps
    unsigned int total_num_global_iters;

    //! number of Local Newton iterations accumulated in the current timestep
    unsigned int num_lnl_iters;

    //! number of Local Newton iterations accumulated over all timesteps
    unsigned int total_num_lnl_iters;

    //! computation time for constitutive update accumulated in the current timestep
    double constitutive_update_time;

    //! computation time for constitutive update accumulated over all timesteps
    double total_constitutive_update_time;

    //! adaptive estimate interpolation: number of plastic predictor construction iterations
    //! accumulated in the current timestep
    std::optional<CsvWritingTimestepAEIData> aei_data;
  };



  /// write timestep data to csv (the csv writer can write at a single Gauss point, or over
  /// all Gauss points -> is_overall_table = true in the latter case)
  void write_timestep_data_to_csv(Core::IO::RuntimeCsvWriter& csv_writer,
      const CsvWritingTimestepData& data,
      const Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
          TrackingSettings& tracking_settings,
      const bool is_overall_table = false)
  {
    // output data
    std::map<std::string, std::vector<double>> output_data;
    output_data["Eval. iterations (global)"] = {static_cast<double>(data.num_global_iters)};
    output_data["Total iterations (global)"] = {static_cast<double>(data.total_num_global_iters)};
    output_data["Eval. iterations (LNL)"] = {static_cast<double>(data.num_lnl_iters)};
    output_data["Total iterations (LNL)"] = {static_cast<double>(data.total_num_lnl_iters)};
    output_data["Eval. time (CU)"] = {data.constitutive_update_time};
    output_data["Total time (CU)"] = {data.total_constitutive_update_time};
    if (data.aei_data.has_value())
    {
      output_data["Eval. PPC iters (AEI)"] = {
          static_cast<double>(data.aei_data->num_plastic_pred_construct_iters)};
      output_data["Total PPC iters (AEI)"] = {
          static_cast<double>(data.aei_data->total_num_plastic_pred_construct_iters)};
      output_data["Eval. interp. iters (AEI)"] = {
          static_cast<double>(data.aei_data->num_interp_iters)};
      output_data["Total interp. iters (AEI)"] = {
          static_cast<double>(data.aei_data->total_num_interp_iters)};
      output_data["Eval. re-estimations (AEI)"] = {
          static_cast<double>(data.aei_data->num_reestimations)};
      output_data["Total re-estimations (AEI)"] = {
          static_cast<double>(data.aei_data->total_num_reestimations)};
      output_data["Eval. start. point det. time (AEI)"] = {
          data.aei_data->starting_point_determination_time};
      output_data["Total start. point det. time (AEI)"] = {
          data.aei_data->total_starting_point_determination_time};
      if (!is_overall_table)
      {
        output_data["Interp. point equiv. stress history (AEI)"] = {
            data.aei_data->interp_point_equiv_stress_history};


        output_data["Starting point (AEI)"] = {data.aei_data->starting_point};
      }
    }

    // write output data to csv
    csv_writer.write_data_to_file(tracking_settings.time, tracking_settings.timestep, output_data);
  }


  /// struct used for detailed AEI data writing (csv) --> vectors with the same size tracking the
  /// evolution of interpolation points and intervals within the adaptive estimate interpolation
  /// including re-estimation
  struct CsvWritingAEIData
  {
    //! adaptive estimate interpolation: vector tracking current interpolation points in the
    //! current timestep
    std::vector<double> aei_current_interp_points;

    //! adaptive estimate interpolation: vector tracking lower interpolation bounds in the
    //! current timestep
    std::vector<double> aei_lower_interp_bounds;

    //! adaptive estimate interpolation: vector tracking upper interpolation bounds in the
    //! current timestep
    std::vector<double> aei_upper_interp_bounds;

    //! adaptive estimate interpolation: vector tracking the global iteration in the current
    //! timestep
    std::vector<double> aei_global_iters;

    //! adaptive estimate interpolation: vector tracking the local iteration in the current
    //! timestep
    std::vector<double> aei_local_iters;

    //! verify equal lengths of the vectors
    void verify_equal_lengths() const
    {
      auto l = {aei_current_interp_points.size(), aei_lower_interp_bounds.size(),
          aei_upper_interp_bounds.size(), aei_global_iters.size(), aei_local_iters.size()};

      auto all_same =
          std::all_of(l.begin(), l.end(), [&](unsigned int v) { return v == *l.begin(); });
      FOUR_C_ASSERT_ALWAYS(all_same,
          "Your vectors for the AEI don't have equal lengths! Current interpolation points: {}, "
          "lower interpolation bounds: {}, upper interpolation bounds: {}, global iterations: "
          "{}, "
          "local iterations: {}",
          aei_current_interp_points.size(), aei_lower_interp_bounds.size(),
          aei_upper_interp_bounds.size(), aei_global_iters.size(), aei_local_iters.size());
    }
  };

  /// initialize csv adaptive estimate interpolation data writer
  void init_aei_data_csv_writer(Core::IO::RuntimeCsvWriter& csv_writer)
  {
    csv_writer.register_data_vector("Lower interpolation bound", 1, 16);
    csv_writer.register_data_vector("Current interpolation point", 1, 16);
    csv_writer.register_data_vector("Upper interpolation bound", 1, 16);
    csv_writer.register_data_vector("Global iteration", 1, 16);
    csv_writer.register_data_vector("Local iteration", 1, 16);
  }


  /// write detailed adaptive estimate interpolation data to csv
  void write_aei_data_to_csv(Core::IO::RuntimeCsvWriter& csv_writer, const CsvWritingAEIData& data,
      const Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
          TrackingSettings& tracking_settings)
  {
    data.verify_equal_lengths();
    const size_t length = data.aei_global_iters.size();

    for (unsigned int l = 0; l < length; ++l)
    {
      // output data
      std::map<std::string, std::vector<double>> output_data;

      output_data["Lower interpolation bound"] = {
          static_cast<double>(data.aei_lower_interp_bounds[l])};
      output_data["Current interpolation point"] = {
          static_cast<double>(data.aei_current_interp_points[l])};
      output_data["Upper interpolation bound"] = {
          static_cast<double>(data.aei_upper_interp_bounds[l])};
      output_data["Global iteration"] = {static_cast<double>(data.aei_global_iters[l])};
      output_data["Local iteration"] = {static_cast<double>(data.aei_local_iters[l])};

      // write output data to csv
      csv_writer.write_data_to_file(
          tracking_settings.time, tracking_settings.timestep, output_data);
    }
  }


  /// struct used for detailed Local Newton data writing (csv) --> vectors with the same size
  /// tracking the evolution of interpolation points and intervals within the adaptive estimate
  /// interpolation including re-estimation
  struct CsvWritingLNLData
  {
    //! vector tracking global iterations in the
    //! current timestep
    std::vector<unsigned int> global_iters;

    //! vector tracking Local Newton iterations in the
    //! current timestep
    std::vector<unsigned int> local_iters;

    //! vector tracking the error status in the
    //! current timestep
    std::vector<ErrorType> error_status;

    //! vector tracking the residual norms in the
    //! current timestep
    std::vector<double> residual_norms;

    //! vector tracking the increment norms in the
    //! current timestep
    std::vector<double> increment_norms;

    //! vector tracking the convergence status over the Local iterations in the
    //! current timestep
    std::vector<bool> is_converged;

    //! vector tracking the current interpolation point \f$ \xi \f$ (from the previous Adaptive
    //! Estimate Interpolation) over the Local iterations in the current timestep
    std::optional<std::vector<double>> current_interpolation_points = std::nullopt;

    //! vector tracking the equivalent stresses in the
    //! current timestep
    std::vector<double> equiv_stresses;

    //! vector tracking the plastic strains in the
    //! current timestep
    std::vector<double> plastic_strains;

    //! vector tracking the plastic strain increments in the
    //! current timestep
    std::vector<double> plastic_strain_increments;

    //! verify equal lengths of the vectors
    void verify_equal_lengths() const
    {
      auto l = {global_iters.size(), local_iters.size(), error_status.size(), residual_norms.size(),
          increment_norms.size(), is_converged.size(),
          current_interpolation_points.has_value() ? current_interpolation_points->size()
                                                   : global_iters.size(),
          equiv_stresses.size(), plastic_strains.size(), plastic_strain_increments.size()};

      auto all_same =
          std::all_of(l.begin(), l.end(), [&](unsigned int v) { return v == *l.begin(); });
      FOUR_C_ASSERT_ALWAYS(all_same,
          "Your vectors for the LNL don't have equal lengths! Global iters: {}, "
          "local iters: {}, error_status: {}, residual_norms: "
          "{}, "
          "increment_norms: {}, current_interpolation_points: {}, equiv_stresses: {}, "
          "plastic_strains: {}, "
          "plastic_strain_increments: {}",
          global_iters.size(), local_iters.size(), error_status.size(), residual_norms.size(),
          increment_norms.size(), is_converged.size(),
          current_interpolation_points.has_value() ? current_interpolation_points->size()
                                                   : global_iters.size(),
          equiv_stresses.size(), plastic_strains.size(), plastic_strain_increments.size());
    }
  };


  /// initialize csv Local Newton data writer
  void init_lnl_data_csv_writer(
      Core::IO::RuntimeCsvWriter& csv_writer, const bool use_adaptive_estimate_interpolation)
  {
    csv_writer.register_data_vector("Global iteration", 1, 16);
    csv_writer.register_data_vector("Local iteration", 1, 16);
    csv_writer.register_data_vector("Error status", 1, 16);
    csv_writer.register_data_vector("Residual norm", 1, 16);
    csv_writer.register_data_vector("Increment norm", 1, 16);
    csv_writer.register_data_vector("Is converged?", 1, 16);
    if (use_adaptive_estimate_interpolation)
      csv_writer.register_data_vector("Current interpolation point", 1, 16);
    csv_writer.register_data_vector("Equivalent stress", 1, 16);
    csv_writer.register_data_vector("Plastic strain", 1, 16);
    csv_writer.register_data_vector("Plastic strain increment", 1, 16);
  }


  /// write detailed Local Newton data to csv
  void write_lnl_data_to_csv(Core::IO::RuntimeCsvWriter& csv_writer, const CsvWritingLNLData& data,
      const Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
          TrackingSettings& tracking_settings)
  {
    data.verify_equal_lengths();
    const size_t length = data.global_iters.size();

    for (unsigned int l = 0; l < length; ++l)
    {
      // output data
      std::map<std::string, std::vector<double>> output_data;

      output_data["Global iteration"] = {static_cast<double>(data.global_iters[l])};
      output_data["Local iteration"] = {static_cast<double>(data.local_iters[l])};
      output_data["Error status"] = {error_status_to_double(data.error_status[l])};
      output_data["Residual norm"] = {static_cast<double>(data.residual_norms[l])};
      output_data["Increment norm"] = {static_cast<double>(data.increment_norms[l])};
      output_data["Is converged?"] = {static_cast<double>(data.is_converged[l])};
      if (data.current_interpolation_points.has_value())
        output_data["Current interpolation point"] = {
            static_cast<double>(data.current_interpolation_points->at(l))};
      output_data["Equivalent stress"] = {static_cast<double>(data.equiv_stresses[l])};
      output_data["Plastic strain"] = {static_cast<double>(data.plastic_strains[l])};
      output_data["Plastic strain increment"] = {
          static_cast<double>(data.plastic_strain_increments[l])};

      // write output data to csv
      csv_writer.write_data_to_file(
          tracking_settings.time, tracking_settings.timestep, output_data);
    }
  }


}  // namespace


void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ThermoMechanicalCouplingCache::resize(
    const unsigned int numgp)
{
  std::apply([numgp](auto&... quantity) { (quantity.resize(numgp), ...); }, quantities());
}

void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ThermoMechanicalCouplingCache::reset(
    const int gp)
{
  std::apply([gp](auto&... quantity) { (quantity.reset(gp), ...); }, quantities());
}


std::string
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::get_detailed_error_message_for_error_type(
    ErrorType err_type)
{
  switch (err_type)
  {
    case ErrorType::negative_plastic_strain:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: negative plastic strain!";
    case ErrorType::overflow_error:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: overflow error related to "
             "the evaluation of the plastic strain increment!";
    case ErrorType::failed_solution_linear_system_lnl:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the Local Newton Loop failed!";
    case ErrorType::no_convergence_local_newton:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Local Newton Loop did not "
             "converge for the given loop settings!";
    case ErrorType::singular_jacobian:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: singular Jacobian after "
             "converged Local Newton Loop, which does not allow for the analytical evaluation "
             "of the linearization!";
    case ErrorType::failed_solution_analytic_linearization:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: solution of the linear "
             "system in the analytical linearization failed";
    case ErrorType::failed_computation_flow_resistance:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the flow resistance for the viscoplasticity law";
    case ErrorType::failed_computation_flow_resistance_derivs:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed while computing "
             "the derivatives of the flow resistance for the viscoplasticity law";
    case ErrorType::failed_matrix_log_evaluation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
             "matrix logarithm or its derivative with respect to the argument";
    case ErrorType::failed_matrix_exp_evaluation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in evaluating the "
             "matrix exponential or its derivative with respect to the argument";
    case ErrorType::failed_right_cg_interpolation:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: Failed in interpolating "
             "the right Cauchy-Green deformation tensor";
    case ErrorType::under_yield_surface:
      return "Error in InelasticDefgradTransvIsotropElastViscoplast: we are 'under' the yield "
             "surface, sigma < sigma_yield!";
    default:
      FOUR_C_THROW("to_string(ErrorType): {}: No error message provided!", err_type);
  }
}

Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ConstNonMatTensors::ConstNonMatTensors()
{  // auxiliaries
  Core::LinAlg::Matrix<3, 3> unit3x3(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) unit3x3(i, i) = 1.0;
  Core::LinAlg::Matrix<6, 6> temp6x6(Core::LinAlg::Initialization::zero);

  // set constant non-material tensors

  // 3x3 identity
  id3x3 = unit3x3;

  // Voigt stress form of 3x3 identity
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      id3x3, id6x1);

  // symmetric identity four tensor
  Core::LinAlg::FourTensorOperations::add_kronecker_tensor_product(id4_6x6, 1.0, id3x3, id3x3, 0.0);

  // deviatoric operator
  Core::LinAlg::FourTensor<3> dev_op_four_tensor =
      Core::LinAlg::setup_deviatoric_projection_tensor<3>();
  Core::LinAlg::Voigt::setup_6x6_voigt_matrix_from_four_tensor(temp6x6, dev_op_four_tensor);
  dev_op = Core::LinAlg::Voigt::modify_voigt_representation(temp6x6, 1.0, 2.0);

  // identity four tensor
  id4_9x9.clear();
  Core::LinAlg::FourTensorOperations::add_non_symmetric_product(1.0, id3x3, id3x3, id4_9x9);

  // 10x10 identity
  id10x10.clear();
  for (int i = 0; i < 10; ++i) id10x10(i, i) = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ConstMatTensors::
    set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m)
{
  // get instance of constant non-material tensors
  const auto& const_non_mat_tensors = ConstNonMatTensors::instance();

  // set material-dependent tensors (fiber orientation)

  // structural tensor
  mm.multiply_nt(1.0, m, m, 0.0);

  // deviatoric part of the structural tensor
  double tr_mm_ = mm(0, 0) + mm(1, 1) + mm(2, 2);
  mm_dev.update(1.0, mm, -1.0 / 3.0 * tr_mm_, const_non_mat_tensors.id3x3);

  // dyadic product of structural tensors
  Core::LinAlg::Matrix<6, 1> mm_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm, mm_V);
  mm_dyad_mm.multiply_nt(1.0, mm_V, mm_V, 0.0);

  // dyadic product of deviatoric structural tensor with the structural tensor
  Core::LinAlg::Matrix<6, 1> mm_dev_V(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::VoigtUtils<Core::LinAlg::Voigt::NotationType::stress>::matrix_to_vector(
      mm_dev, mm_dev_V);
  mm_dev_dyad_mm.multiply_nt(1.0, mm_dev_V, mm_V, 0.0);

  // dyadic product of identity with the structural tensor
  id_dyad_mm.multiply_nt(1.0, const_non_mat_tensors.id6x1, mm_V, 0.0);

  // sum of identity with the structural tensor
  id_plus_mm.update(1.0, const_non_mat_tensors.id3x3, 1.0, mm, 0.0);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::reset(
    const double dt)
{
  t_ = 0.0;
  substep_counter_ = 1;
  curr_dt_ = dt;
  time_step_halving_counter_ = 0;
  total_num_of_substeps_ = 1;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::
    increment_substep()
{
  t_ += curr_dt_;
  substep_counter_++;
};

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils::halve_substep()
{
  curr_dt_ *= 1.0 / 2.0;
  time_step_halving_counter_ += 1;
  total_num_of_substeps_ += (total_num_of_substeps_ - substep_counter_ + 1);
};


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::init(
    const double ref_temperature)
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> id3x3{Core::LinAlg::Initialization::zero};
  for (unsigned int i = 0; i < 3; ++i)
  {
    id3x3(i, i) = 1.0;
  }

  // ----- set last_ and current_ variables referring to values at different time instants
  // ----- for now: the number of Gauss points is unknown -> we set the values only for 1
  // Gauss point and update the number of Gauss points in the setup method

  // default values of the inverse plastic deformation gradient: unit tensor
  last_plastic_defgrad_inverse.resize(1, id3x3);
  current_plastic_defgrad_inverse.resize(1, id3x3);  // value irrelevant at this point
  last_substep_plastic_defgrad_inverse.resize(1, id3x3);

  // update last_ and current_ values of the plastic strain
  last_plastic_strain.resize(1, 0.0);
  current_plastic_strain.resize(1, 0.0);  // value irrelevant at this point
  last_substep_plastic_strain.resize(1, 0.0);

  // update last_ and current_ values of the temperature
  last_temperature.resize(1, ref_temperature);
  current_temperature.resize(1, ref_temperature);  // value irrelevant at this point

  // update last_ and current_ values of the equivalent stress
  last_equiv_stress.resize(1, 0.0);
  current_equiv_stress.resize(1, 0.0);  // value irrelevant at this point

  // default values of the right CG tensor: unit tensor
  last_rightCG.resize(1, id3x3);
  current_rightCG.resize(1, id3x3);  // value irrelevant at this point

  // default value for the current deformation gradient: zero tensor \f$ \boldsymbol{0} f$ (to
  // make sure that the inverse inelastic deformation gradient is evaluated in the first method
  // call)
  last_defgrad.resize(1, Core::LinAlg::Matrix<3, 3>{id3x3});
  current_defgrad.resize(1, Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::resize(
    const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called,
      "You already called resize for the time step quantities! The number of current GP is {} "
      "and "
      "you attempt to set it to {}",
      last_plastic_strain.size(), numgp);


  // default values of the inverse plastic deformation gradient for ALL Gauss Points
  last_plastic_defgrad_inverse.resize(numgp, last_plastic_defgrad_inverse[0]);
  current_plastic_defgrad_inverse.resize(numgp,
      last_plastic_defgrad_inverse[0]);  // value irrelevant at this point
  last_substep_plastic_defgrad_inverse.resize(numgp, last_substep_plastic_defgrad_inverse[0]);

  // default values of the plastic strain for ALL Gauss Points
  last_plastic_strain.resize(numgp, last_plastic_strain[0]);
  current_plastic_strain.resize(numgp, last_plastic_strain[0]);  // value irrelevant at this point
  last_substep_plastic_strain.resize(numgp, last_substep_plastic_strain[0]);

  // default values of the right CG deformation tensor for ALL Gauss Points
  last_rightCG.resize(numgp, last_rightCG[0]);
  current_rightCG.resize(numgp, last_rightCG[0]);  // value irrelevant at this point

  // default values of the equivalent stress for ALL Gauss Points
  last_equiv_stress.resize(numgp, last_equiv_stress[0]);
  current_equiv_stress.resize(numgp, current_equiv_stress[0]);  // value irrelevant at this point

  // default values of the deformation gradient
  last_defgrad.resize(numgp, last_defgrad[0]);
  current_defgrad.resize(numgp, current_defgrad[0]);

  // default values of the temperature for ALL Gauss Points
  last_temperature.resize(numgp, last_temperature[0]);        // value irrelevant at this point
  current_temperature.resize(numgp, current_temperature[0]);  // value irrelevant at this point

  resize_called = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::pre_evaluate(
    const unsigned int gp)
{
  FOUR_C_ASSERT_ALWAYS(gp < last_plastic_defgrad_inverse.size(),
      "You try to pre-evaluate the time step quantities at GP {}, but the object has only {} "
      "Gauss "
      "points",
      gp, last_plastic_defgrad_inverse.size());

  // set consistent last substep values
  last_substep_plastic_defgrad_inverse[gp] = last_plastic_defgrad_inverse[gp];
  last_substep_plastic_strain[gp] = last_plastic_strain[gp];
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::update(
    const unsigned int gp)
{
  FOUR_C_ASSERT_ALWAYS(gp < last_plastic_defgrad_inverse.size(),
      "You try to update the time step quantities at GP {}, but the object has only {} Gauss "
      "points",
      gp, last_plastic_defgrad_inverse.size());

  // update history variables for the next time step
  last_defgrad[gp] = current_defgrad[gp];
  last_rightCG[gp] = current_rightCG[gp];
  last_plastic_defgrad_inverse[gp] = current_plastic_defgrad_inverse[gp];
  last_substep_plastic_defgrad_inverse[gp] = current_plastic_defgrad_inverse[gp];
  last_plastic_strain[gp] = current_plastic_strain[gp];
  last_substep_plastic_strain[gp] = current_plastic_strain[gp];
  last_equiv_stress[gp] = current_equiv_stress[gp];
  last_temperature[gp] = current_temperature[gp];
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, last_defgrad);
  add_to_pack(data, last_rightCG);
  add_to_pack(data, last_plastic_defgrad_inverse);
  add_to_pack(data, last_plastic_strain);
  add_to_pack(data, last_equiv_stress);
  add_to_pack(data, last_substep_plastic_defgrad_inverse);
  add_to_pack(data, last_substep_plastic_strain);
  add_to_pack(data, last_temperature);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  // extract last values
  extract_from_pack(buffer, last_defgrad);
  extract_from_pack(buffer, last_rightCG);
  extract_from_pack(buffer, last_plastic_defgrad_inverse);
  extract_from_pack(buffer, last_plastic_strain);
  extract_from_pack(buffer, last_equiv_stress);
  extract_from_pack(buffer, last_substep_plastic_defgrad_inverse);
  extract_from_pack(buffer, last_substep_plastic_strain);
  extract_from_pack(buffer, last_temperature);

  // fill current_ values with the last_ values
  current_rightCG.resize(last_rightCG.size(),
      last_rightCG[0]);  // value irrelevant
  current_plastic_defgrad_inverse.resize(last_plastic_defgrad_inverse.size(),
      last_plastic_defgrad_inverse[0]);  // value irrelevant
  current_plastic_strain.resize(last_plastic_strain.size(),
      last_plastic_strain[0]);  // value irrelevant
  current_equiv_stress.resize(last_equiv_stress.size(),
      last_equiv_stress[0]);  // value irrelevant
  current_temperature.resize(last_temperature.size(),
      0.0);  // value irrelevant

  // set evaluated deformation gradient to 0, to make sure that the inverse inelastic deformation
  // gradient is evaluated fully after the restart
  current_defgrad.resize(last_substep_plastic_defgrad_inverse.size(),
      Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});
}

Core::LinAlg::Matrix<1, 6>
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::compute_taylor_quinney_wrt_cauchygreen(
    const double taylor_quinney_coefficient, const ThermoMechanicalCouplingState& state,
    const ThermoMechanicalCouplingStateDerivatives& state_derivatives,
    const HistoryVariablesDerivativesWrtCauchyGreen& history_variables_derivatives)
{
  Core::LinAlg::Matrix<1, 6> total_dequiv_stress_dC =
      state_derivatives.equiv_stress_wrt_cauchy_green;
  total_dequiv_stress_dC.multiply(1.0, state_derivatives.equiv_stress_wrt_inverse_plastic_defgrad,
      history_variables_derivatives.inv_plastic_defgrad_wrt_cauchy_green, 1.0);

  Core::LinAlg::Matrix<1, 6> total_dpsr_dC{Core::LinAlg::Initialization::zero};
  total_dpsr_dC.update(state_derivatives.plastic_strain_rate_derivs.deriv_plastic_strain,
      history_variables_derivatives.plastic_strain_wrt_cauchy_green, 0.0);
  total_dpsr_dC.update(
      state_derivatives.plastic_strain_rate_derivs.deriv_equiv_stress, total_dequiv_stress_dC, 1.0);

  Core::LinAlg::Matrix<1, 6> dR_TQ_dCV{Core::LinAlg::Initialization::zero};
  dR_TQ_dCV.update(
      taylor_quinney_coefficient * state.plastic_strain_rate, total_dequiv_stress_dC, 0.0);
  dR_TQ_dCV.update(taylor_quinney_coefficient * state.equiv_stress, total_dpsr_dC, 1.0);

  return dR_TQ_dCV;
}

double
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::compute_taylor_quinney_wrt_temperature(
    const double taylor_quinney_coefficient, const ThermoMechanicalCouplingState& state,
    const ThermoMechanicalCouplingStateDerivatives& state_derivatives,
    const HistoryVariablesDerivativesWrtTemperature& history_variables_derivatives)
{
  double total_dequiv_stress_dT = state_derivatives.equiv_stress_wrt_temperature;
  for (int i = 0; i < 9; ++i)
  {
    total_dequiv_stress_dT += state_derivatives.equiv_stress_wrt_inverse_plastic_defgrad(i) *
                              history_variables_derivatives.inv_plastic_defgrad_wrt_temperature(i);
  }

  const double total_dpsr_dT =
      state_derivatives.plastic_strain_rate_derivs.deriv_temperature +
      state_derivatives.plastic_strain_rate_derivs.deriv_plastic_strain *
          history_variables_derivatives.plastic_strain_wrt_temperature +
      state_derivatives.plastic_strain_rate_derivs.deriv_equiv_stress * total_dequiv_stress_dT;

  return taylor_quinney_coefficient *
         (total_dequiv_stress_dT * state.plastic_strain_rate + state.equiv_stress * total_dpsr_dT);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::LocalNewtonManager(
    const LocalNewtonParams& lnl_params)
    : params_(lnl_params)
{
  // set number of Gauss points to 1 temporarily, since we don't
  // know it at this point in time
  curr_num_iters_.resize(1, 0);

  // set initial number of iterations
  iter_ = 0;

  // initialize solution vector and convergence quantities with dummy values; they will be set
  // anyway to more meaningful values when starting the local Newton within the material model
  sol_ = Core::LinAlg::Matrix<10, 1>(Core::LinAlg::Initialization::zero);
  convergence_quantities_.residual_norm = 0.0;
  convergence_quantities_.increment_norm = 0.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::resize(
    const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called_,
      "You already called resize for the Local Newton manager! The number of current GP is {} "
      "and "
      "you attempt to set it to {}",
      curr_num_iters_.size(), numgp);

  // resize arrays
  curr_num_iters_.resize(numgp, curr_num_iters_[0]);


  resize_called_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    update_after_local_newton(const unsigned int gp)
{
  // increment number of local Newton iterations for the current timestep at the
  // current GP
  curr_num_iters_[gp] += iter_;
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    reset_curr_num_iters(const unsigned int gp)
{
  FOUR_C_ASSERT_ALWAYS(gp < curr_num_iters_.size(),
      "You try to reset the current number of iterations within the Local Newton manager at "
      "Gauss "
      "point {}, but the object only has {} Gauss points",
      gp, curr_num_iters_.size());

  curr_num_iters_[gp] = 0;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    save_init_estimate_and_reset_convergence_quantities(
        const Core::LinAlg::Matrix<10, 1>& init_estimate)
{
  // --> set initial estimate
  sol_ = init_estimate;

  // --> set quantities used for convergence checks

  // residual norm
  convergence_quantities_.residual_norm = 0.0;
  // if the convergence check requires verifying the residual norm, we must ensure that the value
  // set here is larger than the tolerance, to perform the check at least once, in the next
  // iteration
  if (params_.conv_check == LocalNewtonConvCheck::residual ||
      params_.conv_check == LocalNewtonConvCheck::residual_and_increment_ratio)
  {
    convergence_quantities_.residual_norm = 2.0 * params_.res_tol;
  }

  // increment norm: ratio of increment to current solution
  convergence_quantities_.increment_norm = 0.0;
  // if the convergence check requires verifying the increment norm, we must ensure that the value
  // set here is larger than the tolerance, to perform the check at least once, in the next
  // iteration
  if (params_.conv_check == LocalNewtonConvCheck::increment_ratio ||
      params_.conv_check == LocalNewtonConvCheck::residual_and_increment_ratio)
  {
    convergence_quantities_.increment_norm = 2.0 * params_.incr_tol;
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    increment_solution_vector(const Core::LinAlg::Matrix<10, 1>& delta_sol)
{
  sol_.update(1.0, delta_sol, 1.0);

  const double sol_norm = sol_.norm2();
  const double delta_sol_norm = delta_sol.norm2();
  FOUR_C_ASSERT_ALWAYS(sol_norm >= 1.0e-8,
      "The solution vector in local iteration {} is nearly 0, with 2-norm: {}! Something went "
      "wrong, since such mechanical states are not expected!",
      iter_, sol_norm);
  convergence_quantities_.increment_norm = delta_sol_norm / sol_norm;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    is_local_newton_converged() const
{
  // check for convergence
  switch (params_.conv_check)
  {
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::residual:
      return (convergence_quantities_.residual_norm <= params_.res_tol);
      break;
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::increment_ratio:
      return (convergence_quantities_.increment_norm <= params_.incr_tol);
      break;
    case InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonConvCheck::
        residual_and_increment_ratio:
      return (convergence_quantities_.residual_norm <= params_.res_tol &&
              convergence_quantities_.increment_norm <= params_.incr_tol);
      break;
    default:
      FOUR_C_THROW("You should not be here (convergence checking of the Local Newton Loop)");
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::
    is_local_newton_stuck() const
{
  // check for "stuck" Local Newton, i.e., the increment does not change much but there is not a
  // converged state (check only feasible after the first iteration, since dx must be available)
  if ((iter_ > 0) && (convergence_quantities_.increment_norm < 1.0e-15))
  {
    // only in the case that the residual is verified, we set an
    // error status
    switch (params_.conv_check)
    {
      case LocalNewtonConvCheck::residual:
      case LocalNewtonConvCheck::residual_and_increment_ratio:
      {
        return (convergence_quantities_.residual_norm > params_.res_tol);
      }
      case LocalNewtonConvCheck::increment_ratio:
      {
        return false;
      }
      default:
        FOUR_C_THROW(
            "You should not be here with convergence check type {} (check: is Local Newton "
            "stuck?)",
            EnumTools::enum_name(params_.conv_check));
    }
  }

  return false;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::pack(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, curr_num_iters_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalNewtonManager::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  // extract last values
  extract_from_pack(buffer, curr_num_iters_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationInput::
    LocalIntegrationInput(const Config& cfg)
{
  defgrad = cfg.defgrad;
  inv_defgrad.invert(defgrad);
  right_cg.multiply_tn(1.0, defgrad, defgrad, 0.0);
  elastic_predictor_inverse_plastic_defgrad = cfg.last_inv_inelastic_defgrad;
  elastic_predictor_elastic_defgrad.multiply(
      1.0, defgrad, elastic_predictor_inverse_plastic_defgrad, 0.0);
  temperature = cfg.temperature;
  last_plastic_strain = cfg.last_plastic_strain;
  step = cfg.step;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::construct_prelim_plastic_pred(
    const Core::LinAlg::Matrix<3, 3>& elastic_defgrad_elastic_pred,
    const double elastic_predictor_zero_component_threshold,
    const PlasticPredictorConstructionParams& plastic_predictor_construction_params)
{
  // get (preconditioned) elastic deformation gradient to be considered as elastic predictor
  const auto precond_elastic_defgrad_elastic_pred =
      precondition_matrix(elastic_defgrad_elastic_pred, elastic_predictor_zero_component_threshold);

  //  perform polar-spectral decomposition of elastic defgrad within elastic predictor
  Core::LinAlg::Matrix<3, 3> material_stretch_elast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> eigenval_elast_pred_temp{
      Core::LinAlg::Initialization::zero};  // needed here for the function call, but we will grab
                                            // the ordered eigenvalues from the spectral pairs
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_elast_pred;
  Core::LinAlg::matrix_3x3_polar_decomposition(precond_elastic_defgrad_elastic_pred,
      rot_elast_pred_, material_stretch_elast_pred, eigenval_elast_pred_temp,
      spectral_pairs_elast_pred);
  for (int i = 0; i < 3; ++i)
  {
    FOUR_C_ASSERT_ALWAYS(spectral_pairs_elast_pred[i].first >= 1.0e-8,
        "The eigenvalue {} of the elastic deformation gradient within the elastic predictor is {}, "
        "such that its logarithm can not be computed!",
        i, spectral_pairs_elast_pred[i].first);
    eigenval_elast_pred_(i, i) = spectral_pairs_elast_pred[i].first;
    for (int j = 0; j < 3; ++j)
    {
      eigenvect_rot_elast_pred_(i, j) = spectral_pairs_elast_pred[i].second(j);
    }
  }

  // -->  construct a preliminary plastic predictor based on the parameter specifications

  // elastic stretch eigenvectors
  switch (plastic_predictor_construction_params.elastic_stretch_eigenvect_type)
  {
    case AEI::PrelimPlasticPredictor::ElasticStretchEigenvectType::from_elastic_predictor:
    {
      rel_eigenvect_rot_plast_pred_ = UNIT_QUATERNION;
      break;
    }
    default:
    {
      // other eigenvector rotation types not yet enabled; in case of multiple eigenvalues, a
      // canonicalization approach for the related eigenvectors must be first implemented for the
      // spectral decomposition to avoid artificial rotation contributions
      FOUR_C_THROW(
          "Elastic stretch eigenvector type {} not yet enabled for the preliminary plastic "
          "predictor",
          EnumTools::enum_name(
              plastic_predictor_construction_params.elastic_stretch_eigenvect_type));
    }
  }

  // elastic rotation
  switch (plastic_predictor_construction_params.elastic_rotation_type)
  {
    case AEI::PrelimPlasticPredictor::ElasticRotationType::from_elastic_predictor:
    {
      rel_rot_plast_pred_ = UNIT_QUATERNION;
      break;
    }
    default:
    {
      // same as in the case of the elastic stretch eigenvectors
      FOUR_C_THROW("Elastic rotation type {} not yet enabled for the plastic predictor!",
          EnumTools::enum_name(plastic_predictor_construction_params.elastic_rotation_type));
    }
  }

  // elastic stretch eigenvalues
  const double detF = precond_elastic_defgrad_elastic_pred.determinant();
  switch (plastic_predictor_construction_params.elastic_stretch_eigenval_type)
  {
    case AEI::PrelimPlasticPredictor::ElasticStretchEigenvalType::scale_unit:
    {
      eigenval_plast_pred_ = Core::LinAlg::diagonal_matrix<3>(std::cbrt(detF));

      break;
    }
    default:
    {
      FOUR_C_THROW("Elastic stretch eigenvalue type {} not yet enabled for the plastic predictor",
          EnumTools::enum_name(
              plastic_predictor_construction_params.elastic_stretch_eigenval_type));
    }
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::interpolate_elastic_defgrad_contributions(const double interp_loc,
    Core::LinAlg::Matrix<4, 1>& interp_rel_rot_quat, std::vector<double>& interp_eigenval,
    Core::LinAlg::Matrix<4, 1>& interp_rel_eigenvect_rot_quat) const
{
  // consistency checks
  FOUR_C_ASSERT(
      ELASTIC_PREDICTOR_LOCATION <= interp_loc && interp_loc <= PLASTIC_PREDICTOR_LOCATION,
      "Interpolation is constrained to the interval between the elastic and the plastic "
      "predictors! The current plastic predictor location "
      "{} is out of these bounds: [{}, {}]",
      interp_loc, ELASTIC_PREDICTOR_LOCATION, PLASTIC_PREDICTOR_LOCATION);

  // auxiliaries
  Core::LinAlg::Matrix<1, 1> matrix_interp_loc{Core::LinAlg::Initialization::zero};
  matrix_interp_loc(0) = interp_loc;

  // --> interpolate eigenvalues
  interp_eigenval =
      interpolate_eigenvalues(eigenval_elast_pred_, eigenval_plast_pred_, matrix_interp_loc);

  // interpolate quaternions
  interp_rel_eigenvect_rot_quat = Core::LinAlg::spherical_linear_interpolation(
      UNIT_QUATERNION, rel_eigenvect_rot_plast_pred_, interp_loc);
  interp_rel_rot_quat = Core::LinAlg::spherical_linear_interpolation(
      UNIT_QUATERNION, rel_rot_plast_pred_, interp_loc);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> AEI::PredictorInterpolator::interpolate_elastic_defgrad(
    const double interp_loc) const
{
  // consistency checks
  FOUR_C_ASSERT(
      ELASTIC_PREDICTOR_LOCATION <= interp_loc && interp_loc <= PLASTIC_PREDICTOR_LOCATION,
      "Interpolation is constrained to the interval between the elastic and the plastic "
      "predictors! The current plastic predictor location "
      "{} is out of these bounds: [{}, {}]",
      interp_loc, ELASTIC_PREDICTOR_LOCATION, PLASTIC_PREDICTOR_LOCATION);

  // interpolate contributions
  Core::LinAlg::Matrix<4, 1> interp_rel_rot_quat{Core::LinAlg::Initialization::zero};
  std::vector<double> interp_eigenval;
  Core::LinAlg::Matrix<4, 1> interp_rel_eigenvect_rot_quat{Core::LinAlg::Initialization::zero};
  interpolate_elastic_defgrad_contributions(
      interp_loc, interp_rel_rot_quat, interp_eigenval, interp_rel_eigenvect_rot_quat);

  return compute_elastic_defgrad_wrt_elastic_predictor(interp_eigenval, eigenvect_rot_elast_pred_,
      interp_rel_eigenvect_rot_quat, rot_elast_pred_, interp_rel_rot_quat);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::PredictorInterpolator::update_plastic_predictor_after_construction_algo(
    const double plastic_pred_loc)
{
  // consistency checks
  FOUR_C_ASSERT(ELASTIC_PREDICTOR_LOCATION <= plastic_pred_loc &&
                    plastic_pred_loc <= PLASTIC_PREDICTOR_LOCATION,
      "Interpolation constrained to the interval between the elastic and the plastic "
      "predictors! The current plastic predictor location "
      "{} is out of these bounds: [{}, {}]",
      plastic_pred_loc, ELASTIC_PREDICTOR_LOCATION, PLASTIC_PREDICTOR_LOCATION);

  // set all quantities relevant for the plastic predictor
  std::vector<double> interp_eigenval;
  interpolate_elastic_defgrad_contributions(
      plastic_pred_loc, rel_rot_plast_pred_, interp_eigenval, rel_eigenvect_rot_plast_pred_);
  for (unsigned int i = 0; i < 3; ++i)
  {
    eigenval_plast_pred_(i, i) = interp_eigenval[i];
  }
}

AEI::InterpolationPointContainer::InterpolationPointContainer(
    const EstimateInterpolationParams& estimate_interpolation_params)
{
  // set starting points for both constant starting points and the strategy based on the
  // equivalent stress history (starting point will evolve for the latter based on the material
  // evaluation of the subsequent timesteps)
  switch (estimate_interpolation_params.starting_point_type)
  {
    case AEI::StartingPointType::constant:
    case AEI::StartingPointType::equiv_stress_history:
    {
      starting_point = estimate_interpolation_params.user_set_starting_point;
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Interpolation point container initialization not supported for starting point type {}",
          EnumTools::enum_name(estimate_interpolation_params.starting_point_type));
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::InterpolationPointContainer::reset_bounds_and_current_interp_point()
{
  current_interp_point = starting_point;
  lower_interp_bound = ELASTIC_PREDICTOR_LOCATION;
  upper_interp_bound = PLASTIC_PREDICTOR_LOCATION;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
AEI::AEIManager::AEIManager(const AEI::AEIParams& aei_params) : params_(aei_params)
{
  // initialize class variables
  num_plastic_pred_construct_iters_ = 0;
  num_estimate_interp_iters_ = 0;
  num_reestimations_ = 0;
  interpolation_point_containers_.resize(
      1, InterpolationPointContainer(aei_params.estimate_interpolation));
  predictor_interpolators_.resize(1);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::resize(const unsigned int num_gp)
{
  FOUR_C_ASSERT(!resize_called_,
      "You already called resize for the adaptive estimate interpolation manager! You attempt to "
      "set it to {}",
      num_gp);

  interpolation_point_containers_.resize(num_gp, interpolation_point_containers_[0]);
  predictor_interpolators_.resize(num_gp, predictor_interpolators_[0]);

  resize_called_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::reset_and_construct_prelim_plastic_pred(
    const unsigned int gp, const LocalIntegrationInput& local_integration_input)
{
  // reset tracking variables
  num_plastic_pred_construct_iters_ = 0;
  num_estimate_interp_iters_ = 0;
  num_reestimations_ = 0;
  interpolation_point_containers_[gp].reset_bounds_and_current_interp_point();

  // construct the preliminary predictor
  predictor_interpolators_[gp].construct_prelim_plastic_pred(
      local_integration_input.elastic_predictor_elastic_defgrad,
      params_.elastic_predictor_zero_component_threshold, params_.plastic_predictor_construction);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::pack(Core::Communication::PackBuffer& data) const
{
  // pack number of Gauss points
  Core::Communication::add_to_pack(data, interpolation_point_containers_.size());

  // pack starting points
  for (const auto& interp_point_container : interpolation_point_containers_)
  {
    Core::Communication::add_to_pack(data, interp_point_container.starting_point);
  }
  Core::Communication::add_to_pack(data, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::unpack(Core::Communication::UnpackBuffer& buffer)
{
  // unpack number of Gauss points
  std::size_t num_gp;
  Core::Communication::extract_from_pack(buffer, num_gp);

  // create interpolation point containers and predictor interpolators according to the number of
  // Gauss points; for the interpolation point containers, also extract the respective starting
  // point from the buffer
  interpolation_point_containers_.resize(
      num_gp, InterpolationPointContainer(params_.estimate_interpolation));
  predictor_interpolators_.resize(num_gp);
  for (unsigned int gp = 0; gp < num_gp; ++gp)
  {
    Core::Communication::extract_from_pack(
        buffer, interpolation_point_containers_[gp].starting_point);
  }
  Core::Communication::extract_from_pack(buffer, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> AEI::AEIManager::interpolate_inverse_inelastic_defgrad(
    const unsigned int gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad) const
{
  Core::LinAlg::Matrix<3, 3> interp_elastic_defgrad =
      predictor_interpolators_[gp].interpolate_elastic_defgrad(
          interpolation_point_containers_[gp].current_interp_point);

  Core::LinAlg::Matrix<3, 3> inv_inelastic_defgrad{Core::LinAlg::Initialization::zero};
  inv_inelastic_defgrad.multiply(1.0, inv_defgrad, interp_elastic_defgrad);

  return inv_inelastic_defgrad;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> AEI::AEIManager::get_inverse_inelastic_defgrad_plastic_pred(
    const unsigned int gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad) const
{
  // the plastic predictor lies at the location 1.0
  Core::LinAlg::Matrix<3, 3> interp_elastic_defgrad =
      predictor_interpolators_[gp].interpolate_elastic_defgrad(1.0);

  Core::LinAlg::Matrix<3, 3> inv_inelastic_defgrad{Core::LinAlg::Initialization::zero};
  inv_inelastic_defgrad.multiply(1.0, inv_defgrad, interp_elastic_defgrad);

  return inv_inelastic_defgrad;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::update_plastic_predictor_after_construction_algo(const unsigned int gp)
{
  // update the plastic predictor quantities
  predictor_interpolators_[gp].update_plastic_predictor_after_construction_algo(
      interpolation_point_containers_[gp].current_interp_point);

  // reset the interpolation point container
  interpolation_point_containers_[gp].reset_bounds_and_current_interp_point();
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::adapt_interpolation_interval(
    const unsigned int gp, const InterpolationIntervalShift& interval_shift)
{
  switch (interval_shift)
  {
    case InterpolationIntervalShift::towards_elastic_pred:
    {
      interpolation_point_containers_[gp].upper_interp_bound =
          interpolation_point_containers_[gp].current_interp_point;
      break;
    }
    case InterpolationIntervalShift::towards_plastic_pred:
    {
      interpolation_point_containers_[gp].lower_interp_bound =
          interpolation_point_containers_[gp].current_interp_point;
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "You should not be here in the interpolation routine! The shift direction {} is not "
          "supported!",
          EnumTools::enum_name(interval_shift));
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::set_current_interp_point(
    const unsigned int gp, const CurrentInterpPointPreset preset)
{
  FOUR_C_ASSERT(gp < interpolation_point_containers_.size(), "GP index out of range");
  switch (preset)
  {
    case CurrentInterpPointPreset::plastic_pred_construct_update:
    {
      interpolation_point_containers_[gp].current_interp_point =
          interpolation_point_containers_[gp].lower_interp_bound +
          params_.plastic_predictor_construction.interval_scanning_param *
              (interpolation_point_containers_[gp].upper_interp_bound -
                  interpolation_point_containers_[gp].lower_interp_bound);
      return;
    }
    case CurrentInterpPointPreset::estimate_interpolation_update:
    {
      interpolation_point_containers_[gp].current_interp_point =
          interpolation_point_containers_[gp].lower_interp_bound +
          params_.estimate_interpolation.interval_scanning_param *
              (interpolation_point_containers_[gp].upper_interp_bound -
                  interpolation_point_containers_[gp].lower_interp_bound);
      return;
    }
    case CurrentInterpPointPreset::lower_interp_bound:
    {
      interpolation_point_containers_[gp].current_interp_point =
          interpolation_point_containers_[gp].lower_interp_bound;
      return;
    }
    case CurrentInterpPointPreset::upper_interp_bound:
    {
      interpolation_point_containers_[gp].current_interp_point =
          interpolation_point_containers_[gp].upper_interp_bound;
      return;
    }
    case CurrentInterpPointPreset::elastic_predictor:
    {
      interpolation_point_containers_[gp].current_interp_point = ELASTIC_PREDICTOR_LOCATION;
      return;
    }
    case CurrentInterpPointPreset::plastic_predictor:
    {
      interpolation_point_containers_[gp].current_interp_point = PLASTIC_PREDICTOR_LOCATION;
      return;
    }
    case CurrentInterpPointPreset::starting_point:
    {
      interpolation_point_containers_[gp].current_interp_point =
          interpolation_point_containers_[gp].starting_point;
      return;
    }
    case CurrentInterpPointPreset::intermediate_point:
    {
      interpolation_point_containers_[gp].current_interp_point =
          interpolation_point_containers_[gp].lower_interp_bound +
          params_.reestimation.interval_scanning_param *
              (interpolation_point_containers_[gp].current_interp_point -
                  interpolation_point_containers_[gp].lower_interp_bound);
      return;
    }
    default:
      FOUR_C_THROW(
          "Unsupported current interpolation point preset {}", EnumTools::enum_name(preset));
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::set_user_starting_point(const unsigned gp)
{
  FOUR_C_ASSERT(gp < interpolation_point_containers_.size(), "GP index out of range");
  FOUR_C_ASSERT_ALWAYS(
      params_.estimate_interpolation.starting_point_type == AEI::StartingPointType::constant,
      "Setter should only be called for user-set starting points, not for {}",
      EnumTools::enum_name(params_.estimate_interpolation.starting_point_type));


  interpolation_point_containers_[gp].starting_point =
      params_.estimate_interpolation.user_set_starting_point;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void AEI::AEIManager::set_stress_based_starting_point(
    const unsigned gp, InputEquivStressStartingPoint input_equiv_stress_starting_point)
{
  FOUR_C_ASSERT(gp < interpolation_point_containers_.size(), "GP index out of range");
  FOUR_C_ASSERT_ALWAYS(params_.estimate_interpolation.starting_point_type ==
                           AEI::StartingPointType::equiv_stress_history,
      "Setter should only be called for stress-based starting points, not for {}",
      EnumTools::enum_name(params_.estimate_interpolation.starting_point_type));

  interpolation_point_containers_[gp].starting_point =
      calculate_equiv_stress_starting_point(input_equiv_stress_starting_point);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::LocalTimIntAnalysis(
    const bool use_adaptive_estimate_interpolation, const unsigned int num_gp)
{
  // set initial tracking settings
  tracking_settings_ = TrackingSettings{
      .time = 0.0,
      .timestep = 0,
      .is_new_timestep = true,
      .global_iter = -1,  // we set it to -1, since there is an initial evaluation when the
                          // simulation starts which will increment this value
  };


  // initialize stored quantities
  total_num_global_iters_ = 0;
  num_lnl_iters_.resize(num_gp, 0);
  total_num_lnl_iters_.resize(num_gp, 0);
  constitutive_update_time_.resize(num_gp, 0.0);
  total_constitutive_update_time_.resize(num_gp, 0.0);

  // initialize and resize the Local Newton data tracker
  local_newton_data_.global_iters.resize(num_gp, std::vector<unsigned int>{});
  local_newton_data_.local_iters.resize(num_gp, std::vector<unsigned int>{});
  local_newton_data_.error_status.resize(num_gp, std::vector<ErrorType>{});
  local_newton_data_.residual_norms.resize(num_gp, std::vector<double>{});
  local_newton_data_.increment_norms.resize(num_gp, std::vector<double>{});
  local_newton_data_.is_converged.resize(num_gp, std::vector<bool>{});
  if (use_adaptive_estimate_interpolation)
  {
    local_newton_data_.current_interpolation_points = std::vector<std::vector<double>>(num_gp);
  }
  local_newton_data_.equiv_stresses.resize(num_gp, std::vector<double>{});
  local_newton_data_.plastic_strains.resize(num_gp, std::vector<double>{});
  local_newton_data_.plastic_strain_increments.resize(num_gp, std::vector<double>{});

  // initialize and resize the Adaptive Estimate Interpolation data
  if (use_adaptive_estimate_interpolation)
  {
    adaptive_estimate_interp_data_ =
        std::make_optional<LocalTimIntAnalysis::AdaptiveEstimateInterpolationData>();

    adaptive_estimate_interp_data_->num_plastic_pred_construct_iters.resize(num_gp, 0);
    adaptive_estimate_interp_data_->total_num_plastic_pred_construct_iters.resize(num_gp, 0);
    adaptive_estimate_interp_data_->num_interp_iters.resize(num_gp, 0);
    adaptive_estimate_interp_data_->total_num_interp_iters.resize(num_gp, 0);
    adaptive_estimate_interp_data_->num_reestimations.resize(num_gp, 0);
    adaptive_estimate_interp_data_->total_num_reestimations.resize(num_gp, 0);
    adaptive_estimate_interp_data_->current_interp_points.resize(num_gp, std::vector<double>{});
    adaptive_estimate_interp_data_->lower_interp_bounds.resize(num_gp, std::vector<double>{});
    adaptive_estimate_interp_data_->upper_interp_bounds.resize(num_gp, std::vector<double>{});
    adaptive_estimate_interp_data_->global_iters.resize(num_gp, std::vector<double>{});
    adaptive_estimate_interp_data_->local_iters.resize(num_gp, std::vector<double>{});
    adaptive_estimate_interp_data_->interp_point_equiv_stress_history.resize(num_gp, 0.0);
    adaptive_estimate_interp_data_->starting_point.resize(num_gp, 0.0);
    adaptive_estimate_interp_data_->starting_point_determination_time.resize(num_gp, 0.0);
    adaptive_estimate_interp_data_->total_starting_point_determination_time.resize(num_gp, 0.0);
  }

  // create csv writer for timestep data (overall)
  csv_timestep_data_writer_overall_.emplace(
      0, *Global::Problem::instance()->output_control_file(), "timint_output");
  init_timestep_data_csv_writer(
      adaptive_estimate_interp_data_.has_value(), csv_timestep_data_writer_overall_.value(), true);

  // initialize Gauss point csv writers
  for (unsigned int gp = 0; gp < num_gp; ++gp)
  {
    csv_timestep_data_writer_at_gp_.insert(std::make_pair(gp, std::nullopt));
    csv_timestep_data_writer_at_gp_[gp].emplace(0,
        *Global::Problem::instance()->output_control_file(),
        "timint_output_gp_" + std::to_string(gp));
    init_timestep_data_csv_writer(
        adaptive_estimate_interp_data_.has_value(), csv_timestep_data_writer_at_gp_[gp].value());

    csv_lnl_data_writer_at_gp_[gp].emplace(0, *Global::Problem::instance()->output_control_file(),
        "lnl_output_gp_" + std::to_string(gp));
    init_lnl_data_csv_writer(
        csv_lnl_data_writer_at_gp_[gp].value(), use_adaptive_estimate_interpolation);

    if (adaptive_estimate_interp_data_.has_value())
    {
      csv_aei_data_writer_at_gp_[gp].emplace(0, *Global::Problem::instance()->output_control_file(),
          "aei_output_gp_" + std::to_string(gp));
      init_aei_data_csv_writer(csv_aei_data_writer_at_gp_[gp].value());
    }
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::update_total(
    const unsigned int gp)
{
  if (gp == 0) total_num_global_iters_ += tracking_settings_.global_iter;
  total_num_lnl_iters_[gp] += num_lnl_iters_[gp];
  total_constitutive_update_time_[gp] += constitutive_update_time_[gp];
  if (adaptive_estimate_interp_data_.has_value())
  {
    adaptive_estimate_interp_data_->total_num_plastic_pred_construct_iters[gp] +=
        adaptive_estimate_interp_data_->num_plastic_pred_construct_iters[gp];
    adaptive_estimate_interp_data_->total_num_interp_iters[gp] +=
        adaptive_estimate_interp_data_->num_interp_iters[gp];
    adaptive_estimate_interp_data_->total_num_reestimations[gp] +=
        adaptive_estimate_interp_data_->num_reestimations[gp];
    adaptive_estimate_interp_data_->total_starting_point_determination_time[gp] +=
        adaptive_estimate_interp_data_->starting_point_determination_time[gp];
  }
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::reset(
    const unsigned int gp)
{
  num_lnl_iters_[gp] = 0;
  constitutive_update_time_[gp] = 0.0;

  // reset Local Newton data tracker
  local_newton_data_.global_iters[gp] = {};
  local_newton_data_.local_iters[gp] = {};
  local_newton_data_.error_status[gp] = {};
  local_newton_data_.residual_norms[gp] = {};
  local_newton_data_.increment_norms[gp] = {};
  local_newton_data_.is_converged[gp] = {};
  if (local_newton_data_.current_interpolation_points.has_value())
    local_newton_data_.current_interpolation_points->at(gp) = {};
  local_newton_data_.equiv_stresses[gp] = {};
  local_newton_data_.plastic_strains[gp] = {};
  local_newton_data_.plastic_strain_increments[gp] = {};


  // reset Adaptive Estimate Interpolation data tracker
  if (adaptive_estimate_interp_data_.has_value())
  {
    adaptive_estimate_interp_data_->num_plastic_pred_construct_iters[gp] = 0;
    adaptive_estimate_interp_data_->num_interp_iters[gp] = 0;
    adaptive_estimate_interp_data_->num_reestimations[gp] = 0;
    adaptive_estimate_interp_data_->current_interp_points[gp] = {};
    adaptive_estimate_interp_data_->lower_interp_bounds[gp] = {};
    adaptive_estimate_interp_data_->upper_interp_bounds[gp] = {};
    adaptive_estimate_interp_data_->global_iters[gp] = {};
    adaptive_estimate_interp_data_->local_iters[gp] = {};
    adaptive_estimate_interp_data_->interp_point_equiv_stress_history[gp] = -1.0;
    adaptive_estimate_interp_data_->starting_point[gp] = -1.0;
    adaptive_estimate_interp_data_->starting_point_determination_time[gp] = 0.0;
  }
  tracking_settings_.global_iter =
      0;  // here we set it to 0, since the new timestep should start with the 0-th iteration
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
    update_gp_totals_and_write_gp_tables(const unsigned int gp)
{
  update_total(gp);
  write_timestep_tables_at_gp_to_csv(gp);
  write_lnl_tables_to_csv(gp);
  if (adaptive_estimate_interp_data_.has_value()) write_aei_tables_to_csv(gp);
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
    write_timestep_tables_at_gp_to_csv(const unsigned int gp)
{
  const auto csv_writing_timestep_aei_data =
      adaptive_estimate_interp_data_.has_value()
          ? std::make_optional(CsvWritingTimestepAEIData{
                .num_plastic_pred_construct_iters =
                    adaptive_estimate_interp_data_->num_plastic_pred_construct_iters[gp],
                .total_num_plastic_pred_construct_iters =
                    adaptive_estimate_interp_data_->total_num_plastic_pred_construct_iters[gp],
                .num_interp_iters = adaptive_estimate_interp_data_->num_interp_iters[gp],
                .total_num_interp_iters =
                    adaptive_estimate_interp_data_->total_num_interp_iters[gp],
                .num_reestimations = adaptive_estimate_interp_data_->num_reestimations[gp],
                .total_num_reestimations =
                    adaptive_estimate_interp_data_->total_num_reestimations[gp],
                .interp_point_equiv_stress_history =
                    adaptive_estimate_interp_data_->interp_point_equiv_stress_history[gp],
                .starting_point = adaptive_estimate_interp_data_->starting_point[gp],
                .starting_point_determination_time =
                    adaptive_estimate_interp_data_->starting_point_determination_time[gp],
                .total_starting_point_determination_time =
                    adaptive_estimate_interp_data_->total_starting_point_determination_time[gp],
            })
          : std::nullopt;

  const auto csv_writing_timestep_data = CsvWritingTimestepData{
      .num_global_iters = static_cast<unsigned int>(tracking_settings_.global_iter),
      .total_num_global_iters = total_num_global_iters_,
      .num_lnl_iters = num_lnl_iters_[gp],
      .total_num_lnl_iters = total_num_lnl_iters_[gp],
      .constitutive_update_time = constitutive_update_time_[gp],
      .total_constitutive_update_time = total_constitutive_update_time_[gp],
      .aei_data = csv_writing_timestep_aei_data,
  };



  write_timestep_data_to_csv(
      csv_timestep_data_writer_at_gp_[gp].value(), csv_writing_timestep_data, tracking_settings_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
    write_overall_timestep_table_to_csv()
{
  const auto csv_writing_timestep_aei_data =
      adaptive_estimate_interp_data_.has_value()
          ? std::make_optional(CsvWritingTimestepAEIData{
                .num_plastic_pred_construct_iters = std::reduce(
                    adaptive_estimate_interp_data_->num_plastic_pred_construct_iters.begin(),
                    adaptive_estimate_interp_data_->num_plastic_pred_construct_iters.end()),
                .total_num_plastic_pred_construct_iters = std::reduce(
                    adaptive_estimate_interp_data_->total_num_plastic_pred_construct_iters.begin(),
                    adaptive_estimate_interp_data_->total_num_plastic_pred_construct_iters.end()),
                .num_interp_iters =
                    std::reduce(adaptive_estimate_interp_data_->num_interp_iters.begin(),
                        adaptive_estimate_interp_data_->num_interp_iters.end()),
                .total_num_interp_iters =
                    std::reduce(adaptive_estimate_interp_data_->total_num_interp_iters.begin(),
                        adaptive_estimate_interp_data_->total_num_interp_iters.end()),
                .num_reestimations =
                    std::reduce(adaptive_estimate_interp_data_->num_reestimations.begin(),
                        adaptive_estimate_interp_data_->num_reestimations.end()),
                .total_num_reestimations =
                    std::reduce(adaptive_estimate_interp_data_->total_num_reestimations.begin(),
                        adaptive_estimate_interp_data_->total_num_reestimations.end()),
                .interp_point_equiv_stress_history = -1.0,
                .starting_point = -1.0,
                .starting_point_determination_time = std::reduce(
                    adaptive_estimate_interp_data_->starting_point_determination_time.begin(),
                    adaptive_estimate_interp_data_->starting_point_determination_time.end()),
                .total_starting_point_determination_time = std::reduce(
                    adaptive_estimate_interp_data_->total_starting_point_determination_time.begin(),
                    adaptive_estimate_interp_data_->total_starting_point_determination_time.end()),
            })
          : std::nullopt;

  const auto csv_writing_timestep_data = CsvWritingTimestepData{
      .num_global_iters = static_cast<unsigned int>(tracking_settings_.global_iter),
      .total_num_global_iters = total_num_global_iters_,
      .num_lnl_iters = std::reduce(num_lnl_iters_.begin(), num_lnl_iters_.end()),
      .total_num_lnl_iters = std::reduce(total_num_lnl_iters_.begin(), total_num_lnl_iters_.end()),
      .constitutive_update_time =
          std::reduce(constitutive_update_time_.begin(), constitutive_update_time_.end()),
      .total_constitutive_update_time = std::reduce(
          total_constitutive_update_time_.begin(), total_constitutive_update_time_.end()),
      .aei_data = csv_writing_timestep_aei_data};
  write_timestep_data_to_csv(csv_timestep_data_writer_overall_.value(), csv_writing_timestep_data,
      tracking_settings_, true);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
    write_aei_tables_to_csv(const unsigned int gp)
{
  const auto csv_writing_aei_data = CsvWritingAEIData{
      .aei_current_interp_points = adaptive_estimate_interp_data_->current_interp_points[gp],
      .aei_lower_interp_bounds = adaptive_estimate_interp_data_->lower_interp_bounds[gp],
      .aei_upper_interp_bounds = adaptive_estimate_interp_data_->upper_interp_bounds[gp],
      .aei_global_iters = adaptive_estimate_interp_data_->global_iters[gp],
      .aei_local_iters = adaptive_estimate_interp_data_->local_iters[gp]};
  write_aei_data_to_csv(
      csv_aei_data_writer_at_gp_[gp].value(), csv_writing_aei_data, tracking_settings_);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
    write_lnl_tables_to_csv(const unsigned int gp)
{
  const auto csv_writing_lnl_data = CsvWritingLNLData{
      .global_iters = local_newton_data_.global_iters[gp],
      .local_iters = local_newton_data_.local_iters[gp],
      .error_status = local_newton_data_.error_status[gp],
      .residual_norms = local_newton_data_.residual_norms[gp],
      .increment_norms = local_newton_data_.increment_norms[gp],
      .is_converged = local_newton_data_.is_converged[gp],
      .current_interpolation_points =
          local_newton_data_.current_interpolation_points.has_value()
              ? std::make_optional(local_newton_data_.current_interpolation_points->at(gp))
              : std::nullopt,
      .equiv_stresses = local_newton_data_.equiv_stresses[gp],
      .plastic_strains = local_newton_data_.plastic_strains[gp],
      .plastic_strain_increments = local_newton_data_.plastic_strain_increments[gp],
  };

  write_lnl_data_to_csv(
      csv_lnl_data_writer_at_gp_[gp].value(), csv_writing_lnl_data, tracking_settings_);
}

void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalTimIntAnalysis::
    set_aei_interp_point_equiv_stress_history(const unsigned int gp,
        const AdaptiveEstimateInterpolation::AEIManager& aei_manager,
        const AdaptiveEstimateInterpolation::InputEquivStressStartingPoint&
            input_equiv_stress_starting_point)
{
  FOUR_C_ASSERT_ALWAYS(
      adaptive_estimate_interp_data_.has_value(), "This method should not be called!");

  FOUR_C_ASSERT_ALWAYS(
      gp < adaptive_estimate_interp_data_->interp_point_equiv_stress_history.size(),
      "You try to set interp_point_equiv_stress_history at GP {}, but the current "
      "size "
      "is {}",
      gp, adaptive_estimate_interp_data_->interp_point_equiv_stress_history.size());
  adaptive_estimate_interp_data_->interp_point_equiv_stress_history[gp] =
      calculate_equiv_stress_starting_point(input_equiv_stress_starting_point);
}


FOUR_C_NAMESPACE_CLOSE
