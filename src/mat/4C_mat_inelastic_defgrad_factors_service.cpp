// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_inelastic_defgrad_factors_service.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_linalg_utlis_quaternion_interpolation.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <vector>


FOUR_C_NAMESPACE_OPEN

using namespace Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

namespace
{
  // sets the elastic and plastic predictor locations for the adaptive estimate interpolation,
  // specifically 0 and 1
  std::vector<Core::LinAlg::Matrix<1, 1>> set_elast_and_plast_predictor_locs()
  {
    std::vector<Core::LinAlg::Matrix<1, 1>> locs;
    Core::LinAlg::Matrix<1, 1> loc_elast_pred(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<1, 1> loc_plast_pred(Core::LinAlg::Initialization::zero);
    loc_elast_pred(0, 0) = 0.0;
    loc_plast_pred(0, 0) = 1.0;
    locs.push_back(loc_elast_pred);
    locs.push_back(loc_plast_pred);
    return locs;
  }

  // creates the eigenvalue interpolator used for the adaptive estimate interpolation
  Core::LinAlg::ScalarInterpolator<1> create_eigenvalue_interpolator()
  {
    Core::LinAlg::ScalarInterpolationType interp_type =
        Core::LinAlg::ScalarInterpolationType::logarithmic_weighted_average;
    Core::LinAlg::ScalarInterpolationWeightingFunction weight_func =
        Core::LinAlg::ScalarInterpolationWeightingFunction::inverse_distance;
    Core::LinAlg::ScalarInterpolationParams interp_params;

    return Core::LinAlg::ScalarInterpolator<1>(interp_type, weight_func, interp_params);
  }


  // adaptive estimate interpolation: compute the elastic deformation gradient, using the
  // interpolated eigenvalues and rotation contributions (quaternions) with respect to the elastic
  // deformation gradient within the elastic predictor
  Core::LinAlg::Matrix<3, 3> compute_elast_defgrad_wrt_elast_predictor(
      const std::vector<double>& interp_eigenval,
      const Core::LinAlg::Matrix<3, 3>& eigenvect_rot_elast_pred,
      const Core::LinAlg::Matrix<4, 1>& interp_rel_eigenvect_rot_quat,
      const Core::LinAlg::Matrix<3, 3>& rot_elast_pred,
      const Core::LinAlg::Matrix<4, 1>& interp_rel_rot_quat)
  {
    Core::LinAlg::Matrix<3, 3> out{Core::LinAlg::Initialization::zero};

    // construct diagonal eigenvalue matrix
    Core::LinAlg::Matrix<3, 3> eigenval_matrix{Core::LinAlg::Initialization::zero};
    for (unsigned int i = 0; i < 3; ++i)
    {
      eigenval_matrix(i, i) = interp_eigenval[i];
    }


    // construct eigenvector matrix
    Core::LinAlg::Matrix<3, 3> rel_interp_eigenvect_matrix{Core::LinAlg::Initialization::zero};
    Core::LargeRotations::quaterniontotriad(
        interp_rel_eigenvect_rot_quat, rel_interp_eigenvect_matrix);
    Core::LinAlg::Matrix<3, 3> interp_eigenvect_matrix{Core::LinAlg::Initialization::zero};
    interp_eigenvect_matrix.multiply_nn(
        1.0, eigenvect_rot_elast_pred, rel_interp_eigenvect_matrix, 0.0);


    // construct rotation matrix
    Core::LinAlg::Matrix<3, 3> rel_interp_rot_matrix{Core::LinAlg::Initialization::zero};
    Core::LargeRotations::quaterniontotriad(interp_rel_rot_quat, rel_interp_rot_matrix);
    Core::LinAlg::Matrix<3, 3> interp_rot_matrix{Core::LinAlg::Initialization::zero};
    interp_rot_matrix.multiply_nn(1.0, rot_elast_pred, rel_interp_rot_matrix, 0.0);


    // multiply contributions
    Core::LinAlg::Matrix<3, 3> LQ{Core::LinAlg::Initialization::zero};
    LQ.multiply(1.0, eigenval_matrix, interp_eigenvect_matrix, 0.0);
    Core::LinAlg::Matrix<3, 3> QTLQ{Core::LinAlg::Initialization::zero};
    QTLQ.multiply_tn(1.0, interp_eigenvect_matrix, LQ, 0.0);
    out.multiply(1.0, interp_rot_matrix, QTLQ, 0.0);

    return out;
  }


  // precondition matrix: values smaller than a set tolerance are set to 0.0
  void precondition_matrix(Core::LinAlg::Matrix<3, 3>& input_matrix, const double tol)
  {
    for (unsigned i = 0; i < 3; ++i)
    {
      for (unsigned j = 0; j < 3; ++j)
      {
        if (std::abs(input_matrix(i, j)) < tol)
        {
          input_matrix(i, j) = 0.0;
        }
      }
    }
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
        csv_writer.register_data_vector("Interp. point opt. equiv. stress (AEI)", 1, 16);
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

    //! adaptive estimate interpolation: interpolation point leading to optimal equivalent stress
    //! in the current timestep
    const double interp_point_optimal_equiv_stress;

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
        output_data["Interp. point opt. equiv. stress (AEI)"] = {
            data.aei_data->interp_point_optimal_equiv_stress};


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

    //! vector tracking whether the iterations have errors in the
    //! current timestep
    std::vector<bool> has_error;

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

    //! vector tracking the plastic strain increments in the
    //! current timestep
    std::vector<double> plastic_strain_increments;

    //! verify equal lengths of the vectors
    void verify_equal_lengths() const
    {
      auto l = {global_iters.size(), local_iters.size(), has_error.size(), residual_norms.size(),
          increment_norms.size(), is_converged.size(),
          current_interpolation_points.has_value() ? current_interpolation_points->size()
                                                   : global_iters.size(),
          equiv_stresses.size(), plastic_strain_increments.size()};

      auto all_same =
          std::all_of(l.begin(), l.end(), [&](unsigned int v) { return v == *l.begin(); });
      FOUR_C_ASSERT_ALWAYS(all_same,
          "Your vectors for the LNL don't have equal lengths! Global iters: {}, "
          "local iters: {}, has_error: {}, residual_norms: "
          "{}, "
          "increment_norms: {}, current_interpolation_points: {}, equiv_stresses: {}, "
          "plastic_strain_increments: {}",
          global_iters.size(), local_iters.size(), has_error.size(), residual_norms.size(),
          increment_norms.size(), is_converged.size(),
          current_interpolation_points.has_value() ? current_interpolation_points->size()
                                                   : global_iters.size(),
          equiv_stresses.size(), plastic_strain_increments.size());
    }
  };


  /// initialize csv Local Newton data writer
  void init_lnl_data_csv_writer(
      Core::IO::RuntimeCsvWriter& csv_writer, const bool use_adaptive_estimate_interpolation)
  {
    csv_writer.register_data_vector("Global iteration", 1, 16);
    csv_writer.register_data_vector("Local iteration", 1, 16);
    csv_writer.register_data_vector("Has error?", 1, 16);
    csv_writer.register_data_vector("Residual norm", 1, 16);
    csv_writer.register_data_vector("Increment norm", 1, 16);
    csv_writer.register_data_vector("Is converged?", 1, 16);
    if (use_adaptive_estimate_interpolation)
      csv_writer.register_data_vector("Current interpolation point", 1, 16);
    csv_writer.register_data_vector("Equivalent stress", 1, 16);
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
      output_data["Has error?"] = {static_cast<double>(data.has_error[l])};
      output_data["Residual norm"] = {static_cast<double>(data.residual_norms[l])};
      output_data["Increment norm"] = {static_cast<double>(data.increment_norms[l])};
      output_data["Is converged?"] = {static_cast<double>(data.is_converged[l])};
      if (data.current_interpolation_points.has_value())
        output_data["Current interpolation point"] = {
            static_cast<double>(data.current_interpolation_points->at(l))};
      output_data["Equivalent stress"] = {static_cast<double>(data.equiv_stresses[l])};
      output_data["Plastic strain increment"] = {
          static_cast<double>(data.plastic_strain_increments[l])};

      // write output data to csv
      csv_writer.write_data_to_file(
          tracking_settings.time, tracking_settings.timestep, output_data);
    }
  }



}  // namespace


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
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
  id3x3.update(1.0, unit3x3, 0.0);

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
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities::init()
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
  last_plastic_strain[gp] = current_plastic_strain[gp];
  last_equiv_stress[gp] = current_equiv_stress[gp];
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

  // fill current_ values with the last_ values
  current_rightCG.resize(last_rightCG.size(),
      last_rightCG[0]);  // value irrelevant
  current_plastic_defgrad_inverse.resize(last_plastic_defgrad_inverse.size(),
      last_plastic_defgrad_inverse[0]);  // value irrelevant
  current_plastic_strain.resize(last_plastic_strain.size(),
      last_plastic_strain[0]);  // value irrelevant
  current_equiv_stress.resize(last_equiv_stress.size(),
      last_equiv_stress[0]);  // value irrelevant

  // set evaluated deformation gradient to 0, to make sure that the inverse inelastic deformation
  // gradient is evaluated fully after the restart
  current_defgrad.resize(last_substep_plastic_defgrad_inverse.size(),
      Core::LinAlg::Matrix<3, 3>{Core::LinAlg::Initialization::zero});
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
  if ((iter_ > 1) && (convergence_quantities_.increment_norm < 1.0e-15))
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
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::LocalIntegrationDeformationTensors::
    LocalIntegrationDeformationTensors(
        const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<3, 3>& last_iFp)
{
  defgrad = F;
  inv_defgrad.invert(defgrad);
  right_cg.multiply_tn(1.0, defgrad, defgrad, 0.0);
  elastic_predictor_inverse_plastic_defgrad = last_iFp;
  elastic_predictor_elastic_defgrad.multiply(
      1.0, defgrad, elastic_predictor_inverse_plastic_defgrad, 0.0);
}

Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::InterpolationPointContainer::
    InterpolationPointContainer(const AdaptiveEstimateInterpolationParams& aei_params)
{
  current_interp_points.resize(1, 0.0);
  lower_interp_bounds.resize(1, 0.0);
  upper_interp_bounds.resize(1, 1.0);
  switch (aei_params.starting_point_type)
  {
    case AdaptiveEstimateInterpolationStartingPointType::user_set:
    {
      starting_points.resize(1, aei_params.user_set_starting_point);
      break;
    }
    default:
    {
      starting_points.resize(1, aei_params.interval_scanning_param);
      break;
    }
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::InterpolationPointContainer::
    reset_bounds_and_current_interp_point(const unsigned int gp)
{
  // consistency checks
  FOUR_C_ASSERT_ALWAYS(gp < current_interp_points.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      current_interp_points.size());

  current_interp_points[gp] = starting_points[gp];
  lower_interp_bounds[gp] = 0.0;
  upper_interp_bounds[gp] = 1.0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::InterpolationPointContainer::resize(
    const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called,
      "You already called resize for the interpolation point container! The number of "
      "current GP is "
      "{} and "
      "you attempt to set it to {}",
      current_interp_points.size(), numgp);

  current_interp_points.resize(numgp, current_interp_points[0]);
  lower_interp_bounds.resize(numgp, lower_interp_bounds[0]);
  upper_interp_bounds.resize(numgp, upper_interp_bounds[0]);
  starting_points.resize(numgp, starting_points[0]);

  resize_called = true;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::InterpolationPointContainer::pack(
    Core::Communication::PackBuffer& data) const
{
  Core::Communication::add_to_pack(data, current_interp_points);
  Core::Communication::add_to_pack(data, lower_interp_bounds);
  Core::Communication::add_to_pack(data, upper_interp_bounds);
  Core::Communication::add_to_pack(data, starting_points);
  Core::Communication::add_to_pack(data, resize_called);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::InterpolationPointContainer::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_from_pack(buffer, current_interp_points);
  Core::Communication::extract_from_pack(buffer, lower_interp_bounds);
  Core::Communication::extract_from_pack(buffer, upper_interp_bounds);
  Core::Communication::extract_from_pack(buffer, starting_points);
  Core::Communication::extract_from_pack(buffer, resize_called);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PredictorInterpolator::
    PredictorInterpolator()
    : ref_predictor_locs_(set_elast_and_plast_predictor_locs()),
      eigenval_interpolator_(create_eigenvalue_interpolator())
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> unit_3x3{Core::LinAlg::Initialization::zero};
  for (unsigned int i = 0; i < 3; ++i)
  {
    unit_3x3(i, i) = 1.0;
  }
  std::vector<std::vector<double>> vector_of_ones(2, {1.0, 1.0, 1.0});
  Core::LinAlg::Matrix<4, 1> unit_quaternion{Core::LinAlg::Initialization::zero};
  unit_quaternion(3) = 1.0;

  // initialize variables for a single Gauss point
  eigenval_elast_pred_.resize(1, unit_3x3);
  eigenval_plast_pred_.resize(1, unit_3x3);
  scalar_interp_eigenval_.resize(1, vector_of_ones);
  eigenvect_rot_elast_pred_.resize(1, unit_3x3);
  rel_eigenvect_rot_plast_pred_.resize(1, unit_quaternion);
  rot_elast_pred_.resize(1, unit_3x3);
  rel_rot_plast_pred_.resize(1, unit_quaternion);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PredictorInterpolator::resize(
    const unsigned int numgp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called_,
      "You already called resize for the predictor interpolator! The number of current GP is {} "
      "and "
      "you attempt to set it to {}",
      eigenval_elast_pred_.size(), numgp);

  eigenval_elast_pred_.resize(numgp, eigenval_elast_pred_[0]);
  eigenval_plast_pred_.resize(numgp, eigenval_plast_pred_[0]);
  scalar_interp_eigenval_.resize(numgp, scalar_interp_eigenval_[0]);
  eigenvect_rot_elast_pred_.resize(numgp, eigenvect_rot_elast_pred_[0]);
  rel_eigenvect_rot_plast_pred_.resize(numgp, rel_eigenvect_rot_plast_pred_[0]);
  rot_elast_pred_.resize(numgp, rot_elast_pred_[0]);
  rel_rot_plast_pred_.resize(numgp, rel_rot_plast_pred_[0]);

  resize_called_ = true;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PredictorInterpolator::pack(
    Core::Communication::PackBuffer& data) const
{
  Core::Communication::add_to_pack(data, eigenval_elast_pred_);
  Core::Communication::add_to_pack(data, eigenval_plast_pred_);
  Core::Communication::add_to_pack(data, scalar_interp_eigenval_);
  Core::Communication::add_to_pack(data, eigenvect_rot_elast_pred_);
  Core::Communication::add_to_pack(data, rel_eigenvect_rot_plast_pred_);
  Core::Communication::add_to_pack(data, rot_elast_pred_);
  Core::Communication::add_to_pack(data, rel_rot_plast_pred_);
  Core::Communication::add_to_pack(data, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PredictorInterpolator::unpack(
    Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_from_pack(buffer, eigenval_elast_pred_);
  Core::Communication::extract_from_pack(buffer, eigenval_plast_pred_);
  Core::Communication::extract_from_pack(buffer, scalar_interp_eigenval_);
  Core::Communication::extract_from_pack(buffer, eigenvect_rot_elast_pred_);
  Core::Communication::extract_from_pack(buffer, rel_eigenvect_rot_plast_pred_);
  Core::Communication::extract_from_pack(buffer, rot_elast_pred_);
  Core::Communication::extract_from_pack(buffer, rel_rot_plast_pred_);
  Core::Communication::extract_from_pack(buffer, resize_called_);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PredictorInterpolator::
    construct_prelim_plastic_pred(const unsigned int gp,
        const Core::LinAlg::Matrix<3, 3>& elastic_defgrad_elastic_pred,
        const AdaptiveEstimateInterpolationParams& aei_params)
{
  // consistency checks
  FOUR_C_ASSERT_ALWAYS(gp < eigenval_elast_pred_.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      eigenval_elast_pred_.size());

  // get elastic deformation gradient to be considered as elastic predictor
  Core::LinAlg::Matrix<3, 3> precond_elastic_defgrad_elastic_pred{elastic_defgrad_elastic_pred};
  if (aei_params.precondition_elastic_pred)
  {
    precondition_matrix(
        precond_elastic_defgrad_elastic_pred, aei_params.tol_precondition_elastic_pred);
  }

  //  perform polar-spectral decomposition of elastic defgrad within elastic predictor
  Core::LinAlg::Matrix<3, 3> material_stretch_elast_pred{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<3, 3> eigenval_elast_pred_temp{Core::LinAlg::Initialization::zero};
  std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3> spectral_pairs_elast_pred;
  Core::LinAlg::matrix_3x3_polar_decomposition(precond_elastic_defgrad_elastic_pred,
      rot_elast_pred_[gp], material_stretch_elast_pred, eigenval_elast_pred_temp,
      spectral_pairs_elast_pred);
  for (int i = 0; i < 3; ++i)
  {
    FOUR_C_ASSERT_ALWAYS(spectral_pairs_elast_pred[i].first >= 1.0e-8,
        "The eigenvalue {} of the elastic deformation gradient within the elastic predictor at "
        "GP "
        "{} is {}, "
        "such that its logarithm can not be computed!",
        i, gp, eigenval_elast_pred_[gp](i, i));
    eigenval_elast_pred_[gp](i, i) = spectral_pairs_elast_pred[i].first;
    for (int j = 0; j < 3; ++j)
    {
      eigenvect_rot_elast_pred_[gp](i, j) = spectral_pairs_elast_pred[i].second(j);
    }
  }

  // -->  construct a preliminary plastic predictor based on the parameter specifications
  Core::LinAlg::Matrix<4, 1> unit_quaternion{Core::LinAlg::Initialization::zero};
  unit_quaternion(3) = 1.0;

  // elastic stretch eigenvectors
  switch (aei_params.plastic_pred_elastic_stretch_eigenvect_type)
  {
    case PlasticPredictorElasticStretchEigenvectType::from_elastic_predictor:
    {
      rel_eigenvect_rot_plast_pred_[gp].update(1.0, unit_quaternion, 0.0);
      break;
    }
    default:
    {
      // other eigenvector rotation types not yet enabled; in case of multiple eigenvalues, a
      // canonicalization approach for the eigenvectors must be first implemented for the spectral
      // decomposition to avoid artificial rotation contributions
      FOUR_C_THROW("Elastic stretch eigenvector type {} not yet enabled for the plastic predictor",
          EnumTools::enum_name(aei_params.plastic_pred_elastic_stretch_eigenvect_type));
    }
  }

  // elastic rotation
  switch (aei_params.plastic_pred_elastic_rotation_type)
  {
    case PlasticPredictorElasticRotationType::from_elastic_predictor:
    {
      rel_rot_plast_pred_[gp].update(1.0, unit_quaternion, 0.0);
      break;
    }
    default:
    {
      FOUR_C_THROW("Elastic rotation type {} not yet enabled for the plastic predictor!",
          EnumTools::enum_name(aei_params.plastic_pred_elastic_rotation_type));
    }
  }

  // elastic stretch eigenvalues
  const double detF = precond_elastic_defgrad_elastic_pred.determinant();
  switch (aei_params.plastic_pred_elastic_stretch_eigenval_type)
  {
    case PlasticPredictorElasticStretchEigenvalType::scale_unit:
    {
      const double scaled_detF = std::pow(detF, 1.0 / 3.0);
      for (unsigned int i = 0; i < 3; ++i)
      {
        eigenval_plast_pred_[gp](i, i) = scaled_detF;
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("Elastic stretch eigenvalue type {} not yet enabled for the plastic predictor",
          EnumTools::enum_name(aei_params.plastic_pred_elastic_stretch_eigenval_type));
    }
  }

  // store eigenvalues in a form to be directly used within the interpolator
  scalar_interp_eigenval_[gp] = {{eigenval_elast_pred_[gp](0, 0), eigenval_elast_pred_[gp](1, 1),
                                     eigenval_elast_pred_[gp](2, 2)},
      {eigenval_plast_pred_[gp](0, 0), eigenval_plast_pred_[gp](1, 1),
          eigenval_plast_pred_[gp](2, 2)}};
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PredictorInterpolator::
    interpolate_elastic_defgrad_contributions(const unsigned int gp, const double interp_loc,
        std::vector<double>& interp_eigenval,
        Core::LinAlg::Matrix<4, 1>& interp_rel_eigenvect_rot_quat,
        Core::LinAlg::Matrix<4, 1>& interp_rel_rot_quat) const
{
  // consistency checks
  FOUR_C_ASSERT_ALWAYS(0.0 <= interp_loc && interp_loc <= 1.0,
      "Interpolation is constrained to the [0, 1] interval, with 0 specifying the elastic "
      "predictor and 1 specifying the plastic predictor! The current plastic predictor location "
      "is "
      "{}",
      interp_loc);
  FOUR_C_ASSERT_ALWAYS(gp < eigenval_elast_pred_.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      eigenval_elast_pred_.size());

  // auxiliaries
  Core::LinAlg::Matrix<4, 1> unit_quaternion{Core::LinAlg::Initialization::zero};
  unit_quaternion(3) = 1.0;

  Core::LinAlg::Matrix<1, 1> matrix_interp_loc{Core::LinAlg::Initialization::zero};
  matrix_interp_loc(0) = interp_loc;

  // interpolate eigenvalues
  interp_eigenval = eigenval_interpolator_.get_interpolated_scalar(
      scalar_interp_eigenval_[gp], ref_predictor_locs_, matrix_interp_loc);
  FOUR_C_ASSERT_ALWAYS(interp_eigenval.size() == 3,
      "The number of eigenvalues {} must actually be == 3", interp_eigenval.size());

  // interpolate quaternions
  interp_rel_eigenvect_rot_quat = Core::LinAlg::spherical_linear_interpolation(
      unit_quaternion, rel_eigenvect_rot_plast_pred_[gp], interp_loc);
  interp_rel_rot_quat = Core::LinAlg::spherical_linear_interpolation(
      unit_quaternion, rel_rot_plast_pred_[gp], interp_loc);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
    PredictorInterpolator::interpolate_elastic_defgrad(
        const unsigned int gp, const double interp_loc) const
{
  // consistency checks
  FOUR_C_ASSERT_ALWAYS(0.0 <= interp_loc && interp_loc <= 1.0,
      "Interpolation is constrained to the [0, 1] interval, with 0 specifying the elastic "
      "predictor and 1 specifying the plastic predictor! The current interpolation location is "
      "{}",
      interp_loc);
  FOUR_C_ASSERT_ALWAYS(gp < eigenval_elast_pred_.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      eigenval_elast_pred_.size());


  // interpolate contributions
  std::vector<double> interp_eigenval;
  Core::LinAlg::Matrix<4, 1> interp_rel_eigenvect_rot_quat{Core::LinAlg::Initialization::zero};
  Core::LinAlg::Matrix<4, 1> interp_rel_rot_quat{Core::LinAlg::Initialization::zero};
  interpolate_elastic_defgrad_contributions(
      gp, interp_loc, interp_eigenval, interp_rel_eigenvect_rot_quat, interp_rel_rot_quat);

  return compute_elast_defgrad_wrt_elast_predictor(interp_eigenval, eigenvect_rot_elast_pred_[gp],
      interp_rel_eigenvect_rot_quat, rot_elast_pred_[gp], interp_rel_rot_quat);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PredictorInterpolator::
    set_plastic_predictor_after_construction_algo(
        const unsigned int gp, const double plastic_pred_loc)
{
  // consistency checks
  FOUR_C_ASSERT_ALWAYS(0.0 <= plastic_pred_loc && plastic_pred_loc <= 1.0,
      "Interpolation is constrained to the [0, 1] interval, with 0 specifying the elastic "
      "predictor and 1 specifying the plastic predictor! The current plastic predictor location "
      "is "
      "{}",
      plastic_pred_loc);
  FOUR_C_ASSERT_ALWAYS(gp < eigenval_elast_pred_.size(),
      "Inconsistent Gauss point index {}, with set Gauss point size {}", gp,
      eigenval_elast_pred_.size());

  // set all quantities relevant for the plastic predictor
  std::vector<double> interp_eigenval;
  interpolate_elastic_defgrad_contributions(gp, plastic_pred_loc, interp_eigenval,
      rel_eigenvect_rot_plast_pred_[gp], rel_rot_plast_pred_[gp]);
  for (unsigned int i = 0; i < 3; ++i)
  {
    eigenval_plast_pred_[gp](i, i) = interp_eigenval[i];
  }
  scalar_interp_eigenval_[gp] = {{eigenval_elast_pred_[gp](0, 0), eigenval_elast_pred_[gp](1, 1),
                                     eigenval_elast_pred_[gp](2, 2)},
      {eigenval_plast_pred_[gp](0, 0), eigenval_plast_pred_[gp](1, 1),
          eigenval_plast_pred_[gp](2, 2)}};
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    AdaptiveEstimateInterpolationManager(const AdaptiveEstimateInterpolationParams& aei_params)
    : params_(aei_params), interp_point_container_(aei_params), predictor_interpolator_()
{
  // auxiliaries
  Core::LinAlg::Matrix<3, 3> unit3x3{Core::LinAlg::Initialization::zero};
  for (int i = 0; i < 3; ++i) unit3x3(i, i) = 1.0;


  // initialize class variables (for a single Gauss point for now)
  num_reestimations_ = 0;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    resize(const unsigned int num_gp)
{
  FOUR_C_ASSERT_ALWAYS(!resize_called_,
      "You already called resize for the adaptive estimate interpolation manager! You attempt to "
      "set it to {}",
      num_gp);

  interp_point_container_.resize(num_gp);
  predictor_interpolator_.resize(num_gp);

  resize_called_ = true;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    is_plastic_pred_construct_possible(const unsigned int gp)
{
  return (num_plastic_pred_construct_iters_ < params_.max_num_plastic_pred_construct_iters);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    is_estimate_interp_possible(const unsigned int gp)
{
  // check interpolation interval
  const double diff_bounds = interp_point_container_.upper_interp_bounds[gp] -
                             interp_point_container_.lower_interp_bounds[gp];
  bool check_min_interp_interval = (diff_bounds > params_.min_interp_interval);

  // check number of interpolation iterations
  bool check_interp_iters = (num_estimate_interp_iters_ < params_.max_num_estimate_interp_iters);

  return check_min_interp_interval && check_interp_iters;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
bool Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    is_reestimation_possible(const unsigned int gp)
{
  return (num_reestimations_ < params_.max_num_reestimations);
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    reset_and_construct_prelim_plastic_pred(
        const unsigned int gp, const LocalIntegrationDeformationTensors& deftensors)
{
  // reset certain variables
  num_plastic_pred_construct_iters_ = 0;
  num_estimate_interp_iters_ = 0;
  num_reestimations_ = 0;
  interp_point_container_.reset_bounds_and_current_interp_point(gp);



  // construct the preliminary predictor
  predictor_interpolator_.construct_prelim_plastic_pred(
      gp, deftensors.elastic_predictor_elastic_defgrad, params_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    pack(Core::Communication::PackBuffer& data) const
{
  interp_point_container_.pack(data);
  predictor_interpolator_.pack(data);
  Core::Communication::add_to_pack(data, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    unpack(Core::Communication::UnpackBuffer& buffer)
{
  interp_point_container_.unpack(buffer);
  predictor_interpolator_.unpack(buffer);
  Core::Communication::extract_from_pack(buffer, resize_called_);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
    AdaptiveEstimateInterpolationManager::interpolate_inverse_inelastic_defgrad(
        const unsigned int gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad)
{
  Core::LinAlg::Matrix<3, 3> interp_elastic_defgrad =
      predictor_interpolator_.interpolate_elastic_defgrad(
          gp, interp_point_container_.current_interp_points[gp]);

  Core::LinAlg::Matrix<3, 3> inv_inelastic_defgrad{Core::LinAlg::Initialization::zero};
  inv_inelastic_defgrad.multiply(1.0, inv_defgrad, interp_elastic_defgrad);

  return inv_inelastic_defgrad;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<3, 3> Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
    AdaptiveEstimateInterpolationManager::get_inverse_inelastic_defgrad_plastic_pred(
        const unsigned int gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad)
{
  // the plastic predictor lies at the location 1.0
  Core::LinAlg::Matrix<3, 3> interp_elastic_defgrad =
      predictor_interpolator_.interpolate_elastic_defgrad(gp, 1.0);

  Core::LinAlg::Matrix<3, 3> inv_inelastic_defgrad{Core::LinAlg::Initialization::zero};
  inv_inelastic_defgrad.multiply(1.0, inv_defgrad, interp_elastic_defgrad);

  return inv_inelastic_defgrad;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    set_plastic_predictor_after_construction_algo(const unsigned int gp)
{
  // set the plastic predictor quantities
  predictor_interpolator_.set_plastic_predictor_after_construction_algo(
      gp, interp_point_container_.current_interp_points[gp]);

  // reset the interpolation point container
  interp_point_container_.reset_bounds_and_current_interp_point(gp);
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
void Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolationManager::
    adapt_interpolation_interval_and_point(const unsigned int gp, const ErrorType& eval_err_type)
{
  FOUR_C_ASSERT_ALWAYS(eval_err_type != ErrorType::no_errors,
      "You should not call this adaptation routine for error_type {}",
      EnumTools::enum_name(ErrorType::no_errors));

  // shift interpolation interval
  InterpolationShiftAction interp_shift_action = get_interpolation_shift_action(eval_err_type);
  switch (interp_shift_action)
  {
    case InterpolationShiftAction::shift_towards_elastic_pred:
    {
      interp_point_container_.upper_interp_bounds[gp] =
          interp_point_container_.current_interp_points[gp];
      break;
    }
    case InterpolationShiftAction::shift_towards_plastic_pred:
    {
      interp_point_container_.lower_interp_bounds[gp] =
          interp_point_container_.current_interp_points[gp];
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "You should not be here in the interpolation routine! The shift action {} is not "
          "supported!",
          EnumTools::enum_name(interp_shift_action));
    }
  }

  // reset interpolation point
  set_current_interp_point(gp, CurrentInterpPointPreset::standard);
}


double Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::
    AdaptiveEstimateInterpolationManager::calculate_optimal_equiv_stress_interp_point(
        const OptimalEquivStressStartingPointInput& optimal_equiv_stress_input) const
{
  // set to elastic predictor if the stress of the elastic predictor is numerically 0.0 ->
  // this is theoretically
  // possible for viscoplastic laws without yield surfaces, which may have plastic flow
  // even in this case; however, the determination of the optimal point requires dividing
  // over this stress, which will not be possible in this specific case.
  // Same goes for the case where the elastic predictor and the plastic predictor are
  // associated with effectively the same stress value
  // -> set starting
  // point associated with the elastic predictor
  if (optimal_equiv_stress_input.equiv_stress_elast_pred <= 1.0e-12 ||
      std::abs(optimal_equiv_stress_input.equiv_stress_plast_pred -
               optimal_equiv_stress_input.equiv_stress_elast_pred) /
              optimal_equiv_stress_input.equiv_stress_elast_pred <
          1.0e-8)
  {
    return 0.0;
  }


  // compute optimal interpolation point based on the equivalent stress: we clamp between
  // 0.0 and 1.0 because in some special cases such as stress relaxation, the starting
  // point may be slightly under 0.0 or over 1.0 (machine precision)
  return std::clamp((optimal_equiv_stress_input.equiv_stress_solution -
                        optimal_equiv_stress_input.equiv_stress_elast_pred) /
                        (optimal_equiv_stress_input.equiv_stress_plast_pred -
                            optimal_equiv_stress_input.equiv_stress_elast_pred),
      0.0, 1.0);
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
  local_newton_data_.has_error.resize(num_gp, std::vector<bool>{});
  local_newton_data_.residual_norms.resize(num_gp, std::vector<double>{});
  local_newton_data_.increment_norms.resize(num_gp, std::vector<double>{});
  local_newton_data_.is_converged.resize(num_gp, std::vector<bool>{});
  if (use_adaptive_estimate_interpolation)
  {
    local_newton_data_.current_interpolation_points = std::vector<std::vector<double>>(num_gp);
  }
  local_newton_data_.equiv_stresses.resize(num_gp, std::vector<double>{});
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
    adaptive_estimate_interp_data_->interp_point_optimal_equiv_stress.resize(num_gp, 0.0);
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
  local_newton_data_.has_error[gp] = {};
  local_newton_data_.residual_norms[gp] = {};
  local_newton_data_.increment_norms[gp] = {};
  local_newton_data_.is_converged[gp] = {};
  if (local_newton_data_.current_interpolation_points.has_value())
    local_newton_data_.current_interpolation_points->at(gp) = {};
  local_newton_data_.equiv_stresses[gp] = {};
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
    adaptive_estimate_interp_data_->interp_point_optimal_equiv_stress[gp] = -1.0;
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
                .interp_point_optimal_equiv_stress =
                    adaptive_estimate_interp_data_->interp_point_optimal_equiv_stress[gp],
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
                .interp_point_optimal_equiv_stress = -1.0,
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
      .constitutive_update_time = std::reduce(constitutive_update_time_.begin(), constitutive_update_time_.end()),
      .total_constitutive_update_time =
          std::reduce(total_constitutive_update_time_.begin(), total_constitutive_update_time_.end()),
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
      .has_error = local_newton_data_.has_error[gp],
      .residual_norms = local_newton_data_.residual_norms[gp],
      .increment_norms = local_newton_data_.increment_norms[gp],
      .is_converged = local_newton_data_.is_converged[gp],
      .current_interpolation_points =
          local_newton_data_.current_interpolation_points.has_value()
              ? std::make_optional(local_newton_data_.current_interpolation_points->at(gp))
              : std::nullopt,
      .equiv_stresses = local_newton_data_.equiv_stresses[gp],
      .plastic_strain_increments = local_newton_data_.plastic_strain_increments[gp],
  };

  write_lnl_data_to_csv(
      csv_lnl_data_writer_at_gp_[gp].value(), csv_writing_lnl_data, tracking_settings_);
}


FOUR_C_NAMESPACE_CLOSE
