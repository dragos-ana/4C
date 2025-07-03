// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP

#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <magic_enum/magic_enum.hpp>

#include <map>
#include <ostream>
#include <string>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /// namespace: utilities for
  /// InelasticDefgradTransvIsotropElastViscoplast
  namespace InelasticDefgradTransvIsotropElastViscoplastUtils
  {
    /// enum class for error types in InelasticDefgradTransvIsotropElastViscoplast, used for
    /// triggering different procedures (e.g. repredictorization,
    /// substepping, line search) during the
    /// Local Newton Loop
    enum class ErrorType
    {
      no_errors,                ///< no errors
      negative_plastic_strain,  ///< negative plastic strain which does not allow for evaluations
                                ///< inside the viscoplasticity laws
      overflow_error,  ///< overflow error of the term \f$ \Delta t \dot{\varepsilon}^{\text{p}} \f$
                       ///< (and \f$ \mathsymbol{E}^{\text{p}}  = \exp(- \Delta t
                       ///< \dot{\varepsilon}^{\text{p}} \mathsymbol{N}^{\text{p}}) \f$)
      no_flow_resistance,            ///< the material has no flow resistance anymore, such that the
                                     ///< evaluations model non-physical phenomena
      no_plastic_incompressibility,  ///< no plastic incompressibility, meaning that the determinant
                                     ///< of the inelastic defgrad is far from 1
      failed_solution_linear_system_lnl,  ///< solution of the linear system in the Local
                                          ///< Newton-Raphson Loop failed
      failed_determ_line_search_step,     ///< the computation of a suitable line search step failed
      no_convergence_local_newton,  ///< the Local Newton Loop did not converge for the given loop
                                    ///< settings
      singular_jacobian,  ///< singular Jacobian after converged LNL, which does not enable our
                          ///< analytical evaluation of the linearization
      failed_solution_analytic_linearization,  ///< solution of the linear system in the analytical
                                               ///< linearization failed
      failed_computation_flow_resistance,  ///< failed in the computation of the flow resistance via
                                           ///< time integration of the hardening-rate equation
                                           ///(e.g., when using the Anand law)
      failed_computation_flow_resistance_derivs,  ///< failed in the computation of the flow
                                                  ///< resistance derivatives (e.g., when using the
                                                  ///< Anand law)
      failed_matrix_log_evaluation,   ///< failed evaluation of the matrix logarithm or its
                                      ///< derivative
      failed_matrix_exp_evaluation,   ///< failed evaluation of the matrix exponential or its
                                      ///< derivative
      failed_right_cg_interpolation,  ///< failed interpolation of the right Cauchy-Green tensor
      under_yield_surface  ///< mechanical state is "under" the yield surface, i.e., the evaluated
                           ///< stress is smaller than the yield stress, which should not occur
                           ///< in the Local Newton loop
    };


    /// enum class for error management actions in the iterations of the
    // Local Newton loop
    enum class ErrorAction
    {
      continue_iteration,           ///< continue iteration without any errors (NoErrors)
      return_solution_with_errors,  ///< return the current solution with errors (if the current
                                    ///< simulation settings cannot lead to a solution)
      next_iteration,               ///< go to next iteration after performing certain reset steps
    };

    /// convert error type to detailed error message
    std::string get_detailed_error_message_for_error_type(ErrorType err_type);

    /// enum class: success status of single Local Newton Loop
    /// iterations to be tracked in the analysis utilities
    enum class LocalIterationStatus
    {
      residual_evaluation_successful,  // residual could be evaluated without errors
      residual_evaluation_failed,      // residual evaluation failed
      converged,                       // Local Newton loop converged in this iteration
      not_evaluated,  // the iteration has not yet been evaluated (also the case when a previous
                      // iteration has already converged)
      final_error,    // the LNL has finally failed after performing all possible error management
                      // actions or/and after the maximum number of
                      // iterations was reached
    };

    /// convert enum for the success status of single Local
    /// Newton iterations to double (required for Gauss-Point output,
    /// which needs to be of type <double>)
    double local_iteration_status_enum_to_double(const LocalIterationStatus iter_status);

    /// enum class for material behavior types
    enum class MatBehavior
    {
      isotrop,         ///< isotropic material behavior
      transv_isotrop,  ///< transversely isotropic material behavior
    };

    /// enum class for time integration types (Local Newton integration)
    enum class TimIntType
    {
      standard,     ///< standard time integration,
      logarithmic,  ///< time integration with logarithmically transformed residual equation for the
                    ///< evolution of the plastic deformation gradient
    };

    /// enum class for material linearization types
    enum class LinearizationType
    {
      analytic,  ///< analytical linearization involving the solution of a linear system of
                 ///< equations,
      perturbation_based,  ///< linearization based on perturbing the current state
    };


    /// class containing utilities for general analysis of the material
    /// time integration (including predictor adaptation, Local Newton
    /// loop, line search):
    /// error types, number of line searches, ... Currently only
    /// employed for single-element single-processor simulations.
    class GeneralLocalTimIntAnalysisUtils
    {
     public:
      //! number of LNL steps for the current timestep evaluation (LNL)
      unsigned int eval_num_of_LNL_steps_ = 0;

      //! total number of steps
      unsigned int total_num_of_LNL_steps_ = 0;

      //! number of iterations for the current timestep evaluation (LNL)
      unsigned int eval_num_of_iters_ = 0;

      //! total number of LNL iterations
      unsigned int total_num_of_iters_ = 0;

      //! number of repredictorizations for the current timestep evaluation (LNL)
      unsigned int eval_num_of_repredict_ = 0;

      //! total number of LNL repredictorizations
      unsigned int total_num_of_repredict_ = 0;

      //! number of iterations spent in the predictor adaptation for the
      //! current timestep evaluation (LNL)
      unsigned int eval_num_of_pred_adapt_iters_ = 0;

      //! total number of iterations spent in the predictor adaptation
      unsigned int total_num_of_pred_adapt_iters_ = 0;

      //! number of iterations spent in the predictor adaptation for the
      //! current timestep evaluation (LNL), in the specific case of repredictorization
      unsigned int eval_num_of_repredict_iters_ = 0;

      //! total number of iterations spent in the predictor adaptation,
      //! in the specific case of repredictorization
      unsigned int total_num_of_repredict_iters_ = 0;

      //! number of line searches for the current timestep evaluation (LNL)
      unsigned int eval_num_of_line_search_ = 0;

      //! line search: the number of times the step size \f$ \alpha \f$ of
      //! the last iteration (Local Newton Loop) deviates from 1.0
      //! (currently evaluated time step)
      unsigned int eval_num_of_alpha_neq_1_last_iter = 0;

      //! line search: the number of times the step size \f$ \alpha \f$
      //! deviates from 1.0 in all iterations of the Local Newton Loop
      //! (currently evaluated time step)
      unsigned int eval_num_of_alpha_neq_1 = 0;

      //! total number of LNL line searches
      unsigned int total_num_of_line_search_ = 0;

      //! line search: total number of times the step size \f$ \alpha \f$ of
      //! the last iteration (Local Newton Loop) deviates from 1.0
      unsigned int total_num_of_alpha_neq_1_last_iter = 0;

      //! line search: the number of times the step size \f$ \alpha \f$
      //! deviates from 1.0 in all iterations of the Local Newton Loop
      //! (currently evaluated time step)
      unsigned int total_num_of_alpha_neq_1 = 0;

      //! number of iterations of the line searches for the current timestep evaluation (LNL)
      unsigned int eval_num_of_line_search_iters_ = 0;

      //! total number of line search iterations
      unsigned int total_num_of_line_search_iters_ = 0;

      //! number of times the LNL convergences directly in its first
      //! iteration (due to a good predictor!) for the current timestep evaluation
      unsigned int eval_num_of_first_iter_convergences = 0;

      //! total number of times the LNL convergences directly in its first
      //! iteration (due to a good predictor!)
      unsigned int total_num_of_first_iter_convergences = 0;

      //! predictor interpolation factor obtained from the predictor
      //! adaptation routine (for set GP, current
      //! time step, last global iteration)
      double curr_pred_interp_factor_ = 0;

      //! predictor interpolation factor obtained from the predictor
      //! adaptation routine (for set GP, current
      //! time step, maximum over all global iterations)
      double curr_max_pred_interp_factor_ = 0;

      //! optimal predictor interpolation factor obtained from the time
      //! step solution (for GP 0 of element 0 after the current
      //! time step)
      double optimal_pred_interp_factor_ = 0;

      //! Local Newton residual obtained from the optimal predictor interpolation factor
      double lnl_res_optimal_pred_interp_factor_ = 0;

      //! timer for the current timestep evaluation, from the start of preevaluate to the end of
      //! update
      Teuchos::Time eval_teuchos_timer_{
          "InelasticDefgradTransvIsotropElastViscoplast::from_preevaluate_to_update"};

      //! timer for the time spent in the Local Newton loop
      Teuchos::Time eval_teuchos_timer_LNL_{
          "InelasticDefgradTransvIsotropElastViscoplast::time spent in the LNL"};

      //! timer for the time spent adapting the predictor (including repredictorization)
      Teuchos::Time eval_teuchos_timer_pred_adapt_{
          "InelasticDefgradTransvIsotropElastViscoplast::time spent in the predictor adaptation"};

      //! timer for the time spent adapting the predictor only in the
      //! specific case of repredictorization
      Teuchos::Time eval_teuchos_timer_repredict_{
          "InelasticDefgradTransvIsotropElastViscoplast::time spent in the predictor adaptation "
          "(repredictorization)"};

      //! timer for the time spent in the line search scheme
      Teuchos::Time eval_teuchos_timer_line_search_{
          "InelasticDefgradTransvIsotropElastViscoplast::time spent in the line search"};

      //! evaluation time for the current time step, from the start of
      //! preevaluate to the end of update
      double eval_time_;

      //! total evaluation time, from the start of
      //! preevaluate to the end of update
      double total_time_;

      //! evaluation time spent in the Local Newton loop (current
      //! time step)
      double eval_time_LNL_;

      //! total time spent in the Local Newton Loop
      double total_time_LNL_;

      //! evaluation time spent in the predictor adaptation (current
      //! time step)
      double eval_time_pred_adapt_;

      //! total time spent in the predictor adaptation
      double total_time_pred_adapt_;

      //! evaluation time spent in the predictor adaptation (current
      //! time step), in the specific case of repredictorization
      double eval_time_repredict_;

      //! total time spent in the predictor adaptation, in the specific
      //! case of repredictorization
      double total_time_repredict_;

      //! evaluation time spent in the line search (current
      //! time step)
      double eval_time_line_search_;

      //! total time spent in the line search
      double total_time_line_search_;

      //! error map (how many times an error occurs) of the current timestep evaluation, from the
      //! first preevaluate of this time step to the first preevaluate of the next
      std::map<ErrorType, unsigned int> eval_error_map_ = {
          {ErrorType::negative_plastic_strain, 0},
          {ErrorType::overflow_error, 0},
          {ErrorType::no_plastic_incompressibility, 0},
          {ErrorType::failed_solution_linear_system_lnl, 0},
          {ErrorType::failed_determ_line_search_step, 0},
          {ErrorType::no_convergence_local_newton, 0},
          {ErrorType::singular_jacobian, 0},
          {ErrorType::failed_solution_analytic_linearization, 0},
          {ErrorType::failed_matrix_log_evaluation, 0},
          {ErrorType::failed_matrix_exp_evaluation, 0},
          {ErrorType::under_yield_surface, 0},
      };

      //! error map (how many times an error occurs) of the total evaluation
      std::map<ErrorType, unsigned int> total_error_map_ = {
          {ErrorType::negative_plastic_strain, 0},
          {ErrorType::overflow_error, 0},
          {ErrorType::no_plastic_incompressibility, 0},
          {ErrorType::failed_solution_linear_system_lnl, 0},
          {ErrorType::failed_determ_line_search_step, 0},
          {ErrorType::no_convergence_local_newton, 0},
          {ErrorType::singular_jacobian, 0},
          {ErrorType::failed_solution_analytic_linearization, 0},
          {ErrorType::failed_matrix_log_evaluation, 0},
          {ErrorType::failed_matrix_exp_evaluation, 0},
          {ErrorType::failed_right_cg_interpolation, 0},
          {ErrorType::under_yield_surface, 0},
      };

      //! simulation time instant
      double sim_time_ = 0.0;
      //! simulation time step index
      int sim_timestep_ = 0;

      //! was the pre_evaluate method of the first element called?
      bool pre_eval_called_ = false;

      //! how often was the update method called? (maximum:
      //! num_of_global_elements, if only one processor
      //! is considered)
      int num_update_calls_ = 0;

      //! reset method: reset the stored internal variables for a new
      //! evaluation / new timestep
      void reset();

      //! update total values based on the evaluated values
      void update_total();

      //! write the stored internal variables to csv
      void write_to_csv();

      //! output routine of the csv writer in the case of an error
      //! during the Local Newton Loop routine
      void output_error_local_newton_loop(unsigned int step_counter);
    };

    /// enum class for state quantity evaluations in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// plastic strain rate,...)
    enum class StateQuantityEvalType
    {
      FullEval,  ///< full evaluation (full call of the evaluate_state_quantities method)
      PlasticStrainRateOnly,  ///< return in evaluate_state_quantities once the plastic strain rate
                              ///< has been evaluated
    };

    /// enum class for evaluations of the state quantity derivatives in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// derivatives of the plastic strain rate,...)
    enum class StateQuantityDerivEvalType
    {
      FullEval,  ///< full evaluation (full call of the evaluate_state_quantity_derivatives method)
      PlasticStrainRateDerivsOnly,  ///< return in evaluate_state_quantity_derivatives once the
                                    ///< derivatives of the plastic strain rate have been evaluated
    };

    //! struct holding relevant tracking data when writing to csv
    //! files
    struct CSVOutputTrackingData
    {
      //! global element id
      const int ele_gid;

      //! Gauss point index
      const int gp;

      //! time instant \f$t_{n}\f$
      const double tn;

      //! time instant \f$t_{n+1}\f$
      const double tnp;

      //! tracker for the global iteration (if we have output every
      //! iteration) or the timestep index; increased by 1 every time
      //! the Gauss point output routine / or the update method (if no
      //! Gauss point output is considered) is called
      unsigned int globiter_or_timestep_index_;

      //! tracker for the local NR iteration
      unsigned int lnl_iter;
    };

    //! struct holding the relevant output data of all
    //! microiterations of a single predictor adaptation which can be
    //! written to a csv file
    struct CSVOutputPredAdaptMicroIterData
    {
      /*! @brief Constructor.
       *
       * @param[in] csv_output_tracking_data tracking data used to
       * specify the settings for the csv output.
       */
      CSVOutputPredAdaptMicroIterData(const CSVOutputTrackingData csv_output_tracking_data);

      //! data collector for a single micro iteration within the
      //! predictor adaptation -> used to assign the values at the specific microiterations
      struct MicroIterDataCollector
      {
        //! current interpolation factor \f$ \xi \f$
        double current_xi_ = -1;
        //! current equivalent stress
        double current_equiv_stress_ = -1;
        //! current plastic strain
        double current_plastic_strain_ = -1;
        //! current error status
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType current_error_status_ =
            ErrorType::overflow_error;
      };

      //! all indices of the microiterations (iterations within the predictor
      //! adaptation)
      std::vector<unsigned int> all_microiter_;

      //! current interpolation factors \f$ \xi \f$ of all
      //! microiterations
      std::vector<double> all_current_xi_;

      //! current equivalent stresses of all microiterations (associated
      //! with the current interpolation factors)
      std::vector<double> all_current_equiv_stress_;

      //! current plastic strains of all microiterations (associated
      //! with the current interpolation factors)
      std::vector<double> all_current_plastic_strain_;

      //! current error status of all microiterations (associated with
      //! the current interpolation factors)
      std::vector<ErrorType> all_current_error_status_;

      //! tracking data used to specify the settings for csv output
      const CSVOutputTrackingData csv_output_tracking_data_;

      //! append collected data for specific microiteration
      void append_micro_iter_data(
          const MicroIterDataCollector mi_data_collector, const unsigned micro_iter);

      //! writes data from each microiteration of a single predictor
      //! adaptation (specified via tracking data) to a dedicated csv file
      void write_pred_adapt_micro_iter_data_to_csv();
    };

    //! struct holding the relevant output data of all
    //! microiterations of a single line search which can be
    //! written to a csv file
    struct CSVOutputLineSearchMicroIterData
    {
      /*!
       * @param[in] csv_output_tracking_data tracking data used to
       * specify the settings for the csv output.
       */
      CSVOutputLineSearchMicroIterData(const CSVOutputTrackingData csv_output_tracking_data);

      //! data collector for a single micro iteration within the
      //! predictor adaptation -> assigns the values at the specific microiterations
      struct MicroIterDataCollector
      {
        //! current step size \f$ \alpha \f$
        double current_alpha_ = -1;
        //! maximum allowed step size \f$ \alpha_{\mathrm{max}} \f$
        //! accounting for eventual errors
        double max_alpha_ = -1;
        //! current equivalent stress
        double current_equiv_stress_ = -1;
        //! current plastic strain
        double current_plastic_strain_ = -1;
        //! current quadratic residual norm for the current step size
        double current_quadratic_residual_norm_ = -1;
        //! maximum allowed quadratic residual norm for the current step size
        double max_quadratic_residual_norm_ = -1;
        //! current error status
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType current_error_status_ =
            ErrorType::overflow_error;
      };

      //! all indices of the microiterations (iterations within the line search)
      std::vector<unsigned int> all_microiter_;

      //! current step sizes \f$ \alpha \f$ of all
      //! microiterations
      std::vector<double> all_current_alpha_;

      //! maximum step sizes \f$ \alpha_{\mathrm{}} \f$ of all
      //! microiterations
      std::vector<double> all_max_alpha_;

      //! current equivalent stresses of all microiterations (associated
      //! with the current line search step sizes)
      std::vector<double> all_current_equiv_stress_;

      //! current plastic strains of all microiterations (associated
      //! with the current line search step sizes)
      std::vector<double> all_current_plastic_strain_;

      //! current quadratic residual norm of all microiterations (associated
      //! with the current line search step sizes)
      std::vector<double> all_current_quadratic_residual_norm_;

      //! maximum allowed quadratic residual norm of all microiterations (associated
      //! with the current line search step sizes)
      std::vector<double> all_max_quadratic_residual_norm_;

      //! current error status of all microiterations (associated with
      //! the current line search ste sizes)
      std::vector<ErrorType> all_current_error_status_;

      //! append collected data for specific microiteration
      void append_micro_iter_data(
          const MicroIterDataCollector mi_data_collector, const unsigned micro_iter);

      //! tracking data used to specify the settings for csv output
      const CSVOutputTrackingData csv_output_tracking_data_;

      //! writes data from each microiteration of a single line search (specified via tracking data)
      //! to a dedicated csv file
      void write_line_search_micro_iter_data_to_csv();
    };

  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
