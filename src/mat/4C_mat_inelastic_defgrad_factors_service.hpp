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
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <magic_enum/magic_enum.hpp>

#include <array>
#include <map>
#include <optional>
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

    //! matrix exponential and logarithm evaluation utilities
    struct MatrixExpLogUtils
    {
      //! Pade approximation order (to be used consistently: the
      //! derivative of the matrix functions should use the same Pade
      //! order as the evaluation of the matrix functions)
      unsigned int pade_order_ = 16;  // by default we set the highest order currently implemented
    };

    //! struct containing time step settings and time trackers
    struct TimeStepTracker
    {
      //! time step length
      double dt_;
      //! currently computed time instant \f$ t_{n+1} \f$
      double tnp_;
      //! minimum substep length
      double min_dt_;
    };


    //! struct containing quantities at the last and current time points (i.e., at \f[ t_n \f] and
    //! \f[ t_{n+1} \f], respectively). The quantities are tracked at all Gauss points, in order to
    //! update them simultaneously during the update method call
    struct TimeStepQuantities
    {
      //! right Cauchy-Green deformation tensor at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_rightCG_;

      //! inverse plastic deformation gradient at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_plastic_defgrad_inverse_;

      //! spatial stretch of the plastic deformation gradient
      //! at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_plastic_defgrad_spatial_stretch_;

      //! rotation of the inverse plastic deformation gradient
      //! at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_plastic_defgrad_inverse_rot_;

      //! (equivalent) plastic strain at the last time step (for all Gauss points)
      std::vector<double> last_plastic_strain_;

      //! last (reduced) deformation gradient: used to in the predictor
      //! adaptation routine
      std::vector<Core::LinAlg::Matrix<3, 3>> last_defgrad_;

      //! temporary variable, for which we store the right Cauchy-Green deformation tensor at each
      //! evaluation (used in order to update last_rightCG_ once outer NR converges) (for all Gauss
      //! points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_rightCG_;

      //! current (reduced) deformation gradient: used to check whether the inverse inelastic
      //! deformation gradient has already been evaluated (to improve the computation performance)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_defgrad_;


      //! current inverse plastic deformation gradient (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_plastic_defgrad_inverse_;

      //! current plastic strain (for all Gauss points)
      std::vector<double> current_plastic_strain_;

      //! current equivalent stress (for all Gauss points)
      std::vector<double> current_stress_;

      //! inverse plastic deformation gradient at the last computed time instant (after the last
      //! converged substep)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_substep_plastic_defgrad_inverse_;
      //! plastic strain at the last computed time instant (after the last converged substep)
      std::vector<double> last_substep_plastic_strain_;
    };



    /// struct: constant non-material tensors, such as different
    /// identity tensors
    struct ConstNonMatTensors
    {
      static const ConstNonMatTensors& instance()
      {
        static ConstNonMatTensors instance;
        return instance;
      }

      // constructor
      ConstNonMatTensors();
      // second-order 3x3 identity tensor in matrix form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<3, 3> id3x3_{Core::LinAlg::Initialization::zero};
      // second-order 3x3 identity in Voigt stress form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 1> id6x1_{Core::LinAlg::Initialization::zero};
      // symmetric identity four tensor of dimension 3 \f$ \mathbb{I}_\text{S} \f$
      Core::LinAlg::Matrix<6, 6> id4_6x6_{Core::LinAlg::Initialization::zero};
      // deviatoric operator \f$ \mathbb{P}_{\text{dev}}  =  \mathbb{I}_\text{S} -
      // \frac{1}{3} \boldsymbol{I} \otimes \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 6> dev_op_{Core::LinAlg::Initialization::zero};
      // identity fourth-order tensor in Voigt notation: delta_AC delta_BD in index notation
      Core::LinAlg::Matrix<9, 9> id4_9x9_{Core::LinAlg::Initialization::zero};
      // second-order 10x10 identity tensor in matrix form
      Core::LinAlg::Matrix<10, 10> id10x10_{Core::LinAlg::Initialization::zero};
    };



    //! struct containing constant tensors which depend on the constant fiber direction \f$
    //! \boldsymbol{m} \f$
    struct ConstMatTensors
    {
      //! \f$ \boldsymbol{I} + \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> id_plus_mm_;
      //! \f$ \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> mm_{Core::LinAlg::Initialization::zero};
      //! deviatoric part \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //! \right)_\text{dev}\f$
      Core::LinAlg::Matrix<3, 3> mm_dev_{Core::LinAlg::Initialization::zero};
      //! \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right) \otimes \left( \boldsymbol{m}
      //! \otimes \boldsymbol{m} \right) \f$ (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dyad_mm_{Core::LinAlg::Initialization::zero};
      //!  \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right)_\text{dev} \otimes \left(
      //!  \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dev_dyad_mm_{Core::LinAlg::Initialization::zero};
      //!  \f$ \boldsymbol{I} \otimes \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> id_dyad_mm_;

      //! set tensors for a given fiber direction \f$ \boldsymbol{m} \f$
      void set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m);
    };


    //! class containing utilities for predictor interpolation
    struct PredictorAdaptationUtils
    {
      //! first eigenvalue \f$ \lambda_1 \f$ of the inverse plastic deformation
      //! gradient inside the elastic predictor (saved for all Gauss points), utilized as reference
      //! point for interpolation
      std::vector<double> lambda_1_elast_pred_;

      //! first eigenvalue \f$ \lambda_1 \f$ of the inverse plastic deformation
      //! gradient inside the plastic predictor (saved for all Gauss points), utilized as reference
      //! point for interpolation
      std::vector<double> lambda_1_plast_pred_;

      //! second eigenvalue \f$ \lambda_2 \f$ of the inverse plastic deformation
      //! gradient inside the elastic predictor (saved for all Gauss points), utilized as reference
      //! point for interpolation
      std::vector<double> lambda_2_elast_pred_;

      //! second eigenvalue \f$ \lambda_2 \f$ of the inverse plastic deformation
      //! gradient inside the plastic predictor (saved for all Gauss points), utilized as reference
      //! point for interpolation
      std::vector<double> lambda_2_plast_pred_;

      //! relative rotation vector associated with the eigenvector (rotation)
      //! matrix \f$ \boldsymbol{Q} \f$ for the inverse plastic deformation
      //! gradient inside the plastic predictor (saved for all Gauss
      //! points); relative with respect to the eigenvector matrix
      //! inside the elastic predictor
      std::vector<Core::LinAlg::Matrix<3, 1>> rel_eigenvect_rot_vect_plast_pred_;


      //! eigenvector (rotation)
      //! matrix \f$ \boldsymbol{Q} \f$ for the inverse plastic deformation
      //! gradient inside the elastic predictor (saved for all Gauss
      //! points)
      std::vector<Core::LinAlg::Matrix<3, 3>> eigenvect_rot_matrix_elast_pred_;


      //! rotation matrix \f$ \boldsymbol{R} \f$ for the inverse plastic deformation
      //! gradient inside the elastic predictor AND the plastic
      //! predictor (saved for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> rot_matrix_;

      //! spectral pairs of the elastic predictor, already accounting
      //! for multiple eigenvalues
      std::vector<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>>
          spectral_pairs_elast_pred_;

      //! interval scanning parameter set by the user \f$ k_{\mathrm{scan}} \f$
      const double k_scan_;

      //! interpolation factor \f$ \xi_{\lambda_1} \f$ for the
      //! eigenvalue \f$\lambda_1\f$ (saved for all GP) for the
      //! current evaluation (current time step, current global iteration)
      std::vector<double> current_xi_lambda_1_;

      //! interpolation factor \f$ \xi_{\lambda_2} \f$ for the
      //! eigenvalue \f$\lambda_2\f$ (saved for all GP) for the
      //! current evaluation (current time step, current global iteration)
      std::vector<double> current_xi_lambda_2_;

      //! interpolation factors \f$ \xi_{\boldsymbol{Q}} \f$ for the
      //! rotation vector associated with the eigenvector (rotation) matrix \f$\boldsymbol{Q}\f$
      //! (saved for all GP) for the current evaluation (current time step, current global
      //! iteration)
      std::vector<std::array<double, 3>> current_xi_eigenvect_rot_;

      //! maximum interpolation factor \f$ \xi_{\lambda_1, \mathrm{max}}
      //! \f$ for the eigenvalue \f$\ lambda_1 \f$ (saved for all GP)
      //! current evaluation (maximum over current time step)
      std::vector<double> current_max_xi_lambda_1_;

      //! maximum interpolation factor \f$ \xi_{\lambda_2, \mathrm{max}}
      //! \f$ for the eigenvalue \f$ \lambda_2 \f$ (saved for all GP)
      //! current evaluation (maximum over current time step)
      std::vector<double> current_max_xi_lambda_2_;

      //! maximum interpolation factors \f$ \xi_{\boldsymbol{Q}, \mathrm{max}}
      //! \f$ for the rotation vector associatedwith the eigenvector
      //! (rotation) matrix \f$ \boldsymbol{Q} \f$ (saved for all GP)
      //! current evaluation (maximum over current time step)
      std::vector<std::array<double, 3>> current_max_xi_eigenvect_rot_;

      //! interpolation factor \f$ \xi_{\lambda_1,n} \f$ for the
      //! eigenvalue \f$ \lambda_1 \f$ (saved for all GP)
      //! evaluated during the last global iteration of the previous
      //! time step (previous time step, last global iteration)
      std::vector<double> last_xi_lambda_1_;

      //! interpolation factor \f$ \xi_{\lambda_2,n} \f$ for the
      //! eigenvalue \f$ \lambda_1 \f$ (saved for all GP)
      //! evaluated during the last global iteration of the previous
      //! time step (previous time step, last global iteration)
      std::vector<double> last_xi_lambda_2_;

      //! interpolation factors \f$ \xi_{\boldsymbol{Q},n} \f$ for the
      //! rotation vector associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$
      //! (saved for all GP) evaluated during the last global iteration of the previous time step
      //! (previous time step, last global iteration)
      std::vector<std::array<double, 3>> last_xi_eigenvect_rot_;

      //! maximum interpolation factor \f$
      //! \xi_{\lambda_1,n,\mathrm{max}} \f$ for the eigenvalue \f$ \lambda_1 \f$
      //! (saved for all GP) evaluated during the last time step
      //! (maximum over previous time step)
      std::vector<double> last_max_xi_lambda_1_;

      //! maximum interpolation factor \f$
      //! \xi_{\lambda_2,n,\mathrm{max}} \f$ for the eigenvalue \f$ \lambda_2 \f$
      //! (saved for all GP) evaluated during the last time step
      //! (maximum over previous time step)
      std::vector<double> last_max_xi_lambda_2_;

      //! maximum interpolation factors \f$
      //! \xi_{\boldsymbol{Q},n,\mathrm{max}} \f$ for the rotation
      //! vector associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$
      //! (saved for all GP) evaluated during the last time step
      //! (maximum over previous time step)
      std::vector<std::array<double, 3>> last_max_xi_eigenvect_rot_;


      //! optimal interpolation factor \f$ \xi_{\lambda_1, n,
      //! \mathrm{optimal}} \f$ for the eigenvalue \f$ \lambda_1 \f$
      //! (saved for all GP) from the previous time step (determined
      //! such that it leads to the previous LNL solution at the
      //! considered GP)
      std::vector<double> optimal_xi_lambda_1_;

      //! optimal interpolation factor \f$ \xi_{\lambda_2, n,
      //! \mathrm{optimal}} \f$ for the eigenvalue \f$ \lambda_2 \f$
      //! (saved for all GP) from the previous time step (determined
      //! such that it leads to the previous LNL solution at the
      //! considered GP)
      std::vector<double> optimal_xi_lambda_2_;

      //! optimal interpolation factors \f$ \xi_{\boldsymbol{Q}, n,
      //! \mathrm{optimal}} \f$ for the rotation associated with the
      //! eigenvector rotation matrix \f$ \boldsymbol{Q} \f$
      //! (saved for all GP) from the previous time step (determined
      //! such that it leads to the previous LNL solution at the
      //! considered GP)
      std::vector<std::array<double, 3>> optimal_xi_eigenvect_rot_;


      //! lower interpolation factor (\f$ \xi_{\lambda_1,\mathrm{l}}
      //! \f$) for the eigenvalue \f$ \lambda_1 \f$:
      //! effectively, this is the lower bound for which the predictor
      //! leads to a numerically evaluable state
      double xi_l_lambda_1_;

      //! lower interpolation factor (\f$ \xi_{\lambda_2,\mathrm{l}}
      //! \f$) for the eigenvalue \f$ \lambda_2 \f$:
      //! effectively, this is the lower bound for which the predictor
      //! leads to a numerically evaluable state
      double xi_l_lambda_2_;

      //! lower interpolation factors (\f$ \xi_{\boldsymbol{Q},\mathrm{l}}
      //! \f$) for the rotation vector associated with the eigenvector
      //! (rotation) matrix \f$ \boldsymbol{Q} \f$:
      //! effectively, this is the lower bound for which the predictor
      //! leads to a numerically evaluable state
      std::array<double, 3> xi_l_eigenvect_rot_;

      //! upper interpolation factor (\f$ \xi_{\lambda_1, \mathrm{u}}
      //! \f$) for the eigenvalue \f$ \lambda_1 \f$:
      //! effectively, this is the upper bound for which the predictor
      //! leads to plastic strain rate == 0.0
      double xi_u_lambda_1_;

      //! upper interpolation factor (\f$ \xi_{\lambda_2, \mathrm{u}}
      //! \f$) for the eigenvalue \f$ \lambda_2 \f$:
      //! effectively, this is the upper bound for which the predictor
      //! leads to plastic strain rate == 0.0
      double xi_u_lambda_2_;

      //! upper interpolation factors (\f$ \xi_{\boldsymbol{Q}, \mathrm{u}}
      //! \f$) for the rotation vector associated with the eigenvector
      //! (rotation) matrix \f$ \boldsymbol{Q} \f$:
      //! effectively, this is the upper bound for which the predictor
      //! leads to plastic strain rate == 0.0
      std::array<double, 3> xi_u_eigenvect_rot_;

      //! current number of predictor adaptations
      unsigned int num_of_pred_adapt_;

      //! maximum number of allowed
      //! repredictorizations (including the initial predictor adaptation) before throwing error
      const unsigned int max_num_pred_adapt_;

      // maximum allowed number number of predictor adaptation iterations
      static constexpr unsigned int MAX_NUM_PRED_ADAPT_ITERS = 50;

      //! current predictor containing the inverse inelastic deformation
      //! gradient (components 0-8) and the plastic strain (component 9)
      Core::LinAlg::Matrix<10, 1> pred_;

      /*!
       * @brief Constructor!
       *
       * @param[in] k_scan Interval scanning parameter \f$ k_{\mathrm{scan}} \f$
       * @param[in] max_num_pred_adapt Maximum number of allowed
       * repredictorizations (including the initial predictor adaptation) before throwing error
       */
      PredictorAdaptationUtils(const double k_scan, const unsigned int max_num_pred_adapt);

      //! setup method: set the correct number of Gauss Points to track the internal variables of
      //! the class
      void setup(const int num_gp);

      /*!
       * @brief Preevaluation method, performing reset tasks and setting the reference values for
       *  interpolation
       *
       * @param[in] gp Gauss point index
       * @param[in] inv_plastic_defgrad_elast_pred inverse plastic deformation
       * gradient inside the elastic predictor
       * @param[in] inv_plastic_defgrad_plast_pred inverse plastic deformation
       * gradient inside the plastic predictor
       */
      void pre_evaluate(const int gp,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_elast_pred,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_plast_pred);

      /*!
       * @brief update method: update the internal variables of the predictor
       * interpolation struct based on the time step quantities of the
       * material. We have to specify whether we want to compute and
       * update the optimal xi value.
       *
       */
      void update();

      //! pack method
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack method
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! update the maximum interpolation factor in the current time
      //! step evaluation for the given Gauss point
      void update_current_max_xi(const int gp);

      /*!
       * @brief extract all components of the inverse plastic deformation gradient that
       * are relevant for the predictor adaptation
       *
       * @param[in] gp Gauss point index
       * @param[in] inv_plastic_defgrad_elast_pred inverse plastic deformation
       * gradient inside the elastic predictor
       * @param[in] inv_plastic_defgrad_plast_pred inverse plastic deformation
       * gradient inside the plastic predictor
       *
       */
      void extract_inv_plastic_defgrad_components(const int gp,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_elast_pred,
          const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_plast_pred);

      /*!
       * @brief interpolate inverse_plastic deformation gradient between the
       * elastic and the plastic predictor, given the
       * current several interpolation factors
       * @note Linear interpolation is employed for each eigenvalue, and
       * for the eigenvector rotation vector
       *
       * @param[in] gp Gauss point index
       *
       */
      Core::LinAlg::Matrix<3, 3> interpolate_inv_plastic_defgrad(const int gp);

      /*!
       * @brief Adapt interpolation parameter and interpolation interval
       * based on evaluation error
       *
       * @param[in] gp Gauss point index
       * @param[in] eval_err_type type of evaluation error necessitating
       * an adaptation of the parameter and the interval
       *
       */
      void adapt_interpolation_interval(const int gp, const ErrorType eval_err_type);

      /*!
       * @brief Adapt interpolation parameter based on the current
       * parameter bounds
       *
       * @param[in] gp Gauss point index
       *
       */
      void adapt_interpolation_parameter(const int gp);

      //! verify whether we are at the elastic predictor based on the
      //! current interpolation factors
      bool verify_interp_factors_elast_pred(const int gp);

      //! compute optimal interpolation factors, given an inverse
      //! plastic deformation gradient as a solution of the Local
      //! Newton Loop (reference)
      void compute_optimal_interp_factors(
          const int gp, const Core::LinAlg::Matrix<3, 3>& inv_plastic_defgrad_reference);
    };

    //! struct with local substepping utilities
    struct LocalSubsteppingUtils
    {
      //! current time parameter ranging from 0 to the problem time step \f$ \Delta t \f$
      double t_;
      //! counter of evaluated substeps
      unsigned int substep_counter_;
      //! current substep size
      double curr_dt_;
      //! number of times the problem time step \f$ \Delta t \f$ has been halved
      unsigned int time_step_halving_counter_;
      //!  current total number of substeps to be evaluated within the time step \f$ \Delta t
      //! \f$; this is not always given by time_step_halving_counter, since the
      //! halving does not have to be uniform (e.g. we could halve the time step twice and still
      //! have 3 substeps to evaluate instead of 4, i.e. if the first substep was evaluable
      //! numerically, but the second substep not, leading to another halving of the substep
      //! length)
      unsigned int total_num_of_substeps_;

      //! reset routine: basically, create a new empty object
      void reset();
    };

    /// class containing utilities for general analysis of the material
    /// time integration (including predictor adaptation, Local Newton
    /// loop, line search):
    /// error types, number of line searches, timers, ... Currently only
    /// employed for single-element single-processor simulations.
    class GeneralLocalTimIntAnalysisUtils
    {
     public:
      //! number of LNL steps for the current timestep evaluation (LNL)
      unsigned int eval_num_of_LNL_steps_ = 0;

      //! total number of LNL steps over all time steps
      unsigned int total_num_of_LNL_steps_ = 0;

      //! number of iterations for the current timestep evaluation (LNL)
      unsigned int eval_num_of_iters_ = 0;

      //! total number of LNL iterations over all time steps
      unsigned int total_num_of_iters_ = 0;

      //! number of repredictorizations for the current timestep evaluation (LNL)
      unsigned int eval_num_of_repredict_ = 0;

      //! total number of LNL repredictorizations over all time steps
      unsigned int total_num_of_repredict_ = 0;

      //! number of iterations spent in the predictor adaptation for the
      //! current timestep evaluation (including repredictorization)
      unsigned int eval_num_of_pred_adapt_iters_ = 0;

      //! total number of iterations spent in the predictor adaptation
      //! over all time steps (including repredictorization)
      unsigned int total_num_of_pred_adapt_iters_ = 0;

      //! number of iterations spent in the predictor adaptation for the
      //! current timestep evaluation (LNL), in the specific case of repredictorization
      unsigned int eval_num_of_repredict_iters_ = 0;

      //! total number of iterations spent in the predictor adaptation
      //! over all time steps,
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

      //! total number of LNL line searches over all time steps
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

      //! total number of line search iterations over all time steps
      unsigned int total_num_of_line_search_iters_ = 0;

      //! number of times the LNL convergences directly in its first
      //! iteration (due to a good predictor!) for the current timestep evaluation
      unsigned int eval_num_of_first_iter_convergences = 0;

      //! total number of times the LNL convergences directly in its first
      //! iteration (due to a good predictor!)
      unsigned int total_num_of_first_iter_convergences = 0;

      //! predictor interpolation factor for eigenvalue \f$ \lambda_1 \f$ obtained from the
      //! predictor adaptation routine (for set GP, current time step, last global iteration)
      double curr_pred_interp_factor_lambda_1_ = 0;

      //! predictor interpolation factor for eigenvalue \f$ \lambda_2 \f$ obtained from the
      //! predictor adaptation routine (for set GP, current time step, last global iteration)
      double curr_pred_interp_factor_lambda_2_ = 0;

      //! predictor interpolation factor for rotation vector (component
      //! 0) associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$ obtained from
      //! the predictor adaptation routine (for set GP, current time step, last global iteration)
      double curr_pred_interp_factor_eigenvect_rot_comp_0_ = 0;

      //! predictor interpolation factor for rotation vector (component
      //! 1) associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$ obtained from
      //! the predictor adaptation routine (for set GP, current time step, last global iteration)
      double curr_pred_interp_factor_eigenvect_rot_comp_1_ = 0;

      //! predictor interpolation factor for rotation vector (component
      //! 2) associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$ obtained from
      //! the predictor adaptation routine (for set GP, current time step, last global iteration)
      double curr_pred_interp_factor_eigenvect_rot_comp_2_ = 0;

      //! predictor interpolation factor (eigenvalue \f$ \lambda_1 \f$) obtained from the predictor
      //! adaptation routine (for set GP, current
      //! time step, maximum over all global iterations)
      double curr_max_pred_interp_factor_lambda_1_ = 0;

      //! predictor interpolation factor (eigenvalue \f$ \lambda_2 \f$) obtained from the predictor
      //! adaptation routine (for set GP, current
      //! time step, maximum over all global iterations)
      double curr_max_pred_interp_factor_lambda_2_ = 0;

      //! predictor interpolation factor (rotation vector associated
      //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
      //! component 0) obtained from the predictor
      //! adaptation routine (for set GP, current
      //! time step, maximum over all global iterations)
      double curr_max_pred_interp_factor_eigenvect_rot_comp_0_ = 0;

      //! predictor interpolation factor (rotation vector associated
      //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
      //! component 1) obtained from the predictor
      //! adaptation routine (for set GP, current
      //! time step, maximum over all global iterations)
      double curr_max_pred_interp_factor_eigenvect_rot_comp_1_ = 0;

      //! predictor interpolation factor (rotation vector associated
      //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
      //! component 2) obtained from the predictor
      //! adaptation routine (for set GP, current
      //! time step, maximum over all global iterations)
      double curr_max_pred_interp_factor_eigenvect_rot_comp_2_ = 0;

      //! optimal predictor interpolation factor (eigenvalue \f$ \lambda_1 \f$) obtained from the
      //! time step solution (for GP 0 of element 0 after the current time step)
      double optimal_pred_interp_factor_lambda_1_ = 0;

      //! optimal predictor interpolation factor (eigenvalue \f$ \lambda_2 \f$) obtained from the
      //! time step solution (for GP 0 of element 0 after the current time step)
      double optimal_pred_interp_factor_lambda_2_ = 0;

      //! optimal predictor interpolation factor (rotation vector associated
      //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
      //! component 0) obtained from the
      //! time step solution (for GP 0 of element 0 after the current time step)
      double optimal_pred_interp_factor_eigenvect_rot_comp_0_ = 0;

      //! optimal predictor interpolation factor (rotation vector associated
      //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
      //! component 1) obtained from the
      //! time step solution (for GP 0 of element 0 after the current time step)
      double optimal_pred_interp_factor_eigenvect_rot_comp_1_ = 0;

      //! optimal predictor interpolation factor (rotation vector associated
      //! with eigenvector rotation matrix \f$ \boldsymbol{Q} \f$,
      //! component 2) obtained from the
      //! time step solution (for GP 0 of element 0 after the current time step)
      double optimal_pred_interp_factor_eigenvect_rot_comp_2_ = 0;

      //! Local Newton residual obtained from the optimal predictor interpolation factor
      double lnl_res_optimal_pred_interp_factor_ = 0;

      //! timer for the current inelastic deformation gradient evaluation in the current timestep
      Teuchos::Time eval_teuchos_timer_inelastic_defgrad_{
          "InelasticDefgradTransvIsotropElastViscoplast::inelastic deformation gradient "
          "evaluation"};

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

      //! timer for the current linearization evaluation (additional cmat) in the current timestep
      Teuchos::Time eval_teuchos_timer_additional_cmat_{
          "InelasticDefgradTransvIsotropElastViscoplast::linearization "
          "evaluation (additional cmat)"};

      //! evaluation time for the current time step to compute the
      //! inelastic deformation gradient
      double eval_time_inelastic_defgrad_;

      //! total evaluation time for the inelastic deformation gradient
      //! over all times
      double total_time_inelastic_defgrad_;

      //! evaluation time spent in the Local Newton loop (current
      //! time step)
      double eval_time_LNL_;

      //! total time spent in the Local Newton Loop over all time steps
      double total_time_LNL_;

      //! evaluation time spent in the predictor adaptation (current
      //! time step)
      double eval_time_pred_adapt_;

      //! total time spent in the predictor adaptation over all time steps
      double total_time_pred_adapt_;

      //! evaluation time spent in the predictor adaptation (current
      //! time step), in the specific case of repredictorization
      double eval_time_repredict_;

      //! total time spent in the predictor adaptation over all time steps, in the specific
      //! case of repredictorization
      double total_time_repredict_;

      //! evaluation time spent in the line search (current
      //! time step)
      double eval_time_line_search_;

      //! total time spent in the line search scheme over all time steps
      double total_time_line_search_;

      //! evaluation time for the current time step to compute the
      //! linearization (additional cmat)
      double eval_time_additional_cmat_;

      //! total evaluation time for the linearization (additional cmat)
      //! over all times
      double total_time_additional_cmat_;

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

      //! reset called for current time step?
      bool is_reset_current_timestep_ = false;

      //! how often was the update method called? (maximum:
      //! num_of_global_elements, if only one processor
      //! is considered)
      int num_update_calls_ = 0;

      //! runtime csv writer
      std::optional<Core::IO::RuntimeCsvWriter> csv_writer_;

      //! initialize csv writer
      void init_csv_writer();

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

    //! struct containing quantities computed from a given elasticity/plasticity state;
    //! given: current right Cauchy-Green deformation tensor, inelastic deformation gradient and
    //! plastic strain at the previous time instant
    struct StateQuantities
    {
      // ----- current state quantities (for the evaluated Gauss points) ----- //

      //! elastic right Cauchy-Green deformation tensor
      Core::LinAlg::Matrix<3, 3> curr_CeM_{Core::LinAlg::Initialization::zero};

      //! isotropic stress factors
      Core::LinAlg::Matrix<3, 1> curr_gamma_{Core::LinAlg::Initialization::zero};

      //! isotropic constitutive tensor factors
      Core::LinAlg::Matrix<8, 1> curr_delta_{Core::LinAlg::Initialization::zero};

      //! elastic 2nd PK stress tensors (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<3, 3> curr_SeM_{Core::LinAlg::Initialization::zero};

      //! elastic stiffness tensor (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<6, 6> curr_dSedCe_{Core::LinAlg::Initialization::zero};

      //! deviatoric, symmetric part of the Mandel stress tensor
      Core::LinAlg::Matrix<3, 3> curr_Me_dev_sym_M_{Core::LinAlg::Initialization::zero};

      //! equivalent tensile stress
      double curr_equiv_stress_{0.0};

      //! equivalent plastic strain rate
      double curr_equiv_plastic_strain_rate_{0.0};

      //! plastic flow direction tensor
      Core::LinAlg::Matrix<3, 3> curr_NpM_{Core::LinAlg::Initialization::zero};

      //! plastic stretching tensor
      Core::LinAlg::Matrix<3, 3> curr_dpM_{Core::LinAlg::Initialization::zero};

      //! plastic velocity gradient tensor
      Core::LinAlg::Matrix<3, 3> curr_lpM_{Core::LinAlg::Initialization::zero};

      //! plastic update tensor
      Core::LinAlg::Matrix<3, 3> curr_EpM_{Core::LinAlg::Initialization::zero};
    };


    //! struct containing specific derivatives of quantities computed from a given
    //! elasticity/plasticity state; given: current right Cauchy-Green deformation tensor, inelastic
    //! deformation gradient and plastic strain at the previous time instant
    struct StateQuantityDerivatives
    {
      // ----- current state variable derivatives (for the evaluated Gauss points)----- //

      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the inverse
      //! inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dCediFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the right
      //! Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dCedC_{Core::LinAlg::Initialization::zero};

      //! derivatives of the equivalent tensile stress w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<1, 9> curr_dequiv_stress_diFin_{Core::LinAlg::Initialization::zero};
      //! derivatives of the equivalent tensile stress w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress form)
      Core::LinAlg::Matrix<1, 6> curr_dequiv_stress_dC_{Core::LinAlg::Initialization::zero};

      //! derivative of the deviatoric, symmetric part of the Mandel stress tensor w.r.t. the
      //! inverse inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dMe_dev_sym_diFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the deviatoric, symmetric part of the Mandel stress tensor w.r.t. the right
      //! Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dMe_dev_sym_dC_{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic strain rate w.r.t. the equivalent stress
      double curr_dpsr_dequiv_stress_{0.0};
      //! derivative of the plastic strain rate w.r.t. the equivalent plastic strain
      double curr_dpsr_depsp_{0.0};

      //! derivative of the plastic stretching tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_ddpdiFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the equivalent plastic strain (Voigt
      //! stress form)
      Core::LinAlg::Matrix<6, 1> curr_ddpdepsp_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_ddpdC_{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic velocity gradient tensor w.r.t. the inverse inelastic
      //! deformation gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dlpdiFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the equivalent plastic strain
      //! (Voigt notation)
      Core::LinAlg::Matrix<9, 1> curr_dlpdepsp_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the right Cauchy-Green
      //! deformation tensor (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dlpdC_{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic update tensor w.r.t. the inverse inelastic deformation gradient
      //! (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dEpdiFin_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the equivalent plastic strain (Voigt
      //! notation)
      Core::LinAlg::Matrix<9, 1> curr_dEpdepsp_{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the right Cauchy-Green deformation tensor
      //! (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dEpdC_{Core::LinAlg::Initialization::zero};
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
      int ele_gid_;

      //! Gauss point index
      int gp_;

      //! time instant \f$t_{n}\f$
      double tn_;

      //! time instant \f$t_{n+1}\f$
      double tnp_;

      //! tracker for the global iteration (if we have output every
      //! iteration) or the timestep index; increased by 1 every time
      //! the Gauss point output routine / or the update method (if no
      //! Gauss point output is considered) is called
      unsigned int globiter_or_timestep_index_;

      //! tracker for the local NR iteration
      unsigned int lnl_iter_;
    };


    // enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonConvCheck
    {
      ResidualOnly,   ///< only verify convergence based on the absolute value of the Local Newton
                      ///< residual
      IncrementOnly,  ///< only verify convergence based on the 2-norm of the Local Newton
                      ///< solution increment
      ResidualAndIncrement,  ///< verify convergence based on both the Local Newton residual and the
                             ///< solution increment
    };



    // enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonDiverCont
    {
      Stop,      ///< stop the simulation entirely
      Continue,  ///<  continue the simulation, and display warning in regards to the current state
                 ///< within the Local Newton Loop
      ContinueWithSafeGuard  ///< continue the simulation only if the convergence tolerances are not
                             ///< exceeded excessively
    };


    //! struct containing settings and iteration data from the Local Newton-Raphson
    //! Loop (time integration of the viscoplasticity equations)
    //! (used for Gauss-Point output). In contrast to
    //! GeneralLocalTimIntAnalysisUtils which tracks general information
    //! over time steps (such as how many iterations were performed for
    //! a specific time step), this utility struct considers the data
    //! for each specific iteration of the Local Newton-Raphson Loop.
    struct LocalNewtonData
    {
      /*!
       *   @brief Constructor
       *
       * @param[in] res_tol tolerance for the Local Newton-Raphson
       * scheme (absolute residual value)
       * @param[in] incr_tol tolerance for the Local Newton-Raphson
       * scheme (2-norm of solution increment)
       * @param[in] conv_check convergence check strategy for the Local
       * Newton Loop
       * @param[in] diver_cont strategy for dealing with divergence of the Local
       * Newton Loop
       *
       */
      //! constructor of Local Newton data, based on the
      LocalNewtonData(const double res_tol, const double incr_tol,
          const LocalNewtonConvCheck conv_check, const LocalNewtonDiverCont diver_cont);

      //! convergence tolerance of the Local Newton Loop (absolute
      //! residual value)
      const double res_tol_;

      //! convergence tolerance of the Local Newton Loop (2-norm of the
      //! solution increment)
      const double incr_tol_;

      //! convergence check strategy of the Local Newton Loop
      const LocalNewtonConvCheck conv_check_;

      //! strategy for dealing with divergence of the Local Newton Loop
      const LocalNewtonDiverCont diver_cont_;

      //! maximum number of Local Newton Loop iterations
      static constexpr unsigned max_iter_ = 200;

      //! maximum exceedance factor of the residual tolerance (to be used when
      //! using the divergence management strategy for continuation with
      //! safeguard)
      static constexpr double max_exceedance_fact_res_tol_ = 1.0e2;

      //! maximum exceedance of the solution increment tolerance (to be used when
      //! using the divergence management strategy for continuation with
      //! safeguard)
      static constexpr double max_exceedance_fact_incr_tol_ = 1.0e2;



      //! current LNL iteration
      unsigned int iter_;

      //! tracker for the global iteration (if we have output every
      //! iteration) or the timestep index; increased by 1 every time
      //! the Gauss point output routine is called
      unsigned int globiter_or_timestep_index_;

      //! do we have Gauss point output every global iteration?
      bool is_Gauss_point_output_every_global_iter_ = false;

      //! success status of the iteration (can it even evaluate the
      //! residual?); vector of GP values
      std::vector<std::array<LocalIterationStatus, max_iter_>> all_iter_status_;

      //! all iteration values of the LNL residual; vector of GP values
      std::vector<std::array<double, max_iter_>> all_residual_;

      //! all iteration values of the equivalent stress; vector of GP values
      std::vector<std::array<double, max_iter_>> all_equiv_stress_;

      //! all iteration values of the plastic strain; vector of GP values
      std::vector<std::array<double, max_iter_>> all_plastic_strain_;

      //! resize all relevant vectors based on the number of Gauss
      //! points known only after setting up the problem -> each vector
      //! item gets the same value for now
      void set_num_of_gp(const unsigned int num_of_gp);

      //! reset all arrays holding values for all iterations (for a
      //! given Gauss point)
      void reset_all_iteration_data(const unsigned int gp);

      // maybe we need some pack and unpack methods perspectively? If
      // this is to be used consistently in the future...-> would mainly
      // concern the global iteration / time step tracker, but nothing else.

      //! data collector for a single Local Newton iteration
      struct LocalIterDataCollector
      {
        //! success status of the iteration (can it even evaluate the
        //! residual?); vector of GP values
        const LocalIterationStatus iter_status_;

        //! residual of the current iteration
        const double residual_;

        //! equivalent stress of the current iteration
        const double equiv_stress_;

        //! plastic strain of the current iteration
        const double plastic_strain_;
      };

      //! append data for a given iteration (specified via output
      //! tracking data)
      void set_iteration_data(const CSVOutputTrackingData csv_output_tracking_data,
          const LocalIterDataCollector local_iter_data_collector);

      //! write LNL iteration data to csv file, when the LNL fails
      void write_failed_lnl_iteration_data_to_csv(
          const CSVOutputTrackingData csv_output_tracking_data);
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
        //! current interpolation factor \f$ \xi_{\lambda_1} \f$ for the
        //! eigenvalue \f$ \lambda_1 \f$
        double current_xi_lambda_1_ = -1;
        //! current interpolation factor \f$ \xi_{\lambda_2} \f$ for the
        //! eigenvalue \f$ \lambda_2 \f$
        double current_xi_lambda_2_ = -1;
        //! current interpolation factor \f$ \xi_{\boldsymbol{Q}} \f$ for the
        //! rotation vector associated with the eigenvector (rotation) matrix \f$ \boldsymbol{Q} \f$
        std::array<double, 3> current_xi_eigenvect_rot_{1.0, 1.0, 1.0};
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

      //! current interpolation factors \f$ \xi_{\lambda_1} \f$ for
      //! eigenvalue \f$ \lambda_1 \f$ of all
      //! microiterations
      std::vector<double> all_current_xi_lambda_1_;

      //! current interpolation factors \f$ \xi_{\lambda_2} \f$ for
      //! eigenvalue \f$ \lambda_2 \f$ of all
      //! microiterations
      std::vector<double> all_current_xi_lambda_2_;

      //! current interpolation factors \f$ \xi_{\boldsymbol{Q}} \f$ for
      //! the rotation vector associated with the eigenvalue (rotation) matrix \f$ \boldsymbol{Q}
      //! \f$ of all microiterations
      std::vector<std::array<double, 3>> all_current_xi_eigenvect_rot_;

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
      CSVOutputTrackingData csv_output_tracking_data_;

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
      CSVOutputTrackingData csv_output_tracking_data_;

      //! writes data from each microiteration of a single line search (specified via tracking data)
      //! to a dedicated csv file
      void write_line_search_micro_iter_data_to_csv();
    };

    // display / log evaluation warnings
#define DISPLAY_WARNINGS ;

  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
