// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_SERVICE_HPP


#include "4C_config.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_four_tensor_generators.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <vector>


FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /// namespace: utilities for
  /// InelasticDefgradTransvIsotropElastViscoplast
  namespace InelasticDefgradTransvIsotropElastViscoplastUtils
  {
    /// declare numerical tolerance to be used in the verification of (numerically) zero plastic
    /// strain increments
    constexpr double zero_plastic_strain_increment{1.0e-14};

    /// enum class for error types in InelasticDefgradTransvIsotropElastViscoplast, used for
    /// triggering different procedures (e.g. Reinterpolation,
    /// substepping, line search) during the
    /// Local Newton Loop
    enum class ErrorType
    {
      no_errors,                ///< no errors
      negative_plastic_strain,  ///< negative plastic strain which does not allow for evaluations
                                ///< inside the viscoplasticity laws
      overflow_error,  ///< overflow error of the term \f$ \Delta t \dot{\varepsilon}^{\text{p}} \f$
                       ///< (and \f$ \boldsymbol{E}^{\text{p}}  = \exp(- \Delta t
                       ///< \dot{\varepsilon}^{\text{p}} \boldsymbol{N}^{\text{p}}) \f$)
      no_flow_resistance,  ///< the material has no flow resistance anymore, such that the
                           ///< evaluations model non-physical phenomena
      failed_solution_linear_system_lnl,  ///< solution of the linear system in the Local
                                          ///< Newton-Raphson Loop failed
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
      under_yield_surface,  ///< mechanical state is "under" the yield surface, i.e., the evaluated
                            ///< stress is smaller than the yield stress
    };


    /// struct: settings for registering errors within the procedures used for return mapping
    struct ErrorRegistrationSettings
    {
      //! should overflow error be registered via ErrorType when the plastic strain increment
      //! exceeds the specified tolerance
      const bool register_plastic_strain_incr_overflow;

      //! maximum, numerically evaluable plastic strain increment before overflow error is
      //! registered
      const double max_plastic_strain_incr;

      //! should overflow error be registered via ErrorType when any of the plastic strain
      //! derivative increments exceeds the specified tolerance
      const bool register_plastic_strain_incr_derivs_overflow;

      //! maximum, numerically evaluable increment of
      //! plastic strain derivatives (time_step * derivative)
      const double max_plastic_strain_deriv_incr;
    };

    /// enum class for evaluation management actions in the iterations of the
    /// Local Newton loop
    enum class EvaluationAction
    {
      continue_current_iteration,    ///< continue current iteration
      continue_with_next_iteration,  ///< go to next iteration after performing certain reset steps
      exit_with_error,               ///< exit Local Newton Loop with the set error status
    };

    /// convert error type to detailed error message
    std::string get_detailed_error_message_for_error_type(ErrorType err_type);

    /// enum class for material behavior types
    enum class MatBehavior
    {
      isotropic,         ///< isotropic material behavior
      transv_isotropic,  ///< transversely isotropic material behavior
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
      unsigned int pade_order = 16;  // by default we set the highest order currently implemented
    };

    //! struct containing time step settings and time trackers
    struct TimeStepTracker
    {
      //! time step length
      double dt;
      //! currently computed time instant \f$ t_{n+1} \f$
      double tnp;
      //! minimum substep length
      double min_dt;
    };


    //! struct containing quantities at the last and current time points (i.e., at \f[ t_n \f] and
    //! \f[ t_{n+1} \f], respectively). The quantities are tracked at all Gauss points, in order to
    //! update them simultaneously during the update method call
    struct TimeStepQuantities
    {
      //! right Cauchy-Green deformation tensor at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_rightCG;

      //! inverse plastic deformation gradient at the last time step (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_plastic_defgrad_inverse;

      //! (equivalent) plastic strain at the last time step (for all Gauss points)
      std::vector<double> last_plastic_strain;

      //! equivalent stress at the previous time instant (for all Gauss points)
      std::vector<double> last_equiv_stress;

      //! last (reduced) deformation gradient (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_defgrad;

      //! temporary variable, for which we store the right Cauchy-Green deformation tensor at each
      //! evaluation (used in order to update last_rightCG_ once outer NR converges) (for all Gauss
      //! points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_rightCG;

      //! current (reduced) deformation gradient: used to check whether the inverse inelastic
      //! deformation gradient has already been evaluated (to improve the computation performance)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_defgrad;

      //! current inverse plastic deformation gradient (for all Gauss points)
      std::vector<Core::LinAlg::Matrix<3, 3>> current_plastic_defgrad_inverse;

      //! current plastic strain (for all Gauss points)
      std::vector<double> current_plastic_strain;

      //! current equivalent stress (for all Gauss points)
      std::vector<double> current_equiv_stress;

      //! inverse plastic deformation gradient at the last computed time instant (after the last
      //! converged substep)
      std::vector<Core::LinAlg::Matrix<3, 3>> last_substep_plastic_defgrad_inverse;
      //! plastic strain at the last computed time instant (after the last converged substep)
      std::vector<double> last_substep_plastic_strain;

      /*!
       * @brief Set meaningful initial values. Done first for one single Gauss point (extended later
       * on using the resizing function).
       *
       */
      void init();

      /*!
       * @brief Resizing based on a given number of Gauss points
       *
       * @note The value saved within the first item is taken for all items during resizing. We only
       * enable resizing if the current sizes of the involved vectors are 1
       *
       * @param[in] numgp Number of Gauss points
       */
      void resize(const unsigned int numgp);

      /*!
       * @brief Perform pre-evaluation tasks at a given Gauss point
       *
       * @param[in] gp Gauss point index
       */
      void pre_evaluate(const unsigned int gp);

      //!  Update values between time steps: last <- current, at a given Gauss points
      void update(const unsigned int gp);

      //! Pack values
      void pack(Core::Communication::PackBuffer& data) const;

      //! Unpack values
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called{false};
    };



    //! struct: constant non-material tensors, such as different
    //! identity tensors
    struct ConstNonMatTensors
    {
      static const ConstNonMatTensors& instance()
      {
        static ConstNonMatTensors instance;
        return instance;
      }

      //! constructor
      ConstNonMatTensors();
      //! second-order 3x3 identity tensor in matrix form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<3, 3> id3x3{Core::LinAlg::Initialization::zero};
      //! second-order 3x3 identity in Voigt stress form \f$ \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 1> id6x1{Core::LinAlg::Initialization::zero};
      //! symmetric identity four tensor of dimension 3 \f$ \mathbb{I}_\text{S} \f$
      Core::LinAlg::Matrix<6, 6> id4_6x6{Core::LinAlg::Initialization::zero};
      //! deviatoric operator \f$ \mathbb{P}_{\text{dev}}  =  \mathbb{I}_\text{S} -
      //! \frac{1}{3} \boldsymbol{I} \otimes \boldsymbol{I} \f$
      Core::LinAlg::Matrix<6, 6> dev_op{Core::LinAlg::Initialization::zero};
      //! identity fourth-order tensor in Voigt notation: delta_AC delta_BD in index notation
      Core::LinAlg::Matrix<9, 9> id4_9x9{Core::LinAlg::Initialization::zero};
      //! second-order 10x10 identity tensor in matrix form
      Core::LinAlg::Matrix<10, 10> id10x10{Core::LinAlg::Initialization::zero};
    };



    //! struct containing constant tensors which depend on the constant fiber direction \f$
    //! \boldsymbol{m} \f$
    struct ConstMatTensors
    {
      //! \f$ \boldsymbol{I} + \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> id_plus_mm;
      //! \f$ \boldsymbol{m} \otimes \boldsymbol{m} \f$
      Core::LinAlg::Matrix<3, 3> mm{Core::LinAlg::Initialization::zero};
      //! deviatoric part \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //! \right)_\text{dev}\f$
      Core::LinAlg::Matrix<3, 3> mm_dev{Core::LinAlg::Initialization::zero};
      //! \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right) \otimes \left( \boldsymbol{m}
      //! \otimes \boldsymbol{m} \right) \f$ (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dyad_mm{Core::LinAlg::Initialization::zero};
      //!  \f$ \left( \boldsymbol{m} \otimes \boldsymbol{m} \right)_\text{dev} \otimes \left(
      //!  \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> mm_dev_dyad_mm{Core::LinAlg::Initialization::zero};
      //!  \f$ \boldsymbol{I} \otimes \left( \boldsymbol{m} \otimes \boldsymbol{m}
      //!  \right) \f$
      //! (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> id_dyad_mm;

      //! set tensors for a given fiber direction \f$ \boldsymbol{m} \f$
      void set_material_const_tensors(const Core::LinAlg::Matrix<3, 1>& m);
    };

    //! class with local substepping utilities
    class LocalSubsteppingUtils
    {
     public:
      LocalSubsteppingUtils() = delete;
      //! Constructor (calling reset under the hood)
      explicit LocalSubsteppingUtils(double dt) { reset(dt); }

      //! reset routine: set a single substep of a given size dt
      void reset(const double dt);

      //! verify whether the substepping routine has reached its end
      [[nodiscard]] bool end_substepping() const
      {
        return substep_counter_ > total_num_of_substeps_;
      };

      //! increment substep
      void increment_substep();

      //! halve current substep and update relevant quantities
      void halve_substep();

      //! get substep size
      [[nodiscard]] double get_substep_size() const { return curr_dt_; }

      //! retrieve the normalized time parameter at the next time instant during substepping, i.e.,
      //! \f$ \frac{\left(t_{m} + \Delta t_{m}\right)}{\Delta t} \f$, where \f$t_{m}\f$ denotes the
      //! previously converged time instant, \f$\Delta t_{m}\f$ denotes the current substep size,
      //! and \f$\Delta t\f$ specifies the global timestep
      [[nodiscard]] double get_normalized_next_time_param(const double dt) const
      {
        return (t_ + curr_dt_) / dt;
      }

      //! get counter for the current number of time step halving procedures
      [[nodiscard]] unsigned int get_halving_counter() const { return time_step_halving_counter_; }

      //! get substepping info as string
      [[nodiscard]] std::string get_info() const
      {
        std::string out;
        out += "\nSubstepping info: \n";
        out += std::format(
            "t: {}, substep_counter: {}, curr_dt: {}, time_step_halving_counter: {}, "
            "total_num_of_substeps: {} \n",
            t_, substep_counter_, curr_dt_, time_step_halving_counter_, total_num_of_substeps_);
        return out;
      };

     private:
      //! current time parameter ranging from 0 to the problem time step \f$ \Delta t \f$
      double t_;
      //! counter of evaluated substeps
      unsigned int substep_counter_;
      //! current substep size
      double curr_dt_;
      //! number of times the problem time step \f$ \Delta t \f$ has been halved
      unsigned int time_step_halving_counter_;
      //!  total number of substeps to be evaluated within the time step \f$ \Delta t
      //! \f$; this is not always directly proportional to time_step_halving_counter, since the
      //! halving does not have to be uniform (e.g. we could halve the time step twice and still
      //! have 3 substeps to evaluate instead of 4, i.e. if the first substep was evaluable
      //! numerically, but the second substep not, leading to another halving of the substep
      //! length)
      unsigned int total_num_of_substeps_;
    };

    /// enum class for state quantity evaluations in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// plastic strain rate,...)
    enum class StateQuantityEvalType
    {
      full_eval,  ///< full evaluation (full call of the evaluate_state_quantities method)
      plastic_strain_rate_only,  ///< return in evaluate_state_quantities once the plastic strain
                                 ///< rate has been evaluated
      equiv_stress_only,         ///< return in evaluate_state_quantities once the
                                 ///< equivalent stress has been evaluated
    };



    //! struct containing quantities computed from a given elasticity/plasticity state;
    //! given: current right Cauchy-Green deformation tensor, inelastic deformation gradient and
    //! plastic strain at the previous time instant
    struct StateQuantities
    {
      // ----- current state quantities (for the evaluated Gauss points) ----- //

      //! elastic right Cauchy-Green deformation tensor
      Core::LinAlg::Matrix<3, 3> curr_CeM{Core::LinAlg::Initialization::zero};

      //! isotropic stress factors
      Core::LinAlg::Matrix<3, 1> curr_gamma{Core::LinAlg::Initialization::zero};

      //! isotropic constitutive tensor factors
      Core::LinAlg::Matrix<8, 1> curr_delta{Core::LinAlg::Initialization::zero};

      //! elastic 2nd PK stress tensors (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<3, 3> curr_SeM{Core::LinAlg::Initialization::zero};

      //! elastic stiffness tensor (specifically only transversely-isotropic components)
      Core::LinAlg::Matrix<6, 6> curr_dSedCe{Core::LinAlg::Initialization::zero};

      //! deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      Core::LinAlg::Matrix<3, 3> curr_Mtheta_dev_sym_M{Core::LinAlg::Initialization::zero};

      //! equivalent tensile stress
      double curr_equiv_stress{0.0};

      //! equivalent plastic strain rate
      double curr_equiv_plastic_strain_rate{0.0};

      //! plastic flow direction tensor
      Core::LinAlg::Matrix<3, 3> curr_NpM{Core::LinAlg::Initialization::zero};

      //! plastic stretching tensor
      Core::LinAlg::Matrix<3, 3> curr_dpM{Core::LinAlg::Initialization::zero};

      //! plastic velocity gradient tensor
      Core::LinAlg::Matrix<3, 3> curr_lpM{Core::LinAlg::Initialization::zero};

      //! plastic update tensor
      Core::LinAlg::Matrix<3, 3> curr_EpM{Core::LinAlg::Initialization::zero};

      //! evaluation type
      StateQuantityEvalType eval_type;
    };

    /// enum class for evaluations of the state quantity derivatives in
    /// InelasticDefgradTransvIsotropElastViscoplast: what is the aim of
    /// the evaluation? (full evaluation, or only partial, e.g. only the
    /// derivatives of the plastic strain rate,...)
    enum class StateQuantityDerivEvalType
    {
      full_eval,  ///< full evaluation (full call of the evaluate_state_quantity_derivatives
                  ///< method)
      plastic_strain_rate_derivs_only,  ///< return in evaluate_state_quantity_derivatives once the
                                        ///< derivatives of the plastic strain rate have been
                                        ///< evaluated
      equiv_stress_derivs_only,  ///< return in evaluate_state_quantities once the derivatives of
                                 ///< the equivalent stress has been evaluated
    };



    //! struct containing specific derivatives of quantities computed from a given
    //! elasticity/plasticity state; given: current right Cauchy-Green deformation tensor,
    //! inelastic deformation gradient and plastic strain at the previous time instant
    struct StateQuantityDerivatives
    {
      // ----- current state variable derivatives (for the evaluated Gauss points)----- //

      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the inverse
      //! inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dCediFin{Core::LinAlg::Initialization::zero};
      //! derivative of the elastic right Cauchy_Green deformation tensor w.r.t. the right
      //! Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dCedC{Core::LinAlg::Initialization::zero};

      //! derivatives of the equivalent tensile stress w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<1, 9> curr_dequiv_stress_diFin{Core::LinAlg::Initialization::zero};
      //! derivatives of the equivalent tensile stress w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress form)
      Core::LinAlg::Matrix<1, 6> curr_dequiv_stress_dC{Core::LinAlg::Initialization::zero};

      //! derivative of the deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      //! w.r.t. the inverse inelastic deformation gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_dMtheta_dev_sym_diFin{Core::LinAlg::Initialization::zero};
      //! derivative of the deviatoric, symmetric part of the thermo-elastic Mandel stress tensor
      //! w.r.t. the right Cauchy-Green deformation tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_dMtheta_dev_sym_dC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic strain rate w.r.t. the equivalent stress
      double curr_dpsr_dequiv_stress{0.0};
      //! derivative of the plastic strain rate w.r.t. the equivalent plastic strain
      double curr_dpsr_depsp{0.0};

      //! derivative of the plastic stretching tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt stress form)
      Core::LinAlg::Matrix<6, 9> curr_ddpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the equivalent plastic strain (Voigt
      //! stress form)
      Core::LinAlg::Matrix<6, 1> curr_ddpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic stretching tensor w.r.t. the right Cauchy-Green deformation
      //! tensor (Voigt stress-stress form)
      Core::LinAlg::Matrix<6, 6> curr_ddpdC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic velocity gradient tensor w.r.t. the inverse inelastic
      //! deformation gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dlpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the equivalent plastic strain
      //! (Voigt notation)
      Core::LinAlg::Matrix<9, 1> curr_dlpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic velocity gradient tensor w.r.t. the right Cauchy-Green
      //! deformation tensor (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dlpdC{Core::LinAlg::Initialization::zero};

      //! derivative of the plastic update tensor w.r.t. the inverse inelastic deformation
      //! gradient (Voigt notation)
      Core::LinAlg::Matrix<9, 9> curr_dEpdiFin{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the equivalent plastic strain (Voigt
      //! notation)
      Core::LinAlg::Matrix<9, 1> curr_dEpdepsp{Core::LinAlg::Initialization::zero};
      //! derivative of the plastic update tensor w.r.t. the right Cauchy-Green deformation tensor
      //! (Voigt stress form)
      Core::LinAlg::Matrix<9, 6> curr_dEpdC{Core::LinAlg::Initialization::zero};

      //! evaluation type
      StateQuantityDerivEvalType eval_type;
    };

    //! struct containing specific derivatives of the scalar plastic strain rate used in
    //! InelasticDefgradTransvIsotropElastViscoplast
    struct PlasticStrainRateDerivs
    {
      //! derivative with respect to the equivalent stress
      double deriv_equiv_stress;


      //! derivative with respect to the plastic strain
      double deriv_plastic_strain;

      //! derivative with respect to the temperature
      double deriv_temperature;
    };


    /// enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonConvCheck
    {
      residual,         ///< verify convergence based on the absolute value of the Local Newton
                        ///< residual 2-norm
      increment_ratio,  ///< verify convergence based on the ratio of solution increment to current
                        ///< solution: \f$ \frac{\left| \Delta \boldsymbol{s}^{l+1} \right|}{\left|
                        ///< \boldsymbol{s}^{l} \right|}  \f$
      residual_and_increment_ratio,  ///< verify convergence based on both the absolute Local Newton
                                     ///< residual and the ratio of solution increment to current
                                     ///< solution
    };


    /// enum: strategy in dealing with divergence of the Local Newton Loop
    enum class LocalNewtonDiverCont
    {
      stop,          ///< stop the simulation entirely
      continue_sim,  ///<  continue the simulation, and display warning in regards to the current
                     ///<  state within the Local Newton Loop
      continue_sim_with_safeguard  ///< continue the simulation only if the convergence tolerances
                                   ///< are not exceeded excessively, as specified with specific
                                   ///< exceedance factors for the tolerances
    };

    /// enum: quantities relevant for convergence checking within the Local Newton Loop
    struct LocalNewtonConvQuantities
    {
      //! residual 2-norm
      double residual_norm;

      //! ratio of solution increment to current solution: \f$ \frac{\left| \Delta
      //! \boldsymbol{s}^{l} \right|}{\left| \boldsymbol{s}^{l} \right|}  \f$
      double increment_norm;
    };



    //! struct containing parameter specifications for the Local Newton loop
    struct LocalNewtonParams
    {
      //! convergence tolerance: absolute residual value
      const double res_tol;

      //! convergence tolerance: ratio of solution increment to current solution
      const double incr_tol;

      //! convergence check strategy
      const LocalNewtonConvCheck conv_check;

      //! strategy for dealing with divergence
      const LocalNewtonDiverCont diver_cont;

      //! maximum number of local iterations
      const unsigned int max_iter;

      //! maximum exceedance factor for the residual tolerance (to be used when
      //! employing the divergence management strategy for continuation with
      //! safeguard)
      const double max_exceedance_fact_res_tol;

      //! maximum exceedance factor for the solution increment tolerance (to be used when
      //! employing the divergence management strategy for continuation with
      //! safeguard)
      const double max_exceedance_fact_incr_tol;
    };


    //! class for managing the Local Newton loop, containing the utilized parameters and iteration
    //! data
    class LocalNewtonManager
    {
     public:
      LocalNewtonManager() = delete;
      /*!
       * @brief Constructor
       *
       * @param[in] lnl_params Local Newton parameters
       *
       */
      explicit LocalNewtonManager(const LocalNewtonParams& lnl_params);

      /// getter for Local Newton parameters
      [[nodiscard]] LocalNewtonParams params() const { return params_; }

      /// getter for local iteration count
      [[nodiscard]] unsigned int iter() const { return iter_; }

      /// getter for total number of local iterations evaluated in this time step (vector over all
      /// Gauss points)
      [[nodiscard]] const std::vector<unsigned int>& curr_num_iters() const
      {
        return curr_num_iters_;
      }

      /*!
       * @brief Resizing based on a given number of Gauss points
       *
       * @param[in] numgp Number of Gauss points
       */
      void resize(const unsigned int numgp);

      /*!
       * @brief Initialize the solution vector, and the iteration counter (optional), for the
       * subsequent Local Newton at the currently considered Gauss point
       *
       * @param[in] init_estimate initial estimate \f$ \boldsymbol{s}^{(0)} \f$
       * @param[in] reset_iter_counter reset the iteration counter?
       */
      void init_local_newton(
          const Core::LinAlg::Matrix<10, 1>& init_estimate, const bool reset_iter_counter);

      /// sets the residual norm based on the given residual vector
      void set_residual_norm(const Core::LinAlg::Matrix<10, 1>& residual)
      {
        convergence_quantities_.residual_norm = residual.norm2();
      }

      /*!
       * @brief Determine whether the Local Newton Loop has converged, based on the saved
       * convergence quantities and the specified convergence checks.
       *
       * @return boolean: true = converged
       */
      [[nodiscard]] bool is_local_newton_converged() const;


      /*!
       * @brief   After an unsuccessful convergence check: determine whether the Local Newton is
       * stuck, i.e., the relative solution increment is nearly 0, but there is no convergence yet,
       * based on the saved convergence quantities.
       *
       * @return boolean: true = stuck
       */
      [[nodiscard]] bool is_local_newton_stuck() const;


      /// is the maximum number of iterations exceeded?
      [[nodiscard]] bool is_max_iter_exceeded() { return iter_ > params_.max_iter; }


      /*!
       * @brief Increments the solution vector \f$ \boldsymbol{s}^{(l+1)} = \boldsymbol{s}^{(l)}
       * +
       * \Delta \boldsymbol{s}^{(l+1)} \f$ after the current iteration \f$ l \f$, along with the
       * iteration counter
       *
       * @note Also updates the increment norm (ratio of increment to solution) internally based on
       * the provided increment
       *
       * @param[in] delta_sol increment vector for the next iteration \f$\Delta
       * \boldsymbol{s}^{(l+1)}\f$
       */
      void increment_solution_vector_and_iter(const Core::LinAlg::Matrix<10, 1>& delta_sol);

      /// getter for the solution vector
      [[nodiscard]] Core::LinAlg::Matrix<10, 1> sol() const { return sol_; }


      /// getter for the convergence quantities
      [[nodiscard]] LocalNewtonConvQuantities convergence_quantities() const
      {
        return convergence_quantities_;
      }

      /*!
       * @brief Routine to be run after the Local Newton-Raphson at a given Gauss point
       *
       * @param[in] gp Gauss point index
       */
      void update_after_local_newton(const unsigned int gp);

      //! reset the saved number of iterations at a given Gauss point
      void reset_curr_num_iters(const unsigned int gp);

      //! pack values
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack values
      void unpack(Core::Communication::UnpackBuffer& buffer);

     private:
      //! Local Newton parameters
      const LocalNewtonParams params_;

      //! current local iteration
      unsigned int iter_;

      //! total number of local iterations for the current timestep; vector of Gauss point values
      std::vector<unsigned int> curr_num_iters_;

      //! solution vector in the current iteration \f$ \boldsymbol{s}^{(l)} \f$ (used at the
      //! currently considered Gauss point)
      Core::LinAlg::Matrix<10, 1> sol_;

      //! quantities used for convergence checks
      LocalNewtonConvQuantities convergence_quantities_;

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called_{false};
    };

    //! helper struct containing tensors associated with the deformation, passed as input
    //! for the local time integration
    struct LocalIntegrationDeformationTensors
    {
      /*!
       * @brief constructor
       *
       * @param[in] F deformation gradient \f$ \mathbf{F}_{n+1} \f$
       * @param[in] last_iFp previous inelastic/plastic deformation gradient \f$
       \mathbf{F}_{\text{p},n}^{-1} \f$
       *
       */
      LocalIntegrationDeformationTensors(
          const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<3, 3>& last_iFp);

      //! deformation gradient \f$ \mathbf{F}_{n+1} \f$
      Core::LinAlg::Matrix<3, 3> defgrad;

      //! inverse deformation gradient \f$ \mathbf{F}_{n+1}^{-1} \f$
      Core::LinAlg::Matrix<3, 3> inv_defgrad;

      //! right Cauchy-Green deformation tensor \f$ \mathbf{C}_{n+1} \f$
      Core::LinAlg::Matrix<3, 3> right_cg;

      //! inverse plastic deformation gradient within the elastic predictor \f$
      //! \left[ \mathbf{F}_{\mathrm{p},n+1}^{(\mathrm{E})} \right]^{-1} \f$
      Core::LinAlg::Matrix<3, 3> elastic_predictor_inverse_plastic_defgrad;

      //! elastic deformation gradient within the elastic predictor \f$
      //! \mathbf{F}_{\mathrm{e},n+1}^{(\mathrm{E})} \f$
      Core::LinAlg::Matrix<3, 3> elastic_predictor_elastic_defgrad;
    };

    //! plastic predictor within the adaptive estimate interpolation: strategies for choosing the
    //! elastic stretch eigenvalues \f$
    //! \boldsymbol{\Lambda} \f$
    enum class PlasticPredictorElasticStretchEigenvalType
    {
      scale_unit,  ///< the unit tensor is scaled with the deformation gradient determinant to
                   ///< maintain plastic incompressibility
    };


    //! plastic predictor within the adaptive estimate interpolation: strategies for choosing the
    //! elastic stretch eigenvectors \f$
    //! \boldsymbol{Q} \f$
    enum class PlasticPredictorElasticStretchEigenvectType
    {
      from_elastic_predictor,  ///< the elastic stretch eigenvectors are taken directly from the
                               ///< elastic predictor, which is a consistent assumption for
                               ///< isotropic material behavior
    };

    //! plastic predictor within the adaptive estimate interpolation: strategies for choosing the
    //! elastic stretch rotations \f$
    //! \boldsymbol{R} \f$
    enum class PlasticPredictorElasticRotationType
    {
      from_elastic_predictor,  ///< the elastic rotation is taken directly from the
                               ///< elastic predictor, which is a consistent assumption for
                               ///< isotropic material behavior
    };


    //! starting point type to be used for the adaptive estimate interpolation
    enum class AdaptiveEstimateInterpolationStartingPointType
    {
      user_set,                  ///< user-set constant factor
      last_interpolation_point,  ///< takes the interpolation point from the last global iteration
                                 ///< of the previous timestep, which resulted in a valid initial
                                 ///< guess, as the starting point for the interpolation within
                                 ///< the current timestep
      optimal_equiv_stress       ///< computes the interpolation factor based on the equivalent
                                 ///< stress from the previous timestep with respect to the its
                                 ///< corresponding elastic and plastic predictors.
    };


    //! struct: input for the optimal interpolation point determination based on the equivalent
    //! stress of the solution, between both predictors (adaptive estimate interpolation)
    struct OptimalEquivStressStartingPointInput
    {
      //! equivalent stress of the solution: \f$ \overline{\sigma}_{n} \f$
      double equiv_stress_solution;

      //! equivalent stress of the elastic predictor: \f$ \overline{\sigma}_{n}^{\text{E}} \f$
      double equiv_stress_elast_pred;

      //! equivalent stress of the plastic predictor: \f$ \overline{\sigma}_{n}^{\text{P}} \f$
      double equiv_stress_plast_pred;
    };

    //! enum class: method to be used for handling hardening variables within the adaptive
    //! estimate interpolation algorithm
    enum class AdaptiveEstimateInterpolationHardeningMethod
    {
      use_previous,            ///< use hardening variables from the previously converged time step
      integrate_via_evol_eqs,  ///< integrate the hardening variables via their dedicated
                               ///< evolution equations, using the interpolated elastic
                               ///< deformation gradient as input --> "smaller local Newton"
    };


    //! struct containing information required for integrating the hardening variables according to
    //! their evolution equations (currently only the equivalent plastic strain) within the adaptive
    //! estimate interpolation
    struct HardeningIntegrationInput
    {
      //! interpolated equivalent stress \f$ \overline_{\sigma}(\xi) \f$
      double interp_equiv_stress;

      //! previous plastic strain \f$ \varepsilon_{\text{p}}(\xi) \f$
      double last_plastic_strain;

      //! integration time step / substep \f$ \Delta t \f$
      double dt;
    };

    //! struct containing parameters dedicated to handling the hardening variables within
    //! the adaptive estimate interpolation
    struct AdaptiveEstimateInterpolationHardeningParams
    {  //! method to use
      const AdaptiveEstimateInterpolationHardeningMethod method;

      //! should failure of hardening integration (via evolution equations) be allowed?
      //! if so, the mechanical state is marked with an error status depending on the location
      //! of the stress state w.r.t. yield surface and the estimate interpolation continues;
      //! otherwise, an error is thrown
      const bool allow_integration_failure;

      //! maximum relative yield stress deviation within the integration failure strategy, deciding
      //! whether the state is too elastic (i.e., shifted too much towards the elastic predictor) or
      //! too plastic
      const double failure_relative_yield_stress_deviation;

      //! maximum number of iterations for hardening integration
      const unsigned int max_iter_integration;

      //! tolerance for hardening integration
      const double tol_integration;
    };


    //! struct: parameters used for the adaptive estimate interpolation (AEI)
    struct AdaptiveEstimateInterpolationParams
    {
      //! starting point type to be used for the adaptive estimate interpolation
      const AdaptiveEstimateInterpolationStartingPointType starting_point_type;

      //! specified starting point in case that the starting point is user_set
      const double user_set_starting_point;

      //! elastic stretch eigenvalue specification for the plastic predictor to be used within the
      //! adaptive estimate interpolation
      const PlasticPredictorElasticStretchEigenvalType plastic_pred_elastic_stretch_eigenval_type;

      //! elastic stretch eigenvector specification for the plastic predictor to be used within the
      //! adaptive estimate interpolation
      const PlasticPredictorElasticStretchEigenvectType plastic_pred_elastic_stretch_eigenvect_type;

      //! elastic rotation specification for the plastic predictor to be used within the adaptive
      //! estimate interpolation
      const PlasticPredictorElasticRotationType plastic_pred_elastic_rotation_type;

      //! maximum number of plastic predictor construction iterations \f$ i_{\text{C,max}} \f$
      const unsigned int max_num_plastic_pred_construct_iters;

      //! maximum relative deviation between the equivalent stress and the yield stress \f$
      //! \overline{\sigma} / \sigma_{\text{Y}} - 1 \f$; if elastic predictor has a smaller stress
      //! deviation than this this value, it is directly used as the initial Local Newton estimate
      //! without performing interpolation; otherwise, the plastic predictor is updated such that
      //! its relative stress deviation is smaller than this value
      const double max_relative_yield_stress_deviation;

      //! maximum number of estimate interpolation iterations \f$ i_{\text{EI,max}} \f$
      const unsigned int max_num_estimate_interp_iters;

      //! minimum interval length \f$ ( \xi_{\text{P}} - \xi_{\text{E}} )_{\text{min}} \f$ for
      //! estimate interpolation
      const double min_interp_interval;

      //! interval scanning parameter (bisection: = 1/2) used for plastic predictor construction and
      //! estimate interpolation
      const double interval_scanning_param;

      //! maximum number of adaptive re-estimations allowed
      const unsigned int max_num_reestimations;

      //! minimum interval \f$ \xi -  \xi_{\text{E}} \f$ required for verifying the intermediate
      //! point \f$ \xi_{\text{I}} = 1 / 2 (\xi + \xi_{\text{E}}) \f$ as an updated estimate
      //! candidate within the re-estimation procedure
      const double min_reestimation_interval;

      //! precondition the elastic deformation gradient within the elastic predictor to
      //! stabilize interpolation, i.e., components smaller than a set tolerance are set to 0.0 to
      //! avoid unnecessary, "numerical" rotations
      const bool precondition_elastic_pred;

      //! tolerance for preconditioning the elastic deformation gradient within the elastic
      //! predictor to stabilize interpolation
      const double tol_precondition_elastic_pred;


      //! hardening parameters
      const AdaptiveEstimateInterpolationHardeningParams hardening_params;
    };


    //! class: interpolator of elastic deformation gradients between the elastic and plastic
    //! predictors as presented in Ana, Schmidt, Wall: Adaptive
    //! Estimate Interpolation: Accelerating Local Newton-Raphson Schemes in Computational
    //! (Visco)Plasticity (Preprint). This class stores the values of the involved quantities at
    //! all Gauss points.
    class PredictorInterpolator
    {
     public:
      /*!
       * @brief Constructor
       *
       */
      PredictorInterpolator();

      //! resizing based on a given number of Gauss points
      void resize(const unsigned int numgp);

      //! pack method
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack method
      void unpack(Core::Communication::UnpackBuffer& buffer);

      /*!
       * @brief Constructs a preliminary plastic predictor.
       *
       *
       * @param[in] gp Gauss point index
       * @param[in] elastic_defgrad_elastic_pred elastic deformation gradient within the elastic
       * predictor
       * @param[in] aei_params parameters for the AEI procedure (containing plastic predictor
       * specifications)
       * @param[in] last_elastic_defgrad elastic deformation gradient at the previous time instant
       * \f$ t_{n} \f$
       */
      void construct_prelim_plastic_pred(const unsigned int gp,
          const Core::LinAlg::Matrix<3, 3>& elastic_defgrad_elastic_pred,
          const AdaptiveEstimateInterpolationParams& aei_params);


      /*!
       * @brief Interpolates an elastic deformation gradient based on the stored predictor
       * quantities
       *
       *
       * @param[in] gp Gauss point index
       * @param[in] interp_loc location used for interpolation; either \f$ \tau \f$ (plastic
       * predictor construction) or \f$ \xi \f$ estimate interpolation
       */
      Core::LinAlg::Matrix<3, 3> interpolate_elastic_defgrad(
          const unsigned int gp, const double interp_loc) const;


      /*!
       * @brief Sets the plastic predictor quantities using a given interpolation location, based
       * on the plastic predictor construction algorithm
       *
       *
       * @param[in] gp Gauss point index
       * @param[in] plastic_pred_loc location \f$ \tau \f$ determined in the plastic
       * predictor construction
       */
      void set_plastic_predictor_after_construction_algo(
          const unsigned int gp, const double plastic_pred_loc);

     private:
      /*!
       * @brief Interpolates eigenvalues and rotational contributions based on the stored
       * predictor quantities
       *
       *
       * @param[in] gp Gauss point index
       * @param[in] interp_loc location used for interpolation; either \f$ \tau \f$ (plastic
       * predictor construction) or \f$ \xi \f$ estimate interpolation
       */
      void interpolate_elastic_defgrad_contributions(const unsigned int gp, const double interp_loc,
          std::vector<double>& interp_eigenval,
          Core::LinAlg::Matrix<4, 1>& interp_rel_eigenvect_rot_quat,
          Core::LinAlg::Matrix<4, 1>& interp_rel_rot_quat) const;

      //! elastic predictor: elastic eigenvalues \f$ \lambda_{\mathrm{elast}, i} \f$ for all Gauss
      //! points
      std::vector<Core::LinAlg::Matrix<3, 3>> eigenval_elast_pred_;
      //! plastic predictor: elastic eigenvalues \f$ \lambda_{\mathrm{plast}, i} \f$ for all Gauss
      //! points
      std::vector<Core::LinAlg::Matrix<3, 3>> eigenval_plast_pred_;
      //! eigenvalue data for all Gauss points used
      //! directly in the scalar interpolator; contains all eigenvalues of the elastic predictor
      //! in the first item of the Gauss point data, and all eigenvalues of the plastic predictor in
      //! the second item
      std::vector<std::vector<std::vector<double>>> scalar_interp_eigenval_;
      //! interpolation locations for the elastic and plastic predictor, saved to be directly used
      //! within the interpolator
      const std::vector<Core::LinAlg::Matrix<1, 1>> ref_predictor_locs_;
      //! elastic predictor: elastic eigenvector rotation matrix \f$ \mathbf{Q}_{\mathrm{elast}}
      //! \f$ for all Gauss points
      std::vector<Core::LinAlg::Matrix<3, 3>> eigenvect_rot_elast_pred_;
      //! plastic predictor: relative elastic eigenvector quaternion \f$
      //! \mathbf{q}_{\mathbf{Q}_{\mathrm{plast,rel}}} \f$  associated with \f$
      //! \mathbf{Q}_{\mathrm{plast,rel}} =  \mathbf{Q}_{\mathrm{elast}}^{T}
      //! \mathbf{Q}_{\mathrm{plast}}
      //! \f$ for all Gauss points
      std::vector<Core::LinAlg::Matrix<4, 1>> rel_eigenvect_rot_plast_pred_;
      //! elastic predictor: elastic rotation matrix \f$ \mathbf{R}_{\mathrm{elast}} \f$ for all
      //! Gauss points
      std::vector<Core::LinAlg::Matrix<3, 3>> rot_elast_pred_;
      //! plastic predictor: relative elastic rotation quaternion \f$
      //! \mathbf{q}_{\mathbf{R}_{\mathrm{plast,rel}}}
      //! \f$ associated with \f$ \mathbf{R}_{\mathrm{plast,rel}} =
      //! \mathbf{R}_{\mathrm{elast}}^{T} \mathbf{R}_{\mathrm{plast}}\f$ for all Gauss points
      std::vector<Core::LinAlg::Matrix<4, 1>> rel_rot_plast_pred_;
      //! eigenvalue interpolator
      Core::LinAlg::ScalarInterpolator<1> eigenval_interpolator_;
      //! tracks whether the resizing function has been called, to set the current number of Gauss
      //! points exactly once!
      bool resize_called_{false};
    };

    //! struct: container of interpolation points / bounds for all Gauss points used in the adaptive
    //! estimate interpolation, as presented in Ana, Schmidt, Wall: Adaptive
    //! Estimate Interpolation: Accelerating Local Newton-Raphson Schemes in Computational
    //! (Visco)Plasticity (Preprint). This class stores the values of the involved quantities at
    //! all Gauss points.
    struct InterpolationPointContainer
    {
     public:
      /*!
       * @brief Constructor
       *
       * @param[in] aei_params parameters for adaptive estimate interpolation
       */
      InterpolationPointContainer(const AdaptiveEstimateInterpolationParams& aei_params);

      //! reset interpolation interval and set the current interpolation point to its saved
      //! starting point at a given Gauss point
      void reset_bounds_and_current_interp_point(const unsigned int gp);

      //! resizing based on a given number of Gauss points
      void resize(const unsigned int numgp);

      //! pack method
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack method
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! current interpolation point \f$ \xi \f$ for all Gauss points
      std::vector<double> current_interp_points;

      //! lower interpolation bound \f$ \xi_{\text{E}} \f$ for all Gauss points
      std::vector<double> lower_interp_bounds;

      //! upper interpolation bound \f$ \xi_{\text{P}} \f$ for all Gauss points
      std::vector<double> upper_interp_bounds;

      //! starting points for interpolation \f$ \hat{\xi} \f$ for all Gauss points
      std::vector<double> starting_points;

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called{false};
    };



    //! class: manager for the adaptive estimate interpolation algorithm presented in
    //! Ana, Schmidt, Wall: Adaptive Estimate Interpolation: Accelerating Local Newton-Raphson
    //! Schemes in Computational (Visco)Plasticity (Preprint)
    class AdaptiveEstimateInterpolationManager
    {
     public:
      AdaptiveEstimateInterpolationManager() = delete;
      /*!
       * @brief Constructor
       *
       * @param[in] aei_params Adaptive Estimate Interpolation parameters
       */
      explicit AdaptiveEstimateInterpolationManager(
          const AdaptiveEstimateInterpolationParams& aei_params);

      //! resize method: set the correct number of Gauss Points
      void resize(const unsigned int num_gp);

      //! get info as string
      [[nodiscard]] std::string get_info(const unsigned int gp) const
      {
        std::string out;
        out += "\nAdaptive estimate interpolation info: \n";
        out += std::format(
            "number of plastic predictor construction iterations: {} / {}, number of estimate "
            "interpolation iterations: {} / {}, number of re-estimations: {} / {}, interpolation "
            "interval: [{}, {}] / {}, "
            "current interpolation point: {} \n",
            num_plastic_pred_construct_iters_, params_.max_num_plastic_pred_construct_iters,
            num_estimate_interp_iters_, params_.max_num_estimate_interp_iters, num_reestimations_,
            params_.max_num_reestimations, lower_interp_bound(gp), upper_interp_bound(gp),
            params_.min_interp_interval, current_interp_point(gp));
        return out;
      };



      /*!
       * @brief Verify whether plastic predictor construction is still possible, based on the set
       * maximum number of iterations
       *
       * @param[in] gp Gauss point index
       *
       */
      bool is_plastic_pred_construct_possible(const unsigned int gp);

      /*!
       * @brief Verify whether estimate interpolation is still possible, based on the
       * set minimum interpolation interval and the set maximum number of interpolation
       * iterations
       *
       * @param[in] gp Gauss point index
       *
       */
      bool is_estimate_interp_possible(const unsigned int gp);

      /*!
       * @brief Verify whether re-estimations are still possible, based on the
       * set maximum number of re-estimations
       *
       * @param[in] gp Gauss point index
       *
       */
      bool is_reestimation_possible(const unsigned int gp);

      /*!
       * @brief Reset tasks and construction of the preliminary plastic
       * predictor at a given Gauss point
       *
       * @param[in] gp Gauss point index
       * @param[in] deftensors deformation tensors used for local time integration
       */
      void reset_and_construct_prelim_plastic_pred(
          const unsigned int gp, const LocalIntegrationDeformationTensors& deftensors);

      //! pack method
      void pack(Core::Communication::PackBuffer& data) const;

      //! unpack method
      void unpack(Core::Communication::UnpackBuffer& buffer);

      /*!
       * @brief Interpolate the inverse inelastic deformation gradient required by the viscoplastic
       * material.
       *
       * @note The interpolation takes place between the elastic deformation gradients
       * associated with the elastic and the plastic predictor based on the current
       * interpolation point saved internally.
       *
       * @note The eigenvalues are interpolated using the logarithmic weighted average method
       * (see Satheesh et al. 2022, 10.1002/nme.7373) with linear weighting
       * between the predictors.
       *
       * @param[in] gp Gauss point index
       * @param[in] inv_defgrad inverse inelastic deformation gradient
       * used within the AEI
       */
      Core::LinAlg::Matrix<3, 3> interpolate_inverse_inelastic_defgrad(
          const unsigned gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad);

      /*!
       * Retrieves the inverse inelastic deformation gradient associated with the plastic predictor,
       * via interpolation at the value \f$ \xi = 1.0 \f$, at the specified Gauss point
       *
       * @param[in] gp Gauss point index
       * @param[in] inv_defgrad inverse inelastic deformation gradient
       * used within the AEI
       */
      Core::LinAlg::Matrix<3, 3> get_inverse_inelastic_defgrad_plastic_pred(
          const unsigned int gp, const Core::LinAlg::Matrix<3, 3>& inv_defgrad);

      /*!
       * @brief Sets the plastic predictor quantities based on the current interpolation point; then
       * resets the interpolation point container consistently, at the specified Gauss point
       *
       * @param[in] gp Gauss point
       */
      void set_plastic_predictor_after_construction_algo(const unsigned int gp);

      /*!
       * @brief Adapt interpolation interval \f$ \left[\xi_{\mathrm{E}}, \xi_{\mathrm{P} \right] \f$
       * based on evaluation error, and reset the current interpolation point \f$ \xi \f$
       *
       * @param[in] gp Gauss point
       * @param[in] eval_err_type type of evaluation error necessitating
       * an adaptation of the parameter and the interval
       *
       */
      void adapt_interpolation_interval_and_point(
          const unsigned int gp, const ErrorType& eval_err_type);

      /// enum class: types of interpolation points which can be used to set the current
      /// interpolation point
      enum class CurrentInterpPointPreset
      {
        standard,  ///< standard current interpolation point between the lower and upper bound (\f$
                   ///< \xi \gets \xi_{\text{E}} k_{scan} \left( \xi_{\text{P}} - \xi_{\text{P}}
                   ///< \right) \f$)
        elastic_predictor,  ///< elastic predictor (\f$ \xi \gets 0.0 \f$)
        plastic_predictor,  ///< plastic predictor (\f$ \xi \gets 1.0 \f$)
        starting_point,     ///< starting point (\f$ \xi \gets \hat{\xi} \f$)
        intermediate_point  ///<  midpoint between lower bound and current interpolation point
      };

      //! get current interpolation point at a specified Gauss point
      double current_interp_point(const unsigned int gp) const
      {
        FOUR_C_ASSERT_ALWAYS(
            gp < interp_point_container_.current_interp_points.size(), "GP index out of range");
        return interp_point_container_.current_interp_points[gp];
      }


      //! set current interpolation point at the specified Gauss
      //! point
      void set_current_interp_point(const unsigned int gp, const CurrentInterpPointPreset preset)
      {
        FOUR_C_ASSERT_ALWAYS(
            gp < interp_point_container_.current_interp_points.size(), "GP index out of range");
        switch (preset)
        {
          case CurrentInterpPointPreset::standard:
          {
            interp_point_container_.current_interp_points[gp] =
                interp_point_container_.lower_interp_bounds[gp] +
                params_.interval_scanning_param *
                    (interp_point_container_.upper_interp_bounds[gp] -
                        interp_point_container_.lower_interp_bounds[gp]);
            return;
          }
          case CurrentInterpPointPreset::elastic_predictor:
          {
            interp_point_container_.current_interp_points[gp] = 0.0;
            return;
          }
          case CurrentInterpPointPreset::plastic_predictor:
          {
            interp_point_container_.current_interp_points[gp] = 1.0;
            return;
          }
          case CurrentInterpPointPreset::starting_point:
          {
            interp_point_container_.current_interp_points[gp] =
                interp_point_container_.starting_points[gp];
            return;
          }
          case CurrentInterpPointPreset::intermediate_point:
          {
            interp_point_container_.current_interp_points[gp] =
                0.5 * (interp_point_container_.current_interp_points[gp] +
                          interp_point_container_.lower_interp_bounds[gp]);
            return;
          }
          default:
            FOUR_C_THROW(
                "Unsupported current interpolation point preset {}", EnumTools::enum_name(preset));
        }
      }

      //! get lower interpolation bound at the specified Gauss point
      [[nodiscard]] double lower_interp_bound(const unsigned int gp) const
      {
        FOUR_C_ASSERT_ALWAYS(
            gp < interp_point_container_.lower_interp_bounds.size(), "GP index out of range");
        return interp_point_container_.lower_interp_bounds[gp];
      }

      //! set lower interpolation bound to current interpolation point at the specified Gauss point
      void set_lower_interp_bound_to_current_interp_point(const unsigned int gp)
      {
        FOUR_C_ASSERT_ALWAYS(
            gp < interp_point_container_.lower_interp_bounds.size(), "GP index out of range");

        interp_point_container_.lower_interp_bounds[gp] =
            interp_point_container_.current_interp_points[gp];
      }

      //! get upper interpolation bound at the specified Gauss point
      [[nodiscard]] double upper_interp_bound(const unsigned int gp) const
      {
        FOUR_C_ASSERT_ALWAYS(
            gp < interp_point_container_.upper_interp_bounds.size(), "GP index out of range");
        return interp_point_container_.upper_interp_bounds[gp];
      }

      //! get starting point at the specified Gauss point
      [[nodiscard]] double starting_point(const unsigned int gp) const
      {
        FOUR_C_ASSERT_ALWAYS(
            gp < interp_point_container_.starting_points.size(), "GP index out of range");
        return interp_point_container_.starting_points[gp];
      }

      //! set starting point at a specified Gauss point, based on the set starting point type
      void set_starting_point(const unsigned gp,
          std::optional<OptimalEquivStressStartingPointInput> optimal_equiv_stress_input)
      {
        FOUR_C_ASSERT_ALWAYS(
            gp < interp_point_container_.starting_points.size(), "GP index out of range");
        switch (params_.starting_point_type)
        {
          case AdaptiveEstimateInterpolationStartingPointType::user_set:
          {
            FOUR_C_ASSERT_ALWAYS(!optimal_equiv_stress_input.has_value(),
                "The starting point should not be set using the optimal equivalent stress input "
                "for the starting point type {}",
                EnumTools::enum_name(params_.starting_point_type));

            interp_point_container_.starting_points[gp] = params_.user_set_starting_point;
            return;
          }
          case AdaptiveEstimateInterpolationStartingPointType::last_interpolation_point:
          {
            FOUR_C_ASSERT_ALWAYS(!optimal_equiv_stress_input.has_value(),
                "The starting point should not be set using the optimal equivalent stress input "
                "for the "
                "starting point type {}",
                EnumTools::enum_name(params_.starting_point_type));


            interp_point_container_.starting_points[gp] =
                interp_point_container_.current_interp_points[gp];
            return;
          }
          case AdaptiveEstimateInterpolationStartingPointType::optimal_equiv_stress:
          {
            FOUR_C_ASSERT_ALWAYS(optimal_equiv_stress_input.has_value(),
                "No input has been provided for calculating the optimal interpolation point based "
                "on the equivalent stress!");

            // set to elastic predictor if the stress of the elastic predictor is numerically 0.0 ->
            // this is theoretically
            // possible for viscoplastic laws without yield surfaces, which may have plastic flow
            // even in this case; however, the determination of the optimal point requires dividing
            // over this stress, which will not be possible in this specific case.
            // Same goes for the case where the elastic predictor and the plastic predictor are
            // associated with effectively the same stress value
            // -> set starting
            // point associated with the elastic predictor
            if (optimal_equiv_stress_input->equiv_stress_elast_pred <= 1.0e-12 ||
                std::abs(optimal_equiv_stress_input->equiv_stress_plast_pred -
                         optimal_equiv_stress_input->equiv_stress_elast_pred) /
                        optimal_equiv_stress_input->equiv_stress_elast_pred <
                    1.0e-8)
            {
              interp_point_container_.starting_points[gp] = 0.0;
              return;
            }


            // compute optimal interpolation point based on the equivalent stress: we clamp between
            // 0.0 and 1.0 because in some special cases such as stress relaxation, the starting
            // point may be slightly under 0.0 or over 1.0 (machine precision)
            interp_point_container_.starting_points[gp] =
                std::clamp((optimal_equiv_stress_input->equiv_stress_solution -
                               optimal_equiv_stress_input->equiv_stress_elast_pred) /
                               (optimal_equiv_stress_input->equiv_stress_plast_pred -
                                   optimal_equiv_stress_input->equiv_stress_elast_pred),
                    0.0, 1.0);
            return;
          }
          default:
            FOUR_C_THROW("Starting point type {} not yet supported!",
                EnumTools::enum_name(params_.starting_point_type));
        }
      }


      //! increment number of re-estimations
      void increment_num_reestimations() { ++num_reestimations_; }

      //! disable further re-estimations, by setting the specific counter at
      //! its maximum
      void disable_further_reestimations()
      {
        num_reestimations_ = params_.max_num_reestimations + 1;
      }

      //! reset number of estimate interpolation iterations
      void reset_num_estimate_interp_iters() { num_estimate_interp_iters_ = 0; }

      //! increment number of estimate interpolation iterations
      void increment_num_estimate_interp_iters() { ++num_estimate_interp_iters_; }

      //! increment number of plastic predictor construction iterations
      void increment_num_plastic_pred_construct_iters() { ++num_plastic_pred_construct_iters_; }

      //! reset number of plastic predictor construction iterations
      void reset_num_plastic_pred_construct_iters() { num_plastic_pred_construct_iters_ = 0; }

     private:
      //! enum class: shift direction for the interpolation
      enum class InterpolationShiftAction
      {
        shift_towards_elastic_pred,  ///< shift towards the elastic predictor (e.g., in case of
                                     ///< vanishing plastic strain increments)
        shift_towards_plastic_pred,  ///< shift towards the plastic predictor (e.g., in case of
                                     ///< overflow errors due to high overstresses)
      };

      //! get shift direction for evaluation  errors encountered in the plastic predictor
      //! construction / the estimate interpolation
      inline InterpolationShiftAction get_interpolation_shift_action(ErrorType error_type)
      {
        switch (error_type)
        {
          case ErrorType::overflow_error:
          case ErrorType::failed_computation_flow_resistance:
          case ErrorType::failed_computation_flow_resistance_derivs:
          case ErrorType::failed_matrix_log_evaluation:
          case ErrorType::failed_matrix_exp_evaluation:
            return InterpolationShiftAction::shift_towards_plastic_pred;
          case ErrorType::under_yield_surface:
            return InterpolationShiftAction::shift_towards_elastic_pred;
          default:
            FOUR_C_THROW("No action specified for error {} within the estimate interpolation!",
                EnumTools::enum_name(error_type));
        }
      }

      //! Adaptive Estimate Interpolation parameters
      const AdaptiveEstimateInterpolationParams params_;

      //! tracks whether the resizing function has been called, to set the current number of
      //! Gauss points exactly once!
      bool resize_called_{false};

      //! current number of plastic predictor construction iterations
      unsigned int num_plastic_pred_construct_iters_;

      //! current number of estimation interpolation iterations
      unsigned int num_estimate_interp_iters_;

      //! current number of re-estimations
      unsigned int num_reestimations_;

      //! container of interpolation points / bounds for all Gauss points
      InterpolationPointContainer interp_point_container_;

      //! predictor interpolator (containing data and logic for the elastic and plastic
      //! predictors for all Gauss points) in the current timestep
      PredictorInterpolator predictor_interpolator_;
    };

  }  // namespace InelasticDefgradTransvIsotropElastViscoplastUtils

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
