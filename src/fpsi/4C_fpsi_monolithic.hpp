// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FPSI_MONOLITHIC_HPP
#define FOUR_C_FPSI_MONOLITHIC_HPP

// FPSI includes
#include "4C_config.hpp"

#include "4C_adapter_ale_fpsi.hpp"
#include "4C_adapter_fld_fluid_fpsi.hpp"
#include "4C_ale.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fpsi.hpp"
#include "4C_fpsi_coupling.hpp"
#include "4C_fsi_monolithic.hpp"
#include "4C_inpar_fpsi.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_poroelast_base.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | forward declarations                                                  |
 *----------------------------------------------------------------------*/
namespace Core::LinAlg
{
  class SparseMatrix;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace PoroElast
{
  class Monolithic;
}

namespace Adapter
{
  class AleFpsiWrapper;
}  // namespace Adapter

namespace FSI
{
  namespace Utils
  {
    class MatrixColTransform;
  }  // namespace Utils
}  // namespace FSI
/*----------------------------------------------------------------------*/

namespace FPSI
{
  class MonolithicBase : public FpsiBase
  {
   public:
    //! ctor
    explicit MonolithicBase(MPI_Comm comm, const Teuchos::ParameterList& fpsidynparams,
        const Teuchos::ParameterList& poroelastdynparams);


    //! read restart data
    void read_restart(int step) override;

    //! start a new time step
    void prepare_time_step() override;

    //! take current results for converged and save for next time step
    void update() override;

    //! calculate stresses, strains, energies
    virtual void prepare_output(bool force_prepare);

    //! Output routine accounting for Lagrange multiplier at the interface
    void output() override;

    //! @name access sub-fields
    const std::shared_ptr<PoroElast::Monolithic>& poro_field() { return poroelast_subproblem_; };
    const std::shared_ptr<Adapter::FluidFPSI>& fluid_field() { return fluid_subproblem_; };
    const std::shared_ptr<Adapter::AleFpsiWrapper>& ale_field() { return ale_; };

    //@}

    std::shared_ptr<std::map<int, int>> Fluid_PoroFluid_InterfaceMap;
    std::shared_ptr<std::map<int, int>> PoroFluid_Fluid_InterfaceMap;

   protected:
    //! underlying poroelast problem
    std::shared_ptr<PoroElast::Monolithic> poroelast_subproblem_;
    //! underlying fluid of the FSI problem
    std::shared_ptr<Adapter::FluidFPSI> fluid_subproblem_;
    //! underlying ale of the FSI problem
    std::shared_ptr<Adapter::AleFpsiWrapper> ale_;

    //! flag defines if FSI Interface exists for this problem
    bool FSI_Interface_exists_;

   public:
    //! FPSI coupling object (does the interface evaluations)
    std::shared_ptr<FPSI::FpsiCoupling>& fpsi_coupl() { return fpsicoupl_; }

    //! @name Access General Couplings
    Coupling::Adapter::Coupling& fluid_ale_coupling() { return *coupfa_; }

    const Coupling::Adapter::Coupling& fluid_ale_coupling() const { return *coupfa_; }

    // Couplings for FSI
    Coupling::Adapter::Coupling& structure_fluid_coupling_fsi() { return *coupsf_fsi_; }
    Coupling::Adapter::Coupling& structure_ale_coupling_fsi() { return *coupsa_fsi_; }
    Coupling::Adapter::Coupling& interface_fluid_ale_coupling_fsi() { return *icoupfa_fsi_; }

    const Coupling::Adapter::Coupling& structure_fluid_coupling_fsi() const { return *coupsf_fsi_; }
    const Coupling::Adapter::Coupling& structure_ale_coupling_fsi() const { return *coupsa_fsi_; }
    const Coupling::Adapter::Coupling& interface_fluid_ale_coupling_fsi() const
    {
      return *icoupfa_fsi_;
    }

    //@}

   protected:
    //! @name Transfer helpers
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fluid_to_ale(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> ale_to_fluid(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;

    virtual std::shared_ptr<Core::LinAlg::Vector<double>> struct_to_fluid_fsi(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fluid_to_struct_fsi(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> struct_to_ale_fsi(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> ale_to_struct_fsi(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fluid_to_ale_fsi(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> ale_to_fluid_fsi(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> ale_to_fluid_interface_fsi(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;

    //@}

   private:
    //! FPSI - COUPLING
    //! coupling of fluid and ale in the entire fluid volume
    std::shared_ptr<Coupling::Adapter::Coupling> coupfa_;

    //! FSI - COUPLING
    //! coupling of structure and fluid at the interface
    std::shared_ptr<Coupling::Adapter::Coupling> coupsf_fsi_;
    //! coupling of structure and ale at the interface
    std::shared_ptr<Coupling::Adapter::Coupling> coupsa_fsi_;
    //! coupling of fluid and ale in the entire fluid volume
    std::shared_ptr<Coupling::Adapter::Coupling> coupfa_fsi_;
    //! coupling of all interface fluid and ale dofs
    std::shared_ptr<Coupling::Adapter::Coupling> icoupfa_fsi_;
    //! coupling of FPSI+FSI interface overlapping dofs of structure and freefluid
    std::shared_ptr<Coupling::Adapter::Coupling> iffcoupsf_fsi_;

    //! FPSI Coupling Object
    std::shared_ptr<FPSI::FpsiCoupling> fpsicoupl_;
  };
  // MonolithicBase

  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  //<<<<<<<<<<<<<<<<<<<<<<  MonolithicBase -> Monolithic  >>>>>>>>>>>>>>>>>>>>>
  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  class Monolithic : public MonolithicBase
  {
   public:
    //! ctor
    explicit Monolithic(MPI_Comm comm, const Teuchos::ParameterList& fpsidynparams,
        const Teuchos::ParameterList& poroelastdynparams);

    //! setup fpsi system
    void setup_system() override;

    //! setup fsi part of the system
    virtual void setup_system_fsi();

    //! perform time loop
    void timeloop() override;

    //! prepare time loop
    void prepare_timeloop();

    //! solve one time step
    virtual void time_step();

    //! perform result test
    void test_results(MPI_Comm comm) override;

    //! build RHS vector from sub fields
    virtual void setup_rhs(bool firstcall = false) = 0;

    //! build system matrix form sub fields + coupling
    virtual void setup_system_matrix(Core::LinAlg::BlockSparseMatrixBase& mat) = 0;
    //! build system matrix form sub fields + coupling
    virtual void setup_system_matrix() { setup_system_matrix(*system_matrix()); }

    //! access system matrix
    virtual std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> system_matrix() const = 0;

    /// setup solver
    void setup_solver() override;

    /// Recover the Lagrange multiplier at the interface   mayr.mt (03/2012)
    virtual void recover_lagrange_multiplier()
    {
      FOUR_C_THROW("recover_lagrange_multiplier: Not Implemented in Base Class!");
    }

    /// Extract specific columns from Sparse Matrix
    void extract_columnsfrom_sparse(Core::LinAlg::SparseMatrix& src,  ///< source Matrix
        const Core::LinAlg::Map& colmap,   ///< map with column gids to be extracted! (gid which are
                                           ///< not in the source Matrix will be ignored!)
        Core::LinAlg::SparseMatrix& dst);  ///< destination Matrix (will be filled!)

    //! Evaluate all fields at x^n+1 with x^n+1 = x_n + stepinc
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
            stepinc);  ///< increment between time step n and n+1

    //! setup of newton scheme
    void setup_newton();

    //! finite difference check for fpsi systemmatrix
    void fpsifd_check();

    //! solve linear system
    void linear_solve();

    //! solve using line search method
    void line_search(Core::LinAlg::SparseMatrix& sparse);

    //! create linear solver (setup of parameter lists, etc...)
    void create_linear_solver();

    //! build convergence norms after solve
    void build_convergence_norms();

    //! print header and results of newton iteration to screen
    void print_newton_iter();

    //! print header of newton iteration
    void print_newton_iter_header(FILE* ofile);

    //! print results of newton iteration
    void print_newton_iter_text(FILE* ofile);

    //! perform convergence check
    bool converged();

    //! full monolithic dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const
    {
      return blockrowdofmap_.full_map();
    }

    //! map of all dofs on Dirichlet-Boundary
    virtual std::shared_ptr<Core::LinAlg::Map> combined_dbc_map();

    //! extractor to communicate between full monolithic map and block maps
    const Core::LinAlg::MultiMapExtractor& extractor() const { return blockrowdofmap_; }

    //! set conductivity (for fps3i)
    void set_conductivity(double conduct);

    //! external access to rhs vector (used by xfpsi)
    std::shared_ptr<Core::LinAlg::Vector<double>>& rhs()
    {
      return rhs_;
    }  // TodoAge: will be removed again!

   protected:
    //! block systemmatrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> systemmatrix_;
    //! dof row map split in (field) blocks
    Core::LinAlg::MultiMapExtractor blockrowdofmap_;
    //! dof row map (not split)
    std::shared_ptr<Core::LinAlg::Map> fullmap_;
    //! increment between Newton steps k and k+1
    std::shared_ptr<Core::LinAlg::Vector<double>> iterinc_;
    std::shared_ptr<Core::LinAlg::Vector<double>> iterincold_;
    //! zero vector of full length
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;
    //! linear algebraic solver
    std::shared_ptr<Core::LinAlg::Solver> solver_;
    //! rhs of FPSI system
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;
    std::shared_ptr<Core::LinAlg::Vector<double>> rhsold_;

    std::shared_ptr<const Core::LinAlg::Vector<double>> meshdispold_;

    std::shared_ptr<Core::LinAlg::Vector<double>> porointerfacedisplacementsold_;

    //! adapt solver tolerancePoroField()->SystemSparseMatrix()
    bool solveradapttol_;
    bool linesearch_;
    double linesearch_counter;
    double solveradaptolbetter_;

    bool active_FD_check_;  // indicates if evaluate() is called from fd_check (firstiter should not
                            // be added anymore!!!)

    /// extract the three field vectors from a given composed vector
    /*
     \param x  (i) composed vector that contains all field vectors
     \param sx (o) poroelast dofs
     \param fx (o) free fluid velocities and pressure
     \param ax (o) ale displacements
     \param firstiter_ (i) firstiteration? - how to evaluate FSI-velocities
     */
    virtual void extract_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> x,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& pfx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& fx,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& ax, bool firstiter_) = 0;

    /// setup list with default parameters
    void set_default_parameters(const Teuchos::ParameterList& fpsidynparams);

    //! block ids of the monolithic system
    int porofluid_block_;
    int structure_block_;
    int fluid_block_;
    int ale_i_block_;

   private:
    //! flag for direct solver of linear system
    bool directsolve_;

    enum Inpar::FPSI::ConvergenceNorm normtypeinc_;
    enum Inpar::FPSI::ConvergenceNorm normtypefres_;
    enum Inpar::FPSI::BinaryOp combinedconvergence_;

    double toleranceiterinc_;
    double toleranceresidualforces_;
    std::vector<double>
        toleranceresidualforceslist_;  // order of fields: porofluidvelocity, porofluidpressure,
                                       // porostructure, fluidvelocity, fluidpressure, ale
    std::vector<double>
        toleranceiterinclist_;  // order of fields: porofluidvelocity, porofluidpressure,
                                // porostructure, fluidvelocity, fluidpressure, ale

    int maximumiterations_;
    int minimumiterations_;
    double normofrhs_;
    double normofrhsold_;
    double normofiterinc_;
    double normofiterincold_;

    double normrhsfluidvelocity_;
    double normrhsfluidpressure_;
    double normrhsporofluidvelocity_;
    double normrhsporofluidpressure_;
    double normrhsporointerface_;
    double normrhsfluidinterface_;
    double normrhsporostruct_;
    double normrhsfluid_;
    double normrhsale_;

    double normofiterincporostruct_;
    double normofiterincporofluid_;
    double normofiterincfluid_;
    double normofiterincporofluidvelocity_;
    double normofiterincporofluidpressure_;
    double normofiterincfluidvelocity_;
    double normofiterincfluidpressure_;
    double normofiterincale_;
    double normofiterincfluidinterface_;
    double normofiterincporointerface_;

    double sqrtnall_;  //!< squareroot of length of all dofs
    double sqrtnfv_;   //!< squareroot of length of fluid velocity dofs
    double sqrtnfp_;   //!< squareroot of length of fluid pressure dofs
    double sqrtnpfv_;  //!< squareroot of length of porofluid velocity dofs
    double sqrtnpfp_;  //!< squareroot of length of porofluid pressure dofs
    double sqrtnps_;   //!< squareroot of length of porostruct dofs
    double sqrtna_;    //!< squareroot of length of ale dofs

    double norm1_alldof_;  //!< sum of absolute values of all dofs
    double norm1_fv_;      //!< sum of absolute fluid velocity values
    double norm1_fp_;      //!< sum of absolute fluid pressure values
    double norm1_pfv_;     //!< sum of absolute poro fluid velocity values
    double norm1_pfp_;     //!< sum of absolute poro fluid pressure values
    double norm1_ps_;      //!< sum of absolute poro structural displacements values
    double norm1_a_;       //!< sum of absolute ale displacements values

    //! iteration step
    int iter_;

    int printscreen_;  ///> print infos to standard out every printscreen_ steps
    bool printiter_;   ///> print intermediate iterations during solution
    //! timer for solution technique
    Teuchos::Time timer_;

    bool isfirsttimestep_;

    //! hydraulic conductivity (needed for coupling in case of probtype fps3i)
    double conductivity_;

   protected:
    bool islinesearch_;
    //!  flag is true if this is the first Newton iteration, false otherwise
    bool firstcall_;
  };
  // class Monolithic

}  // namespace FPSI

FOUR_C_NAMESPACE_CLOSE

#endif
