/*-----------------------------------------------------------*/
/*! \file

\brief Global state data container for the structural (time)
       integration


\level 3

*/
/*-----------------------------------------------------------*/


#ifndef BACI_BEAMINTERACTION_STR_MODEL_EVALUATOR_DATASTATE_HPP
#define BACI_BEAMINTERACTION_STR_MODEL_EVALUATOR_DATASTATE_HPP

#include "baci_config.hpp"

#include "baci_timestepping_mstep.hpp"

#include <set>

// forward declaration
class Epetra_FEVector;

BACI_NAMESPACE_OPEN

namespace CORE::LINALG
{
  class SparseOperator;
  class SparseMatrix;
  class MultiMapExtractor;
}  // namespace CORE::LINALG

namespace DRT
{
  class Discretization;
  namespace ELEMENTS
  {
    class Beam3Base;
  }
}  // namespace DRT

namespace STR
{
  namespace MODELEVALUATOR
  {
    /** \brief Global state data container for the beaminteraction model
     *
     * This data container holds everything that needs to be updated each
     * iteration step
     */
    class BeamInteractionDataState
    {
     public:
      /// constructor
      BeamInteractionDataState();

      /// destructor
      virtual ~BeamInteractionDataState() = default;

      /// initialize class variables
      void Init();

      /// setup of the new class variables
      void Setup(Teuchos::RCP<const DRT::Discretization> const& ia_discret);

     protected:
      inline const bool& IsInit() const { return isinit_; };

      inline const bool& IsSetup() const { return issetup_; };

      inline void CheckInitSetup() const
      {
        if (!IsInit() or !IsSetup()) dserror("Call Init() and Setup() first!");
      }

      inline void CheckInit() const
      {
        if (!IsInit()) dserror("Init() has not been called, yet!");
      }

     public:
      /// @name General purpose algorithm members
      ///@{

      /// ID of actual processor in parallel
      int const& GetMyRank() const
      {
        CheckInitSetup();
        return myrank_;
      };
      ///@}

      /// @name search/interaction related stuff
      ///@{

      /// get extended bin to ele map
      std::map<int, std::set<int>> const& GetBinToRowEleMap() const
      {
        CheckInitSetup();
        return bintorowelemap_;
      };

      /// get mutable extended bin to ele map
      std::map<int, std::set<int>>& GetBinToRowEleMap()
      {
        CheckInitSetup();
        return bintorowelemap_;
      };

      /// get extended bin to ele map
      std::map<int, std::set<int>> const& GetExtendedBinToRowEleMap() const
      {
        CheckInitSetup();
        return exbintorowelemap_;
      };

      /// get mutable extended bin to ele map
      std::map<int, std::set<int>>& GetExtendedBinToRowEleMap()
      {
        CheckInitSetup();
        return exbintorowelemap_;
      };

      /// get extended ele to bin map
      std::map<int, std::set<int>> const& GetRowEleToBinMap() const
      {
        CheckInitSetup();
        return roweletobinmap_;
      };

      /// get extended ele to bin map
      std::set<int> const& GetRowEleToBinSet(int const i)
      {
        CheckInitSetup();
        return roweletobinmap_[i];
      };

      /// get mutable extended ele to bin map
      std::map<int, std::set<int>>& GetRowEleToBinMap()
      {
        CheckInitSetup();
        return roweletobinmap_;
      };
      ///@}

      /// @name Get state variables (read only access)
      ///@{

      /// Return displacements at the restart step \f$D_{restart}\f$
      Teuchos::RCP<const Epetra_Vector> GetDisRestart() const
      {
        CheckInitSetup();
        return dis_restart_;
      }

      /// Return displacements at the restart step \f$D_{restart}\f$
      Teuchos::RCP<const Epetra_Vector> GetDisRestartCol() const
      {
        CheckInitSetup();
        return dis_restart_col_;
      }

      /// Return displacements \f$D_{n+1}\f$
      Teuchos::RCP<const Epetra_Vector> GetDisNp() const
      {
        CheckInitSetup();
        return disnp_;
      }

      /// Return displacements \f$D_{n+1}\f$
      Teuchos::RCP<const Epetra_Vector> GetDisColNp() const
      {
        CheckInitSetup();
        return discolnp_;
      }

      /// Return displacements \f$D_{n}\f$
      Teuchos::RCP<const Epetra_Vector> GetDisN() const
      {
        CheckInitSetup();
        return (*dis_)(0);
      }

      /// Return internal force \f$fint_{n}\f$
      Teuchos::RCP<const Epetra_FEVector> GetForceN() const
      {
        CheckInitSetup();
        return forcen_;
      }

      /// Return internal force \f$fint_{n+1}\f$
      Teuchos::RCP<const Epetra_FEVector> GetForceNp() const
      {
        CheckInitSetup();
        return forcenp_;
      }

      /// @name Get system matrices (read only access)
      ///@{
      /// returns the entire structural jacobian
      Teuchos::RCP<const CORE::LINALG::SparseMatrix> GetStiff() const
      {
        CheckInitSetup();
        return stiff_;
      }
      ///@}

      /// @name Get mutable state variables (read and write access)
      ///@{

      /// Return displacements at the restart step \f$D_{restart}\f$
      Teuchos::RCP<Epetra_Vector>& GetDisRestart()
      {
        CheckInitSetup();
        return dis_restart_;
      }

      /// Return displacements at the restart step \f$D_{restart}\f$
      Teuchos::RCP<Epetra_Vector>& GetDisRestartCol()
      {
        CheckInitSetup();
        return dis_restart_col_;
      }

      /// Return displacements \f$D_{n+1}\f$
      Teuchos::RCP<Epetra_Vector>& GetDisNp()
      {
        CheckInitSetup();
        return disnp_;
      }

      /// Return displacements \f$D_{n+1}\f$
      Teuchos::RCP<Epetra_Vector>& GetDisColNp()
      {
        CheckInitSetup();
        return discolnp_;
      }

      /// Return displacements \f$D_{n}\f$
      Teuchos::RCP<Epetra_Vector> GetDisN()
      {
        CheckInitSetup();
        return (*dis_)(0);
      }

      /// Return multi-displacement vector \f$D_{n}, D_{n-1}, ...\f$
      Teuchos::RCP<TIMESTEPPING::TimIntMStep<Epetra_Vector>> GetMultiDis()
      {
        CheckInitSetup();
        return dis_;
      }

      /// Return internal force \f$fint_{n}\f$
      Teuchos::RCP<Epetra_FEVector>& GetForceN()
      {
        CheckInitSetup();
        return forcen_;
      }

      /// Return internal force \f$fint_{n+1}\f$
      Teuchos::RCP<Epetra_FEVector>& GetForceNp()
      {
        CheckInitSetup();
        return forcenp_;
      }

      ///@}

      /// @name Get mutable system matrices
      ///@{
      /// returns the entire structural jacobian
      Teuchos::RCP<CORE::LINALG::SparseMatrix>& GetStiff()
      {
        CheckInitSetup();
        return stiff_;
      }

      /// Return the restart coupling flag.
      bool GetRestartCouplingFlag() const
      {
        CheckInitSetup();
        return is_restart_coupling_;
      }

      /// Set the restart coupling flag.
      void SetRestartCouplingFlag(const bool is_restart_coupling)
      {
        is_restart_coupling_ = is_restart_coupling;
      }

     protected:
      /// @name variables for internal use only
      ///@{
      /// flag indicating if Init() has been called
      bool isinit_;

      /// flag indicating if Setup() has been called
      bool issetup_;
      ///@}

     private:
      /// @name General purpose algorithm members
      ///@{

      /// ID of actual processor in parallel
      int myrank_;

      ///@}

      /// @name search/interaction related stuff
      ///@{

      //! bin to ele map
      std::map<int, std::set<int>> bintorowelemap_;

      //! extended bin to ele map
      std::map<int, std::set<int>> exbintorowelemap_;

      //! extended ele to bin map
      std::map<int, std::set<int>> roweletobinmap_;

      //! element
      Teuchos::RCP<CORE::LINALG::MultiMapExtractor> rowelemapextractor_;
      ///@}

      /// @name Global state vectors
      ///@{

      /// global displacements \f${D}_{n}, D_{n-1}, ...\f$
      Teuchos::RCP<TIMESTEPPING::TimIntMStep<Epetra_Vector>> dis_;

      /// global displacements at the restart step \f${D}_{restart}\f$ at \f$t_{restart}\f$
      Teuchos::RCP<Epetra_Vector> dis_restart_;

      /// global displacements at the restart step \f${D}_{restart}\f$ at \f$t_{restart}\f$. This
      /// vector will be used to export disrestart_ to the current partitioning.
      Teuchos::RCP<Epetra_Vector> dis_restart_col_;

      /// flag if coupling, i.e. mesh tying terms should be evaluated at the restart configuration.
      /// This is stored here, since it is directly related to the vectors dis_restart_ and
      /// dis_restart_col_.
      bool is_restart_coupling_;

      /// global displacements \f${D}_{n+1}\f$ at \f$t_{n+1}\f$
      Teuchos::RCP<Epetra_Vector> disnp_;

      /// global displacements \f${D}_{n+1}\f$ at \f$t_{n+1}\f$
      Teuchos::RCP<Epetra_Vector> discolnp_;

      /// global internal force vector at \f$t_{n}\f$
      Teuchos::RCP<Epetra_FEVector> forcen_;

      /// global internal force vector at \f$t_{n+1}\f$
      Teuchos::RCP<Epetra_FEVector> forcenp_;
      ///@}

      /// @name System matrices
      ///@{
      /// supposed to hold the entire jacobian (saddle point system if desired)
      Teuchos::RCP<CORE::LINALG::SparseMatrix> stiff_;

      ///@}
    };
  }  // namespace MODELEVALUATOR
}  // namespace STR


BACI_NAMESPACE_CLOSE

#endif