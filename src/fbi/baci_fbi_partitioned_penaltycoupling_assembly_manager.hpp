/*-----------------------------------------------------------*/
/*! \file

\brief Class to assemble pair based contributions into global coupling matrices.


\level 1

*/
/*-----------------------------------------------------------*/


#ifndef BACI_FBI_PARTITIONED_PENALTYCOUPLING_ASSEMBLY_MANAGER_HPP
#define BACI_FBI_PARTITIONED_PENALTYCOUPLING_ASSEMBLY_MANAGER_HPP

#include "baci_config.hpp"

#include "baci_utils_exceptions.hpp"

#include <Teuchos_RCP.hpp>

#include <vector>

// Forward declarations.
class Epetra_FEVector;
class Epetra_Vector;

BACI_NAMESPACE_OPEN

namespace CORE::LINALG
{
  class SparseMatrix;
  class SparseOperator;
}  // namespace CORE::LINALG
namespace DRT
{
  class Discretization;
}
namespace BEAMINTERACTION
{
  class BeamContactPair;
}

namespace BEAMINTERACTION
{
  namespace SUBMODELEVALUATOR
  {
    /**
     * \brief This class assembles the contribution of beam contact pairs into the global force
     * vector and stiffness matrix for partitioned algorithms. The method EvaluateForceStiff has to
     * be overloaded in the derived classes to implement the correct assembly method.
     */
    class PartitionedBeamInteractionAssemblyManager
    {
     public:
      /**
       * \brief Constructor.
       * \param[in] assembly_contact_elepairs Vector with element pairs to be evaluated by this
       * class.
       */
      PartitionedBeamInteractionAssemblyManager(
          std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>>& assembly_contact_elepairs);

      /**
       * \brief Destructor.
       */
      virtual ~PartitionedBeamInteractionAssemblyManager() = default;

      /**
       * \brief Evaluate all force and stiffness terms and add them to the global matrices.
       * \param[in] discret (in) Pointer to the disretization.
       * \params[inout] ff Global force vector acting on the fluid
       * \params[inout] fb Global force vector acting on the beam
       * \params[inout] cff  Global stiffness matrix coupling fluid to fluid DOFs
       * \params[inout] cbb  Global stiffness matrix coupling beam to beam DOFs
       * \params[inout] cfb  Global stiffness matrix coupling beam to fluid DOFs
       * \params[inout] cbf  Global stiffness matrix coupling fluid to beam DOFs
       */
      virtual void EvaluateForceStiff(const DRT::Discretization& discretization1,
          const DRT::Discretization& discretization2, Teuchos::RCP<Epetra_FEVector>& ff,
          Teuchos::RCP<Epetra_FEVector>& fb, Teuchos::RCP<CORE::LINALG::SparseOperator> cff,
          Teuchos::RCP<CORE::LINALG::SparseMatrix>& cbb,
          Teuchos::RCP<CORE::LINALG::SparseMatrix>& cfb,
          Teuchos::RCP<CORE::LINALG::SparseMatrix>& cbf,
          Teuchos::RCP<const Epetra_Vector> fluid_vel,
          Teuchos::RCP<const Epetra_Vector> beam_vel) = 0;

     protected:
      //! Vector of pairs to be evaluated by this class.
      std::vector<Teuchos::RCP<BEAMINTERACTION::BeamContactPair>> assembly_contact_elepairs_;
    };

  }  // namespace SUBMODELEVALUATOR
}  // namespace BEAMINTERACTION

BACI_NAMESPACE_CLOSE

#endif