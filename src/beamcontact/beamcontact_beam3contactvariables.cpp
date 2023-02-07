/*----------------------------------------------------------------------------*/
/*! \file

\brief One beam contact segment living on an element pair

\level 2

*/
/*----------------------------------------------------------------------------*/

#include "beamcontact_beam3contact.H"
#include "beamcontact_beam3contactvariables.H"
#include "beaminteraction_beam3contact_utils.H"
#include "inpar_beamcontact.H"
#include "inpar_contact.H"
#include "lib_discret.H"
#include "lib_exporter.H"
#include "lib_dserror.H"
#include "linalg_utils_sparse_algebra_math.H"
#include "fem_general_utils_fem_shapefunctions.H"
#include "lib_globalproblem.H"

#include "structure_timint_impl.H"
#include "beam3.H"
#include "beam3_reissner.H"
#include "beam3_euler_bernoulli.H"

#include "Teuchos_TimeMonitor.hpp"
#include "beaminteraction_beam3contact_defines.H"
#include "beaminteraction_beam3contact_tangentsmoothing.H"

/*----------------------------------------------------------------------*
 |  constructor (public)                                     meier 01/14|
 *----------------------------------------------------------------------*/
template <const int numnodes, const int numnodalvalues>
CONTACT::Beam3contactvariables<numnodes, numnodalvalues>::Beam3contactvariables(
    std::pair<TYPE, TYPE>& closestpoint, std::pair<int, int>& segids, std::pair<int, int>& intids,
    const double& pp, TYPE jacobi)
    : closestpoint_(closestpoint),
      segids_(segids),
      intids_(intids),
      jacobi_(jacobi),
      gap_(0.0),
      normal_(LINALG::Matrix<3, 1, TYPE>(true)),
      pp_(pp),
      ppfac_(0.0),
      dppfac_(0.0),
      fp_(0.0),
      dfp_(0.0),
      energy_(0.0),
      integratedenergy_(0.0),
      angle_(0.0)
{
  return;
}
/*----------------------------------------------------------------------*
 |  end: constructor
 *----------------------------------------------------------------------*/

// Possible template cases: this is necessary for the compiler
template class CONTACT::Beam3contactvariables<2, 1>;
template class CONTACT::Beam3contactvariables<3, 1>;
template class CONTACT::Beam3contactvariables<4, 1>;
template class CONTACT::Beam3contactvariables<5, 1>;
template class CONTACT::Beam3contactvariables<2, 2>;