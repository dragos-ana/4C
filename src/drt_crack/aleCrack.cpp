/*----------------------------------------------------------------------*/
/*!
\file aleCrack.cpp

\brief Perform all operations related to ALE cracking

<pre>
Maintainer: Sudhakar
            sudhakar@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15257
</pre>
*/
/*----------------------------------------------------------------------*/

#include "aleCrack.H"
#include "crack_tolerance.H"
#include "crackUtils.H"

#include "../linalg/linalg_utils.H"
#include "../drt_ale/ale_utils_clonestrategy.H"
#include "../drt_ale/ale.H"
#include "../drt_lib/drt_utils_createdis.H"
#include "../drt_fs3i/biofilm_fsi_utils.H"


/*---------------------------------------------------------------------------------------------------*
 * Constructor                                                                             sudhakar 06/14
 *---------------------------------------------------------------------------------------------------*/
DRT::CRACK::aleCrack::aleCrack( Teuchos::RCP<DRT::Discretization> dis )
:structdis_( dis )
{
  Teuchos::RCP<DRT::Discretization> aledis = Teuchos::null;
  aledis = DRT::Problem::Instance()->GetDis("ale");
  if (!aledis->Filled()) aledis->FillComplete();

  DRT::UTILS::CloneDiscretization<ALE::UTILS::AleCloneStrategy>(structdis_,aledis);
  if( aledis->NumGlobalElements() == 0 )
    dserror( "No elements found in ALE discretization" );

  Teuchos::RCP<ALE::AleBaseAlgorithm> ale
        = Teuchos::rcp(new ALE::AleBaseAlgorithm(DRT::Problem::Instance()->StructuralDynamicParams(), DRT::Problem::Instance()->GetDis("ale")));
  ale_ = ale->AleFieldrcp();

  myrank_ = structdis_->Comm().MyPID();
}

/*---------------------------------------------------------------------------------------------------*
 * Perform all operations of ALE step                                                      sudhakar 07/14
 *---------------------------------------------------------------------------------------------------*/
void DRT::CRACK::aleCrack::ALE_step( const std::map<int, std::vector<double> >& ale_bc_nodes,
                                     const std::set<int>& new_ale_bc )
{
  // should have at least two, one for each plane
  if( ale_bc_nodes.size() < 2 )
    dserror( "No tip displacement supplied for ALE operation\n" );

  // confirm at least at one node we have sufficient displacement BC
  if( not sufficientMovement( ale_bc_nodes ) )
    return;

  modifyConditions( ale_bc_nodes, new_ale_bc );

  ALE_Solve();

  //Teuchos::RCP<Epetra_Vector> ale_disp_vec = getDisALE( ale_bc_nodes );
}

/*---------------------------------------------------------------------------------------------------*
 * ALE step is performed only if displacement expected at least at one node                sudhakar 07/14
 * is higher than the predefined tolerance
 *---------------------------------------------------------------------------------------------------*/
bool DRT::CRACK::aleCrack::sufficientMovement( const std::map<int, std::vector<double> >& ale_bc_nodes )
{
  for( std::map<int, std::vector<double> >::const_iterator it = ale_bc_nodes.begin(); it != ale_bc_nodes.end(); it++ )
  {
    const std::vector<double> disp = it->second;
    for( unsigned i=0; i<3; i++ )
    {
      if( fabs( disp[i] ) > ALE_LEN_TOL )
        return true;
    }
  }

  return false;
}

/*Teuchos::RCP<Epetra_Vector> DRT::CRACK::aleCrack::getDisALE( const std::map<int, std::vector<double> >& ale_bc_nodes )
{
  Teuchos::RCP<Epetra_Vector> vec = LINALG::CreateVector(*structdis_->DofRowMap(),true);

  for( std::map<int, std::vector<double> >::const_iterator it = ale_bc_nodes.begin(); it != ale_bc_nodes.end(); it++ )
  {
    const int nodeid = it->first;
    if( structdis_->HaveGlobalNode( nodeid ) )
    {
      DRT::Node * tipnode = structdis_->gNode( nodeid );
      if( tipnode->Owner() == myrank_ )
      {
        const std::vector<double> disp = it->second;
        std::vector<int> lm;
        structdis_->Dof( tipnode, lm );

        for( unsigned i = 0; i < 3; i++ )
          (*vec)[lm[i]] = disp[i];
      }
    }
  }

  return vec;
}*/


/*---------------------------------------------------------------------------------------------------*
 * Set ALE displacement conditions are new crack tip nodes                                   sudhakar 06/14
 * This moves the nodes to desired location so that crack propagation angle is maintained
 *---------------------------------------------------------------------------------------------------*/
void DRT::CRACK::aleCrack::modifyConditions( const std::map<int, std::vector<double> >& ale_bc_nodes,
                                             const std::set<int>& new_ale_bc )
{
  if( new_ale_bc.size() > 0 )
  {
    std::vector<Teuchos::RCP<Condition> > alldiri;
    ale_->Discretization()->GetCondition( "Dirichlet", alldiri );
    for( unsigned i=0; i<alldiri.size(); i++ )
    {
      Teuchos::RCP<Condition> cond = alldiri[i];
      DRT::CRACK::UTILS::addNodesToConditions( &(*cond), new_ale_bc );
    }
  }

  DRT::CRACK::UTILS::AddConditions( ale_->Discretization(), ale_bc_nodes );
  ale_->Discretization()->FillComplete();

  // just adding the Conditions alone do not get the work done
  // we need to build the Dirichlet boundary condition maps again for these conditions to be implemented
  ale_->SetupDBCMapEx( false );
}

/*---------------------------------------------------------------------------------------------------*
 * Build ALE system matrix and solve the system                                              sudhakar 06/14
 *---------------------------------------------------------------------------------------------------*/
void DRT::CRACK::aleCrack::ALE_Solve()
{
  ale_->BuildSystemMatrix();

  // No need because we added this part in the condition???
  //ale_->ApplyInterfaceDisplacements( ale_disp_vec );

  ale_->Solve();

  Teuchos::RCP<const Epetra_Vector> ale_final = ale_->Dispnp();

  // convert displacement in ALE field into structural field
  Teuchos::RCP<Epetra_Vector> ale_str = Teuchos::rcp(new Epetra_Vector(*(structdis_->DofRowMap())));
  for( int i=0; i<ale_str->MyLength(); i++ )
    (*ale_str)[i] = (*ale_final)[i];

  FS3I::Biofilm::UTILS::updateMaterialConfigWithALE_Disp( structdis_, ale_str );

  ale_->Discretization()->ClearDiscret();

  DRT::CRACK::UTILS::deleteConditions( ale_->Discretization() );
  ale_->Discretization()->FillComplete();
}