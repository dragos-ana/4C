/*----------------------------------------------------------------------*/
/*!
 \file scatra_timint_poromulti.cpp

 \brief time integration schemes for scalar transport within multiphase porous medium

   \level 3

   \maintainer  Anh-Tu Vuong
                vuong@lnm.mw.tum.de
                http://www.lnm.mw.tum.de
                089 - 289-15251
 *----------------------------------------------------------------------*/


#include "scatra_timint_poromulti.H"

#include "../drt_io/io.H"

/*----------------------------------------------------------------------*
 | constructor                                             vuong  08/16 |
 *----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMulti::ScaTraTimIntPoroMulti(
        Teuchos::RCP<DRT::Discretization>        dis,
        Teuchos::RCP<LINALG::Solver>             solver,
        Teuchos::RCP<Teuchos::ParameterList>     params,
        Teuchos::RCP<Teuchos::ParameterList>     sctratimintparams,
        Teuchos::RCP<Teuchos::ParameterList>     extraparams,
        Teuchos::RCP<IO::DiscretizationWriter>   output)
  : ScaTraTimIntImpl(dis,solver,sctratimintparams,extraparams,output),
    nds_sat_(-1),
    nds_solid_pressure_(-1)
{
  // DO NOT DEFINE ANY STATE VECTORS HERE (i.e., vectors based on row or column maps)
  // this is important since we have problems which require an extended ghosting
  // this has to be done before all state vectors are initialized
  return;
}


/*----------------------------------------------------------------------*
 | initialize algorithm                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMulti::Init()
{
  return;
}

/*----------------------------------------------------------------------*
 | set solution fields on given dof sets                    vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMulti::SetSolutionFields(
    Teuchos::RCP<const Epetra_MultiVector>   multiflux,
    const int                                nds_flux,
    Teuchos::RCP<const Epetra_Vector>        pressure,
    const int                                nds_pres,
    Teuchos::RCP<const Epetra_Vector>        saturation,
    const int                                nds_sat,
    Teuchos::RCP<const Epetra_Vector>        solid_pressure,
    const int                                nds_solid_pressure
    )
{
  // safety check
  if (nds_flux >= discret_->NumDofSets())
      dserror("Too few dofsets on scatra discretization!");
  if (nds_pres >= discret_->NumDofSets())
      dserror("Too few dofsets on scatra discretization!");
  if (nds_sat >= discret_->NumDofSets())
      dserror("Too few dofsets on scatra discretization!");
  if (nds_solid_pressure >= discret_->NumDofSets())
      dserror("Too few dofsets on scatra discretization!");

  // store number of dof-set
  nds_pres_ = nds_pres;
  // provide scatra discretization with pressure field
  discret_->SetState(nds_pres_,"pressure",pressure);

  // store number of dof-set
  nds_sat_ = nds_sat;
  // provide scatra discretization with pressure field
  discret_->SetState(nds_pres_,"saturation",saturation);

  // store number of dof-set
  nds_solid_pressure_ = nds_solid_pressure;
  // provide scatra discretization with solid pressure field
  discret_->SetState(nds_solid_pressure_,"solid_pressure",solid_pressure);

  // store number of dof-set associated with velocity related dofs
  nds_vel_ = nds_flux;

  if(multiflux->NumVectors()%nsd_!=0)
    dserror("Unexpected length of flux vector: %i",multiflux->NumVectors());

  const int numphases = multiflux->NumVectors() / nsd_;

  std::string stateprefix = "flux";

  for(int curphase=0;curphase<numphases;++curphase)
  {
    // initialize velocity vectors
    Teuchos::RCP<Epetra_Vector> phaseflux = LINALG::CreateVector(*discret_->DofRowMap(nds_flux),true);

    std::stringstream statename;
    statename << stateprefix << curphase;

    // loop all nodes on the processor
    for(int lnodeid=0;lnodeid<discret_->NumMyRowNodes();lnodeid++)
    {
      // get the processor local node
      DRT::Node*  lnode      = discret_->lRowNode(lnodeid);

      // get dofs associated with current node
      std::vector<int> nodedofs = discret_->Dof(nds_flux,lnode);

      if((int)nodedofs.size()!=nsd_)
        dserror("Expected number of DOFs to be equal to the number of space dimensions for flux state!");

      for(int index=0;index<nsd_;++index)
      {
        // get global and local dof IDs
        const int gid = nodedofs[index];
        const int lid = phaseflux->Map().LID(gid);
        if (lid < 0) dserror("Local ID not found in map for given global ID!");

        const double value = (*(*multiflux)(curphase*nsd_+index))[lnodeid];

        int err = phaseflux->ReplaceMyValue(lid, 0, value);
        if (err!=0) dserror("error while inserting a value into convel");
      }
    }

    // provide scatra discretization with convective velocity
    discret_->SetState(nds_flux,statename.str(),phaseflux);
  }

  return;

} // ScaTraTimIntImpl::SetSolutionFields

/*----------------------------------------------------------------------*
 | add parameters depending on the problem                  vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMulti::AddProblemSpecificParametersAndVectors(
  Teuchos::ParameterList& params //!< parameter list
)
{
  // set dof set numbers
  // note: the velocity dof set is set by the standard time integrator

  // provide pressure field
  params.set<int>("ndspres",nds_pres_);
  // provide saturation field
  params.set<int>("ndssat",nds_sat_);
  // provide solid pressure field
  params.set<int>("ndssolidpressure",nds_solid_pressure_);

  return;
}

/*----------------------------------------------------------------------*
 |  write current state to BINIO                           vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMulti::OutputState()
{
  // solution
  output_->WriteVector("phinp", phinp_);

//  // convective velocity (written in case of coupled simulations since volmortar is now possible)
//  if ( cdvel_ == INPAR::SCATRA::velocity_function or cdvel_ == INPAR::SCATRA::velocity_function_and_curve or cdvel_ == INPAR::SCATRA::velocity_Navier_Stokes)
//  {
//    Teuchos::RCP<const Epetra_Vector> convel = discret_->GetState(nds_vel_, "convective velocity field");
//    if(convel == Teuchos::null)
//      dserror("Cannot get state vector convective velocity");
//
//    // convert dof-based Epetra vector into node-based Epetra multi-vector for postprocessing
//    Teuchos::RCP<Epetra_MultiVector> convel_multi = Teuchos::rcp(new Epetra_MultiVector(*discret_->NodeRowMap(),nsd_,true));
//    for (int inode=0; inode<discret_->NumMyRowNodes(); ++inode)
//    {
//      DRT::Node* node = discret_->lRowNode(inode);
//      for (int idim=0; idim<nsd_; ++idim)
//        (*convel_multi)[idim][inode] = (*convel)[convel->Map().LID(discret_->Dof(nds_vel_,node,idim))];
//    }
//
//    output_->WriteVector("convec_velocity", convel_multi, IO::DiscretizationWriter::nodevector);
//  }

  // displacement field
  if (isale_)
  {
    Teuchos::RCP<const Epetra_Vector> dispnp = discret_->GetState(nds_disp_,"dispnp");
    if (dispnp == Teuchos::null)
      dserror("Cannot extract displacement field from discretization");

    // convert dof-based Epetra vector into node-based Epetra multi-vector for postprocessing
    Teuchos::RCP<Epetra_MultiVector> dispnp_multi = Teuchos::rcp(new Epetra_MultiVector(*discret_->NodeRowMap(),nsd_,true));
    for (int inode=0; inode<discret_->NumMyRowNodes(); ++inode)
    {
      DRT::Node* node = discret_->lRowNode(inode);
      for (int idim=0; idim<nsd_; ++idim)
        (*dispnp_multi)[idim][inode] = (*dispnp)[dispnp->Map().LID(discret_->Dof(nds_disp_,node,idim))];
    }

    output_->WriteVector("dispnp", dispnp_multi, IO::DiscretizationWriter::nodevector);
  }

  return;
} // ScaTraTimIntImpl::OutputState

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMultiOST::ScaTraTimIntPoroMultiOST(
  Teuchos::RCP<DRT::Discretization>      actdis,
  Teuchos::RCP<LINALG::Solver>           solver,
  Teuchos::RCP<Teuchos::ParameterList>   params,
  Teuchos::RCP<Teuchos::ParameterList>   sctratimintparams,
  Teuchos::RCP<Teuchos::ParameterList>   extraparams,
  Teuchos::RCP<IO::DiscretizationWriter> output)
  : ScaTraTimIntImpl(actdis,solver,sctratimintparams,extraparams,output),
    ScaTraTimIntPoroMulti(actdis,solver,params,sctratimintparams,extraparams,output),
    TimIntOneStepTheta(actdis,solver,sctratimintparams,extraparams,output)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMultiOST::Init()
{
  // call Init()-functions of base classes
  // note: this order is important
  TimIntOneStepTheta::Init();
  ScaTraTimIntPoroMulti::Init();

  return;
}


/*----------------------------------------------------------------------*
| Destructor dtor (public)                                  vuong  08/16 |
*-----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMultiOST::~ScaTraTimIntPoroMultiOST()
{
  return;
}

/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMultiOST::Update(const int num)
{
  TimIntOneStepTheta::Update(num);
  ScaTraTimIntPoroMulti::Update(num);

  return;
}

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMultiBDF2::ScaTraTimIntPoroMultiBDF2(
  Teuchos::RCP<DRT::Discretization>      actdis,
  Teuchos::RCP<LINALG::Solver>           solver,
  Teuchos::RCP<Teuchos::ParameterList>   params,
  Teuchos::RCP<Teuchos::ParameterList>   sctratimintparams,
  Teuchos::RCP<Teuchos::ParameterList>   extraparams,
  Teuchos::RCP<IO::DiscretizationWriter> output)
  : ScaTraTimIntImpl(actdis,solver,sctratimintparams,extraparams,output),
    ScaTraTimIntPoroMulti(actdis,solver,params,sctratimintparams,extraparams,output),
    TimIntBDF2(actdis,solver,sctratimintparams,extraparams,output)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMultiBDF2::Init()
{
  // call Init()-functions of base classes
  // note: this order is important
  TimIntBDF2::Init();
  ScaTraTimIntPoroMulti::Init();

  return;
}


/*----------------------------------------------------------------------*
| Destructor dtor (public)                                  vuong  08/16 |
*-----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMultiBDF2::~ScaTraTimIntPoroMultiBDF2()
{
  return;
}

/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMultiBDF2::Update(const int num)
{
  TimIntBDF2::Update(num);
  ScaTraTimIntPoroMulti::Update(num);

  return;
}


/*----------------------------------------------------------------------*
 |  Constructor (public)                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMultiGenAlpha::ScaTraTimIntPoroMultiGenAlpha(
  Teuchos::RCP<DRT::Discretization>      actdis,
  Teuchos::RCP<LINALG::Solver>           solver,
  Teuchos::RCP<Teuchos::ParameterList>   params,
  Teuchos::RCP<Teuchos::ParameterList>   sctratimintparams,
  Teuchos::RCP<Teuchos::ParameterList>   extraparams,
  Teuchos::RCP<IO::DiscretizationWriter> output)
  : ScaTraTimIntImpl(actdis,solver,sctratimintparams,extraparams,output),
    ScaTraTimIntPoroMulti(actdis,solver,params,sctratimintparams,extraparams,output),
    TimIntGenAlpha(actdis,solver,sctratimintparams,extraparams,output)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMultiGenAlpha::Init()
{
  // call Init()-functions of base classes
  // note: this order is important
  TimIntGenAlpha::Init();
  ScaTraTimIntPoroMulti::Init();

  return;
}


/*----------------------------------------------------------------------*
| Destructor dtor (public)                                  vuong  08/16 |
*-----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMultiGenAlpha::~ScaTraTimIntPoroMultiGenAlpha()
{
  return;
}

/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                            gjb 08/08 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMultiGenAlpha::Update(const int num)
{
  TimIntGenAlpha::Update(num);
  ScaTraTimIntPoroMulti::Update(num);

  return;
}

/*----------------------------------------------------------------------*
 |  Constructor (public)                                    vuong  08/16 |
 *----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMultiStationary::ScaTraTimIntPoroMultiStationary(
  Teuchos::RCP<DRT::Discretization>      actdis,
  Teuchos::RCP<LINALG::Solver>           solver,
  Teuchos::RCP<Teuchos::ParameterList>   params,
  Teuchos::RCP<Teuchos::ParameterList>   sctratimintparams,
  Teuchos::RCP<Teuchos::ParameterList>   extraparams,
  Teuchos::RCP<IO::DiscretizationWriter> output)
  : ScaTraTimIntImpl(actdis,solver,sctratimintparams,extraparams,output),
    ScaTraTimIntPoroMulti(actdis,solver,params,sctratimintparams,extraparams,output),
    TimIntStationary(actdis,solver,sctratimintparams,extraparams,output)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize time integration                             vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMultiStationary::Init()
{
  // call Init()-functions of base classes
  // note: this order is important
  TimIntStationary::Init();
  ScaTraTimIntPoroMulti::Init();

  return;
}


/*----------------------------------------------------------------------*
| Destructor dtor (public)                                  vuong  08/16 |
*-----------------------------------------------------------------------*/
SCATRA::ScaTraTimIntPoroMultiStationary::~ScaTraTimIntPoroMultiStationary()
{
  return;
}

/*----------------------------------------------------------------------*
 | current solution becomes most recent solution of next timestep       |
 |                                                         vuong  08/16 |
 *----------------------------------------------------------------------*/
void SCATRA::ScaTraTimIntPoroMultiStationary::Update(const int num)
{
  TimIntStationary::Update(num);
  ScaTraTimIntPoroMulti::Update(num);

  return;
}
