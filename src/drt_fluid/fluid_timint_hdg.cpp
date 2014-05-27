/*----------------------------------------------------------------------*/
/*!
\file fluid_timint_hdg.cpp
\brief HDG time-integration scheme

<pre>
Maintainers: Ursula Rasthofer & Martin Kronbichler
             {rasthofer,kronbichler}@lnm.mw.tum.de
             http://www.lnm.mw.tum.de
             089 - 289-15236/-235
</pre>
*/
/*----------------------------------------------------------------------*/

#include "fluid_timint_hdg.H"
#include "../drt_lib/drt_discret_hdg.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_fluid_ele/fluid_ele_hdg.H"
#include "../drt_fluid_ele/fluid_ele_action.H"
#include "../linalg/linalg_utils.H"
#include "../drt_io/io.H"


/*----------------------------------------------------------------------*
 |  Constructor (public)                              kronbichler 05/14 |
 *----------------------------------------------------------------------*/
FLD::TimIntHDG::TimIntHDG(const Teuchos::RCP<DRT::Discretization>&      actdis,
                          const Teuchos::RCP<LINALG::Solver>&           solver,
                          const Teuchos::RCP<Teuchos::ParameterList>&   params,
                          const Teuchos::RCP<IO::DiscretizationWriter>& output,
                          bool                                          alefluid /*= false*/)
: FluidImplicitTimeInt(actdis,solver,params,output,alefluid),
  TimIntGenAlpha(actdis,solver,params,output,alefluid),
  firstAssembly_(false)
{
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                              kronbichler 05/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::Init()
{
  DRT::DiscretizationHDG* hdgdis = dynamic_cast<DRT::DiscretizationHDG*>(discret_.get());
  if (hdgdis == NULL)
    dserror("Did not receive an HDG discretization");

  const int nsd = DRT::Problem::Instance()->NDim();
  int degreep1 = DRT::ELEMENTS::FluidHDG::degree + 1;
  int nscalardofs = 1;
  for(int i=0; i<nsd; ++i)
    nscalardofs *= degreep1;

  int elementndof = nscalardofs * ( nsd * nsd + nsd + 1);

  // set degrees of freedom in the discretization
  discret_->BuildDofSetAuxProxy(0,elementndof,0,false);
  discret_->FillComplete();

  // build velocity/pressure splitting
  std::set<int> conddofset;
  std::set<int> otherdofset;

  for (int j=0; j<hdgdis->NumMyRowElements(); ++j)
  {
    std::vector<int> dof = hdgdis->Dof(0,hdgdis->lRowElement(j));
    dsassert(dof.size() >= 1, "Internal error: could not find HDG pressure dof");
    for (unsigned int i=0; i<dof.size(); ++i)
      conddofset.insert(dof[i]);
  }
  for (int i=0; i<hdgdis->NumMyRowFaces(); ++i)
  {
    std::vector<int> dof = hdgdis->Dof(0,hdgdis->lRowFace(i));
    for (unsigned int j=0; j<dof.size(); ++j)
      otherdofset.insert(dof[j]);
  }

  std::vector<int> conddofmapvec;
  conddofmapvec.reserve(conddofset.size());
  conddofmapvec.assign(conddofset.begin(), conddofset.end());
  conddofset.clear();
  Teuchos::RCP<Epetra_Map> conddofmap =
    Teuchos::rcp(new Epetra_Map(-1,
                                conddofmapvec.size(),
                                &conddofmapvec[0],
                                0,
                                hdgdis->Comm()));
  std::vector<int> otherdofmapvec;
  otherdofmapvec.reserve(otherdofset.size());
  otherdofmapvec.assign(otherdofset.begin(), otherdofset.end());
  otherdofset.clear();
  Teuchos::RCP<Epetra_Map> otherdofmap =
    Teuchos::rcp(new Epetra_Map(-1,
                                otherdofmapvec.size(),
                                &otherdofmapvec[0],
                                0,
                                hdgdis->Comm()));
  velpressplitter_.Setup(*hdgdis->DofRowMap(),conddofmap, otherdofmap);

  // implement ost and bdf2 through gen-alpha facilities
  if (timealgo_ == INPAR::FLUID::timeint_bdf2)
  {
    alphaM_ = 1.5;
    alphaF_ = 1.0;
    gamma_ = 1.0;
  }
  else if (timealgo_ == INPAR::FLUID::timeint_one_step_theta)
  {
    alphaM_ = 1.0;
    alphaF_ = 1.0;
    gamma_ = params_->get<double>("theta");
  }
  else if (timealgo_ == INPAR::FLUID::timeint_stationary)
    dserror("Stationary case not implemented for HDG");

  timealgoset_ = timealgo_;
  timealgo_ = INPAR::FLUID::timeint_afgenalpha;

  // call Init()-functions of base classes
  // note: this order is important
  FLD::TimIntGenAlpha::Init();
}



/*----------------------------------------------------------------------*
| calculate pseudo-theta for startalgo_, modified for HDG  kronbi 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::SetTheta()
{
  // -------------------------------------------------------------------
  //  For af-generalized-alpha time-integration scheme:
  //  set "pseudo-theta", calculate initial accelerations according to
  //  prescribed Dirichlet values for generalized-alpha time
  //  integration and values at intermediate time steps
  // -------------------------------------------------------------------
  // starting algorithm
  if (startalgo_ || (step_ <= 2 && timealgoset_ == INPAR::FLUID::timeint_bdf2))
  {
    // use backward-Euler-type parameter combination
    if (step_<=numstasteps_ || (step_ <= 1 && timealgoset_ == INPAR::FLUID::timeint_bdf2))
    {
      if (myrank_==0)
      {
        std::cout<<"Starting algorithm for Af_GenAlpha active. "
            <<"Performing step "<<step_ <<" of "<<numstasteps_
            <<" Backward Euler starting steps"<<std::endl;
      }
      alphaM_ = 1.0;
      alphaF_ = 1.0;
      gamma_  = 1.0;
    }
    else
    {
      // recall original user wish
      if (timealgoset_ == INPAR::FLUID::timeint_one_step_theta)
      {
        alphaM_ = alphaF_ = 1.0;
        gamma_  = params_->get<double>("theta");
      }
      else if (timealgoset_ == INPAR::FLUID::timeint_bdf2)
      {
        alphaF_ = gamma_ = 1.0;
        alphaM_ = 3./2.;
      }
      else
      {
        alphaM_ = params_->get<double>("alpha_M");
        alphaF_ = params_->get<double>("alpha_F");
        gamma_  = params_->get<double>("gamma");
      }

      // do not enter starting algorithm section in the future
      startalgo_ = false;
    }
  }

  // compute "pseudo-theta" for af-generalized-alpha scheme
  theta_ = alphaF_*gamma_/alphaM_;
}


/*----------------------------------------------------------------------*
 * Explicit predictor                                 kronbichler 05/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::ExplicitPredictor()
{
  if (predictor_=="steady_state")
  {
    // this has already been done in TimeUpdate()
  }
  else if(predictor_=="zero_acceleration")
  {
    velnp_->Update(1.0, *veln_, (1.0-theta_)*dta_, *accn_, 0.0);
    intvelnp_->Update(1.0, *intveln_, (1.0-theta_)*dta_, *intaccn_, 0.0);
  }
  else if(predictor_=="constant_acceleration")
  {
    velnp_->Update(1.0, *veln_, dta_, *accn_, 0.0);
    intvelnp_->Update(1.0, *intveln_, dta_, *intaccn_, 0.0);
  }
  else if(predictor_=="constant_increment")
  {
    velnp_->Update(2.0, *veln_, -1.0, *velnm_, 0.0);
    intvelnp_->Update(2.0, *intveln_, -1.0, *intvelnm_, 0.0);
  }
  else if(predictor_=="explicit_second_order_midpoint")
  {
    velnp_->Update(1.0, *velnm_, 2.0*dta_, *accn_, 0.0);
    intvelnp_->Update(1.0, *intvelnm_, 2.0*dta_, *intaccn_, 0.0);
  }
  else
    dserror("Unknown fluid predictor %s", predictor_.c_str());
}

/*----------------------------------------------------------------------*
| set old part of right hand side                     kronbichler 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::SetOldPartOfRighthandside()
{
  hist_->PutScalar(0.0);

  // This code is entered at the beginning of the nonlinear iteration, so
  // store that the assembly to be done next is going to be the first one
  // (without combined vector update) for HDG.
  firstAssembly_ = true;
}



/*----------------------------------------------------------------------*
 | update acceleration for generalized-alpha time integration kro 05/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::GenAlphaUpdateAcceleration()
{

  //                                  n+1     n
  //                               vel   - vel
  //       n+1      n  gamma-1.0      (i)
  //    acc    = acc * --------- + ------------
  //       (i)           gamma      gamma * dt
  //

  // as opposed to standard fluid, update all variables including pressure

  // compute factors
  const double fact1 = 1.0/(gamma_*dta_);
  const double fact2 = 1.0 - (1.0/gamma_);

  accnp_->Update(fact2,*accn_,0.0);
  accnp_->Update(fact1,*velnp_,-fact1,*veln_,1.0);

  intaccnp_->Update(fact2,*intaccn_,0.0);
  intaccnp_->Update(fact1,*intvelnp_,-fact1,*intveln_,1.0);
}



/*----------------------------------------------------------------------*
 | compute values at intermediate time steps for gen.-alpha  kron 05/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::GenAlphaIntermediateValues()
{
  // set intermediate values for accelerations
  //
  //       n+alphaM                n+1                      n
  //    acc         = alpha_M * acc     + (1-alpha_M) *  acc
  //       (i)                     (i)
  accam_->Update((alphaM_),*accnp_,(1.0-alphaM_),*accn_,0.0);
  intaccam_->Update((alphaM_),*intaccnp_,(1.0-alphaM_),*intaccn_,0.0);

  // set intermediate values for velocity, pressure, velocity gradient
  //
  //       n+alphaF              n+1                   n
  //      u         = alpha_F * u     + (1-alpha_F) * u
  //       (i)                   (i)
  //
  // note that its af-genalpha with mid-point treatment of the pressure,
  // not implicit treatment as for the genalpha according to Whiting
  velaf_->Update((alphaF_),*velnp_,(1.0-alphaF_),*veln_,0.0);
  intvelaf_->Update((alphaF_),*intvelnp_,(1.0-alphaF_),*intveln_,0.0);
}


/*----------------------------------------------------------------------*
| set HDG state vectors                               kronbichler 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::SetStateTimInt()
{
  discret_->SetState(0, "velaf",velaf_);
  discret_->SetState(1, "intvelaf",intvelaf_);
  discret_->SetState(1, "intaccam",intaccam_);
  discret_->SetState(1, "intvelnp",intvelnp_);
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state               kronbichler 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::SetCustomEleParamsAssembleMatAndRHS(Teuchos::ParameterList& eleparams)
{
  eleparams.set<bool>("needslocalupdate",!firstAssembly_);
}

/*----------------------------------------------------------------------*
| set integration-scheme-specific state               kronbichler 05/14 |
*-----------------------------------------------------------------------*/
void FLD::TimIntHDG::ClearStateAssembleMatAndRHS()
{
  if (!firstAssembly_) {
    // Wrote into the state vector during element calls, need to transfer the
    // data back before it disappears when clearing the state (at least for nproc>1)
    const Epetra_Vector& intvelnpGhosted = *discret_->GetState(1,"intvelnp");
    for (int i=0; i<intvelnp_->MyLength(); ++i)
      (*intvelnp_)[i] = intvelnpGhosted[intvelnpGhosted.Map().LID(intvelnp_->Map().GID(i))];
  }
  firstAssembly_ = false;
  FluidImplicitTimeInt::ClearStateAssembleMatAndRHS();
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::TimeUpdate()
{
  FluidImplicitTimeInt::TimeUpdate();

  // velocities/pressures of this step become most recent
  // velocities/pressures of the last step
  intvelnm_->Update(1.0,*intveln_ ,0.0);
  intveln_ ->Update(1.0,*intvelnp_,0.0);

  intaccnm_->Update(1.0,*intaccn_,0.0);
  intaccn_->Update(1.0,*intaccnp_,0.0);

}



/*----------------------------------------------------------------------*
 |  set initial flow field for test cases              kronbichler 05/14|
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::SetInitialFlowField(
  const INPAR::FLUID::InitialField initfield,
  const int startfuncno
  )
{
  const Epetra_Map* dofrowmap = discret_->DofRowMap();
  const Epetra_Map* intdofrowmap = discret_->DofRowMap(1);
  Epetra_SerialDenseVector elevec1, elevec2, elevec3;
  Epetra_SerialDenseMatrix elemat1, elemat2;
  Teuchos::ParameterList initParams;
  initParams.set<int>("action",FLD::project_fluid_field);
  initParams.set("startfuncno",startfuncno);
  initParams.set<int>("initfield",initfield);
  // loop over all elements on the processor
  DRT::Element::LocationArray la(2);
  double error = 0;
  for (int el=0; el<discret_->NumMyColElements();++el) {
    DRT::Element *ele = discret_->lColElement(el);

    ele->LocationVector(*discret_,la,false);
    if (static_cast<std::size_t>(elevec1.M()) != la[0].lm_.size())
      elevec1.Shape(la[0].lm_.size(), 1);
    if (elevec2.M() != discret_->NumDof(1,ele))
      elevec2.Shape(discret_->NumDof(1,ele), 1);

    ele->Evaluate(initParams,*discret_,la[0].lm_,elemat1,elemat2,elevec1,elevec2,elevec3);

    // now fill the ele vector into the discretization
    for (unsigned int i=0; i<la[0].lm_.size(); ++i) {
      const int lid = dofrowmap->LID(la[0].lm_[i]);
      if (lid >= 0) {
        if ((*velnp_)[lid] != 0)
          error += std::abs((*velnp_)[lid]-elevec1(i));
        (*velnp_)[lid] = elevec1(i);
        (*veln_)[lid] = elevec1(i);
        (*velnm_)[lid] = elevec1(i);
      }
    }

    if (ele->Owner() == discret_->Comm().MyPID()) {
      std::vector<int> localDofs = discret_->Dof(1, ele);
      dsassert(localDofs.size() == static_cast<std::size_t>(elevec2.M()), "Internal error");
      for (unsigned int i=0; i<localDofs.size(); ++i)
        localDofs[i] = intdofrowmap->LID(localDofs[i]);
      intvelnp_->ReplaceMyValues(localDofs.size(), elevec2.A(), &localDofs[0]);
      intveln_->ReplaceMyValues(localDofs.size(), elevec2.A(), &localDofs[0]);
      intvelnm_->ReplaceMyValues(localDofs.size(), elevec2.A(), &localDofs[0]);
    }
  }
}



/*----------------------------------------------------------------------*
 | evaluate error for test cases with analytical solutions  kronbi 05/14|
 *----------------------------------------------------------------------*/
Teuchos::RCP<std::vector<double> >
FLD::TimIntHDG::EvaluateErrorComparedToAnalyticalSol()
{
  // HDG needs one more state vector for the interior solution (i.e., the actual solution)
  INPAR::FLUID::CalcError calcerr = DRT::INPUT::get<INPAR::FLUID::CalcError>(*params_,"calculate error");

  switch(calcerr)
  {
  case INPAR::FLUID::beltrami_flow:
  case INPAR::FLUID::channel2D:
  case INPAR::FLUID::topoptchannel:
  case INPAR::FLUID::gravitation:
  case INPAR::FLUID::shear_flow:
  case INPAR::FLUID::fsi_fluid_pusher:
  case INPAR::FLUID::byfunct1:
    discret_->SetState(1, "intvelnp",intvelnp_);
    break;
  default:
    break;
  };

  return FluidImplicitTimeInt::EvaluateErrorComparedToAnalyticalSol();
}



/*------------------------------------------------------------------------------------------------*
 |
 *------------------------------------------------------------------------------------------------*/
void FLD::TimIntHDG::Reset(
    bool completeReset,
    bool newFiles,
    int iter)
{
  FluidImplicitTimeInt::Reset(completeReset, newFiles, iter);
  const Epetra_Map* intdofrowmap = discret_->DofRowMap(1);
  intvelnp_  = LINALG::CreateVector(*intdofrowmap,true);
  intvelaf_  = LINALG::CreateVector(*intdofrowmap,true);
  intvelnm_  = LINALG::CreateVector(*intdofrowmap,true);
  intveln_   = LINALG::CreateVector(*intdofrowmap,true);
  intaccnp_  = LINALG::CreateVector(*intdofrowmap,true);
  intaccam_  = LINALG::CreateVector(*intdofrowmap,true);
  intaccnm_  = LINALG::CreateVector(*intdofrowmap,true);
  intaccn_   = LINALG::CreateVector(*intdofrowmap,true);
  if (discret_->Comm().MyPID()==0)
    std::cout << "Number of degrees of freedom in HDG system: "
              << discret_->DofRowMap(0)->NumGlobalElements() << std::endl;
}



namespace
{
  // internal helper function for output
  void getNodeVectorsHDG (DRT::Discretization               &dis,
                          const Teuchos::RCP<Epetra_Vector> &interiorValues,
                          const Teuchos::RCP<Epetra_Vector> &traceValues,
                          const int                          ndim,
                          Teuchos::RCP<Epetra_Vector>       &velocity,
                          Teuchos::RCP<Epetra_Vector>       &pressure,
                          Teuchos::RCP<Epetra_Vector>       &tracevel,
                          Teuchos::RCP<Epetra_Vector>       &cellPres)
  {
    // create dofsets for velocity and pressure at nodes
    if (pressure.get() == NULL || pressure->MyLength() != dis.NumMyRowNodes()) {
      std::vector<int> conddofset;
      std::vector<int> otherdofset;
      int numrownodes = dis.NumMyRowNodes();
      conddofset.reserve(numrownodes);
      otherdofset.reserve(numrownodes*ndim);

      for (int i=0; i<numrownodes; ++i)
      {
        DRT::Node* node = dis.lRowNode(i);
        const int index = node->Id()*(ndim+1);
        for (int d=0; d<ndim; ++d)
          otherdofset.push_back(index+d);
        conddofset.push_back(index+ndim);
      }

      Teuchos::RCP<Epetra_Map> conddofmap =
          Teuchos::rcp(new Epetra_Map(-1, conddofset.size(), &conddofset[0], 0, dis.Comm()));
      Teuchos::RCP<Epetra_Map> otherdofmap =
          Teuchos::rcp(new Epetra_Map(-1, otherdofset.size(), &otherdofset[0], 0, dis.Comm()));

      velocity.reset(new Epetra_Vector(*otherdofmap));
      pressure.reset(new Epetra_Vector(*conddofmap));
    }
    tracevel.reset(new Epetra_Vector(velocity->Map()));
    cellPres.reset(new Epetra_Vector(*dis.ElementRowMap()));

    // call element routine for interpolate HDG to elements
    Teuchos::ParameterList params;
    params.set<int>("action",FLD::interpolate_hdg_to_node);
    dis.SetState(1,"intvelnp",interiorValues);
    dis.SetState(0,"velnp",traceValues);
    std::vector<int> dummy;
    Epetra_SerialDenseMatrix dummyMat;
    Epetra_SerialDenseVector dummyVec;
    Epetra_SerialDenseVector interpolVec;
    std::vector<unsigned char> touchCount(pressure->MyLength());
    velocity->PutScalar(0.);
    pressure->PutScalar(0.);
    for (int el=0; el<dis.NumMyColElements();++el) {
      DRT::Element *ele = dis.lColElement(el);
      if (interpolVec.M() == 0)
        interpolVec.Resize(ele->NumNode()*(2*ndim+1)+1);

      ele->Evaluate(params,dis,dummy,dummyMat,dummyMat,interpolVec,dummyVec,dummyVec);

      // sum values on nodes into vectors and record the touch count (build average of values)
      for (int i=0; i<ele->NumNode(); ++i)
      {
        DRT::Node* node = ele->Nodes()[i];
        const int localIndex = pressure->Map().LID(node->Id()*(ndim+1)+ndim);
        if (localIndex < 0)
          continue;
        touchCount[localIndex]++;
        for (int d=0; d<ndim; ++d)
          (*velocity)[ndim*localIndex+d] += interpolVec(i+d*ele->NumNode());
        (*pressure)[localIndex] += interpolVec(i+ndim*ele->NumNode());
        for (int d=0; d<ndim; ++d)
          (*tracevel)[ndim*localIndex+d] += interpolVec(i+(ndim+1+d)*ele->NumNode());

        const int eleIndex = dis.ElementRowMap()->LID(ele->Id());
        if (eleIndex >= 0)
          (*cellPres)[eleIndex] += interpolVec((2*ndim+1)*ele->NumNode());
      }
    }
    for (int i=0; i<pressure->MyLength(); ++i) {
      (*pressure)[i] /= touchCount[i];
      for (int d=0; d<ndim; ++d)
        (*velocity)[i*ndim+d] /= touchCount[i];
      for (int d=0; d<ndim; ++d)
        (*tracevel)[i*ndim+d] /= touchCount[i];
    }
    dis.ClearState();
  }
}



/*----------------------------------------------------------------------*
 | output of solution vector to binio                  kronbichler 05/14|
 *----------------------------------------------------------------------*/
void FLD::TimIntHDG::Output()
{
  // output of solution, currently only small subset of functionality
  if (step_%upres_ == 0)
  {
    // step number and time
    output_->NewStep(step_,time_);

    Teuchos::RCP<Epetra_Vector> traceVel, cellPres;
    getNodeVectorsHDG(*discret_, intvelnp_, velnp_,
                      params_->get<int>("number of velocity degrees of freedom"),
                      interpolatedVelocity_, interpolatedPressure_, traceVel, cellPres);
    output_->WriteVector("velnp",interpolatedVelocity_);
    output_->WriteVector("pressure",interpolatedPressure_);
    output_->WriteVector("par_vel",traceVel);
    output_->WriteVector("pressure_avg",cellPres);

    if (step_==upres_ or step_ == 0) output_->WriteElementData(true,false);

    if (uprestart_ != 0 && step_%uprestart_ == 0) //add restart data
    {
      // acceleration vector at time n+1 and n, velocity/pressure vector at time n and n-1
      //output_->WriteVector("accnp",intaccnp_);
      //output_->WriteVector("accn", intaccn_);
      //output_->WriteVector("veln", intveln_);
      //output_->WriteVector("velnm",intvelnm_);
    }
  }
}
