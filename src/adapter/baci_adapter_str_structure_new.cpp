/*-----------------------------------------------------------*/
/*! \file

\brief Adapter for the new structural time integration framework.


\level 3

*/
/*-----------------------------------------------------------*/

#include "baci_adapter_str_structure_new.H"
#include "baci_adapter_str_timint_adaptive.H"
#include "baci_adapter_str_fsi_timint_adaptive.H"
#include "baci_adapter_str_constr_merged.H"
#include "baci_adapter_str_wrapper.H"
#include "baci_adapter_str_lung.H"
#include "baci_adapter_str_redairway.H"
#include "baci_adapter_str_fpsiwrapper.H"
#include "baci_adapter_str_fbiwrapper.H"
#include "baci_adapter_str_fsiwrapper_immersed.H"
#include "baci_adapter_str_structalewrapper.H"
#include "baci_adapter_str_timeloop.H"
#include "baci_adapter_str_ssiwrapper.H"
#include "baci_adapter_str_pasiwrapper.H"

#include "baci_structure_timada_create.H"

#include "baci_utils_exceptions.H"

#include "baci_structure_new_timint_factory.H"
#include "baci_structure_new_solver_factory.H"
#include "baci_structure_new_timint_base.H"
#include "baci_structure_new_model_evaluator.H"

#include "baci_lib_globalproblem.H"
#include "baci_lib_discret.H"
#include "baci_lib_condition.H"
#include "baci_lib_utils_parallel.H"

#include "baci_io.H"
#include "baci_io_control.H"
#include "baci_io_pstream.H"

#include "baci_comm_utils.H"

#include "baci_mat_par_bundle.H"

#include "baci_inpar_fsi.H"
#include "baci_inpar_poroelast.H"
#include "baci_inpar_beamcontact.H"
#include "baci_inpar_beaminteraction.H"
#include "baci_inpar_beam_to_solid.H"
#include "baci_inpar_validparameters.H"
#include "baci_inpar_contact.H"

#include "baci_so3_sh8p8.H"
#include "baci_so3_plast_ssn_eletypes.H"
#include "baci_so3_plast_ssn_sosh8.H"
#include "baci_so3_plast_ssn_sosh18.H"
#include "baci_so3_hex8fbar.H"
#include "baci_s8.H"

#include "baci_beam3_reissner.H"
#include "baci_beam3_kirchhoff.H"

#include "baci_solver_nonlin_nox_group.H"
#include "baci_solver_nonlin_nox_group_prepostoperator.H"

#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
ADAPTER::StructureBaseAlgorithmNew::StructureBaseAlgorithmNew()
    : str_wrapper_(Teuchos::null),
      prbdyn_(Teuchos::null),
      sdyn_(Teuchos::null),
      isinit_(false),
      issetup_(false)
{
  // empty
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::Init(const Teuchos::ParameterList& prbdyn,
    Teuchos::ParameterList& sdyn, Teuchos::RCP<DRT::DiscretizationInterface> actdis)
{
  issetup_ = false;

  // save a pointer of the input variables as class variables
  prbdyn_ = Teuchos::rcpFromRef<const Teuchos::ParameterList>(prbdyn);
  sdyn_ = Teuchos::rcpFromRef<Teuchos::ParameterList>(sdyn);
  actdis_ = actdis;

  isinit_ = true;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::Setup()
{
  if (not IsInit()) dserror("You have to call Init() first!");

  // major switch to different time integrators
  switch (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(*sdyn_, "DYNAMICTYP"))
  {
    case INPAR::STR::dyna_statics:
    case INPAR::STR::dyna_genalpha:
    case INPAR::STR::dyna_genalpha_liegroup:
    case INPAR::STR::dyna_onesteptheta:
    case INPAR::STR::dyna_gemm:
    case INPAR::STR::dyna_expleuler:
    case INPAR::STR::dyna_centrdiff:
    case INPAR::STR::dyna_ab2:
    case INPAR::STR::dyna_euma:
    case INPAR::STR::dyna_euimsto:
      SetupTimInt();  // <-- here is the show
      break;
    default:
      dserror(
          "Unknown time integration scheme '%s'", sdyn_->get<std::string>("DYNAMICTYP").c_str());
      break;
  }

  issetup_ = true;

  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::RegisterModelEvaluator(
    const std::string name, Teuchos::RCP<::STR::MODELEVALUATOR::Generic> me)
{
  // safety checks
  if (not IsInit()) dserror("Init(...) must be called before RegisterModelEvaluator(...) !");
  if (IsSetup()) dserror("RegisterModelEvaluator(...) must be called before Setup() !");

  // set RCP ptr to model evaluator in problem dynamic parameter list
  const_cast<Teuchos::ParameterList&>(*prbdyn_).set<Teuchos::RCP<::STR::MODELEVALUATOR::Generic>>(
      name, me);

  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::SetupTimInt()
{
  if (not IsInit()) dserror("You have to call Init() first!");

  // get the problem instance
  DRT::Problem* problem = DRT::Problem::Instance();
  // get the restart step
  const int restart = problem->Restart();

  // ---------------------------------------------------------------------------
  // Define, initialize and start the timer
  // ---------------------------------------------------------------------------
  Teuchos::RCP<Teuchos::Time> t =
      Teuchos::TimeMonitor::getNewTimer("ADAPTER::StructureTimIntBaseAlgorithm::SetupStructure");
  Teuchos::TimeMonitor monitor(*t);

  // ---------------------------------------------------------------------------
  // Here we read the discretization at the current
  // time step from restart files
  // ---------------------------------------------------------------------------
  if (actdis_->GetCondition("PointCoupling") != NULL)
  {
    std::vector<Teuchos::RCP<DRT::Discretization>> actdis_vec(
        1, Teuchos::rcp_dynamic_cast<DRT::Discretization>(actdis_, true));
    actdis_vec[0]->FillComplete(false, false, false);
    DRT::UTILS::RedistributeDiscretizationsByBinning(actdis_vec, true);
  }
  else if (not actdis_->Filled() || not actdis_->HaveDofs())
  {
    actdis_->FillComplete();
  }

  // ---------------------------------------------------------------------------
  // Setup a model type set by checking
  // the different conditions
  // ---------------------------------------------------------------------------
  // define and initial with default value
  Teuchos::RCP<std::set<enum INPAR::STR::ModelType>> modeltypes =
      Teuchos::rcp(new std::set<enum INPAR::STR::ModelType>());
  modeltypes->insert(INPAR::STR::model_structure);
  SetModelTypes(*modeltypes);

  // ---------------------------------------------------------------------------
  // Setup a element technology set by checking
  // the elements of the discretization
  // ---------------------------------------------------------------------------
  Teuchos::RCP<std::set<enum INPAR::STR::EleTech>> eletechs =
      Teuchos::rcp(new std::set<enum INPAR::STR::EleTech>());
  DetectElementTechnologies(*eletechs);

  // ---------------------------------------------------------------------------
  // Setup the parameter lists for structural
  // time integration
  // ---------------------------------------------------------------------------
  Teuchos::RCP<Teuchos::ParameterList> ioflags =
      Teuchos::rcp(new Teuchos::ParameterList(problem->IOParams()));
  Teuchos::RCP<Teuchos::ParameterList> taflags =
      Teuchos::rcp(new Teuchos::ParameterList(sdyn_->sublist("TIMEADAPTIVITY")));
  Teuchos::RCP<Teuchos::ParameterList> xparams = Teuchos::rcp(new Teuchos::ParameterList());
  SetParams(*ioflags, *xparams, *taflags);

  // ---------------------------------------------------------------------------
  // Setup and create model specific linear solvers
  // ---------------------------------------------------------------------------
  Teuchos::RCP<std::map<enum INPAR::STR::ModelType, Teuchos::RCP<CORE::LINALG::Solver>>>
      linsolvers = STR::SOLVER::BuildLinSolvers(*modeltypes, *sdyn_, *actdis_);

  // ---------------------------------------------------------------------------
  // Checks in case of multi-scale simulations
  // ---------------------------------------------------------------------------
  {
    // make sure we IMR-like generalised-alpha requested for multi-scale
    // simulations
    Teuchos::RCP<MAT::PAR::Bundle> materials = problem->Materials();
    for (std::map<int, Teuchos::RCP<MAT::PAR::Material>>::const_iterator i =
             materials->Map()->begin();
         i != materials->Map()->end(); ++i)
    {
      Teuchos::RCP<MAT::PAR::Material> mat = i->second;
      if (mat->Type() == INPAR::MAT::m_struct_multiscale)
      {
        if (DRT::INPUT::IntegralValue<INPAR::STR::DynamicType>(*sdyn_, "DYNAMICTYP") !=
            INPAR::STR::dyna_genalpha)
          dserror("In multi-scale simulations, you have to use DYNAMICTYP=GenAlpha");
        else if (DRT::INPUT::IntegralValue<INPAR::STR::MidAverageEnum>(
                     sdyn_->sublist("GENALPHA"), "GENAVG") != INPAR::STR::midavg_trlike)
          dserror(
              "In multi-scale simulations, you have to use DYNAMICTYP=GenAlpha with GENAVG=TrLike");
        break;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Create context for output and restart
  // ---------------------------------------------------------------------------
  Teuchos::RCP<IO::DiscretizationWriter> output = actdis_->Writer();
  if (DRT::INPUT::IntegralValue<int>(*ioflags, "OUTPUT_BIN"))
  {
    output->WriteMesh(0, 0.0);
  }

  // ---------------------------------------------------------------------------
  // initialize/setup the input/output data container
  // ---------------------------------------------------------------------------
  Teuchos::RCP<STR::TIMINT::BaseDataIO> dataio = Teuchos::rcp(new STR::TIMINT::BaseDataIO());
  dataio->Init(*ioflags, *sdyn_, *xparams, output);
  dataio->Setup();

  // ---------------------------------------------------------------------------
  // initialize/setup the structural dynamics data
  // container
  // ---------------------------------------------------------------------------
  Teuchos::RCP<STR::TIMINT::BaseDataSDyn> datasdyn = STR::TIMINT::BuildDataSDyn(*sdyn_);
  datasdyn->Init(actdis_, *sdyn_, *xparams, modeltypes, eletechs, linsolvers);
  datasdyn->Setup();

  // ---------------------------------------------------------------------------
  // initialize/setup the global state data container
  // ---------------------------------------------------------------------------
  Teuchos::RCP<STR::TIMINT::BaseDataGlobalState> dataglobalstate = Teuchos::null;
  SetGlobalState(dataglobalstate, datasdyn);

  // ---------------------------------------------------------------------------
  // in case of non-additive rotation (pseudo-)vector DOFs:
  // ---------------------------------------------------------------------------
  if (eletechs->find(INPAR::STR::eletech_rotvec) != eletechs->end())
  {
    // -------------------------------------------------------------------------
    // set the RotVecUpdater as new precomputeX operator for the nox nln group
    // -------------------------------------------------------------------------
    Teuchos::ParameterList& p_grp_opt = datasdyn->GetMutableNoxParams().sublist("Group Options");
    // Get the current map. If there is no map, return a new empty one. (reference)
    NOX::NLN::GROUP::PrePostOperator::Map& prepostgroup_map =
        NOX::NLN::GROUP::PrePostOp::GetMutableMap(p_grp_opt);
    // create the new rotation vector update pre/post operator
    Teuchos::RCP<NOX::NLN::Abstract::PrePostOperator> prepostrotvec_ptr =
        Teuchos::rcp(new NOX::NLN::GROUP::PrePostOp::TIMINT::RotVecUpdater(dataglobalstate));
    // insert/replace the old pointer in the map
    prepostgroup_map[NOX::NLN::GROUP::prepost_rotvecupdate] = prepostrotvec_ptr;
  }

  // ---------------------------------------------------------------------------
  // Build time integrator
  // ---------------------------------------------------------------------------
  Teuchos::RCP<STR::TIMINT::Base> ti_strategy = Teuchos::null;
  SetTimeIntegrationStrategy(ti_strategy, dataio, datasdyn, dataglobalstate, restart);


  // ---------------------------------------------------------------------------
  // Create wrapper for the time integration strategy
  // ---------------------------------------------------------------------------
  SetStructureWrapper(*ioflags, *sdyn_, *xparams, *taflags, ti_strategy);

  // see you
  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::SetModelTypes(
    std::set<enum INPAR::STR::ModelType>& modeltypes) const
{
  if (not IsInit()) dserror("You have to call Init() first!");
  // ---------------------------------------------------------------------------
  // check for meshtying and contact conditions
  // ---------------------------------------------------------------------------
  // --- contact conditions
  std::vector<DRT::Condition*> ccond(0);
  actdis_->GetCondition("Contact", ccond);
  if (ccond.size())
  {
    // what's the current problem type?
    ProblemType probtype = DRT::Problem::Instance()->GetProblemType();
    // ToDo: once the new structural time integration can handle
    //       condensed contact formulations, the model_evaluator
    //       can have its contact model. For now, the TSI Lagrange
    //       strategy resides in the TSI algorithm.
    if (probtype == ProblemType::tsi)
    {
      const Teuchos::ParameterList& contact = DRT::Problem::Instance()->ContactDynamicParams();
      if (DRT::INPUT::IntegralValue<INPAR::CONTACT::SolvingStrategy>(contact, "STRATEGY") ==
          INPAR::CONTACT::solution_nitsche)
        modeltypes.insert(INPAR::STR::model_contact);
    }
    else
      modeltypes.insert(INPAR::STR::model_contact);
  }
  // --- meshtying conditions
  std::vector<DRT::Condition*> mtcond(0);
  actdis_->GetCondition("Mortar", mtcond);
  if (mtcond.size()) modeltypes.insert(INPAR::STR::model_meshtying);

  // check for 0D cardiovascular conditions
  // ---------------------------------------------------------------------------
  std::vector<DRT::Condition*> cardiovasc0dcond_4elementwindkessel(0);
  std::vector<DRT::Condition*> cardiovasc0dcond_arterialproxdist(0);
  std::vector<DRT::Condition*> cardiovasc0dcond_syspulcirculation(0);
  std::vector<DRT::Condition*> cardiovascrespir0dcond_syspulperiphcirculation(0);
  actdis_->GetCondition(
      "Cardiovascular0D4ElementWindkesselStructureCond", cardiovasc0dcond_4elementwindkessel);
  actdis_->GetCondition(
      "Cardiovascular0DArterialProxDistStructureCond", cardiovasc0dcond_arterialproxdist);
  actdis_->GetCondition("Cardiovascular0DSysPulCirculationStructureCond",
      cardiovascrespir0dcond_syspulperiphcirculation);
  actdis_->GetCondition("CardiovascularRespiratory0DSysPulPeriphCirculationStructureCond",
      cardiovasc0dcond_syspulcirculation);
  if (cardiovasc0dcond_4elementwindkessel.size() or cardiovasc0dcond_arterialproxdist.size() or
      cardiovasc0dcond_syspulcirculation.size() or
      cardiovascrespir0dcond_syspulperiphcirculation.size())
    modeltypes.insert(INPAR::STR::model_cardiovascular0d);

  // ---------------------------------------------------------------------------
  // check for constraint conditions
  // ---------------------------------------------------------------------------
  bool have_lag_constraint = false;
  bool have_pen_constraint = false;
  // --- enforcement by Lagrange multiplier
  std::vector<DRT::Condition*> lagcond_volconstr3d(0);
  std::vector<DRT::Condition*> lagcond_areaconstr3d(0);
  std::vector<DRT::Condition*> lagcond_areaconstr2d(0);
  std::vector<DRT::Condition*> lagcond_mpconline2d(0);
  std::vector<DRT::Condition*> lagcond_mpconplane3d(0);
  std::vector<DRT::Condition*> lagcond_mpcnormcomp3d(0);
  actdis_->GetCondition("VolumeConstraint_3D", lagcond_volconstr3d);
  actdis_->GetCondition("AreaConstraint_3D", lagcond_areaconstr3d);
  actdis_->GetCondition("AreaConstraint_2D", lagcond_areaconstr2d);
  actdis_->GetCondition("MPC_NodeOnLine_2D", lagcond_mpconline2d);
  actdis_->GetCondition("MPC_NodeOnPlane_3D", lagcond_mpconplane3d);
  actdis_->GetCondition("MPC_NormalComponent_3D", lagcond_mpcnormcomp3d);
  if (lagcond_volconstr3d.size() or lagcond_areaconstr3d.size() or lagcond_areaconstr2d.size() or
      lagcond_mpconline2d.size() or lagcond_mpconplane3d.size() or lagcond_mpcnormcomp3d.size())
    have_lag_constraint = true;
  // --- enforcement by penalty law
  std::vector<DRT::Condition*> pencond_volconstr3d(0);
  std::vector<DRT::Condition*> pencond_areaconstr3d(0);
  std::vector<DRT::Condition*> pencond_mpcnormcomp3d(0);
  actdis_->GetCondition("VolumeConstraint_3D_Pen", pencond_volconstr3d);
  actdis_->GetCondition("AreaConstraint_3D_Pen", pencond_areaconstr3d);
  actdis_->GetCondition("MPC_NormalComponent_3D_Pen", pencond_mpcnormcomp3d);
  if (pencond_volconstr3d.size() or pencond_areaconstr3d.size() or pencond_mpcnormcomp3d.size())
    have_pen_constraint = true;
  if (have_lag_constraint or have_pen_constraint)
    modeltypes.insert(INPAR::STR::model_lag_pen_constraint);

  // ---------------------------------------------------------------------------
  // check for spring dashpot conditions
  // ---------------------------------------------------------------------------
  std::vector<DRT::Condition*> sdp_cond(0);
  actdis_->GetCondition("RobinSpringDashpot", sdp_cond);
  if (sdp_cond.size()) modeltypes.insert(INPAR::STR::model_springdashpot);
  // ---------------------------------------------------------------------------
  // check for coupled problems
  // ---------------------------------------------------------------------------
  // get the problem instance
  DRT::Problem* problem = DRT::Problem::Instance();
  // what's the current problem type?
  ProblemType probtype = problem->GetProblemType();
  switch (probtype)
  {
    case ProblemType::fsi:
    case ProblemType::immersed_fsi:
    case ProblemType::fbi:
    case ProblemType::fsi_redmodels:
    case ProblemType::fsi_lung:
    case ProblemType::gas_fsi:
    case ProblemType::ac_fsi:
    case ProblemType::biofilm_fsi:
    case ProblemType::thermo_fsi:
    case ProblemType::fsi_xfem:
    case ProblemType::pasi:
    case ProblemType::ssi:
    case ProblemType::ssti:
    {
      if (prbdyn_->INVALID_TEMPLATE_QUALIFIER isType<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>(
              "Partitioned Coupling Model"))
      {
        if (prbdyn_->INVALID_TEMPLATE_QUALIFIER isType<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>(
                "Monolithic Coupling Model"))
          dserror("Cannot have both partitioned and monolithic coupling at the same time!");
        const auto coupling_model_ptr =
            prbdyn_->INVALID_TEMPLATE_QUALIFIER get<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>(
                "Partitioned Coupling Model");
        if (coupling_model_ptr.is_null())
          dserror("The partitioned coupling model pointer is not allowed to be Teuchos::null!");
        // set the model type
        modeltypes.insert(INPAR::STR::model_partitioned_coupling);
        // copy the coupling model object pointer into the (temporal) sdyn parameter list
        sdyn_->set<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>(
            "Partitioned Coupling Model", coupling_model_ptr);
      }

      else if (prbdyn_->INVALID_TEMPLATE_QUALIFIER
                   isType<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>("Monolithic Coupling Model"))
      {
        const auto coupling_model_ptr =
            prbdyn_->INVALID_TEMPLATE_QUALIFIER get<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>(
                "Monolithic Coupling Model");
        if (coupling_model_ptr.is_null())
          dserror("The monolithic coupling model pointer is not allowed to be Teuchos::null!");
        // set the model type
        modeltypes.insert(INPAR::STR::model_monolithic_coupling);
        // copy the coupling model object pointer into the (temporal) sdyn parameter list
        sdyn_->set<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>(
            "Monolithic Coupling Model", coupling_model_ptr);
      }

      else if (prbdyn_->INVALID_TEMPLATE_QUALIFIER
                   isType<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>("Basic Coupling Model"))
      {
        const auto coupling_model_ptr =
            prbdyn_->INVALID_TEMPLATE_QUALIFIER get<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>(
                "Basic Coupling Model");
        if (coupling_model_ptr.is_null())
          dserror("The basic coupling model pointer is not allowed to be Teuchos::null!");
        // set the model type
        modeltypes.insert(INPAR::STR::model_basic_couping);
        // copy the coupling model object pointer into the (temporal) sdyn parameter list
        sdyn_->set<Teuchos::RCP<STR::MODELEVALUATOR::Generic>>(
            "Basic Coupling Model", coupling_model_ptr);
      }
      break;
    }
    default:
      // do nothing
      break;
  }  // switch (probtype)

  // ---------------------------------------------------------------------------
  // check for beam interactions (either contact or potential-based)
  // ---------------------------------------------------------------------------
  // get beam contact strategy
  const Teuchos::ParameterList& beamcontact = DRT::Problem::Instance()->BeamContactParams();
  INPAR::BEAMCONTACT::Strategy strategy =
      DRT::INPUT::IntegralValue<INPAR::BEAMCONTACT::Strategy>(beamcontact, "BEAMS_STRATEGY");

  INPAR::BEAMCONTACT::Modelevaluator modelevaluator =
      DRT::INPUT::IntegralValue<INPAR::BEAMCONTACT::Modelevaluator>(beamcontact, "MODELEVALUATOR");

  // conditions for potential-based beam interaction
  std::vector<DRT::Condition*> beampotconditions(0);
  actdis_->GetCondition("BeamPotentialLineCharge", beampotconditions);

  // conditions for beam penalty point coupling
  std::vector<DRT::Condition*> beampenaltycouplingconditions(0);
  actdis_->GetCondition("PenaltyPointCouplingCondition", beampenaltycouplingconditions);


  if (strategy != INPAR::BEAMCONTACT::bstr_none and modelevaluator == INPAR::BEAMCONTACT::bstr_old)
    modeltypes.insert(INPAR::STR::model_beam_interaction_old);

  // ---------------------------------------------------------------------------
  // check for brownian dynamics
  // ---------------------------------------------------------------------------
  if (DRT::INPUT::IntegralValue<int>(
          DRT::Problem::Instance()->BrownianDynamicsParams(), "BROWNDYNPROB"))
    modeltypes.insert(INPAR::STR::model_browniandyn);

  // ---------------------------------------------------------------------------
  // check for beam interaction
  // ---------------------------------------------------------------------------
  if (DRT::INPUT::IntegralValue<int>(
          DRT::Problem::Instance()->BeamInteractionParams().sublist("CROSSLINKING"),
          "CROSSLINKER") or
      DRT::INPUT::IntegralValue<int>(
          DRT::Problem::Instance()->BeamInteractionParams().sublist("SPHERE BEAM LINK"),
          "SPHEREBEAMLINKING") or
      DRT::INPUT::IntegralValue<INPAR::BEAMINTERACTION::Strategy>(
          DRT::Problem::Instance()->BeamInteractionParams().sublist("BEAM TO BEAM CONTACT"),
          "STRATEGY") != INPAR::BEAMINTERACTION::bstr_none or
      DRT::INPUT::IntegralValue<INPAR::BEAMINTERACTION::Strategy>(
          DRT::Problem::Instance()->BeamInteractionParams().sublist("BEAM TO SPHERE CONTACT"),
          "STRATEGY") != INPAR::BEAMINTERACTION::bstr_none or
      Teuchos::getIntegralValue<INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization>(
          DRT::Problem::Instance()->BeamInteractionParams().sublist(
              "BEAM TO SOLID VOLUME MESHTYING"),
          "CONTACT_DISCRETIZATION") != INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization::none or
      Teuchos::getIntegralValue<INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization>(
          DRT::Problem::Instance()->BeamInteractionParams().sublist(
              "BEAM TO SOLID SURFACE MESHTYING"),
          "CONTACT_DISCRETIZATION") != INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization::none or
      Teuchos::getIntegralValue<INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization>(
          DRT::Problem::Instance()->BeamInteractionParams().sublist(
              "BEAM TO SOLID SURFACE CONTACT"),
          "CONTACT_DISCRETIZATION") != INPAR::BEAMTOSOLID::BeamToSolidContactDiscretization::none or
      beampotconditions.size() > 0 or beampenaltycouplingconditions.size() > 0)
    modeltypes.insert(INPAR::STR::model_beaminteraction);

  // hopefully we haven't forgotten anything
  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::DetectElementTechnologies(
    std::set<enum INPAR::STR::EleTech>& eletechs) const
{
  int isplasticity_local = 0;
  int isplasticity_global = 0;

  int iseas_local = 0;
  int iseas_global = 0;

  int isfbar_local = 0;
  int isfbar_global = 0;

  int ispressure_local = 0;
  int ispressure_global = 0;

  int isrotvec_local = 0;
  int isrotvec_global = 0;

  for (int i = 0; i < actdis_->NumMyRowElements(); ++i)
  {
    DRT::Element* actele = actdis_->lRowElement(i);
    // Detect plasticity -------------------------------------------------------
    if (actele->ElementType() == DRT::ELEMENTS::So_hex8PlastType::Instance() or
        actele->ElementType() == DRT::ELEMENTS::So_hex27PlastType::Instance() or
        actele->ElementType() == DRT::ELEMENTS::So_sh8PlastType::Instance() or
        actele->ElementType() == DRT::ELEMENTS::So_hex18PlastType::Instance() or
        actele->ElementType() == DRT::ELEMENTS::So_sh18PlastType::Instance())
    {
      if (actele->Material()->MaterialType() == INPAR::MAT::m_plelasthyper)
        isplasticity_local = true;
    }

    // Detect EAS --------------------------------------------------------------
    DRT::ELEMENTS::So_base* so_base_ele = dynamic_cast<DRT::ELEMENTS::So_base*>(actele);
    if (so_base_ele != NULL)
    {
      if (so_base_ele->HaveEAS()) iseas_local = 1;
    }
    /* additional check for shell8 elements, since these elements are not derived
     * from the So_base class */
    else
    {
      DRT::ELEMENTS::Shell8* shell8_ele = dynamic_cast<DRT::ELEMENTS::Shell8*>(actele);
      if (shell8_ele != NULL)
        if (shell8_ele->HaveEAS()) iseas_local = 1;
    }

    // Detect additional pressure dofs -----------------------------------------
    if (actele->ElementType() == DRT::ELEMENTS::So_sh8p8Type::Instance()) ispressure_local = 1;

    // Detect fbar
    DRT::ELEMENTS::So_hex8fbar* so_hex8fbar_ele = dynamic_cast<DRT::ELEMENTS::So_hex8fbar*>(actele);
    if (so_hex8fbar_ele != NULL) isfbar_local = 1;

    // Detect non-additive rotation-vector DOFs --------------------------------
    if (actele->ElementType() == DRT::ELEMENTS::Beam3rType::Instance() or
        actele->ElementType() == DRT::ELEMENTS::Beam3kType::Instance())
    {
      isrotvec_local = true;
      break;
    }
  }

  // plasticity - sum over all processors
  actdis_->Comm().SumAll(&isplasticity_local, &isplasticity_global, 1);
  if (isplasticity_global > 0) eletechs.insert(INPAR::STR::eletech_plasticity);

  // eas - sum over all processors
  actdis_->Comm().SumAll(&iseas_local, &iseas_global, 1);
  if (iseas_global > 0) eletechs.insert(INPAR::STR::eletech_eas);

  // pressure - sum over all processors
  actdis_->Comm().SumAll(&ispressure_local, &ispressure_global, 1);
  if (ispressure_global > 0) eletechs.insert(INPAR::STR::eletech_pressure);

  // fbar - sum over all processors
  actdis_->Comm().SumAll(&isfbar_local, &isfbar_global, 1);
  if (isfbar_global > 0) eletechs.insert(INPAR::STR::eletech_fbar);

  // rotation vector DOFs - sum over all processors
  actdis_->Comm().SumAll(&isrotvec_local, &isrotvec_global, 1);
  if (isrotvec_global > 0) eletechs.insert(INPAR::STR::eletech_rotvec);

  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::SetParams(Teuchos::ParameterList& ioflags,
    Teuchos::ParameterList& xparams, Teuchos::ParameterList& taflags)
{
  // get the problem instance and the problem type
  DRT::Problem* problem = DRT::Problem::Instance();
  ProblemType probtype = problem->GetProblemType();

  // ---------------------------------------------------------------------------
  // show default parameters
  // ---------------------------------------------------------------------------
  if ((actdis_->Comm()).MyPID() == 0) DRT::INPUT::PrintDefaultParameters(IO::cout, *sdyn_);

  // ---------------------------------------------------------------------------
  // get input parameter lists and copy them,
  // because a few parameters are overwritten
  // ---------------------------------------------------------------------------
  // nox parameter list
  Teuchos::RCP<Teuchos::ParameterList> snox =
      Teuchos::rcp(new Teuchos::ParameterList(problem->StructuralNoxParams()));
  Teuchos::ParameterList& nox = xparams.sublist("NOX");
  nox = *snox;
  // add extra parameters (a kind of work-around)
  xparams.set<FILE*>("err file", problem->ErrorFile()->Handle());

  /* overrule certain parameters
   *
   * These parameters are overwritten by the parameters of the current
   * problem type. This is done only once temporally, since we need the
   * structural dynamics parameter-list only for the setup routines of the
   * different data containers.                                         */
  sdyn_->set<double>("TIMESTEP", prbdyn_->get<double>("TIMESTEP"));
  sdyn_->set<double>("MAXTIME", prbdyn_->get<double>("MAXTIME"));

  // overrule certain parameters
  sdyn_->set<int>("NUMSTEP", prbdyn_->get<int>("NUMSTEP"));
  sdyn_->set<int>("RESTARTEVRY", prbdyn_->get<int>("RESTARTEVRY"));
  sdyn_->set<int>("RESULTSEVRY", prbdyn_->get<int>("RESULTSEVRY"));

  // Check if for chosen Rayleigh damping the regarding parameters are given explicitly in the .dat
  // file
  if (DRT::INPUT::IntegralValue<INPAR::STR::DampKind>(*sdyn_, "DAMPING") ==
      INPAR::STR::damp_rayleigh)
  {
    if (sdyn_->get<double>("K_DAMP") < 0.0)
    {
      dserror("Rayleigh damping parameter K_DAMP not explicitly given.");
    }
    if (sdyn_->get<double>("M_DAMP") < 0.0)
    {
      dserror("Rayleigh damping parameter M_DAMP not explicitly given.");
    }
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  /* Overwrite certain parameters in STRUCTURAL DYNAMIC/TIMADAPTIVITY by those from
   * FSI DYNAMIC/TIMEADAPTIVITY
   *
   * In case, that the structure field is part of an FSI simulation with time step
   * size adaptivity based on structure field error estimation, we have to provide
   * the following algorithmic control parameters:
   *
   * - ADAPTSTEPMAX
   * - STEPSIZEMAX
   * - STEPSIZEMIN
   * - SIZERATIOMAX
   * - SIZERATIOMIN
   * - SIZERATIOSCALE
   *
   * They are specified by the FSI algorithm since they have to be the same for
   * the structure and fluid field. Hence, we overwrite the corresponding
   * parameters in the structural parameter list in order to avoid redundant
   * parameter specification in the input file.
   *
   * Note: This is really ugly, but currently the only way to avoid that the user
   * has to specify these parameters twice in the input file.
   *
   * ToDO: Find something nicer here!
   *
   * \author mayr.mt \date 12/2013
   */
  // ---------------------------------------------------------------------------
  switch (probtype)
  {
    case ProblemType::fsi:
    case ProblemType::fsi_redmodels:
    {
      const Teuchos::ParameterList& fsidyn = problem->FSIDynamicParams();
      const Teuchos::ParameterList& fsiada = fsidyn.sublist("TIMEADAPTIVITY");
      if (DRT::INPUT::IntegralValue<bool>(fsiada, "TIMEADAPTON"))
      {
        // overrule time step size adaptivity control parameters
        if (taflags.get<std::string>("KIND") != "NONE")
        {
          taflags.set<int>("ADAPTSTEPMAX", fsiada.get<int>("ADAPTSTEPMAX"));
          taflags.set<double>("STEPSIZEMAX", fsiada.get<double>("DTMAX"));
          taflags.set<double>("STEPSIZEMIN", fsiada.get<double>("DTMIN"));
          taflags.set<double>("SIZERATIOMAX", fsiada.get<double>("SIZERATIOMAX"));
          taflags.set<double>("SIZERATIOMIN", fsiada.get<double>("SIZERATIOMIN"));
          taflags.set<double>("SIZERATIOSCALE", fsiada.get<double>("SAFETYFACTOR"));

          if (actdis_->Comm().MyPID() == 0)
          {
            IO::cout
                << "*** Due to FSI time step size adaptivity with structure based error "
                   "estimation,\n"
                   "algorithmic control parameters in STRUCTURAL DYNAMIC/TIMEADAPTIVITY have been\n"
                   "overwritten by those from FSI DYNAMIC/TIMEADAPTIVITY."
                << IO::endl
                << IO::endl;
          }
        }
      }
      break;
    }
    default:
    {
      // do nothing
      break;
    }
  }

  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::SetGlobalState(
    Teuchos::RCP<STR::TIMINT::BaseDataGlobalState>& dataglobalstate,
    const Teuchos::RCP<const STR::TIMINT::BaseDataSDyn>& datasdyn)
{
  dataglobalstate = STR::TIMINT::BuildDataGlobalState();
  dataglobalstate->Init(actdis_, *sdyn_, datasdyn);
  dataglobalstate->Setup();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::SetTimeIntegrationStrategy(
    Teuchos::RCP<STR::TIMINT::Base>& ti_strategy,
    const Teuchos::RCP<STR::TIMINT::BaseDataIO>& dataio,
    const Teuchos::RCP<STR::TIMINT::BaseDataSDyn>& datasdyn,
    const Teuchos::RCP<STR::TIMINT::BaseDataGlobalState>& dataglobalstate, const int& restart)
{
  ti_strategy = STR::TIMINT::BuildStrategy(*sdyn_);
  ti_strategy->Init(dataio, datasdyn, dataglobalstate);

  /* In the restart case, we Setup the structural time integration after the
   * discretization has been redistributed. See STR::TIMINT::Base::ReadRestart()
   * for more information.                                     hiermeier 05/16*/
  if (not restart) ti_strategy->Setup();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::SetStructureWrapper(const Teuchos::ParameterList& ioflags,
    const Teuchos::ParameterList& sdyn, const Teuchos::ParameterList& xparams,
    const Teuchos::ParameterList& taflags, Teuchos::RCP<STR::TIMINT::Base> ti_strategy)
{
  // if no adaptive wrapper was found, we try to create a standard one
  if (str_wrapper_.is_null()) CreateWrapper(ti_strategy);

  if (str_wrapper_.is_null()) dserror("No proper time integration found!");

  return;
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void ADAPTER::StructureBaseAlgorithmNew::CreateWrapper(Teuchos::RCP<STR::TIMINT::Base> ti_strategy)
{
  // get the problem instance and the problem type
  DRT::Problem* problem = DRT::Problem::Instance();
  ProblemType probtype = problem->GetProblemType();

  switch (probtype)
  {
    case ProblemType::fsi:
    case ProblemType::fsi_redmodels:
    case ProblemType::fsi_lung:
    case ProblemType::gas_fsi:
    case ProblemType::ac_fsi:
    case ProblemType::biofilm_fsi:
    case ProblemType::thermo_fsi:
    case ProblemType::fsi_xfem:
    {
      const Teuchos::ParameterList& fsidyn = problem->FSIDynamicParams();
      const int coupling = DRT::INPUT::IntegralValue<int>(fsidyn, "COUPALGO");

      // Are there any constraint conditions active?
      const std::set<INPAR::STR::ModelType>& modeltypes =
          ti_strategy->GetDataSDyn().GetModelTypes();
      if (modeltypes.find(INPAR::STR::model_lag_pen_constraint) != modeltypes.end())
      {
        if ((actdis_->Comm()).MyPID() == 0)
          IO::cout << "Using StructureNOXCorrectionWrapper()..." << IO::endl;

        if (coupling == fsi_iter_constr_monolithicstructuresplit or
            coupling == fsi_iter_constr_monolithicfluidsplit)
          str_wrapper_ = Teuchos::rcp(new FSIStructureWrapper(
              Teuchos::rcp(new StructureNOXCorrectionWrapper(ti_strategy))));
        else
          str_wrapper_ = Teuchos::rcp(new StructureConstrMerged(
              Teuchos::rcp(new StructureNOXCorrectionWrapper(ti_strategy))));
      }
      else
      {
        if (coupling == fsi_iter_lung_monolithicstructuresplit or
            coupling == fsi_iter_lung_monolithicfluidsplit)
        {
          if ((actdis_->Comm()).MyPID() == 0)
            IO::cout << "Using StructureNOXCorrectionWrapper()..." << IO::endl;
          str_wrapper_ = Teuchos::rcp(
              new StructureLung(Teuchos::rcp(new StructureNOXCorrectionWrapper(ti_strategy))));
        }
        else
        {
          str_wrapper_ =
              Teuchos::rcp(new FSIStructureWrapper(ti_strategy));  // case of partitioned fsi
        }
      }
      break;
    }
    case ProblemType::fbi:
    {
      const Teuchos::ParameterList& fsidyn = problem->FSIDynamicParams();
      if (DRT::INPUT::IntegralValue<INPAR::FSI::PartitionedCouplingMethod>(
              fsidyn.sublist("PARTITIONED SOLVER"), "PARTITIONED") == INPAR::FSI::DirichletNeumann)
        str_wrapper_ = Teuchos::rcp(new FBIStructureWrapper(ti_strategy));
      else
        dserror("Only DirichletNeumann is implemented for FBI so far");
      break;
    }
    case ProblemType::immersed_fsi:
    {
      str_wrapper_ = Teuchos::rcp(new FSIStructureWrapperImmersed(ti_strategy));
      break;
    }
    case ProblemType::ssi:
    case ProblemType::ssti:
    {
      str_wrapper_ = Teuchos::rcp(new SSIStructureWrapper(ti_strategy));
      break;
    }
    case ProblemType::pasi:
    {
      str_wrapper_ = Teuchos::rcp(new PASIStructureWrapper(ti_strategy));
      break;
    }
    case ProblemType::redairways_tissue:
      str_wrapper_ = Teuchos::rcp(new StructureRedAirway(ti_strategy));
      break;
    case ProblemType::poroelast:
    case ProblemType::poroscatra:
    case ProblemType::fpsi:
    case ProblemType::fps3i:
    case ProblemType::fpsi_xfem:
    {
      const Teuchos::ParameterList& porodyn = problem->PoroelastDynamicParams();
      const INPAR::POROELAST::SolutionSchemeOverFields coupling =
          DRT::INPUT::IntegralValue<INPAR::POROELAST::SolutionSchemeOverFields>(
              porodyn, "COUPALGO");
      // Are there any constraint conditions active?
      const std::set<INPAR::STR::ModelType>& modeltypes =
          ti_strategy->GetDataSDyn().GetModelTypes();
      if (modeltypes.find(INPAR::STR::model_lag_pen_constraint) != modeltypes.end())
      {
        if (coupling == INPAR::POROELAST::Monolithic_structuresplit or
            coupling == INPAR::POROELAST::Monolithic_fluidsplit or
            coupling == INPAR::POROELAST::Monolithic_nopenetrationsplit)
          str_wrapper_ = Teuchos::rcp(new FPSIStructureWrapper(ti_strategy));
        else
          str_wrapper_ = Teuchos::rcp(new StructureConstrMerged(ti_strategy));
      }
      else
      {
        str_wrapper_ = Teuchos::rcp(new FPSIStructureWrapper(ti_strategy));
      }
      break;
    }
    case ProblemType::struct_ale:
    {
      str_wrapper_ = Teuchos::rcp(new StructAleWrapper(ti_strategy));
      dserror("ProblemType::struct_ale not supported, yet");
      break;
    }
    default:
      /// wrap time loop for pure structure problems
      str_wrapper_ = (Teuchos::rcp(new StructureTimeLoop(ti_strategy)));
      break;
  }

  return;
}