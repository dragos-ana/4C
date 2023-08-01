/*---------------------------------------------------------------------------*/
/*! \file
\brief surface tension interface viscosity handler for smoothed particle hydrodynamics (SPH)
       interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "baci_particle_interaction_sph_surface_tension_interface_viscosity.H"

#include "baci_particle_interaction_sph_kernel.H"
#include "baci_particle_interaction_material_handler.H"
#include "baci_particle_interaction_sph_equationofstate.H"
#include "baci_particle_interaction_sph_equationofstate_bundle.H"
#include "baci_particle_interaction_sph_neighbor_pairs.H"
#include "baci_particle_interaction_sph_artificialviscosity.H"

#include "baci_particle_interaction_utils.H"

#include "baci_particle_engine_interface.H"
#include "baci_particle_engine_container.H"

#include "baci_utils_exceptions.H"

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
PARTICLEINTERACTION::SPHInterfaceViscosity::SPHInterfaceViscosity(
    const Teuchos::ParameterList& params)
    : params_sph_(params),
      liquidtype_(PARTICLEENGINE::Phase1),
      gastype_(PARTICLEENGINE::Phase2),
      artvisc_lg_int_(params_sph_.get<double>("INTERFACE_VISCOSITY_LIQUIDGAS")),
      artvisc_sl_int_(params_sph_.get<double>("INTERFACE_VISCOSITY_SOLIDLIQUID")),
      trans_ref_temp_(params_sph_.get<double>("TRANS_REF_TEMPERATURE")),
      trans_dT_intvisc_(params_sph_.get<double>("TRANS_DT_INTVISC"))
{
  // empty constructor
}

PARTICLEINTERACTION::SPHInterfaceViscosity::~SPHInterfaceViscosity() = default;

void PARTICLEINTERACTION::SPHInterfaceViscosity::Init()
{
  // init artificial viscosity handler
  InitArtificialViscosityHandler();

  // init fluid particle types
  fluidtypes_ = {liquidtype_, gastype_};

  // init with potential boundary particle types
  boundarytypes_ = {PARTICLEENGINE::BoundaryPhase, PARTICLEENGINE::RigidPhase};

  // safety check
  if (trans_dT_intvisc_ > 0.0)
  {
    if (DRT::INPUT::IntegralValue<INPAR::PARTICLE::TemperatureEvaluationScheme>(
            params_sph_, "TEMPERATUREEVALUATION") == INPAR::PARTICLE::NoTemperatureEvaluation)
      dserror("temperature evaluation needed for linear transition of interface viscosity!");
  }
}

void PARTICLEINTERACTION::SPHInterfaceViscosity::Setup(
    const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface,
    const std::shared_ptr<PARTICLEINTERACTION::SPHKernelBase> kernel,
    const std::shared_ptr<PARTICLEINTERACTION::MaterialHandler> particlematerial,
    const std::shared_ptr<PARTICLEINTERACTION::SPHEquationOfStateBundle> equationofstatebundle,
    const std::shared_ptr<PARTICLEINTERACTION::SPHNeighborPairs> neighborpairs)
{
  // set interface to particle engine
  particleengineinterface_ = particleengineinterface;

  // set particle container bundle
  particlecontainerbundle_ = particleengineinterface_->GetParticleContainerBundle();

  // set kernel handler
  kernel_ = kernel;

  // set equation of state handler
  equationofstatebundle_ = equationofstatebundle;

  // set neighbor pair handler
  neighborpairs_ = neighborpairs;

  // setup artificial viscosity handler
  artificialviscosity_->Setup();

  // safety check
  for (const auto& type_i : fluidtypes_)
    if (not particlecontainerbundle_->GetParticleTypes().count(type_i))
      dserror("no particle container for particle type '%s' found!",
          PARTICLEENGINE::EnumToTypeName(type_i).c_str());

  // update with actual boundary particle types
  const auto boundarytypes = boundarytypes_;
  for (const auto& type_i : boundarytypes)
    if (not particlecontainerbundle_->GetParticleTypes().count(type_i))
      boundarytypes_.erase(type_i);

  // determine size of vectors indexed by particle types
  const int typevectorsize = *(--particlecontainerbundle_->GetParticleTypes().end()) + 1;

  // allocate memory to hold particle types
  fluidmaterial_.resize(typevectorsize);

  // iterate over all fluid particle types
  for (const auto& type_i : fluidtypes_)
  {
    fluidmaterial_[type_i] = dynamic_cast<const MAT::PAR::ParticleMaterialSPHFluid*>(
        particlematerial->GetPtrToParticleMatParameter(type_i));
  }
}

void PARTICLEINTERACTION::SPHInterfaceViscosity::ComputeInterfaceViscosityContribution() const
{
  // compute interface viscosity contribution (particle contribution)
  ComputeInterfaceViscosityParticleContribution();

  // compute interface viscosity contribution (particle-boundary contribution)
  ComputeInterfaceViscosityParticleBoundaryContribution();
}

void PARTICLEINTERACTION::SPHInterfaceViscosity::InitArtificialViscosityHandler()
{
  // create artificial viscosity handler
  artificialviscosity_ = std::unique_ptr<PARTICLEINTERACTION::SPHArtificialViscosity>(
      new PARTICLEINTERACTION::SPHArtificialViscosity());

  // init artificial viscosity handler
  artificialviscosity_->Init();
}

void PARTICLEINTERACTION::SPHInterfaceViscosity::ComputeInterfaceViscosityParticleContribution()
    const
{
  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->GetRelevantParticlePairIndicesForEqualCombination(fluidtypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->GetRefToParticlePairData()[particlepairindex];

    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->GetSpecificContainer(type_j, status_j);

    // get material for particle types
    const MAT::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[type_i];
    const MAT::PAR::ParticleMaterialSPHFluid* material_j = fluidmaterial_[type_j];

    // get pointer to particle states
    const double* rad_i = container_i->GetPtrToState(PARTICLEENGINE::Radius, particle_i);
    const double* mass_i = container_i->GetPtrToState(PARTICLEENGINE::Mass, particle_i);
    const double* dens_i = container_i->GetPtrToState(PARTICLEENGINE::Density, particle_i);
    const double* vel_i = container_i->GetPtrToState(PARTICLEENGINE::Velocity, particle_i);
    const double* cfg_i =
        container_i->GetPtrToState(PARTICLEENGINE::ColorfieldGradient, particle_i);
    const double* temp_i = container_i->CondGetPtrToState(PARTICLEENGINE::Temperature, particle_i);
    double* acc_i = container_i->GetPtrToState(PARTICLEENGINE::Acceleration, particle_i);

    const double* rad_j = container_j->GetPtrToState(PARTICLEENGINE::Radius, particle_j);
    const double* mass_j = container_j->GetPtrToState(PARTICLEENGINE::Mass, particle_j);
    const double* dens_j = container_j->GetPtrToState(PARTICLEENGINE::Density, particle_j);
    const double* vel_j = container_j->GetPtrToState(PARTICLEENGINE::Velocity, particle_j);
    const double* cfg_j =
        container_j->GetPtrToState(PARTICLEENGINE::ColorfieldGradient, particle_j);
    const double* temp_j = container_j->CondGetPtrToState(PARTICLEENGINE::Temperature, particle_j);
    double* acc_j = container_j->GetPtrToState(PARTICLEENGINE::Acceleration, particle_j);

    // get smoothing length
    const double h_i = kernel_->SmoothingLength(rad_i[0]);
    const double h_j = (rad_i[0] == rad_j[0]) ? h_i : kernel_->SmoothingLength(rad_j[0]);

    // evaluate transition factor above reference temperature
    double tempfac_i = 1.0;
    double tempfac_j = 1.0;
    if (trans_dT_intvisc_ > 0.0)
    {
      tempfac_i =
          UTILS::CompLinTrans(temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_dT_intvisc_);
      tempfac_j =
          UTILS::CompLinTrans(temp_j[0], trans_ref_temp_, trans_ref_temp_ + trans_dT_intvisc_);
    }

    // compute artificial viscosity
    const double artvisc_i =
        UTILS::VecNormTwo(cfg_i) * h_i * artvisc_lg_int_ + tempfac_i * artvisc_sl_int_;
    const double artvisc_j =
        UTILS::VecNormTwo(cfg_j) * h_j * artvisc_lg_int_ + tempfac_j * artvisc_sl_int_;

    // evaluate artificial viscosity
    if (artvisc_i > 0.0 or artvisc_j > 0.0)
    {
      // particle averaged smoothing length
      const double h_ij = 0.5 * (h_i + h_j);

      // get speed of sound
      const double c_i = material_i->SpeedOfSound();
      const double c_j = (type_i == type_j) ? c_i : material_j->SpeedOfSound();

      // particle averaged speed of sound
      const double c_ij = 0.5 * (c_i + c_j);

      // particle averaged density
      const double dens_ij = 0.5 * (dens_i[0] + dens_j[0]);

      // evaluate artificial viscosity
      artificialviscosity_->ArtificialViscosity(vel_i, vel_j, mass_i, mass_j, artvisc_i, artvisc_j,
          particlepair.dWdrij_, particlepair.dWdrji_, dens_ij, h_ij, c_ij, particlepair.absdist_,
          particlepair.e_ij_, acc_i, acc_j);
    }
  }
}

void PARTICLEINTERACTION::SPHInterfaceViscosity::
    ComputeInterfaceViscosityParticleBoundaryContribution() const
{
  // get relevant particle pair indices
  std::vector<int> relindices;
  neighborpairs_->GetRelevantParticlePairIndicesForDisjointCombination(
      fluidtypes_, boundarytypes_, relindices);

  // iterate over relevant particle pairs
  for (const int particlepairindex : relindices)
  {
    const SPHParticlePair& particlepair =
        neighborpairs_->GetRefToParticlePairData()[particlepairindex];

    // access values of local index tuples of particle i and j
    PARTICLEENGINE::TypeEnum type_i;
    PARTICLEENGINE::StatusEnum status_i;
    int particle_i;
    std::tie(type_i, status_i, particle_i) = particlepair.tuple_i_;

    PARTICLEENGINE::TypeEnum type_j;
    PARTICLEENGINE::StatusEnum status_j;
    int particle_j;
    std::tie(type_j, status_j, particle_j) = particlepair.tuple_j_;

    // swap fluid particle and boundary particle
    const bool swapparticles = boundarytypes_.count(type_i);
    if (swapparticles)
    {
      std::tie(type_i, status_i, particle_i) = particlepair.tuple_j_;
      std::tie(type_j, status_j, particle_j) = particlepair.tuple_i_;
    }

    // absolute distance between particles
    const double absdist = particlepair.absdist_;

    // versor from particle j to i
    double e_ij[3];
    UTILS::VecSet(e_ij, particlepair.e_ij_);
    if (swapparticles) UTILS::VecScale(e_ij, -1.0);

    // first derivative of kernel
    const double dWdrij = (swapparticles) ? particlepair.dWdrji_ : particlepair.dWdrij_;

    // get corresponding particle containers
    PARTICLEENGINE::ParticleContainer* container_i =
        particlecontainerbundle_->GetSpecificContainer(type_i, status_i);

    PARTICLEENGINE::ParticleContainer* container_j =
        particlecontainerbundle_->GetSpecificContainer(type_j, status_j);

    // get material for particle types
    const MAT::PAR::ParticleMaterialSPHFluid* material_i = fluidmaterial_[type_i];

    // get equation of state for particle types
    const PARTICLEINTERACTION::SPHEquationOfStateBase* equationofstate_i =
        equationofstatebundle_->GetPtrToSpecificEquationOfState(type_i);

    // get pointer to particle states
    const double* rad_i = container_i->GetPtrToState(PARTICLEENGINE::Radius, particle_i);
    const double* mass_i = container_i->GetPtrToState(PARTICLEENGINE::Mass, particle_i);
    const double* dens_i = container_i->GetPtrToState(PARTICLEENGINE::Density, particle_i);
    const double* vel_i = container_i->GetPtrToState(PARTICLEENGINE::Velocity, particle_i);
    const double* cfg_i =
        container_i->GetPtrToState(PARTICLEENGINE::ColorfieldGradient, particle_i);
    const double* temp_i = container_i->CondGetPtrToState(PARTICLEENGINE::Temperature, particle_i);

    double* acc_i = nullptr;
    if (status_i == PARTICLEENGINE::Owned)
      acc_i = container_i->GetPtrToState(PARTICLEENGINE::Acceleration, particle_i);

    // get pointer to boundary particle states
    const double* mass_j = container_i->GetPtrToState(PARTICLEENGINE::Mass, particle_i);
    const double* press_j =
        container_j->GetPtrToState(PARTICLEENGINE::BoundaryPressure, particle_j);
    const double* vel_j = container_j->GetPtrToState(PARTICLEENGINE::BoundaryVelocity, particle_j);

    double temp_dens(0.0);
    temp_dens = equationofstate_i->PressureToDensity(press_j[0], material_i->initDensity_);
    const double* dens_j = &temp_dens;

    // get smoothing length
    const double h_i = kernel_->SmoothingLength(rad_i[0]);

    // evaluate transition factor above reference temperature
    double tempfac_i = 1.0;
    if (trans_dT_intvisc_ > 0.0)
      tempfac_i =
          UTILS::CompLinTrans(temp_i[0], trans_ref_temp_, trans_ref_temp_ + trans_dT_intvisc_);

    // compute artificial viscosity
    const double artvisc_i =
        UTILS::VecNormTwo(cfg_i) * h_i * artvisc_lg_int_ + tempfac_i * artvisc_sl_int_;

    // evaluate artificial viscosity
    if (artvisc_i > 0.0)
    {
      // get speed of sound
      const double c_i = material_i->SpeedOfSound();

      // particle averaged density
      const double dens_ij = 0.5 * (dens_i[0] + dens_j[0]);

      // evaluate artificial viscosity
      artificialviscosity_->ArtificialViscosity(vel_i, vel_j, mass_i, mass_j, artvisc_i, 0.0,
          dWdrij, 0.0, dens_ij, h_i, c_i, absdist, e_ij, acc_i, nullptr);
    }
  }
}