/*---------------------------------------------------------------------*/
/*! \file

\brief Input for RedAirway, RedAcinus, RedInterAcinarDep, RedAirBloodScatra and
RedAirBloodScatraLine3 elements


\level 3

*/
/*---------------------------------------------------------------------*/

#include "4C_global_data.hpp"
#include "4C_io_linedefinition.hpp"
#include "4C_mat_maxwell_0d_acinus.hpp"
#include "4C_red_airways_elementbase.hpp"

FOUR_C_NAMESPACE_OPEN

using namespace DRT::UTILS;


/*----------------------------------------------------------------------*
| Read in the RED_AIRWAY elements                                       |
*-----------------------------------------------------------------------*/
bool DRT::ELEMENTS::RedAirway::ReadElement(
    const std::string& eletype, const std::string& distype, INPUT::LineDefinition* linedef)
{
  const int ndim = GLOBAL::Problem::Instance()->NDim();
  if (ndim != 3)
    FOUR_C_THROW("Problem defined as %dd, but found Reduced dimensional AIRWAY element.", ndim);

  // Read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  // Read the element type, the element specific variables and store them to airwayParams_
  linedef->ExtractString("TYPE", elem_type_);
  if (elem_type_ == "Resistive" || elem_type_ == "InductoResistive" ||
      elem_type_ == "ComplientResistive" || elem_type_ == "RLC" ||
      elem_type_ == "ViscoElasticRLC" || elem_type_ == "ConvectiveViscoElasticRLC")
  {
    linedef->ExtractString("Resistance", resistance_);
    linedef->ExtractString("ElemSolvingType", elemsolving_type_);

    double Ew, tw, A, Ts, Phis, nu, velPow;
    int generation;
    linedef->ExtractDouble("PowerOfVelocityProfile", velPow);
    linedef->ExtractDouble("WallElasticity", Ew);
    linedef->ExtractDouble("PoissonsRatio", nu);
    linedef->ExtractDouble("ViscousTs", Ts);
    linedef->ExtractDouble("ViscousPhaseShift", Phis);
    linedef->ExtractDouble("WallThickness", tw);
    linedef->ExtractDouble("Area", A);
    linedef->ExtractInt("Generation", generation);

    if (linedef->HaveNamed("AirwayColl"))
    {
      double airwayColl;
      linedef->ExtractDouble("AirwayColl", airwayColl);
      double sc, so, Pcrit_c, Pcrit_o, open_init;
      linedef->ExtractDouble("S_Close", sc);
      linedef->ExtractDouble("S_Open", so);
      linedef->ExtractDouble("Pcrit_Open", Pcrit_o);
      linedef->ExtractDouble("Pcrit_Close", Pcrit_c);
      linedef->ExtractDouble("Open_Init", open_init);
      airway_params_.airway_coll = airwayColl;
      airway_params_.s_close = sc;
      airway_params_.s_open = so;
      airway_params_.p_crit_open = Pcrit_o;
      airway_params_.p_crit_close = Pcrit_c;
      airway_params_.open_init = open_init;
    }

    // Correct the velocity profile power
    // this is because the 2.0 is the minimum energy consumtive laminar profile
    if (velPow < 2.0) velPow = 2.0;
    airway_params_.power_velocity_profile = velPow;
    airway_params_.wall_elasticity = Ew;
    airway_params_.poisson_ratio = nu;
    airway_params_.wall_thickness = tw;
    airway_params_.area = A;
    airway_params_.viscous_Ts = Ts;
    airway_params_.viscous_phase_shift = Phis;
    airway_params_.generation = generation;
    if (linedef->HaveNamed("BranchLength"))
    {
      double l_branch = 0.0;
      linedef->ExtractDouble("BranchLength", l_branch);
      airway_params_.branch_length = l_branch;
    }
  }
  else
  {
    FOUR_C_THROW(
        "Reading type of RED_AIRWAY element failed. Possible types: ComplientResistive/"
        "PoiseuilleResistive/TurbulentPoiseuilleResistive/InductoResistive/RLC/ViscoElasticRLC/"
        "ConvectiveViscoElasticRLC");
    exit(1);
  }

  return true;
}


/*----------------------------------------------------------------------*
| Read in the RED_ACINUS elements                                       |
*-----------------------------------------------------------------------*/
bool DRT::ELEMENTS::RedAcinus::ReadElement(
    const std::string& eletype, const std::string& distype, INPUT::LineDefinition* linedef)
{
  const int ndim = GLOBAL::Problem::Instance()->NDim();
  if (ndim != 3)
    FOUR_C_THROW("Problem defined as %dd, but found Reduced dimensional ACINUS element.", ndim);

  // Read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  // Read the element type, the element specific variables and store them to acinusParams_
  linedef->ExtractString("TYPE", elem_type_);
  if (elem_type_ == "NeoHookean" || elem_type_ == "Exponential" ||
      elem_type_ == "DoubleExponential" || elem_type_ == "VolumetricOgden")
  {
    double acinusVol, alveolarDuctVol, A;
    const int generation = -1;
    linedef->ExtractDouble("AcinusVolume", acinusVol);
    linedef->ExtractDouble("AlveolarDuctVolume", alveolarDuctVol);
    linedef->ExtractDouble("Area", A);

    acinus_params_.volume_relaxed = acinusVol;
    acinus_params_.alveolar_duct_volume = alveolarDuctVol;
    acinus_params_.area = A;
    acinus_params_.volume_init = acinusVol;
    acinus_params_.generation = generation;

    // Setup material, calls overloaded function Setup(linedef) for each Maxwell_0d_acinus material
    Teuchos::RCP<CORE::MAT::Material> mat = Material();
    Teuchos::RCP<MAT::Maxwell0dAcinus> acinus_mat =
        Teuchos::rcp_dynamic_cast<MAT::Maxwell0dAcinus>(Material());
    acinus_mat->Setup(linedef);
  }
  else
  {
    FOUR_C_THROW(
        "Reading type of RED_ACINUS element failed. Possible types: NeoHookean/ Exponential"
        "/ DoubleExponential/ VolumetricOgden");
    exit(1);
  }

  return true;
}


/*----------------------------------------------------------------------*
| Read in the RED_ACINAR_INTER_DEP elements                             |
*-----------------------------------------------------------------------*/
bool DRT::ELEMENTS::RedInterAcinarDep::ReadElement(
    const std::string& eletype, const std::string& distype, INPUT::LineDefinition* linedef)
{
  const int ndim = GLOBAL::Problem::Instance()->NDim();
  if (ndim != 3)
    FOUR_C_THROW(
        "Problem defined as %dd, but found Reduced dimensional INTER ACINAR DEPENDENCE element.",
        ndim);

  // set generation
  const int generation = -2;
  generation_ = generation;

  // Read number of material model
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);


  return true;
}


/*----------------------------------------------------------------------*
| Read in the Scatra elements                                           |
*-----------------------------------------------------------------------*/
bool DRT::ELEMENTS::RedAirBloodScatra::ReadElement(
    const std::string& eletype, const std::string& distype, INPUT::LineDefinition* linedef)
{
  const int ndim = GLOBAL::Problem::Instance()->NDim();
  if (ndim != 3)
    FOUR_C_THROW("Problem defined as %dd, but found Reduced dimensional Scatra element.", ndim);

  // read number of material model
  const int generation = -2;
  generation_ = generation;

  double diff = 0.0;
  linedef->ExtractDouble("DiffusionCoefficient", diff);
  elem_params_["DiffusionCoefficient"] = diff;

  double th = 0.0;
  linedef->ExtractDouble("WallThickness", th);
  elem_params_["WallThickness"] = th;

  double percDiffArea = 0.0;
  linedef->ExtractDouble("PercentageOfDiffusionArea", percDiffArea);
  elem_params_["PercentageOfDiffusionArea"] = percDiffArea;

  return true;
}


/*----------------------------------------------------------------------*
| Read in the Scatra elements                                           |
*-----------------------------------------------------------------------*/
bool DRT::ELEMENTS::RedAirBloodScatraLine3::ReadElement(
    const std::string& eletype, const std::string& distype, INPUT::LineDefinition* linedef)
{
  const int ndim = GLOBAL::Problem::Instance()->NDim();
  if (ndim != 3)
    FOUR_C_THROW("Problem defined as %dd, but found Reduced dimensional Scatra element.", ndim);

  double diff = 0.0;
  linedef->ExtractDouble("DiffusionCoefficient", diff);
  elem_params_["DiffusionCoefficient"] = diff;

  double th = 0.0;
  linedef->ExtractDouble("WallThickness", th);
  elem_params_["WallThickness"] = th;

  double percDiffArea = 0.0;
  linedef->ExtractDouble("PercentageOfDiffusionArea", percDiffArea);
  elem_params_["PercentageOfDiffusionArea"] = percDiffArea;

  return true;
}

FOUR_C_NAMESPACE_CLOSE