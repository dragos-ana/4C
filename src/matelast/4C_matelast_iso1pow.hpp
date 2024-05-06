/*----------------------------------------------------------------------*/
/*! \file
\brief Definition of classes for the isochoric contribution of an isotropic general power-type
material in terms of the first Cauchy-Green invariant

\level 1
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MATELAST_ISO1POW_HPP
#define FOUR_C_MATELAST_ISO1POW_HPP

#include "4C_config.hpp"

#include "4C_matelast_summand.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace MAT
{
  namespace ELASTIC
  {
    namespace PAR
    {
      /*!
       * @brief material parameters for isochoric contribution of a general power material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_Iso1Pow C 1 D 1
       */
      class Iso1Pow : public CORE::MAT::PAR::Parameter
      {
       public:
        /// standard constructor
        Iso1Pow(const Teuchos::RCP<CORE::MAT::PAR::Material>& matdata);

        /// @name material parameters
        //@{

        /// material parameters
        double c_;
        int d_;

        //@}

        /// Override this method and throw error, as the material should be created in within the
        /// Factory method of the elastic summand
        Teuchos::RCP<CORE::MAT::Material> CreateMaterial() override
        {
          FOUR_C_THROW(
              "Cannot create a material from this method, as it should be created in "
              "MAT::ELASTIC::Summand::Factory.");
          return Teuchos::null;
        };
      };  // class Iso1Pow

    }  // namespace PAR

    /*!
     * @brief Isochoric general power material
     *
     * This is the isochoric part of a third order hyperelastic, isotropic
     * material depending on the modified trace of the right Cauchy-Green tensor.
     *
     * Strain energy function is given by
     * \f[
     *    \Psi = C (\overline{I}_{\boldsymbol{C}}-3)^D.
     * \f]
     */
    class Iso1Pow : public Summand
    {
     public:
      /// constructor with given material parameters
      Iso1Pow(MAT::ELASTIC::PAR::Iso1Pow* params);

      /// @name Access material constants
      //@{

      /// material type
      CORE::Materials::MaterialType MaterialType() const override
      {
        return CORE::Materials::mes_iso1pow;
      }

      //@}

      // add strain energy
      void AddStrainEnergy(double& psi,  ///< strain energy function
          const CORE::LINALG::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const CORE::LINALG::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const CORE::LINALG::Matrix<6, 1>& glstrain,  ///< Green-Lagrange strain
          int gp,                                      ///< Gauss point
          int eleGID                                   ///< element GID
          ) override;

      // Add derivatives with respect to modified invariants.
      void AddDerivativesModified(
          CORE::LINALG::Matrix<3, 1>&
              dPmodI,  ///< first derivative with respect to modified invariants
          CORE::LINALG::Matrix<6, 1>&
              ddPmodII,  ///< second derivative with respect to modified invariants
          const CORE::LINALG::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          int gp,      ///< Gauss point
          int eleGID   ///< element GID
          ) override;

      /// Indicator for formulation
      void SpecifyFormulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic splitted formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic splitted formulation
          bool& viscogeneral  ///< general indicator, if one viscoelastic formulation is used
          ) override
      {
        isomod = true;
        return;
      };

     private:
      /// my material parameters
      MAT::ELASTIC::PAR::Iso1Pow* params_;
    };

  }  // namespace ELASTIC
}  // namespace MAT

FOUR_C_NAMESPACE_CLOSE

#endif