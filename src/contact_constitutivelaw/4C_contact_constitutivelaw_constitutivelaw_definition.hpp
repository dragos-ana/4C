// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_CONSTITUTIVELAW_DEFINITION_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_CONSTITUTIVELAW_DEFINITION_HPP

#include "4C_config.hpp"

#include "4C_io_linecomponent.hpp"

#include <Teuchos_Array.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputFile;
}  // namespace Core::IO

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Global
{
  class Problem;
}

namespace Inpar::CONTACT
{
  enum class ConstitutiveLawType;
}

namespace CONTACT
{
  class Interface;
  namespace CONSTITUTIVELAW
  {
    class Container;
    class Bundle;
    /**
     * \brief Definition of a valid contact constitutive law in 4C input
     *
     * This is basically a clone of \see MaterialDefinition
     * A ConstitutivelawDefinition is the definition of a --Contact Constitutive Law in the DAT
     * file section. This definition includes the knowledge what this section looks like, how to
     * read it and how to write it. In particular given a ConstitutivelawDefinition object it is
     * possible to read a DAT file and create Constitutivelaw::Container objects for each line in
     * this section.
     */
    class LawDefinition
    {
     public:
      /**
       * \brief default constructor
       *
       * \param[in] name name of contact constitutive law
       * \param[in] description of contact constitutive law type
       * \param[in] type of contact constitutive law to be built
       */
      LawDefinition(std::string name, std::string description,
          Inpar::CONTACT::ConstitutiveLawType contactconstitutivelawlawtype);

      // destructor
      virtual ~LawDefinition() = default;

      /** add a concrete component to the contact constitutive law line definition
       *
       * \param[in] c new component needed as parameter for the contact constitutive law
       */
      void add_component(std::shared_ptr<Input::LineComponent> c);

      /** \brief read all materials from my input file section
       *
       * \param[in] problem global problem instance that manages the input
       * \param[in] input the input file
       * \params[inout] mmap map mapping IDs to parameters of the contact constitutive model
       */
      void read(const Global::Problem& problem, Core::IO::InputFile& input,
          CONTACT::CONSTITUTIVELAW::Bundle& bundle);

      /// print my DAT file section and possible contact constitutive laws
      std::ostream& print(std::ostream& stream,  ///< the output stream
          const Core::FE::Discretization* dis = nullptr);

      /// get contact constitutive law name
      [[nodiscard]] std::string name() const { return coconstlawname_; }

      // get contact constitutive law type
      [[nodiscard]] Inpar::CONTACT::ConstitutiveLawType type() const { return coconstlawtype_; }

      /// get contact constitutive law description
      [[nodiscard]] std::string description() const { return description_; }

      /// get contact constitutive law inputline
      [[nodiscard]] std::vector<std::shared_ptr<Input::LineComponent>> inputline() const
      {
        return inputline_;
      }

     private:
      /// name of ontact constitutive law
      std::string coconstlawname_;
      /// description of ontact constitutive law type
      std::string description_;
      /// type of ontact constitutive law to be build
      Inpar::CONTACT::ConstitutiveLawType coconstlawtype_;
      /// the list of valid components
      std::vector<std::shared_ptr<Input::LineComponent>> inputline_;
    };


    /** Add contact constitutive law definition to list of defined definitions
     *
     * \params[inout] list list of defined contact constitutive law definitions
     * \param[in] def contact constitutive law definition to add to the list
     *
     */
    void append_co_const_law_component_definition(
        std::vector<std::shared_ptr<CONTACT::CONSTITUTIVELAW::LawDefinition>>& list,
        std::shared_ptr<CONTACT::CONSTITUTIVELAW::LawDefinition> def);

  }  // namespace CONSTITUTIVELAW
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
