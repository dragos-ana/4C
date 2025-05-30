// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_GEOMETRY_PAIR_HPP
#define FOUR_C_GEOMETRY_PAIR_HPP


#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// Forward declarations.
namespace Core::Elements
{
  class Element;
}


namespace GeometryPair
{
  /**
   * \brief Base class for all geometry pairs.
   */
  class GeometryPairBase
  {
   public:
    /**
     * \brief Constructor.
     * @param element1 Pointer to the first element.
     * @param element2 Pointer to the second element.
     */
    GeometryPairBase(
        const Core::Elements::Element* element1, const Core::Elements::Element* element2)
        : element1_(element1), element2_(element2) {};

    /**
     * \brief Destructor.
     */
    virtual ~GeometryPairBase() = default;

    /**
     * \brief Pointer to the first element.
     */
    inline const Core::Elements::Element* element1() const { return element1_; };

    /**
     * \brief Pointer to the second element.
     */
    inline const Core::Elements::Element* element2() const { return element2_; };

    /**
     * \brief Set the pointer to the second element.
     *
     * For surface elements the pairs need the face element representing the surface, not the volume
     * element.
     */
    virtual inline void set_element2(const Core::Elements::Element* element2)
    {
      element2_ = element2;
    };

   private:
    //! Link to the element containing the first geometry.
    const Core::Elements::Element* element1_;

    //! Link to the element containing the second geometry.
    const Core::Elements::Element* element2_;
  };
}  // namespace GeometryPair

FOUR_C_NAMESPACE_CLOSE

#endif
