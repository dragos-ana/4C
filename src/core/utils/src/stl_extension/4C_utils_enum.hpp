// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_ENUM_HPP
#define FOUR_C_UTILS_ENUM_HPP

#include "4C_config.hpp"

// 1) Include <format> before the magic_enum header to enable std::format support.
#include <format>

// 2) Include magic_enum header.
#include <magic_enum/magic_enum_format.hpp>
#include <magic_enum/magic_enum_iostream.hpp>
#include <magic_enum/magic_enum_switch.hpp>

FOUR_C_NAMESPACE_OPEN

/**
 * This namespace contains helpers for static reflection of enums.
 *
 * @note The implementation is currently provided by the magic_enum library.
 */
namespace EnumTools
{
  // Import the stream insertion and extraction operators for enums.
  using magic_enum::iostream_operators::operator<<;
  using magic_enum::iostream_operators::operator>>;

  // Import everything from magic_enum into our namespace.
  using namespace magic_enum;
}  // namespace EnumTools

FOUR_C_NAMESPACE_CLOSE

#endif
