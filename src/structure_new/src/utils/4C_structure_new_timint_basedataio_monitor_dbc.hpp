// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_MONITOR_DBC_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAIO_MONITOR_DBC_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace TimeInt
  {
    /** \brief Input data container for monitoring reaction forces for structural (time) integration
     *
     * */
    class ParamsMonitorDBC
    {
     public:
      /// constructor
      ParamsMonitorDBC();

      /// destructor
      virtual ~ParamsMonitorDBC() = default;

      /// initialize the class variables
      void init(const Teuchos::ParameterList& IO_monitor_dbc_structure_paramslist);

      /// setup new class variables
      void setup();


      /// output interval regarding steps: write output every INTERVAL_STEPS steps
      int output_interval_in_steps() const
      {
        check_init_setup();
        return output_interval_steps_;
      };

      /// precision for file output
      int file_precision() const
      {
        check_init_setup();
        return of_precision_;
      };

      /// precision for screen output
      int screen_precision() const
      {
        check_init_setup();
        return os_precision_;
      };

      /// file type ending
      std::string const& file_type() const
      {
        check_init_setup();
        return file_type_;
      };

      /// whether to write header in csv files
      bool write_header() const
      {
        check_init_setup();
        return write_header_;
      }


     private:
      /// get the init indicator status
      const bool& is_init() const { return isinit_; };

      /// get the setup indicator status
      const bool& is_setup() const { return issetup_; };

      /// Check if init() and setup() have been called, yet.
      void check_init_setup() const;


     private:
      /// @name variables for internal use only
      /// @{
      ///
      bool isinit_;

      bool issetup_;
      /// @}

      /// @name variables controlling output
      /// @{

      /// output interval regarding steps: write output every INTERVAL_STEPS steps
      int output_interval_steps_;

      /// precision for file output
      unsigned of_precision_;

      /// precision for screen output
      unsigned os_precision_;

      /// file type
      std::string file_type_;

      /// write header in csv files
      bool write_header_;

      /// @}
    };

  }  // namespace TimeInt
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
