# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

#
# Set up the main documentation based on Sphinx

four_c_process_global_option(FOUR_C_BUILD_DOCUMENTATION "Build the documentation" OFF)
if(FOUR_C_BUILD_DOCUMENTATION)

  # add doc-specific cmake modules
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

  # set input and output files and directories
  set(_sphinx_SOURCE_DIR ${PROJECT_SOURCE_DIR}/doc/documentation)
  set(_sphinx_TPLDEPEND_DIR ${PROJECT_SOURCE_DIR}/dependencies/current)
  set(_sphinx_CONFIG_IN ${_sphinx_SOURCE_DIR}/conf.py.in)
  set(_sphinx_OUT_DIR ${PROJECT_BINARY_DIR}/doc/documentation)
  set(_sphinx_CONFIG_OUT ${_sphinx_OUT_DIR}/conf.py)
  set(_sphinx_HTML_DIR ${_sphinx_OUT_DIR}/html)

  #install the python virtual environment with sphinx
  find_package(Sphinx REQUIRED)

  set(_sphinx_THEME sphinx_rtd_theme)

  configure_file(${_sphinx_CONFIG_IN} ${_sphinx_CONFIG_OUT} @ONLY)
  message("Sphinx configure file conf.py is written based on ${_sphinx_CONFIG_IN}")

  add_custom_target(
    documentation ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_BINARY_DIR}/doc/tmp"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${_sphinx_OUT_DIR}/reference_docs"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${_sphinx_HTML_DIR}"
    # the following command also creates the directory ${_sphinx_OUT_DIR}/reference_docs again
    COMMAND create_rtd
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/doc/tmp"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sphinx_OUT_DIR}/reference_docs/tutorials"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_sphinx_OUT_DIR}/reference_docs/reference_images"
    COMMAND ${FOUR_C_EXECUTABLE_NAME} -h > ${_sphinx_OUT_DIR}/reference_docs/4C-help.txt
    COMMAND
      ${CMAKE_COMMAND} ${PROJECT_SOURCE_DIR} --list-presets >
      ${_sphinx_OUT_DIR}/reference_docs/4C-cmake-presets.txt
    COMMAND
      bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/extract_cmake_variables.sh
      ${PROJECT_BINARY_DIR}/CMakeCache.txt >
      ${_sphinx_OUT_DIR}/reference_docs/4C-cmake-variables.csv
    COMMAND
      ${SPHINX_EXECUTABLE} -W -n -q -b html # warnings are errors and be nit-picky
      -i "${CMAKE_CURRENT_SOURCE_DIR}/src" -i "${_sphinx_OUT_DIR}/reference_docs" -i
      "${PROJECT_SOURCE_DIR}/tests/tutorials" -i "${_sphinx_TPLDEPEND_DIR}" -o "${_sphinx_HTML_DIR}"
      -c "${_sphinx_OUT_DIR}" -s "${PROJECT_BINARY_DIR}/doc/tmp"
    DEPENDS create_rtd ${FOUR_C_EXECUTABLE_NAME}
    COMMENT "Build documentation"
    )

  list(REMOVE_ITEM CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()
