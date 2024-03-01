/*----------------------------------------------------------------------*/
/*! \file

\brief computes element volume

\level 2

*----------------------------------------------------------------------*/


#ifndef BACI_DISCRETIZATION_GEOMETRY_ELEMENT_VOLUME_HPP
#define BACI_DISCRETIZATION_GEOMETRY_ELEMENT_VOLUME_HPP


#include "baci_config.hpp"

#include "baci_cut_kernel.hpp"
#include "baci_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "baci_discretization_fem_general_utils_gausspoints.hpp"

BACI_NAMESPACE_OPEN

namespace CORE::GEO
{
  //! calculates the length of an element in given configuration
  template <CORE::FE::CellType distype, class matrixtype>
  double ElementLengthT(const matrixtype& xyze)  ///> xyze nsd = 3 coords, number of nodes
  {
    // gaussian points
    static constexpr int numnode = CORE::FE::num_nodes<distype>;
    const CORE::FE::GaussIntegration intpoints(distype);

    CORE::LINALG::Matrix<1, 1> eleCoord;
    CORE::LINALG::Matrix<1, numnode> deriv;
    CORE::LINALG::Matrix<1, 3> xjm;

    double length = 0.0;

    // integration loop
    for (int iquad = 0; iquad < intpoints.NumPoints(); ++iquad)
    {
      // coordinates of the current integration point in element coordinates \xi
      eleCoord(0) = intpoints.Point(iquad)[0];

      // shape functions and their first derivatives
      CORE::FE::shape_function_1D_deriv1(deriv, eleCoord(0), distype);

      // get transposed of the jacobian matrix d x / d \xi
      xjm = 0;

      for (int inode = 0; inode < numnode; ++inode)
        for (int i = 0; i < 1; ++i)
          for (int j = 0; j < 3; ++j) xjm(i, j) += deriv(i, inode) * xyze(j, inode);

      const double fac = intpoints.Weight(iquad) * xjm.Norm2();

      length += fac;
    }  // end loop over gauss points

    return length;
  }

  //! calculates the area of an element in given configuration
  template <CORE::FE::CellType distype, class matrixtype>
  double ElementAreaT(const matrixtype& xyze)  ///> xyze nsd = 3 coords, number of nodes
  {
    // gaussian points
    static constexpr int numnode = CORE::FE::num_nodes<distype>;
    const CORE::FE::GaussIntegration intpoints(distype);

    CORE::LINALG::Matrix<2, 1> eleCoord;
    CORE::LINALG::Matrix<2, numnode> deriv;
    CORE::LINALG::Matrix<2, 3> xjm;
    CORE::LINALG::Matrix<2, 2> xjm_xjmt;

    double area = 0.0;

    // integration loop
    for (int iquad = 0; iquad < intpoints.NumPoints(); ++iquad)
    {
      // coordinates of the current integration point in element coordinates \xi
      eleCoord(0) = intpoints.Point(iquad)[0];
      eleCoord(1) = intpoints.Point(iquad)[1];

      // shape functions and their first derivatives
      CORE::FE::shape_function_2D_deriv1(deriv, eleCoord(0), eleCoord(1), distype);

      // get transposed of the jacobian matrix d x / d \xi
      xjm = 0;

      for (int inode = 0; inode < numnode; ++inode)
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 3; ++j) xjm(i, j) += deriv(i, inode) * xyze(j, inode);

      xjm_xjmt.MultiplyNT<3>(xjm, xjm);

      const double det = xjm_xjmt.Determinant();
      const double fac = intpoints.Weight(iquad) * std::sqrt(det);

      area += fac;
    }  // end loop over gauss points

    return area;
  }

  //! calculates the volume of an element in given configuration          u.may
  template <CORE::FE::CellType distype, class matrixtype>
  double ElementVolumeT(const matrixtype& xyze  ///> xyze nsd = 3 coords, number of nodes)
  )
  {
    // number of nodes for element
    const int numnode = CORE::FE::num_nodes<distype>;
    // gaussian points
    const CORE::FE::GaussIntegration intpoints(distype);

    CORE::LINALG::Matrix<3, 1> eleCoord;
    CORE::LINALG::Matrix<3, numnode> deriv;
    CORE::LINALG::Matrix<3, 3> xjm;

    double vol = 0.0;

    // integration loop
    for (int iquad = 0; iquad < intpoints.NumPoints(); ++iquad)
    {
      // coordinates of the current integration point in element coordinates \xi
      eleCoord(0) = intpoints.Point(iquad)[0];
      eleCoord(1) = intpoints.Point(iquad)[1];
      eleCoord(2) = intpoints.Point(iquad)[2];

      // shape functions and their first derivatives
      CORE::FE::shape_function_3D_deriv1(deriv, eleCoord(0), eleCoord(1), eleCoord(2), distype);

      // get transposed of the jacobian matrix d x / d \xi
      xjm = 0;

      for (int inode = 0; inode < numnode; ++inode)
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) xjm(i, j) += deriv(i, inode) * xyze(j, inode);

      const double det = xjm.Determinant();
      const double fac = intpoints.Weight(iquad) * det;

      if (det <= 0.0) dserror("NEGATIVE JACOBIAN DETERMINANT: %g", det);

      vol += fac;
    }  // end loop over gauss points
    return vol;
  }

  /** \brief calculates the length of a edge element in given configuration
   *
   *  \params distype (in) : discretization type of the given element
   *  \params xyze    (in) : spatial coordinates of the elememnt nodes
   *                         (row = dim, col = number of nodes)
   */
  template <class matrixtype>
  double ElementLength(const CORE::FE::CellType& distype, const matrixtype& xyze)
  {
    if (distype != CORE::FE::CellType::line2 or xyze.numCols() != 2)
      dserror("Currently only line2 elements are supported!");

    // calculate the distance between the two given nodes and return
    // the value
    CORE::LINALG::Matrix<3, 1> d(&xyze(0, 0));
    const CORE::LINALG::Matrix<3, 1> x1(&xyze(0, 1));

    d.Update(1.0, x1, -1.0);
    return d.Norm2();
  }


  /** \brief calculates the area of a surface element in given configuration
   */
  template <class matrixtype>
  double ElementArea(const CORE::FE::CellType distype, const matrixtype& xyze)
  {
    switch (distype)
    {
      // --- 2-D boundary elements
      case CORE::FE::CellType::line2:
        return ElementLength(distype, xyze);
      case CORE::FE::CellType::line3:
        return ElementLengthT<CORE::FE::CellType::line3>(xyze);
      // --- 3-D boundary elements
      case CORE::FE::CellType::tri3:
        return ElementAreaT<CORE::FE::CellType::tri3>(xyze);
      case CORE::FE::CellType::tri6:
        return ElementAreaT<CORE::FE::CellType::tri6>(xyze);
      case CORE::FE::CellType::quad4:
        return ElementAreaT<CORE::FE::CellType::quad4>(xyze);
      case CORE::FE::CellType::quad8:
        return ElementAreaT<CORE::FE::CellType::quad8>(xyze);
      case CORE::FE::CellType::quad9:
        return ElementAreaT<CORE::FE::CellType::quad9>(xyze);
      default:
        dserror("Unsupported surface element type!");
        exit(EXIT_FAILURE);
    }

    return -1.0;
  }

  //! calculates the volume of an element in given configuration          u.may
  template <class matrixtype>
  double ElementVolume(const CORE::FE::CellType distype,
      const matrixtype& xyze  ///> xyze nsd = 3 coords, number of nodes
  )
  {
    switch (distype)
    {
      // --- 1-D elements -----------------------------------------------------
      case CORE::FE::CellType::line2:
        return ElementLength(distype, xyze);
      // --- 2-D elements -----------------------------------------------------
      case CORE::FE::CellType::quad4:
      case CORE::FE::CellType::tri3:
        return ElementArea(distype, xyze);
      // --- 3-D elements -----------------------------------------------------
      case CORE::FE::CellType::hex8:
        return ElementVolumeT<CORE::FE::CellType::hex8>(xyze);
      case CORE::FE::CellType::hex20:
        return ElementVolumeT<CORE::FE::CellType::hex20>(xyze);
      case CORE::FE::CellType::hex27:
        return ElementVolumeT<CORE::FE::CellType::hex27>(xyze);
      case CORE::FE::CellType::tet4:
        return ElementVolumeT<CORE::FE::CellType::tet4>(xyze);
      case CORE::FE::CellType::tet10:
        return ElementVolumeT<CORE::FE::CellType::tet10>(xyze);
      case CORE::FE::CellType::wedge6:
        return ElementVolumeT<CORE::FE::CellType::wedge6>(xyze);
      case CORE::FE::CellType::wedge15:
        return ElementVolumeT<CORE::FE::CellType::wedge15>(xyze);
      case CORE::FE::CellType::pyramid5:
        return ElementVolumeT<CORE::FE::CellType::pyramid5>(xyze);
      default:
        dserror("add you distype here");
    }
    return -1.0;
  }

}  // namespace CORE::GEO


BACI_NAMESPACE_CLOSE

#endif  // DISCRETIZATION_GEOMETRY_ELEMENT_VOLUME_H