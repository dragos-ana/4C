/*----------------------------------------------------------------------*/
/*! \file

\brief Utility functions for the geometry pairs elements.

\level 1
*/
// End doxygen header.


#ifndef BACI_GEOMETRY_PAIR_ELEMENT_FUNCTIONS_HPP
#define BACI_GEOMETRY_PAIR_ELEMENT_FUNCTIONS_HPP


#include "baci_config.hpp"

#include "baci_geometry_pair_element.hpp"
#include "baci_geometry_pair_utility_functions.hpp"
#include "baci_utils_fad.hpp"

BACI_NAMESPACE_OPEN

namespace GEOMETRYPAIR
{
  /**
   * \brief For some face elements the normals face inside the volume, this factor corrects this.
   *
   * @tparam element_type Type of the surface element.
   */
  template <typename element_type>
  struct SurfaceNormalFactor
  {
   public:
    /**
     * \brief Return the factor to scale the normals with.
     *
     * A factor of 1 means that the normals already face outside of the volume, -1 means that the
     * direction has to be changed.
     *
     * The function has to be inline to avoid a compiler error.
     *
     * @param element (in) Face element for which the normal is evaluated.
     * @return Normal factor.
     */
    static inline double GetFactor(const DRT::Element* element) { return 1.0; };
  };

  /**
   * \brief Template specialization for nurbs9 elements.
   */
  template <>
  inline double SurfaceNormalFactor<GEOMETRYPAIR::t_nurbs9>::GetFactor(const DRT::Element* element)
  {
    auto face_element = dynamic_cast<const DRT::FaceElement*>(element);
    if (face_element == nullptr)
      dserror(
          "SurfaceNormalFactor<GEOMETRYPAIR::t_nurbs9>::GetFactor needs a face element pointer.");

    switch (face_element->FaceMasterNumber())
    {
      case 0:
      case 3:
      case 5:
        return -1.0;
      case 1:
      case 2:
      case 4:
        return 1.0;
      default:
        dserror("Invalid face number in SurfaceNormalFactor<GEOMETRYPAIR::t_nurbs9>::GetFactor");
    }
    return 0.0;
  }

  /**
   * \brief Get the shape function matrix of the element at xi.
   *
   * Multiplying this shape function matrix with the element DOF vector results in the field
   * variable evaluated at xi.
   *
   * @param xi (in) Parameter coordinate on the element.
   * @param N (out) shape function matrix.
   * @param element (in) Pointer to the element, needed for beam elements.
   */
  template <typename element_type, typename T, typename scalar_type, unsigned int n_dim>
  inline void EvaluateShapeFunctionMatrix(const T& xi,
      CORE::LINALG::Matrix<n_dim, element_type::n_dof_, scalar_type>& N,
      const DRT::Element* element = nullptr)
  {
    // Matrix for shape function values.
    CORE::LINALG::Matrix<1, element_type::n_nodes_ * element_type::n_val_, scalar_type> N_flat(
        true);

    // Evaluate the shape function values.
    element_type::EvaluateShapeFunction(
        N_flat, xi, std::integral_constant<unsigned int, element_type::dim_>{}, element);

    // Fill up the full shape function matrix.
    N.Clear();
    for (unsigned int node = 0; node < element_type::n_nodes_; node++)
      for (unsigned int dim = 0; dim < n_dim; dim++)
        for (unsigned int val = 0; val < element_type::n_val_; val++)
          N(dim, n_dim * element_type::n_val_ * node + n_dim * val + dim) =
              N_flat(element_type::n_val_ * node + val);
  }

  /**
   * \brief Get the position in the element.
   * @param xi (in) Parameter coordinate on the element.
   * @param q (in) Degrees of freedom for the element.
   * @param r (out) Position on the element.
   * @param element (in) Pointer to the element, needed for beam elements.
   */
  template <typename element_type, typename T, typename V, typename scalar_type, unsigned int n_dim>
  inline void EvaluatePosition(const T& xi, const V& q,
      CORE::LINALG::Matrix<n_dim, 1, scalar_type>& r, const DRT::Element* element = nullptr)
  {
    // Matrix for shape function values.
    CORE::LINALG::Matrix<1, element_type::n_nodes_ * element_type::n_val_, scalar_type> N(true);

    // Evaluate the shape function values.
    element_type::EvaluateShapeFunction(
        N, xi, std::integral_constant<unsigned int, element_type::dim_>{}, element);

    // Calculate the position.
    r.Clear();
    for (unsigned int node = 0; node < element_type::n_nodes_; node++)
      for (unsigned int dim = 0; dim < n_dim; dim++)
        for (unsigned int val = 0; val < element_type::n_val_; val++)
          r(dim) += q(n_dim * element_type::n_val_ * node + n_dim * val + dim) *
                    N(element_type::n_val_ * node + val);
  }

  /**
   * \brief Get the derivative of the position w.r.t xi in the element.
   * @param xi (in) Parameter coordinate on the element.
   * @param q (in) Degrees of freedom for the element.
   * @param dr (out) Derivative of the position on the element.
   * @param element (in) Pointer to the element, needed for beam elements.
   */
  template <typename element_type, typename T, typename V, typename scalar_type>
  inline void EvaluatePositionDerivative1(const T& xi, const V& q,
      CORE::LINALG::Matrix<3, element_type::dim_, scalar_type>& dr,
      const DRT::Element* element = nullptr)
  {
    // Matrix for shape function values.
    CORE::LINALG::Matrix<element_type::dim_, element_type::n_nodes_ * element_type::n_val_,
        scalar_type>
        dN(true);

    // Evaluate the shape function values.
    element_type::EvaluateShapeFunctionDeriv1(
        dN, xi, std::integral_constant<unsigned int, element_type::dim_>{}, element);

    // Calculate the derivative of the position.
    dr.Clear();
    for (unsigned int dim = 0; dim < 3; dim++)
      for (unsigned int direction = 0; direction < element_type::dim_; direction++)
        for (unsigned int node = 0; node < element_type::n_nodes_; node++)
          for (unsigned int val = 0; val < element_type::n_val_; val++)
            dr(dim, direction) += q(3 * element_type::n_val_ * node + 3 * val + dim) *
                                  dN(direction, element_type::n_val_ * node + val);
  }

  /**
   * \brief Evaluate the normal at a position on the surface.
   *
   * @param xi (in) Parameter coordinates on the surface (the first two are in the surface
   * parameter coordiantes, the third one is in the normal direction).
   * @param q_surface (in) Degrees of freedom for the surface.
   * @param normal (out) Normal on the surface.
   * @param element (in) Pointer to the element object.
   * @param nodal_normals (in) Optional - Normals on the nodes.
   */
  template <typename surface, typename T, typename scalar_type_dof, typename scalar_type_result>
  void EvaluateSurfaceNormal(const T& xi,
      const CORE::LINALG::Matrix<surface::n_dof_, 1, scalar_type_dof>& q_surface,
      CORE::LINALG::Matrix<3, 1, scalar_type_result>& normal, const DRT::Element* element,
      const CORE::LINALG::Matrix<3 * surface::n_nodes_, 1, scalar_type_dof>* nodal_normals =
          nullptr)
  {
    // Check at compile time if a surface (2D) element is given.
    static_assert(surface::dim_ == 2, "EvaluateSurfaceNormal can only be called for 2D elements!");

    if (nodal_normals == nullptr)
    {
      // Calculate the normal as the geometrical normal on the element.
      CORE::LINALG::Matrix<3, 2, scalar_type_result> dr;
      CORE::LINALG::Matrix<3, 1, scalar_type_result> dr_0;
      CORE::LINALG::Matrix<3, 1, scalar_type_result> dr_1;
      GEOMETRYPAIR::EvaluatePositionDerivative1<surface>(xi, q_surface, dr, element);
      for (unsigned int i_dir = 0; i_dir < 3; i_dir++)
      {
        dr_0(i_dir) = dr(i_dir, 0);
        dr_1(i_dir) = dr(i_dir, 1);
      }
      normal.CrossProduct(dr_0, dr_1);
      normal.Scale(
          SurfaceNormalFactor<surface>::GetFactor(element) / CORE::FADUTILS::VectorNorm(normal));
    }
    else
    {
      // Calculate the normal as a interpolation of nodal normals.
      GEOMETRYPAIR::EvaluatePosition<surface>(xi, *nodal_normals, normal, element);
      normal.Scale(1.0 / CORE::FADUTILS::VectorNorm(normal));
    }
  }

  /**
   * \brief Evaluate a position on the surface.
   *
   * @param xi (in) Parameter coordinates on the surface (the first two are in the surface
   * parameter coordinates, the third one is in the normal direction).
   * @param q_surface (in) Degrees of freedom for the surface.
   * @param r (out) Position on the surface.
   * @param element (in) Pointer to the element object.
   * @param nodal_normals (in) Optional - Normals on the nodes.
   */
  template <typename surface, typename scalar_type_xi, typename scalar_type_dof,
      typename scalar_type_result>
  void EvaluateSurfacePosition(const CORE::LINALG::Matrix<3, 1, scalar_type_xi>& xi,
      const CORE::LINALG::Matrix<surface::n_dof_, 1, scalar_type_dof>& q_surface,
      CORE::LINALG::Matrix<3, 1, scalar_type_result>& r, const DRT::Element* element,
      const CORE::LINALG::Matrix<3 * surface::n_nodes_, 1, scalar_type_dof>* nodal_normals =
          nullptr)
  {
    // Check at compile time if a surface (2D) element is given.
    static_assert(surface::dim_ == 2, "EvaluateSurfaceNormal can only be called for 2D elements!");

    // Evaluate the normal.
    CORE::LINALG::Matrix<3, 1, scalar_type_result> normal;
    EvaluateSurfaceNormal<surface>(xi, q_surface, normal, element, nodal_normals);

    // Evaluate the position on the surface.
    GEOMETRYPAIR::EvaluatePosition<surface>(xi, q_surface, r, element);

    // Add the normal part to the position.
    normal.Scale(xi(2));
    r += normal;
  }

  /**
   * \brief Evaluate a triad on a plane curde.
   *
   * The curve has to lie in the x-y plane. The first basis vector of the triad is the tangent along
   * the curve, the second basis vector is the tangent rotated clockwise around the z axis by pi/2.
   * The third basis vector is the Cartesian e_z basis vector.
   *
   * @param xi (in) Parameter coordinate on the line
   * @param q_line (in) DOFs of the beam
   * @param triad (out) Beam cross section triad
   * @param element (in) Pointer to the element object.
   */
  template <typename line, typename scalar_type_xi, typename scalar_type_dof,
      typename scalar_type_result>
  void EvaluateTriadAtPlaneCurve(const scalar_type_xi xi,
      const CORE::LINALG::Matrix<line::n_dof_, 1, scalar_type_dof>& q_line,
      CORE::LINALG::Matrix<3, 3, scalar_type_result>& triad, const DRT::Element* element)
  {
    CORE::LINALG::Matrix<3, 1, scalar_type_result> tangent, cross_section_director_2,
        cross_section_director_3;
    EvaluatePositionDerivative1<line>(xi, q_line, tangent, element);

    if (std::abs(tangent(2)) > CONSTANTS::pos_tol)
      dserror(
          "EvaluateTriadAtPlaneCurve: The tangent vector can not have a component in z direction! "
          "The component is %f!",
          CORE::FADUTILS::CastToDouble(tangent(2)));

    // Create the director vectors in the cross section.
    // Director 2 is the one in the y-axis (reference configuration).
    // Director 3 is the one in the z-axis (reference configuration).
    tangent.Scale(1. / CORE::FADUTILS::VectorNorm(tangent));
    cross_section_director_2.Clear();
    cross_section_director_2(0) = -tangent(1);
    cross_section_director_2(1) = tangent(0);
    cross_section_director_3.Clear();
    cross_section_director_3(2) = 1.;

    // Set the triad.
    for (unsigned int dir = 0; dir < 3; dir++)
    {
      triad(dir, 0) = tangent(dir);
      triad(dir, 1) = cross_section_director_2(dir);
      triad(dir, 2) = cross_section_director_3(dir);
    }
  }

  /**
   * \brief Evaluate the Jacobi matrix for a volume element.
   *
   * @param xi (in) Parameter coordinate on the element.
   * @param X_volume (in) Reference positions for the element.
   * @param J (out) 3D Jacobi matrix.
   * @param element (in) Pointer to the element, needed for beam elements.
   */
  template <typename volume, typename scalar_type>
  void EvaluateJacobian(const CORE::LINALG::Matrix<3, 1, scalar_type>& xi,
      const CORE::LINALG::Matrix<volume::n_dof_, 1, double>& X_volume,
      CORE::LINALG::Matrix<3, 3, scalar_type>& J, const DRT::Element* element)
  {
    // Check at compile time if a volume (3D) element is given.
    static_assert(volume::dim_ == 3, "EvaluateJacobian can only be called for 3D elements!");

    // Get the derivatives of the reference position w.r.t the parameter coordinates. This is the
    // transposed Jacobi matrix.
    CORE::LINALG::Matrix<3, 3, scalar_type> dXdxi(true);
    EvaluatePositionDerivative1<volume>(xi, X_volume, dXdxi, element);
    J.Clear();
    J.UpdateT(dXdxi);
  }

  /**
   * \brief Evaluate the 3D deformation gradient F in a volume element.
   *
   * @param xi (in) Parameter coordinate on the element.
   * @param X_volume (in) Reference positions for the element.
   * @param q_volume (in) Degrees of freedom for the element.
   * @param F (out) 3D deformation gradient.
   * @param element (in) Pointer to the element, needed for beam elements.
   */
  template <typename volume, typename scalar_type_xi, typename scalar_type_dof,
      typename scalar_type_result>
  void EvaluateDeformationGradient(const CORE::LINALG::Matrix<3, 1, scalar_type_xi>& xi,
      const CORE::LINALG::Matrix<volume::n_dof_, 1, double>& X_volume,
      const CORE::LINALG::Matrix<volume::n_dof_, 1, scalar_type_dof>& q_volume,
      CORE::LINALG::Matrix<3, 3, scalar_type_result>& F, const DRT::Element* element)
  {
    // Check at compile time if a volume (3D) element is given.
    static_assert(
        volume::dim_ == 3, "EvaluateDeformationGradient can only be called for 3D elements!");

    // Get the inverse of the Jacobian.
    CORE::LINALG::Matrix<3, 3, scalar_type_xi> inv_J(true);
    EvaluateJacobian<volume>(xi, X_volume, inv_J, element);
    CORE::LINALG::Inverse(inv_J);

    // Get the derivatives of the shape functions w.r.t to the parameter coordinates.
    CORE::LINALG::Matrix<volume::dim_, volume::n_nodes_ * volume::n_val_, scalar_type_xi> dNdxi(
        true);
    volume::EvaluateShapeFunctionDeriv1(
        dNdxi, xi, std::integral_constant<unsigned int, volume::dim_>{}, element);

    // Transform to derivatives w.r.t physical coordinates.
    CORE::LINALG::Matrix<volume::dim_, volume::n_nodes_ * volume::n_val_, scalar_type_xi> dNdX(
        true);
    for (unsigned int i_row = 0; i_row < 3; i_row++)
      for (unsigned int i_col = 0; i_col < volume::n_nodes_ * volume::n_val_; i_col++)
        for (unsigned int i_sum = 0; i_sum < 3; i_sum++)
          dNdX(i_row, i_col) += inv_J(i_row, i_sum) * dNdxi(i_sum, i_col);

    // Calculate F.
    F.Clear();
    for (unsigned int i_row = 0; i_row < 3; i_row++)
      for (unsigned int i_col = 0; i_col < 3; i_col++)
        for (unsigned int i_sum = 0; i_sum < volume::n_nodes_ * volume::n_val_; i_sum++)
          F(i_col, i_row) += dNdX(i_row, i_sum) * q_volume(3 * i_sum + i_col);
  }

  /**
   * \brief Check if the parameter coordinate xi is in the valid range for a 1D element.
   * @param xi (in) Parameter coordinate on the line.
   * @return True if xi is in a valid range for the element.
   */
  template <typename T>
  bool ValidParameter1D(const T& xi)
  {
    const double xi_limit = 1.0 + CONSTANTS::projection_xi_eta_tol;
    if (fabs(xi) < xi_limit)
      return true;
    else
      return false;
  }

  /**
   * \brief Check if the parameter coordinate xi is in the valid range for a 2D element.
   * @param xi (in) Parameter coordinate on the line.
   * @return True if xi is in a valid range for the element.
   */
  template <typename element_type, typename T>
  bool ValidParameter2D(const T& xi)
  {
    const double xi_limit = 1.0 + CONSTANTS::projection_xi_eta_tol;
    if (element_type::geometry_type_ == DiscretizationTypeGeometry::quad)
    {
      if (fabs(xi(0)) < xi_limit && fabs(xi(1)) < xi_limit) return true;
    }
    else if (element_type::geometry_type_ == DiscretizationTypeGeometry::triangle)
    {
      if (xi(0) > -CONSTANTS::projection_xi_eta_tol && xi(1) > -CONSTANTS::projection_xi_eta_tol &&
          xi(0) + xi(1) < 1.0 + CONSTANTS::projection_xi_eta_tol)
        return true;
    }
    else
    {
      dserror("Wrong DiscretizationTypeGeometry given!");
    }

    // Default value.
    return false;
  }

  /**
   * \brief Check if the parameter coordinate xi is in the valid range for a 3D element.
   * @param xi (in) Parameter coordinate on the line.
   * @return True if xi is in a valid range for the element.
   */
  template <typename element_type, typename T>
  bool ValidParameter3D(const T& xi)
  {
    const double xi_limit = 1.0 + CONSTANTS::projection_xi_eta_tol;
    if (element_type::geometry_type_ == DiscretizationTypeGeometry::hexahedron)
    {
      if (fabs(xi(0)) < xi_limit && fabs(xi(1)) < xi_limit && fabs(xi(2)) < xi_limit) return true;
    }
    else if (element_type::geometry_type_ == DiscretizationTypeGeometry::tetraeder)
    {
      if (xi(0) > -CONSTANTS::projection_xi_eta_tol && xi(1) > -CONSTANTS::projection_xi_eta_tol &&
          xi(2) > -CONSTANTS::projection_xi_eta_tol &&
          xi(0) + xi(1) + xi(2) < 1.0 + CONSTANTS::projection_xi_eta_tol)
        return true;
    }
    else
    {
      dserror("Wrong DiscretizationTypeGeometry given!");
    }

    // Default value.
    return false;
  }

  /**
   * \brief This struct sets parameter coordinates start values for local Newton iterations on an
   * element.
   *
   * @tparam geometry_type The geometry type of the element, this type is typically known at
   * compile time.
   */
  template <DiscretizationTypeGeometry geometry_type>
  struct StartValues
  {
   public:
    /**
     * \brief Set the start values for a local Newton iterations on the element.
     *
     * This method will be specialized for each geometry type. By doing so the values can be set at
     * compile time.
     *
     * @param xi (out) Parameter coordinate on element.
     */
    template <typename T>
    static void Set(T& xi)
    {
      std::string error_string =
          "This is the default implementation of StartValues::Set and should never be called, "
          "since we only want to call the templated versions. You are calling it with the "
          "DiscretizationTypeGeometry ";
      error_string += DiscretizationTypeGeometryToString(geometry_type);
      dserror(error_string.c_str());
    }
  };

  /**
   * \brief Template specialization for line elements.
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::line>::Set(T& xi)
  {
    xi = 0.0;
  }

  /**
   * \brief Template specialization for triangle surface elements.
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::triangle>::Set(T& xi)
  {
    // We do not use xi.PutScalar(0.25) here, since this might be a surface element which has a
    // normal direction and we do not want to set an initial value in the normal direction.
    xi.PutScalar(0.0);
    for (int i_dim = 0; i_dim < 2; i_dim++) xi(i_dim) = 0.25;
  }

  /**
   * \brief Template specialization for quad surface elements.
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::quad>::Set(T& xi)
  {
    xi.PutScalar(0.0);
  }

  /**
   * \brief Template specialization for tetraeder volume elements.
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::tetraeder>::Set(T& xi)
  {
    xi.PutScalar(0.25);
  }

  /**
   * \brief Template specialization for hexahedron volume elements.
   */
  template <>
  template <typename T>
  void StartValues<DiscretizationTypeGeometry::hexahedron>::Set(T& xi)
  {
    xi.PutScalar(0.0);
  }

}  // namespace GEOMETRYPAIR


BACI_NAMESPACE_CLOSE

#endif