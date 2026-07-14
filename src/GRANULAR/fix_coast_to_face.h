/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(coast/to/face,FixCoastToFace);
// clang-format on
#else

#ifndef LMP_FIX_COAST_TO_FACE_H
#define LMP_FIX_COAST_TO_FACE_H

#include "fix.h"

namespace LAMMPS_NS {

// Holds atoms in the fix group at constant velocity (immune to gravity,
// pair forces, etc.) until they cross a defined plane, then releases
// them to normal physics. Meant to reproduce the "coasting" integrator
// that LIGGGHTS' fix insert/stream uses internally to hold particles at
// a fixed velocity between their insertion point and the insertion face.

class FixCoastToFace : public Fix {
 public:
  FixCoastToFace(class LAMMPS *, int, char **);
  ~FixCoastToFace() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  double compute_vector(int) override;

  // per-atom data (the coast_active flag) migration callbacks: without
  // these, atom->sort()/comm->exchange() silently leave coast_active
  // behind while x/v/etc move, corrupting which atom is "coasting"
  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

 protected:
  double normal[3], point[3];
  int coast_index;
  bigint ncoasting, ncoasting_all;
};

}    // namespace LAMMPS_NS

#endif
#endif
