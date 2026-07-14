// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_coast_to_face.h"

#include "atom.h"
#include "error.h"
#include "math_extra.h"
#include "memory.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixCoastToFace::FixCoastToFace(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 9) error->all(FLERR,"Illegal fix coast/to/face command");

  normal[0] = utils::numeric(FLERR,arg[3],false,lmp);
  normal[1] = utils::numeric(FLERR,arg[4],false,lmp);
  normal[2] = utils::numeric(FLERR,arg[5],false,lmp);
  point[0] = utils::numeric(FLERR,arg[6],false,lmp);
  point[1] = utils::numeric(FLERR,arg[7],false,lmp);
  point[2] = utils::numeric(FLERR,arg[8],false,lmp);

  double nlen = MathExtra::len3(normal);
  if (nlen < 1.0e-12) error->all(FLERR,"Fix coast/to/face normal vector has zero length");
  normal[0] /= nlen;
  normal[1] /= nlen;
  normal[2] /= nlen;

  vector_flag = 1;
  size_vector = 1;
  global_freq = 1;
  extvector = 0;

  // per-atom flag: 1 while an atom is still coasting (held at constant
  // velocity), 0 once it has crossed the face and been released to
  // normal physics. New atoms default to 0 (not coasting) unless
  // explicitly marked in setup().

  int flag, cols;
  coast_index = atom->find_custom("coast_active", flag, cols);
  if (coast_index < 0) coast_index = atom->add_custom("coast_active", 0, 0, 0);

  // this fix owns the coast_active array's lifetime across atom
  // sort/exchange/migration, since atom->add_custom() alone does not
  // register it for that
  atom->add_callback(Atom::GROW);

  ncoasting = ncoasting_all = 0;
}

/* ---------------------------------------------------------------------- */

FixCoastToFace::~FixCoastToFace()
{
  atom->delete_callback(id, Atom::GROW);

  int flag, cols;
  int index = atom->find_custom("coast_active", flag, cols);
  if (index >= 0) atom->remove_custom(index, 0, 0);
}

/* ---------------------------------------------------------------------- */

int FixCoastToFace::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCoastToFace::init()
{
  if (!atom->torque_flag)
    error->all(FLERR,"Fix coast/to/face requires atom attribute torque");
}

/* ----------------------------------------------------------------------
   mark every atom currently in the fix group as coasting; this is the
   standalone-test entry point. When wired up to an insertion fix, that
   fix would instead set coast_active = 1 for each newly-inserted atom
   at the moment of insertion.
------------------------------------------------------------------------- */

void FixCoastToFace::setup(int vflag)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *coast = atom->ivector[coast_index];

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) coast[i] = 1;

  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixCoastToFace::post_force(int /*vflag*/)
{
  double **x = atom->x;
  double **f = atom->f;
  double **torque = atom->torque;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *coast = atom->ivector[coast_index];

  ncoasting = 0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (!coast[i]) continue;

    double d = normal[0]*(x[i][0]-point[0]) +
               normal[1]*(x[i][1]-point[1]) +
               normal[2]*(x[i][2]-point[2]);

    if (d >= 0.0) {
      // crossed the face this step: release to normal physics,
      // forces computed this step are left untouched
      coast[i] = 0;
      continue;
    }

    // still coasting: zero the force/torque so velocity Verlet leaves
    // velocity (and angular velocity) exactly unchanged this step
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    f[i][2] = 0.0;
    torque[i][0] = 0.0;
    torque[i][1] = 0.0;
    torque[i][2] = 0.0;
    ncoasting++;
  }
}

/* ---------------------------------------------------------------------- */

void FixCoastToFace::post_force_respa(int vflag, int /*ilevel*/, int /*iloop*/)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   number of atoms still coasting (held at constant velocity)
------------------------------------------------------------------------- */

double FixCoastToFace::compute_vector(int /*n*/)
{
  MPI_Allreduce(&ncoasting,&ncoasting_all,1,MPI_LMP_BIGINT,MPI_SUM,world);
  return static_cast<double>(ncoasting_all);
}

/* ---------------------------------------------------------------------- */

double FixCoastToFace::memory_usage()
{
  return static_cast<double>(atom->nmax) * sizeof(int);
}

/* ---------------------------------------------------------------------- */

void FixCoastToFace::grow_arrays(int nmax)
{
  memory->grow(atom->ivector[coast_index], nmax, "fix_coast_to_face:coast_active");
}

/* ---------------------------------------------------------------------- */

void FixCoastToFace::copy_arrays(int i, int j, int /*delflag*/)
{
  atom->ivector[coast_index][j] = atom->ivector[coast_index][i];
}

/* ---------------------------------------------------------------------- */

int FixCoastToFace::pack_exchange(int i, double *buf)
{
  buf[0] = ubuf(atom->ivector[coast_index][i]).d;
  return 1;
}

/* ---------------------------------------------------------------------- */

int FixCoastToFace::unpack_exchange(int nlocal, double *buf)
{
  atom->ivector[coast_index][nlocal] = (int) ubuf(buf[0]).i;
  return 1;
}
