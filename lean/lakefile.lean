import Lake
open Lake DSL

package «gspacegroup» where
  packagesDir := "../../.lake/packages"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.29.0"

@[default_target]
lean_lib «GSpacegroup» where
  roots := #[`GSpacegroup.Plancherel, `GSpacegroup.FrobeniusSchur,
             `GSpacegroup.SpectralWeights, `GSpacegroup.Chi3Reduction]
