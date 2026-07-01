# -A-Bayesian-approach-for-accurate-alignment-and-smoothing
R implementation of the Bayesian functional alignment model for amplitude and phase variability described in ADDRESSING PHASE DISCREPANCIES IN FUNCTIONAL DATA: A BAYESIAN APPROACH FOR ACCURATE ALIGNMENT AND SMOOTHING (Gardella et al. 2026)

# JARA_warping_GROUP

Bayesian curve registration (warping) and smoothing for functional data.

## Overview

This code implements 4 models in total for smoothing and alignment of
functional data. The models are those of Telesca and Inoue (2012), Gardella
et al. (2026), and 2 additional models given by combining the smoothing and
warping components of the two. The parameters that select the model are
`WARP` and `SMOOT`.

## Usage

The function is called by passing **a single list of parameters**:

```r
result <- JARA_warping_GROUP(list(y = y, nburn = 500, niter = 2000,
                                   WARP = "gaussian", SMOOT = "telesca", ...))
```

## 1. Data & MCMC parameters (always required)

| Parameter | Default | Description |
|---|---|---|
| `y` | *required* | List of matrices (one per group), columns = subjects, rows = time points (NA-padded) |
| `n_groups` | inferred from `y` | Number of groups (consistency check) |
| `n_per_group` | inferred from `y` | Subjects per group (consistency check) |
| `nburn` | 100 | Burn-in iterations |
| `niter` | 500 | Post burn-in iterations saved |
| `n_knots_beta` | 10 | Knots of mean curve spline |
| `n_knots_phi` | 6 | Knots of warping spline |
| `degree_beta` | 3 | Degree of mean curve spline |
| `degree_phi` | 3 | Degree of warping spline |
| `intercept_beta` | FALSE | Intercept in beta spline basis |
| `intercept_phi` | FALSE | Intercept in phi spline basis |
| `boundary_beta` | `[0,1]` | Boundary knots for beta spline |
| `boundary_phi` | `[0,1]` | Boundary knots for phi spline |
| `a_eps`, `b_eps` | 3, 0.1 | IG prior on residual sigma² |
| `a_lambda`, `b_lambda` | 3, 0.1 | IG prior on beta penalization |
| `beta0` | 0 | Prior mean for beta |
| `WARP` | `"gamma-adaptation"` | Warping type: `"gamma-adaptation"` or `"gaussian"` |
| `SMOOT` | `"gardella"` | Smoothing model: `"gardella"` or `"telesca"` |

## 2. Parameters when `SMOOT = "gardella"`

| Parameter | Default | Description |
|---|---|---|
| `n_knots_gamma` | 3 | Knots of individual component spline |
| `degree_gamma` | 3 | Degree of individual component spline |
| `intercept_gamma` | FALSE | Intercept in gamma spline basis |
| `boundary_gamma` | `[0,1]` | Boundary knots for gamma spline |
| `a_gamma`, `b_gamma` | 3, 0.1 | IG prior on gamma sigma² |
| `gamma0` | 0 | Prior mean for gamma |

## 3. Parameters when `SMOOT = "telesca"`

| Parameter | Default | Description |
|---|---|---|
| `m_a0`, `sigma_a0` | 5, 10 | Prior on population mean of individual scale a |
| `m_c0`, `sigma_c0` | 0, 10 | Prior on population mean of individual shift c |
| `a_a`, `b_a` | 25, 100 | IG prior on between-subject variance of a |
| `a_c`, `b_c` | 250, 600 | IG prior on between-subject variance of c |

> **Note:** with `SMOOT = "telesca"` and multiple groups, the model runs
> **separately per group** (no shared beta across groups). Output is a list
> of lists (`group_1`, `group_2`, ...).

## 4. Parameters when `WARP = "gamma-adaptation"`

| Parameter | Default | Description |
|---|---|---|
| `a_f`, `b_f` | —, 1.0 | Gamma prior on warping increments |
| `zeta_0`, `zeta_0_coeff` | auto, 0.001 | Initial covariance for adaptive MH proposal |
| `sd0`, `sd_min`, `sd_max` | auto, 0.01, 1.0 | Initial scale/bounds of adaptive proposal |
| `target_alpha` | 0.234 | Target acceptance rate |
| `LAMB` | 0.6 | Adaptation decay rate |
| `EPS` | 1e-6 | Adaptive covariance regularization |
| `n0_zeta` | 10 | Iteration when covariance adaptation starts |
| `INIT_WARP`, `phi_init` | FALSE, — | Warping initialization from a provided vector |
| `SOMMA_csi` | 200 | Scale of initial increment sum |

## 5. Parameters when `WARP = "gaussian"`

| Parameter | Default | Description |
|---|---|---|
| `coeff_var_phi` | — | MH proposal step size for each phi component |
| `a_phi`, `b_phi` | 4, 4 | IG prior on warping sigma² |
| `phi0` | auto | Prior mean for phi |
| `INIT_WARP`, `phi_init` | FALSE, — | Warping initialization from a provided vector |

## Output

The function returns a list containing:

- **`post`**: posterior estimates of all parameters
- **`full`**: complete MCMC chains
- **`acceptance`**: MH acceptance rates
- **`h`**: estimated warping functions
- **`Bm`**: final B-spline basis matrices
- **`y_info`**: information extracted from the data
- **`curves`**: `y` (original data), `y_smoot` (fitted), `y_star` (data on
  the warped scale), `y_star_smoot` (fitted on the warped scale)

## References

- Telesca, D. and Inoue, L. Y. T. (2012).
- Gardella et al. (2026).
