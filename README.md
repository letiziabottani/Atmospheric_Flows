# Atmospheric Flow Solver over Orography

This branch provides a solver for the simulation of three-dimensional atmospheric flows over idealized orography, specifically based on the **versiera of Agnesi** profile.
The solver is highly configurable via a parameter file, allowing flexible setup of physical, geometrical, and numerical aspects of the simulation.

---

## Configuration

### Domain and Orography Parameters

The computational domain and terrain geometry can be defined through

- Domain bounds:
  - `x_min`, `x_max`
  - `y_min`, `y_max`
  - `z_min`, `z_max`
- Orography parameters:
  - Mountain height: `h`
  - Mountain center coordinates: `xc`, `yc`
  - Mountain semi-width: `ac`
---

### Initial and Background Conditions

The initial atmospheric state is defined by

- Buoyancy frequency: `N`
- Background flow and thermodynamic conditions:
  - Background velocity: `u_bar`
  - Pressure at ground level (`z = 0`): `p_bar`
  - Temperature at ground level (`z = 0`): `T_bar`
---

### Numerical Parameters

#### Mesh

- Initial mesh resolution
  - `n_elements_x`
  - `n_elements_y`
  - `n_elements_z`

#### Rayleigh Damping

Rayleigh damping layers can be configured independently along each spatial direction

- Vertical damping:
  - `z_start`, `lambda_z`
- Horizontal damping (x-direction):
  - `x_start_left`, `lambda_x_left`
  - `x_start_right`, `lambda_x_right`
- Horizontal damping (y-direction):
  - `y_start_left`, `lambda_y_left`
  - `y_start_right`, `lambda_y_right`

#### Solver Tolerances

- Fixed-point iteration convergence:
  - `||dp|| < atol_iterative + rtol_iterative * ||p||`
- Linear solver convergence:
  - `||Ax - b|| < atol_fixed_point + rtol_fixed_point * ||b||`
---

## Non-Dimensional Formulation

The solver supports a fully non-dimensional formulation through the specification of reference quantities

- Reference length: `L_ref`
- Reference velocity: `u_ref`
- Reference pressure: `p_ref`
- Reference density: `rho_ref`

> **Note:**
> - The **Mach** and **Froude** numbers must still be provided explicitly and are enforced during the simulation.
> - These reference quantities do not necessarily correspond to physical scales; they are used purely for numerical normalization.
---

## Running the Solver

After compilation, the executable can be launched as

```bash
./Atmospheric_Flow
```

If no input file is specified, the solver attempts to read

```bash
parameter-file.prm
```

A custom parameter file can be specified via

```bash
./Atmospheric_Flow -p <parameter_file>
```

A minimal help message can be displayed with

```bash
./Atmospheric_Flow -h
```

## Additional Modules

The subfolder `LAMINAR_INVERSE_MASS_MATRICES_PERTURBATION` contains an alternative solver implementation based on a perturbation formulation 
(as described, e.g., in https://www.sciencedirect.com/science/article/abs/pii/S002199911730030X).
