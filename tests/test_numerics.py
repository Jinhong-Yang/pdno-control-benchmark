import numpy as np

from pdno.physics.burgers import burgers_rhs, energy_rate, evolve_burgers, periodic_grid, spectral_derivative
from pdno.physics.heat import heat_cn_step, heat_eigenmode
from pdno.physics.finite_volume import burgers_fv_rhs


def test_periodic_constant_state_is_stationary():
    q = np.full(128, 0.4)
    assert np.max(np.abs(burgers_rhs(q, nu=0.02))) < 1e-12


def test_zero_burgers_state_with_zero_force_is_stationary():
    q = np.zeros(128)
    assert np.max(np.abs(evolve_burgers(q, 0.02, 2.5e-4, 10))) == 0.0


def test_split_solver_is_stable_at_truth_grid_and_maximum_viscosity():
    x = periodic_grid(256)
    q = 0.5 * np.sin(2 * np.pi * x) + 0.1 * np.cos(4 * np.pi * x)
    trajectory = evolve_burgers(q, nu=0.045, dt=2.5e-4, steps=512)
    assert np.isfinite(trajectory).all()
    assert np.max(np.abs(trajectory)) < 1.0
    assert abs(np.mean(trajectory[-1]) - np.mean(q)) < 2e-12


def test_burgers_mean_balance_and_energy_dissipation():
    x = periodic_grid(128)
    q = 0.3 * np.sin(2 * np.pi * x) + 0.1 * np.cos(4 * np.pi * x)
    f = np.sin(2 * np.pi * x)  # zero mean
    rhs = burgers_rhs(q, 0.02, f)
    assert abs(np.mean(rhs) - np.mean(f)) < 1e-12
    assert energy_rate(q, 0.02) < 0.0
    assert np.isfinite(np.mean(q * rhs))


def test_spectral_derivative_matches_periodic_manufactured_mode():
    x = periodic_grid(256)
    q = np.sin(6 * np.pi * x)
    expected = 6 * np.pi * np.cos(6 * np.pi * x)
    assert np.max(np.abs(spectral_derivative(q) - expected)) < 2e-11


def test_manufactured_smooth_forced_burgers_residual():
    x = periodic_grid(256)
    nu, speed, t = 0.02, 0.1, 0.17
    phase = 2 * np.pi * (x - speed * t)
    q = 0.2 + 0.15 * np.sin(phase) + 0.03 * np.cos(4 * np.pi * (x - speed * t))
    qt = -speed * (0.15 * 2 * np.pi * np.cos(phase) - 0.03 * 4 * np.pi * np.sin(4 * np.pi * (x - speed * t)))
    qx = 0.15 * 2 * np.pi * np.cos(phase) - 0.03 * 4 * np.pi * np.sin(4 * np.pi * (x - speed * t))
    qxx = -0.15 * (2 * np.pi) ** 2 * np.sin(phase) - 0.03 * (4 * np.pi) ** 2 * np.cos(4 * np.pi * (x - speed * t))
    forcing = qt + q * qx - nu * qxx
    residual = burgers_rhs(q, nu, forcing)
    assert np.max(np.abs(residual - qt)) < 2e-10


def test_independent_fv_and_spectral_initial_residual_agree_under_refinement():
    errors = []
    for n in (64, 128, 256):
        x = periodic_grid(n)
        q = 0.15 * np.sin(2 * np.pi * x) + 0.05 * np.cos(4 * np.pi * x)
        fv = burgers_fv_rhs(q, 1.0 / n, 0.02)
        spectral = burgers_rhs(q, 0.02, dealias=False)
        errors.append(np.sqrt(np.mean((fv - spectral) ** 2)))
    assert errors[2] < errors[1] < errors[0]


def test_heat_eigenmode_matches_cn_refinement_and_boundaries():
    errors = []
    t, kappa, decay = 0.05, 0.01, 0.3
    for n in (64, 128):
        dx = 1.0 / (n + 1)
        x = dx * np.arange(1, n + 1)
        dt, steps = 2.5e-4, round(t / 2.5e-4)
        u = np.sin(np.pi * x)
        for _ in range(steps):
            u = heat_cn_step(u, dx, dt, kappa, decay)
        exact = heat_eigenmode(x, steps * dt, kappa, decay)
        errors.append(np.sqrt(np.mean((u - exact) ** 2)))
        assert np.isfinite(u).all()
    assert errors[1] < errors[0] / 2.5


def test_heat_zero_state_is_fixed_point():
    assert np.max(np.abs(heat_cn_step(np.zeros(31), 1 / 32, 1e-3, 0.01, 0.3))) == 0.0
