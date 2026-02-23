"""
Unit tests for equation of state implementations.

Tests thermodynamic consistency and accuracy of EOS classes.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lagrangian_solver.equations.eos import IdealGasEOS, ThermodynamicState


class TestIdealGasEOS:
    """Tests for IdealGasEOS class."""

    @pytest.fixture
    def eos(self):
        """Create standard air EOS (gamma=1.4, R=287.05)."""
        return IdealGasEOS(gamma=1.4, R=287.05)

    def test_initialization(self, eos):
        """Test EOS initialization with correct properties."""
        assert eos.gamma == 1.4
        assert eos.R == 287.05
        assert abs(eos.cv - 287.05 / 0.4) < 1e-10
        assert abs(eos.cp - 1.4 * 287.05 / 0.4) < 1e-10

    def test_invalid_gamma(self):
        """Test that invalid gamma raises error."""
        with pytest.raises(ValueError):
            IdealGasEOS(gamma=1.0)  # gamma must be > 1
        with pytest.raises(ValueError):
            IdealGasEOS(gamma=0.5)

    def test_invalid_R(self):
        """Test that invalid R raises error."""
        with pytest.raises(ValueError):
            IdealGasEOS(R=-1.0)
        with pytest.raises(ValueError):
            IdealGasEOS(R=0.0)

    def test_pressure_from_rho_e(self, eos):
        """Test pressure calculation: p = (gamma-1) * rho * e."""
        rho = np.array([1.0, 2.0, 0.5])
        e = np.array([250000.0, 300000.0, 200000.0])

        p = eos.pressure(rho, e)

        expected = (eos.gamma - 1) * rho * e
        np.testing.assert_allclose(p, expected, rtol=1e-10)

    def test_internal_energy_from_rho_p(self, eos):
        """Test internal energy calculation: e = p / (rho * (gamma-1))."""
        rho = np.array([1.0, 2.0, 0.5])
        p = np.array([100000.0, 200000.0, 50000.0])

        e = eos.internal_energy(rho, p)

        expected = p / (rho * (eos.gamma - 1))
        np.testing.assert_allclose(e, expected, rtol=1e-10)

    def test_pressure_energy_consistency(self, eos):
        """Test that pressure and energy are consistent (round-trip)."""
        rho = np.array([1.225, 0.5, 2.0])
        p_original = np.array([101325.0, 50000.0, 200000.0])

        e = eos.internal_energy(rho, p_original)
        p_recovered = eos.pressure(rho, e)

        np.testing.assert_allclose(p_recovered, p_original, rtol=1e-10)

    def test_temperature(self, eos):
        """Test temperature calculation: T = e / cv."""
        e = np.array([200000.0, 300000.0, 250000.0])
        T = eos.temperature(np.array([1.0, 1.0, 1.0]), e)

        expected = e / eos.cv
        np.testing.assert_allclose(T, expected, rtol=1e-10)

    def test_sound_speed(self, eos):
        """Test sound speed calculation: c = sqrt(gamma * p / rho)."""
        rho = np.array([1.225, 0.5, 2.0])
        p = np.array([101325.0, 50000.0, 200000.0])

        c = eos.sound_speed(rho, p)

        expected = np.sqrt(eos.gamma * p / rho)
        np.testing.assert_allclose(c, expected, rtol=1e-10)

    def test_sound_speed_standard_conditions(self, eos):
        """Test sound speed at standard sea level conditions."""
        # Standard atmosphere: T = 288.15 K, p = 101325 Pa, rho = 1.225 kg/m³
        rho = np.array([1.225])
        p = np.array([101325.0])

        c = eos.sound_speed(rho, p)

        # Expected sound speed ~ 340 m/s
        assert 330 < c[0] < 350, f"Sound speed {c[0]} not in expected range"

    def test_get_gamma(self, eos):
        """Test that gamma is constant for ideal gas."""
        rho = np.array([1.0, 2.0, 0.5])
        p = np.array([100000.0, 200000.0, 50000.0])

        gamma = eos.get_gamma(rho, p)

        np.testing.assert_allclose(gamma, np.full(3, 1.4), rtol=1e-10)

    def test_complete_state(self, eos):
        """Test complete state calculation from rho and p."""
        rho = 1.225
        p = 101325.0

        state = eos.complete_state(rho, p=p)

        assert isinstance(state, ThermodynamicState)
        assert state.rho == rho
        assert abs(state.p - p) < 1e-6
        assert state.T > 0
        assert state.e > 0
        assert state.c > 0
        assert abs(state.gamma - 1.4) < 1e-10

    def test_complete_state_from_e(self, eos):
        """Test complete state calculation from rho and e."""
        rho = 1.225
        e = 200000.0

        state = eos.complete_state(rho, e=e)

        assert isinstance(state, ThermodynamicState)
        assert state.rho == rho
        assert abs(state.e - e) < 1e-6
        assert state.p > 0
        assert state.T > 0

    def test_complete_state_invalid_args(self, eos):
        """Test that complete_state raises error with invalid arguments."""
        with pytest.raises(ValueError):
            eos.complete_state(1.0)  # Neither p nor e

        with pytest.raises(ValueError):
            eos.complete_state(1.0, p=100000.0, e=200000.0)  # Both p and e

    def test_temperature_from_rho_p(self, eos):
        """Test temperature from density and pressure."""
        rho = np.array([1.225])
        p = np.array([101325.0])

        T = eos.temperature_from_rho_p(rho, p)

        # T = p / (rho * R)
        expected = p / (rho * eos.R)
        np.testing.assert_allclose(T, expected, rtol=1e-10)

        # Should be near standard temperature (288 K)
        assert 280 < T[0] < 300

    def test_density_from_p_T(self, eos):
        """Test density from pressure and temperature."""
        p = np.array([101325.0])
        T = np.array([288.15])

        rho = eos.density_from_p_T(p, T)

        # rho = p / (R * T)
        expected = p / (eos.R * T)
        np.testing.assert_allclose(rho, expected, rtol=1e-10)

        # Should be near standard density (1.225 kg/m³)
        assert 1.2 < rho[0] < 1.3

    def test_ideal_gas_law_consistency(self, eos):
        """Test that p = rho * R * T holds."""
        rho = np.array([1.225, 0.5, 2.0])
        T = np.array([288.15, 300.0, 250.0])

        # Get pressure from ideal gas law
        p = rho * eos.R * T

        # Get temperature from EOS
        e = eos.cv * T  # e = cv * T
        T_from_eos = eos.temperature(rho, e)

        np.testing.assert_allclose(T_from_eos, T, rtol=1e-10)

    def test_scalar_inputs(self, eos):
        """Test that scalar inputs work correctly."""
        rho = 1.225
        p = 101325.0

        e = eos.internal_energy(rho, p)
        T = eos.temperature(rho, e)
        c = eos.sound_speed(rho, p)

        # Should return arrays even for scalar input
        assert isinstance(e, np.ndarray)
        assert isinstance(T, np.ndarray)
        assert isinstance(c, np.ndarray)


class TestDifferentGammas:
    """Test EOS with different gamma values."""

    @pytest.mark.parametrize(
        "gamma,description",
        [
            (1.4, "diatomic gas (air)"),
            (1.667, "monatomic gas (argon)"),
            (1.3, "polyatomic gas"),
            (1.1, "near incompressible"),
        ],
    )
    def test_different_gammas(self, gamma, description):
        """Test EOS with various gamma values."""
        eos = IdealGasEOS(gamma=gamma)

        rho = np.array([1.0])
        p = np.array([100000.0])

        e = eos.internal_energy(rho, p)
        p_back = eos.pressure(rho, e)
        c = eos.sound_speed(rho, p)

        np.testing.assert_allclose(p_back, p, rtol=1e-10)
        assert c[0] > 0, f"Sound speed must be positive for {description}"
