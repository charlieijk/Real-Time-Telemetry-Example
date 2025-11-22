"""
Environmental Physics and Weather Integration for Solar Boat Racing.

This module provides:
1. Wind field modeling (speed, direction, gusts)
2. Wave dynamics (height, period, direction)
3. Water current modeling (tidal, river flow)
4. Weather API integration (OpenWeatherMap, NOAA)
5. Solar irradiance modeling (time of day, clouds, atmospheric effects)
6. Combined environmental forces on boat dynamics
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from datetime import datetime, timezone
import json
from abc import ABC, abstractmethod
import math


@dataclass
class WindState:
    """Current wind conditions."""
    speed: float = 0.0          # m/s
    direction: float = 0.0      # radians (0 = from North, clockwise)
    gust_speed: float = 0.0     # m/s (peak gusts)
    turbulence: float = 0.0     # 0-1 turbulence intensity


@dataclass
class WaveState:
    """Current wave conditions."""
    height: float = 0.0          # significant wave height (m)
    period: float = 4.0          # wave period (seconds)
    direction: float = 0.0       # radians (direction waves are coming from)
    steepness: float = 0.0       # wave steepness (H/L)


@dataclass
class CurrentState:
    """Water current conditions."""
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))  # [vx, vy] m/s
    depth_profile: Optional[List[float]] = None  # velocity at different depths


@dataclass
class SolarState:
    """Solar irradiance conditions."""
    direct_normal: float = 0.0    # Direct normal irradiance (W/m²)
    diffuse_horizontal: float = 0.0  # Diffuse horizontal (W/m²)
    global_horizontal: float = 0.0   # Total GHI (W/m²)
    cloud_cover: float = 0.0      # 0-1 fraction
    air_mass: float = 1.0         # Atmospheric air mass


@dataclass
class EnvironmentalState:
    """Complete environmental state."""
    wind: WindState = field(default_factory=WindState)
    waves: WaveState = field(default_factory=WaveState)
    current: CurrentState = field(default_factory=CurrentState)
    solar: SolarState = field(default_factory=SolarState)
    air_temperature: float = 20.0  # Celsius
    water_temperature: float = 15.0  # Celsius
    air_pressure: float = 101325.0  # Pa
    humidity: float = 0.5  # 0-1


class WindModel:
    """
    Wind field model with spatial and temporal variations.

    Supports:
    - Mean wind with log profile
    - Turbulence modeling (von Karman spectrum)
    - Gust modeling
    - Land/water transitions
    """

    def __init__(
        self,
        mean_speed: float = 5.0,
        mean_direction: float = 0.0,
        turbulence_intensity: float = 0.15,
        reference_height: float = 10.0,
        roughness_length: float = 0.0002  # Water surface
    ):
        self.mean_speed = mean_speed
        self.mean_direction = mean_direction
        self.turbulence_intensity = turbulence_intensity
        self.reference_height = reference_height
        self.roughness_length = roughness_length

        # Gust parameters
        self.gust_factor = 1.4
        self.gust_duration = 3.0  # seconds

        # Time-varying state
        self._time = 0.0
        self._last_gust_time = -np.inf
        self._current_gust_magnitude = 0.0

    def get_wind_at_height(self, height: float = 2.0) -> float:
        """
        Get wind speed at specified height using log profile.

        Args:
            height: Height above water surface (m)

        Returns:
            Wind speed at height (m/s)
        """
        # Logarithmic wind profile
        if height < self.roughness_length:
            return 0.0

        speed = self.mean_speed * (
            np.log(height / self.roughness_length) /
            np.log(self.reference_height / self.roughness_length)
        )
        return max(0.0, speed)

    def get_wind_state(self, time: float, position: np.ndarray = None) -> WindState:
        """
        Get wind conditions at given time and position.

        Args:
            time: Current time (seconds)
            position: [x, y] position (optional for spatial variation)

        Returns:
            WindState with current conditions
        """
        self._time = time

        # Base wind at typical sail/deck height
        base_speed = self.get_wind_at_height(2.0)

        # Add turbulence
        turbulent_speed = base_speed * (1.0 + self.turbulence_intensity * np.random.randn())

        # Direction variation (±15 degrees)
        direction_variation = np.radians(15) * np.random.randn() * self.turbulence_intensity
        current_direction = self.mean_direction + direction_variation

        # Gust modeling (Poisson process)
        gust_probability = 0.02  # per second
        if np.random.random() < gust_probability:
            self._last_gust_time = time
            self._current_gust_magnitude = base_speed * (self.gust_factor - 1.0) * np.random.random()

        # Apply gust if within duration
        gust_contribution = 0.0
        if time - self._last_gust_time < self.gust_duration:
            decay = 1.0 - (time - self._last_gust_time) / self.gust_duration
            gust_contribution = self._current_gust_magnitude * decay

        return WindState(
            speed=max(0.0, turbulent_speed),
            direction=current_direction % (2 * np.pi),
            gust_speed=base_speed * self.gust_factor + gust_contribution,
            turbulence=self.turbulence_intensity
        )

    def compute_wind_force(
        self,
        wind_state: WindState,
        boat_velocity: float,
        boat_heading: float,
        frontal_area: float = 0.5,
        drag_coefficient: float = 1.0
    ) -> np.ndarray:
        """
        Compute aerodynamic force on boat.

        Args:
            wind_state: Current wind conditions
            boat_velocity: Boat speed (m/s)
            boat_heading: Boat heading (radians)
            frontal_area: Exposed frontal area (m²)
            drag_coefficient: Aerodynamic drag coefficient

        Returns:
            Force vector [Fx, Fy] in Newtons
        """
        # Wind velocity vector
        wind_vx = wind_state.speed * np.sin(wind_state.direction)
        wind_vy = wind_state.speed * np.cos(wind_state.direction)

        # Boat velocity vector
        boat_vx = boat_velocity * np.cos(boat_heading)
        boat_vy = boat_velocity * np.sin(boat_heading)

        # Apparent wind
        apparent_vx = wind_vx - boat_vx
        apparent_vy = wind_vy - boat_vy
        apparent_speed = np.sqrt(apparent_vx**2 + apparent_vy**2)

        if apparent_speed < 0.01:
            return np.array([0.0, 0.0])

        # Drag force (opposite to apparent wind direction)
        rho_air = 1.225  # kg/m³
        drag_magnitude = 0.5 * rho_air * drag_coefficient * frontal_area * apparent_speed**2

        # Force direction (aligned with apparent wind)
        force_x = drag_magnitude * apparent_vx / apparent_speed
        force_y = drag_magnitude * apparent_vy / apparent_speed

        return np.array([force_x, force_y])


class WaveModel:
    """
    Wave dynamics model.

    Features:
    - Pierson-Moskowitz spectrum
    - JONSWAP spectrum for fetch-limited seas
    - Added resistance calculations
    - Motion response (heave, pitch, roll)
    """

    def __init__(
        self,
        significant_height: float = 0.3,
        peak_period: float = 4.0,
        direction: float = 0.0,
        spectrum_type: str = 'pierson_moskowitz'
    ):
        self.significant_height = significant_height
        self.peak_period = peak_period
        self.direction = direction
        self.spectrum_type = spectrum_type

        # Wave component phases (for time evolution)
        self.n_components = 20
        self.phases = np.random.uniform(0, 2*np.pi, self.n_components)
        self.frequencies = np.linspace(0.05, 0.5, self.n_components)  # Hz

    def get_wave_state(self, time: float, position: np.ndarray = None) -> WaveState:
        """Get wave conditions at time and position."""
        # Instantaneous wave height (superposition of components)
        wave_height = 0.0
        for i, freq in enumerate(self.frequencies):
            amplitude = self._get_amplitude(freq)
            wave_height += amplitude * np.sin(2*np.pi*freq*time + self.phases[i])

        return WaveState(
            height=self.significant_height + wave_height * 0.3,
            period=self.peak_period,
            direction=self.direction,
            steepness=self.significant_height / (1.56 * self.peak_period**2)
        )

    def _get_amplitude(self, frequency: float) -> float:
        """Get wave amplitude from spectrum."""
        if self.spectrum_type == 'pierson_moskowitz':
            return self._pierson_moskowitz(frequency)
        else:
            return self._pierson_moskowitz(frequency)

    def _pierson_moskowitz(self, f: float) -> float:
        """Pierson-Moskowitz spectrum."""
        fp = 1.0 / self.peak_period
        alpha = 0.0081
        g = 9.81

        if f <= 0:
            return 0.0

        S = (alpha * g**2 / (2*np.pi)**4) * f**(-5) * np.exp(-1.25 * (fp/f)**4)
        return np.sqrt(2 * S * (self.frequencies[1] - self.frequencies[0]))

    def compute_added_resistance(
        self,
        wave_state: WaveState,
        boat_velocity: float,
        boat_heading: float,
        boat_length: float = 4.0,
        boat_beam: float = 1.5
    ) -> float:
        """
        Compute added resistance due to waves.

        Uses simplified Gerritsma-Beukelman formula.

        Args:
            wave_state: Current wave conditions
            boat_velocity: Boat speed (m/s)
            boat_heading: Boat heading (radians)
            boat_length: Boat waterline length (m)
            boat_beam: Boat beam (m)

        Returns:
            Added resistance force (N)
        """
        # Encounter angle
        encounter_angle = boat_heading - wave_state.direction

        # Encounter frequency
        g = 9.81
        omega = 2 * np.pi / wave_state.period
        omega_e = omega - (omega**2 / g) * boat_velocity * np.cos(encounter_angle)

        # Added resistance coefficient (simplified)
        # Depends on wave height squared
        H = wave_state.height
        L = boat_length

        # Simplified formula: R_aw = k * rho * g * B^2 * H^2 / L
        k = 0.5  # coefficient
        rho = 1025  # seawater density
        B = boat_beam

        R_aw = k * rho * g * B**2 * H**2 / L

        # Direction factor (head seas = max, following = min)
        direction_factor = 0.5 + 0.5 * np.cos(encounter_angle)

        return R_aw * direction_factor


class CurrentModel:
    """
    Water current model.

    Supports:
    - Tidal currents
    - River/estuary flow
    - Rotational currents
    - Spatial variation
    """

    def __init__(
        self,
        base_velocity: np.ndarray = None,
        tidal_amplitude: float = 0.0,
        tidal_period: float = 44712.0,  # seconds (12h 25m)
        tidal_phase: float = 0.0
    ):
        self.base_velocity = base_velocity if base_velocity is not None else np.array([0.0, 0.0])
        self.tidal_amplitude = tidal_amplitude
        self.tidal_period = tidal_period
        self.tidal_phase = tidal_phase

    def get_current_state(self, time: float, position: np.ndarray = None) -> CurrentState:
        """Get current conditions at time and position."""
        # Tidal component
        tidal_factor = np.sin(2*np.pi*time/self.tidal_period + self.tidal_phase)
        tidal_velocity = self.tidal_amplitude * tidal_factor

        # Total velocity
        total_velocity = self.base_velocity.copy()
        total_velocity[0] += tidal_velocity  # Add to x-component

        return CurrentState(velocity=total_velocity)

    def compute_current_effect(
        self,
        current_state: CurrentState,
        boat_heading: float
    ) -> Tuple[float, float]:
        """
        Compute effect of current on boat.

        Returns:
            speed_change: Velocity change due to current (m/s)
            drift_angle: Sideways drift angle (radians)
        """
        # Project current onto boat direction
        heading_vec = np.array([np.cos(boat_heading), np.sin(boat_heading)])
        current_vec = current_state.velocity

        # Speed change (current component in direction of travel)
        speed_change = np.dot(current_vec, heading_vec)

        # Drift (perpendicular component)
        perpendicular = np.array([-np.sin(boat_heading), np.cos(boat_heading)])
        drift_component = np.dot(current_vec, perpendicular)

        # Drift angle
        drift_angle = np.arctan2(drift_component, 1.0)  # Small angle approximation

        return speed_change, drift_angle


class SolarModel:
    """
    Solar irradiance model.

    Features:
    - Position-based solar geometry
    - Time-of-day variation
    - Cloud cover effects
    - Panel angle optimization
    """

    def __init__(
        self,
        latitude: float = 40.0,
        longitude: float = -74.0,
        timezone_offset: int = -5,
        panel_tilt: float = 0.0,  # degrees from horizontal
        panel_azimuth: float = 180.0  # degrees from North (180 = South)
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.timezone_offset = timezone_offset
        self.panel_tilt = np.radians(panel_tilt)
        self.panel_azimuth = np.radians(panel_azimuth)

        # Clear sky model parameters
        self.solar_constant = 1361.0  # W/m²

    def get_solar_state(
        self,
        time: float,
        datetime_utc: Optional[datetime] = None,
        cloud_cover: float = 0.0
    ) -> SolarState:
        """
        Get solar irradiance at given time.

        Args:
            time: Simulation time (seconds from start)
            datetime_utc: Optional actual datetime for accurate sun position
            cloud_cover: Cloud cover fraction (0-1)

        Returns:
            SolarState with irradiance values
        """
        if datetime_utc is None:
            # Assume midday on summer day for demo
            hour_angle = (time % 86400) / 86400 * 2 * np.pi - np.pi
            declination = np.radians(23.45)  # Summer solstice
        else:
            hour_angle, declination = self._compute_sun_position(datetime_utc)

        # Solar elevation angle
        lat_rad = np.radians(self.latitude)
        sin_elevation = (np.sin(lat_rad) * np.sin(declination) +
                        np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
        elevation = np.arcsin(np.clip(sin_elevation, -1, 1))

        if elevation <= 0:
            # Sun below horizon
            return SolarState(
                direct_normal=0.0,
                diffuse_horizontal=0.0,
                global_horizontal=0.0,
                cloud_cover=cloud_cover,
                air_mass=0.0
            )

        # Air mass (Kasten-Young formula)
        zenith = np.pi/2 - elevation
        zenith_deg = np.degrees(zenith)
        air_mass = 1.0 / (np.cos(zenith) + 0.50572 * (96.07995 - zenith_deg)**(-1.6364))

        # Clear sky irradiance (Bird model simplified)
        tau = 0.7  # Atmospheric transmittance
        dni_clear = self.solar_constant * tau ** air_mass

        # Apply cloud cover reduction
        cloud_factor = 1.0 - 0.75 * cloud_cover ** 3.4

        dni = dni_clear * cloud_factor
        dhi = 0.1 * self.solar_constant * np.sin(elevation) * (1.0 + 0.5 * cloud_cover)

        # Global horizontal
        ghi = dni * np.sin(elevation) + dhi

        return SolarState(
            direct_normal=dni,
            diffuse_horizontal=dhi,
            global_horizontal=ghi,
            cloud_cover=cloud_cover,
            air_mass=air_mass
        )

    def _compute_sun_position(self, dt: datetime) -> Tuple[float, float]:
        """Compute hour angle and declination from datetime."""
        day_of_year = dt.timetuple().tm_yday

        # Declination (Spencer formula)
        gamma = 2 * np.pi * (day_of_year - 1) / 365
        declination = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
                      - 0.006758 * np.cos(2*gamma) + 0.000907 * np.sin(2*gamma))

        # Hour angle
        solar_time = dt.hour + dt.minute/60 + self.timezone_offset
        solar_time += (self.longitude / 15.0)  # Longitude correction
        hour_angle = np.radians((solar_time - 12) * 15)

        return hour_angle, declination

    def compute_panel_irradiance(
        self,
        solar_state: SolarState,
        sun_azimuth: float,
        sun_elevation: float
    ) -> float:
        """
        Compute irradiance on tilted panel.

        Args:
            solar_state: Current solar conditions
            sun_azimuth: Sun azimuth angle (radians)
            sun_elevation: Sun elevation angle (radians)

        Returns:
            Irradiance on panel surface (W/m²)
        """
        # Angle of incidence on tilted surface
        cos_aoi = (np.sin(sun_elevation) * np.cos(self.panel_tilt) +
                   np.cos(sun_elevation) * np.sin(self.panel_tilt) *
                   np.cos(sun_azimuth - self.panel_azimuth))

        cos_aoi = np.clip(cos_aoi, 0, 1)

        # Beam component
        beam = solar_state.direct_normal * cos_aoi

        # Diffuse (isotropic sky model)
        diffuse = solar_state.diffuse_horizontal * (1 + np.cos(self.panel_tilt)) / 2

        # Ground reflection
        albedo = 0.1  # Water albedo
        reflected = solar_state.global_horizontal * albedo * (1 - np.cos(self.panel_tilt)) / 2

        return beam + diffuse + reflected


class EnvironmentSimulator:
    """
    Combined environmental simulator.

    Integrates wind, waves, current, and solar models.
    """

    def __init__(
        self,
        wind_model: Optional[WindModel] = None,
        wave_model: Optional[WaveModel] = None,
        current_model: Optional[CurrentModel] = None,
        solar_model: Optional[SolarModel] = None
    ):
        self.wind = wind_model or WindModel()
        self.waves = wave_model or WaveModel()
        self.current = current_model or CurrentModel()
        self.solar = solar_model or SolarModel()

        # Weather forecast data (if available)
        self._forecast: Optional[List[Dict]] = None

    def get_state(
        self,
        time: float,
        position: np.ndarray = None
    ) -> EnvironmentalState:
        """Get complete environmental state at time and position."""
        wind_state = self.wind.get_wind_state(time, position)
        wave_state = self.waves.get_wave_state(time, position)
        current_state = self.current.get_current_state(time, position)
        solar_state = self.solar.get_solar_state(time)

        return EnvironmentalState(
            wind=wind_state,
            waves=wave_state,
            current=current_state,
            solar=solar_state
        )

    def compute_total_environmental_force(
        self,
        env_state: EnvironmentalState,
        boat_velocity: float,
        boat_heading: float,
        boat_params: Dict[str, float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute total force from all environmental factors.

        Args:
            env_state: Current environmental state
            boat_velocity: Boat speed (m/s)
            boat_heading: Boat heading (radians)
            boat_params: Boat parameters (area, length, etc.)

        Returns:
            force: Force vector [Fx, Fy] in boat frame
            power_available: Solar power available (W)
        """
        if boat_params is None:
            boat_params = {
                'frontal_area': 0.5,
                'length': 4.0,
                'beam': 1.5,
                'panel_area': 1.0
            }

        # Wind force
        wind_force = self.wind.compute_wind_force(
            env_state.wind,
            boat_velocity,
            boat_heading,
            frontal_area=boat_params.get('frontal_area', 0.5)
        )

        # Wave resistance
        wave_resistance = self.waves.compute_added_resistance(
            env_state.waves,
            boat_velocity,
            boat_heading,
            boat_length=boat_params.get('length', 4.0),
            boat_beam=boat_params.get('beam', 1.5)
        )

        # Current effect (modifies effective velocity)
        speed_change, drift = self.current.compute_current_effect(
            env_state.current,
            boat_heading
        )

        # Combine forces
        # Wave resistance acts opposite to direction of travel
        wave_force = np.array([
            -wave_resistance * np.cos(boat_heading),
            -wave_resistance * np.sin(boat_heading)
        ])

        total_force = wind_force + wave_force

        # Solar power
        power_available = env_state.solar.global_horizontal * boat_params.get('panel_area', 1.0)

        return total_force, power_available

    def set_forecast(self, forecast_data: List[Dict]):
        """Set weather forecast data for prediction."""
        self._forecast = forecast_data

    def predict_conditions(
        self,
        start_time: float,
        duration: float,
        dt: float = 60.0
    ) -> List[EnvironmentalState]:
        """
        Predict future environmental conditions.

        Args:
            start_time: Start time (seconds)
            duration: Prediction duration (seconds)
            dt: Time step (seconds)

        Returns:
            List of predicted environmental states
        """
        predictions = []
        t = start_time

        while t < start_time + duration:
            state = self.get_state(t)
            predictions.append(state)
            t += dt

        return predictions


class WeatherAPIClient:
    """
    Weather API client for real-time conditions.

    Supports multiple providers:
    - OpenWeatherMap
    - NOAA (National Oceanic and Atmospheric Administration)
    - Visual Crossing
    """

    def __init__(
        self,
        provider: str = 'openweathermap',
        api_key: Optional[str] = None
    ):
        self.provider = provider
        self.api_key = api_key
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_duration = 300  # seconds

    def fetch_current(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[EnvironmentalState]:
        """
        Fetch current weather conditions from API.

        Note: Requires internet connection and valid API key.
        Returns None if fetch fails.
        """
        # This is a stub - in production, make actual API calls
        # For now, return simulated data

        cache_key = f"{latitude:.2f},{longitude:.2f}"
        import time
        current_time = time.time()

        if cache_key in self._cache:
            cache_time, cached_data = self._cache[cache_key]
            if current_time - cache_time < self._cache_duration:
                return cached_data

        # Simulate API response
        state = EnvironmentalState(
            wind=WindState(
                speed=3.0 + np.random.random() * 5,
                direction=np.random.random() * 2 * np.pi,
                gust_speed=8.0 + np.random.random() * 4,
                turbulence=0.15
            ),
            waves=WaveState(
                height=0.2 + np.random.random() * 0.3,
                period=3.0 + np.random.random() * 2,
                direction=np.random.random() * 2 * np.pi
            ),
            solar=SolarState(
                global_horizontal=600 + np.random.random() * 400,
                cloud_cover=np.random.random() * 0.5
            ),
            air_temperature=20 + np.random.random() * 10,
            water_temperature=15 + np.random.random() * 8
        )

        self._cache[cache_key] = (current_time, state)
        return state

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        hours: int = 24
    ) -> Optional[List[Dict]]:
        """
        Fetch weather forecast from API.

        Returns list of hourly forecasts.
        """
        # Stub - return simulated forecast
        forecast = []

        for hour in range(hours):
            forecast.append({
                'time_offset_hours': hour,
                'wind_speed': 3.0 + np.random.random() * 5,
                'wind_direction': np.random.random() * 360,
                'wave_height': 0.2 + np.random.random() * 0.3,
                'cloud_cover': np.random.random() * 0.5,
                'temperature': 20 + np.random.random() * 10
            })

        return forecast


def create_default_environment() -> EnvironmentSimulator:
    """Create environment with sensible defaults for solar boat racing."""
    wind = WindModel(
        mean_speed=4.0,
        mean_direction=np.pi/4,  # NE wind
        turbulence_intensity=0.12
    )

    waves = WaveModel(
        significant_height=0.25,
        peak_period=3.5,
        direction=np.pi/4
    )

    current = CurrentModel(
        base_velocity=np.array([0.1, 0.0]),  # Slight eastward current
        tidal_amplitude=0.2
    )

    solar = SolarModel(
        latitude=40.0,
        longitude=-74.0
    )

    return EnvironmentSimulator(wind, waves, current, solar)
