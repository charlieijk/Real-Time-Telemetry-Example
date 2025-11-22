"""
Environmental Physics Module for Solar Regatta.

Provides wind, wave, current, and solar irradiance modeling for realistic
race simulation and strategy optimization.
"""

from .weather import (
    WindState,
    WaveState,
    CurrentState,
    SolarState,
    EnvironmentalState,
    WindModel,
    WaveModel,
    CurrentModel,
    SolarModel,
    EnvironmentSimulator,
    WeatherAPIClient,
    create_default_environment,
)

__all__ = [
    'WindState',
    'WaveState',
    'CurrentState',
    'SolarState',
    'EnvironmentalState',
    'WindModel',
    'WaveModel',
    'CurrentModel',
    'SolarModel',
    'EnvironmentSimulator',
    'WeatherAPIClient',
    'create_default_environment',
]
