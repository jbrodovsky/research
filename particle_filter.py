""" Particle filter algorithim and simulation code"""

from scipy.stats import norm
from pandas import DataFrame
from xarray import DataArray
import numpy as np
from haversine import haversine, Unit
from filterpy.monte_carlo import residual_resample
from gmt_tool import get_map_point
from pyins.pyins import earth

OVERFLOW = 500


def propagate(
    particles: np.ndarray,
    control: np.ndarray,
    dt: float = 1.0,
    noise=np.diag([0, 0, 0]),
) -> np.ndarray:
    """
    Process model according to Groves. State vector is in NED:
    [lat, lon, depth, vN, vE, vD]. Controls are currently none, dt is in seconds.

    """
    n, _ = particles.shape
    # Velocity update
    velocity = np.random.multivariate_normal(control, noise, (n,))

    # Depth update
    previous_depth = particles[:, 2]
    new_depth = previous_depth + 0.5 * (particles[:, 5] + velocity[:, 2]) * dt
    # Latitude update
    previous_lat = particles[:, 0]
    lat_rad = np.deg2rad(previous_lat)
    RN, RE_0, _ = earth.principal_radii(
        previous_lat, np.zeros_like(previous_lat)
    )  # pricipal_radii expect degrees
    previous_lat = np.deg2rad(previous_lat)
    lat_rad = previous_lat + 0.5 * dt * (
        particles[:, 3] / (RN - previous_depth) + velocity[:, 0] / (RN - new_depth)
    )
    # Longitude update
    _, RE_1, _ = earth.principal_radii(np.rad2deg(lat_rad), np.zeros_like(lat_rad))
    lon_rad = np.deg2rad(particles[:, 1]) + 0.5 * dt * (
        particles[:, 4] / ((RE_0 - previous_depth) * np.cos(previous_lat))
        + velocity[:, 1] / ((RE_1 - new_depth) * np.cos(lat_rad))
    )

    particles = np.array([np.rad2deg(lat_rad), np.rad2deg(lon_rad), new_depth]).T
    particles = np.hstack([particles, velocity])
    return particles


def update_relief(
    particles: np.ndarray,
    # weights: np.ndarray,
    geo_map: DataArray,
    observation,
    relief_sigma: float,
) -> np.ndarray:
    """
    Measurement update
    """

    n, _ = particles.shape
    observation = np.asarray(observation)
    observation = np.tile(observation, (n,))
    observation -= particles[:, 2]
    z_bar = -get_map_point(geo_map, particles[:, 1], particles[:, 0])
    dz = observation - z_bar
    w = np.zeros_like(dz)

    inds = np.abs(dz) < OVERFLOW
    w[inds] = norm(loc=0, scale=relief_sigma).pdf(dz[inds])

    w[np.isnan(w)] = 1e-16
    if np.any(np.isnan(w)):
        print("NAN elements found")
        w[np.isnana(w)] = 1e-16

    w_sum = np.sum(w)
    try:
        new_weights = w / w_sum
    except ZeroDivisionError:
        return np.ones_like(w) / n
    return new_weights


def rmse(particles, truth):
    """
    root mean square error calculation
    """
    diffs = [haversine(truth, (p[0], p[1]), Unit.METERS) for p in particles]
    diffs = np.asarray(diffs)
    return np.sqrt(np.mean(diffs**2))


def weighted_rmse(particles, weights, truth):
    """
    Weighted root mean square error calculation
    """
    diffs = [haversine(truth, (p[0], p[1]), Unit.METERS) for p in particles]
    diffs = np.asarray(diffs) * weights
    return np.sqrt(diffs**2)


def run_particle_filter(
    mu: np.ndarray,
    cov: np.ndarray,
    N: int,
    data: DataFrame,
    geo_map: DataArray,
    noise: np.ndarray = np.diag([0.1, 0.01, 0]),
    measurement_sigma: float = 15,
):
    """
    Run through an instance of the particle filter
    """
    particles = np.random.multivariate_normal(mu, cov, (N,))
    weights = np.ones((N,)) / N
    error = np.zeros(len(data))
    rms_error = np.zeros_like(error)
    # wrmse = np.zeros_like(error)
    # Initial values
    estimate = [weights @ particles]
    error[0] = haversine(estimate[0][:2], (data.LAT[0], data.LON[0]), Unit.METERS)
    rms_error[0] = rmse(particles, (data["LAT"][0], data["LAT"][0]))
    # wrmse[0] = weighted_rmse(particles, weights, (data["LAT"][0], data["LAT"][0]))

    for i, item in enumerate(data.iterrows()):
        if i > 0:
            row = item[1]
            # Propagate
            u = np.asarray([row["vN"], row["vE"], 0])
            particles = propagate(particles, u, row["dt"].seconds, noise)
            # Update
            obs = row["CORR_DEPTH"]
            weights = update_relief(particles, geo_map, obs, measurement_sigma)
            # Resample
            inds = residual_resample(weights)
            particles[:] = particles[inds]
            # Calculate estimate and error
            estimate.append(weights @ particles)
            error[i] = haversine(
                estimate[i][:2], (data.LAT[i], data.LON[i]), Unit.METERS
            )
            rms_error[i] = rmse(particles, (row.LAT, row.LON))
            # wrmse[i] = weighted_rmse(particles, weights, (row.LAT, row.LON))
    estimate = np.asarray(estimate)
    return estimate, error, rms_error  # , wrmse
