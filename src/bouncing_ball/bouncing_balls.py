
"""
This script comes from the RTRBM code by Ilya Sutskever from 
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar

Modified & Updated for Python 3 by Peter O'Connor
"""
import itertools
from typing import Union, List, Generator
import numpy as np


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2
    

def norm(x, axis, keepdims=False):
    return np.sqrt((x**2).sum(axis=axis, keepdims=keepdims))


SIZE=10   # This is an arbitrary size for the "pen" in which balls bounce around.  radii is defined against it.


def generate_bouncing_ball_positions(
        n_samples: int = None,
        n_balls : int = 2,
        n_steps : Union[int, type(None)] = None,
        radii: Union[float, List[float]] = 1.2,
        masses: Union[float, List[float]] = 1.,
        rng: Union[np.random.RandomState, type(None), int] = None,
        ) -> Generator['ndarray(n_samples, n_balls, 2)[float]', None, None]:
    """
    Generate simulated ball positions.
    :param n_samples: Number of parallel-bouncing ball simulations to generata at a time.  None for a single sample
    :param n_balls: Number of balls int the pen.
    :param n_steps: Number of time steps (If None, generator just keeps on going)
    :param radii: Either a number of a List[n_balls] of the radii of each ball
    :param masses: Either a number or a List[n_balls] of the masses of each ball.
    :param rng: A random number generator or seed
    :return: A generator yielding a ndarray(n_samples, n_balls, 2)[float] of ball-positions at a given time-step.
        , or ndarray(n_balls, 2)[float] if n_samples = None
    """

    radii = np.array([radii]*n_balls) if np.isscalar(radii) else np.array(radii)
    masses = np.array([masses]*n_balls) if np.isscalar(masses) else np.array(masses)

    single_sample = n_samples is None
    if single_sample:
        n_samples = 1

    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    v = rng.randn(n_samples, n_balls, 2)
    v = v / norm(v, axis=None, keepdims=False)*.5

    positions = np.empty((n_samples, n_balls, 2))

    for n in range(n_samples):
        good_config = False
        while not good_config:  # Randomly position balls until they're nnot overlapping with eachother or the wall.
            good_config = True
            positions[n] = rng.rand(n_balls, 2)*(SIZE - 2*radii[:, None]) + radii[:, None]

            for i in range(n_balls):
                for j in range(i):
                    if norm(positions[n, i, :]-positions[n, j, :], axis=-1)<radii[i]+radii[j]:
                        good_config=False

    eps = .5
    for t in itertools.count(0):  # for how long do we show small simulation
        if n_steps is not None and t>=n_steps:
            break

        yield positions.copy() if not single_sample else positions[0].copy()

        for mu in range(int(1/eps)):
            positions += eps*v
            v[:] = np.where(positions - radii[:, None] < 0, -v, v)
            v[:] = np.where(positions + radii[:, None] > SIZE, -v, v)

            random_pertubation = np.random.normal(0, 0.05, v.shape)
            v += random_pertubation
            for i in range(n_balls):  # For each ball
                for j in range(i):  # For all balls before ball i
                    separations = positions[..., i, :] - positions[..., j, :]
                    separation_directions = separations/norm(separations, axis=-1, keepdims=True)  # ndarray(..., n_dim)
                    colliding = norm(separations, axis=-1) < radii[i] + radii[j]
                    vi = (separation_directions * v[..., i, :]).sum(axis=-1)
                    vj = (separation_directions * v[..., j, :]).sum(axis=-1)
                    new_v_i, new_v_j = new_speeds(masses[i], masses[j], vi, vj)
                    v[..., i, :] += np.where(colliding[:, None], separation_directions*(new_v_i-vi)[:, None], 0)
                    v[..., j, :] += np.where(colliding[:, None], separation_directions*(new_v_j-vj)[:, None], 0)


def generate_bouncing_ball_data(resolution, n_samples, radii=1.2, save_positions=False, **kwargs) -> Generator['ndarray(n_samples, resolution, resolution)[float]', None, None]:
    """
    Generate images of simulated bouncing balls.
    :param resolution: The resolution of the images to create
    :param kwargs: (To be passed to generate_bouncing_ball_positions)
    :return: A generator yielding an images of batches of bouncing balls.
    """
    for x in generate_bouncing_ball_positions(radii = radii, n_samples=n_samples, **kwargs):
        img = draw_balls(x, resolution=resolution, radii=radii)
        if save_positions:
            yield img, x
        else:
            yield img


def load_bouncing_ball_data(resolution, n_steps, n_samples, save_positions=False, **kwargs):
    """
    Load the bouncing ball dataset.
    :return: An ndarray[n_steps, n_samples, resolution, resolution] containing images of the bouncing balls over time.
    """
    N_BALLS = 1
    data = np.empty((n_steps, n_samples, resolution, resolution))
    positions = np.empty((n_steps, n_samples, N_BALLS, 2))
    for t, imgs in enumerate(generate_bouncing_ball_data(resolution=resolution, n_steps=n_steps, n_samples=n_samples, save_positions=save_positions, **kwargs)):
        data[t] = imgs[0]
        positions[t] = imgs[1]
    if save_positions:
        return data, positions
    else:
        return data


def draw_balls(positions: 'ndarray(n_samples,n_balls,n_dim)[float]', resolution, radii) -> 'ndarray(n_samples, res, res)[float])':
    """
    :param positions: An (n_samples, n_balls, n_dim) array of position data
    :param resolution: The resulution at which to draw the balls
    :param radii: The radii of the balls.
    :return: An ndarray(n_samples,n_balls,n_dim)[float]
    """

    n_steps, n_balls, _ = positions.shape
    if np.isscalar(radii):
        radii=np.array([radii] * n_balls)
    radii = np.array([radii]*n_balls) if np.isscalar(radii) else np.array(radii)
    image=np.zeros((n_steps, resolution, resolution), dtype='float')
    y = x = (.5/resolution+np.arange(0, 1, 1./resolution, dtype='float')) * SIZE
    for i in range(n_balls):
        image += np.exp(-(((y - positions[:, i, 0][:, None, None]) ** 2 + (x[:, None] - positions[:, i, 1][:, None, None]) ** 2) / (radii[i] ** 2)) ** 4)
    image[image>1]=1
    return image


def live_plot_bouncing_ball_data(resolution, n_balls, n_samples, n_steps=None, **kwargs):

    try:
        from artemis.plotting.db_plotting import dbplot
    except ImportError:
        raise Exception('For live plots, install Artemis with `pip install artemis-ml`')

    for d in generate_bouncing_ball_data(n_samples = n_samples, n_balls=n_balls, resolution=resolution, n_steps=n_steps, **kwargs):
        dbplot(d, 'Bouncing Balls')

        
if __name__ == "__main__":

    RESOLUTION = 30
    N_BALLS = 1
    N_SAMPLES = 1
    RADII = 1.2,
    N_STEPS = 1600
    PLOT = True
    #RADII = 2.

    if PLOT:
        live_plot_bouncing_ball_data(n_steps = N_STEPS, resolution=RESOLUTION, n_balls=N_BALLS, n_samples=N_SAMPLES, radii=RADII)
    else:
        d = load_bouncing_ball_data(n_steps=N_STEPS, resolution=RESOLUTION, n_balls = N_BALLS, n_samples = N_SAMPLES, radii=RADII, save_positions=False)
        #print(f'Generated a shape {d.shape} array of bouncing ball data')

