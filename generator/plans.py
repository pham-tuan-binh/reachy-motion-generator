"""Recipe conditioning used by the published generator evaluation."""
import numpy as np


class NominalRNG:
    def uniform(self, low=0., high=1., size=None):
        value = (low + high) / 2
        return value if size is None else np.full(size, value)
    def normal(self, loc=0., scale=1., size=None):
        return float(loc) if size is None else np.full(size, float(loc))

def recipe_to_plan(recipe, seed, mode):
    from planner.dsl import expand, to_plan
    if mode == "nominal":
        rng, amplitude, tempo = NominalRNG(), 1., 1.
    elif mode == "randomized":
        rng = np.random.default_rng(seed)
        amplitude, tempo = rng.uniform(.75, 1.25), rng.uniform(.8, 1.25)
    else:
        raise ValueError("mode must be nominal or randomized")
    frames = expand(recipe, rng, amp=amplitude, tsc=tempo)
    return to_plan(frames, rng, mirror=False, fc=2., kdt=.25)
