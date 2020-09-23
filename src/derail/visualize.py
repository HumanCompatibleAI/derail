import itertools
import time

import numpy as np
import gym
import derail.envs


class NoisyObsRenderer:
    def __init__(self):
        self.viewer = None

    def render(self, env, data, mode='human'):
        import derail.rendering as rendering

        width = 5
        height = 5

        grid_side = 30
        gs = grid_side

        g_x0 = -110 + 1.5*grid_side
        g_y0 = -110 + 1*grid_side

        make_rect = lambda x, y, w, h : rendering.make_polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])

        make_grid_rect = lambda i, j, di, dj : make_rect(g_x0 + i*gs, g_y0 + j*gs, di*gs, dj*gs)

        if self.viewer is None:
            k = 1.0
            viewer_width = int(k * 500)
            viewer_height = int(k * 800)

            self.viewer = rendering.Viewer(viewer_width, viewer_height)
            self.viewer.set_bounds(-130, 120, -150, 250)

            self.grid = rendering.Grid(start=(g_x0, g_y0), grid_side=grid_side, shape=(width, height))
            self.grid.set_color(0.85, 0.85, 0.85)
            self.viewer.add_geom(self.grid)


        def coords_from_pos(pos):
            if not isinstance(pos, tuple):
                pos = poss[pos]
            return self.grid.coords_from_pos(pos)

        l = 5
        L = range(l)

        for pos in itertools.product(L, L):
            x, y = self.grid.coords_from_pos(pos)
            text = data[pos]

            self.viewer.add_geom(
                rendering.Text(
                    x=x,
                    y=y,
                    text=text,
                    font_size=10,
                )
            )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def run():
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str)
    args = parser.parse_args()

    data = load_data(args.filepath)

    renderer = NoisyObsRenderer()

    data = np.random.randn(5, 5)
    data = np.vectorize(lambda x : f'{x:.2f}\nX')(data)

    env = gym.make('seals/NoisyObs-v0')
    renderer.render(env, data)
    time.sleep(10)


if __name__ == '__main__':
    run()
