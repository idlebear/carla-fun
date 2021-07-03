# Copyright (C) 2021 - Barry Gilhuly

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import pi, sqrt, atan2, cos, sin, pow
from casadi.casadi import le
import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt


class Cubic():
    def __init__(self, waypoints, dt):

        self.waypoints = waypoints
        self.dt = dt

        # calculate the angle/orientation for each segment
        self.angles = []
        last_wp = None
        for wp in self.waypoints:
            if last_wp is not None:
                if wp == last_wp:
                    self.angles.append(self.angles[-1])
                else:
                    dx = wp[0] - last_wp[0]
                    dy = wp[1] - last_wp[1]
                    self.angles.append(atan2(dy, dx))
            last_wp = wp

        # copy the last angle for the last waypoint
        self.angles.append(self.angles[-1])

    @staticmethod
    def _calculate_coefficients(p0, p1, v0, v1, T):
        mat = np.array([
            0, 0, 0, 1,
            T*T*T, T*T, T, 1,
            0, 0, 1, 0,
            3*T*T, 2*T, 1, 0
        ]).reshape((4, 4))

        inv_T_mat = np.linalg.inv(mat)

        x = np.array([p0[0], p1[0], v0[0], v1[0]]).reshape((-1, 1))
        y = np.array([p0[1], p1[1], v0[1], v1[1]]).reshape((-1, 1))

        ax = np.squeeze(inv_T_mat @ x)
        ay = np.squeeze(inv_T_mat @ y)

        return (ax, ay)

    def trajectory(self, waypoint_speeds, theta_i, theta_f):
        self.X = []
        self.X2 = []
        self.V = []
        self.t = []

        v_iter = iter(waypoint_speeds)

        self.angles[-1] = theta_f

        for i in range(1, len(self.waypoints)):

            if self.waypoints[i] == self.waypoints[i-1]:
                continue

            if i < len(self.waypoints) - 1:
                angle_out = self.angles[i] - self.angles[i-1]
                if angle_out > np.pi:
                    angle_out -= np.pi * 2
                elif angle_out < -np.pi:
                    angle_out -= np.pi * 2
                angle_out = angle_out / 2 + self.angles[i-1]
            else:
                angle_out = self.angles[i]

            # initial speed is either the last one from the previous interval or
            # we're just starting
            try:
                v0 = self.V[-1]
                vi = sqrt(v0[0] * v0[0] + v0[1] * v0[1])
            except IndexError:
                vi = next(v_iter)
                v0 = (vi*cos(theta_i), vi*sin(theta_i))

            # with multiple speeds available, we can limit the acceleration over
            # each interval and keep it within reasonable bounds
            try:
                vf = next(v_iter)
            except StopIteration:
                # no more speeds -- reuse the final velocity
                vf = waypoint_speeds[-1]
            v1 = (vf * cos(angle_out), vf * sin(angle_out))

            # traversal time is the approximate (euclidean) distance divided by the
            # average desired velocity
            dx = self.waypoints[i][0] - self.waypoints[i-1][0]
            dy = self.waypoints[i][1] - self.waypoints[i-1][1]
            dist = sqrt(dx * dx + dy * dy)
            T = dist / ((vi + vf)/2)
            try:
                t0 = self.t[-1]
            except IndexError:
                t0 = 0

            (ts, x, v) = self.cubic_time_scaling_spline(self.waypoints[i-1], self.waypoints[i], v0, v1, t0, t0 + T)
            self.t.extend(ts)
            self.V.extend(v)
            self.X.extend(x)

    def trajectory_alt(self, waypoint_speeds, theta_i, theta_f):
        self.X = []
        self.X2 = []
        self.V = []
        self.t = []

        v_iter = iter(waypoint_speeds)

        self.angles[-1] = theta_f

        for i in range(1, len(self.waypoints)):

            if self.waypoints[i] == self.waypoints[i-1]:
                continue

            if i < len(self.waypoints) - 1:
                angle_out = self.angles[i] - self.angles[i-1]
                if angle_out > np.pi:
                    angle_out -= np.pi * 2
                elif angle_out < -np.pi:
                    angle_out -= np.pi * 2
                angle_out = angle_out / 2 + self.angles[i-1]
            else:
                angle_out = self.angles[i]

            # initial speed is either the last one from the previous interval or
            # we're just starting
            try:
                v0 = self.V[-1]
                vi = sqrt(v0[0] * v0[0] + v0[1] * v0[1])
            except IndexError:
                vi = next(v_iter)
                v0 = (vi*cos(theta_i), vi*sin(theta_i))

            # with multiple speeds available, we can limit the acceleration over
            # each interval and keep it within reasonable bounds
            try:
                vf = next(v_iter)
            except StopIteration:
                # no more speeds -- reuse the final velocity
                vf = waypoint_speeds[-1]
            v1 = (vf * cos(angle_out), vf * sin(angle_out))

            # traversal time is the approximate (euclidean) distance divided by the
            # average desired velocity
            dx = self.waypoints[i][0] - self.waypoints[i-1][0]
            dy = self.waypoints[i][1] - self.waypoints[i-1][1]
            dist = sqrt(dx * dx + dy * dy)
            T = dist / ((vi + vf)/2)
            try:
                t0 = self.t[-1]
            except IndexError:
                t0 = 0

            (ts, x, v) = self.polynomial_time_scaling_3rd_order(
                self.waypoints[i-1], self.waypoints[i], v0, v1, t0, t0+T)
            self.t.extend(ts)
            self.V.extend(v)
            self.X.extend(x)

    @staticmethod
    def test():
        dt = 0.05
        waypoints = [[0, 0], [0.5, 0], [0.5, -0.5], [1, -0.5], [1, 0], [1, 0.5],
                     [1.5, 0.5], [1.5, 0], [1.5, -0.5], [1, -0.5], [1, 0],
                     [1, 0.5], [0.5, 0.5], [0.5, 0], [0, 0]]

        cubic_trajectory = Cubic(waypoints=waypoints, dt=dt)

        # set the target parameters
        initial_theta = 0
        final_theta = 0
        cubic_trajectory.trajectory([0, 0.5], initial_theta, final_theta)

        return cubic_trajectory

    @staticmethod
    def test2():
        dt = 0.05
        waypoints = [[0, 0], [0.5, 0], [0.5, -0.5], [1, -0.5], [1, 0], [1, 0.5],
                     [1.5, 0.5], [1.5, 0], [1.5, -0.5], [1, -0.5], [1, 0],
                     [1, 0.5], [0.5, 0.5], [0.5, 0], [0, 0]]

        cubic_trajectory = Cubic(waypoints=waypoints, dt=dt)

        # set the target parameters
        initial_theta = 0
        final_theta = 0
        cubic_trajectory.trajectory_alt([0, 0.5], initial_theta, final_theta)

        return cubic_trajectory

    def plot(self, scale=0.1):
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(self.t, [x for x, _ in self.X])
        ax[0].plot(self.t, [y for _, y in self.X])

        ax[1].plot(self.t, [x for x, _ in self.V])
        ax[1].plot(self.t, [y for _, y in self.V])

        mag_v = [sqrt(vx*vx + vy * vy) for vx, vy in self.V]
        ax[1].plot(self.t, mag_v)

        fig2, ax2 = plt.subplots()
        for wp in self.waypoints:
            c = patches.Circle([wp[0], wp[1]],
                               scale,
                               linewidth=1, edgecolor='k', facecolor='gray')
            ax2.add_patch(c)

        ax2.plot([x for x, _ in self.X], [y for _, y in self.X])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        plt.close(fig)
        plt.close(fig2)

    @staticmethod
    def eval_cubic(a0, a1, a2, a3, t):
        return (a0*t*t*t + a1*t*t + a2*t + a3, 3*a0*t*t + 2*a1*t + a2)

    # Cubic spline implementation based on course content from
    #
    #    https://ucr-ee144.readthedocs.io/en/latest/lab6.html
    #
    def cubic_time_scaling_spline(self, p0, p1, v0, v1, t0, t1):
        T = t1 - t0

        ax, ay = Cubic._calculate_coefficients(p0, p1, v0, v1, T)

        X = []
        V = []
        ts = []
        for t in np.arange(0, T, self.dt):
            ts.append(t + t0)
            (x, vx) = Cubic.eval_cubic(*ax, t)
            (y, vy) = Cubic.eval_cubic(*ay, t)
            X.append((x, y))
            V.append((vx, vy))

        # add the final endpoint
        ts.append(t1)
        (x, vx) = Cubic.eval_cubic(*ax, T)
        (y, vy) = Cubic.eval_cubic(*ay, T)
        X.append((x, y))
        V.append((vx, vy))

        return (ts, X, V)

    @staticmethod
    def eval_cubic_alt(a0, a1, a2, a3, t):
        return ((1/6)*a0*t*t*t + (1/2)*a1*t*t + a2*t + a3, (1/2)*a0*t*t + a1*t + a2)

    #
    # Cubic spline based on an informal paper by G.W.Lucas, published at the Rossum project:
    #
    #   http://rossum.sourceforge.net/papers/CalculationsForRobotics/CubicPath.htm
    #
    # This one is approximately twice as fast as the lab version, with similar results.  There
    # is a small variation between the two that expands as the trajectory gets longer.  Don't
    # know yet if this is significant.
    #
    def polynomial_time_scaling_3rd_order(self, p0, p1, v0, v1, t0, t1):
        '''
            Given a position, velocity and orientation for both the starting and ending position, as well 
            as the time interval, derive the cubic equation that describes the trajectory.
        '''

        # total duration
        T = t1 - t0

        ax = 6 * ((v1[0] + v0[0]) * T - 2 * (p1[0] - p0[0])) / pow(T, 3)
        bx = -2 * ((v1[0] + 2 * v0[0]) * T - 3 * (p1[0] - p0[0])) / pow(T, 2)

        ay = 6 * ((v1[1] + v0[1]) * T - 2 * (p1[1] - p0[1])) / pow(T, 3)
        by = -2 * ((v1[1] + 2 * v0[1]) * T - 3 * (p1[1] - p0[1])) / pow(T, 2)

        X = []
        V = []
        ts = []
        for t in np.arange(0, T, self.dt):
            ts.append(t + t0)
            (x, vx) = Cubic.eval_cubic_alt(ax, bx, v0[0], p0[0], t)
            (y, vy) = Cubic.eval_cubic_alt(ay, by, v0[1], p0[1], t)
            X.append((x, y))
            V.append((vx, vy))

        # add the final endpoint
        ts.append(t1)
        (x, vx) = Cubic.eval_cubic_alt(ax, bx, v0[0], p0[0], T)
        (y, vy) = Cubic.eval_cubic_alt(ay, by, v0[1], p0[1], T)
        X.append((x, y))
        V.append((vx, vy))

        return (ts, X, V)


if __name__ == '__main__':
    import timeit
    # num = 100000
    # t = timeit.timeit(stmt='Cubic.test2()',
    #                   setup='from __main__ import Cubic',
    #                   number=num)
    # print('time: {}, per iteration: {}'.format(t, t/num))

    whatever = Cubic.test()
    whatever.plot()
