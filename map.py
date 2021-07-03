#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
import numpy as np

from math import atan2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D


from map import GlobalRoutePlannerDAO
from map import GlobalRoutePlanner

from trajectory import Cubic


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enanble car lights')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    np.random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()

        map = world.get_map()

        map_data = GlobalRoutePlannerDAO(map, 2.0)
        map_graph = GlobalRoutePlanner(map_data)

        wp_list = map.generate_waypoints(10.0)

        wp1 = wp_list[np.random.randint(len(wp_list))]
        wp2 = wp_list[np.random.randint(len(wp_list))]

        start = time.time()
        map_graph.setup()
        print('Created graph in {} seconds'.format(time.time() - start))

        start = time.time()
        route = map_graph.trace_route(wp1.transform.location, wp2.transform.location)
        print('Found route from ({},{}) to ({},{}) in {} seconds'.format(wp1.transform.location.x, wp1.transform.location.y,
                                                                         wp2.transform.location.x, wp2.transform.location.y, time.time() - start))

        waypoints = [(wp[0].transform.location.x, wp[0].transform.location.y) for wp in route]
        cubic_path = Cubic(waypoints, 0.05)

        # cubic_path.trajectory([0, 2, 4, 5],
        #                       atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0]),
        #                       atan2(waypoints[-1][1] - waypoints[-2][1], waypoints[-1][0] - waypoints[-2][0]))
        # cubic_path.plot()

        # traj = cubic_path.X.copy()

        cubic_path.trajectory_alt([0, 2, 4, 5],
                                  atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0]),
                                  atan2(waypoints[-1][1] - waypoints[-2][1], waypoints[-1][0] - waypoints[-2][0]))
        cubic_path.plot()
        traj2 = cubic_path.X.copy()

        diff = [(t1[0]-t2[0], t1[1]-t2[1]) for t1, t2 in zip(traj, traj2)]

        f = plt.figure()
        plt.plot(range(len(diff)), [x for x, _ in diff])
        plt.plot(range(len(diff)), [y for _, y in diff])
        plt.show()

    finally:
        pass


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
