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

from PIL import Image, ImageDraw

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (sys.version_info.major, sys.version_info.minor, "win-amd64" if os.name == "nt" else "linux-x86_64")
        )[0]
    )
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
import numpy as np

import carla
from carla import TrafficLightState as tls

import argparse
import logging
import datetime
import weakref
import math
import random
import json


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


USE_PYGAME = True

# Colors

# We will use the color palette used in Tango Desktop Project (Each color is indexed depending on brightness level)
# See: https://en.wikipedia.org/wiki/Tango_Desktop_Project

if USE_PYGAME:
    COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
    COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
    COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

    COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
    COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
    COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

    COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
    COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
    COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

    COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
    COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
    COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

    COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
    COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
    COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

    COLOR_PLUM_0 = pygame.Color(173, 127, 168)
    COLOR_PLUM_1 = pygame.Color(117, 80, 123)
    COLOR_PLUM_2 = pygame.Color(92, 53, 102)

    COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
    COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
    COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

    COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
    COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
    COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
    COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
    COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
    COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
    COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

    # Basic colors
    COLOR_WHITE = pygame.Color(255, 255, 255)
    COLOR_BLACK = pygame.Color(0, 0, 0)

else:
    COLOR_BUTTER_0 = (252, 233, 79)
    COLOR_BUTTER_1 = (237, 212, 0)
    COLOR_BUTTER_2 = (196, 160, 0)

    COLOR_ORANGE_0 = (252, 175, 62)
    COLOR_ORANGE_1 = (245, 121, 0)
    COLOR_ORANGE_2 = (209, 92, 0)

    COLOR_CHOCOLATE_0 = (233, 185, 110)
    COLOR_CHOCOLATE_1 = (193, 125, 17)
    COLOR_CHOCOLATE_2 = (143, 89, 2)

    COLOR_CHAMELEON_0 = (138, 226, 52)
    COLOR_CHAMELEON_1 = (115, 210, 22)
    COLOR_CHAMELEON_2 = (78, 154, 6)

    COLOR_SKY_BLUE_0 = (114, 159, 207)
    COLOR_SKY_BLUE_1 = (52, 101, 164)
    COLOR_SKY_BLUE_2 = (32, 74, 135)

    COLOR_PLUM_0 = (173, 127, 168)
    COLOR_PLUM_1 = (117, 80, 123)
    COLOR_PLUM_2 = (92, 53, 102)

    COLOR_SCARLET_RED_0 = (239, 41, 41)
    COLOR_SCARLET_RED_1 = (204, 0, 0)
    COLOR_SCARLET_RED_2 = (164, 0, 0)

    COLOR_ALUMINIUM_0 = (238, 238, 236)
    COLOR_ALUMINIUM_1 = (211, 215, 207)
    COLOR_ALUMINIUM_2 = (186, 189, 182)
    COLOR_ALUMINIUM_3 = (136, 138, 133)
    COLOR_ALUMINIUM_4 = (85, 87, 83)
    COLOR_ALUMINIUM_4_5 = (66, 62, 64)
    COLOR_ALUMINIUM_5 = (46, 52, 54)

    # Basic colors
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)


# Other constants
MAX_PIXELS_PER_METER = 12
DEFAULT_MAP_SCALE = 0.1


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class MapImage:
    """Class encharged of rendering a 2D image from top view of a carla world."""

    def __init__(self, carla_world, carla_map, pixels_per_meter, show_triggers, show_connections, show_spawn_points):
        """Renders the map image generated based on the world, its map and additional flags that provide extra information about the road network"""

        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0
        self.show_triggers = show_triggers
        self.show_connections = show_connections
        self.show_spawn_points = show_spawn_points

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        # Maximum size of a Pygame surface
        width_in_pixels = (1 << 14) - 1

        # Adapt Pixels per meter to make world fit in surface
        surface_pixel_per_meter = int(width_in_pixels / self.width)
        if surface_pixel_per_meter > MAX_PIXELS_PER_METER:
            surface_pixel_per_meter = MAX_PIXELS_PER_METER

        self._pixels_per_meter = surface_pixel_per_meter
        width_in_pixels = int(self._pixels_per_meter * self.width)

        if USE_PYGAME:
            self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        else:
            self.map_image = Image.new("RGBA", (width_in_pixels, width_in_pixels), (0, 0, 0, 0))
            self.big_map_surface = ImageDraw.Draw(self.map_image)

        # Load OpenDrive content
        opendrive_content = carla_map.to_opendrive()

        # Render map
        self.draw_road_map(self.big_map_surface, carla_world, carla_map, self.world_to_pixel, self.world_to_pixel_width)

        # Save rendered map for next executions of same map
        map = carla_map.name.split("/")[-1]
        filename = f"{map}-map.png"

        # TODO: make the map image name an argument
        # full_path = str(os.path.join(dirname, filename))

        if USE_PYGAME:
            pygame.image.save(self.big_map_surface, filename)
        else:
            self.map_image.save(filename)

        # write the parameters to a json file
        with open(f"{map}-map.json", "w") as f:
            json.dump(
                {
                    "pixel_width": width_in_pixels,
                    "pixels_per_meter": self._pixels_per_meter,
                    "width": self.width,
                    "world_offset": self._world_offset,
                    "scale": self.scale,
                },
                f,
            )

        self.surface = self.big_map_surface

    def draw_road_map(self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        """Draws all the roads, including lane markings, arrows and traffic signs"""
        if USE_PYGAME:
            map_surface.fill(COLOR_ALUMINIUM_4)
        else:
            map_surface.rectangle(
                [(0, 0), (self._pixels_per_meter * self.width, self._pixels_per_meter * self.width)],
                fill=COLOR_ALUMINIUM_4,
            )
        precision = 0.05

        def lane_marking_color_to_tango(lane_marking_color):
            """Maps the lane marking color enum specified in PythonAPI to a Tango Color"""
            tango_color = COLOR_BLACK

            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2

            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0

            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0

            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0

            return tango_color

        def draw_solid_line(surface, color, closed, points, width):
            """Draws solid lines in a surface given a set of points, width and color"""
            if len(points) >= 2:
                if USE_PYGAME:
                    pygame.draw.lines(surface, color, closed, points, width)
                else:
                    if closed:
                        points.append(points[0])
                    surface.line(points, fill=color, width=width)

        def draw_broken_line(surface, color, closed, points, width):
            """Draws broken lines in a surface given a set of points, width and color"""
            # Select which lines are going to be rendered from the set of lines
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]

            # Draw selected lines
            for line in broken_lines:
                if USE_PYGAME:
                    pygame.draw.lines(surface, color, closed, line, width)
                else:
                    surface.line(line, fill=color, width=width)

        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
            as a combination of Broken and Solid lines"""
            margin = 0.25
            marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
                return [(lane_marking_type, lane_marking_color, marking_1)]
            else:
                marking_2 = [
                    world_to_pixel(lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2)))
                    for w in waypoints
                ]
                if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                    return [
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
                    ]
                elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                    return [
                        (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
                    ]
                elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                    return [
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
                    ]
                elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                    return [
                        (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
                    ]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

        def draw_lane(surface, lane, color):
            """Renders a single lane in a surface and with a specified color"""
            for side in lane:
                lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
                lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

                polygon = lane_left_side + [x for x in reversed(lane_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    if USE_PYGAME:
                        pygame.draw.polygon(surface, color, polygon, 5)
                        pygame.draw.polygon(surface, color, polygon)
                    else:
                        surface.polygon(polygon, fill=color)

        def draw_lane_marking(surface, waypoints):
            """Draws the left and right side of lane markings"""
            # Left Side
            draw_lane_marking_single_side(surface, waypoints[0], -1)

            # Right Side
            draw_lane_marking_single_side(surface, waypoints[1], 1)

        def draw_lane_marking_single_side(surface, waypoints, sign):
            """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
            the waypoint based on the sign parameter"""
            lane_marking = None

            marking_type = carla.LaneMarkingType.NONE
            previous_marking_type = carla.LaneMarkingType.NONE

            marking_color = carla.LaneMarkingColor.Other
            previous_marking_color = carla.LaneMarkingColor.Other

            markings_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE
            for sample in waypoints:
                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    # Get the list of lane markings to draw
                    markings = get_lane_markings(
                        previous_marking_type, lane_marking_color_to_tango(previous_marking_color), temp_waypoints, sign
                    )
                    current_lane_marking = marking_type

                    # Append each lane marking in the list
                    for marking in markings:
                        markings_list.append(marking)

                    temp_waypoints = temp_waypoints[-1:]

                else:
                    temp_waypoints.append((sample))
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            # Add last marking
            last_markings = get_lane_markings(
                previous_marking_type, lane_marking_color_to_tango(previous_marking_color), temp_waypoints, sign
            )
            for marking in last_markings:
                markings_list.append(marking)

            # Once the lane markings have been simplified to Solid or Broken lines, we draw them
            for markings in markings_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(surface, markings[1], False, markings[2], 2)
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(surface, markings[1], False, markings[2], 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            """Draws an arrow with a specified color given a transform"""
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            end = transform.location
            start = end - 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir

            # Draw lines
            if USE_PYGAME:
                pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [start, end]], 4)
                pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [left, start, right]], 4)
            else:
                points = [world_to_pixel(x) for x in [start, end]]
                surface.line(points, fill=color, width=4)
                points = [world_to_pixel(x) for x in [left, start, right]]
                surface.lines(points, fill=color, width=4)

        def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
            """Draw stop traffic signs and its bounding box if enabled"""
            transform = actor.get_transform()
            waypoint = carla_map.get_waypoint(transform.location)

            # angle = -waypoint.transform.rotation.yaw - 90.0
            # font_surface = pygame.transform.rotate(font_surface, angle)
            # pixel_pos = world_to_pixel(waypoint.transform.location)
            # offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            # surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = (
                carla.Location(-forward_vector.y, forward_vector.x, forward_vector.z) * waypoint.lane_width / 2 * 0.7
            )

            line = [
                (waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                (waypoint.transform.location + (forward_vector * 1.5) - (left_vector)),
            ]

            line_pixel = [world_to_pixel(p) for p in line]
            if USE_PYGAME:
                pygame.draw.lines(surface, color, True, line_pixel, 2)
            else:
                line_pixel.append(line_pixel[0])  # closed!
                surface.line(line_pixel, fill=color, width=2)

            # Draw bounding box of the stop trigger
            if self.show_triggers:
                # corners = Util.get_bounding_box(actor)
                corners = actor.bounding_box.get_world_vertices(obj.transform)
                corners = [world_to_pixel(p) for p in corners]
                if USE_PYGAME:
                    pygame.draw.lines(surface, trigger_color, True, corners, 2)
                else:
                    corners.append(corners[0])  # Closed
                    surface.line(corners, fill=trigger_color, width=2)

        def draw_crosswalk(surface, polygon, color=COLOR_WHITE):
            """Draws a crosswalk given a polygon and a color"""
            polygon = [world_to_pixel(p) for p in polygon]
            if USE_PYGAME:
                pygame.draw.polygon(surface, color, polygon)
            else:
                surface.polygon(polygon, fill=color)

        def draw_object(surface, obj):
            """Draws an object given a surface and an object"""
            corners = obj.bounding_box.get_world_vertices(obj.transform)[::2]
            corners = [world_to_pixel(p) for p in corners[::-1]]

            center = world_to_pixel(obj.transform.location)
            radius = int(world_to_pixel_width(min(obj.bounding_box.extent.x, obj.bounding_box.extent.y)))

            color = None
            if obj.type == carla.libcarla.CityObjectLabel.Buildings:
                color = None  # COLOR_ALUMINIUM_5
            # elif obj.type == carla.libcarla.CityObjectLabel.Vegetation:
            #     color = COLOR_CHAMELEON_2
            # elif obj.type == carla.libcarla.CityObjectLabel.Fences:
            #     color = COLOR_WHITE
            # elif obj.type == carla.libcarla.CityObjectLabel.Walls:
            #     color = None  # COLOR_ALUMINIUM_4
            # elif obj.type == carla.libcarla.CityObjectLabel.TrafficSign:
            #     color = None
            elif obj.type == carla.libcarla.CityObjectLabel.Poles:
                color = COLOR_BLACK
            # elif obj.type == carla.libcarla.CityObjectLabel.TrafficLight:
            #     color = None
            # elif obj.type == carla.libcarla.CityObjectLabel.Car:
            #     color = COLOR_ALUMINIUM_5
            # elif obj.type == carla.libcarla.CityObjectLabel.Ground:
            #     color = COLOR_CHOCOLATE_1
            # elif obj.type == carla.libcarla.CityObjectLabel.Terrain:
            #     color = None  # COLOR_CHAMELEON_1
            # elif obj.type == carla.libcarla.CityObjectLabel.Person:
            #     color = None

            if color is not None:
                if USE_PYGAME:
                    pygame.draw.circle(surface, color, center, radius)
                else:
                    bounding_box = [(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)]
                    surface.ellipse(bounding_box, fill=color, width=2)

        # def draw_crosswalk(surface, transform=None, color=COLOR_ALUMINIUM_2):
        #     """Given two points A and B, draw white parallel lines from A to B"""
        #     a = carla.Location(0.0, 0.0, 0.0)
        #     b = carla.Location(10.0, 10.0, 0.0)

        #     ab = b - a
        #     length_ab = math.sqrt(ab.x**2 + ab.y**2)
        #     unit_ab = ab / length_ab
        #     unit_perp_ab = carla.Location(-unit_ab.y, unit_ab.x, 0.0)

        #     # Crosswalk lines params
        #     space_between_lines = 0.5
        #     line_width = 0.7
        #     line_height = 2

        #     current_length = 0
        #     while current_length < length_ab:

        #         center = a + unit_ab * current_length

        #         width_offset = unit_ab * line_width
        #         height_offset = unit_perp_ab * line_height
        #         list_point = [center - width_offset - height_offset,
        #                       center + width_offset - height_offset,
        #                       center + width_offset + height_offset,
        #                       center - width_offset + height_offset]

        #         list_point = [world_to_pixel(p) for p in list_point]
        #         pygame.draw.polygon(surface, color, list_point)
        #         current_length += (line_width + space_between_lines) * 2

        def lateral_shift(transform, shift):
            """Makes a lateral shift of the forward vector of a transform"""
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology, index):
            """Draws traffic signs and the roads network with sidewalks, parking and shoulders by generating waypoints"""
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            set_waypoints = []
            for waypoint in topology:
                waypoints = [waypoint]

                # Generate waypoints of a road id. Stop when road id differs
                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break
                set_waypoints.append(waypoints)

                # Draw Shoulders, Parkings and Sidewalks
                PARKING_COLOR = COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_5
                SIDEWALK_COLOR = COLOR_ALUMINIUM_3

                shoulder = [[], []]
                parking = [[], []]
                sidewalk = [[], []]

                for w in waypoints:
                    # Classify lane types until there are no waypoints by going left
                    l = w.get_left_lane()
                    while l and l.lane_type != carla.LaneType.Driving:

                        if l.lane_type == carla.LaneType.Shoulder:
                            shoulder[0].append(l)

                        if l.lane_type == carla.LaneType.Parking:
                            parking[0].append(l)

                        if l.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[0].append(l)

                        l = l.get_left_lane()

                    # Classify lane types until there are no waypoints by going right
                    r = w.get_right_lane()
                    while r and r.lane_type != carla.LaneType.Driving:

                        if r.lane_type == carla.LaneType.Shoulder:
                            shoulder[1].append(r)

                        if r.lane_type == carla.LaneType.Parking:
                            parking[1].append(r)

                        if r.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[1].append(r)

                        r = r.get_right_lane()

                # Draw classified lane types
                draw_lane(map_surface, shoulder, SHOULDER_COLOR)
                draw_lane(map_surface, parking, PARKING_COLOR)
                draw_lane(map_surface, sidewalk, SIDEWALK_COLOR)

            # Draw Roads
            for waypoints in set_waypoints:
                waypoint = waypoints[0]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # Draw Lane Markings and Arrows
                if not waypoint.is_junction:
                    draw_lane_marking(map_surface, [waypoints, waypoints])
                    for n, wp in enumerate(waypoints):
                        if ((n + 1) % 400) == 0:
                            draw_arrow(map_surface, wp.transform)

        topology = carla_map.get_topology()
        draw_topology(topology, 0)

        # if self.show_spawn_points:
        #     for sp in carla_map.get_spawn_points():
        #         draw_arrow(map_surface, sp, color=COLOR_CHOCOLATE_0)

        if self.show_connections:
            dist = 1.5

            def to_pixel(wp):
                return world_to_pixel(wp.transform.location)

            for wp in carla_map.generate_waypoints(dist):
                col = (0, 255, 255) if wp.is_junction else (0, 255, 0)
                for nxt in wp.next(dist):
                    if USE_PYGAME:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(nxt), 2)
                    else:
                        map_surface.line([to_pixel(wp), to_pixel(nxt)], fill=col, width=2)
                if wp.lane_change & carla.LaneChange.Right:
                    r = wp.get_right_lane()
                    if r and r.lane_type == carla.LaneType.Driving:
                        if USE_PYGAME:
                            pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(r), 2)
                        else:
                            map_surface.line([to_pixel(wp), to_pixel(r)], fill=col, width=2)
                if wp.lane_change & carla.LaneChange.Left:
                    l = wp.get_left_lane()
                    if l and l.lane_type == carla.LaneType.Driving:
                        if USE_PYGAME:
                            pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(l), 2)
                        else:
                            map_surface.line([to_pixel(wp), to_pixel(l)], fill=col, width=2)

        # draw the crosswalks
        crosswalk_locations = carla_map.get_crosswalks()
        if len(crosswalk_locations):
            crosswalk_poly = []
            for location in crosswalk_locations:
                if len(crosswalk_poly) > 1 and location.x == crosswalk_poly[0].x and location.y == crosswalk_poly[0].y:
                    # completed polygon -- draw it and start the next
                    draw_crosswalk(map_surface, crosswalk_poly)
                    crosswalk_poly = []
                else:
                    crosswalk_poly.append(location)

        objs = carla_world.get_environment_objects()
        for obj in objs:
            draw_object(map_surface, obj)

        # # Find and Draw Traffic Signs: Stops and Yields
        # font_size = world_to_pixel_width(1)
        # font = pygame.font.SysFont("Arial", font_size, True)

        # stops = [actor for actor in actors if "stop" in actor.type_id]
        # yields = [actor for actor in actors if "yield" in actor.type_id]

        # stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        # stop_font_surface = pygame.transform.scale(
        #     stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2)
        # )

        # yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        # yield_font_surface = pygame.transform.scale(
        #     yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2)
        # )

        # for ts_stop in stops:
        #     draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

        # for ts_yield in yields:
        #     draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)

    def world_to_pixel(self, location, offset=(0, 0)):
        """Converts the world coordinates to pixel coordinates"""
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        # return [int(x - offset[0]), int(y - offset[1])]
        return (int(x - offset[0]), int(y - offset[1]))  # tuple!

    def world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        """Scales the map surface"""
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            if USE_PYGAME:
                self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))
            else:
                self.surface = self.big_map_surface.resize((width, width), Image.BICUBIC)


class World(object):
    """Class that contains all the information of a carla world that is running on the server side"""

    def __init__(self, name, args, timeout):
        self.client = None
        self.name = name
        self.args = args
        self.timeout = timeout
        self.server_fps = 0.0
        self.simulation_time = 0
        # self.server_clock = pygame.time.Clock()

        # World data
        self.world = None
        self.town_map = None
        self.actors_with_transforms = []

        self._hud = None
        self._input = None

        self.surface_size = [0, 0]
        self.prev_scaled_size = 0
        self.scaled_size = 0

        # Hero actor
        self.hero_actor = None
        self.spawned_hero = None
        self.hero_transform = None

        self.scale_offset = [0, 0]

        self.vehicle_id_surface = None
        self.result_surface = None

        self.traffic_light_surfaces = None
        self.affected_traffic_light = None

        # Map info
        self.map_image = None
        self.border_round_surface = None
        self.original_surface_size = None
        self.hero_surface = None
        self.actors_surface = None

    def _get_data_from_carla(self):
        """Retrieves the data from the server side"""
        try:
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(self.timeout)

            if self.args.map is None:
                world = self.client.get_world()
            else:
                world = self.client.load_world(self.args.map)

            town_map = world.get_map()
            return (world, town_map)

        except RuntimeError as ex:
            logging.error(ex)
            raise (ex)

    def start(self, hud, input_control):
        """Build the map image, stores the needed modules and prepares rendering in Hero Mode"""
        self.world, self.town_map = self._get_data_from_carla()

        # Create Surfaces
        self.map_image = MapImage(
            carla_world=self.world,
            carla_map=self.town_map,
            pixels_per_meter=MAX_PIXELS_PER_METER,
            show_triggers=self.args.show_triggers,
            show_connections=self.args.show_connections,
            show_spawn_points=False,  # self.args.show_spawn_points,
        )

        print("Map image created")


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--host", metavar="H", default="127.0.0.1", help="IP of the host server (default: 127.0.0.1)"
    )
    argparser.add_argument(
        "-p", "--port", metavar="P", default=2000, type=int, help="TCP port to listen to (default: 2000)"
    )
    argparser.add_argument(
        "--tm-port", metavar="P", default=8000, type=int, help="port to communicate with TM (default: 8000)"
    )
    argparser.add_argument("-s", "--seed", metavar="S", type=int, help="Random device seed")
    argparser.add_argument("--show-triggers", action="store_true", default=False, help="Enanble car lights")
    argparser.add_argument("--show-connections", action="store_true", default=False, help="Enanble car lights")
    argparser.add_argument("--map", default="Town03", type=str, help="Name of Town Map")
    argparser.add_argument(
        "--res", metavar="WIDTHxHEIGHT", default="1280x720", help="window resolution (default: 1280x720)"
    )

    args = argparser.parse_args()
    args.description = argparser.description
    args.width, args.height = [int(x) for x in args.res.split("x")]

    # Init Pygame
    if USE_PYGAME:
        pygame.init()
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    world = World(args.map, args, 1000000)
    world.start(None, None)

    return


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
