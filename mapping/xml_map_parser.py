# XML Map Parser

from xml.dom.minidom import parse, parseString
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

import numpy as np
import carla
import cv2

import argparse


MAP_MARGIN = 50.0
MAP_RESOLUTION = 0.1


from enum import Enum


class RoadType(Enum):
    DRIVING = 0
    SIDEWALK = 1
    CROSSWALK = 2
    NONE = 3


element_to_road_type = {
    "driving": RoadType.DRIVING,
    "shoulder": RoadType.DRIVING,
    "parking": RoadType.DRIVING,
    "bidirectional": RoadType.DRIVING,
    "driving": RoadType.DRIVING,
    "sidewalk": RoadType.SIDEWALK,
    "crosswalk": RoadType.CROSSWALK,
    "border": RoadType.NONE,
    "median": RoadType.NONE,
    "none": RoadType.NONE,
}

road_type_to_colour = {
    RoadType.DRIVING: (125, 125, 125),
    RoadType.SIDEWALK: (100, 255, 100),
    RoadType.CROSSWALK: (255, 255, 255),
    RoadType.NONE: (55, 55, 55),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenDRIVE File Loader")
    parser.add_argument("--town", type=str, default="Town03", help="CARLA town name to map")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server IP address")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port number")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout for connecting to CARLA server")

    args = parser.parse_args()
    return args


def parse_geometry(road):
    # parse the planview element and pull the geometry
    # and link elements
    planView = road.find("planView")

    min_x = float("inf")
    min_y = float("inf")
    max_x = -float("inf")
    max_y = -float("inf")

    geometry = []
    for geometry_data in planView:
        mode = list(geometry_data)[0]
        if mode.tag == "line":
            points = line_road_to_polyline(geometry_data)
        elif mode.tag == "arc":
            points = arc_road_to_polyline(geometry_data, float(mode.attrib["curvature"]))
        else:
            raise ValueError(f"Unknown geometry type: {mode.tag}")

        geometry.append(points)

    geometry = np.concatenate(geometry, axis=0)
    min_pts = np.min(geometry[:, 1:3], axis=0)
    max_pts = np.max(geometry[:, 1:3], axis=0)
    min_x = min(min_x, min_pts[0])
    min_y = min(min_y, min_pts[1])
    max_x = max(max_x, max_pts[0])
    max_y = max(max_y, max_pts[1])

    return geometry, (min_x, min_y), (max_x, max_y)


def parse_lanes(road):
    # parse the lanes element and pull the lane elements
    lanes_data = road.find("lanes")

    offsets = []
    for element in lanes_data:
        if element.tag == "laneOffset":
            s = float(element.get("s"))
            a = float(element.get("a"))
            b = float(element.get("b"))
            c = float(element.get("c"))
            d = float(element.get("d"))
            offset = [s, a + s * b + s**2 * c + s**3 * d]
            offsets.append(offset)
    else:
        offset = None
    offsets = np.array(offsets)

    def parse_lane(lane):
        id = int(lane.get("id"))
        type = lane.get("type")
        level = lane.get("level")  # don't care

        widths = []
        widths_data = lane.findall("width")
        for width_data in widths_data:
            s = float(width_data.get("sOffset"))
            a = float(width_data.get("a"))
            b = float(width_data.get("b"))
            c = float(width_data.get("c"))
            d = float(width_data.get("d"))
            width = [s, a + s * b + s**2 * c + s**3 * d]
            widths.append(width)
        widths = np.array(widths)

        lane_dict = {
            "id": id,
            "type": type,
            "level": level,
            "widths": widths,
        }

        return lane_dict

    # get the section with lane data
    lane_section_data = lanes_data.find("laneSection")

    # parse the lanes separately, starting with the left lanes
    left_lanes = []
    lane_data = lane_section_data.find("left")
    if lane_data is not None:
        for lane in lane_data:
            left_lanes.append(parse_lane(lane))
    left_lanes.sort(key=lambda x: abs(int(x["id"])))

    # TODO: Ignore the
    # lane_data = lane_section_data.find("center")
    # if lane_data is not None:
    #     for lane in lane_data:
    #         lanes.append(parse_lane(lane))

    right_lanes = []
    lane_data = lane_section_data.find("right")
    if lane_data is not None:
        for lane in lane_data:
            right_lanes.append(parse_lane(lane))
    left_lanes.sort(key=lambda x: abs(int(x["id"])))

    return {"offsets": offsets, "left": left_lanes, "right": right_lanes}


def parse_objects(road):
    # crosswalks are listed as objects in a road element
    objects_data = road.find("objects")
    if objects_data is None:
        return []

    objects = []
    for object_data in objects_data:
        ob_type = object_data.get("type")
        if ob_type == "crosswalk":
            s = float(object_data.get("s"))
            t = float(object_data.get("t"))
            heading = float(object_data.get("hdg"))
            length = float(object_data.get("length"))
            width = float(object_data.get("width"))
            orientation = object_data.get("orientation")

            outline_data = object_data.find("outline")
            points_data = outline_data.findall("cornerLocal")
            points = []
            for point_data in points_data:
                u = float(point_data.get("u"))
                v = float(point_data.get("v"))
                points.append([u, v])
            points = np.array(points)

            object = {
                "type": element_to_road_type[ob_type],
                "s": s,
                "t": t,
                "orientation": orientation,
                "heading": heading,
                "length": length,
                "width": width,
                "outline": points,
            }

            objects.append(object)

    return objects


def real_to_pixel(x, y, origin, resolution):
    return [int((x - origin[0]) / resolution), int((y - origin[1]) / resolution)]


def line_road_to_polyline(geometry):
    x1 = float(geometry.get("x"))
    y1 = float(geometry.get("y"))
    s = float(geometry.get("s"))
    heading = float(geometry.get("hdg"))
    length = float(geometry.get("length"))

    x2 = x1 + (length) * np.cos(heading)
    y2 = y1 + (length) * np.sin(heading)

    return np.array([[s, x1, y1, heading], [s, x2, y2, heading]])


def arc_road_to_polyline(geometry, curvature):
    x = float(geometry.get("x"))
    y = float(geometry.get("y"))
    s = float(geometry.get("s"))
    heading = float(geometry.get("hdg"))
    length = float(geometry.get("length"))

    # calculate arc parameters
    side = np.sign(curvature)
    radius = 1.0 / abs(curvature)
    angle = length / radius

    # find the center of the arc
    center_x = x + radius * np.cos(heading + np.pi / 2) * side
    center_y = y + radius * np.sin(heading + np.pi / 2) * side

    t = np.linspace(0, angle, 250)
    headings = heading + t * side
    angles = headings - np.pi / 2 * side
    xs = center_x + radius * np.cos(angles)
    ys = center_y + radius * np.sin(angles)
    points = np.array([[s, x, y, heading] for x, y, heading in zip(xs, ys, headings)])

    return points


def draw_road_polyline(img, points, origin, resolution):
    polyline = np.array([real_to_pixel(x, y, origin, resolution) for _, x, y, _ in points], dtype=np.int32).reshape(
        -1, 1, 2
    )
    cv2.polylines(img, [polyline], isClosed=False, color=(255, 255, 255), thickness=3)


def main():
    args = parse_arguments()

    # Connect to the CARLA server and download the map
    # client = carla.Client(args.host, args.port)
    # client.set_timeout(args.timeout)
    # world = client.load_world(args.town)
    # carla_map = world.get_map()
    # map_data = carla_map.to_opendrive()

    # # write the map data to a file
    # with open("map_data.xml", "w") as f:
    #     f.write(map_data)

    # Parse the map data
    map_data = open("sample_map_data.xml", "r").read()
    root = ET.fromstring(map_data)

    roads_data = root.findall("road")

    # Extract and print all subelements for each road element
    roads = []
    mins = []
    maxs = []
    for road_data in roads_data:
        id = int(road_data.get("id"))

        # if id != 30 and id != 1772 and id != 10 and id != 1214 and id != 46 and id != 1192:
        # if id != 10: #and id != 1772 and id != 10 and id != 1214 and id != 46 and id != 1192:
        #     continue

        name = road_data.get("name")
        length = float(road_data.get("length"))

        geometry, geo_mins, geo_maxs = parse_geometry(road_data)
        mins.append(geo_mins)
        maxs.append(geo_maxs)

        lanes = parse_lanes(road_data)
        objects = parse_objects(road_data)

        road = {"id": id, "name": name, "length": length, "geometry": geometry, "lanes": lanes, "objects": objects}
        roads.append(road)

    min_x, min_y = np.min(np.array(mins), axis=0)
    max_x, max_y = np.max(np.array(maxs), axis=0)

    print(f"Map bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    map_origin = [min_x - MAP_MARGIN, min_y - MAP_MARGIN]

    map_width = int((max_x - min_x + 2 * MAP_MARGIN) / MAP_RESOLUTION)
    map_height = int((max_y - min_y + 2 * MAP_MARGIN) / MAP_RESOLUTION)

    # create a blank image
    img = np.zeros((map_height, map_width, 3), np.uint8)

    # # draw the roads
    # for road in roads:
    #     draw_road_polyline(img, road["geometry"], map_origin, MAP_RESOLUTION)

    # consruct the road polygons
    polygons = []
    for road in roads:

        def construct_polygon_array(geometry, offsets, lanes, side="left"):
            sign = 1
            if side == "right":
                sign = -1

            # adjust the centerline
            inner_lane_edge = np.zeros([len(geometry), 2])
            for index, (s, x, y, heading) in enumerate(geometry):
                offset = np.interp(s, offsets[:, 0], offsets[:, 1])
                inner_lane_edge[index, 0] = x + offset * np.cos(heading + np.pi / 2)
                inner_lane_edge[index, 1] = y + offset * np.sin(heading + np.pi / 2)

            polygons = []
            road_types = []
            for lane in lanes:
                outer_lane_edge = np.zeros_like(inner_lane_edge)
                for index, (s, x, y, heading) in enumerate(geometry):
                    width = np.interp(s, lane["widths"][:, 0], lane["widths"][:, 1])
                    outer_lane_edge[index, 0] = inner_lane_edge[index, 0] + sign * width * np.cos(heading + np.pi / 2)
                    outer_lane_edge[index, 1] = inner_lane_edge[index, 1] + sign * width * np.sin(heading + np.pi / 2)

                polygon = np.concatenate([inner_lane_edge, np.flip(outer_lane_edge, axis=0)], axis=0)
                polygons.append(polygon)

                road_types.append(element_to_road_type[lane["type"]])

                # move the inner lane edge to the outer lane edge
                inner_lane_edge = outer_lane_edge

            return polygons, road_types

        def construct_object_polygons(geometry, objects):
            polygons = []
            road_types = []
            for object in objects:
                s = object["s"]

                center_x = np.interp(s, geometry[:, 0], geometry[:, 1])
                center_y = np.interp(s, geometry[:, 0], geometry[:, 2])
                heading = np.interp(s, geometry[:, 0], geometry[:, 3]) + object["heading"]
                # heading = np.interp(s, geometry[:, 0], geometry[:, 3])

                # rotate the object outline to match the road heading
                rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
                corner_points = np.dot(rotation_matrix, np.array(object["outline"]).T).T

                # calculate the corner points of the crosswalk
                corner_points = corner_points + np.array([center_x, center_y])

                polygons.append(corner_points)
                road_types.append(object["type"])

            return polygons, road_types

        left_polys, road_types = construct_polygon_array(
            road["geometry"], road["lanes"]["offsets"], road["lanes"]["left"], side="left"
        )
        for poly, road_type in zip(left_polys, road_types):
            if road_type == RoadType.DRIVING:
                colour = road_type_to_colour[road_type]
                points = np.array([real_to_pixel(x, y, map_origin, MAP_RESOLUTION) for x, y in poly], dtype=np.int32)
                cv2.fillPoly(img, [points], color=colour)
        for poly, road_type in zip(left_polys, road_types):
            if road_type != RoadType.DRIVING:
                colour = road_type_to_colour[road_type]
                points = np.array([real_to_pixel(x, y, map_origin, MAP_RESOLUTION) for x, y in poly], dtype=np.int32)
                cv2.fillPoly(img, [points], color=colour)

        right_polys, road_types = construct_polygon_array(
            road["geometry"], road["lanes"]["offsets"], road["lanes"]["right"], side="right"
        )
        for poly, road_type in zip(right_polys, road_types):
            if road_type == RoadType.DRIVING:
                colour = road_type_to_colour[road_type]
                points = np.array([real_to_pixel(x, y, map_origin, MAP_RESOLUTION) for x, y in poly], dtype=np.int32)
                cv2.fillPoly(img, [points], color=colour)
        for poly, road_type in zip(right_polys, road_types):
            if road_type != RoadType.DRIVING:
                colour = road_type_to_colour[road_type]
                points = np.array([real_to_pixel(x, y, map_origin, MAP_RESOLUTION) for x, y in poly], dtype=np.int32)
                cv2.fillPoly(img, [points], color=colour)

        object_polys, road_types = construct_object_polygons(road["geometry"], road["objects"])
        for poly, road_type in zip(object_polys, road_types):
            colour = road_type_to_colour[road_type]
            points = np.array([real_to_pixel(x, y, map_origin, MAP_RESOLUTION) for x, y in poly], dtype=np.int32)
            cv2.fillPoly(img, [points], color=colour)

    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.savefig("xml_map.png")
    plt.show()

    print("done")


if __name__ == "__main__":
    main()
