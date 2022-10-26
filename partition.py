from math import sqrt, pi
import random
from shapely import affinity
from shapely.ops import polygonize
from scipy.spatial import Voronoi
from shapely.geometry import Point, LineString
from shapely.ops import unary_union


def random_partition(polygon, n_points, iterations, cover=False):

    points = points_within(n_points, polygon, cut=False, expand=1/5)

    min_x = min([p.x for p in points])
    min_y = min([p.y for p in points])

    coords = list(zip([p.x-min_x for p in points], [p.y-min_y for p in points]))
    shape = affinity.translate(polygon, -min_x, -min_y)

    for i in range(5):
        vor = Voronoi(coords)
        lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
        polygons = list(polygonize(lines))
        polygons = [poly.intersection(shape) for poly in polygons]
        good_polygons = []
        for poly in polygons:
            if poly.area > 0:
                good_polygons.append(poly)
        polygons = good_polygons
        points = [poly.centroid for poly in polygons]

    polygons = [affinity.translate(poly, min_x, min_y) for poly in polygons]

    if cover:
        covered = unary_union(polygons)

        uncovered = polygon.difference(covered)
        uncovered = uncovered.buffer(0)
        if uncovered.geom_type == 'MultiPolygon':
            for poly in uncovered.geoms:
                poly = poly.buffer(0)
                if poly.area > 0:
                    polygons.append(poly)
        else:
            uncovered = uncovered.buffer(0)
            if uncovered.area > 0:
                polygons.append(uncovered)

    return polygons


def points_within(number, polygon, cut=True, expand=0):
    """Get points within a given polygon

    Parameters
    ----------
    number : int
        Number of points
    polygon : Shapely Polygon
        Polygon to get points within
    cut : bool
        Whether or not to remove points falling outside of the polygon
    expand : float
        proportion of polygon radius by which to expand

    Returns
    -------
    list of Shapely Point
        Points within the polygon
    """
    polygon = polygon.buffer(sqrt(polygon.area/pi)*expand)

    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt) or not cut:
            points.append(pnt)

    return points
