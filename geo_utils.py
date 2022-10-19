from math import sqrt, pi, degrees, radians, acos, asin, cos, sin
from shapely.geometry import Point
import math
import numpy as np
import random




def get_flash_density(flash_data, lat, lon):
    lat_idx, lon_idx = idx_from_latlon(lat, lon)
    try:
        output = flash_data['HRFC_COM_FR'][lat_idx][lon_idx]
        return output
    except IndexError:
        print(lat_idx, lon_idx, lat, lon)


def idx_from_latlon(lat, lon):
    lon_idx = math.floor((lon+180)*2)
    lat_idx = math.floor((lat+90)*2)
    return lat_idx, lon_idx


def get_pitch(mapping, lat, lon):
    x_pitches = np.array([30, 28, 26, 24])
    x_thresh = np.array([280, 344, 408])
    y_pitches = np.array([30, 28, 26, 24, 22, 20])
    y_thresh = np.array([280, 344, 408, 472, 536])

    x, y = get_pixel(mapping, lat, lon)
    quad_x = abs(x-650)
    quad_y = abs(y-686)

    x_pitch = x_pitches[np.argmin(quad_x > x_thresh)]
    if all(quad_x > x_thresh):
        x_pitch = 24
    y_pitch = y_pitches[np.argmin(quad_y > y_thresh)]
    if all(quad_y > y_thresh):
        y_pitch = 20

    return float(x_pitch*y_pitch)


def get_pixel(mapping, lat, lon):
    lat_diff = np.abs(mapping.meanLatitude - lat)
    lon_diff = np.abs(mapping.meanLongitude - lon)
    diff = lat_diff + lon_diff
    idx = np.argmin(diff)
    if min(diff) > 2:
        raise ValueError('Difference from pixel center is greater than 2. Is this the right mapping?')
    return mapping.x.iloc[idx], mapping.y.iloc[idx]


def angle_from_nadir(lat, lon, central_lon):
    lon -= central_lon
    return degrees(acos(cos(radians(lat))*cos(radians(lon))))


def distance_to_angle(dist):
    r = 6357
    goes_height = 35786
    alpha = dist / r
    point_dist = sqrt((goes_height+r)**2 + r**2 - 2*(goes_height+r)*r*cos(alpha))
    theta = asin(r*sin(alpha)/point_dist)
    return degrees(theta)
