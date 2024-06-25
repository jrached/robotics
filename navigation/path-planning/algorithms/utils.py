#!/bin/usr/python3 

from shapely.geometry import Point, LineString, Polygon 
import numpy as np

#########################
###Collision Functions###
#########################
def point_in_collision(pnt, obstacles):
    """
    Uses shapely library to check if a point is in collision with different polygons.

    Input(s): A point in 2D as a tuple, pnt. A list of obstacles represented as 
              lists of the points at the corners of the polygon, obstacles.
    Output(s): Bool, wheter or not the point is in collision.
    """
    point, collision = Point(pnt), False
    for obstacle in obstacles: 
        polygon = Polygon(obstacle)
        collision = collision or polygon.contains(point)
    return collision 

def segment_in_collision(segment, obstacles):
    """
    Uses shapely library to check if a line segment is in collision with different polygons.

    Input(s): A line segment in 2D represented as a list containing the two endpoints of the 
              segment represented as tuples, segment. A list of obstacles represented as 
              lists of the points at the corners of the polygon, obstacles.
    Output(s): Bool, wheter or not the segment is in collision.
    """
    line_segment, collision = LineString(segment), False
    for obstacle in obstacles: 
        polygon = Polygon(obstacle)
        collision = collision or polygon.intersects(line_segment)
    return collision 

##########################
###Path Post-Processing###
##########################
def shortcutting(path, obstacles, factor=5):
    """
    Randomly selects two points down the path, draws a straight line between them,
    if that segment isn't in collision with any obstacles, then it gets rid of every point 
    in between the two points and connects them with the new line.

    Input(s): A path as a list of tuples, where each tuple is a point along the path, path.
              A list of obstacles as lists of points at the corners of each polygon, obstacles.
              The number of iterations as a factor of the length of the path, factor. (optional).
    Output(s): The shorcutted path.
    """
    iters = factor * len(path)
    for _ in range(iters): 
        ndx1, ndx2 = np.sort(np.random.randint(0, len(path), 2))
        pnt1, pnt2 = path[ndx1], path[ndx2]
        if pnt1 != pnt2 and not segment_in_collision((pnt1, pnt2), obstacles):
            path = path[:ndx1+1] + path[ndx2:] 

    return path