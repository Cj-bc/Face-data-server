import dlib


def points2dpoints(ps: dlib.points) -> dlib.dpoints:
    """convert dlib.points object to dlib.dpoints object.
        All points() are should be converted to dpoints,
        as we use float values
    """
    ret = dlib.dpoints()
    for p in ps:
        ret.append(dlib.dpoint(float(p.x), float(p.y)))

    return ret
