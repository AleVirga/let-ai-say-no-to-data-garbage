
from geopy.distance import geodesic


class DistanceMatching:
    def __init__(self):
        pass

    def distance_match(
        self,
        latitude1,
        longitude1,
        latitude2,
        longitude2,
    ):
        """
        Returns the distance between the input arrays in meters
        input:
            lat1, lon1, lat2, long2: pyspark.array
        output:
            distance: float
        """
        try:
            latitude1 = float(latitude1)
            longitude1 = float(longitude1)
            latitude2 = float(latitude2)
            longitude2 = float(longitude2)
        except (TypeError, ValueError):
            return None

        if (
            latitude1 is None
            or not (-90 <= float(latitude1) <= 90)
            or longitude1 is None
            or not (-180 <= float(longitude1) <= 180)
            or latitude2 is None
            or not (-90 <= float(latitude2) <= 90)
            or longitude2 is None
            or not (-180 <= float(longitude2) <= 180)
        ):
            return None
        else:
            point1 = (latitude1, longitude1)
            point2 = (latitude2, longitude2)
            return round(geodesic(point1, point2).meters, 2)
