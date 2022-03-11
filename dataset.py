"""Dataset.


Read and process the dataset. This module follows a factory design
pattern. Therefore, the dataset class must not be directly instanciated.
Instead, an instance of the dataset should be used
"""
from typing import Dict, Tuple, Optional, List, Union
from config import DATASET, DATASET_ROOT_DIR
from json import load, dumps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

import sys
import argparse


class Calibration(object):
    """Calibration.

    Process calibration data. Those data are mainly calibration matrices
    for converting measurement from each sensor coordinate frame into a
    reference frame.

    Attributes:
        camera: calibration matrix of the camera sensor.
                Convert data from camera-coordinate frame into reference frame
        cameraCoef: matrix for mapping an object from the coordinate frame
                    of the  camera to its field of View
        lidar: calibration matrix of the lidar sensor.
               Convert data from lidar-coordinate frame into reference frame
        radar: calibration matrix of the radar sensor.
               Convert data from radar-coordinate frame into reference frame
    """

    def __init__(self, filepath: str) -> None:
        """Contructor of a config."""
        self.sensors_name: Dict[str, str] = {
            "camera_front": "camera",
            "lidar_vlp16": "lidar",
            "radar_6455": "radar",
        }
        with open(filepath, "r") as file_handler:
            for sensor in load(file_handler)["sensors"]:
                setattr(
                    self,
                    self.sensors_name[sensor["sensor_uid"]],
                    np.array(sensor["calib_data"]["T_to_ref_COS"]),
                )
                if "camera" in sensor["sensor_uid"]:
                    # The coefficient K from the dataset map an object from the
                    # coordinate frame of the  camera to its field of View

                    # TODO: What does "the field of view of the camera" mean?
                    setattr(
                        self,
                        self.sensors_name[sensor["sensor_uid"]] + "Coef",
                        np.array(sensor["calib_data"]["K"]),
                    )

    @classmethod
    def toRef(cls, config: np.array, pointcloud: np.array) -> np.array:
        """Transform a pointcloud into the the reference coordinate frame.
        
        Arguments:
            config: transmation matrix composed of rotation matrix and
                    translation vector.

                    - shape: (4, 4)
                    - structure
                        r11 r12 r13 t1
                        r21 r22 r23 t2
                        r31 r32 r33 t3
                         0   0   0   1
            pointcloud: dataset composed of x, y, z coordinates of sensor
                        reading

                        - shape: (n, 3)
                        - structure
                            X001 Y001 Z001
                            X002 Y002 Z002
                            X003 Y003 Z003
                            ...
                            .
        Return:
            Shape: (n, 3)
        """
        # Read the number of rows
        rows, _ = np.shape(pointcloud)
        # Change the structure of the point cloud
        #
        # X001 Y001 Z001 1
        # X002 Y002 Z002 1
        # X003 Y003 Z003 1
        # ...
        pointcloud = np.column_stack([pointcloud, np.ones(((rows, 1)))])
        pointcloud = np.dot(config, np.transpose(pointcloud))
        pointcloud = np.transpose(pointcloud)[:, :3]
        return pointcloud

    @classmethod
    def fromRef(cls, config: np.array,
                          pointcloud: np.array) -> np.array:
        """Transform a pointcloud from the reference CoF into a given CoF.

        CoF: Coordinate Frame

        Basically, the purpose of this function is to perform the inverse  of
        the transformation operation of "sensor CoF -> Reference frame"

        Arguments:
            config: transmation matrix composed of rotation matrix and
                    translation vector.

                    - shape: (4, 4)
                    - structure
                        r11 r12 r13 t1
                        r21 r22 r23 t2
                        r31 r32 r33 t3
                         0   0   0   1
                    
                    tx: component of the translation vector
                    rxx: component of the rotation matrix

            pointcloud: dataset composed of x, y, z coordinates of sensor
                        reading within the reference CoF

                        - shape: (n, 3)
                        - structure
                            X001 Y001 Z001
                            X002 Y002 Z002
                            X003 Y003 Z003
                            ...
                            .

            K: Matrix for mapping dataset from the camera's coordinate frame
               into its field of view

                - shape: (3, 3)
        """
        # The operation is the same as the one for transforming a pointcloud
        # into the reference coordinate frame.
        # The difference is that here, we use the inverse transformation matrix
        return cls.toRef(cls._inv(config), pointcloud)

    @classmethod
    def _inv(cls, config: np.array) -> np.array:
        """Compute the inverse transformation matrix.

        Gereral form of the inverse of a transformation matrix

                   __               __
                  |                   |
            [M] = | R^(-1)  -R^(-1)t  |
                  |   0         1     |
                  |__               __|

        Arguments:
            config: transmation matrix composed of rotation matrix and
                    translation vector.

                    - shape: (4, 4)
                    - structure
                        r11 r12 r13 t1
                        r21 r22 r23 t2
                        r31 r32 r33 t3
                         0   0   0   1
                    
                    tx: component of the translation vector
                    rxx: component of the rotation matrix
        """
        # The inverse of a rotation matrix equal it's transpose
        inv_rotation = np.transpose(config[0:3, 0:3])

        inv_translation = -1 * np.dot(inv_rotation, config[0:3, 3])

        # Add a row of zeros below the rotation matrix
        extended_inv_rotation = np.row_stack([inv_rotation, np.zeros((1, 3))])

        # Extend the vector with a new row of "1"
        extended_inv_translation = np.append(inv_translation, 1)

        # Return the matrix equivalent to the inverse of the transformation
        return np.column_stack(
            [extended_inv_rotation, extended_inv_translation]
        )


class Camera(object):
    """Camera.

    Process camera data
    """

    def __init__(self, filepath: str, calibration: np.array,
            cameraCoef: np.array) -> None:
        """Contructor of camera measurement data.
        
        Arguments:
            filepath: path of the camera image file
            calibration: Calibration matrix of the camera coordinate system
            cameraCoef: Matrix that maps an element from the reference Cof to
                        the camera's field of view
        """
        self.filepath: str = filepath
        self.calibration = calibration
        self.cameraCoef = cameraCoef
        self.image = image.imread(self.filepath)

    def plot(self, show: bool = True, **kwargs) -> None:
        """Plot the picture captured by the camera."""
        plt.imshow(self.image, zorder=1)
        if show:
            plt.show()


class Lidar(object):
    """Lidar.

    Process lidar data

    Attributes:
        filepath: path to the data file
        pointcloud: matrix of data contains all the measurement points
                    Row format: x, y, z, reflectivity
    """

    DATASET_DESCRIPTION_HEADER_ROW_LENGTH: int = 1

    def __init__(self, filepath: str, calibration: np.array) -> None:
        """Contructor of the lidar measurement data."""
        self.filepath: str = filepath
        self.calibration = calibration
        self.pointcloud: Optional[np.ndarray] = self._load()

    def _load(self, **kwargs) -> np.array:
        """Load data and convert it into the reference coordinate frame.

        Arguments:
            filepath: path to the file containing the data
            calibration: Transformation matrix for converting data points
                           into reference coordinate system.
        """
        data: Optional[np.array] = np.array(np.loadtxt(
            self.filepath,
            delimiter=" ",
            dtype=float,
            encoding="utf-8",
            skiprows=self.DATASET_DESCRIPTION_HEADER_ROW_LENGTH,
        ))
        return np.hstack((
            Calibration.toRef(self.calibration, data[:, :3]),
            data[:, 3:],
        ))

    def getPointCloud(self, **kwargs) -> np.array:
        """Get the x, y, z coordinate from the lidar dataset.

        Return
            Format:
                X001 Y001 Z001
                X002 Y002 Z002
                X003 Y003 Z003
                ...
            Shape: (n, 3)
        """
        return self.pointcloud[:, 0:3]

    def getPointCloudSample(self, x_filter: Optional[Tuple[float, float]] = None,
                            y_filter: Optional[Tuple[float, float]] = None,
                            z_filter: Optional[Tuple[float, float]] = None,
                            full: bool = False
                           ) -> np.array:
        """Extract a part of the pointcloud based on the filters.
        
        Arguments:
            x_filter: tuple containing the min and max values defining
                      the x-axis boundary
            y_filter: tuple containing the min and max values defining
                      the y-axis boundary
            z_filter: tuple containing the min and max values defining
                      the z-axis boundary
                      If z_filter is "None", the pointcloud is not filtered
                      in regard of the z-axis.
            full: Boolean flag. When it's "True" all the entries of the pointcould
                  are returned. If not, only the (x, y, z) columns of the dataset
                  are returned.

        Note: Each filtering tuple is compose of min value and max value (in
              that order)
        """
        if full:
            pointcloud = self.pointcloud
        else:
            pointcloud = self.getPointCloud()
        filtering_mask = True
        if x_filter:
            x_min, x_max = x_filter
            filtering_mask = filtering_mask & (pointcloud[:, 0] >= x_min)
            filtering_mask = filtering_mask & (pointcloud[:, 0] <= x_max)
        if y_filter:
            y_min, y_max = y_filter
            filtering_mask = filtering_mask & (pointcloud[:, 1] >= y_min)
            filtering_mask = filtering_mask &  (pointcloud[:, 1] <= y_max)
        if z_filter:
            z_min, z_max = z_filter
            filtering_mask = filtering_mask & (pointcloud[:, 2] >= z_min)
            filtering_mask = filtering_mask &  (pointcloud[:, 2] <= z_max)
        return pointcloud[filtering_mask]

    def getBirdEyeView(self, resolution: float,
                       srange: Tuple[float, float], # side range
                       frange: Tuple[float, float], # forward range
                       ) -> None:
        """Generate the bird eye view of the pointcloud.

        Arguments:
            resoluton: The pixel resolution of the image to generate
            srange: Side range to cover.
                Format: (srange_min, srange_max)
            frange: Forward range to cover.
                Format (frange_min, frange_max)

        Note: The ranges are expected to be provided the minimum first and then
        the maxnimum

                -----------------   <-- frange_max
                |               |
                |        ^      |
                |        |      |
                |    <-- 0      |
                |               |
                |               |
                -----------------   <-- frange_min
                ^               ^
            srange_min      srange_max
        """
        pointcloud = self.getPointCloudSample(frange, srange, z_filter=None, full=True)
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        ximg = (-y / resolution).astype(np.int32)
        yimg = (-x / resolution).astype(np.int32)

        ximg -= int(np.floor(srange[0] / resolution))
        yimg += int(np.floor(frange[1] / resolution))

        # Prepare the three channels of the bird eye view image
        pixels = np.zeros((len(z), 3), dtype=np.uint8)

        # Encode distance
        norm = np.sqrt(x **2 + y **2 + z**2)
        pixels[:, 0] = (255.0 / (1.0 + np.exp(-norm))).astype(np.uint8)

        # Encode height information
        pixels[:, 1] = (255.0 / (1.0 + np.exp(-z))).astype(np.uint8)

        # Encode intensity (for lidar) and radial velosity (for radar)
        pixels[:, 2] = pointcloud[:, 3]
        # Scale pixels between 0 and 2555
        minv = min(pointcloud[:, 3])
        maxv = max(pointcloud[:, 3])
        pixels[:, 2] = (
            ((pixels[:, 2] - minv) / np.abs(maxv - minv)) * 255
        ).astype(np.uint8)

        # Create the frame for the bird eye view
        # Note: the "+1" to estimate the width and height of the image is
        # to count for the (0, 0) position in the center of the pointcloud
        img_width: int = 1 + int((srange[1] - srange[0])/resolution)
        img_height: int = 1 + int((frange[1] - frange[0])/resolution)
        bev_img = np.zeros([img_height, img_width, 3], dtype=np.uint8)

        # Set the height information in the created image frame
        bev_img[yimg, ximg] = pixels
        return bev_img

    def plot(self, **kwargs) -> None:
        """Plot the lidar point cloud."""
        axes = plt.axes(projection="3d")
        plotting = axes.scatter(
            # x-axis
            self.pointcloud[:,0],
            # y-axis
            self.pointcloud[:, 1],
            # z-axis
            self.pointcloud[:, 2],
            # lazer ID
            c=self.pointcloud[:, 4],
            # Colormap
            cmap=plt.cm.get_cmap(),
        )
        plt.colorbar(plotting)
        plt.show()


class Radar(Lidar):
    """Radar.

    Process radar data
    """

    DATASET_DESCRIPTION_HEADER_ROW_LENGTH: int = 2

    def __init__(self, filepath: str, calibration: np.array) -> None:
        """Contructor of the radar measurement data."""
        self.filepath: str = filepath
        self.calibration = calibration
        self.pointcloud: Optional[np.ndarray] = self._load()


class Object(object):
    """Object.

    Represent an object marked in the groundtruth data of each entry
    """

    def __init__(self, **kwargs) -> None:
        """Constructor of ground truth object."""
        self.id = kwargs["id"]
        self.name = kwargs["classname"]
        self.created_by = kwargs["created_by"]
        self.centerX, self.centerY, self.centerZ = kwargs["center3d"]
        self.length, self.width, self.height = kwargs["dimension3d"]
        self.orientation: List[float] = kwargs["orientation_quat"]
        self.occlusion = kwargs["occlusion"]
        self.label_certainty = kwargs["label_certainty"]
        self.score = kwargs["score"]
        if kwargs.get("lidar"):
            self.lidar = self.extractFromPointcloud(kwargs["lidar"])
        if kwargs.get("radar"):
            self.radar = self.extractFromPointcloud(kwargs["radar"])
        if kwargs.get("camera"):
            self.camera = kwargs["camera"]

    def getQuaternion(self) -> np.array:
        """Return the quaternion-based orientation of the object."""
        return np.asarray(self.orientation)

    def getEulerAngles(self) -> Tuple[float, float, float]:
        """Return the Euler angles indicating the orientation of the object."""
        # components of the orientation quaternion
        q0, q1, q2, q3 = self.orientation

        roll: float = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2 * (q1*q1 + q2*q2))
        pitch: float = np.arcsin(2*(q0*q2 + q1*q3))
        yaw: float = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2 * (q2*q2 + q3*q3))
        return (roll, pitch, yaw)

    def _getUnitQuaternion(self) -> np.matrix:
        """Return the unit quaternion."""
        quaternion = self.getQuaternion()
        quaternion_norm = np.sqrt(np.dot(quaternion, quaternion))
        # Normalized quaternion
        return quaternion / quaternion_norm

    def getRotationMatrix(self) -> np.array:
        """Return the rotation matrix based on the quaternion orentation."""
        q0, q1, q2, q3 = self._getUnitQuaternion()
        r11: float = 1 - 2*(q2*q2 + q3*q3)
        r12: float = 2*(q1*q2 - q0*q3)
        r13: float = 2*(q1*q3 + q0*q2)

        r21: float = 2*(q1*q2 + q0*q3)
        r22: float = 1 - 2*(q1*q1 + q3*q3)
        r23: float = 2*(q2*q3 - q0*q1)

        r31: float = 2*(q1*q3 - q0*q2)
        r32: float = 2*(q2*q3 + q0*q1)
        r33: float = 1 - 2*(q1*q1 + q2*q2)
        return np.array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33],
        ])

    @property
    def center(self) -> np.array:
        """Return the coordinate of the center of the object's bounding box."""
        return np.array([self.centerX, self.centerY, self.centerZ])

    @property
    def distance(self) -> float:
        """Return the distance of the object from the "ego" vehicle in meter."""
        return np.sqrt(np.sum(np.square(self.center)))

    @property
    def angle(self) -> float:
        """Return the yaw angle of the object in degree."""
        roll, pitch, yaw = self.getEulerAngles()
        yaw_degree = yaw * 180 / np.pi
        return yaw_degree if (yaw_degree > 0) else (yaw_degree + 360)

    def getBoundingBox(self):
        """Get bounding box.

        Return the coordinates of the corners of the bounding box

                      z
                      |  /x
                      | /
                y_____|/

                c1__________ c2               
                /|         /|
             c4/_|________/ |     
               | /--------|-/ c6
             c8|/_________|/ c7
                <--------->
                  w: width

            cx: corner order number

        Notations:
            centerX: x-coordinate of the center of the box in reference CoF
            centerY: y-coordinate of the center of the box in reference CoF
            centerZ: z-coordinate of the center of the box in reference CoF

            width: width of the box
            height: height of the box
            length: length of the 3D box
        
        Method to compute corners coordinates:
            corner 1 (c1):
                x: CenterX + length/2
                y: CenterY + width/2
                z: CenterZ + height/2
            corner 2 (c2):
                x: CenterX + length/2
                y: CenterY - width/2
                z: CenterZ + height/2
            corner 3 (c3):
                x: CenterX - length/2
                y: CenterY - width/2
                z: CenterZ + height/2
            corner 4 (c4):
                x: CenterX - length/2
                y: CenterY + width/2
                z: CenterZ + height/2

            corner 5 (c5):
                x: CenterX + length/2
                y: CenterY + width/2
                z: CenterZ - height/2
            corner 6 (c6):
                x: CenterX + length/2
                y: CenterY - width/2
                z: CenterZ - height/2
            corner 7 (c7):
                x: CenterX - length/2
                y: CenterY - width/2
                z: CenterZ - height/2
            corner 8 (c8):
                x: CenterX - length/2
                y: CenterY + width/2
                z: CenterZ - height/2

        Return: array of corners coordinates
                c1.x, c1.y, c1.z
                c2.x, c2.y, c2.z
                c3.x, c3.y, c3.z
                ...
        """
        corners_coordinates = np.array([
            [self.length/2, self.width/2, self.height/2],       # c1
            [self.length/2, -self.width/2, self.height/2],      # c2
            [-self.length/2, -self.width/2, self.height/2],     # c3
            [-self.length/2, self.width/2, self.height/2],      # c4
            [self.length/2, self.width/2, -self.height/2],      # c5
            [self.length/2, -self.width/2, -self.height/2],     # c6
            [-self.length/2, -self.width/2, -self.height/2],    # c7
            [-self.length/2, self.width/2, -self.height/2],     # c8
        ])
        # Rotate the corners coordinates into the reference coordinate frame
        corners_coordinates = np.dot(
            self.getRotationMatrix(),
            np.transpose(corners_coordinates),
        )
        # Add the coordinates of the center of the box in order to move it to the
        # appropriate coornidate in the reference coordinate frame
        # NOTE: This assumes that the coordinates of the center of the bounding boxes
        #       have been provided in the reference frame by the dataset.
        corners_coordinates = np.transpose(corners_coordinates) + self.center
        return corners_coordinates

    def extractFromPointcloud(self, sensor: Union[Lidar, Radar]):
        """Get the lidar points that covers the object.
        
        Argument:
            sensor: object representing either lidar or radar model
        
        Note: The "sensor" object must implement the method
              "getPointCloudSample"
        """
        bounding_box = self.getBoundingBox()
        # return the max value of each colunm
        x_max, y_max, z_max = bounding_box.max(0)
        # return the min value of each colunm
        x_min, y_min, z_min = bounding_box.min(0)
        return sensor.getPointCloudSample(
            (x_min, x_max),
            (y_min, y_max),
            (z_min, z_max),
        )

    def view(self, is_lidar: bool = True) -> None:
        """Plot some data point over the camera image.
        
        It's a projection of the lidar/radar data onto the camera data.
        """
        data = self.lidar if is_lidar else self.radar
        self.camera.plot(show=False)
        pointcloud = Calibration.fromRef(
            self.camera.calibration,
            data,
        )
        pointcloud = np.dot(
            self.camera.cameraCoef,
            np.transpose(pointcloud[:, :3]),
        )
        pointcloud = pointcloud / pointcloud[2]
        pointcloud = np.transpose(pointcloud)
        plt.scatter(
            [pointcloud[:, 0]],
            [pointcloud[:, 1]],
            c="b",
            alpha=0.8,
            s=4,
            zorder=2,
        )
        plt.show()

    def __str__(self) -> str:
        """String representation of an Object instance."""
        return f"""
        id              : {self.id}
        name            : {self.name}
        lidar-points    : {len(self.lidar)}
        radar-points    : {len(self.radar)}
        distance        : {self.distance:.3f} m
        yaw-angle       : {self.angle:.3f} deg
        Dimensions
            length      : {self.length:.3f} m
            width       : {self.width:.3f} m
            height      : {self.height:.3f} m
        """


class GroundTruth(object):
    """Ground truth.

    Process ground truth data

    It's possible to feed this class with the lidar and radar class objects
    as keyword arguments.
    """

    def __init__(self, filepath, **kwargs) -> None:
        """Contructor of the ground truth class representation."""
        self.lidar: Optional[Lidar] = kwargs.get("lidar")
        self.radar: Optional[Radar] = kwargs.get("radar")
        self.camera: Optional[Camera] = kwargs.get("camera")
        with open(filepath, "r") as file_handler:
            self.objects: List[Object] = [
                Object(
                    **item,
                    id=index,
                    lidar=self.lidar,
                    radar=self.radar,
                    camera=self.camera,
                )
                for (index, item) in enumerate(load(file_handler)["objects"])
            ]

    def all(self) -> List[Object]:
        """Return all the object present in the current entry."""
        return self.objects


class Data(object):
    """Data.

    Single entry in the dataset
    """

    def __init__(self, **kwargs) -> None:
        """Build a data entry of the dataset."""
        self.calibration = Calibration(kwargs["calibration"])
        self.camera = Camera(
            kwargs["camera"],
            self.calibration.camera,
            self.calibration.cameraCoef,
        )
        self.lidar = Lidar(kwargs["lidar"], self.calibration.lidar)
        self.radar = Radar(kwargs["radar"], self.calibration.radar)
        self.groundtruth = GroundTruth(
            kwargs["groundtruth"],
            lidar=self.lidar,
            radar=self.radar,
            camera=self.camera,
        )

    def __str__(self) -> str:
        """String representation of the data."""
        pass


class AstyxDataset(object):
    """Astyx Dataset."""

    def __init__(self, dataset: str) -> None:
        """Constructor of the dataset.

        Arguments:
            dataset: path to the dataset
        """
        # Channels represent the type of information available in the dataset
        # It's an interface for the dataset internal naming convention
        # in order to have user-friendly channel names
        self.channels: Dict[str, str] = {
            "calibration": "calibration",
            "camera": "camera_front",
            "groundtruth": "groundtruth_obj3d",
            "lidar": "lidar_vlp16",
            "radar": "radar_6455",
        }
        # Entry point of the dataset
        self.dataset: str = dataset
        # Store of the dataset
        self.data: Dict[str, Dict[str, str]] = {}

        with open(self.dataset, "r") as file_handler:
            self.data = load(file_handler)["data"]

    def overview(self) -> str:
        """Return a json string formated overview of the dataset."""
        return dumps(self.data, indent=2)

    def get_channels(self) -> Tuple[str, str, str, str, str]:
        """Return the channels available on each item of the dataset.

        By channel we should understand: type of information.
        Mainly here the channels are:
            - calibration
            - camera
            - groundtruth
            - lidar
            - radar
        """
        return tuple(self.channels.keys())

    def __len__(self) -> int:
        """Return the size (number of data entry) of the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Return a data object.
        
        Argument:
            index: irderorder number of a data within the dataset
        """
        store: Dict[str, str] = {
            "calibration": DATASET_ROOT_DIR + "/" + (
                self.data[str(index)][self.channels["calibration"]]
            ),
            "camera": DATASET_ROOT_DIR + "/" + (
                self.data[str(index)][self.channels["camera"]]
            ),
            "lidar": DATASET_ROOT_DIR + "/" + (
                self.data[str(index)][self.channels["lidar"]]
            ),
            "radar": DATASET_ROOT_DIR + "/" + (
                self.data[str(index)][self.channels["radar"]]
            ),
            "groundtruth": DATASET_ROOT_DIR + "/" + (
                self.data[str(index)][self.channels["groundtruth"]]
            ),
        }
        return Data(**store)

    def __str__(self) -> str:
        """String representation of the class."""
        return f"AstyxDataset{self.get_channels()}, lenght: {len(self)}"

    def show(self) -> None:
        """Render all pending graph."""
        plt.show()


# Single instance of the Dataset class that must be used to interact with
# the data store
dataset = AstyxDataset(DATASET)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Astyx Dataset api",
        description="Interface to interact with the astyx dataset",
    )
    parser.add_argument(
        "-i", "--index",
        type=int,
        help="Read a specific dataset",
    )
    parser.add_argument(
        "-v", "--version",
        help="Print the size of the dataset",
        action="store_true",
    )
    parser.add_argument(
        "--overview",
        help="Print the general structure of the dataset",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--groundtruth",
        help="Print the Ground thruth information",
        action="store_true",
    )
    parser.add_argument(
        "--view",
        help="Trigger a graphical view of a ground truth object",
        type=int,
    )
    parser.add_argument(
        "--camera",
        help="Render the camera view of the an entry of the dataset",
        action="store_true",
    )
    parser.add_argument(
        "--lidar",
        help="Render the lidar pointcloud",
        action="store_true",
    )
    parser.add_argument(
        "--bird-eye-view",
        "-bev",
        help="Request a bird eye view rendering",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--resolution",
        "-res",
        help="Bird eye view resolution",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--width",
        help="Bird eye view image width",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--height",
        help="Bird eye view image height",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--radar",
        help="Render the radar pointcloud",
        action="store_true",
    )
    subparsers = parser.add_subparsers()
    filtering_subparser = subparsers.add_parser("filter")
    filtering_subparser.add_argument(
        "-d", "--distance",
        type=float,
        help="Print the Ground thruth information",
    )
    filtering_subparser.add_argument(
        "-a", "--angle",
        type=float,
        help="Print the Ground thruth information",
    )
    filtering_subparser.add_argument(
        "-m", "--max",
        type=float,
        help="Maximum number of result",
        default=10,
    )

    args = parser.parse_args()

    if args.version:
        print(dataset)

    elif args.overview:
        print(dataset.overview())
        sys.exit(0)

    elif args.index and args.groundtruth:
        data: Data = dataset[args.index]
        if args.view is not None:
            if 0 <= args.view < len(data.groundtruth.objects):
                groundtruth_object = data.groundtruth.objects[args.view]
                if args.lidar:
                    groundtruth_object.view()
                elif args.radar:
                    groundtruth_object.view(is_lidar=False)
            else:
                print("Invalid ground truth object id")
        else:
            for groundtruth_object in data.groundtruth.objects:
                print(groundtruth_object)

    elif args.index and args.lidar:
        data: Data = dataset[args.index]
        if args.bird_eye_view:
            # render bird eye view
            bev = data.lidar.getBirdEyeView(
                args.resolution,
                (-args.width/2, args.width/2),
                (-args.height/2, args.height/2),
            )
            plt.imshow(bev)
            plt.show()
            sys.exit(0)
        data.lidar.plot()

    elif args.index and args.radar:
        data: Data = dataset[args.index]
        if args.bird_eye_view:
            # render bird eye view
            bev = data.radar.getBirdEyeView(
                args.resolution,
                (-args.width/2, args.width/2),
                (0, args.height/2),
            )
            plt.imshow(bev)
            plt.show()
            sys.exit(0)
        data.radar.plot()

    elif args.index and args.camera:
        data: Data = dataset[args.index]
        data.camera.plot()

    elif hasattr(args, "distance") or hasattr(args, "angle"):
        DISTANCE_MARGING: float = 1
        ANGLE_MARGING: float = 1
        number_of_results: int = 0
        for index in range(len(dataset)):
            data: Data = dataset[index]
            if (number_of_results >= args.max):
                break
            for groundtruth_object in data.groundtruth.objects:
                if (number_of_results >= args.max):
                    break
                criteria = True
                if args.distance:
                    criteria = criteria and (args.distance - DISTANCE_MARGING) <= (
                        groundtruth_object.distance) <= (
                            args.distance + DISTANCE_MARGING)
                if args.angle:
                    criteria = criteria and ((args.angle - ANGLE_MARGING) <= (
                        groundtruth_object.angle) <= (
                            args.angle + ANGLE_MARGING))
                if criteria:
                    print(f"    Index: {index}{groundtruth_object}")
                    number_of_results += 1
    else:
        parser.print_help()
