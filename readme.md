# Astyx

Astyx is an API to access the Astyx Hires 2019 dataset.

## Requirements

1. Create a virtual environment

```bash
python -m venv venv

source venv/bin/activate
```

2. Install the required packages in the virtual environment

```bash
python -m pip install -r requirements.txt
```

## Dataset

1. Download the dataset at:
https://www.astyx.com/development/astyx-hires2019-dataset.html


2. Unzip the downloaded file and copy the dataset in this folder

NOTE: The structure of the folder must be similar to the following

<pre>
dataset_astyx_hires2019
  ├── calibration
  ├── camera_front
  ├── dataset.json
  ├── groundtruth_obj3d
  ├── lidar_vlp16
  └── radar_6455
</pre>

Make sure that the `dataset.json` is at the root of the dataset directory.
The library uses that file to load the dataset.

## CLI interface usage

The Astyx Hires dataset has 546 entries (from `0` to `545`). Each entry has
lidar, radar, camera, calibration data as well as ground truth data.

Just lead by examples here

0. Overview of the dataset

```bash
python dataset.py --overview
```


1. Plot the lidar point cloud of a given entry

```bash
python dataset.py -i <index> --lidar
```

```bash
python dataset.py -i 2 --lidar
```

2. Plot the radar point cloud of a given entry

```bash
python dataset.py -i <index> --radar
```

```bash
python dataset.py -i 56 --radar
```

3. View the camera image of a given entry

```bash
python dataset.py -i <index> --camera
```

```bash
python dataset.py -i 43 --camera
```

4. Read all the ground truth data of a given entry of the dataset

```bash
python dataset.py -i <index> --groundtruth
```

```bash
python dataset.py -i 2 --groundtruth
```

Example of output expected
<pre>
        id              : 0
        name            : Car
        lidar-points    : 20
        radar-points    : 7
        distance        : 18.520 m
        yaw-angle       : 264.847 deg
        Dimensions
            length      : 4.040 m
            width       : 1.790 m
            height      : 1.390 m
</pre>

The `id` field of the result is the order in which the ground truth object
appears in the dataset.

5. View Camera + Lidar data for a given ground truth object.

```bash
python dataset.py -i <index> --groundtruth --view <object-id> --lidar
```

```bash
python dataset.py -i 127 --groundtruth --view 2 --lidar
```

NOTE: It's recommended to list all the available groundtruth object of an entry
first (see 4.) in order to provide a valid `object-id`.

6. View Camera + Radar data for a given ground truth object.

```bash
python dataset.py -i <index> --groundtruth --view <object-id> --radar
```

```bash
python dataset.py -i 127 --groundtruth --view 2 --radar
```

7. Search for a ground truth object that match some criteria

```bash
# <d> in meter
# <a> in degree
python dataset.py filter --distance <d> --angle <a> --max <number-objects>
```

```bash
# distance: 30 meters
# angle: 20°
python dataset.py filter --distance 30 --angle 20 --max 2
```

```bash
python dataset.py filter --distance 10

python dataset.py filter --angle 1
```

Both filtering criteria are not mandatory. At leat one must be provided. If the
`max` argument is not provided, `10` result will be displayed.

NOTE: The search starts from the entry `0` and stops when the `max` number of
objects have been found. This operation is quite slow as for each entry some
data need to be loaded along with some computation.

8. Bird eye view of lidar/radar pointcloud

```bash
# Lidar pointcloud
python dataset.py -i <index> --lidar -bev [--resolution <res> --width <width> --height <height>]

# Radar pointcloud
python dataset.py -i <index> --radar -bev [--resolution <res> --width <width> --height <height>]
```

Resolution, width and height parameters are optional. The default values are:
- Resolution: `0.05` (`5cm/pixel`)
- Width: `80` (meter)
- Height: `80` (meter)

The RGB channels of each pixel in the bird-eye-view are encoded as follow:
- R (red channel): Distance informatiom of each data point in the pointcloud
- G (green channel): Height information (z-axis)
- B (blue channel): Intensity of reflection for lidar pointcloud and radial velocity for radar pointcloud

```bash
# Bird eye view output of the lidar point cloud from the dataset entry 146
python dataset.py -i 146 --lidar -bev

# Bird eye view output of the radar point cloud from the dataset entry 146
python dataset.py -i 146 --radar -bev
```

## References

- https://github.com/BerensRWU/Complex_YOLO
- https://www.astyx.com/fileadmin/redakteur/dokumente/Astyx_Dataset_HiRes2019_specification.pdf
