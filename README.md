# Open Azure Kinect [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/open-azure-kinect)](https://pypi.org/project/open-azure-kinect/)

Open playback functions for Azure Kinect.

It is possible to playback [Azure Kinect](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) videos files (mkv) without using the official SDK. This allows the software to be used on systems where the depth engine is not implemented, such as MacOS. The library currently **only** supports the **playback** of mkv files and does **not provide direct access** to the Azure Kinect device.

The following functions are currently supported:

- Reading colour, infrared and depth stream from mkv
- Reading calibration data from mkv

## Installation

```terminal
pip install open-azure-kinect
```

## Usage
In order to load an MKV file, it is necessary to create a new instance of the `OpenK4APlayback` class.

```python
from openk4a.playback import OpenK4APlayback

azure = OpenK4APlayback("my-file.mkv")
azure.open()
```

After that, it is possible to read the available stream information.

```python
for stream in azure.streams:
    print(stream)
```

And read the actual capture information (image data).

```python
while capture := azure.read():
    # read color frame as numpy array
    color_image = capture.color
```

## Development and Examples
To run the examples or develop the library please install the `dev-requirements.txt` and `requirements.txt`.

```terminal
pip install -r dev-requirements.txt
pip install -r requirements.txt
```

There is already an example script [demo.py](demo.py) which provides insights in how to use the library.