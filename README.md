# HDF5 to PLY Gaussian Splat Convertor
This project provides functions to convert gaussian splats stored in HDF5 format to PLY format. 


## Installation
You can install the required dependencies using pip:
```
pip install h5_to_ply
```


## Usage
```python
h5file_path = "/path/to/model.h5"
plyfile_path = "/path/to/model.ply"
in_buffer: BytesIO = BytesIO()
out_buffer: BytesIO = BytesIO()

with open(h5file_path, "rb") as f:
    in_buffer = BytesIO(f.read())

with open(plyfile_path, "wb") as f:
    convert_h5_to_ply(in_buffer, out_buffer)
    out_buffer.seek(0)
    f.write(out_buffer.getbuffer())
```