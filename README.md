# eMLP
The electron machine learning potential package


# Installation instructions
**Requirements:**

 - Tensorflow 2.7 or newer
 - [yaff](https://github.com/molmod/yaff)
 - [molmod](https://github.com/molmod/molmod)
 - h5py

**Before installing the package**, execute the following bash script

    bash compile_op.sh

to compile the `cell_list_op.so` custom tensorflow op. Next, install the package in the usual way. For instance, run

    pip install .

in the cloned directory.


# Examples
Templates of scripts to train and use the model are located in the `emlp/examples/` directory.

