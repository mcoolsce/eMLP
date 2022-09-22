# eMLP
The electron machine learning potential package.


# Installation instructions
**Requirements:**

 - Tensorflow 2.10 or newer
 - [yaff](https://github.com/molmod/yaff)
 - [molmod](https://github.com/molmod/molmod)
 - h5py

**Before installing the package**, execute the following bash script

    bash compile_op.sh

to compile the `cell_list_op.so` custom tensorflow op. Next, install the package in the usual way. For instance, run

    pip install .

in the cloned directory.


# Examples
Templates of scripts and jupyter notebooks to train and use the model are located in the `emlp/examples/` directory.


# Reference
The eMLP has been published in the following paper: 
 - [https://doi.org/10.1021/acs.jctc.1c00978](https://doi.org/10.1021/acs.jctc.1c00978)

