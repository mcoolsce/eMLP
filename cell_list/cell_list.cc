#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "cell_list.h"
#include <math.h>
#include <tuple>
#include <list>
#include <map>

using namespace tensorflow;

REGISTER_OP("CellList")
    .Input("atcarts: float32")
    .Input("rvecs: float32")
    .Input("rcut: float32")
    .Output("pairlist: int32")
    .Doc(R"doc(3D periodic neighbor search using cell lists, in c++.)doc");

template<typename T>
T** newMatrix(int N, int M){
	 T** matrix = new T*[N];
     for (int i = 0; i < N; i++){
          matrix[i] = new T[M];
     } 

	 return matrix;
}


CellInfo getCellInfo(Tensor rvecs_tensor){
     // Getting the rvecs data and making a new cellinfo struct
     auto m = rvecs_tensor.matrix<float>();

     // Inverting the matrix
     float det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) -
             m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
             m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

     float invdet = 1.0 / det;

     // An empty 3 by 3 matrix
     CellInfo cellinfo;
     cellinfo.gvecs = newMatrix<float>(3, 3);

     cellinfo.gvecs[0][0] = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * invdet;
     cellinfo.gvecs[0][1] = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet;
     cellinfo.gvecs[0][2] = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
     cellinfo.gvecs[1][0] = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
     cellinfo.gvecs[1][1] = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
     cellinfo.gvecs[1][2] = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
     cellinfo.gvecs[2][0] = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
     cellinfo.gvecs[2][1] = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet;
     cellinfo.gvecs[2][2] = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet;

     cellinfo.spacings = new float[3];

     for (int i = 0; i < 3; i++){
         cellinfo.spacings[i] = 1.0 / sqrt(cellinfo.gvecs[0][i] * cellinfo.gvecs[0][i] + cellinfo.gvecs[1][i] * cellinfo.gvecs[1][i] + cellinfo.gvecs[2][i] * cellinfo.gvecs[2][i]);
     }

     return cellinfo; 
}

std::array<float, 2> getCellExtrema(float** atfracs, int N){
	float maximum = atfracs[0][0];
	float minimum = atfracs[0][0];

	for (int i = 0; i < N; i++){
		for (int j = 0; j < 3; j++){
			if (atfracs[i][j] > maximum){
				maximum = atfracs[i][j];
			}

			if (atfracs[i][j] < minimum){
				minimum = atfracs[i][j];
			}
		}
	}
	
	std::array<float, 2> extrema = {minimum, maximum};
	
	return extrema;
}

int wrap_bin(int neighbour, int nbin){
	while (neighbour < 0){
		neighbour += nbin;
	}
	return (neighbour % nbin);
}

class CellListOp : public OpKernel {
 public:
  explicit CellListOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& atcarts_tensor = context->input(0);
    const TensorShape& atcarts_shape = atcarts_tensor.shape();
    
    // atcarts should be a ? by 3 matrix
    OP_REQUIRES(context, atcarts_shape.dims() == 2, errors::InvalidArgument("Input 'atcarts' is not a rank 2 tensor. A rank ", atcarts_shape.dims(), " was found."));
    OP_REQUIRES(context, atcarts_shape.dim_size(1) == 3, errors::InvalidArgument("Input 'atcarts' dim 1 size is ", atcarts_shape.dim_size(1), ". A dim size 3 is required."));

    const Tensor& rvecs_tensor = context->input(1);
    const TensorShape& rvecs_shape = rvecs_tensor.shape();

    // Check if rvec is a 3 by 3 matrix
    OP_REQUIRES(context, rvecs_shape.dims() == 2, errors::InvalidArgument("Input 'rvecs' is not a rank 2 tensor. A rank ", rvecs_shape.dims(), " was found."));
    OP_REQUIRES(context, rvecs_shape.dim_size(0) == 3, errors::InvalidArgument("Input 'rvecs' dim 0 size is ", rvecs_shape.dim_size(0), ". A dim size 3 is required."));
    OP_REQUIRES(context, rvecs_shape.dim_size(1) == 3, errors::InvalidArgument("Input 'rvecs' dim 1 size is ", rvecs_shape.dim_size(1), ". A dim size 3 is required."));
    //auto rvecs = rvecs_tensor.matrix<float>();

    const Tensor& rcut_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(rcut_tensor.shape()), errors::InvalidArgument("Input 'rcut' is not a scalar."));
    float rcut = rcut_tensor.scalar<float>()(0);
    
    // The number of atoms
    int N = atcarts_shape.dim_size(0);

    // Getting the cell properties
    CellInfo cellinfo = getCellInfo(rvecs_tensor);
    
    auto rvecs_matrix = rvecs_tensor.matrix<float>();
    
    // Calculating the number of bins    
  	int* nbins = new int[3];
  	int* nmirrors = new int[3];
	for (int i = 0; i < 3; i++){
		nmirrors[i] = static_cast<int>(ceil(rcut / cellinfo.spacings[i])); // number of periodic images when the cutoff radius is larger than the cell spacings
		nbins[i] = static_cast<int>(floor(cellinfo.spacings[i] * nmirrors[i] / rcut));
	}

    auto atcarts_input = atcarts_tensor.matrix<float>();

    // atfracs = dot(atcarts, gvecs)
    float** atfracs = newMatrix<float>(N, 3);
    int** wrapped_addfrac = newMatrix<int>(N, 3); // A matrix containing the addfracs due to input positions outside the unitcell
    float unwrapped;
    
	for (int i = 0; i < N; i++){
		for (int j = 0; j < 3; j++){
			float sum = 0.0;
			for (int k = 0; k < 3; k++){
				sum += atcarts_input(i, k) * cellinfo.gvecs[k][j];
			}
			
			unwrapped = fmod(sum, 1.0);
            
            // Underflow will result in an atfrac of 1 if the fractional coordinate before wrapping is -epsilon (epsilon very small)
            atfracs[i][j] = (unwrapped < 0 ? unwrapped + 1.0 : unwrapped); // wrap this inside the cell (original atfracs[i][j] = sum;)
            wrapped_addfrac[i][j] = static_cast<int>(floor(sum));
		}
	}

	int** atbins = new int*[N];
	for (int i = 0; i < N; i++){
		atbins[i] = new int[3];
		for (int j = 0; j < 3; j++){
			// atbins[i][j] = static_cast<int>(floor(atfracs[i][j] * nbins[j])); Original code
			int bin_number = static_cast<int>(floor(atfracs[i][j] * nbins[j]));
			atbins[i][j] = (bin_number == nbins[j] ? nbins[j] - 1 : bin_number);
			
		}
	}

	// A dictionary containing atom indices, keys are a bin tuple
	std::map<std::tuple<int, int, int>, std::list<int>> atmap;
	for (int i = 0; i < N; i++){
		std::tuple<int, int, int> bin_tuple(atbins[i][0],  atbins[i][1], atbins[i][2]);
		if (atmap.find(bin_tuple) == atmap.end()){ // Adding a new element
			atmap[bin_tuple] = std::list<int> (1, i);
		} else { // Appending the atom index
			atmap[bin_tuple].push_back(i);
		}
	}

	int incs[14][3] = {{-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1}, {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1}, {0, -1, -1}, {0, -1, 0}, {0, -1, 1}, {0, 0, -1}, {0, 0, 0}};
	
	// Making the temporary data container
	std::list<std::array<int, 5>> list_output;
	// Making the max neighbor list
	int* num_neighbors = new int[N];
	memset(num_neighbors, 0, N * sizeof(int)); // initializes to zero
	
	int max_neighbors = 0;
	
	// Loop over all bins in the atmap
	for (auto const& element : atmap){
		std::tuple<int, int, int> my_bin = element.first;
		// Loop over all the neighbouring bins
		for (int i = 0; i < 14; i++){
			int category;
			if (i < 9){
				category = 0; // The left plane
			} else if (i < 12){
				category = 1; // Middle plane in front
			} else if (i == 12){
				category = 2; // Central bottom cube
			} else {
				category = 3; // Middle cube
			}
			
			std::tuple<int, int, int> neighbour_bin(std::get<0>(my_bin) + incs[i][0], std::get<1>(my_bin) + incs[i][1], std::get<2>(my_bin) + incs[i][2]); // the neighbouring tuple
			
			// Add integer relative vector in fractional coordinates
			std::array<int, 3> addfrac = {static_cast<int>(floor((float) std::get<0>(neighbour_bin) / nbins[0] * nmirrors[0])), 
										  static_cast<int>(floor((float) std::get<1>(neighbour_bin) / nbins[1] * nmirrors[1])), 
										  static_cast<int>(floor((float) std::get<2>(neighbour_bin) / nbins[2] * nmirrors[2]))};
			
			// wrapping the bin indices in [0, nbins[
			std::get<0>(neighbour_bin) = wrap_bin(std::get<0>(neighbour_bin), nbins[0]);
			std::get<1>(neighbour_bin) = wrap_bin(std::get<1>(neighbour_bin), nbins[1]);
			std::get<2>(neighbour_bin) = wrap_bin(std::get<2>(neighbour_bin), nbins[2]);
			
			// Looping over all pairs
			if (atmap.find(neighbour_bin) != atmap.end()){ // both cells contain atoms
				for (auto const& iatom_a : element.second){
					for (auto const& iatom_b : atmap[neighbour_bin]){
						for (int mirrorx = 0; mirrorx < nmirrors[0]; mirrorx++){
						for (int mirrory = 0; mirrory < nmirrors[1]; mirrory++){
						for (int mirrorz = 0; mirrorz < nmirrors[2]; mirrorz++){
							if ((category == 0) || ((category == 1) && (mirrorx == 0)) || ((category == 2) && (mirrorx == 0) && (mirrory == 0)) || 
							   ((category == 3) && (mirrorx == 0) && (mirrory == 0) && (mirrorz == 0) && (iatom_a > iatom_b))){ // Only count the pairs once
						       // (OPTIONAL, CHECK PERFORMANCE): Filter out the neighbors who are situated outside the cutoff radius
						       int rel_vector_cell[3] = {addfrac[0] + wrapped_addfrac[iatom_a][0] - wrapped_addfrac[iatom_b][0] + mirrorx,
												   	     addfrac[1] + wrapped_addfrac[iatom_a][1] - wrapped_addfrac[iatom_b][1] + mirrory, 
												   	     addfrac[2] + wrapped_addfrac[iatom_a][2] - wrapped_addfrac[iatom_b][2] + mirrorz};
							   
							   // relative vector		   	
							   float rel_vector[3] = {atcarts_input(iatom_b, 0) - atcarts_input(iatom_a, 0), atcarts_input(iatom_b, 1) - atcarts_input(iatom_a, 1), atcarts_input(iatom_b, 2) - atcarts_input(iatom_a, 2)};
							   
							   for (int k = 0; k < 3; k++){
							       for (int l = 0; l < 3; l++){
							            rel_vector[k] += rvecs_matrix(l, k) * rel_vector_cell[l];
							       }    
							   }
							   
							   if (rel_vector[0] * rel_vector[0] + rel_vector[1] * rel_vector[1] + rel_vector[2] * rel_vector[2] <= rcut * rcut){
							       // Keep track of how many neighbors each atom has to assign the correct tensor shape
							       if (max_neighbors == num_neighbors[iatom_a]++){
							           max_neighbors++;
							       };
							       if (max_neighbors == num_neighbors[iatom_b]++){
							           max_neighbors++;
							       };
							       
							       // Fill the pairlist 
							       list_output.push_back({rel_vector_cell[0], rel_vector_cell[1], rel_vector_cell[2], iatom_a, iatom_b}); // The relative distance vector points from atom a to atom b!
							   }          
							}
						}}}
					}	
				}
			}
		}
	}
	
	// Create the output_shape of pairlist_indices
    TensorShape output_shape;
    output_shape.AddDim(N);
    output_shape.AddDim(max_neighbors);
    output_shape.AddDim(4);

    // Create an output tensor of neighbor indices and relative distances
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    
    auto output = output_tensor->tensor<int, 3>();
    
    // Copying the c++ list into the output of the right size
    int* indexes = new int[N];
	memset(indexes, 0, N * sizeof(int)); // initializes to zero
	
    int my_index;
	for (auto const& pair : list_output){
	    // centered at atom a
	    my_index = indexes[pair[3]];
	    
		output(pair[3], my_index, 0) = pair[4]; // the index
		output(pair[3], my_index, 1) = pair[0]; // the relative cell vectors
		output(pair[3], my_index, 2) = pair[1];
		output(pair[3], my_index, 3) = pair[2];
		
		indexes[pair[3]]++;
		
		// centered at atom b
	
		my_index = indexes[pair[4]];
	    
		output(pair[4], my_index, 0) = pair[3]; // the index
		output(pair[4], my_index, 1) = -pair[0]; // the relative cell vectors
		output(pair[4], my_index, 2) = -pair[1];
		output(pair[4], my_index, 3) = -pair[2];
		
		indexes[pair[4]]++;
	}
	
	// Put the other non assigned pair indices to -1 to be maskable
	for (int i = 0; i < N; i++){ // Atom loop
	    for (int j = indexes[i]; j < max_neighbors; j++){
	        output(i, j, 0) = -1;
	        output(i, j, 1) = -1;
	        output(i, j, 2) = -1;
	        output(i, j, 3) = -1;
	        // It is not necessary to put the other entries of dim 3 to -1 as they have to be masked away!
	    }
	}
	
	// Deleting the allocated memory
	
	delete[] cellinfo.spacings;
	delete[] num_neighbors;
	delete[] indexes;
	
	for (int i = 0; i < 3; i++){
		delete[] cellinfo.gvecs[i];
	}
	
	delete[] cellinfo.gvecs;
	delete[] nbins;
	delete[] nmirrors;
	
	for (int i = 0; i < N; i++){
		delete[] atbins[i];
		delete[] atfracs[i];
		delete[] wrapped_addfrac[i];
	}
	
	delete[] atbins;
	delete[] atfracs;
	delete[] wrapped_addfrac;
  }
};

REGISTER_KERNEL_BUILDER(Name("CellList").Device(DEVICE_CPU), CellListOp);
