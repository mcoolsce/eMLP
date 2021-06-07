#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

struct CellInfo {
    float** gvecs;
    float* spacings;
};

template<typename T>
T** newMatrix(int N, int M);
CellInfo getCellInfo(Tensor rvecs);
int wrap_bin(int neighbour, int nbin);
std::array<float, 2> getCellExtrema(float** tfracs, int N);
