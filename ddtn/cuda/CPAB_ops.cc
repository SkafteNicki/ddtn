#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "iostream"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("CalcTrans")
    .Input("points: float")         // 2 x nP
    .Input("trels: float")          // n_theta x nC x 2 x 3
    .Input("ntimestep1: int32")
    .Input("ncx: int32")
    .Input("ncy: int32")
    .Input("inc_x: float")
    .Input("inc_y: float")
    .Output("newpoints: float")     // n_theta x 2 x nP
    .Doc(R"doc(CPAB transformation implementation)doc");
    
REGISTER_OP("CalcGrad")
    .Input("points: float")        // 2 x nP
    .Input("as: float")            // n_theta x nC x 2 x 3
    .Input("bs: float")            // d x nC x 2 x 3
    .Input("ntimestep: int32")    
    .Input("ncx: int32")
    .Input("ncy: int32")
    .Input("inc_x: float")
    .Input("inc_y: float")
    .Output("grad: float")         // d x n_theta x 2 x nP
    .Doc(R"doc(Gradient of CPAB transformation implementation)doc");
    
class CalcTransCPU : public OpKernel {
    public:
        explicit CalcTransCPU(OpKernelConstruction* context) : OpKernel(context) {}
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& Trels_in = context->input(1);
            const Tensor& nStepSolver_in = context->input(2);
            const Tensor& ncx_in = context->input(3);
            const Tensor& ncy_in = context->input(4);
            const Tensor& inc_x_in = context->input(5);
            const Tensor& inc_y_in = context->input(6);
                
            // Problem size
            const int nP = points_in.dim_size(1);
            const int batch_size = Trels_in.dim_size(0);

            // Create and allocate output tensor
            const int NDIMS = 3;        
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s0 = {batch_size, 2, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s0), &newpoints_out));            
            typename TTypes<float, NDIMS>::Tensor newpoints = newpoints_out->tensor<float, NDIMS>();
                   
            // Setup data view
            typename TTypes<float>::ConstMatrix points = points_in.matrix<float>();
            const float* Trels = Trels_in.flat<float>().data();
            const int nStepSolver = nStepSolver_in.flat<int>()(0);
            const int ncx = ncx_in.flat<int>()(0);
            const int ncy = ncy_in.flat<int>()(0);
            const float inc_x = inc_x_in.flat<float>()(0);
            const float inc_y = inc_y_in.flat<float>()(0);
        
            // Loop over all transformations and all points    
            for(int t = 0; t < batch_size; t++){
                // Define start index for the matrices belonging to this batch
                // batch * num_elem * 4 triangles pr cell * cell in x * cell in y
                int start_idx = t * 6 * 4 * ncx * ncy; 
                for(int i = 0; i < nP; i++){
                    // Current point
                    float point[2];
                    point[0] = points(0,i);
                    point[1] = points(1,i);
                        
                    // Iterate in nStepSolver
                    int cellidx;
                    for(int n = 0; n < nStepSolver; n++){
                        // Find cell idx
                        cellidx = findcellidx(point, ncx, ncy, inc_x, inc_y);
    
                        // Extract the mapping in the cell
                        const float* Trels_idx = Trels + 6*cellidx + start_idx;                
                        
                        // Calculate trajectory of point
                        float point_updated[2];                
                        A_times_b(point_updated, Trels_idx, point);
                        
                        point[0] = point_updated[0];
                        point[1] = point_updated[1];
                    }
                        
                    // Copy to output
                    newpoints(t,0,i) = point[0];
                    newpoints(t,1,i) = point[1];
                }    
            } 
        } // end compute method
    private:
        int mymin(int a, double b) {
            return !(b<a)?a:round(b);
        }
    
        int findcellidx(const float* p, const int ncx, const int ncy, 
                        const float inc_x, const float inc_y) {
            // Move with respect to the lower bound
            double point[2];
            point[0] = p[0] + 1;
            point[1] = p[1] + 1;
            
            // Find initial row, col placement
            double p0 = std::min((ncx * inc_x - 0.000000001), std::max(0.0, point[0]));
            double p1 = std::min((ncy * inc_y - 0.000000001), std::max(0.0, point[1]));
            double xmod = fmod(p0, inc_x);
            double ymod = fmod(p1, inc_y);
            double x = xmod / inc_x;
            double y = ymod / inc_y;
            
            int cell_idx =  mymin(ncx-1, (p0 - xmod) / inc_x) + 
                            mymin(ncy-1, (p1 - ymod) / inc_y) * ncx;        
            cell_idx *= 4;
            
            // Out of bound (left)
            if(point[0]<=0){
                if(point[1] <= 0 && point[1]/inc_y<point[0]/inc_x){
                    // Nothing to do here
                } else if(point[1] >= ncy * inc_y && point[1]/inc_y-ncy > -point[0]/inc_x) {
                    cell_idx += 2;
                } else {
                    cell_idx += 3;
                }
                return cell_idx;
            }
            
            // Out of bound (right)
            if(point[0] >= ncx*inc_x){
                if(point[1]<=0 && -point[1]/inc_y > point[0]/inc_x - ncx){
                    // Nothing to do here
                } else if(point[1] >= ncy*inc_y && point[1]/inc_y - ncy > point[0]/inc_x-ncx){
                    cell_idx += 2;
                } else {
                    cell_idx += 1;
                }
                return cell_idx;
            }
                
            // Out of bound (up)
            if(point[1] <= 0){
                return cell_idx;
            }
            
            // Out of bound (bottom)
            if(point[1] >= ncy*inc_y){
                cell_idx += 2;
                return cell_idx;
            }
            
            // OK, we are inbound
            if(x<y){
                if(1-x<y){
                    cell_idx += 2;
                } else {
                    cell_idx += 3;
                }
            } else if(1-x<y) {
                cell_idx += 1;
            }
                                
            return cell_idx;
        }
        
        void A_times_b(float x[], const float* A, float* b) {
            x[0] = A[0]*b[0] + A[1]*b[1] + A[2];
              x[1] = A[3]*b[0] + A[4]*b[1] + A[5];
            return;
        }
};
        
// Forward decleration of kernel launcher 
void calcTrans_kernel_launcher(const GPUDevice& device, const int nP, const int batch_size,
                                float* newpoints, const float* points, 
                                const float* Trels, const int* nStepSolver,
                                const int* ncx, const int* ncy, 
                                const float* inc_x, const float* inc_y);

class CalcTransGPU : public OpKernel {
    public:
        explicit CalcTransGPU(OpKernelConstruction* context) : OpKernel(context) {}
        
        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& Trels_in = context->input(1);
            const Tensor& nStepSolver_in = context->input(2);
            const Tensor& ncx_in = context->input(3);
            const Tensor& ncy_in = context->input(4);
            const Tensor& inc_x_in = context->input(5);
            const Tensor& inc_y_in = context->input(6);
            
            // Problem size
            const int nP = points_in.dim_size(1);
            const int batch_size = Trels_in.dim_size(0);
            
            // Create and allocate output tensor
            Tensor* newpoints_out = NULL;
            std::initializer_list< int64 > s = {batch_size, 2, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &newpoints_out));            
            float* newpoints = newpoints_out->flat<float>().data();
                        
            // Setup data view
            const float* points = points_in.flat<float>().data();
            const float* Trels = Trels_in.flat<float>().data();
            const int* nStepSolver = nStepSolver_in.flat<int>().data();
            const int* ncx = ncx_in.flat<int>().data();
            const int* ncy = ncy_in.flat<int>().data();
            const float* inc_x = inc_x_in.flat<float>().data();
            const float* inc_y = inc_y_in.flat<float>().data();
            
            // Grap GPU device
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();

            // Launch kernel
            calcTrans_kernel_launcher(eigen_device, nP, batch_size,
                                      newpoints, points, Trels,
                                      nStepSolver, ncx, ncy, inc_x, inc_y);
            
            return;
        }
};

void calcT_batch_grad_kernel_launcher(  const GPUDevice& device, 
                                        const int n_theta, const int d, const int nP, const int nC,
                                        float* grad, const float* points, const float* As, const float* Bs,
                                        const int* nStepSolver, const int* ncx, const int* ncy, 
                                        const float* inc_x, const float* inc_y);

class CalcGradGPU : public OpKernel {
    public:
        explicit CalcGradGPU(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            // Grap input
            const Tensor& points_in = context->input(0);
            const Tensor& As_in = context->input(1);
            const Tensor& Bs_in = context->input(2);
            const Tensor& nStepSolver_in = context->input(3);
            const Tensor& ncx_in = context->input(4);
            const Tensor& ncy_in = context->input(5);
            const Tensor& inc_x_in = context->input(6);
            const Tensor& inc_y_in = context->input(7);

            // Create and allocate output tensor
            const int n_theta = As_in.dim_size(0);
            const int d = Bs_in.dim_size(0);
            const int nP = points_in.dim_size(1);
            const int nC = Bs_in.dim_size(1);
            
            Tensor* grad_out = NULL;
            std::initializer_list< int64 > s = {d, n_theta, 2, nP};
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape(s), &grad_out));            
            float* grad = (grad_out->flat<float>()).data();
            
            // Setup data view
            const float* points = (points_in.flat<float>()).data();
            const float* As = (As_in.flat<float>()).data();            
            const float* Bs = (Bs_in.flat<float>()).data();            
            const int* nStepSolver = (nStepSolver_in.flat<int>()).data();            
            const int* ncx = (ncx_in.flat<int>()).data();            
            const int* ncy = (ncy_in.flat<int>()).data();            
            const float* inc_x = (inc_x_in.flat<float>()).data();            
            const float* inc_y = (inc_y_in.flat<float>()).data();
                       
            // Get GPU information
            const GPUDevice& eigen_device = context->eigen_device<GPUDevice>();
            
            // Launch kernel
            calcT_batch_grad_kernel_launcher(eigen_device, n_theta, d, nP, nC,
                                             grad, points, As, Bs,
                                             nStepSolver, ncx, ncy, inc_x, inc_y);
            return;
        } // end compute method
}; // end CalcTBatchGradGPU

// Register kernels to OP's
REGISTER_KERNEL_BUILDER(Name("CalcTrans").Device(DEVICE_CPU), CalcTransCPU);
REGISTER_KERNEL_BUILDER(Name("CalcTrans").Device(DEVICE_GPU), CalcTransGPU);
// TODO: Implement CalcGrad for CPU: REGISTER_KERNEL_BUILDER(Name("CalcGrad").Device(DEVICE_CPU), CalcGradCPU);
REGISTER_KERNEL_BUILDER(Name("CalcGrad").Device(DEVICE_GPU), CalcGradGPU);


