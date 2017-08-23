#include <algorithm>
#include <vector>
#include <float.h>

#include "caffe/layers/logsoftmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void LogSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
}

/* Credits to Leon Bottou */
double THExpMinusApprox(const double x)
{
#define EXACT_EXPONENTIAL 0
#if EXACT_EXPONENTIAL
  return exp(-x);
#else
  /* fast approximation of exp(-x) for x positive */
# define A0   (1.0)
# define A1   (0.125)
# define A2   (0.0078125)
# define A3   (0.00032552083)
# define A4   (1.0172526e-5)
  if (x < 13.0)
  {
/*    assert(x>=0); */
    double y;
    y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
    y *= y;
    y *= y;
    y *= y;
    y = 1/y;
    return y;
  }
  return 0;
# undef A0
# undef A1
# undef A2
# undef A3
# undef A4
#endif
}

template <typename Dtype>
void LogSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  int t = 0;
  //std::cout << "on: " << outer_num_ << ", Dim: " << dim << "\n";
  #pragma omp parallel for private(t)
  for(t = 0; t < outer_num_; t++)
  {
  	Dtype* ptr_output = top_data + t*dim;
  	const Dtype* ptr_input = bottom_data + t*dim;

    Dtype logsum = 0;
    Dtype maxInput = -FLT_MAX;
    for(int d = 0; d < dim; d++)
      maxInput = std::max(maxInput, ptr_input[d]);

    for(int d = 0; d < dim; d++)
      logsum += THExpMinusApprox(maxInput-ptr_input[d]);
    logsum = maxInput + log(logsum + 1e-8);

    for(int d = 0; d < dim; d++)
      ptr_output[d] = ptr_input[d] - logsum;
  }
}

template <typename Dtype>
void LogSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* gradOutput_data = top[0]->cpu_diff();
  const Dtype* output_data = top[0]->cpu_data();
  Dtype* gradInput_data = bottom[0]->mutable_cpu_diff();
  int dim = top[0]->count() / outer_num_;

  int t = 0;
  #pragma omp parallel for private(t)
  for(t = 0; t < outer_num_; t++)
  {
  	const Dtype* ptr_output = output_data + t*dim;
  	const Dtype* ptr_gradOutput  = gradOutput_data  + t*dim;
  	Dtype* ptr_gradInput  = gradInput_data  + t*dim;

  	Dtype sum = 0;
    for(int d = 0; d < dim; d++)
      sum += ptr_gradOutput[d];

    for(int d = 0; d < dim; d++)
      ptr_gradInput[d] = ptr_gradOutput[d] - THExpMinusApprox(-ptr_output[d])*sum;
  }

}


#ifdef CPU_ONLY
//STUB_GPU(LogSoftmaxLayer);
#endif

INSTANTIATE_CLASS(LogSoftmaxLayer);
REGISTER_LAYER_CLASS(LogSoftmax);

}  // namespace caffe
