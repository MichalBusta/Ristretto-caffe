#include "caffe/layers/ctc_loss_layer.hpp"
#include "caffe/util/ctc/ctc.h"
#include "caffe/util/ctc/detail/gpu_ctc.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace caffe {

static inline ctcStatus_t get_gpu_workspace_size(const int* label_lengths,
                               const int* input_lengths,
                               int alphabet_size, int minibatch,
                               ctcComputeInfo info,
                               size_t* size_bytes)
{
    if (label_lengths == nullptr ||
        input_lengths == nullptr ||
        size_bytes == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0)
        return CTC_STATUS_INVALID_VALUE;

    // This is the max of all S and T for all examples in the minibatch.
    int maxL = *std::max_element(label_lengths, label_lengths + minibatch);
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch);

    const int S = 2 * maxL + 1;

    *size_bytes = 0;

    // GPU storage
    //nll_forward, nll_backward
    *size_bytes += 2 * sizeof(float) * minibatch;

    //repeats
    *size_bytes += sizeof(int) * minibatch;

    //label offsets
    *size_bytes += sizeof(int) * minibatch;

    //utt_length
    *size_bytes += sizeof(int) * minibatch;

    //label lengths
    *size_bytes += sizeof(int) * minibatch;

    //labels without blanks - overallocate for now
    *size_bytes += sizeof(int) * maxL * minibatch;

    //labels with blanks
    *size_bytes += sizeof(int) * S * minibatch;

    //alphas
    *size_bytes += sizeof(float) * S * maxT * minibatch;

    //denoms
    *size_bytes += sizeof(float) * maxT * minibatch;

    //probs (since we will pass in activations)
    *size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;

    return CTC_STATUS_SUCCESS;
}

template <typename Dtype>
ctcStatus_t compute_ctc_loss(const Dtype* const activations,
														 Dtype* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
														 Dtype *costs,
                             void *workspace,
                             ctcComputeInfo info) {

	if (activations == nullptr ||
			flat_labels == nullptr ||
			label_lengths == nullptr ||
			input_lengths == nullptr ||
			costs == nullptr ||
			workspace == nullptr ||
			alphabet_size <= 0 ||
			minibatch <= 0)
			return CTC_STATUS_INVALID_VALUE;


	GpuCTC<Dtype> ctc(alphabet_size, minibatch, workspace, info.stream);

	if (gradients != NULL)
			return ctc.cost_and_grad(activations, gradients, costs,
															 flat_labels, label_lengths,
															 input_lengths);
	else
			return ctc.score_forward(activations, costs, flat_labels,
															 label_lengths, input_lengths);
}

template <typename Dtype>
static inline void gpu_ctc(Dtype* acts,
						 Dtype* grads,
             int* labels,
             int* label_lengths,
             int* input_lengths,
             int alphabet_size,
             int minibatch,
						 Dtype* cost,
             int num_threads)
{

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	ctcComputeInfo info;
	info.loc = CTC_GPU;
	info.stream = stream;

	size_t gpu_alloc_bytes;
	get_gpu_workspace_size(label_lengths, input_lengths,
										 alphabet_size, minibatch, info,
										 &gpu_alloc_bytes);

	char *ctc_gpu_workspace;
	CUDA_CHECK(cudaMalloc(&ctc_gpu_workspace, gpu_alloc_bytes));

	compute_ctc_loss<Dtype>(acts, grads,
									 labels, label_lengths,
									 input_lengths,
									 alphabet_size,
									 minibatch,
									 cost,
									 ctc_gpu_workspace,
									 info);
	CUDA_CHECK(cudaFree(ctc_gpu_workspace));
	CUDA_CHECK(cudaStreamDestroy(stream));
}

//#define DEBUG 1

template <typename Dtype>
void CTCLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> *acts = bottom[0];
  const Blob<Dtype> *gt_blob = bottom[1];

  std::vector<int> input_lengths;

  std::vector<int> label_lengths;
  std::vector<int> labels;

  const Dtype* gt_blob_data = gt_blob->cpu_data();
  int max_length = gt_blob->shape(1);
  for( size_t b = 0; b < N_; b++ ){
  	const Dtype* gt_blob_data_batch =  &gt_blob_data[b * max_length];
  	int max_size = 0;
  	for(size_t t = 0; t < max_length; t++){
  		if( gt_blob_data_batch[t] > 0 ){
  			max_size += 1;
  			labels.push_back( (int) gt_blob_data_batch[t] );
  			if((int) gt_blob_data_batch[t] < 0 || (int) gt_blob_data_batch[t] >  acts->shape(3)){
  				std::cout << "invalid gt: " << (int) gt_blob_data_batch[t]  << std::endl;
  			}
  		}
  	}
  	label_lengths.push_back(max_size);
  	input_lengths.push_back(acts->shape(0));
  }


  int alphabet_size = acts->shape(3);
  int minibatch = acts->shape(1);

  std::vector<Dtype> cost;
  cost.resize(N_);
  Dtype *grads = acts->mutable_gpu_diff();


  gpu_ctc<Dtype>(acts->mutable_gpu_data(), grads, &labels[0], &label_lengths[0], &input_lengths[0], alphabet_size, minibatch, &cost[0], 1);

#ifdef DEBUG
  if(true){
  	std::cout << "label lengths:" << label_lengths[0] << ", input_lengths: " << input_lengths[0] << ", alphabet_size: " << alphabet_size << ", minibatch:" << minibatch << std::endl;
  	for(size_t i = 0; i < labels.size(); i++){
  		std::cout << labels[i] << " ";
  	}
  	std::cout << std::endl;

  	const Dtype* acts_data = acts->mutable_cpu_data();
  	for(size_t i = 0; i < acts->count(); i++){
  		std::cout << acts_data[i] << " ";
  	}
  	std::cout << std::endl;
  }
  CUDA_POST_KERNEL_CHECK;
#endif
  Dtype loss = 0;
  for(size_t i = 0; i < N_; i++){
  	loss += cost[i];
  }

  bool status = std::isinf(loss) || std::isnan(loss);
  if(status){
  	std::cout << "Inf cost!!!!\n";
  	caffe_gpu_set(acts->count(), (Dtype) 0, grads);
  }else{
		Dtype normalizer = N_;
		//std::cout << "Loss: " << loss << ", N: " << minibatch << " ms: " << label_lengths[0] << " is: " << input_lengths[0] << std::endl;
		top[0]->mutable_cpu_data()[0] = loss / normalizer;
		caffe_gpu_scal(acts->count(), 1/normalizer, grads);
  }

  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void CTCLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(CTCLossLayer);

}  // namespace caffe
