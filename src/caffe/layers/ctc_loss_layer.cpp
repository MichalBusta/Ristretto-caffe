#include "caffe/layers/ctc_loss_layer.hpp"
#include "caffe/util/ctc/ctc.h"
#include "caffe/util/ctc/detail/cpu_ctc.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace caffe {

static inline ctcStatus_t get_workspace_size(const int* label_lengths,
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

    if (info.loc == CTC_GPU) {
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

    } else {
        //cpu can eventually replace all minibatch with
        //max number of concurrent threads if memory is
        //really tight

        //per minibatch memory
        size_t per_minibatch_bytes = 0;

        //output
        per_minibatch_bytes += sizeof(float) * alphabet_size ;

        //alphas
        per_minibatch_bytes += sizeof(float) * S * maxT;

        //betas
        per_minibatch_bytes += sizeof(float) * S;

        //labels w/blanks, e_inc, s_inc
        per_minibatch_bytes += 3 * sizeof(int) * S;

        *size_bytes = per_minibatch_bytes * minibatch;

        //probs
        *size_bytes += sizeof(float) * alphabet_size * maxT * minibatch;
    }

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


		CpuCTC<Dtype> ctc(alphabet_size, minibatch, workspace, info.num_threads);

		if (gradients != NULL)
				return ctc.cost_and_grad(activations, gradients,
																 costs,
																 flat_labels, label_lengths,
																 input_lengths);
		else
				return ctc.score_forward(activations, costs, flat_labels,
																 label_lengths, input_lengths);
}

template <typename Dtype>
void cpu_ctc(Dtype* acts,
						 Dtype* grads,
             int* labels,
             int* label_lengths,
             int* input_lengths,
             int alphabet_size,
             int minibatch,
						 Dtype* cost,
             int num_threads)
{
    ctcComputeInfo info;
    info.loc = CTC_CPU;
    info.num_threads = num_threads;

    size_t cpu_alloc_bytes;
    get_workspace_size(label_lengths, input_lengths,
                       alphabet_size, minibatch, info,
                       &cpu_alloc_bytes);

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    compute_ctc_loss(acts, grads,
                     labels, label_lengths,
                     input_lengths,
                     alphabet_size,
                     minibatch,
                     cost,
                     ctc_cpu_workspace,
                     info);

    free(ctc_cpu_workspace);
}




template <typename Dtype>
CTCLossLayer<Dtype>::CTCLossLayer(const LayerParameter& param)
     : LossLayer<Dtype>(param),
       T_(0),
       N_(0),
       C_(0) {
}

template <typename Dtype>
CTCLossLayer<Dtype>::~CTCLossLayer() {

}

template <typename Dtype>
void CTCLossLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::LayerSetUp(bottom, top);
  const Blob<Dtype>* probs = bottom[0];

  T_ = probs->num();
  N_ = probs->channels();
  C_ = probs->height();

}

template <typename Dtype>
void CTCLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const Blob<Dtype>* probs = bottom[0];
	T_ = probs->num();
	N_ = probs->channels();
	C_ = probs->height();

	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  		}
  	}
  	label_lengths.push_back(max_size);
  	input_lengths.push_back(acts->shape(0));
  }


  int alphabet_size = acts->shape(3);
  int minibatch = acts->shape(1);

  std::vector<Dtype> cost;
  cost.resize(N_);
  Dtype *grads = acts->mutable_cpu_diff();

  cpu_ctc(acts->mutable_cpu_data(), grads, &labels[0], &label_lengths[0], &input_lengths[0], alphabet_size, minibatch, &cost[0], 1);
	
  Dtype loss = 0;
  for(size_t i = 0; i < N_; i++){
  	loss += cost[i];
  }

	Dtype normalizer = N_;
	//std::cout << "Loss: " << loss << ", N: " << minibatch << " ms: " << label_lengths[0] << " is: " << input_lengths[0] << std::endl;
	top[0]->mutable_cpu_data()[0] = loss / normalizer;

	caffe_scal(acts->count(), 1/normalizer, grads);

}

template <typename Dtype>
void CTCLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_CLASS(CTCLossLayer);
REGISTER_LAYER_CLASS(CTCLoss);

}  // namespace caffe
