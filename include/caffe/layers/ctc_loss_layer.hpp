#ifndef CAFFE_CTC_LOSS_LAYER_HPP
#define CAFFE_CTC_LOSS_LAYER_HPP

#include <list>
#include <vector>

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>


class CTCLossLayer : public LossLayer<Dtype> {
 public:

 public:
  explicit CTCLossLayer(const LayerParameter& param);
  virtual ~CTCLossLayer();

  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CTCLoss"; }

  // probabilities, sequence indicators, target sequence
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // loss
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:


  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  /**
   * @brief Unused. Gradient calculation is done in Forward_cpu
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
#ifndef CPU_ONLY
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      										 const vector<Blob<Dtype>*>& top);

	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
														const vector<bool>& propagate_down,
														const vector<Blob<Dtype>*>& bottom);
#endif

 private:

  int T_;
  int N_;
  int C_;

};

}  // namespace caffe

#endif  // CAFFE_CTC_LOSS_LAYER_HPP
