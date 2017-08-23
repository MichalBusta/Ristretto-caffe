#ifndef CAFFE_ONDISK_DATA_LAYER_HPP_
#define CAFFE_ONDISK_DATA_LAYER_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class OnDiskDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit OnDiskDataLayer(const LayerParameter& param);
  virtual ~OnDiskDataLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CMPData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 5; }

  virtual std::string get_image_file_name(int bid);

  virtual int get_crop(int bid, int side);

  virtual void fill_bucket(Blob<Dtype>* imgs, Blob<Dtype>* boxes, int width);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);

  std::vector<std::string> file_list;

  int batch_size;
  int height_ = 100;
  int width_ = 100;
  int channels_ = 1;

  float crop_ratio = 0.1;

  std::string data_dir;

  std::vector<cv::Mat> orig_top;
  std::vector<std::string> src_images;
  std::vector<cv::Rect> crops;

};

}  // namespace caffe

#endif  // CAFFE_ONDISK_DATA_LAYER_HPP_
