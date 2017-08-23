#ifndef CAFFE_REGION_LAYER_HPP_
#define CAFFE_REGION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class RegBox{
public:

	float x;
	float y;
	float w;
	float h;
	float angle;
	int class_no;
	float iou;
	int box_index;
	int nms;
	float classes_prob[6];
};

/*
 * @brief
 *
 */
template<typename Dtype>
class RegionLayer : public Layer<Dtype> {
public:
	explicit RegionLayer(const LayerParameter &param)
	: Layer<Dtype>(param) {}

	virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);

	virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);

	virtual inline const char *type() const { return "Region"; }

	virtual inline int MaxBottomBlobs() const { return 2; }

	virtual inline int ExactNumTopBlobs() const { return 3; }

protected:


	virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
			const vector<Blob<Dtype> *> &top);

	virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
			const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

	virtual void ouputPlaneRegionBoxes(Dtype *top_data, Dtype *top_data2);
	virtual void ouputRegionBoxes(const vector<Blob<Dtype> *> &top);

	int classes_;
	int coords_;
	int boxes_of_each_grid_;
	bool softmax_;
	int batch_num_;
	int channels_;
	int height_, width_;

	std::vector<float> biases;
	std::vector<float> biases_x;
	std::vector<float> biases_y;
	float min_confidence;
	float nms_threshold = 0.6;
	float iou_threshold = 0.6;
	float object_scale = 5;
	int bias_match = 1;

};


}  // namespace caffe

#endif  // CAFFE_REGION_LAYER_HPP_
