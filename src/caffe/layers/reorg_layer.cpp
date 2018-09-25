#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void reorg_cpu(const Dtype *bottom_data, const int b_w, const int b_h,
		const int b_c, const int b_n, const int stride,
		const bool forward, Dtype *top_data) {
	int t_c = b_c / (stride * stride);
	int t_w = b_w * stride;
	int t_h = b_h * stride;
	for (int n = 0; n < b_n; n++) {
		for (int c = 0; c < b_c; c++) {
			for (int h = 0; h < b_h; h++) {
				for (int w = 0; w < b_w; w++) {
					int bottom_index = w + b_w * (h + b_h * (c + b_c * n));
					int c2 = c % t_c;
					int offset = c / t_c;
					int w2 = w * stride + offset % stride;
					int h2 = h * stride + offset / stride;
					int top_index = w2 + t_w * (h2 + t_h * (c2 + t_c * n));
					if (forward) top_data[top_index] = bottom_data[bottom_index];
					else
						top_data[bottom_index] = bottom_data[top_index];
				}
			}
		}
	}
}

template
void reorg_cpu<double>(const double *bottom_data, const int b_w, const int b_h,
		const int b_c, const int b_n, const int stride,
		const bool forward, double *top_data);

template
void reorg_cpu<float>(const float *bottom_data, const int b_w, const int b_h,
		const int b_c, const int b_n, const int stride,
		const bool forward, float *top_data);

template<typename Dtype>
void ReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {
	CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
			"allow in-place computation.";
	ReorgParameter reorg_param = this->layer_param_.reorg_param();
	CHECK_EQ(reorg_param.has_stride(), true) << this->type() << " Layer needs stride param.";
	reverse_ = reorg_param.reverse();
	stride_ = reorg_param.stride();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	batch_num_ = bottom[0]->num();

	if (reverse_) {
		reorged_channels_ = channels_ / (stride_ * stride_);
		reorged_width_ = width_ * stride_;
		reorged_height_ = height_ * stride_;
	} else {
		reorged_channels_ = channels_ * stride_ * stride_;
		reorged_height_ = height_ / stride_;
		reorged_width_ = width_ / stride_;
	}
}

template<typename Dtype>
void ReorgLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {

	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	batch_num_ = bottom[0]->num();

	if (reverse_) {
		reorged_channels_ = channels_ / (stride_ * stride_);
		reorged_width_ = width_ * stride_;
		reorged_height_ = height_ * stride_;
	} else {
		reorged_channels_ = channels_ * stride_ * stride_;
		reorged_height_ = height_ / stride_;
		reorged_width_ = width_ / stride_;
	}

	top[0]->Reshape(batch_num_, reorged_channels_,
			reorged_height_, reorged_width_);
}

template<typename Dtype>
void ReorgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {
	const Dtype *bottom_data = bottom[0]->cpu_data();
	Dtype *top_data = top[0]->mutable_cpu_data();
	reorg_cpu(bottom_data, reorged_width_, reorged_height_,
			reorged_channels_, batch_num_, stride_, reverse_, top_data);
}

template<typename Dtype>
void ReorgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
		const vector<Blob<Dtype> *> &bottom) {
	if(!propagate_down[0]){
		return;
	}
	const Dtype *top_diff = top[0]->cpu_diff();
	Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
	reorg_cpu(top_diff, width_, height_,
			channels_, batch_num_, stride_, !reverse_, bottom_diff);
}

INSTANTIATE_CLASS(ReorgLayer);

REGISTER_LAYER_CLASS(Reorg);

}  // namespace caffe
