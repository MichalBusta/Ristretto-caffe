#include "caffe/layers/region_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core/core.hpp>

namespace caffe {

#define TOP2_OUT 15
#define MAX_BOXES 500
#define LABES_WIDTH 6

static inline double logistic_activate(double x) { return 1. / (1 + exp(-x)); }

static inline float logistic_activate(float x) { return (float) (1. / (1 + exp(-x))); }

static inline float logistic_gradient(float x){return (1-x)*x;}

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(RegBox& a, RegBox& b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

static float box_union(RegBox& a, RegBox& b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

static float box_iou(RegBox& a, RegBox& b)
{
    return box_intersection(a, b)/box_union(a, b);
}

template<typename Dtype>
void softmax(Dtype *input, int n, Dtype temp, Dtype *output) {
		int i;
		Dtype sum = 0;
		Dtype largest = -FLT_MAX;
		for (i = 0; i < n; ++i) {
				if (input[i] > largest) largest = input[i];
		}
		for (i = 0; i < n; ++i) {
				Dtype e = exp(input[i] / temp - largest / temp);
				sum += e;
				output[i] = e;
		}
		for (i = 0; i < n; ++i) {
				output[i] /= sum;
		}
}

template
void softmax(float *input, int n, float temp, float *output);

template
void softmax(double *input, int n, double temp, double *output);

template<typename Dtype>
void RegionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {
	CHECK_NE(top[2], bottom[0]) << this->type() << " Layer does not "
			"allow in-place computation.";
	RegionParameter region_param = this->layer_param_.region_param();
	classes_ = region_param.classes();
	coords_ = region_param.coords();
	boxes_of_each_grid_ = region_param.boxes_of_each_grid();

	channels_ = bottom[0]->shape(3);
	height_ = bottom[0]->shape(1);
	width_ = bottom[0]->shape(2);
	batch_num_ = bottom[0]->shape(0);
	nms_threshold = region_param.nms_threshold();
	iou_threshold = region_param.iou_threshold();
	object_scale = region_param.object_scale();

	biases = {0.317539695732,0.493652380857, 0.66545245494,0.357112901109, 1.01046388351,0.512909981333, 1.3496335045,0.322306706382, 1.58891134384,0.581019084508,  1.20785566117,1.02485042068, 2.28320153664,0.813839044484, 2.5279678299,0.442826767368, 3.01451020961,1.34187668082, 3.79487918772,0.78852430371, 4.7935325462,3.01837445215, 6.1298771774,1.41297354615, 10.4140073713, 2.29083421664, 12.1392691662, 5.06897667207};
	if(region_param.expand_bias()){
		std::vector<float> biases_out;
		for( size_t i = 0; i < biases.size() / 2; i+=2 ){
			if( biases[i] < 0.5 || biases[i + 1] < 0.5 ){

				biases_out.push_back(biases[i]);
				biases_out.push_back(biases[i + 1]);
				biases_x.push_back(0.75);
				biases_y.push_back(0.25);

				biases_out.push_back(biases[i]);
				biases_out.push_back(biases[i + 1]);
				biases_x.push_back(0.25);
				biases_y.push_back(0.25);

				biases_out.push_back(biases[i]);
				biases_out.push_back(biases[i + 1]);
				biases_x.push_back(0.5);
				biases_y.push_back(0.5);

				biases_out.push_back(biases[i]);
				biases_out.push_back(biases[i + 1]);
				biases_x.push_back(0.75);
				biases_y.push_back(0.75);

				biases_out.push_back(biases[i]);
				biases_out.push_back(biases[i + 1]);
				biases_x.push_back(0.75);
				biases_y.push_back(0.25);


			}else{
				biases_out.push_back(biases[i]);
				biases_out.push_back(biases[i + 1]);
				biases_x.push_back(0);
				biases_y.push_back(0);
			}
		}
		biases = biases_out;

		std::cout << "Have " << biases.size() << "biases" << std::endl;
		boxes_of_each_grid_ = biases.size() / 2;
	}else{
		for( size_t i = 0; i < biases.size() / 2; i+=2 ){
			biases_x.push_back(0);
			biases_y.push_back(0);
		}
	}

	min_confidence = 0.01;
	softmax_ = region_param.softmax();
}

template<typename Dtype>
void RegionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {

	channels_ = bottom[0]->shape(3);
	height_ = bottom[0]->shape(1);
	width_ = bottom[0]->shape(2);
	batch_num_ = bottom[0]->shape(0);

	top[2]->Reshape(batch_num_, height_, width_, channels_);
	top[1]->Reshape(batch_num_, 1, MAX_BOXES, TOP2_OUT);
	top[0]->Reshape(1, 1, 1, 1);
}

template<typename Dtype>
RegBox get_region_box(Dtype *x, std::vector<float>& biases, std::vector<float>& biasesx, std::vector<float>& biasesy, int n, size_t index, int i, int j, int w, int h, int coords, float norm)
{
	RegBox b;
	b.x = (i + logistic_activate(x[index + 0]) + biasesx[n]) / w;
	b.y = (j + logistic_activate(x[index + 1] + biasesy[n])) / h;
	b.w = exp(x[index + 2]) * biases[2*n]   / norm;
	b.h = exp(x[index + 3]) * biases[2*n+1] / norm;

	if(coords > 4)
		b.angle = x[index + 4];
	b.box_index = index;
	b.nms = 0;
	return b;
}

//TODO !! - class indices outside of the cpp
//Latin = 0 , Kanji = 1, Arabic = 2 Bangla = 3, Korean = 4
template<typename Dtype>
void RegionLayer<Dtype>::ouputPlaneRegionBoxes(Dtype *top_data, Dtype *top_data2)
{
	float norm = sqrtf(width_* width_ + height_ * height_);
	std::vector<RegBox> out_boxes;
	out_boxes.reserve(MAX_BOXES);
  #pragma omp parallel for
	for (int i = 0; i < width_*height_; ++i){
		int row = i / width_;
		int col = i % width_;
		for(int n = 0; n < boxes_of_each_grid_; ++n){
			size_t index = i*boxes_of_each_grid_ + n;
			size_t p_index = index * (classes_ + coords_ + 1) + coords_;
			float scale = top_data[p_index];
			size_t box_index = index * (classes_ + coords_ + 1);
			RegBox box = get_region_box(top_data, biases, biases_x, biases_y, n, box_index, col, row, width_, height_, coords_, norm);

			size_t class_index = index * (classes_ + coords_ + 1) + coords_ + 1;
			box.iou = scale*top_data[index * (classes_ + coords_ + 1) + coords_];

			if(box.iou < min_confidence)
				continue;

			float best_prob = 0;
			for(int j = 0; j < classes_; ++j){
				float prob = scale*top_data[class_index+j];
				box.classes_prob[j] = prob;
				if(prob > best_prob){
					//box.prob = prob;
					box.class_no = j;
					best_prob = prob;
				}
			}
      #pragma omp critical
			{
				out_boxes.push_back(box);
			}
		}
	}
	sort(out_boxes.begin(), out_boxes.end(),
	    [](const RegBox & a, const RegBox & b)
	{
	    return a.iou > b.iou;
	});

	for(size_t i = 0; i < out_boxes.size(); i++){
		RegBox &a  = out_boxes[i];
		for(size_t j = i + 1; j < out_boxes.size(); j++){
			RegBox &b  = out_boxes[j];
			float iou = box_iou(a, b);

			if(b.iou > a.iou){
				//a.iou -= iou; !!TODO - proper evaluation?
				if(iou >= nms_threshold and a.class_no == b.class_no){
					a.nms = 1;
				}
			}else{
				//b.iou -= iou;
				if(iou >= nms_threshold and a.class_no == b.class_no){
					b.nms = 1;
				}
			}
		}
	}

	sort(out_boxes.begin(), out_boxes.end(),
		    [](const RegBox & a, const RegBox & b)
	{
			if(a.nms == b.nms)
		    return a.iou > b.iou;
			else
				return a.nms < b.nms;
	});

	for(size_t i = 0; i < std::min(out_boxes.size(), (size_t) MAX_BOXES); i++){
		top_data2[i* TOP2_OUT] = out_boxes[i].x;
		top_data2[i* TOP2_OUT + 1] = out_boxes[i].y;
		top_data2[i* TOP2_OUT + 2] = out_boxes[i].w;
		top_data2[i* TOP2_OUT + 3] = out_boxes[i].h;
		top_data2[i* TOP2_OUT + 4] = out_boxes[i].angle;
		top_data2[i* TOP2_OUT + 5] = out_boxes[i].iou;
		top_data2[i* TOP2_OUT + 6] = out_boxes[i].class_no;
		top_data2[i* TOP2_OUT + 7] = out_boxes[i].box_index;
		top_data2[i* TOP2_OUT + 8] = out_boxes[i].nms;
		top_data2[i* TOP2_OUT + 9] = out_boxes[i].classes_prob[0];
		top_data2[i* TOP2_OUT + 10] = out_boxes[i].classes_prob[1];
		top_data2[i* TOP2_OUT + 11] = out_boxes[i].classes_prob[2];
		top_data2[i* TOP2_OUT + 12] = out_boxes[i].classes_prob[3];
		top_data2[i* TOP2_OUT + 13] = out_boxes[i].classes_prob[4];
		top_data2[i* TOP2_OUT + 14] = out_boxes[i].classes_prob[5];
	}
}

template<typename Dtype>
void RegionLayer<Dtype>::ouputRegionBoxes(const vector<Blob<Dtype> *> &top)
{
	Dtype *top_data = top[2]->mutable_cpu_data();
	Dtype *top_data2 = top[1]->mutable_cpu_data();
	memset((void *) top_data2, 0, sizeof(Dtype) * MAX_BOXES * TOP2_OUT * batch_num_);

	int size = coords_ + classes_ + 1;
	size_t outputs = width_*height_* boxes_of_each_grid_* size;
	for (int b = 0; b < batch_num_; ++b) {
		ouputPlaneRegionBoxes(top_data + b * outputs, top_data2 + b * (MAX_BOXES * TOP2_OUT));
	}

}

template<typename Dtype>
RegBox get_gt_box(Dtype* data){
	RegBox truth;
	truth.x = data[0];
	truth.y = data[1];
	truth.w = data[2];
	truth.h = data[3];
	truth.angle = data[4];
	truth.class_no = data[5];
	return truth;
}

template<typename Dtype>
void delta_region_class(const Dtype *output, Dtype *delta, int index, int cls, int classes, float scale)
{
	if(cls == -10)
		cls = 1;
	if( cls < 0 || cls > classes){
		std::cout << "Bad class: " << cls << std::endl;
	}

	for(int n = 0; n < classes; ++n){
		delta[index + n] = - scale * (((n == cls)?1 : 0) - output[index + n]);
	}
}

template<typename Dtype>
static float delta_region_box(RegBox& truth, const Dtype *x, std::vector<float>& biases, std::vector<float>& biases_x, std::vector<float>& biases_y, int n, int index, int i, int j, int w, int h, Dtype *delta, float scale, int coords, float norm)
{
	RegBox pred = get_region_box(x, biases, biases_x, biases_y, n, index, i, j, w, h, coords, norm);
	float iou = box_iou(pred, truth);
	float tx = (truth.x*w - i);
	float ty = (truth.y*h - j);
	if(truth.w  == 0 || truth.h == 0 ){
		std::cout << "Zero log!\n";
		return iou;
	}
	float tw = log(truth.w*norm / biases[2*n]);
	float th = log(truth.h*norm / biases[2*n + 1]);

	//std::cout << "gt: " << truth.x << "x" << truth.y << "x" << truth.w << "x" <<  truth.h << "\n";
	//std::cout << "pr: " << pred.x << "x" << pred.y << "x" << pred.w << "x" <<  pred.h <<  "-" << tw - x[index + 2] << "\n";

	delta[index + 0] = - scale * (tx - logistic_activate(x[index + 0] +  biases_x[n])) * logistic_gradient(logistic_activate(x[index + 0] +  biases_x[n] ));
	delta[index + 1] = - scale * (ty - logistic_activate(x[index + 1] + biases_y[n])) * logistic_gradient(logistic_activate(x[index + 1] +  biases_y[n]));

	//for some annotations we have just axis aligned bbox - then we do not regress w/h/angle
	if( coords > 4 && truth.angle < -50 ){
	  if( truth.w > pred.w )
	    delta[index + 2] = - scale * (tw - x[index + 2]);
		return iou;
	}

	delta[index + 2] = - scale * (tw - x[index + 2]);
	delta[index + 3] = - scale * (th - x[index + 3]);
	if(coords > 4){
		float angle_diff = truth.angle - pred.angle;
		while(angle_diff > 2* M_PI)
			angle_diff -= 2* M_PI;
		while(angle_diff < -2* M_PI)
			angle_diff += 2* M_PI;
		if( fabs(truth.angle - pred.angle) > 2 * M_PI )
			std::cout << "Angle diff: " << (truth.angle - pred.angle) << std::endl;
		delta[index + 4] = - scale *(angle_diff);
	}
	return iou;
}

template<typename Dtype>
void RegionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
		const vector<Blob<Dtype> *> &top) {
	const Dtype *bottom_data = bottom[0]->cpu_data();
	Dtype *top_data = top[2]->mutable_cpu_data();
	int size = coords_ + classes_ + 1;
	size_t ouputs = width_*height_* boxes_of_each_grid_* size;
	caffe_copy(batch_num_ * ouputs, bottom_data, top_data);

	for (int b = 0; b < batch_num_; ++b) {
    #pragma omp parallel for
		for (size_t i = 0; i < height_ * width_ * boxes_of_each_grid_; ++i) {
			size_t index = size * i + b * ouputs;
			top_data[index + coords_] = logistic_activate(top_data[index + coords_]);
		}
	}

	if (softmax_) {
		for (int b = 0; b < batch_num_; ++b) {
      #pragma omp parallel for
			for (int i = 0; i < height_ * width_ * boxes_of_each_grid_; ++i) {
				size_t index = size * i + b * ouputs;
				softmax(top_data + index + coords_ + 1, classes_, (Dtype) 1, top_data + index + coords_ + 1);
			}
		}
	}
	ouputRegionBoxes(top);
	if(this->phase_ == TEST)
		return;

	//compute loss
	Dtype *loss = top[0]->mutable_cpu_data();
	loss[0] = 0;
	const Dtype *gt_boxall = bottom[1]->cpu_data();
	int max_boxes = bottom[1]->shape(1);
	Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
	size_t outputs = width_*height_* boxes_of_each_grid_* size;

	caffe_set(batch_num_ * outputs, (Dtype) 0, bottom_diff);

	float norm = sqrtf( width_* width_ + height_ * height_);
	int coordsp = coords_ + 1;

  #pragma omp parallel for
	for (int b = 0; b < batch_num_; ++b) {
		const Dtype *gt_box = gt_boxall + b * (max_boxes * LABES_WIDTH);
		//negatives ...
		for (int j = 0; j < height_; ++j) {
			for (int i = 0; i < width_; ++i) {
				for (int n = 0; n < boxes_of_each_grid_; ++n) {
					size_t index = size*(j*width_*boxes_of_each_grid_ + i*boxes_of_each_grid_ + n) + b*outputs;
					RegBox truth0 = get_gt_box(gt_box);
					if(truth0.class_no == -10){
						continue;
					}

					bottom_diff[index + coords_] = - 1 * ((0 - top_data[index + coords_]) * logistic_gradient(top_data[index + coords_]));
					#pragma omp critical
					{
						loss[0] += (0 - top_data[index + coords_]) * (0 - top_data[index + coords_]);
					}
				}
			}
		}

		//and positive samples
		for(size_t bi = 0; bi < max_boxes; bi++){

			RegBox truth = get_gt_box(gt_box + bi* LABES_WIDTH);;
			if(truth.x == 0 && truth.y == 0)
				continue;

			if(truth.class_no == -1)
				continue;

			float best_iou = 0;
			size_t best_index = 0;
			int best_n = 0;

			int i = (truth.x * width_);
			int j = (truth.y * height_);
			if(i > width_ - 1){
				continue;
			}
			if(j > height_ - 1){
				continue;
			}

			if(truth.w <= 0 || truth.h <= 0 || truth.w > 1 || truth.h > 1){
				std::cout << "Invalid gt!\n";
				continue;
			}

			//printf("index %d %d\n",i, j);
			for(int n = 0; n < boxes_of_each_grid_; ++n){

				size_t index = size*(j*width_*boxes_of_each_grid_ + i*boxes_of_each_grid_ + n) + b*outputs;

				if(truth.class_no == -1){
					#pragma omp critical
					{
						if(bottom_diff[index + coords_] > 0)
							loss[0] -= (0 - top_data[index + coords_]) * (0 - top_data[index + coords_]);
					}
					bottom_diff[index + coords_] = 0;
					continue;
				}


				RegBox pred = get_region_box(top_data, biases, biases_x, biases_y, n, index, i, j, width_, height_, coords_, norm);
				//printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
				if(bias_match){
					pred.w = biases[2*n]/norm;
					pred.h = biases[2*n+1]/norm;
				}

				RegBox truth_shift = truth;
				if(biases_x[n] == 0 && biases_y[n] == 0){
					truth_shift.x = 0;
					truth_shift.y = 0;
				}

				if(truth.angle < -50){
					cv::RotatedRect rect(cv::Point(pred.x, pred.y), cv::Size(pred.w, pred.h), pred.angle * 180 / M_PI);
					cv::Rect br = rect.boundingRect();
					pred.w = br.width;
					pred.h = br.height;
				}
				if(biases_x[n] == 0 && biases_y[n] == 0){
					pred.x = 0;
					pred.y = 0;
				}
				float iou = box_iou(pred, truth_shift);
				if (iou > best_iou){
					best_index = index;
					best_iou = iou;
					best_n = n;
				}
			}
			float iou = delta_region_box(truth, top_data, biases, biases_x, biases_y, best_n, best_index, i, j, width_, height_, bottom_diff, 1, coords_, norm);
			#pragma omp critical
			{
				if(bottom_diff[best_index + coords_] > 0)
					loss[0] -= (0 - top_data[best_index + coords_]) * (0 - top_data[best_index + coords_]);
					loss[0] += (iou - top_data[best_index + coords_]) * (iou - top_data[best_index + coords_]);
			}
			bottom_diff[best_index + coords_] = - object_scale * (iou - top_data[best_index + coords_]) * logistic_gradient(top_data[best_index + coords_]);
			//std::cout << "Delta-objectness: " << bottom_diff[best_index + coords_] << " - " << x[best_index + coords_] << "\n";
			delta_region_class(top_data, bottom_diff, best_index + coordsp, truth.class_no, classes_, 1);
		}
	}
}

template<typename Dtype>
void RegionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
		const vector<Blob<Dtype> *> &bottom) {
	//nothing to do ...
}

INSTANTIATE_CLASS(RegionLayer);

REGISTER_LAYER_CLASS(Region);

}  // namespace caffe
