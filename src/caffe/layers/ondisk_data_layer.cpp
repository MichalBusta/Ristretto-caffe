#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <random>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/locale.hpp>

#include "caffe/layers/ondisk_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

#define LABES_WIDTH 6

bool doLineDetection = false;

namespace caffe {

static double distance_to_Line(cv::Point line_start, cv::Point line_end, cv::Point point)
{
	double normalLength = hypot(line_end.x - line_start.x, line_end.y - line_start.y);
	double distance = (double)((point.x - line_start.x) * (line_end.y - line_start.y) - (point.y - line_start.y) * (line_end.x - line_start.x)) / normalLength;
	return distance;
}

static void assign_lines(cv::Mat srcImg, std::vector<std::vector<float>>& word_gto, std::vector<std::wstring>& txt, std::vector<cv::RotatedRect>& lines, std::vector<std::wstring>& lines_txt)
{

	float normFactor = sqrtf(srcImg.rows * srcImg.rows + srcImg.cols * srcImg.cols);


	std::vector<cv::RotatedRect> rects;
	std::vector<cv::RotatedRect> rects_o;
	for(size_t i = 0; i < word_gto.size(); i++)
	{

		cv::RotatedRect rr( cv::Point(word_gto[i][0] * srcImg.cols, word_gto[i][1] * srcImg.rows),
				cv::Size(word_gto[i][2] * normFactor + word_gto[i][3] * normFactor, word_gto[i][3] * normFactor), word_gto[i][4] * 180 / 3.14 );
		rects.push_back(rr);

		cv::RotatedRect rr2( cv::Point(word_gto[i][0] * srcImg.cols, word_gto[i][1] * srcImg.rows),
						cv::Size(word_gto[i][2] * normFactor, word_gto[i][3] * normFactor), word_gto[i][4] * 180 / 3.14 );
		rects_o.push_back(rr2);
	}

	for( size_t i = 0; i < rects.size(); i++ ){
		cv::RotatedRect& rect1 = rects[i];
		cv::RotatedRect& rect1o = rects_o[i];
		cv::Rect test = rect1o.boundingRect();
		if(test.x == 0 || test.y == 0 || (test.x + test.width) >= (srcImg.cols - 1) || (test.x + test.width) >= (srcImg.rows - 1))
			continue;

		if(rect1.center.x <= 0 || rect1.center.y <= 0)
			continue;

		if(fabs(rect1.angle) > 45.f ) //TODO - line merging for vertical lines
			continue;

		if(word_gto[i][5] == -1)
			continue;


		for( size_t j = i + 1; j < rects.size(); j++ ){

			if(word_gto[j][5] == -1)
				continue;

			cv::RotatedRect& rect2 = rects[j];
			cv::RotatedRect& rect2o = rects_o[j];
			if( fabs(rect1.angle - rect2.angle) > 45.f / 6){
				continue;
			}

			test = rect2o.boundingRect();
			if(test.x <= 0 || test.y <= 0 || (test.x + test.width) >= (srcImg.cols - 1) || (test.x + test.width) >= (srcImg.rows - 1))
				continue;

			std::vector<cv::Point2f> vertices;
			int ret = rotatedRectangleIntersection(rect1, rect2, vertices);
			if(ret != 0){

				cv::Point2f pts[4];
				rect1.points(pts);

				cv::Point2f pts2[4];
				rect2.points(pts2);

				float height = std::min(rect1.size.height, rect2.size.height) / 3.0;

				double dist = distance_to_Line(pts[3], pts[0], pts2[0]);
				dist = std::max(dist, distance_to_Line(pts[3], pts[0], pts2[3]));
				if(fabs(dist) > height){
					continue;
				}

				std::wstring line_txt;
				if( pts[0].x < pts2[0].x )
					line_txt = txt[i] + L" " + txt[j];
				else
					line_txt = txt[j] + L" " + txt[i];

				if(line_txt.size() < 7)
					continue;

				std::vector<cv::Point> cnt;
				rect1o.points(pts);
				cnt.push_back(pts[0]); cnt.push_back(pts[1]); cnt.push_back(pts[2]); cnt.push_back(pts[3]);
				rect2o.points(pts2);
				cnt.push_back(pts2[0]); cnt.push_back(pts2[1]); cnt.push_back(pts2[2]); cnt.push_back(pts2[3]);

				cv::RotatedRect rect = cv::minAreaRect(cnt);
				if(rect.size.height > rect.size.width){
					std::swap(rect.size.height, rect.size.width);
					rect.angle += 90;
				}

				if(fabs(rect.angle - rect1.angle) > 45.f / 6){
					//std::cout << "Rect skip\n";
					continue;
				}
#ifdef DEBUG
				cv::rectangle(srcImg, rect2.boundingRect(), cv::Scalar(255, 0, 0));
				cv::rectangle(srcImg, rect1.boundingRect(), cv::Scalar(0, 255, 0));
				cv::rectangle(srcImg, rect.boundingRect(), cv::Scalar(0, 255, 0));
				cv::imshow("line-rects", srcImg);
				cv::waitKey(0);
#endif
			  lines.push_back( rect );
			  lines_txt.push_back(line_txt);
			}
		}
	}

	if(doLineDetection){
		for(int i = 0; i < lines.size(); i++){
			word_gto.push_back(std::vector<float>());

			float x = lines[i].center.x / (float) srcImg.cols;
			word_gto.back().push_back(x);
			float y = lines[i].center.y / (float) srcImg.rows;
			word_gto.back().push_back(y);
			float w = lines[i].size.width / normFactor;
			word_gto.back().push_back(w);
			float h = lines[i].size.height / normFactor;
			word_gto.back().push_back(h);
			float angle = lines[i].angle / 180.0 * M_PI;
			word_gto.back().push_back(angle);
			word_gto.back().push_back(1);

			txt.push_back(lines_txt[i]);
		}
	}
}

template <typename Dtype>
OnDiskDataLayer<Dtype>::OnDiskDataLayer(const LayerParameter& param)
  : BaseDataLayer<Dtype>(param), batch_size(1)
{
	std::string input_file = param.data_param().source();
	std::ifstream fr(input_file.c_str());
	std::string line;
	while (std::getline(fr, line))
	{
		boost::trim(line);
		file_list.push_back(line);
		//std::cout << line << "\n";
	}

	boost::filesystem::path p(input_file);
	boost::filesystem::path dir = p.parent_path();
	data_dir = dir.string();

	srand (time(NULL));
}

template <typename Dtype>
OnDiskDataLayer<Dtype>::~OnDiskDataLayer() {

}

template <typename Dtype>
std::string OnDiskDataLayer<Dtype>::get_image_file_name(int bid){
	return src_images[bid];
}


template <typename Dtype>
int OnDiskDataLayer<Dtype>::get_crop(int bid, int side)
{
	if(side == 0)
		return crops[bid].x;
	else if(side == 1)
		return crops[bid].y;
	else if(side == 2)
		return crops[bid].x + crops[bid].width;
	else if(side == 3)
		return crops[bid].y + crops[bid].height;
	return 0;
}

template <typename Dtype>
void OnDiskDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
	const int num_top = top.size();
	const InputParameter& param = this->layer_param_.input_param();
	const int num_shape = param.shape_size();
	CHECK(num_shape == 0 || num_shape == 1 || num_shape == num_top)
			<< "Must specify 'shape' once, once per top blob, or not at all: "
			<< num_top << " tops vs. " << num_shape << " shapes.";
	if (num_shape > 0) {
		for (int i = 0; i < num_top; ++i) {
			const int shape_index = (param.shape_size() == 1) ? 0 : i;
			top[i]->Reshape(param.shape(shape_index));
		}
	}
}

template <typename Dtype>
void OnDiskDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top)
{
	batch_size = top[0]->num();
	height_ = top[0]->height();
	width_ = top[0]->width();
	channels_ = top[0]->channels();
}

template <typename Dtype>
void wrapInputLayers(Blob<Dtype>* input_layer, std::vector<cv::Mat>* input_channels, int nImages) {

	int width = input_layer->width();
	int height = input_layer->height();
	Dtype* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels() * nImages; ++i) {

		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

static void preprocess_multi(const std::vector<cv::Mat>& imgs, std::vector<cv::Mat>* input_channels, cv::Size input_geometry, int channels) {

	/* Convert the input image to the input image format of the network. */
	for( size_t i = 0; i < imgs.size(); i++ )
	{
		const cv::Mat& img = imgs[i];
		cv::Mat sample;

		if(img.type() == CV_8UC3 && channels == 1)
			cv::cvtColor(img, sample, CV_BGR2GRAY);
		else if(img.type() == CV_8UC1 && channels == 3)
			cv::cvtColor(img, sample, CV_GRAY2BGR);
		else
			sample = img;

		cv::Mat sample_resized;
		if (sample.size() != input_geometry)
			cv::resize(sample, sample_resized, input_geometry);
		else
			sample_resized = sample;

		cv::Mat sample_float;
		if(channels == 3){
			sample_resized.convertTo(sample_float, CV_32FC3, 1 / 128.0f, -1.0f);
			std::vector<cv::Mat> ichannels;
			cv::split(sample_float, ichannels);
			for( int j = 0; j <  ichannels.size(); j++){
				//std::cout << "Copy data " << i*channels + j << std::endl;
				ichannels[j].copyTo((*input_channels)[i*channels + j]);
			}
		}else{
			sample_resized.convertTo(sample_float, CV_32FC1, 1 / 128.0f, -1.0f);
			sample_float.copyTo((*input_channels)[i]);
		}

	}
}

static void read_txt_gt(std::string gt_file, bool skip_slash, std::vector<std::vector<float>>& boxes, std::vector<std::wstring>& txt)
{

	std::ifstream fr(gt_file.c_str());
	std::string line;

	while (std::getline(fr, line))
	{
		if(line[0] == '#')
			continue;

		std::wstring wline = boost::locale::conv::utf_to_utf<wchar_t>(line);;

		std::vector<std::wstring> splitLine;
		boost::split(splitLine, wline, boost::is_any_of(L" "));

		if(splitLine.size() < 7){
			std::cout << gt_file << std::endl;
			std::wcout << L"bad gt line: " << wline << std::endl;
			continue;
		}

		std::wstring text = splitLine[6];
		boxes.push_back(std::vector<float>());

		int cls = boost::lexical_cast<int>(splitLine[0]);
		if(text[0] == L'#' && skip_slash){
			cls = -1;
		}else if(cls > 2){
		  cls = 0;
    }else if (cls < 0 )
    	cls = 1;

		float x = boost::lexical_cast<float>(splitLine[1]);
		boxes.back().push_back(x);
		float y = boost::lexical_cast<float>(splitLine[2]);
		boxes.back().push_back(y);
		float w = boost::lexical_cast<float>(splitLine[3]);
		boxes.back().push_back(w);
		float h = boost::lexical_cast<float>(splitLine[4]);
		boxes.back().push_back(h);
		float angle = boost::lexical_cast<float>(splitLine[5]);
		boxes.back().push_back(angle);
		boxes.back().push_back(cls);

		txt.push_back(text);
	}
}

void transform_boxes(cv::Mat& im, cv::Mat& scaled, std::vector<std::vector<float>>& word_gto)
{

	cv::Size image_size = scaled.size();
	cv::Size o_size = im.size();

	float normo = sqrtf(im.rows * im.rows + im.cols * im.cols );
	float normo2 = sqrtf(scaled.rows * scaled.rows + scaled.cols * scaled.cols );
	float scalex = o_size.width / (float) image_size.width;
	float scaley = o_size.height / (float) image_size.height;

	for(size_t i = 0; i < word_gto.size(); i++){
		std::vector<float>& gt = word_gto[i];
		float angle = gt[4];
		if(angle < -50)
			angle = 0;
		cv::RotatedRect gtbox(cv::Point(gt[0] * o_size.width, gt[1] * o_size.height), cv::Size(gt[2] * normo, gt[3] * normo), angle * 180 / M_PI);
		cv::Point2f pts[4];
		gtbox.points(pts);

		for(size_t p = 0; p < 4; p++ ){
			pts[p].x /= scalex;
			pts[p].y /= scaley;
		}

		cv::Point2f dh = pts[0] - pts[1];
		cv::Point2f dw = pts[1] - pts[2];

		float h = sqrtf(dh.x * dh.x + dh.y * dh.y) / normo2;
		float w = sqrtf(dw.x * dw.x + dw.y * dw.y) / normo2;

		gt[2] = w;
		gt[3] = h;

		if( gt[4] < -50)
			gt[4] = -100;
		else
			gt[4] = atan2((pts[2].y - pts[1].y), pts[2].x - pts[1].x);
	}
}

std::random_device rd;
std::mt19937 rng(rd());

static void random_crop(cv::Mat& img, std::vector<std::vector<float>>& word_gto, std::vector<std::wstring>& txt, float crop_ratio, std::vector<cv::Rect>& crops)
{

	std::uniform_int_distribution<int> uni(0, img.cols * crop_ratio);
	std::uniform_int_distribution<int> uni2(0, img.rows * crop_ratio);

	int xs = uni(rng);
	int xe = img.cols - xs -  uni(rng);

	int ys = uni2(rng);
	int ye =  img.rows - ys -  uni2(rng);

	//std::cout <<  img.cols << "x" << img.rows << " - " << xs << "-" << maxx << "x" << ys << "-" << maxy << std::endl;
	cv::Rect crop_rect(xs, ys, xe - xs, ye - ys);
	cv::Mat crop_img = img(crop_rect);
	crops.push_back(crop_rect);

	float normo = sqrtf(img.cols * img.cols + img.rows * img.rows);
	cv::Size image_size(crop_img.cols, crop_img.rows);
	float normo2 = sqrtf(image_size.width * image_size.width + image_size.height * image_size.height );
	cv::Size o_size(img.cols, img.rows);

	std::vector<std::wstring>::iterator itt = txt.begin();
	for( std::vector<std::vector<float>>::iterator it = word_gto.begin(); it != word_gto.end();) {

		std::vector<float>& gt = *it;
		float angle = gt[4];
		if(angle < -50)
			angle = 0;
		cv::RotatedRect gtbox(cv::Point(gt[0] * o_size.width, gt[1] * o_size.height), cv::Size(gt[2] * normo, gt[3] * normo), angle * 180 / M_PI);
		cv::Point2f pts[4];
		gtbox.points(pts);

		float centerx = 0;
		float centery = 0;
		bool cut_out = false;
		for(size_t p = 0; p < 4; p++ ){
			pts[p].x -= xs;
			if( pts[p].x <= 0)
				cut_out = true;
			pts[p].x = MAX(pts[p].x, 0);
			pts[p].x = MIN(pts[p].x, xe);
			pts[p].y -= ys;
			if( pts[p].y <= 0)
				cut_out = true;
			pts[p].y = MAX(pts[p].y, 0);
			pts[p].y = MIN(pts[p].y, ye);

			centerx += pts[p].x;
			centery += pts[p].y;
		}

		if(cut_out){
			//std::cout << "Box cut out!\n";
			gt[5] = -1;
		}

		cv::Point2f dh = pts[0] - pts[1];
		cv::Point2f dw = pts[1] - pts[2];

		centerx /= 4;
		centery /= 4;

		float h = sqrtf(dh.x * dh.x + dh.y * dh.y) / normo2;
		float w = sqrtf(dw.x * dw.x + dw.y * dw.y) / normo2;

		if(w * normo2 < 2 || h * normo2  < 2){
			gt[5] = -1;
		}

		gt[0] = centerx / image_size.width;
		gt[1] = centery / image_size.height;
		gt[2] = w;
		gt[3] = h;
        if(gt[4] > -50)
		  gt[4] = atan2((pts[2].y - pts[1].y), (pts[2].x - pts[1].x));
		it++;
		itt++;
	}
	img = crop_img;
	if(rand() % 10 < 5)
		cv::GaussianBlur( img, img, cv::Size( 5, 5 ), 0, 0 );
}

template <typename Dtype>
void OnDiskDataLayer<Dtype>::fill_bucket(Blob<Dtype>* imgs, Blob<Dtype>* boxes, int width)
{

}

template <typename Dtype>
void OnDiskDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top)
{
	int files_read = 0;

	srand (time(NULL));

	cv::Size input_geometry(width_, height_);

	std::vector<cv::Mat> scaled_images;
	std::vector<std::vector<std::vector<float>>> batch_boxes;
	std::vector<std::vector<std::wstring>> batch_boxes_text;

	std::vector<std::vector<cv::RotatedRect>> batch_lines;
	std::vector<std::vector<std::wstring>> batch_lines_text;

	batch_boxes.resize(batch_size);
	batch_boxes_text.resize(batch_size);
	batch_lines.resize(batch_size);
	batch_lines_text.resize(batch_size);

	orig_top.resize(batch_size);
	src_images.clear();

	int max_boxes = 0;
	int max_text_length = 0;
	crops.clear();

	int max_lines = 0;
	int max_lines_length = 0;

	while( files_read <  batch_size){
		auto random_integer = rand() % file_list.size();
		if(rand() % 100 < 10)
			random_integer = rand() % std::min(20000, (int) file_list.size());
		std::string file_name = file_list[random_integer];

		if(file_name[0] != '/'){
			file_name = data_dir + "/" + file_name;
		}

		std::string gt_file = boost::replace_all_copy(file_name, ".jpg", ".txt");
		gt_file = boost::replace_all_copy(gt_file, ".png", ".txt");

		std::vector<std::vector<float>>& boxes = batch_boxes[files_read];
		std::vector<std::wstring>& txt = batch_boxes_text[files_read];
		read_txt_gt(gt_file, random_integer < 10000, boxes, txt);

		cv::Mat img = cv::imread(file_name.c_str());
		if( img.rows == 0 || img.cols == 0 ){
			std::cout << "Bad sample: " << file_name << "\n";
			continue;
		}

		random_crop(img, boxes, txt, crop_ratio, crops);

		orig_top[files_read] =  img;
		src_images.push_back(file_name);

		cv::Mat scaled;
		cv::resize(img, scaled, input_geometry);
		scaled_images.push_back(scaled);
		transform_boxes(img, scaled, boxes);

		assign_lines(scaled, boxes, txt, batch_lines[files_read], batch_lines_text[files_read]);

		max_lines = std::max(max_lines,  (int) batch_lines[files_read].size());
		max_boxes = std::max(max_boxes, (int) boxes.size());
		for(size_t k = 0; k < boxes.size(); k++)
			max_text_length = MAX(max_text_length, txt[k].size());
		for(size_t k = 0; k < batch_lines_text[files_read].size(); k++)
			max_lines_length = MAX(max_lines_length, batch_lines_text[files_read][k].size());
		files_read++;
	}


	Blob<Dtype>* input_layer = top[0];
	std::vector<cv::Mat> input_channels;
	wrapInputLayers(input_layer, &input_channels, batch_size);
	preprocess_multi(scaled_images, &input_channels, input_geometry, channels_);

	Blob<Dtype>* input_boxes = top[1];
	input_boxes->Reshape(batch_size, max_boxes, 1, LABES_WIDTH);
	Dtype *top_data2 = input_boxes->mutable_cpu_data();
	memset((void *) top_data2, 0, sizeof(Dtype) * max_boxes * LABES_WIDTH * batch_size);

	Blob<Dtype>* gt_txt_boxes = top[2];
	gt_txt_boxes->Reshape(batch_size, max_boxes, 1, max_text_length);
	Dtype *top_data_txt = gt_txt_boxes->mutable_cpu_data();
	memset((void *) top_data_txt, 0, sizeof(Dtype) * max_boxes * max_text_length * batch_size);

	assert(batch_boxes_text.size() == batch_boxes.size());

	Blob<Dtype>* input_lines = top[3];
	input_lines->Reshape(batch_size, max_lines, 1, 5);
	Dtype *top_lines_data = input_lines->mutable_cpu_data();
	memset((void *) top_lines_data, 0, sizeof(Dtype) * max_lines * 5 * batch_size);

	Blob<Dtype>* gt_txt_lines = top[4];
	gt_txt_lines->Reshape(batch_size, max_lines, 1, max_lines_length);
	Dtype *top_lines_txt = gt_txt_lines->mutable_cpu_data();
	memset((void *) top_lines_txt, 0, sizeof(Dtype) * max_lines * max_lines_length * batch_size);


	for(int i = 0; i < batch_size; i++){

		std::vector<std::vector<float>>& boxes = batch_boxes[i];
		std::vector<std::wstring>& txt = batch_boxes_text[i];
		Dtype *top_data2_batch = top_data2 + i * max_boxes * LABES_WIDTH;
		Dtype *top_data_txt_batch = top_data_txt + i * max_boxes * max_text_length;
		for(size_t j = 0; j < boxes.size(); j++){
			top_data2_batch[j* LABES_WIDTH] = boxes[j][0];
			top_data2_batch[j* LABES_WIDTH + 1] = boxes[j][1];
			top_data2_batch[j* LABES_WIDTH + 2] = boxes[j][2];
			top_data2_batch[j* LABES_WIDTH + 3] = boxes[j][3];
			top_data2_batch[j* LABES_WIDTH + 4] = boxes[j][4];
			top_data2_batch[j* LABES_WIDTH + 5] = boxes[j][5];
			for(size_t t = 0; t < txt[j].size(); t++){
				top_data_txt_batch[j* max_text_length + t] = (int) txt[j][t];
			}
		}

		Dtype *top_lines_batch = top_lines_data + i * max_lines * LABES_WIDTH;
		Dtype *top_lines_txt_batch = top_lines_txt + i * max_lines * max_lines_length;

		std::vector<cv::RotatedRect>& bid_lines = batch_lines[i];
		std::vector<std::wstring>& batch_lines_txt = batch_lines_text[i];
		for(size_t j = 0; j < bid_lines.size(); j++){
			top_lines_batch[j* 5] = bid_lines[j].center.x;
			top_lines_batch[j* 5 + 1] = bid_lines[j].center.y;
			top_lines_batch[j* 5 + 2] = bid_lines[j].size.width;
			top_lines_batch[j* 5 + 3] = bid_lines[j].size.height;
			top_lines_batch[j* 5 + 4] = bid_lines[j].angle;
			for(size_t t = 0; t < batch_lines_txt[j].size(); t++){
				top_lines_txt_batch[j* max_lines_length + t] = (int) batch_lines_txt[j][t];
			}
		}

	}
}

INSTANTIATE_CLASS(OnDiskDataLayer);
REGISTER_LAYER_CLASS(OnDiskData);


}  // namespace caffe
