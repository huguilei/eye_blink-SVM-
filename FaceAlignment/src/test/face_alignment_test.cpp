/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face alignment, the
 * face alignment method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "cv.h"
#include "highgui.h"

#include "face_detection.h"
#include "face_alignment.h"
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;
#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif

IplConvKernel*  kernel;
IplImage *diff;
CvCapture   *capture;
struct ret
{
	float out[1][118];
};

//ret alivedetector(String str);
ret LBP_59(int count);
int main(int argc, char** argv)
{
	CvSVM svm;
	char key='a';
	// Initialize face detection model
	IplImage *frame, *eye_template, *template_match, *gray, *prev, *nose_template, *nose_template_match;
	IplImage framme;
	ret result;
	svm.load("E:\\vsfile\\jinglun\\eyeblink_SVM\\eyeblink_SVM\\svmalive-version3.xml");
	seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);
	seeta::FaceAlignment point_detector((MODEL_DIR + "seeta_fa_v1.1.bin").c_str());

	capture = cvCaptureFromCAM(0);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 600);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 500);
	frame = cvQueryFrame(capture);
	cvNamedWindow("video", CV_WINDOW_NORMAL);
	int count = 0;
	ofstream outfile1("d:\\blink.txt");
	while (key != 's')
	{

		//Mat framee = imread("87.bmp");
		frame = cvQueryFrame(capture);
		if (!frame) break;
		cvFlip(frame, frame, 1);
		IplImage *img_grayscale = cvCreateImage(cvGetSize(frame), frame->depth, 1);
		cvCvtColor(frame, img_grayscale, CV_BGR2GRAY);
		
		if (img_grayscale == NULL)
		{
			return 0;
		}
		/* Always check if frame exists */
		if (!frame) break;
		int pts_num = 5;
		int im_width = img_grayscale->width;
		int im_height = img_grayscale->height;
		unsigned char* data = new unsigned char[im_width * im_height];
		unsigned char* data_ptr = data;
		unsigned char* image_data_ptr = (unsigned char*)img_grayscale->imageData;
		int h = 0;
		for (h = 0; h < im_height; h++) {
			memcpy(data_ptr, image_data_ptr, im_width);
			data_ptr += im_width;
			image_data_ptr += img_grayscale->widthStep;
		}
		seeta::ImageData image_data;
		image_data.data = data;
		image_data.width = im_width;
		image_data.height = im_height;
		image_data.num_channels = 1;
		seeta::FacialLandmark points[5];
		// Detect faces
		std::vector<seeta::FaceInfo> faces = detector.Detect(image_data);
		int32_t face_num = static_cast<int32_t>(faces.size());
		CvScalar p;
		//IplImage *res;
		if (face_num > 0)
		{
			point_detector.PointDetectLandmarks(image_data, faces[0], points);

			int width = (points[1].x - points[0].x) / 3;
			int height = width / 2;
			//float feature[1][59];
			//cout << width << endl;		
			Rect eyefield(points[0].x - width / 2, points[0].y - height / 2, width , height );
			IplImage* res = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
			cvSetImageROI(frame, eyefield);
			//提取ROI  
			cvCopy(frame, res);

			//取消设置  
			cvResetImageROI(frame);
			cvRectangle(frame, cvPoint(points[0].x - width / 2, points[0].y - height / 2), cvPoint(points[0].x + width / 2, points[0].y + height / 2), Scalar(0, 0, 255), 1, 1, 0);
			cvShowImage("res", res);
			waitKey(10);
			outfile1 << count;
			outfile1 << '\n';
			String name = to_string(count) + ".bmp";
			cvSaveImage(name.c_str(), res);
			//cvFlip(res, res, -1);
			//cout << count << endl;
			if (count >=2)
			{
				result = LBP_59(count);
				for (int j = 0; j < 118; j++)
				{
					if (j == 59)
					{
						outfile1 << endl;
						outfile1 << "fushu:";
						outfile1 << endl;
					}
					outfile1 << result.out[0][j];
					outfile1 << ' ';
				}
				outfile1 << '\n';
				CvMat testDataMat = cvMat(1, 118, CV_32FC1, result.out);
				float response = (float)svm.predict(&testDataMat);
				if (response == 1)
				{
					cout << "ceshishuju:" << count << endl;
				}
			}
			count++;
		}
		//String name_all = to_string(count) + ".jpg";
		cvShowImage("video", frame);
		//cvSaveImage(name_all.c_str(), frame);
		waitKey(10);

	}
	outfile1.close();
	return 0;
}
