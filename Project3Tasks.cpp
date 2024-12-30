
// Project3Tasks.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <fstream>
#include <cstdint>
#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <strsafe.h>
#include <sstream>
#include <math.h>


using namespace cv;

using namespace std;
int classify(cv::Mat& originalimage);
int collectTrainingData();

// Driver code
int main(int argc, char** argv)
{
	Mat image = imread("C:/Users/srika/Downloads/test2.jpg");

	Mat img;
	image.copyTo(img);
	
	// Error Handling
	if (image.empty()) {
		cout << "Image File "
			<< "Not Found" << endl;

		// wait for any key press
		cin.get();
		return -1;
	}

	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);  //gray conversion
	
	//blur the image before applying threshold, here i am using gaussian blur
	
	Mat blur;
	cv::GaussianBlur(image, blur, cv::Size(5, 5), 0);





	//saturating and unsaturating pixels
	

	Mat saturate_image;
	blur.copyTo(saturate_image);
	int r = saturate_image.rows;
	int c = saturate_image.cols;
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {

			if (saturate_image.at<uint8_t>(i, j) == 255) {
				saturate_image.at<uint8_t>(i, j) = saturate_image.at<uint8_t>(i, j) * 0;
			}
			else {
				saturate_image.at<uint8_t>(i, j) = saturate_image.at<uint8_t>(i, j) * 1;
			}
		}
	}
	
	
	
	//Applying thresholding on saturated image to convert it to a binary image

	Mat threshold;
	saturate_image.copyTo(threshold);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if (threshold.at<uint8_t>(i, j) > 75) {
				threshold.at<uint8_t>(i, j) = threshold.at<uint8_t>(i, j) * 0;
			}
			else {
				threshold.at<uint8_t>(i, j) = 255;
			}
		}
	}
	
	
	
	//Applying morphological close and open operators on binary image to remove noise and holes in the binary image.
	
	Mat src, dest;
	threshold.copyTo(src);
	Mat M = cv::Mat::ones(5, 5, CV_8U);
	Point anchor = cv::Point(-1, -1);
	morphologyEx(src, dest, MORPH_OPEN, M, anchor, 1,
		BORDER_CONSTANT, morphologyDefaultBorderValue());

	dest.copyTo(src);
	morphologyEx(src, dest, MORPH_CLOSE, M);


	
	//Getting connected components of the binary image
	
	
	Mat connect;
	connectedComponents(dest, connect);
	Mat imLabelClone = connect.clone();
	//Find min and max pixel values and their location in the image
	Point minPos, maxPos;
	double min, max;
	minMaxLoc(imLabelClone, &min, &max, &minPos, &maxPos);
	imLabelClone = 255 * (imLabelClone - min) / (max - min); //Normalize the image so that min values in 0 and max value is 255
	imLabelClone.convertTo(imLabelClone, CV_8U); //Convert image to 8 bits
	Mat imLabelCloneColorMap;
	//Apply color map to images
	applyColorMap(imLabelClone, imLabelCloneColorMap, COLORMAP_JET);
	
	
	
	
	//Compute features for each region by finding contours and moments
	
	

	Mat output;
	dest.copyTo(output);
	vector<vector<Point> > contours;
	findContours(output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	vector<Moments> mu(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i]);
	}
	RNG rng(12345);
	vector<RotatedRect> minRectangle(contours.size());
	vector<RotatedRect> minEllipse(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		minRectangle[i] = minAreaRect(contours[i]);
		if (contours[i].size() > 5)
		{
			minEllipse[i] = fitEllipse(contours[i]);
		}
	}
	Mat drawing;
	img.copyTo(drawing);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 255, 0);
		//Contour
		drawContours(drawing, contours, (int)i, color);
		//Ellipse
		//Ellipse(drawing, minEllipse[i], color, 2);
		//Rotated rectangle
		Point2f rect_points[4];
		minRectangle[i].points(rect_points);

		float  angle = minRectangle[i].angle; // angle
		// read center of rotated rectangle
		cv::Point2f center = minRectangle[i].center; // center

		for (int j = 0; j < 4; j++)
		{
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color);
		}

		std::stringstream ss;   ss << angle; // convert float to string
		cv::circle(drawing, center, 5, color); // draw center
		cv::putText(drawing, ss.str(), center + cv::Point2f(-25, 25), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color); // print angle
	}
	
	
	//Collect training data for each label when u enter 'N', it starts capturing images for each label as below
	
	int l_count = 4;
	String Labels[4] = { "spoon", "box", "book", "shoe" };	//here l_count is no of labels (folders)
	collectTrainingData();
	
	
	
	//Nearest neighbour classifier and KNN classifier
	Mat originalimage = imread("box/1.jpeg");
	cv::cvtColor(originalimage, originalimage, cv::COLOR_BGR2GRAY);
	resize(originalimage, originalimage, Size(500, 500), INTER_LINEAR);
	int index1 = classify(originalimage);

	// confusion matrix 

	
	//imshow("Window Name", output);
	//imshow("Window Name1", drawing);

	//printf("%d,%d", image.rows, image.cols);

	// Wait for any keystroke
	waitKey(0);
	return 0;
}



int collectTrainingData() {
	
	//Classify new images
	//here we can implement nearest neighbours for any particular image we want to classify by only using any one label which is near to u.
	//for KNN Classifier we will be using all labels and using K value for classifying which label (class) the image belong to.
	//so i have put both of them in same code with minor chnages while executing.

	int l_count = 4;
	String Labels[4] = { "spoon", "box", "book", "shoe" };	//here l_count is no of labels (folders)
	for (int i = 0; i < l_count; i++) {   // create folders
		string folderName = Labels[i];
		string folderCreateCommand = "mkdir " + folderName;
		system(folderCreateCommand.c_str());
	}

	// capture images for each label
	for (int i = 0; i < l_count; i++) {
		char input;
		cin >> input;
		if (input != 'N') {
			printf("Wrong Input..........");
			continue;
		}

		int captureimagescount = 3;   //no of images we can capture for each label (folder)
		int x = 0;

		while (x < captureimagescount) {
			cv::VideoCapture video(0);
			if (!video.isOpened()) { return -1; }
			cv::Mat frame;

			int frame_width = (int)video.get(cv::CAP_PROP_FRAME_WIDTH);
			int frame_height = (int)video.get(cv::CAP_PROP_FRAME_HEIGHT);

			video.read(frame);
			if (frame.empty()) {
				printf("frame is empty\n");
				break;
			}
			cvtColor(frame, frame, COLOR_BGR2GRAY);
			cv::imshow("Video feed", frame);

			std::string no = std::to_string(x);
			string tmp = Labels[i] + "/";
			tmp = tmp + no + ".jpg";

			resize(frame, frame, Size(100, 100), INTER_LINEAR);
			cout << tmp;
			cv::imwrite(tmp, frame);

			x++;
			char key = cv::waitKey(1);
			if (key == 'q') {
				break;
			}
		}
	}
}







int classify(cv::Mat& originalimage) {
	
	
	//Classify new images
	//here we can implement nearest neighbours for any particular image we want to classify by only using any one label which is near to u.
	//for KNN Classifier we will be using all labels and using K value for classifying which label (class) the image belong to.
	//so i have put both of them in same code with minor chnages while executing.
	
	

	int l_count = 4;
	String Labels[4] = { "spoon", "box", "book", "shoe" };  //here l_count is no of labels (folders)
	
	int k = 4, f = 0;
	int dis[4]; string dist[4];

	for (int i = 0; i < l_count; i++) {
		for (int j = 0; j < 4; j++) {
			std::string no = std::to_string(j);

			string tmp = Labels[i] + "/";
			tmp = tmp + no + ".jpeg";
			Mat dataset_image = imread(tmp);
			cv::cvtColor(dataset_image, dataset_image, cv::COLOR_BGR2GRAY);
			resize(dataset_image, dataset_image, Size(500, 500), INTER_LINEAR);

			//image which is need to be classified to which label (class) it belongs here i taken from storage but we can try capturing a frame
			//as i don't had hardware for that i stored it and used from there.
			//Mat originalimage = imread("box/1.jpeg");
			//cv::cvtColor(originalimage, originalimage, cv::COLOR_BGR2GRAY);
			//resize(originalimage, originalimage, Size(500, 500), INTER_LINEAR);

			cv::imshow("Video feed", dataset_image);
			waitKey(0);

			int distance = 0;
			for (int p = 0; p < originalimage.rows; p++) {
				for (int q = 0; q < originalimage.cols; q++) {

					int x = originalimage.at<uint8_t>(p, q);
					int y = dataset_image.at<uint8_t>(p, q);

					distance = distance + sqrt((x - y) * (x - y));
				}
			}
			cout << Labels[i] << "\t";
			printf("%d\t%d\n", i, distance);

			if (f < k) {
				dis[f] = distance; dist[f] = Labels[i]; f++;
			}
			else {
				int min = dis[0], index = 0;
				int i1 = 0;
				while (i1 < k) {
					if (dis[i1] > distance && dis[i1] > min) {
						index = i1; min = distance;
					}
					i1++;
				}
				dis[index] = distance; dist[index] = Labels[i];
			}
		}
	}

	int labelcount[4] = { 0,0,0,0 }; //here size is same as l_count
	int index = 0, labelmax = -1;

	for (int i = 0; i < k; i++) {
		string g = dist[i];
		for (int j = 0; j < l_count; j++) {

			if (Labels[j] == g) {
				labelcount[j]++;
				if (labelcount[j] > labelmax) { index = j; labelmax = labelcount[j]; }
				cout << Labels[j] << "\n";
				break;
			}
		}
	}
	cout << Labels[index] << "\n";

	return index;  //will return to which class the input image should belongs to.
}
