#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "omr.hpp"

//g++ main.cpp -o main -I /usr/local/include/opencv -lopencv_core -lopencv_imgproc -lopencv_highgui

using namespace cv;
using namespace std;


int main(int argc, char* argv[]){

  Mat img = imread("example.jpg",0);
  //imshow("original", img);

  cleanup_image(img, img);

  //imshow("bitwise", img);

  //imshow("cvtColor", img3);

  vector<Vec4i> lines = search_lines(img); 

  // DEBUG PURPOSE
  cv::Mat img2;  
  cvtColor(img,img2, CV_GRAY2RGB);  

  for(int i = 0; i < lines.size(); i++)
    line(img2, Point(lines[i][0], lines[i][1]), Point(lines[i][2],
          lines[i][3]), Scalar(0,0,255), 3, CV_AA);
  //imshow("lines",img2);

  //Find corners
  vector<cv::Point2f> corners = find_corners(img, lines);

  //DEBUG CORNERS
  for(int i = 0; i < corners.size(); i++)
    circle( img2, corners[i], 3, Scalar(0,255,0), -1, 8, 0 );
  //imshow("corners",img2);

  // Get mass center  
  cv::Point2f center = compute_mass_center(corners);
  for (int i = 0; i < corners.size(); i++)  
    center += corners[i];  
  center *= (1. / corners.size()); 

  //circle( img2, center, 3, Scalar(255,255,0), -1, 8, 0 );
  //imshow("mass",img2);

  sortCorners(corners, center); 

  Rect r = boundingRect(corners); 
  cout << r << endl;
  cv::Mat quad = cv::Mat::zeros(r.height, r.width, CV_8UC3);  
  // Corners of the destination image  
  std::vector<cv::Point2f> quad_pts;  
  quad_pts.push_back(cv::Point2f(0, 0));  
  quad_pts.push_back(cv::Point2f(quad.cols, 0));  
  quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));  
  quad_pts.push_back(cv::Point2f(0, quad.rows));  

  cv::Mat img3;  
  cvtColor(img,img3, CV_GRAY2RGB); 

  fix_perspective(img3, quad, corners, quad_pts);

  //imshow("transform",quad);

  vector<Vec3f> circles = find_circles(quad);
  for( size_t i = 0; i < circles.size(); i++ ){  
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    circle( quad, center, 3, Scalar(0,255,0), -1, 8, 0 );
  }
  //imshow("circles",quad);

  //waitKey();

  vector<int> markers = find_markers(quad, circles);
  for(int i = 0; i < markers.size(); i++){
    printf("%d: %c\n", i+1, 'A'+markers[i]);  
  }

  //// circle outline*/  
  //imshow("circles",quad);
  //waitKey();
  return 0;
}
