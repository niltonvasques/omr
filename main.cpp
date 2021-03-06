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

Mat img;
int thresh = 100;
int max_thresh = 255;

void thresh_callback(int, void*);
void clean_callback(int, void*);

int range = 200; 

#define MAX_WIDTH 600.0

int main(int argc, char* argv[]){

  const char* filename = argc >= 2 ? argv[1] : "example.jpg";
  img = imread(filename, 0);
  //imshow("original", img);
  Size new_size(MAX_WIDTH, (MAX_WIDTH/img.size().width) * img.size().height);
  resize(img, img, new_size);

  //vector<vector<Point> > squares;
  //find_squares(img, squares);

  // Apply Histogram Equalization
//  equalizeHist( img, img );
  //imshow("original", img);
  //cleanup_image(img, img);

  //imshow("bitwise", img);
  // Create Window
  char* source_window = "Source";
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, img );
  createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
  thresh_callback( 0, 0 );
  //createTrackbar( " Canny thresh:", "Source", &range, 255, clean_callback );
  //clean_callback( 0, 0 );
  ////imshow("original", img);

  waitKey();

  return 0;
}

void clean_callback(int, void*){
  if(range % 2 == 0) range++;
  cv::Size size(3,3);  
  Mat dst;
  cv::GaussianBlur(img,dst,size,0);  
  //imshow("blur", img);
  //imshow("thresh", img);
  cv::bitwise_not(dst, dst);    
  threshold( dst, dst, range, 255,0 );
  //Canny(dst, dst, thresh, thresh * 2, 3);
  //adaptiveThreshold(dst, dst,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);  
  imshow("Source", dst);

  ///Mat img3;
  ///cvtColor(drawing,img3, CV_RGB2GRAY);  
  vector<Vec4i> lines = search_lines(dst); 

  // DEBUG PURPOSE
  cv::Mat img2;  
  cvtColor(dst,img2, CV_GRAY2RGB);  

  for(int i = 0; i < lines.size(); i++)
    line(img2, Point(lines[i][0], lines[i][1]), Point(lines[i][2],
          lines[i][3]), Scalar(0,0,255), 3, CV_AA);
  //imshow("lines",img2);

  //Find corners
  vector<cv::Point2f> corners = find_corners(img2, lines);

  //DEBUG CORNERS
  for(int i = 0; i < corners.size(); i++)
    circle( img2, corners[i], 3, Scalar(0,255,0), -1, 8, 0 );
  imshow("corners",img2);

  // Get mass center  
  cv::Point2f center = compute_mass_center(corners);
  for (int i = 0; i < corners.size(); i++)  
    center += corners[i];  
  center *= (1. / corners.size()); 

  circle( img2, center, 3, Scalar(255,255,0), -1, 8, 0 );
  imshow("mass",img2);

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

  rectangle(img, r, Scalar(255, 0, 255));
  imshow("rect",img);
}

void thresh_callback(int, void*){
  
  vector<vector<Point> > contours;
  vector<vector<Point> > biggest;
  vector<Vec4i> hierarchy;
  Mat edges;
  RNG rng(12345);
  Canny(img, edges, thresh, thresh * 2, 3);
  // Find contours
  findContours( edges, contours, hierarchy, CV_RETR_EXTERNAL,
      CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  sort(contours.begin(), contours.end(), compareContourAreas);
  biggest.pb(contours[contours.size()-1]);

  /// Draw contours
  Mat drawing = Mat::zeros( edges.size(), CV_8UC3 );
  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
  //drawContours( edges, contours, -1, color, 2, 8, hierarchy, 0, Point() );
  drawContours( drawing, biggest, -1, color, 2 );

  //vector<vector<Point> > hull(1);
  //Mat tmp(biggest[0]);
  //convexHull( tmp, hull[0], false );
  //printf("biggest size: %d\n", biggest[0].size());
  //drawContours( drawing, hull, -1, Scalar(255,0,0), 2 );
  // Show in a window
  imshow( "Source", drawing );

  Mat normalized = wrap(img, biggest[0]);
  imshow("wrap and cut", normalized);

  cleanup_image(normalized, normalized);
  cvtColor(normalized, drawing, CV_GRAY2RGB);

  int cols = 21;
  int rows = 29;
  double w_step = normalized.size().width / 21.0; 
  double h_step = normalized.size().height / 29.0; 
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      double x = j * w_step;
      double y = i * h_step;

      Rect rect(x,y,w_step,h_step);  
      Mat submat = normalized(rect);  
      Rect rect2(w_step*0.2,h_step*0.2,w_step*0.6,h_step*0.6);  
      submat = submat(rect2);  
      double p =(double)countNonZero(submat)/(submat.size().width*submat.size().height);  
      rectangle(drawing, rect, Scalar(255, 255, 255));
      if(p>=0.5){  
        //max = p;  
        //ind = j;  
        //rectangle(src, c, averR, Scalar(255,0,255), 3, 8, 0);  
        rectangle(drawing, rect, Scalar(255, 0, 255));
        cout << p << endl;
      }  
    }
  }
  imshow("founded", drawing);

  //Mat img4;
  //cvtColor(drawing,img4, CV_RGB2GRAY);  
  //vector<Vec4i> lines = search_lines(img4); 
  //for(int i = 0; i < lines.size(); i++)
  //  line(drawing, Point(lines[i][0], lines[i][1]), Point(lines[i][2],
  //        lines[i][3]), Scalar(0,0,255), 3, CV_AA);
  //imshow("lines",drawing);

  // you could also reuse img1 here
  //Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);

  //// CV_FILLED fills the connected components found
  //drawContours(mask, biggest, -1, Scalar(255), CV_FILLED);
  //imshow("Mask", mask);

  //// let's create a new image now
  //Mat crop(img.rows, img.cols, CV_8UC3);

  //// set background to green
  //crop.setTo(Scalar(0,255,0));

  //// and copy the magic apple
  //img.copyTo(crop, mask);

  //// normalize so imwrite(...)/imshow(...) shows the mask correctly!
  //normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);

  //imshow("cropped", crop);

  //Rect r = boundingRect(biggest[0]); 

  //cout << r << endl;
  //Mat submat = img(r);  
  ////circle( submat, Point(10,10), 3, Scalar(0,0,0), -1, 8, 0 );
  //imshow("cropped2", submat);
  //cv::Mat quad = cv::Mat::zeros(r.height, r.width, CV_8UC3);  
  //// Corners of the destination image  
  //std::vector<cv::Point2f> quad_pts;  
  //quad_pts.push_back(cv::Point2f(0, 0));  
  //quad_pts.push_back(cv::Point2f(quad.cols, 0));  
  //quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));  
  //quad_pts.push_back(cv::Point2f(0, quad.rows));  

  //cv::Mat img3, quad;  
  //cvtColor(img,img3, CV_GRAY2RGB); 

  //fix_perspective(img3, quad, corners, quad_pts);

  //imshow("transform",quad);

  /////imshow("cvtColor", img3);

  /////Mat img3;
  /////cvtColor(drawing,img3, CV_RGB2GRAY);  
  //vector<Vec4i> lines = search_lines(edges); 

  //// DEBUG PURPOSE
  //cv::Mat img2;  
  //cvtColor(img,img2, CV_GRAY2RGB);  

  //for(int i = 0; i < lines.size(); i++)
  //  line(img2, Point(lines[i][0], lines[i][1]), Point(lines[i][2],
  //        lines[i][3]), Scalar(0,0,255), 3, CV_AA);
  //imshow("lines",img2);

  ////Find corners
  //vector<cv::Point2f> corners = find_corners(img, lines);

  ////DEBUG CORNERS
  //for(int i = 0; i < corners.size(); i++)
  //  circle( img2, corners[i], 3, Scalar(0,255,0), -1, 8, 0 );
  ////imshow("corners",img2);

  //// Get mass center  
  //cv::Point2f center = compute_mass_center(corners);
  //for (int i = 0; i < corners.size(); i++)  
  //  center += corners[i];  
  //center *= (1. / corners.size()); 

  ////circle( img2, center, 3, Scalar(255,255,0), -1, 8, 0 );
  ////imshow("mass",img2);

  //sortCorners(corners, center); 

  //Rect r = boundingRect(corners); 
  //cout << r << endl;
  //cv::Mat quad = cv::Mat::zeros(r.height, r.width, CV_8UC3);  
  //// Corners of the destination image  
  //std::vector<cv::Point2f> quad_pts;  
  //quad_pts.push_back(cv::Point2f(0, 0));  
  //quad_pts.push_back(cv::Point2f(quad.cols, 0));  
  //quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));  
  //quad_pts.push_back(cv::Point2f(0, quad.rows));  

  //cv::Mat img3;  
  //cvtColor(img,img3, CV_GRAY2RGB); 

  //fix_perspective(img3, quad, corners, quad_pts);

  ////imshow("transform",quad);

  //vector<Vec3f> circles = find_circles(quad);
  //for( size_t i = 0; i < circles.size(); i++ ){  
  //  Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
  //  circle( quad, center, 3, Scalar(0,255,0), -1, 8, 0 );
  //}
  ////imshow("circles",quad);

  ////waitKey();

  //vector<int> markers = find_markers(quad, circles);
  //for(int i = 0; i < markers.size(); i++){
  //  printf("%d: %c\n", i+1, 'A'+markers[i]);  
  //}

  ////// circle outline*/  
  //imshow("circles",quad);
}
