#ifndef ___OMR_H___
#define ___OMR_H___

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

#define pb push_back

static inline cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
  int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
  int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

  if (float d = ((float)(x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4)))
  {
    cv::Point2f pt;
    pt.x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
    pt.y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
    return pt;
  }
  else
    return cv::Point2f(-1, -1);
}

static inline bool comparator2(double a,double b){  
  return a<b;  
}  
static inline bool comparator3(Vec3f a,Vec3f b){  
  return a[0]<b[0];  
}  

static inline bool comparator(Point2f a,Point2f b){  
  return a.x<b.x;  
}  
static inline void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)  
{  


  std::vector<cv::Point2f> top, bot;  
  for (int i = 0; i < corners.size(); i++)  
  {  
    if (corners[i].y < center.y)  
      top.push_back(corners[i]);  
    else  
      bot.push_back(corners[i]);  
  }  


  sort(top.begin(),top.end(),comparator);  
  sort(bot.begin(),bot.end(),comparator); 

  cv::Point2f tl = top[0];  
  cv::Point2f tr = top[top.size()-1];  
  cv::Point2f bl = bot[0];  
  cv::Point2f br = bot[bot.size()-1]; 
  corners.clear();  
  corners.push_back(tl);  
  corners.push_back(tr);  
  corners.push_back(br);  
  corners.push_back(bl);  
}  

static inline void cleanup_image(cv::Mat &src, cv::Mat &dst){
  cv::Size size(3,3);  
  cv::GaussianBlur(src,dst,size,0);  
  //imshow("blur", img);
  adaptiveThreshold(src, dst,255,CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,75,10);  
  //imshow("thresh", img);
  cv::bitwise_not(src, dst);    
}

static inline vector<Vec4i> search_lines(cv::Mat &src){
  vector<Vec4i> lines;  

  HoughLinesP(src, lines, 1, CV_PI/180, 80, 400, 10);  
  for( size_t i = 0; i < lines.size(); i++ )  
  {  
    Vec4i l = lines[i];  
  }  
  return lines;
}

static inline vector<cv::Point2f> find_corners(cv::Mat &src, vector<Vec4i> &lines){
  std::vector<cv::Point2f> corners;
  for (int i = 0; i < lines.size(); i++)
  {
    for (int j = i+1; j < lines.size(); j++)
    {
      cv::Point2f pt = computeIntersect(lines[i], lines[j]);
      if (pt.x >= 0 && pt.y >= 0 && pt.x < src.cols && pt.y < src.rows){
        corners.push_back(pt);
      }
    }
  }
  return corners;
}

static inline cv::Point2f compute_mass_center(vector<cv::Point2f> corners){
  cv::Point2f center(0,0);  
  for (int i = 0; i < corners.size(); i++)  
    center += corners[i];  
  center *= (1. / corners.size()); 
  return center;
}

static inline void fix_perspective(cv::Mat &src, cv::Mat &dst,
    vector<cv::Point2f> corners, vector<cv::Point2f> quad_pts){
  // Get transformation matrix  
  cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);  
  // Apply perspective transformation  
  cv::warpPerspective(src, dst, transmtx, dst.size());  
}

static inline vector<Vec3f> find_circles(cv::Mat &src){
  Mat cimg;
  cvtColor(src,cimg, CV_BGR2GRAY);
  vector<Vec3f> circles;  
  HoughCircles(cimg, circles, CV_HOUGH_GRADIENT, 1, src.rows/8, 100, 75, 0, 0 );  
  return circles;
}

static inline vector<int> find_markers(cv::Mat &src, vector<Vec3f> circles){
  double averR = 0;
  vector<double> row;
  vector<double> col;

  //Find rows and columns of circles for interpolation
  for(int i=0; i < circles.size(); i++){  
    bool found = false;  
    int r = cvRound(circles[i][2]);
    averR += r;
    int x = cvRound(circles[i][0]);  
    int y = cvRound(circles[i][1]);
    for(int j=0;  j < row.size(); j++){
      double y2 = row[j];
      if(y - r < y2 && y + r > y2){
        found = true;
        break;
      }
    }
    if(!found){
      row.push_back(y);
    }
    found = false;
    for(int j = 0; j < col.size(); j++){
      double x2 = col[j];
      if(x - r < x2 && x + r > x2){
        found = true;
        break;
      }
    }
    if(!found){  
      col.push_back(x);
    }
  }

  averR /= circles.size();

  sort(row.begin(),row.end(),comparator2);
  sort(col.begin(),col.end(),comparator2);

  vector<int> markers;
  Mat cimg;
  cvtColor(src, cimg, CV_BGR2GRAY);
  for(int i = 0; i < row.size(); i++){
    double max = 0;  
    double y = row[i];
    int ind = -1;
    for(int j = 0; j < col.size(); j++){
      double x = col[j];
      Point c(x,y);  

      //Use an actual circle if it exists
      for(int k = 0; k < circles.size(); k++){
        double x2 = circles[k][0];
        double y2 = circles[k][1];
        if(abs(y2-y) < averR && abs(x2-x) < averR){
          x = x2;
          y = y2;
        }
      }

      // circle outline  
      circle(src, c, averR, Scalar(0,0,255), 3, 8, 0);  
      Rect rect(x-averR,y-averR,2*averR,2*averR);  
      Mat submat = cimg(rect);  
      double p =(double)countNonZero(submat)/(submat.size().width*submat.size().height);  
      if(p>=0.3 && p>max){  
        max = p;  
        ind = j;  
        circle(src, c, averR, Scalar(255,0,255), 3, 8, 0);  
      }  
    }
    markers.pb(ind);
  }
  return markers;
}

#endif