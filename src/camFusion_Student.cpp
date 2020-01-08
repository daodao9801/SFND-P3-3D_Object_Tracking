
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    vector<cv::DMatch> matchesROI;
    for(auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::KeyPoint currKeypoint = kptsCurr[it->trainIdx];
        if(boundingBox.roi.contains(currKeypoint.pt))
        {
            matchesROI.push_back(*it);
        } 
    }
    cout << "Matches before filtered: " << matchesROI.size() << endl;

    // remove match outliers
    // compute the matches distance mean and deviation
    vector<double> matchesDist;
    for(auto it = matchesROI.begin(); it != matchesROI.end(); ++it)
    {
        matchesDist.push_back(it->distance);
    }
    cout << "before matchesDist.size(): " << matchesDist.size() << endl;
    // compute mean
    double sum = accumulate(matchesDist.begin(), matchesDist.end(), 0);
    double mean = sum / matchesDist.size();

    // compute covariance
    double diffDistSum = 0;
    for_each(matchesDist.begin(), matchesDist.end(), [&](const double d) {
        diffDistSum += (d - mean) * (d - mean);
    });

    // remove the matches which distance to mean > stddev 
    double stddev = sqrt(diffDistSum / (matchesDist.size() - 1));
    
    for(auto it = matchesDist.begin(); it != matchesDist.end();)
    {
        double dist = *it - mean;
        if(fabs(dist) > 1.0*stddev)
        {
            it = matchesDist.erase(it);
        }else
        {
            ++it;
        }    
    }

    cout << "after matchesDist.size(): " << matchesDist.size() << endl;

    boundingBox.keypoints.clear();
    boundingBox.kptMatches.clear();
    
    for(auto it = matchesROI.begin(); it != matchesROI.end(); ++it)
    {
        auto iter = find(matchesDist.begin(), matchesDist.end(), it->distance);
        if(iter != matchesDist.end())
        {
            boundingBox.kptMatches.push_back(*it);
            boundingBox.keypoints.push_back(kptsCurr[it->trainIdx]);
        }

    }
    cout << "After filtered: " << boundingBox.kptMatches.size() << endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC)
{
     // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
                // cout << "distRatio: " << distRatio << endl;
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    //cout << "distRatios.size(): " << distRatios.size() << endl;

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    // double meanDistRatio = accumulate(distRatios.begin(), distRatios.end(), 0) / distRatios.size();
    cout << "medDistRatio: " << medDistRatio << endl;
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    std::vector<LidarPoint> prevFiltered;
    std::vector<LidarPoint> currFiltered;
    // remove PCD outliers
    prevFiltered = removeLidarOutliers(lidarPointsPrev);
    currFiltered = removeLidarOutliers(lidarPointsCurr);
    
    cout << "Lidar before filter at previous frame: " << lidarPointsPrev.size() << ", " << "After filter: " << prevFiltered.size() << endl; 
    cout << "Lidar before filter at current frame: " << lidarPointsCurr.size() << ", " << "After filter: " << currFiltered.size() << endl; 
    
    /**
    /* 
     * using minimum point to compute TTC
     */
    double minXPrev = 1e9, minXCurr = 1e9;
    for(auto lidarpoint : prevFiltered)
    {
        minXPrev = lidarpoint.x > minXPrev ? minXPrev : lidarpoint.x;
    }

    for(auto lidarpoint : currFiltered)
    {
        minXCurr = lidarpoint.x > minXCurr ? minXCurr : lidarpoint.x;
    }

    double dt = 1.0 / frameRate;
    // cout << "minXprev: " << minXPrev << ", " << "minXCurr: " << minXCurr << endl;

    TTC  = minXCurr * dt / (minXPrev - minXCurr);
    
    /* 
     * using median to compute TTC
     */
    // std::sort(prevFiltered .begin(), prevFiltered .end(), [](LidarPoint a, LidarPoint b) {
    //     return a.x < b.x;  // Sort ascending on the x coordinate only
    // });

    // std::sort(currFiltered .begin(), currFiltered .end(), [](LidarPoint a, LidarPoint b) {
    //     return a.x < b.x;  // Sort ascending on the x coordinate only
    // });
    
    // double d0 = prevFiltered[prevFiltered.size()/2].x;
    // double d1 = currFiltered[currFiltered.size()/2].x;

    // double dt = 1.0 / frameRate;

    // TTC = d1* dt / (d0-d1);    
}

// This function is used to remove LidarPoint outliers before computing TTC.
std::vector<LidarPoint> removeLidarOutliers(std::vector<LidarPoint> &lidarPoints)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
     
    for(auto it = lidarPoints.begin(); it!=lidarPoints.end(); ++it)
    {
        pcl::PointXYZI point;
        point.x = it->x;
        point.y = it->y;
        point.z = it->z;
        point.intensity = it->r;
        cloud->points.push_back(point);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;

    //Removing outliers using a StatiscalOutlierREmoval filter PCL API
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor1;
    sor1.setInputCloud(cloud);
    sor1.setMeanK(100);
    sor1.setStddevMulThresh(0.25);
    sor1.filter(*cloud_filtered);

    // pcl::RadiusOutlierRemoval<pcl::PointXYZI> sor;
    // sor.setInputCloud(cloud);
    // sor.setRadiusSearch(2);
    // sor.setMinNeighborsInRadius(10);

    // Change back to the LidarPoint type
    std::vector<LidarPoint> lidarPointsFiltered;
    for(auto it : cloud_filtered->points)
    {
        LidarPoint lidarPoint;
        lidarPoint.x = it.x;
        lidarPoint.y = it.y;
        lidarPoint.z = it.z;
        lidarPoint.r = it.intensity;
        lidarPointsFiltered.push_back(lidarPoint);
    }

    return lidarPointsFiltered;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    
    // DMatch: queryIdx, trainIdx can find the cooresponding keypoints in prevFrame and currFrame.
    // queryIdx->prevFrame, trainIdx->currFrame
    // First Part: find the preFrame's matches in its every boundingBox, that is to fill 
    // the kptMatches attribute of boundingBox in head file dataStructures.h.
    for(auto it1 = matches.begin(); it1 != matches.end(); ++it1)
    {
        int trainIdx = it1->trainIdx;
        cv::KeyPoint currKeypoint = currFrame.keypoints[trainIdx];
        vector<vector<BoundingBox>::iterator> currEnclosingBoxes;
        for(auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2)
        {
            if(it2->roi.contains(currKeypoint.pt))
            {
                currEnclosingBoxes.push_back(it2);
            }
        }
        if(currEnclosingBoxes.size() == 1)
        {
            currEnclosingBoxes[0]->kptMatches.push_back(*it1);
            currEnclosingBoxes[0]->keypoints.push_back(currKeypoint);
        }
    }
    
    // Second Part: according to the kptMatches of prevFrame's every boundingBox, three for loops  are needed 
    for(auto it1 = currFrame.boundingBoxes.begin(); it1 != currFrame.boundingBoxes.end(); ++it1)
    {// first for loop is to go through all the boundingBoxes of prevFrame 
        multimap<int, int> records; 
        for(auto it2 = it1->kptMatches.begin(); it2 != it1->kptMatches.end(); ++it2)
        {// second loop is to go through the matches in every precFrame's boundingBox.
            int queryIdx = it2->queryIdx;
            for(auto it3 = prevFrame.boundingBoxes.begin(); it3 != prevFrame.boundingBoxes.end(); ++it3)
            {// third loop is to find the currFrame's boundingBox containing the keypoints of 
             // matches which is in every precFrame's boundingBox
                if(it3->roi.contains(prevFrame.keypoints[queryIdx].pt))
                {
                    records.insert(pair<int, int>(it3->boxID, queryIdx));
                }
            } 
        }

        // find the maximum key  of multimap 'record', and that is the current frame boundingBox to the preFrame boundingBox.
        // which is stored in bbBestMatches
        int max = 0;
        int index;
        if(records.size() > 0)          {
            for(auto it4 = records.begin(); it4 != records.end(); ++it4)
            {
               if(records.count(it4->first) > max)
               {
                   max = records.count(it4->first);
                   index = it4->first;
               }
                
            }
        }
        bbBestMatches.insert(pair<int, int>(index, it1->boxID)); // find the vector maximum
    }
   
}
