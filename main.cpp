/*
Basil Reji & Kevin Sebastian Sani
Spring 2024
Pattern Recognition & Computer Vision
Project 3: Real-time 2-D Object Recognition 
*/

#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int append_image_data_csv(char* csvFilename, char* imageLabel, std::vector<float>& imageData, int overwriteFile);
int read_image_data_csv(char* csvFilename, std::vector<char*>& imageLabels, std::vector<std::vector<float>>& imageData, int displayFileContent);

// Euclidian Distance Calculation Function

float calculateScaledEuclideanDistance(const std::vector<float>& vector1, const std::vector<float>& vector2, const std::vector<float>& standardDeviations) {
    float distanceSquared = 0.0;
    for (size_t i = 0; i < vector1.size(); i++) {
        float diff = vector1[i] - vector2[i];
        float scaledDiff = diff / standardDeviations[i];
        distanceSquared += scaledDiff * scaledDiff;
    }
    return sqrt(distanceSquared);
}

// Cosine Distance Calculation Function

float calculateCosineDistance(const std::vector<float>& vector1, const std::vector<float>& vector2) {
    float dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < vector1.size(); i++) {
        dotProduct += vector1[i] * vector2[i];
        normA += vector1[i] * vector1[i];
        normB += vector2[i] * vector2[i];
    }
    return 1.0 - (dotProduct / (sqrt(normA) * sqrt(normB)));
}

// Manhattan Distance Calculation Function

float calculateManhattanDistance(const std::vector<float>& vector1, const std::vector<float>& vector2) {
    float sum = 0.0;
    for (size_t i = 0; i < vector1.size(); i++) {
        sum += fabs(vector1[i] - vector2[i]);
    }
    return sum;
}


void segmentAndLabel(const Mat& inputImage, Mat& outputImage) {
    Mat labelMatrix = Mat::zeros(inputImage.size(), CV_32SC1);
    int nextLabel = 1;
    int neighborOffsets[] = { -1, 0, 1 };
    vector<int> labels;

    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            if (inputImage.at<uchar>(y, x) == 255) {
                vector<int> neighborLabels;
                for (int dy : neighborOffsets) {
                    for (int dx : neighborOffsets) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && nx >= 0 && ny < inputImage.rows && nx < inputImage.cols) {
                            int neighborLabel = labelMatrix.at<int>(ny, nx);
                            if (neighborLabel > 0) {
                                neighborLabels.push_back(neighborLabel);
                            }
                        }
                    }
                }
                if (neighborLabels.empty()) {
                    labelMatrix.at<int>(y, x) = nextLabel++;
                }
                else {
                    int smallestLabel = *min_element(neighborLabels.begin(), neighborLabels.end());
                    labelMatrix.at<int>(y, x) = smallestLabel;
                    for (int neighborLabel : neighborLabels) {
                        if (neighborLabel != smallestLabel) {
                            for (int ly = 0; ly < inputImage.rows; ly++) {
                                for (int lx = 0; lx < inputImage.cols; lx++) {
                                    if (labelMatrix.at<int>(ly, lx) == neighborLabel) {
                                        labelMatrix.at<int>(ly, lx) = smallestLabel;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    vector<Vec3b> colorMap(nextLabel);
    for (int i = 1; i < nextLabel; i++) {
        colorMap[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }
    outputImage = Mat::zeros(inputImage.size(), CV_8UC3);
    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            int label = labelMatrix.at<int>(y, x);
            if (label > 0) {
                outputImage.at<Vec3b>(y, x) = colorMap[label];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Paths for CSV file with object data and labels
    char csvFilePath[] = "database.csv";
    char labelName[] = "object_1";
    char modeSelection;
    vector<char*> namesOfObjects;
    vector<vector<float>> objectFeatures;

    // Setup for background subtraction
    Ptr<BackgroundSubtractor> subtractor = createBackgroundSubtractorMOG2();

    // Configuration for camera stream access
    string ipAddress = "10.0.0.177";
    string portNumber = "4747";
    Mat processedImg;
    Mat noiseReducedImg;
    Mat capturedFrame, maskOfForeground;
    double momentFeatures[10];

    // Establishing connection to camera stream
    VideoCapture capture("http://" + ip_address + ":" + port_number + "/video");
    if (!capture.isOpened()) {
        std::cout << "Unable to connect to the camera stream. Please check the IP address and port number." << std::endl;
        return -1;
    }

    std::cout << "Welcome to the Object Recognition System!" << std::endl;
    std::cout << "Please select an operating mode:" << std::endl;
    std::cout << "  1. Training Mode - Teach the system new objects." << std::endl;
    std::cout << "  2. Classifier Mode - Let the system recognize and classify objects." << std::endl;
    std::cout << "Enter your choice (1/2): ";
    cin >> modeSelection;

    while (true) {
        while (modeSelection == '1') {

            td::cout << "You have selected Training Mode." << std::endl;
            std::cout << "In this mode, you can introduce new objects to the system." << std::endl;
            std::cout << "Please enter the name of the object you wish to add: ";
            cin >> labelName;
            
            while (true) {
                vector<float> featuresOfObject;
                stream >> capturedFrame;
                if (capturedFrame.empty())
                    break;

                // Applying background subtraction to current frame
                subtractor->apply(capturedFrame, maskOfForeground, 0);
                Rect detectedBoundingBox;

                // Display current frame and foreground mask
                rectangle(capturedFrame, Point(10, 2), Point(100, 20), Scalar(255, 255, 255), -1);
                stringstream frameInfo;
                frameInfo << stream.get(CAP_PROP_POS_FRAMES);
                string frameNumber = frameInfo.str();
                putText(capturedFrame, frameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
                imshow("Frame", capturedFrame);
                imshow("FG Mask", maskOfForeground);

                // Use morphological operations to refine foreground mask
                Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5)); 

                // Open operation to reduce background noise
                morphologyEx(maskOfForeground, noiseReducedImg, MORPH_OPEN, morphKernel, Point(-1, -1), 1);

                // Close operation to fill gaps in the foreground
                morphologyEx(noiseReducedImg, noiseReducedImg, MORPH_CLOSE, morphKernel, Point(-1, -1), 3);

                // Erode operation to further eliminate noise
                morphologyEx(noiseReducedImg, noiseReducedImg, MORPH_ERODE, morphKernel, Point(-1, -1), 1);

                // Display the refined foreground image
                imshow("Cleaned Up Image", noiseReducedImg);

                // Segment and label the processed image
                segmentAndLabel(noiseReducedImg, processedImg);

                // Display the image with segmented regions
                imshow("Segmented Image", processedImg);

                // Identifying contours within the segmented image
                RNG randomGenerator(12345);
                vector<vector<Point>> detectedContours;
                findContours(noiseReducedImg, detectedContours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

                // Determine the contour with the maximum area
                double largestArea = -1;
                int indexOfLargestContour = -1;
                for (int i = 0; i < detectedContours.size(); i++) {
                    double contourAreaValue = contourArea(detectedContours[i]);
                    if (contourAreaValue > largestArea) {
                        largestArea = contourAreaValue;
                        indexOfLargestContour = i;
                    }
                }

                // Highlighting the largest contour with a bounding box
                Mat imageWithBoundingBox = capturedFrame.clone();
                if (indexOfLargestContour >= 0) {
                    detectedBoundingBox = boundingRect(detectedContours[indexOfLargestContour]);
                    rectangle(imageWithBoundingBox, detectedBoundingBox, Scalar(0, 255, 0), 2);
                }

                // Visualize the image with the bounding box
                imshow("Detected Object", imageWithBoundingBox);

                // Analyzing the largest contour for its properties
                Moments contourMoments = moments(detectedContours[indexOfLargestContour]);

                // Calculating the centroid of the contour
                Point2f contourCentroid(contourMoments.m10 / contourMoments.m00, contourMoments.m01 / contourMoments.m00);

                // Determining central moments for orientation analysis
                double centralMoment11 = contourMoments.mu11 / contourMoments.m00;
                double centralMoment20 = contourMoments.mu20 / contourMoments.m00;
                double centralMoment02 = contourMoments.mu02 / contourMoments.m00;

                // Orientation angle derived from central moments
                double orientationAngle = 0.5 * atan2(2 * centralMoment11, centralMoment20 - centralMoment02);

                // Utilizing orientation angle to calculate the Axis Least Central Moment (ALCM)
                double cosineOfAngle = cos(orientationAngle);
                double sineOfAngle = sin(orientationAngle);
                double axisEndX1 = contourCentroid.x + cosineOfAngle * centralMoment20 + sineOfAngle * centralMoment11;
                double axisEndY1 = contourCentroid.y + cosineOfAngle * centralMoment11 + sineOfAngle * centralMoment02;
                double axisEndX2 = contourCentroid.x - cosineOfAngle * centralMoment20 - sineOfAngle * centralMoment11;
                double axisEndY2 = contourCentroid.y - cosineOfAngle * centralMoment11 - sineOfAngle * centralMoment02;

                // Drawing the ALCM on the detected object
                line(imageWithBoundingBox, Point(axisEndX1, axisEndY1), Point(axisEndX2, axisEndY2), Scalar(0, 255, 0), 2);

                // Present the image with ALCM
                imshow("ALCM", imageWithBoundingBox);

                // Computing Hu moments for shape analysis
                cv::Moments huMomentsCalculator = cv::moments(detectedContours[indexOfLargestContour]);
                double huMoments[10];
                cv::HuMoments(huMomentsCalculator, huMoments);

                // Normalizing Hu moments for scale invariance
                for (int i = 0; i < 10; i++) {
                    huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
                    featuresOfObject.push_back(huMoments[i]);
                }

                // Displaying normalized central moments and Hu moments for analysis
                cout << "Normalized Central Moments: mu20: " << centralMoment20 << ", mu02: " << centralMoment02 << ", mu11: " << centralMoment11 << endl;
                cout << "Hu Moments: ";
                for (int i = 0; i < 10; i++) {
                    cout << huMoments[i] << " ";
                }
                cout << endl;

                // Wait for a user command to proceed
                int userCommand = waitKey(30);
                if (userCommand == 'n') {
                    append_image_data_csv(csvFilePath, labelName, featuresOfObject, 0);
                }
                
                std::cout << std::endl;
                std::cout << "Object has been added to Database" << std::endl;
                std::cout << "--------------------------------------------------------" << std::endl;
                std::cout << std::endl;
                std::cout << "Welcome to the Object Recognition System!" << std::endl;
                std::cout << "Please select an operating mode:" << std::endl;
                std::cout << "  1. Training Mode - Teach the system new objects." << std::endl;
                std::cout << "  2. Classifier Mode - Let the system recognize and classify objects." << std::endl;
                std::cout << "  3. Exit " << std::endl;
                std::cout << "Enter your choice (1/2/3): ";
                cin >> modeSelection;

                while (modeSelection == '2')
                {
                    std::cout << "\n-- Classifier Mode --" << std::endl;
                    std::cout << "The system will now attempt to recognize objects. Please present an object to the camera." << std::endl;
                    cv::destroyAllWindows(); 
                    vector<vector<float>> featuresList;
                    vector<string> labelsList;

                    vector<float> currentObjectFeatures;

                    vector<char*> objectFileNames; 

                    // Read features and labels from CSV file
                    read_image_data_csv(csvFilePath, objectFileNames, featuresList, 0);

                    // Display loaded features for each object
                    for (size_t i = 0; i < objectFileNames.size(); ++i)
                    {
                        cout << "\nObject: " << objectFileNames[i] << " Features:";
                        for (float feature : featuresList[i])
                        {
                            cout << " " << feature;
                        }
                    }

                    while (true) {
                        stream >> capturedFrame; 
                        if (capturedFrame.empty()) break; 

                        // Apply background subtraction to isolate foreground
                        subtractor->apply(capturedFrame, maskOfForeground, 0);

                        // Display frame number on the video feed
                        rectangle(capturedFrame, Point(10, 2), Point(100, 20), Scalar(255, 255, 255), -1);
                        stringstream frameInfo;
                        frameInfo << stream.get(CAP_PROP_POS_FRAMES);
                        string frameNumber = frameInfo.str();
                        putText(capturedFrame, "Frame: " + frameNumber, Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

                        // Show current video frame and the foreground mask
                        imshow("Live Video Feed", capturedFrame);
                        imshow("Foreground Detection", maskOfForeground);

                        // User command to proceed or quit
                        int userCommand = waitKey(30);
                        if (userCommand == 'n') break; 
                    }

                    while (true)
                    {
                        Rect detectedBoundingBox;
                        string identifiedLabel;
                        // Refine foreground by reducing noise and filling gaps
                        Mat morphologicalKernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

                        // Apply morphological opening to reduce small noise
                        morphologyEx(maskOfForeground, noiseReducedImg, MORPH_OPEN, morphologicalKernel, Point(-1, -1), 1);

                        // Apply morphological closing to fill gaps in the foreground
                        morphologyEx(noiseReducedImg, noiseReducedImg, MORPH_CLOSE, morphologicalKernel, Point(-1, -1), 3);

                        // Apply erosion to fine-tune the edges of the foreground objects
                        morphologyEx(noiseReducedImg, noiseReducedImg, MORPH_ERODE, morphologicalKernel, Point(-1, -1), 1);

                        // Visualize the refined foreground mask
                        imshow("Refined Foreground Mask", noiseReducedImg);

                        // Perform region segmentation and labeling
                        Mat labeledImage;
                        segmentAndLabel(noiseReducedImg, labeledImage);

                        // Display the image with labeled regions
                        imshow("Labeled Regions", labeledImage);

                        // Detect and analyze contours in the cleaned image
                        vector<vector<Point>> detectedContours;
                        findContours(noiseReducedImg, detectedContours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

                        // Identify the contour with the maximum area
                        double largestArea = -1;
                        int indexOfLargestContour = -1;
                        for (int i = 0; i < detectedContours.size(); i++) {
                            double contourAreaValue = contourArea(detectedContours[i]);
                            if (contourAreaValue > largestArea) {
                                largestArea = contourAreaValue;
                                indexOfLargestContour = i;
                            }
                        }

                        // Highlight the largest contour by drawing a bounding box around it
                        Mat imageWithBoundingBox = capturedFrame.clone();
                        if (indexOfLargestContour >= 0) {
                            detectedBoundingBox = boundingRect(detectedContours[indexOfLargestContour]);
                            rectangle(imageWithBoundingBox, detectedBoundingBox, Scalar(0, 255, 0), 2);
                        }

                        // Display the image with bounding box highlighting the detected object
                        imshow("Detected Object Bounding Box", imageWithBoundingBox);


                        // Analyzing shape characteristics of the detected contour
                        Moments shapeMoments = cv::moments(detectedContours[indexOfLargestContour]);

                        // Determining the center point of the contour
                        Point2f contourCenter(shapeMoments.m10 / shapeMoments.m00, shapeMoments.m01 / shapeMoments.m00);

                        // Computing contour's central moments for orientation calculation
                        double centralMoment11 = shapeMoments.mu11 / shapeMoments.m00;
                        double centralMoment20 = shapeMoments.mu20 / shapeMoments.m00;
                        double centralMoment02 = shapeMoments.mu02 / shapeMoments.m00;

                        // Estimating contour's orientation angle
                        double contourAngle = 0.5 * atan2(2 * centralMoment11, centralMoment20 - centralMoment02);

                        // Applying orientation angle to derive Axis Least Central Moment (ALCM)
                        double angleCosine = cos(contourAngle);
                        double angleSine = sin(contourAngle);
                        double axisEndpointX1 = contourCenter.x + angleCosine * centralMoment20 + angleSine * centralMoment11;
                        double axisEndpointY1 = contourCenter.y + angleCosine * centralMoment11 + angleSine * centralMoment02;
                        double axisEndpointX2 = contourCenter.x - angleCosine * centralMoment20 - angleSine * centralMoment11;
                        double axisEndpointY2 = contourCenter.y - angleCosine * centralMoment11 - angleSine * centralMoment02;

                        // Highlighting the ALCM on the detected object within the image
                        line(imageWithBoundingBox, Point(axisEndX1, axisEndY1), Point(axisEndX2, axisEndY2), Scalar(0, 255, 0), 2);

                        // Presenting the processed image with ALCM visualization
                        imshow("Contour Orientation", imageWithBoundingBox);

                        // Executing Hu moments for advanced shape analysis
                        Moments huMomentCalculator = cv::moments(detectedContours[indexOfLargestContour]);
                        double huMomentsArray[10];
                        cv::HuMoments(huMomentCalculator, huMomentsArray);

                        // Normalizing Hu moments to achieve scale invariance
                        for (int i = 0; i < 10; i++) {
                            huMomentsArray[i] = -1 * copysign(1.0, huMomentsArray[i]) * log10(abs(huMomentsArray[i]));
                            currentObjectFeatures.push_back(huMomentsArray[i]);
                        }

                        // Adjusting moments for consistent scale and orientation representation
                        double normalizedCentralMoment20 = copysign(1.0, centralMoment20) * log10(abs(centralMoment20));
                        double normalizedCentralMoment02 = copysign(1.0, centralMoment02) * log10(abs(centralMoment02));
                        double normalizedCentralMoment11 = copysign(1.0, centralMoment11) * log10(abs(centralMoment11));
                        currentObjectFeatures.push_back(normalizedCentralMoment20);
                        currentObjectFeatures.push_back(normalizedCentralMoment02);
                        currentObjectFeatures.push_back(normalizedCentralMoment11);

                        vector<pair<float, string>> similarityScores;

                        // Compute standard deviations for feature normalization
                        vector<float> featureStdDevs(featuresList[0].size(), 0.0);
                        for (size_t featureIndex = 0; featureIndex < featuresList[0].size(); ++featureIndex) {
                            float sum = 0.0, variance = 0.0;
                            for (const auto& features : featuresList) {
                                sum += features[featureIndex];
                            }
                            float mean = sum / featuresList.size();

                            for (const auto& features : featuresList) {
                                variance += pow(features[featureIndex] - mean, 2);
                            }
                            featureStdDevs[featureIndex] = sqrt(variance / featuresList.size());
                        }

                        // Calculate distance between current object features and dataset
                        for (size_t i = 0; i < objectFileNames.size(); ++i) {
                            float distance = calculateScaledEuclideanDistance(currentObjectFeatures, featuresList[i], featureStdDevs);
                            similarityScores.emplace_back(distance, objectFileNames[i]);
                        }
                        sort(similarityScores.begin(), similarityScores.end());

                        // Identify the closest match based on the smallest distance
                        cout << "Closest match: " << similarityScores.front().second << "\n";

                        // Further classification with k-Nearest Neighbor (k-NN) for comparison
                        vector<Mat> queryDescriptors(1, Mat(currentObjectFeatures).reshape(1, 1)); 
                        vector<Mat> trainingDescriptors; 
                        for (const auto& features : featuresList) {
                            trainingDescriptors.push_back(Mat(features).reshape(1, 1));
                        }

                        // k-NN matching
                        int k = 2; 
                        vector<vector<DMatch>> knnMatches;
                        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
                        matcher->knnMatch(queryDescriptors[0], trainingDescriptors, knnMatches, k);
                        string bestMatch = objectFileNames[knnMatches[0][0].trainIdx];
                        
                        // Displaying the closest match found by the normal distance measure  
                        cout << "Top Normal Match: " << distances[0].second << "\n";

                        // For k-NN, displaying the top match based on the nearest neighbor(s)
                        cout << "Top KNN Match: " << bestMatch << "\n";
                 
                        int action = waitKey(30);
                        if (action == 'n') {
                        }

                        std::cout << std::endl;
                        std::cout << "Welcome to the Object Recognition System!" << std::endl;
                        std::cout << "Please select an operating mode:" << std::endl;
                        std::cout << "  1. Training Mode - Teach the system new objects." << std::endl;
                        std::cout << "  2. Classifier Mode - Let the system recognize and classify objects." << std::endl;
                        std::cout << "  3. Exit " << std::endl;
                        std::cout << "Enter your choice (1/2/3): ";
                        cin >> modeSelection;

                        if (modeSelection == '3') {
                            destroyAllWindows();
                            return 0;
                        }
                    }
                }
            }
        }
    }
}
