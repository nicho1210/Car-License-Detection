#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#endif

int main() {
    // Read the image
    IplImage* img = cvLoadImage("car.jpg", CV_LOAD_IMAGE_COLOR);

    // Check if the image was loaded
    if (!img) {
        printf("Could not load image file: %s\n", "car.jpg");
        exit(0);
    }

    //make the img into the lower half of the img
    cvSetImageROI(img, cvRect(0, img->height/2, img->width, img->height/2));
    IplImage* img2 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    cvCopy(img, img2, NULL);
    cvResetImageROI(img);

    //save the img2
    cvSaveImage("car2.jpg", img2);

    //and cut down and keep the middle of the img2
    cvSetImageROI(img2, cvRect(img2->width/4, img2->height/4, img2->width/2, img2->height/2));
    IplImage* img3 = cvCreateImage(cvGetSize(img2), img2->depth, img2->nChannels);
    cvCopy(img2, img3, NULL);
    cvResetImageROI(img2);

    //save the img3
    cvSaveImage("car3.jpg", img3);

    img = img3;

    // Convert the image to grayscale
    IplImage* gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvCvtColor(img, gray, CV_BGR2GRAY);

    // Adjust the thresholds for canny edge detection based on the image characteristics
    cvCanny(gray, gray, 50, 200, 3);

    // Adjust kernel size and shape for the morphological operation if needed
    IplImage* morphed = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);
    cvMorphologyEx(gray, morphed, NULL, NULL, CV_MOP_CLOSE, 2);

    // Find contours on the morphed image
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours = NULL;
    cvFindContours(morphed, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));


    // Draw contours on a clone of the original image for visualization
    IplImage* contoursImage = cvCloneImage(img);
    cvDrawContours(contoursImage, contours, CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), 2, 2, 8, cvPoint(0, 0));
    cvNamedWindow("Contours Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Contours Image", contoursImage);

    CvSeq* carPlateContour = NULL;
    double maxArea = 0;
    // Iterate through each contour
    for (CvSeq* c = contours; c != NULL; c = c->h_next) {
        // Use boundingRect to get a simplified rectangle around the contour
        CvRect r = cvBoundingRect(c, 0);
        double contourArea = cvContourArea(c, CV_WHOLE_SEQ, 0);
        
        // Filter based on aspect ratio and size criteria
        if ((r.width / (double)r.height > 2) && (r.width / (double)r.height < 4) && contourArea > maxArea) {
            carPlateContour = c;
            maxArea = contourArea;
        }
    }

    // Debugging: print details of each contour to help fine-tune the filtering criteria
    for (CvSeq* c = contours; c != NULL; c = c->h_next) {
        CvRect r = cvBoundingRect(c, 0);
        printf("Contour bounding box: x=%d, y=%d, width=%d, height=%d\n", r.x, r.y, r.width, r.height);
        // You can also print out the area of the contour
        double contourArea = cvContourArea(c, CV_WHOLE_SEQ, 0);
        printf("Contour area: %f\n", contourArea);
    }


    // If a car plate contour was found
    if (carPlateContour != NULL) {
        // Draw the found contour on a clone of the original image for visualization
        IplImage* carPlateImage = cvCloneImage(img);
        cvDrawContours(carPlateImage, carPlateContour, CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), 2, 2, 8, cvPoint(0, 0));
        cvNamedWindow("Car Plate Image", CV_WINDOW_AUTOSIZE);
        cvShowImage("Car Plate Image", carPlateImage);
        cvSaveImage("carplate.jpg", carPlateImage);
        cvReleaseImage(&carPlateImage); // Release the car plate image after saving
    }

    // Wait for a key press
    cvWaitKey(0);

    // Release memory and cleanup
    cvReleaseMemStorage(&storage);
    cvReleaseImage(&img);
    cvReleaseImage(&gray);
    cvReleaseImage(&morphed);
    cvReleaseImage(&contoursImage);
    cvDestroyAllWindows();

    return 0;
}
#ifdef _EiC
main(1,"convexhull.c");
#endif
