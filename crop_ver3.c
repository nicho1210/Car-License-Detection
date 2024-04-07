#include "cv.h"
#include "highgui.h"
#include <stdio.h>

int main() {
    // Read the image
    IplImage* img = cvLoadImage("car.jpg", CV_LOAD_IMAGE_COLOR);
    if (!img) {
        printf("Could not load image file\n");
        return -1;
    }
    cvNamedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Original Image", img);

    // Convert the image to grayscale
    IplImage* gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvCvtColor(img, gray, CV_BGR2GRAY);
    cvNamedWindow("Gray Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Gray Image", gray);

    // Gaussian blur
    IplImage* img_blur = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvSmooth(gray, img_blur, CV_GAUSSIAN, 19, 19, 0, 0);
    cvNamedWindow("Blurred Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Blurred Image", img_blur);

    // Edge detection
    IplImage* img_canny = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvCanny(img_blur, img_canny, 30, 150, 3);
    cvNamedWindow("Canny Edge Detection", CV_WINDOW_AUTOSIZE);
    cvShowImage("Canny Edge Detection", img_canny);
    //save the canny edge detection image
    cvSaveImage("canny_edge.jpg", img_canny);

    // Find the contours
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contours = NULL;
    cvFindContours(img_canny, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

    // Assume the largest contour is the license plate
    double maxArea = 0;
    CvRect plate_rect;
    for (CvSeq* c = contours; c != NULL; c = c->h_next) {
        double area = cvContourArea(c, CV_WHOLE_SEQ, 0);
        if (area > maxArea) {
            CvRect rect = cvBoundingRect(c, 0);
            float aspect_ratio = (float)rect.width / rect.height;
            if (aspect_ratio > 2 && aspect_ratio < 5 && rect.width > 100 && rect.height > 20) {
                maxArea = area;
                plate_rect = rect;
            }
        }
    }

    // If a potential plate was found
    if (maxArea > 0) {
        printf("Car plate found\n");
        cvRectangle(img, cvPoint(plate_rect.x, plate_rect.y),
                    cvPoint(plate_rect.x + plate_rect.width, plate_rect.y + plate_rect.height),
                    CV_RGB(0, 255, 0), 2, 8, 0);
        cvNamedWindow("Car Plate Image", CV_WINDOW_AUTOSIZE);
        cvShowImage("Car Plate Image", img);
        //save the car plate image
        cvSetImageROI(img, plate_rect);
        IplImage* car_plate = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        cvCopy(img, car_plate, NULL);
        cvResetImageROI(img);
        cvSaveImage("car_plate.jpg", car_plate);
    }

    cvWaitKey(0);

    // Cleanup
    cvReleaseImage(&img);
    cvReleaseImage(&gray);
    cvReleaseImage(&img_blur);
    cvReleaseImage(&img_canny);
    cvReleaseMemStorage(&storage);
    cvDestroyAllWindows();

    return 0;
}
