#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "FaceSwapper.h"

using namespace dlib;
using namespace std;

cv::Rect dlibRectangleToCV(dlib::rectangle r) 
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

int main()
{
	try
	{
		FaceSwapper face_swapper("shape_predictor_68_face_landmarks.dat");
		cv::VideoCapture cap(0);
		if(!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}

		image_window win;

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		
		while(!win.is_closed()) 
		{
			cv::Mat temp;
			if(!cap.read(temp)) 
			{
				break;
			}

			cv_image<bgr_pixel> cimg(temp);

			std::vector<rectangle> faces = detector(cimg);
			
			for (int i = 0; i < int(faces.size()) - 1; i++)
			{
				cv::Rect rect1 = dlibRectangleToCV(faces[i]);
				cv::Rect rect2 = dlibRectangleToCV(faces[i + 1]);
				face_swapper.swapFaces(temp, rect1, rect2);
			}

			cimg = cv_image<bgr_pixel>(temp);
			
			win.clear_overlay();
			win.set_image(cimg);
		}
	}
	catch(exception& e)
	{
		cerr << e.what() << endl;
	}
}

