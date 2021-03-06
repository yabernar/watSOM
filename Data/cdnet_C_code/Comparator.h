
#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "types.h"
#include "VideoFolder.h"

class Comparator
{
	public:
		Comparator(const VideoFolder &videoFolder);

		void compare(const uint nbSteps);
		void save() const;

	private:
		typedef BinaryFrame ROIFrame;
		typedef BinaryConstIterator ROIIterator;

		const VideoFolder &videoFolder;
		const ROIFrame ROI;

		uint tp, fp, fn, tn;
		uint nbShadowErrors;

		void compare(const BinaryFrame& binary, const GTFrame& gt);
};
