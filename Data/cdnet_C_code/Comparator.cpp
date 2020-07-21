
#include "Comparator.h"

#include <fstream>
#include <iostream>

Comparator::Comparator(const VideoFolder &videoFolder)
	: videoFolder(videoFolder),
	ROI(cv::imread(videoFolder.getVideoPath() + "ROI.bmp", 0))
{}

void Comparator::compare(const uint nbSteps) {
	const Range range = videoFolder.getRange();
	const uint fromIdx = range.first;
	const uint toIdx = range.second;

	tp = fp = fn = tn = 0;
	nbShadowErrors = 0;

	// For each frame in the range, compare and calculate the statistics
	for (uint t = fromIdx; t <= toIdx; t+=nbSteps) {
		compare(cv::imread(videoFolder.binaryFrame(t), 0),
				cv::imread(videoFolder.gtFrame(t), 0));
	}
}

void Comparator::compare(const BinaryFrame& binary, const GTFrame& gt) {
	if (binary.empty()) {
		throw string("Binary frame is null. Probably a bad path or incomplete folder.\n");
	}

	if (gt.empty()) {
		throw string("gt frame is null. Probably a bad path or incomplete folder.\n");
	}

	BinaryConstIterator itBinary = binary.begin();
	GTIterator itGT = gt.begin();
	ROIIterator itROI = ROI.begin();

	BinaryConstIterator itEnd = binary.end();
	for (; itBinary != itEnd; ++itBinary, ++itGT, ++itROI) {
		// Current pixel needs to be in the ROI && it must not be an unknown color
		if (*itROI != BLACK && *itGT != UNKNOWN) {

			if (*itBinary == WHITE) { // Model thinks pixel is foreground
				if (*itGT == WHITE) {
					++tp; // and it is
				} else {
					++fp; // but it's not
				}
			} else { // Model thinks pixel is background
				if (*itGT == WHITE) {
					++fn; // but it's not
				} else {
					++tn; // and it is
				}
			}

			if (*itGT == SHADOW) {
				if (*itBinary == WHITE) {
					++nbShadowErrors;
				}
			}

		}
	}
}

void Comparator::save() const {
	const string filePath = videoFolder.getOutputPath() + "fmeasure.txt";
	ofstream f(filePath.c_str(), ios::out);
	if (f.is_open()) {
	    double precision = tp / (double)(tp + fp);
	    double recall = tp / (double)(tp + fn);
	    double fmeasure = 0;
	    if (precision + recall > 0) {
	        fmeasure = (2*precision*recall)/(precision+recall);
	    }
		f << fmeasure << std::endl;
		f << precision << std::endl;
		f << recall;
		f.close();
	} else {
		throw string("Unable to open the file : ") + filePath;
	}
}
