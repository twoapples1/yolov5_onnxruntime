#ifndef __DETECTOR_YOLOV5_H__
#define __DETECTOR_YOLOV5_H__

#include <CommonDefinition.h>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <opencv2/opencv.hpp>

#include "utils.h"

namespace OrtSamples
{

typedef struct _YOLOV5Parameter
{
    int numberOfClasses;
    float confidenceThreshold;
    float nmsThreshold;
    float iouThreshold;
    float objectThreshold;

}YOLOV5Parameter;

class DetectorYOLOV5
{
public:
    DetectorYOLOV5();
    ~DetectorYOLOV5();

    cv::Size2f inputImageShape;
    std::vector<std::string> classNames;

    int numberOfClasses;
    float confidenceThreshold;
    float nmsThreshold;
    float iouThreshold;
    float objectThreshold;

    ErrorCode Initialize(InitializationParameterOfDetector initializationParameterOfDetector);
    ErrorCode Detect(const cv::Mat &image, const float& confThreshold, const float& iouThreshold,std::vector<Detection> &result);

private:
    cv::FileStorage configurationFile;

    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    ErrorCode PreProcessing(const cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape);
    
    std::vector<Detection> PostProcessing(const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          std::vector<Ort::Value>& outputTensors,
                                          const float& confThreshold, const float& iouThreshold);

    static void GetBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                float& bestConf, int& bestClassId);

    // static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
    //                              float& bestConf, int& bestClassId);

    std::vector<const char*> inputNames={"images"};
    std::vector<const char*> outputNames={"output0"};
 
    bool isDynamicInputShape{};

};

}


#endif

