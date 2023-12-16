#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SimpleLog.h>
#include <Filesystem.h>
#include <yolov5.h>


int main(int argc, char *argv[])
{

    // 创建YOLOV5检测器
    OrtSamples::DetectorYOLOV5 detector;
    OrtSamples::InitializationParameterOfDetector initParamOfDetectorYOLOV5;
    initParamOfDetectorYOLOV5.configFilePath = CONFIG_FILE;
    OrtSamples::ErrorCode errorCode=detector.Initialize(initParamOfDetectorYOLOV5);

    if(errorCode!=OrtSamples::SUCCESS)
    {
        LOG_ERROR(stdout, "fail to initialize detector!\n");
        exit(-1);
    }
    LOG_INFO(stdout, "succeed to initialize detector\n");

    //读取测试图片
    cv::Mat srcImage = cv::imread("../resource/images/dynamicpics/image1.jpg",1);

    const cv::Size& inputSize = cv::Size(640, 640);
    detector.inputImageShape = cv::Size2f(inputSize);

    std::vector<Detection> result;

    //warm up
    detector.Detect(srcImage, detector.confidenceThreshold, detector.iouThreshold,result);

    //推理
    double time1 = cv::getTickCount();
    detector.Detect(srcImage, detector.confidenceThreshold, detector.iouThreshold,result);
    double time2 = cv::getTickCount();
    double elapsedTime = (time2 - time1)*1000 / cv::getTickFrequency();
    LOG_INFO(stdout, "inference time:%f ms\n", elapsedTime);
    
    utils::visualizeDetection(srcImage, result, detector.classNames);
    cv::imwrite("result.jpg", srcImage);
    return 0;
}

