// 常用定义

#ifndef __COMMON_DEFINITION_H__
#define __COMMON_DEFINITION_H__

#include <opencv2/opencv.hpp>

namespace OrtSamples
{
   
// 路径分隔符(Linux:‘/’,Windows:’\\’)
#ifdef _WIN32
#define  PATH_SEPARATOR '\\'
#else
#define  PATH_SEPARATOR '/'
#endif

#define CONFIG_FILE                                                     "../resource/Configuration.xml"

typedef enum _ErrorCode
{
    SUCCESS=0,  // 0
    MODEL_NOT_EXIST, // 模型不存在
    CONFIG_FILE_NOT_EXIST, // 配置文件不存在
    FAIL_TO_LOAD_MODEL, // 加载模型失败
    FAIL_TO_OPEN_CONFIG_FILE, // 加载配置文件失败
    IMAGE_ERROR, // 图像错误
}ErrorCode;

typedef struct _ResultOfPrediction
{
    float confidence;
    int label;
    _ResultOfPrediction():confidence(0.0f),label(0){}

}ResultOfPrediction;

typedef struct _ResultOfDetection
{
    cv::Rect boundingBox;
    float confidence;
    int classID;
    std::string className;
    bool exist;

    _ResultOfDetection():confidence(0.0f),classID(0),exist(true){}

}ResultOfDetection;

typedef struct _InitializationParameterOfDetector
{
    std::string parentPath;
    std::string configFilePath;
}InitializationParameterOfDetector;

}

#endif

