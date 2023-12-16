#include <yolov5.h>
#include <Filesystem.h>
#include <SimpleLog.h>


namespace OrtSamples
{

DetectorYOLOV5::DetectorYOLOV5()
{
}

DetectorYOLOV5::~DetectorYOLOV5()
{
    configurationFile.release();
    
}


ErrorCode DetectorYOLOV5::Initialize(InitializationParameterOfDetector initializationParameterOfDetector)
{
    // 读取配置文件
    std::string configFilePath=initializationParameterOfDetector.configFilePath;
    std::cout<<"configFilePath:"<<configFilePath<<std::endl;
    if(Exists(configFilePath)==false)
    {
        LOG_ERROR(stdout, "no configuration file!\n");
        return CONFIG_FILE_NOT_EXIST;
    }
    if(!configurationFile.open(configFilePath, cv::FileStorage::READ))
    {
       LOG_ERROR(stdout, "fail to open configuration file\n");
       return FAIL_TO_OPEN_CONFIG_FILE;
    }
    LOG_INFO(stdout, "succeed to open configuration file\n");
    
    // 获取配置文件参数
    cv::FileNode OrtsettingNode = configurationFile["OrtSeting"];
    int GraphOptimizationLevel_ =(int)OrtsettingNode["GraphOptimizationLevel"];
    int LogSeverityLevel_ = (int)OrtsettingNode["LogSeverityLevel"];
    int DeviceId = (int)OrtsettingNode["UseDeviceId"];
    bool EnableProfiling_ = (bool)(int)OrtsettingNode["EnableProfiling"];

    cv::FileNode netNode = configurationFile["DetectorYOLOV5"];

    std::string modelPath=(std::string)netNode["ModelPath"];

    std::string pathOfClassNameFile=(std::string)netNode["ClassNameFile"];
    confidenceThreshold = (float)netNode["ConfidenceThreshold"];
    nmsThreshold = (float)netNode["NMSThreshold"];
    objectThreshold = (float)netNode["ObjectThreshold"];
    iouThreshold = (float)netNode["IOUThreshold"];
    numberOfClasses=(int)netNode["NumberOfClasses"];
    // useFP16=(bool)(int)netNode["UseFP16"];

    // 加载模型
    if(Exists(modelPath)==false)
    {
        LOG_ERROR(stdout,"%s not exist!\n",modelPath.c_str());
        return MODEL_NOT_EXIST;
    }

    //对Session进行初始化
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    //设置图优化等级、日志输出等级以及Profiling
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    sessionOptions.SetLogSeverityLevel(LogSeverityLevel_);
    if (EnableProfiling_)  sessionOptions.EnableProfiling("profile_prefix");

    //判断当前ort中所存在的provider
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    auto rocmAvailable = std::find(availableProviders.begin(), availableProviders.end(), "ROCMExecutionProvider");

    if (cudaAvailable == availableProviders.end()&&rocmAvailable == availableProviders.end())
    {
        LOG_INFO(stdout,"GPU is not supported by your ONNXRuntime build. Fallback to CPU.\n");
        LOG_INFO(stdout,"Inference device: CPU.\n");
    }
    else if(cudaAvailable != availableProviders.end())
    {
        //使用 CUDA 端进行推理
        LOG_INFO(stdout,"Inference device: NVIDAI GPU.\n");
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = DeviceId;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else if(rocmAvailable != availableProviders.end())
    {
        //使用 ROCM 端进行推理
        LOG_INFO(stdout,"Inference device: AMD GPU.\n");
        OrtROCMProviderOptions rocmOptions;
        rocmOptions.device_id = DeviceId;
        sessionOptions.AppendExecutionProvider_ROCM(rocmOptions);
    }

    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
    LOG_INFO(stdout,"succeed to load model: %s\n",GetFileName(modelPath).c_str());

    // 获取模型输入/输出节点信息
    // Ort::AllocatorWithDefaultOptions allocator;
    // int numInputs = session.GetInputCount();
    // //inputNames.reserve(numInputs);

    // LOG_INFO(stdout,"input_name: ");
    // for (int i = 0; i < numInputs; i++) {
    //     auto input_name = session.GetInputNameAllocated(i, allocator);
    //     std::string input_name_ = input_name.get();

    //     LOG_INFO(stdout,"%s ",input_name_);
    // }

    // int numOutputs = session.GetOutputCount();
    // LOG_INFO(stdout,"\noutput_name: ");
    // for(size_t i = 0; i < numOutputs; i++)
    // {
    //     auto out_name = session.GetOutputNameAllocated(i, allocator);
    //     std::string output_name_ = out_name.get();
    //     LOG_INFO(stdout,"%s ",output_name_);
    // }
    // LOG_INFO(stdout,"\n");
 
    LOG_INFO(stdout,"ConfidenceThreshold:%f\n",confidenceThreshold);
    LOG_INFO(stdout,"NMSThreshold:%f\n",nmsThreshold);
    LOG_INFO(stdout,"objectThreshold:%f\n",objectThreshold);
    LOG_INFO(stdout,"iouThreshold:%f\n",iouThreshold);
    LOG_INFO(stdout,"NumberOfClasses:%d\n",numberOfClasses);

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        LOG_INFO(stdout, "Dynamic input shape\n");
        this->isDynamicInputShape = true;
    }

    // 读取类别名
    if(!pathOfClassNameFile.empty())
    {
         std::ifstream classNameFile(pathOfClassNameFile);
         std::string line;
         while (getline(classNameFile, line))
         {
             classNames.push_back(line);
         }
    }
    else
    {
        classNames.resize(numberOfClasses);
    }
    return SUCCESS;

}

ErrorCode DetectorYOLOV5::PreProcessing(const cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape)
{

    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);

    //使用letterbox 对输入图像进行填充
    utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
                     cv::Scalar(114, 114, 114), true, 
                     false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize {floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

ErrorCode DetectorYOLOV5::Detect(const cv::Mat &image, const float& confThreshold , const float& iouThreshold,std::vector<Detection> &result)
{
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape {1, 3, -1, -1};
    
    this->PreProcessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              outputNames.data(),
                                                              1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    result = this->PostProcessing(resizedShape,
                                    image.size(),
                                    outputTensors,
                                    confThreshold, iouThreshold);

    delete[] blob;
}


std::vector<Detection> DetectorYOLOV5::PostProcessing(const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int numClasses = (int)outputShape[2] - 5;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = (int) (it[0]);
            int centerY = (int) (it[1]);
            int width = (int) (it[2]);
            int height = (int) (it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->GetBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}


void DetectorYOLOV5::GetBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }
}

}
