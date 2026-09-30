#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>    // std::shuffle
#include <cmath>
#include <map>
#include <filesystem>
#include <windows.h>
#include <numeric>   // std::iota
#include <random>    // std::mt19937

using namespace std;
using namespace cv;
// 1. 定义返回结果的容器
struct GridMeasureResult {
    vector<Vec3f> allCircle; // 圆心信息
    float hDist = 0.0f;  // 水平间距
    float vDist = 0.0f;  // 垂直间距
    float avgRadius = 0.0f; // 平均半径
};
// 定义9个ROI区域
struct NineGridRoiImg
{
    Point RoiCenter{ 0, 0 };

    Rect NineCenter{ 0, 0, 0, 0 };
    Rect NineUp{ 0, 0, 0, 0 };
    Rect NineDown{ 0, 0, 0, 0 };
    Rect NineLeft{ 0, 0, 0, 0 };
    Rect NineRight{ 0, 0, 0, 0 };

    Rect NineLeftUp{ 0, 0, 0, 0 };
    Rect NineLeftDown{ 0, 0, 0, 0 };
    Rect NineRightUp{ 0, 0, 0, 0 };
    Rect NineRightDown{ 0, 0, 0, 0 };

} nineGridRoiImg;

// 每个ROI图像的特征
struct ROIFeature {
    std::string roiName = "";   // ROI 名称
    double area = 0.0;          // 面积
    double perimeter = 0.0;     // 周长
    double width = 0.0;         // 宽
    double height = 0.0;        // 高
    double aspectRatio = 0.0;   // 长宽比
    double circularity = 0.0;   // 圆形度
    double meanGray = 0.0;      // 平均灰度
    double stdGray = 0.0;       // 灰度标准差
} roiFeature;


// 定位图片中目标的 ROI --- 1.矩形高亮区域；2.计算圆孔距离（水平|垂直）3.计算目标整个ROI；4.特征可视化
vector<Rect> FindFullWhiteRectsByIntegral(const Mat& srcImg, int targetWidth = 21, int targetHeight = 13);
GridMeasureResult MeasureCircleGridDistances(const Mat& srcImg);
Rect GetRegionRoi(const vector<Rect>& fullWhite, GridMeasureResult gridMeasureResult);
Mat WhiteRectVisualize(const Mat& img, const vector<Rect>& fullWhite, const GridMeasureResult& gridMeasureResult, const Rect roi);

// 针对旋转图片 提取目标 ROI ---1.提取 旋转的ROI 的RotatedRect；2.旋转的RotatedRect 校正 为ROIimg
cv::RotatedRect GetRegionRotatedROI(const std::vector<cv::Rect>& fullWhite, GridMeasureResult gridMeasureResult);
cv::Mat ExtractRotatedROI(const cv::Mat& img, const cv::RotatedRect& rr);

// 1.划分 雪花胶水区域 为 9 个ROI；2.划分结果-单张图片可视化；3.划分结果-批量可视化
cv::Mat NineGridRoi(cv::Mat& testImg, Rect Roi); // 带可视化
int OneImgVisualize(const std::string& imgPath);
void BatchVisualize(const std::string& folderPath);

// 计算 ROI 图片 多个特征 - 1.计算单个ROI区域特征；2.计算单张图片所有ROI区域特征
ROIFeature ExtractROIFeature(const cv::Mat& srcImg, const cv::Rect& roiRect, const std::string& roiName = "ROIFeat");
Mat ExtractOneImageFeature(const Mat& testImg);

// 1.计算与保存 所有图像 雪花特征；2.加载特征数据
void ExtractAndSaveFeatures(const string& goodImgDir, const string& badImgDir, const string& savePath);
void loadFeatures(const string& loadPath, Mat& trainData, Mat& labels);

// 加载特征数据，训练SVM模型
void TrainSVMFromFile(const string& featureFilePath);

// 1.加载模型，预测图片；2.加载模型，批量预测图片；3.评估预测结果
int PredictImage(const string& modelPath, const string& normPath, const string& imgPath);
void BatchPredictFolder(const string& modelPath, const string& normPath,
    const string& folderPath, bool recursive = false);
void EvaluateSVM(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& samples, const cv::Mat& labels,
    const std::string& tag);

#include "GlueInspection.h"

int main()
{
    //SetConsoleOutputCP(CP_UTF8);
    //SetConsoleCP(CP_UTF8);       // 建议同时设置输入编码
    //const string imgPath = "./GlueInspection200PCS/OK100/ZCG12A72745_0101_20260308034335.jpg";
    //const string imgPath = "./GlueInspection200PCS/Bad100/broken3.jpg"; // 断胶
    const string imgPath = "./GlueInspection200PCS/Bad100/less5.jpg"; // 少胶
    //const string imgPath = "./GlueInspection200PCS/Bad100/more11.jpg"; // 多胶
    //const string imgPath = "./GlueInspection200PCS/Bad100/刮胶59.jpg";
    //const string imgPath = "./GlueInspection200PCS/Bad100/少胶6.jpg";

    const string modelPath = "SVM_Model_R92_39-39_.xml";
    const string normPath = "NormParams_R92_39-39_.xml";
    // 批量处理 --- 一张过于倾斜的-刮胶23-无法匹配
    const string goodImgDir = "./GlueInspection200PCS/OK100/";
    const string badImgDir = "./GlueInspection200PCS/Bad100/";
    const string featureFile = "dataset_R92_features.yml";
    //const string folderPath = "./GlueInspection200PCS/Bad100/";
    // 测试单张图片
    //OneImgVisualize(imgPath);
    // 测试所有图片
    //BatchVisualize(badImgDir);

    // 第一次运行：执行特征提取并保存（跑完后注释掉）
    //ExtractAndSaveFeatures(goodImgDir, badImgDir, featureFile);
    // 后续调试：直接加载数据训练模型
    //TrainSVMFromFile(featureFile);

    // 加载模型，预测图片
    //PredictImage(modelPath, normPath, imgPath);
    BatchPredictFolder(modelPath, normPath, goodImgDir, false);

   //Mat trainData, labels;
   //loadFeatures(featureFile, trainData, labels);
   //trainSVM(goodImgDir, badImgDir);

    // ***********************************************

   //ExtractROIFeature(ImgFeat, nineGridRoiImg.NineCenter);

   // ------------------
   //std::vector<std::vector<cv::Point>> contours;
   //cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
   //cv::Mat dst = drawAllRawContours(testImg, contours);

   //cv::imshow("Main-testImg", testImg);

   //cv::imshow("Debug: Raw Contours", dst);
   //imwrite("less5_fullWhite.jpg", testImg);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

int OneImgVisualize(const std::string& imgPath)
{
    // 测试单张图片
    std::cout << "image :" << imgPath << endl;
    Mat testImg = imread(imgPath);
    if (testImg.empty()) {
        cerr << "Img read error!" << endl;
        return -1;
    }
    else
    {   // 提取目标 ROI
        vector<Rect> fullWhite = FindFullWhiteRectsByIntegral(testImg);
        if (fullWhite.size() == 7)
        {
            Point pointRx3(fullWhite[fullWhite.size() - 3].x, fullWhite[fullWhite.size() - 3].y);
            GridMeasureResult gridMeasureResult = MeasureCircleGridDistances(testImg);

            // ---- 3. 算倾斜角 theta = std::atan2(dy, dx); 逆时针为负 ----// 从左到右
            double dx = fullWhite[fullWhite.size() - 1].x - fullWhite[0].x;
            double dy = fullWhite[fullWhite.size() - 1].y - fullWhite[0].y;
            double theta = std::atan2(dy, dx);
            std::cout << "倾斜角 theta = " << theta * 180.0 / CV_PI << " 度" << std::endl;

            Rect Roi;
            cv::Mat vis = testImg.clone();
            cv::Mat rotationRoiImg;
            if (std::abs(theta) < 1.0 * CV_PI / 180.0) {
                std::cout << "不旋转" << std::endl;
                Roi = GetRegionRoi(fullWhite, gridMeasureResult);
                NineGridRoi(testImg, Roi);
                imshow("NineROI", testImg);
            }
            else {
                // 1. 求倾斜 ROI
                cv::RotatedRect rr = GetRegionRotatedROI(fullWhite, gridMeasureResult);
                // 2. 摆正并提取，输出固定 width x height
                rotationRoiImg = ExtractRotatedROI(testImg, rr);
                Roi = Rect(0, 0, rotationRoiImg.cols, rotationRoiImg.rows);

                cout << "\nretationRoiImg.size:" << rotationRoiImg.size();

                cv::Point2f pts[4];
                rr.points(pts); // 倾斜矩形的 4 个角点坐标填进数组pts
                for (int i = 0; i < 4; ++i)
                    cv::line(vis, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
                cv::circle(vis, cv::Point(pointRx3.x, pointRx3.y), 1, cv::Scalar(0, 0, 255), -1);  // 参考点
                testImg = rotationRoiImg;
                NineGridRoi(testImg, Roi);
                cv::imshow("Rotated ROI", vis);
                imshow("NineRotated ROI", testImg);

            }
            //WhiteRectVisualize(testImg, fullWhite, gridMeasureResult, Roi);
            //imwrite("featROI.jpg", testImg);
            //waitKey(0);
        }
        else
        {
            cerr << "\nerror :fullWhite.size() != 7 !" << endl;
            return -1;
        }
    }

}

void BatchVisualize(const std::string& folderPath) {
    size_t count = 0, failCount = 0;
    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(folderPath))
    {
        std::cout << "\n ------*** 第" << count << "张图片 ***------" << endl;
        std::string ext = entry.path().extension().string();
        // 统一转为小写判断后缀，防止 .JPG 漏掉
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".png" || ext == ".bmp") {
            OneImgVisualize(entry.path().string()) == -1 ? failCount++ : ++count;
        }
    }
    std::cout << "\n条目数: " << count << " ,跳过条目数：" << failCount << std::endl;
}

cv::Mat NineGridRoi(cv::Mat& testImg, Rect Roi)
{
    int Rwidth = Roi.width;
    int Rheight = Roi.height;

    nineGridRoiImg.RoiCenter = Point(Roi.x + Roi.width / 2 - 1, Roi.y + Roi.height / 2 - 1);

    nineGridRoiImg.NineCenter = Rect(nineGridRoiImg.RoiCenter - Point(Rwidth / 6, Rheight / 6),
        nineGridRoiImg.RoiCenter + Point(Rwidth / 6, Rheight / 6));

    nineGridRoiImg.NineUp = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 12,
        nineGridRoiImg.RoiCenter.y - Rheight / 2 + 1,
        Rwidth / 6, Rheight / 3);

    nineGridRoiImg.NineDown = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 12,
        nineGridRoiImg.RoiCenter.y + Rheight / 6 - 1,
        Rwidth / 6, Rheight / 3);

    nineGridRoiImg.NineLeft = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 2 + 1,
        nineGridRoiImg.RoiCenter.y - Rheight / 12,
        Rwidth / 3, Rheight / 6);

    nineGridRoiImg.NineRight = Rect(nineGridRoiImg.RoiCenter.x + Rwidth / 6 - 1,
        nineGridRoiImg.RoiCenter.y - Rheight / 12,
        Rwidth / 3, Rheight / 6);

    nineGridRoiImg.NineLeftUp = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 2 + 1,
        nineGridRoiImg.RoiCenter.y - Rheight / 2 + 1,
        Rwidth / 3, Rheight / 3);
    nineGridRoiImg.NineLeftDown = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 2 + 1,
        nineGridRoiImg.RoiCenter.y + Rheight / 6 + 1,
        Rwidth / 3, Rheight / 3);
    nineGridRoiImg.NineRightUp = Rect(nineGridRoiImg.RoiCenter.x + Rwidth / 6 + 1,
        nineGridRoiImg.RoiCenter.y - Rheight / 2 + 1,
        Rwidth / 3, Rheight / 3);
    nineGridRoiImg.NineRightDown = Rect(nineGridRoiImg.RoiCenter.x + Rwidth / 6 + 1,
        nineGridRoiImg.RoiCenter.y + Rheight / 6,
        Rwidth / 3, Rheight / 3);

    cv::rectangle(testImg, nineGridRoiImg.NineCenter, Scalar(0, 255, 0));
    cv::rectangle(testImg, nineGridRoiImg.NineUp, Scalar(0, 255, 0));
    cv::rectangle(testImg, nineGridRoiImg.NineDown, Scalar(0, 255, 0));
    cv::rectangle(testImg, nineGridRoiImg.NineLeft, Scalar(0, 255, 0));
    cv::rectangle(testImg, nineGridRoiImg.NineRight, Scalar(0, 255, 0));

    cv::rectangle(testImg, nineGridRoiImg.NineLeftUp, Scalar(0, 255, 0));
    cv::rectangle(testImg, nineGridRoiImg.NineLeftDown, Scalar(0, 255, 0));
    cv::rectangle(testImg, nineGridRoiImg.NineRightUp, Scalar(0, 255, 0));
    cv::rectangle(testImg, nineGridRoiImg.NineRightDown, Scalar(0, 255, 0));

    return testImg;
}

ROIFeature ExtractROIFeature(const cv::Mat& srcImg, const cv::Rect& roiRect, const std::string& roiName)
{
    //feat = ExtractROIFeature(rotationRoiImg, item.second, item.first);
    ROIFeature feat;
    feat.roiName = roiName;

    // 1. 裁剪 ROI 区域（防止越界）
    cv::Rect safeRoi = roiRect & cv::Rect(0, 0, srcImg.cols, srcImg.rows);
    cv::Mat roiImg = srcImg(safeRoi).clone();

    // 2. 转灰度
    cv::Mat gray;
    if (roiImg.channels() == 3)
        cv::cvtColor(roiImg, gray, cv::COLOR_BGR2GRAY);
    else
        gray = roiImg.clone();

    // 3. 计算原图 ROI 的平均灰度和标准差（用于后续分类）
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    feat.meanGray = mean[0];
    feat.stdGray = stddev[0];

    // 4. 大津法二值化
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 5. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 6. 取最大轮廓（假设 ROI 内只有一个主要目标）
    double maxArea = 0;
    int maxIdx = -1;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double a = cv::contourArea(contours[i]);
        if (a > maxArea)
        {
            maxArea = a;
            maxIdx = static_cast<int>(i);
        }
    }

    if (maxIdx >= 0)
    {
        const auto& cnt = contours[maxIdx];
        feat.area = maxArea;
        feat.perimeter = cv::arcLength(cnt, true);  // 待确认

        // 外接矩形 → 宽高
        cv::Rect bound = cv::boundingRect(cnt);
        feat.width = bound.width;
        feat.height = bound.height;
        feat.aspectRatio = (feat.height > 0) ? (feat.width / feat.height) : 0;

        // 圆形度 = 4π·area / perimeter²，越接近1越圆
        if (feat.perimeter > 0)
            feat.circularity = 4.0 * CV_PI * feat.area / (feat.perimeter * feat.perimeter);
        else
            feat.circularity = 0;
    }
    else
    {
        // 没找到轮廓，全部置 0
        std::cout << "find contours error！\n";
        feat.area = feat.perimeter = feat.width = feat.height = 0;
        feat.aspectRatio = feat.circularity = 0;
    }

    ////// ================= 新增：轮廓可视化调试 =================
    //// 1. 创建可视化画布（将灰度图转为3通道彩色图，方便画彩色轮廓）
    //cv::Mat visImg;
    //cv::cvtColor(gray, visImg, cv::COLOR_GRAY2BGR);

    //// 2. 绘制所有轮廓（用蓝色表示，便于观察二值化后的所有噪点）
    //cv::drawContours(visImg, contours, -1, cv::Scalar(255, 0, 0), 1);

    //// 3. 绘制最大轮廓（用红色加粗表示，确认选中的目标是否正确）
    //if (maxIdx >= 0)
    //{
    //    cv::drawContours(visImg, contours, maxIdx, cv::Scalar(0, 0, 255), 2);

    //    // 4. 绘制最大轮廓的外接矩形（用绿色表示，直观查看宽高）
    //    cv::Rect bound = cv::boundingRect(contours[maxIdx]);
    //    cv::rectangle(visImg, bound, cv::Scalar(0, 255, 0), 2);
    //}

    //// 5. 在图上标注 ROI 名称和面积，方便多窗口区分
    //std::string label = roiName + " Area:" + std::to_string((int)feat.area);
    //cv::putText(visImg, label, cv::Point(5, 20),
    //    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

    //// 6. 弹窗显示（窗口名带上 ROI 名称，避免被覆盖）
    //cv::namedWindow("ROI_Feat_" + roiName, WINDOW_NORMAL);
    //resizeWindow("ROI_Feat_" + roiName, Size(250, 250));
    //cv::imshow("ROI_Feat_" + roiName, visImg);
    //cv::waitKey(0);  // 建议改成10ms，批量跑图时不会卡住；单图调试可改回 0
    return feat;
}

vector<Rect> FindFullWhiteRectsByIntegral(const Mat& srcImg, int targetWidth, int targetHeight)
{
    vector<Rect> resultRects;

    // 1. 确保输入是单通道灰度图
    Mat gray;
    if (srcImg.channels() == 3) {
        cvtColor(srcImg, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = srcImg.clone();
    }

    // 2. 计算积分图 (必须是 CV_32F 或 CV_64F，防止累加溢出)
    Mat integralImg;
    cv::integral(gray, integralImg, CV_32F);    // 必须使用 cv:: 

    // 计算目标矩形的理论像素总和
    double targetSum = static_cast<double>(targetWidth * targetHeight * 250);

    // 3. 遍历图像，利用积分图 O(1) 计算每个窗口的像素和
    // 注意边界：积分图比原图多一行一列
    for (int y = 0; y <= gray.rows - targetHeight; ++y) {
        for (int x = 0; x <= gray.cols - targetWidth; ++x) {
            // 积分图计算矩形区域和的公式：S = I(x2,y2) - I(x1,y2) - I(x2,y1) + I(x1,y1)
            double sum = integralImg.at<float>(y + targetHeight, x + targetWidth)
                - integralImg.at<float>(y + targetHeight, x)
                - integralImg.at<float>(y, x + targetWidth)
                + integralImg.at<float>(y, x);

            // 如果像素和等于理论值，说明该区域全是255
            if (sum > targetSum) {
                resultRects.push_back(cv::Rect(x, y, targetWidth, targetHeight));
            }
        }
    }

    // 4. 合并重叠矩形
    if (!resultRects.empty()) {
        cv::groupRectangles(resultRects, 1, 0.2);
    }

    // 5.筛选有效矩形
    float minDist = 0;
    vector<Rect> validRects;
    for (size_t i = 0; i < resultRects.size(); ++i)
    {
        int count = 0;
        for (size_t j = 0; j < resultRects.size(); ++j)
        {
            if (i == j) continue;  // 用下标跳过自身，比 == 比较更可靠
            if ((abs(resultRects[i].x - resultRects[j].x) < resultRects[i].width * 2)
                && (abs(resultRects[i].y - resultRects[j].y) < resultRects[i].height * 0.5))
            {
                count++; break;
            }
        }
        if (count > 0) validRects.emplace_back(resultRects[i]);
    }
    //for (auto& reRect1 : resultRects)
    //{
    //    int count = 0;
    //    for (auto& reRect2 : resultRects)
    //    {
    //        if (reRect1 == reRect2) continue;
    //        //minDist = cv::norm(Point2f(reRect1.x, reRect1.y) - Point2f(reRect2.x, reRect2.y));
    //        if ((abs(reRect1.x - reRect2.x) < reRect1.width * 2) && (abs(reRect1.y- reRect2.y) < reRect1.height * 0.5))
    //        {
    //            cout << "\nabs(reRect1.x - reRect2.x)= " << abs(reRect1.x - reRect2.x)
    //                << " ，reRect1.width * 1.5 = " << reRect1.width * 1.5
    //                << " ，abs(reRect1.y- reRect2.y)= " << abs(reRect1.y - reRect2.y)
    //                << " ，reRect1.height * 0.5 = " << reRect1.height * 0.5 << endl;
    //            count++;
    //            //if (count >= 2) break;
    //        }
    //    }
    //    if (count > 0) validRects.emplace_back(reRect1);
    //}

    // 5.矩形水平排序
    std::cout << "\n--------------------****validRects.size() =  :" << validRects.size() << "****--------------------" << endl;

    std::sort(validRects.begin(), validRects.end(),
        [](const Rect& a, const Rect& b) {
            return a.x < b.x;
        });

    return validRects;
}

Mat WhiteRectVisualize(const Mat& img, const vector<Rect>& fullWhite, const GridMeasureResult& gridMeasureResult, const Rect roi)
{
    Mat testImg;
    if (img.channels() == 1) {
        cvtColor(img, testImg, COLOR_GRAY2BGR);
    }
    else {
        testImg = img.clone();
    }
    std::cout << "\n----------fullWhite.size:" << fullWhite.size() << endl;

    int rectDistence = fullWhite[4].x - fullWhite[3].x;

    Point pointRx3(fullWhite[4].x, fullWhite[4].y);
    //Point pointLD(fullWhite[0].x - rectDistence * 1.5, fullWhite[0].y);

    line(testImg, pointRx3, pointRx3 + Point(0, -210), Scalar(127, 255, 255), 1);
    //line(testImg, pointLD, pointLD + Point(0, -210), Scalar(127, 255, 255), 1);

    for (auto fw : fullWhite)
    {
        int x = fw.x + fw.width / 2;
        int y = fw.y + fw.height / 2;
        //circle(testImg, Point(x, y), 1, Scalar(0, 255, 0), -1);
        line(testImg, Point(x - 1, y), Point(x + 1, y), Scalar(0, 255, 0), 1);
        line(testImg, Point(x, y - 1), Point(x, y + 1), Scalar(0, 255, 0), 1);
        rectangle(testImg, fw, cv::Scalar(0, 0, 255), 1);
    }
    for (auto ac : gridMeasureResult.allCircle)
    {
        circle(testImg, Point(ac[0], ac[1]), 1, Scalar(0, 255, 0));

    }
    cv::putText(testImg, cv::format("H: %.1f px", gridMeasureResult.hDist), cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    cv::putText(testImg, cv::format("V: %.1f px", gridMeasureResult.vDist), cv::Point(10, 60),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    cv::putText(testImg, cv::format("R: %.1f px", gridMeasureResult.avgRadius), cv::Point(10, 90),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

    rectangle(testImg, roi, Scalar(0, 255, 0));

    imshow("WhiteRectVisualize", testImg);
    waitKey(0);
    return testImg;
}

Rect GetRegionRoi(const vector<Rect>& fullWhite, GridMeasureResult gridMeasureResult)
{
    Point pointRx3(fullWhite[fullWhite.size() - 3].x, fullWhite[fullWhite.size() - 3].y);
    Point pointLx2(fullWhite[1].x, fullWhite[1].y);

    double dist = cv::norm(pointRx3 - pointLx2);

    int width = 0, radius = 0, height = 0;

    if (gridMeasureResult.hDist > 0 && gridMeasureResult.vDist > 0)
    {
        //width = gridMeasureResult.hDist * 3;
        radius = gridMeasureResult.avgRadius;
        height = gridMeasureResult.vDist * 3 - radius * 3 - 8;
    }
    else
    {
        std::cout << "霍夫圆检测ROI失败，使用经验值\n";
        gridMeasureResult.hDist = 62;
        gridMeasureResult.vDist = 61;
        gridMeasureResult.avgRadius = 12;
        //width = gridMeasureResult.hDist * 3;
        radius = gridMeasureResult.avgRadius;
        height = gridMeasureResult.vDist * 3 - radius * 3 - 8;
    }
    Point pointRD(pointRx3.x, pointRx3.y - gridMeasureResult.vDist - radius - 4);

    width = static_cast<int>(dist) * 2 - fullWhite[0].width;   // 2

    Rect Roi(pointRD.x - width + 1, pointRD.y - height + 1, width, height);

    return Roi;
}

// 求倾斜 ROI 的 RotatedRect（中心、宽高、角度）
cv::RotatedRect GetRegionRotatedROI(const std::vector<cv::Rect>& fullWhite, GridMeasureResult gridMeasureResult)
{
    // ---- 1. 参考点 ----
    cv::Point pointRx3(fullWhite[fullWhite.size() - 3].x, fullWhite[fullWhite.size() - 3].y);
    cv::Point pointLx2(fullWhite[1].x, fullWhite[1].y);
    double dist = cv::norm(pointRx3 - pointLx2);

    // ---- 2. 宽高 ----
    int width = 0, radius = 0, height = 0;
    if (gridMeasureResult.hDist > 0 && gridMeasureResult.vDist > 0) {
        //width = static_cast<int>(gridMeasureResult.hDist * 3);
        radius = static_cast<int>(gridMeasureResult.avgRadius);
        height = static_cast<int>(gridMeasureResult.vDist * 3 - radius * 3);
    }
    else {
        std::cout << "霍夫圆检测ROI失败，使用经验值" << std::endl;
        gridMeasureResult.hDist = 62;
        gridMeasureResult.vDist = 61;
        gridMeasureResult.avgRadius = 12;
        //width = static_cast<int>(gridMeasureResult.hDist * 3);
        radius = static_cast<int>(gridMeasureResult.avgRadius);
        height = static_cast<int>(gridMeasureResult.vDist * 3 - radius * 3);
    }

    width = static_cast<int>(dist) * 2 - fullWhite[0].width;

    // ---- 3. 算倾斜角 theta = std::atan2(dy, dx); 逆时针为负 ----// 从左到右
    double dx = fullWhite[fullWhite.size() - 1].x - fullWhite[0].x;
    double dy = fullWhite[fullWhite.size() - 1].y - fullWhite[0].y;
    double theta = std::atan2(dy, dx);
    //std::cout << "倾斜角 theta = " << theta * 180.0 / CV_PI << " 度" << std::endl;

    // ---- 4. 求倾斜后的右下角点 ----
    double d = gridMeasureResult.vDist + radius;   // 沿“向上”方向偏移的距离

    cv::Point2f pointRD;
    double cosT = std::cos(theta);
    double sinT = std::sin(theta);

    if (std::abs(theta) < 5.0 * CV_PI / 180.0) {
        // 倾斜小，竖直向上
        pointRD = cv::Point2f(pointRx3.x, pointRx3.y - static_cast<float>(d));
        std::cout << "不旋转" << std::endl;
    }
    else {
        // 把偏移向量 (0, -d) 旋转 theta
        double dx = 0.0, dy = -d;
        double dxr = dx * cosT - dy * sinT;
        double dyr = dx * sinT + dy * cosT;
        pointRD = cv::Point2f(pointRx3.x + static_cast<float>(dxr),
            pointRx3.y + static_cast<float>(dyr));
    }

    // ---- 5. 算 RotatedRect 的中心 ----
    // 右下角点 -> 中心：向左上偏移 (width/2, height/2)，再旋转 theta
    double cx = -width / 2.0;
    double cy = -height / 2.0;
    double cxr = cx * cosT - cy * sinT;
    double cyr = cx * sinT + cy * cosT;

    cv::Point2f center(pointRD.x + static_cast<float>(cxr),
        pointRD.y + static_cast<float>(cyr));

    // ---- 6. 构造 RotatedRect ----(矩形中心点Point2f、矩形宽高Size、旋转角度float)
    // 注意：OpenCV 的 RotatedRect::angle 是顺时针为正，theta 是图像坐标系下的角
    cv::RotatedRect rr(center, cv::Size2f(static_cast<float>(width),
        static_cast<float>(height)),
        static_cast<float>(theta * 180.0 / CV_PI));

    return rr;
}

// 从 RotatedRect 提取摆正后的 ROI 图像
cv::Mat ExtractRotatedROI(const cv::Mat& img, const cv::RotatedRect& rr)
{
    int outW = static_cast<int>(rr.size.width);
    int outH = static_cast<int>(rr.size.height);
    // 1. 取 RotatedRect 的 4 个角点
    cv::Point2f pts[4];
    rr.points(pts);

    // 2. 排序：左上、右上、右下、左下
    std::vector<cv::Point2f> v(pts, pts + 4); // 用迭代器区间构造：pts 是首指针，pts + 4 是尾后指针，把 4 个点拷进 v。
    std::sort(v.begin(), v.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) {
            return (a.x + a.y) < (b.x + b.y);
        });
    cv::Point2f tl = v[0], br = v[3];
    cv::Point2f tr, bl;
    if (v[1].x > v[2].x) { tr = v[1]; bl = v[2]; }
    else { tr = v[2]; bl = v[1]; }

    cv::Point2f srcOrdered[4] = { tl, tr, br, bl };
    cv::Point2f dstOrdered[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(outW - 1.0f, 0),
        cv::Point2f(outW - 1.0f, outH - 1.0f),
        cv::Point2f(0, outH - 1.0f)
    };

    // 3. 透视变换，输出固定 W x H
    cv::Mat M = cv::getPerspectiveTransform(srcOrdered, dstOrdered);
    cv::Mat roi;
    cv::warpPerspective(img, roi, M, cv::Size(outW, outH),
        cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return roi;
}

GridMeasureResult MeasureCircleGridDistances(const Mat& srcImg) {
    // 1. 预处理与霍夫圆检测
    GridMeasureResult griMeasureResult;
    Mat gray;
    if (srcImg.channels() == 3) {
        cvtColor(srcImg, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = srcImg.clone();
    }
    medianBlur(gray, gray, 5);

    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 20, 100, 30, 10, 20);

    if (circles.size() < 7) {
        std::cout << "检测到的圆孔数量不足！" << std::endl;
        return griMeasureResult;
    }
    //std::cout << "检测到的圆孔数量：" << circles.size() << std::endl;
    // 2. 提取圆心并按坐标排序
    vector<Point2f> centers;
    for (const auto& c : circles) {
        centers.emplace_back(c[0], c[1]);
    }

    vector<float> xCoords, yCoords;
    for (const auto& p : centers) {
        xCoords.push_back(p.x);
        yCoords.push_back(p.y);
    }
    sort(xCoords.begin(), xCoords.end());
    std::sort(yCoords.begin(), yCoords.end());

    // 3. 统计所有相邻坐标差值（保留浮点精度，不再取整）
    const float offset = 5.0f; // 容差窗口：±5像素
    vector<float> xDists, yDists;

    for (size_t i = 0; i < xCoords.size() - 1; ++i) {
        float d = xCoords[i + 1] - xCoords[i];
        if (d > 10)
        {
            xDists.push_back(d);
            //std::cout << "\n xDists: " << d;
        }
    }
    for (size_t i = 0; i < yCoords.size() - 1; ++i) {
        float d = yCoords[i + 1] - yCoords[i];
        if (d > 10)
        {
            yDists.push_back(d);
            //std::cout << "\n yDists: " << d;
        }
    }
    // 4. 核心：带 offset 容差的众数统计
    // 思路：遍历所有距离值，对每个距离值统计 [val - offset, val + offset] 范围内有多少个距离落入
    // 取落入数量最多的那个距离值的均值作为最终结果
    auto findMostFrequentWithTolerance = [&](const vector<float>& dists, float tol)
        -> pair<float, int> {
        if (dists.empty()) return { 0.0f, 0 };

        float bestCenter = 0.0f;
        int maxCount = 0;

        for (size_t i = 0; i < dists.size(); ++i) {
            float center = dists[i];
            int count = 0;
            float sum = 0.0f;
            for (size_t j = 0; j < dists.size(); ++j) {
                if (std::abs(dists[j] - center) <= tol) {
                    count++;
                    sum += dists[j];
                }
            }
            if (count > maxCount) {
                maxCount = count;
                bestCenter = sum / count; // 用范围内均值作为最终间距
            }
        }
        return { bestCenter, maxCount };
        };

    auto [hDist, hCount] = findMostFrequentWithTolerance(xDists, offset);
    //auto [vDist, vCount] = findMostFrequentWithTolerance(yDists, offset);
    float vDist = 0.0;
    //std::cout << "\nyCoords.size():" << yDists.size() << endl;
    for (int i = 0; i < yDists.size(); i++)
    {
        if (hCount >= 2 && abs(yDists[i] - hDist) < 8)
        {
            vDist = yDists[i];
        }
    }

    // 5. 输出结果
   /* std::cout << "================ 测量结果 ================" << endl;
    std::cout << "水平方向最密集间距: " << hDist << " px (出现 " << hCount << " 次)" << endl;
    std::cout << "垂直方向最密集间距: " << vDist << endl;

    if (hDist > 0 && vDist > 0) {
        double ratio = static_cast<double>(hDist) / vDist;
        std::cout << "水平与垂直间距比例 (H:V) = 1 : " << ratio << endl;
    }*/
    float radiusSum = 0.0f;
    for (const auto& c : circles) {
        radiusSum += c[2];
    }
    griMeasureResult.avgRadius = radiusSum / circles.size();
    //griMeasureResult.avgRadius = circles[0][2];
    griMeasureResult.hDist = hDist;
    griMeasureResult.vDist = vDist;
    griMeasureResult.allCircle = circles;
    //std::cout << "==========================================" << endl;

    //// 6. 可视化
    //Mat debugImg = srcImg.clone();
    //if (srcImg.channels() == 1) cv::cvtColor(debugImg, debugImg, cv::COLOR_GRAY2BGR);

    //for (const auto& center : centers) {
    //    cv::circle(debugImg, center, 3, cv::Scalar(0, 255, 0), -1);
    //}
    //cv::putText(debugImg, cv::format("H: %.1f px", hDist), cv::Point(10, 30),
    //    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    //cv::putText(debugImg, cv::format("V: %.1f px", vDist), cv::Point(10, 60),
    //    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

    //cv::imshow("Simple Grid Measurement", debugImg);
    //cv::waitKey(0);

    return griMeasureResult;
}

// ===== SVM 批量训练函数 =====
#include <opencv2/ml.hpp> 
void TrainSVMFromFile(const string& featureFilePath) 
{
    std::cout << "==========  1. 加载特征数据:" << featureFilePath << " ==========" << std::endl;
    FileStorage fs(featureFilePath, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "无法打开特征文件: " << featureFilePath << endl;
        return;
    }
    Mat trainData, labels;
    fs["features"] >> trainData;
    fs["labels"] >> labels;
    fs.release();

    int totalSamples = trainData.rows;
    int featDim = trainData.cols;

    if (totalSamples < 10) {
        cerr << "错误: 样本数太少 (" << totalSamples << ")，无法训练！" << endl;
        return;
    }
    // cv::Size(width, height) = (宽度、高度) = （列数，行数）
    std::cout << "----训练样本特征trainData.size()=(宽度、高度) = （列数，行数）: " << trainData.size() << " ,训练样本标签labels.size()=(宽度、高度) = （列数，行数）: " << labels.size() << endl;
    std::cout << "----样本数trainData.rows行数: " << totalSamples << " ,特征维度trainData.cols: " << featDim << endl;

    std::cout << "========== 特征归一化（Z-Score） ==========" << std::endl;
    // 算每列均值（行向量 1 x featDim，CV_32F）
    cv::Mat meanRow;
    cv::reduce(trainData, meanRow, 0, cv::REDUCE_AVG, CV_32F);  // dim=0 → 每列均值，1×35
    std::cout << "每列特征平均值meanRow.size()=（列数，行数）: " << meanRow.size() << endl;
    cv::Mat meanMatrix;
    cv::repeat(meanRow, trainData.rows, 1, meanMatrix); // 把均值行向量复制成1×35 → 199×35
    std::cout << "meanMatrix.size()=（列数，行数）: " << meanMatrix.size() << endl;

    std::cout << "========== 特征归一化：3.算每列标准差:减均值，求方差，开方 ==========" << std::endl;
    cv::Mat centered;
    cv::subtract(trainData, meanMatrix, centered);
    cv::Mat sq;
    cv::multiply(centered, centered, sq);

    cv::Mat stdRow;
    cv::reduce(sq, stdRow, 0, cv::REDUCE_AVG, CV_32F);  // 标准差要 总和 除以 样本数量
    cv::sqrt(stdRow, stdRow);
    std::cout << "每列特征标准差 stdRow.size()=（列数，行数）: " << stdRow.size()
        <<",\tstdRow.type():"<< stdRow.type()
        << ",\tstdRow.cols：" << stdRow.cols << ",\tstdRow.rows：" << stdRow.rows
        << endl;

    std::cout << "========== 4. 归一化 ==========" << std::endl;
    for (int col = 0; col < trainData.cols; ++col) {
        float m = meanRow.at<float>(0, col);
        float s = stdRow.at<float>(0, col);
        if (s < 1e-6f) s = 1.0f;   // 除零保护
        for (int row = 0; row < trainData.rows; ++row) {
            // Z-Score 归一化：对每一列（每个特征）单独减均值、除以标准差。
            trainData.at<float>(row, col) = (trainData.at<float>(row, col) - m) / s;
        }
    }
    std::cout << "========== 随机打乱数据 ==========" << std::endl;
    int okNum = 100, badNum = 99;
    //cv::RNG rng(12345); // OpenCV 随机数生成器，固定种子结果可复现；换成 cv::RNG rng(cv::getTickCount()); 每次不同
    const unsigned SEED = 99999;
    std::mt19937 rng(SEED);   // 固定种子，结果可复现
    std::vector<int> idx(totalSamples);
    std::iota(idx.begin(), idx.end(), 0);   // std::iota 从 0 开始递增填充：idx = {0, 1, 2, ..., N-1}。
    // 用 rng 把 idx 里的数字随机重排。
    std::shuffle(idx.begin(), idx.begin() + 100, rng);// 只打乱前 100 个元素（索引 0~99）
    std::shuffle(idx.begin() + 100, idx.end(), rng); // 只打乱后 99 个元素（索引 100~198）
    // === 保存打乱索引 ===
    {
        cv::Mat idxMat(idx);
        cv::FileStorage fs("shuffle_9_idx.yml", cv::FileStorage::WRITE);
        fs << "shuffle_idx" << idxMat;
        fs << "okNum" << okNum;
        fs << "badNum" << badNum;
        fs << "seed" << static_cast<int>(SEED);
        fs.release();
        std::cout << "打乱索引已保存至 shuffle_9_idx.yml" << std::endl;
    }
    Mat dataShuf(totalSamples, featDim, CV_32F);
    Mat labelShuf(totalSamples, 1, CV_32S);
    for (int i = 0; i < totalSamples; ++i) {
        trainData.row(idx[i]).copyTo(dataShuf.row(i));  // 打乱索引的数据特征
        labelShuf.at<int>(i, 0) = labels.at<int>(idx[i], 0);  // 标签
    }

    std::cout << "========== 划分训练/测试集  ==========" << endl;
    Mat trainSamples, trainLabels;
    int trainCount = static_cast<int>(okNum * 0.8);
    cv::vconcat(dataShuf.rowRange(0, trainCount),
        dataShuf.rowRange(100, 100 + trainCount), trainSamples);  // rowRange 是左闭右开：owRange(0, 159)：取第 0~158 行（前 159 行）
    cv::vconcat(labelShuf.rowRange(0, trainCount),
        labelShuf.rowRange(100, 100 + trainCount), trainLabels);

    Mat testSamples, testLabels;
    cv::vconcat(dataShuf.rowRange(trainCount, 100),
        dataShuf.rowRange(100 + trainCount, totalSamples), testSamples);
    cv::vconcat(labelShuf.rowRange(trainCount, 100),
        labelShuf.rowRange(100 + trainCount, totalSamples), testLabels);

    std::cout << "\n========== 数据集统计 ==========" << endl;
    std::cout << "总样本数: " << totalSamples << endl;
    std::cout << "训练集trainSamples.rows: " << trainSamples.rows << " 张" << endl;
    std::cout << "测试集testSamples.rows: " << testSamples.rows << " 张" << endl;

    // 4. 训练 SVM
    auto svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);

    //svm->setKernel(cv::ml::SVM::LINEAR); // 线性核
    svm->setKernel(cv::ml::SVM::RBF);   // RBF 核
    svm->setGamma(1.0 / 63.0);  // 1 / 特征维度

    svm->setC(1.0);
    svm->setTermCriteria(cv::TermCriteria(
        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,  // 两个条件同时用
        1000,      // 最大迭代次数
        1e-6));    // 收敛精度
    std::cout << "\n========== 开始训练SVM ==========" << endl;
    svm->trainAuto(trainSamples, cv::ml::ROW_SAMPLE, trainLabels, 10); // 10折交叉验证
    //svm->train(trainSamples, cv::ml::ROW_SAMPLE, trainLabels);
    // balanced = true 有助于处理类别不平衡 (你的数据 ok:100, bad:99 基本平衡，可设 false)
    std::cout << "训练完成！" << endl;
    std::cout << "自动搜索到的最优参数:" << endl;
    std::cout << "  C     = " << svm->getC() << endl;
    std::cout << "  Gamma = " << svm->getGamma() << endl;
    std::cout << "  支持向量数 = " << svm->getSupportVectors().rows << endl;
    EvaluateSVM(svm, trainSamples, trainLabels, "训练集评估");
    // 5. 测试集评估
    EvaluateSVM(svm, testSamples, testLabels, "测试集评估");
    int correct = 0;
    int total = testSamples.rows;
    std::cout << "\n========== 测试集预测结果 ==========" << endl;
    for (int i = 0; i < total; ++i) {
        float response = svm->predict(testSamples.row(i));
        //int predicted = static_cast<int>(response);
        int predicted = (response > 0.5f) ? 1 : 0;
        int actual = testLabels.at<int>(i, 0);
        //int actual = static_cast<int>(testLabels.at<float>(i, 0));
        string result = (predicted == actual) ? "OK" : "NG";
        if (predicted == actual) correct++;
        std::cout << "  样本[" << i << "] 预测=" << (predicted == 0 ? "好图" : "坏图")
            << " 实际=" << (actual == 0 ? "好图" : "坏图")
            << " [" << result << "]" << endl;
    }

    float accuracy = static_cast<float>(correct) / total * 100.0f;
    std::cout << "\n准确率: " << correct << "/" << total << " = " << accuracy << "%" << endl;

    // 6. 保存模型和归一化参数
    string svmName = "SVM_Model_R92_" + to_string(correct) + "-" + to_string(total) + "_.xml";
    svm->save(svmName);
    std::cout << "模型已保存至: " << svmName << endl;

    string fsNormName = "NormParams_R92_" + to_string(correct) + "-" + to_string(total) + "_.xml";
    FileStorage fsNorm(fsNormName, cv::FileStorage::WRITE);
    fsNorm << "mean" << meanRow;
    fsNorm << "stddev" << stdRow;
    fsNorm.release();
    std::cout << "归一化参数已保存至: " << fsNormName << endl;
}

// ================== 模块1：特征提取与保存 ==================
void ExtractAndSaveFeatures(const string& goodImgDir, const string& badImgDir, const string& savePath) {
    vector<Mat> allFeatures;
    vector<int> allLabels;

    // 辅助 lambda：处理单个文件夹
    auto processDir = [&](const string& dir, int label) {
        int processed = 0;
        int skipped = 0;
        for (const auto& entry : filesystem::directory_iterator(dir)) {
            string ext = entry.path().extension().string(); // 扩展名字符串,如 .jpg
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);  // 转小写
            if (ext != ".jpg" && ext != ".png" && ext != ".bmp") continue;

            string imgPath = entry.path().string(); // 完整路径字符串
            std::cout << "     ==>>第" << processed << "张图片：" << imgPath;
            Mat testImg = imread(imgPath);
            if (testImg.empty()) { skipped++; continue; }

            Mat featVec = ExtractOneImageFeature(testImg); // 提取一张图片特征 1 * 63 :(7 * 9) 一行
            if (featVec.cols != 63 || featVec.type() != CV_32F) {
                std::cout << "异常: " << imgPath
                    << " cols=" << featVec.cols
                    << " type=" << featVec.type() << std::endl;
                skipped++;
                continue;
            }
            allFeatures.push_back(featVec);
            allLabels.push_back(label);
            processed++;

            /*if (processed % 20 == 0)
                std::cout << "  已处理 " << processed << " 张" << endl;*/
        }
        std::cout << "  " << (label == 0 ? "好图" : "坏图") << " 完成: 成功" << processed
            << " 张, 跳过 " << skipped << " 张" << endl;
        };
    std::cout << "========== 开始处理好图 ==========" << endl;
    processDir(goodImgDir, 0);
    std::cout << "\n========== 开始处理坏图 ==========" << endl;
    processDir(badImgDir, 1);

    if (allFeatures.empty()) {
        cerr << "错误: 未提取到任何有效样本！" << endl;
        return;
    }

    FileStorage fs(savePath, FileStorage::WRITE);   // 创建 或 先清空文件
    if (!allFeatures.empty()) {
        cv::Mat featureMatrix;
        cv::vconcat(allFeatures, featureMatrix); // 将 vector<Mat> 拼接成 N行 x M列 的大矩阵
        fs << "features" << featureMatrix;
    }
    if (!allLabels.empty()) {
        cv::Mat labelMatrix(allLabels, true); // 将 vector<int> 转为单列 Mat
        fs << "labels" << labelMatrix;
    }
    fs.release();
    std::cout << "特征数据已保存至: " << savePath << endl;
}

#include <string>
// 假设你之前的特征提取辅助函数和结构体已经定义在头文件中
int PredictImage(const string& modelPath, const string& normPath, const string& imgPath) {
    // 1. 加载 SVM 模型
    Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(modelPath);
    if (svm.empty()) {
        cerr << "无法加载模型: " << modelPath << endl;
        return -1;
    }

    // 2. 加载归一化参数
    FileStorage fsNorm(normPath, FileStorage::READ);
    if (!fsNorm.isOpened()) {
        cerr << "无法加载归一化参数: " << normPath << endl;
        return -1;
    }
    Mat meanRow, stdRow;
    fsNorm["mean"] >> meanRow;
    fsNorm["stddev"] >> stdRow;
    fsNorm.release();

    // 3. 读取图片并提取特征（必须与训练时的逻辑完全一致）
    Mat testImg = imread(imgPath);
    if (testImg.empty()) {
        cerr << "无法读取图片: " << imgPath << endl;
        return -1;
    }

    Mat featVec = ExtractOneImageFeature(testImg); // 提取一张图片特征 1 * 63 :(7 * 9) 一行

    // 4. 使用归一化参数对特征进行标准化（必须与训练时一致）
    if (featVec.cols != 63 || featVec.type() != CV_32F) {
        std::cout << "异常: " << imgPath
            << " cols=" << featVec.cols
            << " type=" << featVec.type() << std::endl;
        return -1;
    }
    for (int c = 0; c < featVec.cols; ++c) {
        float m = meanRow.at<float>(0, c);
        float s = stdRow.at<float>(0, c);
        if (s < 1e-6f) s = 1.0f;
        featVec.at<float>(0, c) = (featVec.at<float>(0, c) - m) / s;
    }

    // 5. 使用 SVM 模型进行预测
    float response = svm->predict(featVec);
    //int predicted = static_cast<int>(response);
    int predicted = (response > 0.5f) ? 1 : 0;

    cout << "图片: " << imgPath << " 预测结果: " << response
        << (predicted == 0 ? " -> 好图" : "坏图") << endl;

    return predicted;
}

void EvaluateSVM(const cv::Ptr<cv::ml::SVM>& svm,
    const cv::Mat& samples,
    const cv::Mat& labels,
    const std::string& tag)
{
    int total = samples.rows;
    int correct = 0, tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i = 0; i < total; ++i) {
        float response = svm->predict(samples.row(i));
        int predicted = (response > 0.5f) ? 1 : 0;
        int actual = labels.at<int>(i, 0);

        if (predicted == actual) correct++;
        if (actual == 1 && predicted == 1) tp++;
        else if (actual == 0 && predicted == 0) tn++;
        else if (actual == 0 && predicted == 1) fp++;
        else if (actual == 1 && predicted == 0) fn++;
    }

    float acc = static_cast<float>(correct) / total * 100.0f;
    float precision = (tp + fp) > 0 ? static_cast<float>(tp) / (tp + fp) * 100.0f : 0.0f;
    float recall = (tp + fn) > 0 ? static_cast<float>(tp) / (tp + fn) * 100.0f : 0.0f;
    float f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;

    std::cout << "\n========== " << tag << " ==========" << endl;
    std::cout << "样本数: " << total << endl;
    std::cout << "准确率: " << correct << "/" << total << " = " << acc << "%" << endl;
    std::cout << "混淆矩阵 (正类=坏图): TP=" << tp
        << "  FP=" << fp << "  FN=" << fn << "  TN=" << tn << endl;
    std::cout << "Precision=" << precision
        << "%  Recall=" << recall
        << "%  F1=" << f1 << "%" << endl;
}

#include <filesystem>
namespace fs = std::filesystem;
// ============================================================
// 批量测试文件夹下所有图片
// ============================================================
void BatchPredictFolder(const string& modelPath,
    const string& normPath,
    const string& folderPath,
    bool recursive)
{
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        cerr << "目录不存在: " << folderPath << endl;
        return;
    }

    // 收集图片路径
    vector<string> imgPaths;
    auto collect = [&](const fs::directory_entry& entry) {
        if (!entry.is_regular_file()) return;
        string ext = entry.path().extension().string();
        transform(ext.begin(), ext.end(), ext.begin(),
            [](unsigned char c) { return static_cast<char>(tolower(c)); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
            ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
            imgPaths.push_back(entry.path().string());
        }
        };

    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(folderPath))
            collect(entry);
    }
    else {
        for (const auto& entry : fs::directory_iterator(folderPath))
            collect(entry);
    }

    sort(imgPaths.begin(), imgPaths.end());

    cout << "\n========== 批量预测开始 ==========" << endl;
    cout << "文件夹: " << folderPath << endl;
    cout << "图片数量: " << imgPaths.size() << endl;

    int goodCount = 0, badCount = 0, failCount = 0;
    vector<string> failList;

    for (size_t i = 0; i < imgPaths.size(); ++i) {
        cout << "[" << (i + 1) << "/" << imgPaths.size() << "] ";
        int r = PredictImage(modelPath, normPath, imgPaths[i]);
        if (r == 0)      goodCount++;
        else if (r == 1) badCount++;
        else {
            failCount++;
            failList.push_back(imgPaths[i]);
        }
    }

    cout << "\n========== 批量预测统计 ==========" << endl;
    cout << "总数: " << imgPaths.size() << endl;
    cout << "好图: " << goodCount << endl;
    cout << "坏图: " << badCount << endl;
    cout << "失败: " << failCount << endl;

    if (!failList.empty()) {
        cout << "\n失败列表:" << endl;
        for (const auto& p : failList)
            cout << "  " << p << endl;
    }
}


// --- 核心图像处理与特征提取逻辑（与你训练时一模一样）---
Mat ExtractOneImageFeature(const Mat& testImg)
{
    vector<Rect> fullWhite = FindFullWhiteRectsByIntegral(testImg);
    if (fullWhite.size() != 7) {
        cerr << "图片特征点数量不符，跳过" << endl;
        return Mat();
    }

    GridMeasureResult gridMeasureResult = MeasureCircleGridDistances(testImg);

    // ---- 3. 算倾斜角 theta = std::atan2(dy, dx); 逆时针为负 ----
    // 从左到右
    double dx = fullWhite[fullWhite.size() - 1].x - fullWhite[0].x;
    double dy = fullWhite[fullWhite.size() - 1].y - fullWhite[0].y;
    double theta = std::atan2(dy, dx);
    std::cout << "倾斜角 theta = " << theta * 180.0 / CV_PI << " 度" << std::endl;
    Rect Roi;
    Mat rotationRoiImg;
    if (std::abs(theta) < 1.0 * CV_PI / 180.0) {
        std::cout << "不旋转" << std::endl;
        Roi = GetRegionRoi(fullWhite, gridMeasureResult);
    }
    else {
        // 1. 求倾斜 ROI
        cv::RotatedRect rr = GetRegionRotatedROI(fullWhite, gridMeasureResult);

        // 2. 摆正并提取，输出固定 width x height
        rotationRoiImg = ExtractRotatedROI(testImg, rr);
        Roi = Rect(0, 0, rotationRoiImg.cols, rotationRoiImg.rows);
    }

    int Rwidth = Roi.width;
    int Rheight = Roi.height;

    // nineGridRoiImg 结构体
    nineGridRoiImg.RoiCenter = Point(Roi.x + Roi.width / 2 - 1, Roi.y + Roi.height / 2 - 1);

    nineGridRoiImg.NineCenter = Rect(nineGridRoiImg.RoiCenter - Point(Rwidth / 6, Rheight / 6),
        nineGridRoiImg.RoiCenter + Point(Rwidth / 6, Rheight / 6));
    nineGridRoiImg.NineUp = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 12,
        nineGridRoiImg.RoiCenter.y - Rheight / 2 + 1, Rwidth / 6, Rheight / 3);
    nineGridRoiImg.NineDown = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 12,
        nineGridRoiImg.RoiCenter.y + Rheight / 6 - 1, Rwidth / 6, Rheight / 3);
    nineGridRoiImg.NineLeft = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 2 + 1,
        nineGridRoiImg.RoiCenter.y - Rheight / 12, Rwidth / 3, Rheight / 6);
    nineGridRoiImg.NineRight = Rect(nineGridRoiImg.RoiCenter.x + Rwidth / 6 - 1,
        nineGridRoiImg.RoiCenter.y - Rheight / 12, Rwidth / 3, Rheight / 6);

    nineGridRoiImg.NineLeftUp = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 2 + 1,
        nineGridRoiImg.RoiCenter.y - Rheight / 2 + 1,
        Rwidth / 3, Rheight / 3);
    nineGridRoiImg.NineLeftDown = Rect(nineGridRoiImg.RoiCenter.x - Rwidth / 2 + 1,
        nineGridRoiImg.RoiCenter.y + Rheight / 6 + 1,
        Rwidth / 3, Rheight / 3);
    nineGridRoiImg.NineRightUp = Rect(nineGridRoiImg.RoiCenter.x + Rwidth / 6 + 1,
        nineGridRoiImg.RoiCenter.y - Rheight / 2 + 1,
        Rwidth / 3, Rheight / 3);
    nineGridRoiImg.NineRightDown = Rect(nineGridRoiImg.RoiCenter.x + Rwidth / 6 + 1,
        nineGridRoiImg.RoiCenter.y + Rheight / 6,
        Rwidth / 3, Rheight / 3);

    vector<pair<string, Rect>> roiList = {
        {"Center", nineGridRoiImg.NineCenter}, {"Up", nineGridRoiImg.NineUp},
        {"Down", nineGridRoiImg.NineDown}, {"Left", nineGridRoiImg.NineLeft},
        {"Right", nineGridRoiImg.NineRight},{"LeftUp",  nineGridRoiImg.NineLeftUp},
        {"LeftDown",  nineGridRoiImg.NineLeftDown}, {"RightUp",  nineGridRoiImg.NineRightUp},
        {"RightDown",  nineGridRoiImg.NineRightDown},
    };

    const int FEAT_PER_ROI = 7;
    const int TOTAL_DIM = roiList.size() * FEAT_PER_ROI;
    Mat featVec(1, TOTAL_DIM, CV_32F);
    int col = 0;
    for (const auto& item : roiList) {
        ROIFeature feat;
        if (std::abs(theta) < 1.0 * CV_PI / 180.0)
            feat = ExtractROIFeature(testImg, item.second, item.first);
        else
            feat = ExtractROIFeature(rotationRoiImg, item.second, item.first);

        featVec.at<float>(0, col++) = static_cast<float>(feat.area);
        featVec.at<float>(0, col++) = static_cast<float>(feat.perimeter);
        featVec.at<float>(0, col++) = static_cast<float>(feat.width);
        featVec.at<float>(0, col++) = static_cast<float>(feat.height);
        featVec.at<float>(0, col++) = static_cast<float>(feat.aspectRatio);
        featVec.at<float>(0, col++) = static_cast<float>(feat.circularity);
        featVec.at<float>(0, col++) = static_cast<float>(feat.meanGray);
    }
    return featVec;
}