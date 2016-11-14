//#include <stdio.h>
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Windows.h>
#include <opencv2\core\cuda_devptrs.hpp>
using namespace cv;
using namespace std;
using namespace cv::gpu;
int main()
{
	double cost1, cost2;
	DWORD startTime, endTime;
	string inputfile;
	cout << "请输入图片编号：";
	cin >> inputfile;
	Mat srcImage = imread(inputfile+".jpg");     //默认.jpg后缀
	imshow("【原始】 canny边缘检测", srcImage);
	Mat dstImage,edge, grayImage,temp;
	
	gpu::GpuMat eg, gray;	
	dstImage.create(srcImage.size(), srcImage.type());
	
    //原图转换成灰度图像
	cvtColor(srcImage, grayImage, CV_BGR2GRAY); 
	//上传灰度图
    gray.upload(grayImage);
	//对图像降噪处理
	blur(grayImage, edge, Size(3, 3));

	//执行canny算子
	startTime = GetTickCount();   //记录处理的起始时间
	for (int i = 0; i < 100;i++)  //图片文件太小的时候加上，文件足够大则注释掉此行
	Canny(edge, edge, 3, 9, 3);
	endTime = GetTickCount();      //记录处理的结束时间
	imshow("【效果图】 canny", edge);
	cost1 = endTime - startTime;
	cout << "处理所用时间为：" << cost1<<"ms"<<endl;


	/*以下为利用GPU加速，进行相同的处理流程*/
	//降噪处理
	gpu::blur(gray, eg, Size(3, 3));
	startTime = GetTickCount();   //记录处理的起始时间
	for (int i = 0; i < 100; i++)          //图片文件太小的时候加上，文件足够大则注释掉此行
	gpu::Canny(eg, eg, 3, 9, 3);
	endTime = GetTickCount();      //记录处理的结束时间
	eg.download(temp);             //载入在GPU处理过的图片并显示
	imshow("【效果图】 GPU加速", temp);
	cost2 = endTime - startTime;
	cout << "gpu加速处理时间为：" << cost2 << "ms" << endl;
	cout << "加速比：" << (cost1 - cost2) / cost1 << endl;

	waitKey(0);
	return 0;
}

