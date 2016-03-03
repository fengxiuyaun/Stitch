/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
*/

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>
#include <stdbool.h>


//窗口名字符串
#define IMG1 "图1"
#define IMG2 "图2"
#define IMG1_FEAT "图1特征点"
#define IMG2_FEAT "图2特征点"
#define IMG_MATCH1 "距离比值筛选后的匹配结果"
#define IMG_MATCH2 "RANSAC筛选后的匹配结果"
#define IMG_MOSAIC_TEMP "临时拼接图像"
#define IMG_MOSAIC_SIMPLE "简易拼接图"
#define IMG_MOSAIC_BEFORE_FUSION "重叠区域融合前"
#define IMG_MOSAIC_PROC "处理后的拼接图"

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49


int main( int argc, char** argv )
{
	IplImage* img1, *img2;//原始图像
	IplImage *stacked, *stacked_ransac;//"距离比值筛选后的匹配结果","RANSAC筛选后的匹配结果"
	struct feature* feat1, * feat2, * feat;//两幅图像sift特征
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, k, i, m = 0;
	argc = 3;
	//  argv[1] = "../Debug/beaver.png";
	//  argv[2] = "../Debug/beaver_xform.png";
	argv[1] = "../Debug/3.jpg";
	argv[2] = "../Debug/4.jpg";
	bool isHorizontal = true;

	if( argc != 3 )
		fatal_error( "usage: %s <img1> <img2>", argv[0] );
  
	img1 = cvLoadImage( argv[1], 1 );
	if( ! img1 )
		fatal_error( "unable to load image from %s", argv[1] );
	img2 = cvLoadImage( argv[2], 1 );
	if( ! img2 )
		fatal_error( "unable to load image from %s", argv[2] );
	if (isHorizontal)
		stacked = stack_imgs_horizontal(img1, img2);//合成图像，显示经距离比值法筛选后的匹配结果
	else
		stacked = stack_imgs( img1, img2 );

	

	fprintf( stderr, "Finding features in %s...\n", argv[1] );
	n1 = sift_features( img1, &feat1 );
	fprintf( stderr, "Finding features in %s...\n", argv[2] );
	n2 = sift_features( img2, &feat2 );
	fprintf( stderr, "Building kd tree...\n" );

	kd_root = kdtree_build( feat1, n1 );
	for( i = 0; i < n2; i++ )
	{
		feat = feat2 + i;
		k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
		if( k == 2 )
		{
			d0 = descr_dist_sq( feat, nbrs[0] );
			d1 = descr_dist_sq( feat, nbrs[1] );
			if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
			{
				pt2 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
				pt1 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
				if (isHorizontal)
					pt2.x += img1->width;
				else
					pt2.y += img1->height;
				cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
				m++;
				feat2[i].fwd_match = nbrs[0];
			}
		}
		free( nbrs );
	}

	fprintf( stderr, "Found %d total matches\n", m );
	display_big_img( stacked, "Matches" );
	cvWaitKey( 0 );

  /* 
     UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
     
     Note that this line above:
     
     feat1[i].fwd_match = nbrs[0];
     
     is important for the RANSAC function to work.
  */
	{
		CvMat* H;
		IplImage* xformed, *xformed_simple, *xformed_proc;
		struct feature **inliers;//精RANSAC筛选后的内点数组
		int n_inliers;//经RANSAC算法筛选后的内点个数,即feat2中具有符合要求的特征点的个数
		H = ransac_xform( feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, &inliers, &n_inliers);
		if( H )
		{
			if (isHorizontal)
				stacked_ransac = stack_imgs_horizontal(img1, img2);//合成图像，显示经距离比值法筛选后的匹配结果
			else
				stacked_ransac = stack_imgs(img1, img2);

			int invertNum = 0;//统计pt2.x > pt1.x的匹配点对的个数，来判断img1中是否右图

			//遍历经RANSAC算法筛选后的特征点集合inliers，找到每个特征点的匹配点，画出连线
			for (int i = 0; i<n_inliers; i++)
			{
				feat = inliers[i];//第i个特征点
				pt2 = cvPoint(cvRound(feat->x), cvRound(feat->y));//图2中点的坐标
				pt1 = cvPoint(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));//图1中点的坐标(feat的匹配点)

				//统计匹配点的左右位置关系，来判断图1和图2的左右位置关系
				if (pt2.x > pt1.x)
					invertNum++;
				if (isHorizontal)
					pt2.x += img1->width;
				else
					pt2.y += img1->height;//由于两幅图是上下排列的，pt2的纵坐标加上图1的高度，作为连线的终点
				cvLine(stacked_ransac, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);//在匹配图上画出连线
			}
			display_big_img(stacked_ransac, IMG_MATCH2);
			cvWaitKey(0);

			/*程序中计算出的变换矩阵H用来将img2中的点变换为img1中的点，正常情况下img1应该是左图，img2应该是右图。
			此时img2中的点pt2和img1中的对应点pt1的x坐标的关系基本都是：pt2.x < pt1.x
			若用户打开的img1是右图，img2是左图，则img2中的点pt2和img1中的对应点pt1的x坐标的关系基本都是：pt2.x > pt1.x
			所以通过统计对应点变换前后x坐标大小关系，可以知道img1是不是右图。
			如果img1是右图，将img1中的匹配点经H的逆阵H_IVT变换后可得到img2中的匹配点*/
			//若pt2.x > pt1.x的点的个数大于内点个数的80%，则认定img1中是右图
			if (invertNum > n_inliers * 0.8)
			{
				CvMat * H_IVT = cvCreateMat(3, 3, CV_64FC1);//变换矩阵的逆矩阵
				//求H的逆阵H_IVT时，若成功求出，返回非零值
				if (cvInvert(H, H_IVT, 0))
				{
					cvReleaseMat(&H);//释放变换矩阵H，因为用不到了
					H = cvCloneMat(H_IVT);//将H的逆阵H_IVT中的数据拷贝到H中
					cvReleaseMat(&H_IVT);//释放逆阵H_IVT
					//将img1和img2对调
					IplImage * temp = img2;
					img2 = img1;
					img1 = temp;
				}
				else//H不可逆时，返回0
				{
					cvReleaseMat(&H_IVT);//释放逆阵H_IVT
				}
			}

			//图2的四个角经矩阵H变换后的坐标
			CvPoint leftTop, leftBottom, rightTop, rightBottom;
			//计算图2的四个角经矩阵H变换后的坐标
			double v2[] = { 0, 0, 1 };//左上角
			double v1[3];//变换后的坐标值
			CvMat V2 = cvMat(3, 1, CV_64FC1, v2);
			CvMat V1 = cvMat(3, 1, CV_64FC1, v1);
			cvGEMM(H, &V2, 1, 0, 1, &V1, 0);//矩阵乘法
			leftTop.x = cvRound(v1[0] / v1[2]);
			leftTop.y = cvRound(v1[1] / v1[2]);

			//将v2中数据设为左下角坐标
			v2[0] = 0;
			v2[1] = img2->height;
			V2 = cvMat(3, 1, CV_64FC1, v2);
			V1 = cvMat(3, 1, CV_64FC1, v1);
			cvGEMM(H, &V2, 1, 0, 1, &V1, 0);
			leftBottom.x = cvRound(v1[0] / v1[2]);
			leftBottom.y = cvRound(v1[1] / v1[2]);

			//将v2中数据设为右上角坐标
			v2[0] = img2->width;
			v2[1] = 0;
			V2 = cvMat(3, 1, CV_64FC1, v2);
			V1 = cvMat(3, 1, CV_64FC1, v1);
			cvGEMM(H, &V2, 1, 0, 1, &V1, 0);
			rightTop.x = cvRound(v1[0] / v1[2]);
			rightTop.y = cvRound(v1[1] / v1[2]);

			//将v2中数据设为右下角坐标
			v2[0] = img2->width;
			v2[1] = img2->height;
			V2 = cvMat(3, 1, CV_64FC1, v2);
			V1 = cvMat(3, 1, CV_64FC1, v1);
			cvGEMM(H, &V2, 1, 0, 1, &V1, 0);
			rightBottom.x = cvRound(v1[0] / v1[2]);
			rightBottom.y = cvRound(v1[1] / v1[2]);

			//为拼接结果图xformed分配空间,高度为图1图2高度的较小者，根据图2右上角和右下角变换后的点的位置决定拼接图的宽度
			xformed = cvCreateImage(cvSize(MAX(rightTop.x, rightBottom.x), MAX(img1->height, img2->height)), IPL_DEPTH_8U, 3);
			//用变换矩阵H对右图img2做投影变换(变换后会有坐标右移)，结果放到xformed中
			cvWarpPerspective(img2, xformed, H, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
			cvNamedWindow(IMG_MOSAIC_TEMP, 1); //显示临时图,即只将图2变换后的图
			cvShowImage(IMG_MOSAIC_TEMP, xformed);
			cvWaitKey(0);

			//简易拼接法：直接将将左图img1叠加到xformed的左边
			xformed_simple = cvCloneImage(xformed);//简易拼接图，可笼子xformed
			cvSetImageROI(xformed_simple, cvRect(0, 0, img1->width, img1->height));
			cvAddWeighted(img1, 1, xformed_simple, 0, 0, xformed_simple);
			cvResetImageROI(xformed_simple);
			cvNamedWindow(IMG_MOSAIC_SIMPLE, 1);//创建窗口
			cvShowImage(IMG_MOSAIC_SIMPLE, xformed_simple);//显示简易拼接图
			cvWaitKey(0);

			//处理后的拼接图，克隆自xformed
			xformed_proc = cvCloneImage(xformed);

			//重叠区域左边的部分完全取自图1
			cvSetImageROI(img1, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
			cvSetImageROI(xformed, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
			cvSetImageROI(xformed_proc, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
			cvAddWeighted(img1, 1, xformed, 0, 0, xformed_proc);
			cvResetImageROI(img1);
			cvResetImageROI(xformed);
			cvResetImageROI(xformed_proc);
			cvNamedWindow(IMG_MOSAIC_BEFORE_FUSION, 1);
			cvShowImage(IMG_MOSAIC_BEFORE_FUSION, xformed_proc);//显示融合之前的拼接图
			cvWaitKey(0);

			//采用加权平均的方法融合重叠区域
			int start = MIN(leftTop.x, leftBottom.x);//开始位置，即重叠区域的左边界
			double processWidth = img1->width - start;//重叠区域的宽度
			double alpha = 1;//img1中像素的权重
			for (int i = 0; i<xformed_proc->height; i++)//遍历行
			{
				const uchar * pixel_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));//img1中第i行数据的指针
				const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));//xformed中第i行数据的指针
				uchar * pixel_xformed_proc = ((uchar *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc中第i行数据的指针
				for (int j = start; j<img1->width; j++)//遍历重叠区域的列
				{
					//如果遇到图像xformed中无像素的黑点，则完全拷贝图1中的数据
					if (pixel_xformed[j * 3] < 50 && pixel_xformed[j * 3 + 1] < 50 && pixel_xformed[j * 3 + 2] < 50)
					{
						alpha = 1;
					}
					else
					{   //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比
						alpha = (processWidth - (j - start)) / processWidth;
					}
					pixel_xformed_proc[j * 3] = pixel_img1[j * 3] * alpha + pixel_xformed[j * 3] * (1 - alpha);//B通道
					pixel_xformed_proc[j * 3 + 1] = pixel_img1[j * 3 + 1] * alpha + pixel_xformed[j * 3 + 1] * (1 - alpha);//G通道
					pixel_xformed_proc[j * 3 + 2] = pixel_img1[j * 3 + 2] * alpha + pixel_xformed[j * 3 + 2] * (1 - alpha);//R通道
				}
			}
			cvNamedWindow(IMG_MOSAIC_PROC, 1);//创建窗口
			cvShowImage(IMG_MOSAIC_PROC, xformed_proc);//显示处理后的拼接图
			cvWaitKey(0);

			//xformed = cvCreateImage( cvGetSize( img1 ), IPL_DEPTH_8U, 3 );
			//cvWarpPerspective( img2, xformed, H, 
			//		   CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
			//		   cvScalarAll( 0 ) );
			//cvNamedWindow( "Xformed", 1 );
			//cvShowImage( "Xformed", xformed );
			//cvWaitKey( 0 );
			cvReleaseImage( &xformed );
			cvReleaseMat( &H );
		}
	}
 

	cvReleaseImage( &stacked );
	cvReleaseImage( &img1 );
	cvReleaseImage( &img2 );
	kdtree_release( kd_root );
	free( feat1 );
	free( feat2 );
	return 0;
}
