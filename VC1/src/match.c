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


//�������ַ���
#define IMG1 "ͼ1"
#define IMG2 "ͼ2"
#define IMG1_FEAT "ͼ1������"
#define IMG2_FEAT "ͼ2������"
#define IMG_MATCH1 "�����ֵɸѡ���ƥ����"
#define IMG_MATCH2 "RANSACɸѡ���ƥ����"
#define IMG_MOSAIC_TEMP "��ʱƴ��ͼ��"
#define IMG_MOSAIC_SIMPLE "����ƴ��ͼ"
#define IMG_MOSAIC_BEFORE_FUSION "�ص������ں�ǰ"
#define IMG_MOSAIC_PROC "������ƴ��ͼ"

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49


int main( int argc, char** argv )
{
	IplImage* img1, *img2;//ԭʼͼ��
	IplImage *stacked, *stacked_ransac;//"�����ֵɸѡ���ƥ����","RANSACɸѡ���ƥ����"
	struct feature* feat1, * feat2, * feat;//����ͼ��sift����
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
		stacked = stack_imgs_horizontal(img1, img2);//�ϳ�ͼ����ʾ�������ֵ��ɸѡ���ƥ����
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
		struct feature **inliers;//��RANSACɸѡ����ڵ�����
		int n_inliers;//��RANSAC�㷨ɸѡ����ڵ����,��feat2�о��з���Ҫ���������ĸ���
		H = ransac_xform( feat2, n2, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, &inliers, &n_inliers);
		if( H )
		{
			if (isHorizontal)
				stacked_ransac = stack_imgs_horizontal(img1, img2);//�ϳ�ͼ����ʾ�������ֵ��ɸѡ���ƥ����
			else
				stacked_ransac = stack_imgs(img1, img2);

			int invertNum = 0;//ͳ��pt2.x > pt1.x��ƥ���Եĸ��������ж�img1���Ƿ���ͼ

			//������RANSAC�㷨ɸѡ��������㼯��inliers���ҵ�ÿ���������ƥ��㣬��������
			for (int i = 0; i<n_inliers; i++)
			{
				feat = inliers[i];//��i��������
				pt2 = cvPoint(cvRound(feat->x), cvRound(feat->y));//ͼ2�е������
				pt1 = cvPoint(cvRound(feat->fwd_match->x), cvRound(feat->fwd_match->y));//ͼ1�е������(feat��ƥ���)

				//ͳ��ƥ��������λ�ù�ϵ�����ж�ͼ1��ͼ2������λ�ù�ϵ
				if (pt2.x > pt1.x)
					invertNum++;
				if (isHorizontal)
					pt2.x += img1->width;
				else
					pt2.y += img1->height;//��������ͼ���������еģ�pt2�����������ͼ1�ĸ߶ȣ���Ϊ���ߵ��յ�
				cvLine(stacked_ransac, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);//��ƥ��ͼ�ϻ�������
			}
			display_big_img(stacked_ransac, IMG_MATCH2);
			cvWaitKey(0);

			/*�����м�����ı任����H������img2�еĵ�任Ϊimg1�еĵ㣬���������img1Ӧ������ͼ��img2Ӧ������ͼ��
			��ʱimg2�еĵ�pt2��img1�еĶ�Ӧ��pt1��x����Ĺ�ϵ�������ǣ�pt2.x < pt1.x
			���û��򿪵�img1����ͼ��img2����ͼ����img2�еĵ�pt2��img1�еĶ�Ӧ��pt1��x����Ĺ�ϵ�������ǣ�pt2.x > pt1.x
			����ͨ��ͳ�ƶ�Ӧ��任ǰ��x�����С��ϵ������֪��img1�ǲ�����ͼ��
			���img1����ͼ����img1�е�ƥ��㾭H������H_IVT�任��ɵõ�img2�е�ƥ���*/
			//��pt2.x > pt1.x�ĵ�ĸ��������ڵ������80%�����϶�img1������ͼ
			if (invertNum > n_inliers * 0.8)
			{
				CvMat * H_IVT = cvCreateMat(3, 3, CV_64FC1);//�任����������
				//��H������H_IVTʱ�����ɹ���������ط���ֵ
				if (cvInvert(H, H_IVT, 0))
				{
					cvReleaseMat(&H);//�ͷű任����H����Ϊ�ò�����
					H = cvCloneMat(H_IVT);//��H������H_IVT�е����ݿ�����H��
					cvReleaseMat(&H_IVT);//�ͷ�����H_IVT
					//��img1��img2�Ե�
					IplImage * temp = img2;
					img2 = img1;
					img1 = temp;
				}
				else//H������ʱ������0
				{
					cvReleaseMat(&H_IVT);//�ͷ�����H_IVT
				}
			}

			//ͼ2���ĸ��Ǿ�����H�任�������
			CvPoint leftTop, leftBottom, rightTop, rightBottom;
			//����ͼ2���ĸ��Ǿ�����H�任�������
			double v2[] = { 0, 0, 1 };//���Ͻ�
			double v1[3];//�任�������ֵ
			CvMat V2 = cvMat(3, 1, CV_64FC1, v2);
			CvMat V1 = cvMat(3, 1, CV_64FC1, v1);
			cvGEMM(H, &V2, 1, 0, 1, &V1, 0);//����˷�
			leftTop.x = cvRound(v1[0] / v1[2]);
			leftTop.y = cvRound(v1[1] / v1[2]);

			//��v2��������Ϊ���½�����
			v2[0] = 0;
			v2[1] = img2->height;
			V2 = cvMat(3, 1, CV_64FC1, v2);
			V1 = cvMat(3, 1, CV_64FC1, v1);
			cvGEMM(H, &V2, 1, 0, 1, &V1, 0);
			leftBottom.x = cvRound(v1[0] / v1[2]);
			leftBottom.y = cvRound(v1[1] / v1[2]);

			//��v2��������Ϊ���Ͻ�����
			v2[0] = img2->width;
			v2[1] = 0;
			V2 = cvMat(3, 1, CV_64FC1, v2);
			V1 = cvMat(3, 1, CV_64FC1, v1);
			cvGEMM(H, &V2, 1, 0, 1, &V1, 0);
			rightTop.x = cvRound(v1[0] / v1[2]);
			rightTop.y = cvRound(v1[1] / v1[2]);

			//��v2��������Ϊ���½�����
			v2[0] = img2->width;
			v2[1] = img2->height;
			V2 = cvMat(3, 1, CV_64FC1, v2);
			V1 = cvMat(3, 1, CV_64FC1, v1);
			cvGEMM(H, &V2, 1, 0, 1, &V1, 0);
			rightBottom.x = cvRound(v1[0] / v1[2]);
			rightBottom.y = cvRound(v1[1] / v1[2]);

			//Ϊƴ�ӽ��ͼxformed����ռ�,�߶�Ϊͼ1ͼ2�߶ȵĽ�С�ߣ�����ͼ2���ϽǺ����½Ǳ任��ĵ��λ�þ���ƴ��ͼ�Ŀ��
			xformed = cvCreateImage(cvSize(MAX(rightTop.x, rightBottom.x), MAX(img1->height, img2->height)), IPL_DEPTH_8U, 3);
			//�ñ任����H����ͼimg2��ͶӰ�任(�任�������������)������ŵ�xformed��
			cvWarpPerspective(img2, xformed, H, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, cvScalarAll(0));
			cvNamedWindow(IMG_MOSAIC_TEMP, 1); //��ʾ��ʱͼ,��ֻ��ͼ2�任���ͼ
			cvShowImage(IMG_MOSAIC_TEMP, xformed);
			cvWaitKey(0);

			//����ƴ�ӷ���ֱ�ӽ�����ͼimg1���ӵ�xformed�����
			xformed_simple = cvCloneImage(xformed);//����ƴ��ͼ��������xformed
			cvSetImageROI(xformed_simple, cvRect(0, 0, img1->width, img1->height));
			cvAddWeighted(img1, 1, xformed_simple, 0, 0, xformed_simple);
			cvResetImageROI(xformed_simple);
			cvNamedWindow(IMG_MOSAIC_SIMPLE, 1);//��������
			cvShowImage(IMG_MOSAIC_SIMPLE, xformed_simple);//��ʾ����ƴ��ͼ
			cvWaitKey(0);

			//������ƴ��ͼ����¡��xformed
			xformed_proc = cvCloneImage(xformed);

			//�ص�������ߵĲ�����ȫȡ��ͼ1
			cvSetImageROI(img1, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
			cvSetImageROI(xformed, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
			cvSetImageROI(xformed_proc, cvRect(0, 0, MIN(leftTop.x, leftBottom.x), xformed_proc->height));
			cvAddWeighted(img1, 1, xformed, 0, 0, xformed_proc);
			cvResetImageROI(img1);
			cvResetImageROI(xformed);
			cvResetImageROI(xformed_proc);
			cvNamedWindow(IMG_MOSAIC_BEFORE_FUSION, 1);
			cvShowImage(IMG_MOSAIC_BEFORE_FUSION, xformed_proc);//��ʾ�ں�֮ǰ��ƴ��ͼ
			cvWaitKey(0);

			//���ü�Ȩƽ���ķ����ں��ص�����
			int start = MIN(leftTop.x, leftBottom.x);//��ʼλ�ã����ص��������߽�
			double processWidth = img1->width - start;//�ص�����Ŀ��
			double alpha = 1;//img1�����ص�Ȩ��
			for (int i = 0; i<xformed_proc->height; i++)//������
			{
				const uchar * pixel_img1 = ((uchar *)(img1->imageData + img1->widthStep * i));//img1�е�i�����ݵ�ָ��
				const uchar * pixel_xformed = ((uchar *)(xformed->imageData + xformed->widthStep * i));//xformed�е�i�����ݵ�ָ��
				uchar * pixel_xformed_proc = ((uchar *)(xformed_proc->imageData + xformed_proc->widthStep * i));//xformed_proc�е�i�����ݵ�ָ��
				for (int j = start; j<img1->width; j++)//�����ص��������
				{
					//�������ͼ��xformed�������صĺڵ㣬����ȫ����ͼ1�е�����
					if (pixel_xformed[j * 3] < 50 && pixel_xformed[j * 3 + 1] < 50 && pixel_xformed[j * 3 + 2] < 50)
					{
						alpha = 1;
					}
					else
					{   //img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ��������
						alpha = (processWidth - (j - start)) / processWidth;
					}
					pixel_xformed_proc[j * 3] = pixel_img1[j * 3] * alpha + pixel_xformed[j * 3] * (1 - alpha);//Bͨ��
					pixel_xformed_proc[j * 3 + 1] = pixel_img1[j * 3 + 1] * alpha + pixel_xformed[j * 3 + 1] * (1 - alpha);//Gͨ��
					pixel_xformed_proc[j * 3 + 2] = pixel_img1[j * 3 + 2] * alpha + pixel_xformed[j * 3 + 2] * (1 - alpha);//Rͨ��
				}
			}
			cvNamedWindow(IMG_MOSAIC_PROC, 1);//��������
			cvShowImage(IMG_MOSAIC_PROC, xformed_proc);//��ʾ������ƴ��ͼ
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
