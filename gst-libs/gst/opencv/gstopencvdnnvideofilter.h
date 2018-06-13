/*
 * GStreamer
 * Copyright (C) 2010 Thiago Santos <thiago.sousa.santos@collabora.co.uk>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef __GST_OPENCV_DNN_VIDEO_FILTER_H__
#define __GST_OPENCV_DNN_VIDEO_FILTER_H__

#include "gstopencvvideofilter.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

G_BEGIN_DECLS


/* #defines don't like whitespacey bits */
#define GST_TYPE_OPENCV_DNN_VIDEO_FILTER \
  (gst_opencv_dnn_video_filter_get_type())
#define GST_OPENCV_DNN_VIDEO_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_OPENCV_DNN_VIDEO_FILTER,GstOpencvDnnVideoFilter))
#define GST_OPENCV_DNN_VIDEO_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_OPENCV_DNN_VIDEO_FILTER,GstOpencvDnnVideoFilterClass))
#define GST_IS_OPENCV_DNN_VIDEO_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_OPENCV_DNN_VIDEO_FILTER))
#define GST_IS_OPENCV_DNN_VIDEO_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_OPENCV_DNN_VIDEO_FILTER))
#define GST_OPENCV_DNN_VIDEO_FILTER_GET_CLASS(obj) \
  (G_TYPE_INSTANCE_GET_CLASS((obj),GST_TYPE_OPENCV_DNN_VIDEO_FILTER,GstOpencvDnnVideoFilterClass))
#define GST_OPENCV_DNN_VIDEO_FILTER_CAST(obj) ((GstOpencvDnnVideoFilter *) (obj))

typedef struct _GstOpencvDnnVideoFilter GstOpencvDnnVideoFilter;
typedef struct _GstOpencvDnnVideoFilterClass GstOpencvDnnVideoFilterClass;

typedef void (*GstOpencvDnnVideoFilterPostProcessFunc)
    (GstOpencvDnnVideoFilter * dnn,  std::vector<cv::Mat> & output_blobs,
    cv::Mat & inframe, cv::Mat & outframe);

typedef void (*GstOpencvDnnVideoFilterPostProcessIPFunc)
    (GstOpencvDnnVideoFilter * dnn, std::vector<cv::Mat> & output_blobs, cv::Mat & frame);

#define GST_TYPE_OPENCV_DNN_CHANNEL_ORDER (gst_opencv_dnn_channel_order_get_type ())

typedef enum _GstOpencvDnnChannelOrder {
  GST_OPENCV_DNN_CHANNEL_ORDER_RGB,
  GST_OPENCV_DNN_CHANNEL_ORDER_BGR,
} GstOpencvDnnChannelOrder;


struct _GstOpencvDnnVideoFilter
{
  GstOpencvVideoFilter filter;

  gchar *model_fn;
  gchar *config_fn;
  gchar *framework;
  gchar *classes_fn;
  gint width;
  gint height;
  GstOpencvDnnChannelOrder channel_order;
  gdouble mean_red;
  gdouble mean_green;
  gdouble mean_blue;
  gdouble scale;

  /* backend */
  /* target */
  /* rgb */

  cv::dnn::Net net;
  std::vector<std::string> classes;
};


struct _GstOpencvDnnVideoFilterClass
{
  GstOpencvVideoFilterClass parent_class;

  GstOpencvDnnVideoFilterPostProcessFunc post_process;
  GstOpencvDnnVideoFilterPostProcessIPFunc post_process_ip;  
};

GST_OPENCV_API
GType gst_opencv_dnn_video_filter_get_type (void);

GST_OPENCV_API
GType gst_opencv_dnn_channel_order_get_type (void);

GST_OPENCV_API
cv::Scalar gst_opencv_dnn_video_filter_get_mean_values (GstOpencvDnnVideoFilter * dnn);

G_END_DECLS

#endif /* __GST_OPENCV_DNN_VIDEO_FILTER_H__ */
