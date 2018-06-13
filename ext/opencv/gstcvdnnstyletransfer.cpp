/*
 * GStreamer
 * Copyright (C) 2018 Stian Selnes <stian@pexip.com>
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

/** // FIXME
 * SECTION:element-dnnobjectdetect
 *
` * Performs object detection on videos and images using OpenCV's Deep Neural
 * Net module.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 autovideosrc ! decodebin ! colorspace ! dnnobjectdetect ! videoconvert ! xvimagesink
 * ]| Detect objects and draw bounding boxes
 *
 * </refsect2>
 */

/*
 * Add examples:
 * gst-launch-1.0 v4l2src ! videoconvert ! dnnobjectdetect model=/share/models/opencv_face_detector.caffemodel config=/share/models/opencv_face_detector.prototxt width=300 height=300 channel-order=bgr mean-red=123 mean-green=177 mean-blue=104 scale=1.0 ! videoconvert ! ximagesink sync=false
 *
 * Set channel-order=rgb
 * gst-launch-1.0 v4l2src ! videoconvert ! dnnobjectdetect model=/share/yolov3-tiny/yolov3-tiny.weights config=/share/yolov3-tiny/yolov3-tiny.cfg classes=/share/yolov3-tiny/coco.names width=416 height=416 scale=0.00392  ! videoconvert ! ximagesink sync=false
 *
 * See https://github.com/opencv/opencv/tree/master/samples/dnn
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include "gstcvdnnstyletransfer.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

struct _GstCvDnnStyleTransfer
{
  GstOpencvDnnVideoFilter element;
};

GST_DEBUG_CATEGORY_STATIC (gst_cv_dnn_style_transfer_debug);
#define GST_CAT_DEFAULT gst_cv_dnn_style_transfer_debug

enum
{
  PROP_0,
};


static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE ("RGB"))
    );

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE ("RGB"))
    );

G_DEFINE_TYPE (GstCvDnnStyleTransfer, gst_cv_dnn_style_transfer, GST_TYPE_OPENCV_DNN_VIDEO_FILTER);
#define parent_class gst_cv_dnn_style_transfer_parent_class

#if 0
static std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

static void
print_mat (Mat & mat)
{
  GST_ERROR ("Matrix dim %d, type %d (%s), depth %d, channels %d, size (%d, %d, %d)",
    mat.dims, mat.type(), type2str(mat.type()).c_str(), mat.depth(), mat.channels(),
      mat.size[0], mat.size[1], mat.size[2]);
}
#endif

static void
gst_cv_dnn_style_transfer_post_process (GstOpencvDnnVideoFilter * dnnfilter,
    vector<Mat> & outs, Mat & inframe, Mat & outframe)
{
  GstCvDnnStyleTransfer *transfer = GST_CV_DNN_STYLE_TRANSFER (dnnfilter);

  g_return_if_fail (outs.size () == 1);

  Mat &outblob = outs[0];
  g_return_if_fail (outblob.dims == 4); //  x channels x rows x cols
  g_return_if_fail (outblob.size[0] == 1); // one frame
  g_return_if_fail (outblob.size[1] == 3); // three channels (planar)

  // TODO: apparently merge() is faster
  // https://stackoverflow.com/questions/43183931

  // Convert the matrix from planar to interleaved rgb
  Mat tmp(3, outblob.size[2] * outblob.size[3], outblob.type(), outblob.data);
  tmp = tmp.t();
  tmp = tmp.reshape(3, outblob.size[2]);

  // Adjust for mean
  tmp += gst_opencv_dnn_video_filter_get_mean_values (dnnfilter); 
  tmp.convertTo(outframe, CV_8UC3);

  if (dnnfilter->channel_order == GST_OPENCV_DNN_CHANNEL_ORDER_BGR)
    cvtColor (outframe, outframe, COLOR_BGR2RGB);
}

static void
gst_cv_dnn_style_transfer_init (GstCvDnnStyleTransfer * transfer)
{
  gst_opencv_video_filter_set_in_place (GST_OPENCV_VIDEO_FILTER_CAST (transfer),
      FALSE);
}

static void
gst_cv_dnn_style_transfer_class_init (GstCvDnnStyleTransferClass * klass)
{
  GstElementClass *element_class;
  GstOpencvDnnVideoFilterClass *opencv_dnnfilter_class;

  element_class = GST_ELEMENT_CLASS (klass);
  opencv_dnnfilter_class = GST_OPENCV_DNN_VIDEO_FILTER_CLASS (klass);

  opencv_dnnfilter_class->post_process =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_style_transfer_post_process);

  gst_element_class_set_static_metadata (element_class,
      "cvdnnstyletransfer",
      "Filter/Effect/Video",
      "Performs DNN style transfer on videos and images",
      "Stian Selnes <stian@pexip.com.com>");

  gst_element_class_add_static_pad_template (element_class, &src_template);
  gst_element_class_add_static_pad_template (element_class, &sink_template);
}

gboolean
gst_cv_dnn_style_transfer_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_cv_dnn_style_transfer_debug, "cvdnnstyletransfer",
      0, "Style transfer using OpenCV's DNN module");

  return gst_element_register (plugin, "cvdnnstyletransfer", GST_RANK_NONE,
      GST_TYPE_CV_DNN_STYLE_TRANSFER);
}
