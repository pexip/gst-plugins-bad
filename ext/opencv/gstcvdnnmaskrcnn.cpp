/*
 * GStreamer
 * Copyright (C) 2019 Stian Selnes <stian@pexip.com>
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

/**
 * SECTION:element-cvdnnmaskrcnn
 *
 ` Performs masking on detected objects on videos and images using OpenCV's Deep Neural
 * Net module.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 autovideosrc ! decodebin ! colorspace ! cvdnnmaskrcnn ! videoconvert ! xvimagesink
 * ]| Detect objects and draw bounding boxes
 *
 * </refsect2>
 */

/*
 * Add examples:
 * gst-launch-1.0 v4l2src ! videoconvert ! cvdnnmaskrcnn model=/share/models/opencv_face_detector.caffemodel config=/share/models/opencv_face_detector.prototxt width=300 height=300 channel-order=bgr mean-red=123 mean-green=177 mean-blue=104 scale=1.0 ! videoconvert ! ximagesink sync=false
 *
 * Set channel-order=rgb
 * gst-launch-1.0 v4l2src ! videoconvert ! cvdnnmaskrcnn model=/share/yolov3-tiny/yolov3-tiny.weights config=/share/yolov3-tiny/yolov3-tiny.cfg classes=/share/yolov3-tiny/coco.names width=416 height=416 scale=0.00392  ! videoconvert ! ximagesink sync=false
 *
 * See https://github.com/opencv/opencv/tree/master/samples/dnn
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include "gstcvdnnmaskrcnn.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

struct _GstCvDnnMaskRcnn
{
  GstOpencvDnnVideoFilter element;

  gdouble confidence_threshold;
  gboolean draw;
  //vector<string> classes;
};

GST_DEBUG_CATEGORY_STATIC (gst_cv_dnn_mask_rcnn_debug);
#define GST_CAT_DEFAULT gst_cv_dnn_mask_rcnn_debug

#define DEFAULT_CONFIDENCE_THRESHOLD 0.5
#define DEFAULT_DRAW TRUE

const float maskThreshold = 0.3;

enum
{
  PROP_0,
  PROP_CONFIDENCE_THRESHOLD,
  PROP_DRAW
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

G_DEFINE_TYPE (GstCvDnnMaskRcnn, gst_cv_dnn_mask_rcnn, GST_TYPE_OPENCV_DNN_VIDEO_FILTER);
#define parent_class gst_cv_dnn_mask_rcnn_parent_class


static void
gst_cv_dnn_mask_rcnn_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCvDnnMaskRcnn *self = GST_CV_DNN_MASK_RCNN_CAST (object);
  (void) self;
  switch (prop_id) {
    case PROP_CONFIDENCE_THRESHOLD:
      self->confidence_threshold = g_value_get_double (value);
      break;
    case PROP_DRAW:
      self->draw = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_cv_dnn_mask_rcnn_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstCvDnnMaskRcnn *self = GST_CV_DNN_MASK_RCNN_CAST (object);
  (void) self;

  switch (prop_id) {
    case PROP_CONFIDENCE_THRESHOLD:
      g_value_set_double (value, self->confidence_threshold);
      break;
    case PROP_DRAW:
      g_value_set_boolean (value, self->draw);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
draw_bounding_box (GstCvDnnMaskRcnn * self, int class_id, float conf, int left,
    int top, int right, int bottom, Mat & frame)
{
  GstOpencvDnnVideoFilter *dnnfilter =
    GST_OPENCV_DNN_VIDEO_FILTER_CAST (self);

  /*
    import seaborn as sns
    for c in sns.hls_palette(8, l=.7, s=1.).as_hex():
    print("Scalar (0x{}, 0x{}, 0x{}),".format(c[1:3], c[3:5], c[5:]))
  */
  static const Scalar colormap[] = {
    Scalar(0xff, 0x6f, 0x66),
    Scalar(0xff, 0xe2, 0x66),
    Scalar(0xa9, 0xff, 0x66),
    Scalar(0x66, 0xff, 0x95),
    Scalar(0x66, 0xf6, 0xff),
    Scalar(0x66, 0x83, 0xff),
    Scalar(0xbc, 0x66, 0xff),
    Scalar(0xff, 0x66, 0xd0),
  };
  static const int colormap_len = sizeof(colormap) / sizeof (*colormap);
  Scalar color = colormap[class_id % colormap_len];

  /* Draw bounding box */
  rectangle(frame, Point(left, top), Point(right, bottom), color, 2);

  /* Draw confidence and class name */
  string label = format("%.2f", conf);
  if (!dnnfilter->classes.empty()) {
    if (class_id < (int) dnnfilter->classes.size()) {
      label = dnnfilter->classes[class_id] + ": " + label;
    } else {
      GST_WARNING_OBJECT (self,
          "class_id %d exceeds number of known classes %d", class_id,
          (int) dnnfilter->classes.size());
    }
  }
  /* Put label above box (if possible) */
  int base_line;
  Size label_size = getTextSize (label, FONT_HERSHEY_DUPLEX, 0.5, 1,
      &base_line);
  int text_top = max(0, top - label_size.height - base_line);
  rectangle (frame, Point(left, text_top), Point (left + label_size.width, top), color, FILLED);
  putText (frame, label, Point(left, top - base_line), FONT_HERSHEY_DUPLEX, 0.5, Scalar::all(30), 1, LINE_AA);
}

// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(GstCvDnnMaskRcnn * self, Mat& frame, int classId, float conf,
    Rect box, Mat& objectMask)
{
  GstOpencvDnnVideoFilter *dnnfilter = GST_OPENCV_DNN_VIDEO_FILTER_CAST (self);

  //Draw a rectangle displaying the bounding box
  rectangle(frame, Point(box.x, box.y),
      Point(box.x+box.width, box.y+box.height), Scalar(255, 178, 50), 3);

  //Get the label for the class name and its confidence
  string label = format("%.2f", conf);
  if (!dnnfilter->classes.empty())
  {
    CV_Assert(classId < (int)dnnfilter->classes.size());
    label = dnnfilter->classes[classId] + ":" + label;
  }

  //Display the label at the top of the bounding box
  // int baseLine;
  // Size labelSize = getTextSize(label, FONT_HERSHEY_DUPLEX, 0.5, 1, &baseLine);
  // box.y = max(box.y, labelSize.height);
  // rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)),
  //     Point(box.x + round(1.5*labelSize.width), box.y + baseLine),
  //     Scalar(255, 255, 255), FILLED);
  // putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_DUPLEX, 0.75,
  //     Scalar(0,0,0),1);

  /*
    import seaborn as sns
    for c in sns.hls_palette(8, l=.7, s=1.).as_hex():
    print("Scalar (0x{}, 0x{}, 0x{}),".format(c[1:3], c[3:5], c[5:]))
  */
  static const Scalar colormap[] = {
    Scalar(0xff, 0x6f, 0x66),
    Scalar(0xff, 0xe2, 0x66),
    Scalar(0xa9, 0xff, 0x66),
    Scalar(0x66, 0xff, 0x95),
    Scalar(0x66, 0xf6, 0xff),
    Scalar(0x66, 0x83, 0xff),
    Scalar(0xbc, 0x66, 0xff),
    Scalar(0xff, 0x66, 0xd0),
  };
  static const int colormap_len = sizeof(colormap) / sizeof (*colormap);
  Scalar color = colormap[classId % colormap_len];


  // Resize the mask, threshold, color and apply it on the image
  resize(objectMask, objectMask, Size(box.width, box.height));
  Mat mask = (objectMask > maskThreshold);
  Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
  coloredRoi.convertTo(coloredRoi, CV_8UC3);

  // Draw the contours on the image
  vector<Mat> contours;
  Mat hierarchy;
  mask.convertTo(mask, CV_8U);
  findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
  drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
  coloredRoi.copyTo(frame(box), mask);
}


static void
gst_cv_dnn_mask_rcnn_post_process (GstOpencvDnnVideoFilter * dnnfilter,
    vector<Mat> & outs, Mat & frame)
{
  GstCvDnnMaskRcnn *self = GST_CV_DNN_MASK_RCNN (dnnfilter);
  Net& net = dnnfilter->net;

  g_assert_cmpint (outs.size(), ==, 2);
  Mat outDetections = outs[0];
  Mat outMasks = outs[1];

  // Output size of masks is NxCxHxW where
  // N - number of detected boxes
  // C - number of classes (excluding background)
  // HxW - segmentation shape
  const int numDetections = outDetections.size[2];
  const int numClasses = outMasks.size[1];

  outDetections = outDetections.reshape(1, outDetections.total() / 7);
  for (int i = 0; i < numDetections; ++i)
  {
    float score = outDetections.at<float>(i, 2);
    if (score > self->confidence_threshold)
    {
      // Extract the bounding box
      int classId = static_cast<int>(outDetections.at<float>(i, 1));
      int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
      int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
      int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
      int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

      left = max(0, min(left, frame.cols - 1));
      top = max(0, min(top, frame.rows - 1));
      right = max(0, min(right, frame.cols - 1));
      bottom = max(0, min(bottom, frame.rows - 1));
      Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

      // Extract the mask for the object
      Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i,classId));

      // Draw bounding box, colorize and show the mask on the image
      drawBox(self, frame, classId, score, box, objectMask);
    }
  }
}

static void
gst_cv_dnn_mask_rcnn_dispose (GObject * obj)
{
  G_OBJECT_CLASS (parent_class)->dispose (obj);
}

static void
gst_cv_dnn_mask_rcnn_init (GstCvDnnMaskRcnn * self)
{
  gst_opencv_video_filter_set_in_place (GST_OPENCV_VIDEO_FILTER_CAST (self),
      TRUE);
}

static void
gst_cv_dnn_mask_rcnn_class_init (GstCvDnnMaskRcnnClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  //GstOpencvVideoFilterClass *opencv_filter_class;
  GstOpencvDnnVideoFilterClass *opencv_dnnfilter_class;

  element_class = GST_ELEMENT_CLASS (klass);
  gobject_class = G_OBJECT_CLASS (klass);
/*   opencv_filter_class = GST_OPENCV_VIDEO_FILTER_CLASS (klass); */
  opencv_dnnfilter_class = GST_OPENCV_DNN_VIDEO_FILTER_CLASS (klass);

  gobject_class->dispose =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_mask_rcnn_dispose);
  gobject_class->set_property =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_mask_rcnn_set_property);
  gobject_class->get_property =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_mask_rcnn_get_property);

  opencv_dnnfilter_class->post_process_ip =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_mask_rcnn_post_process);

  // element_class->change_state =
  //   GST_DEBUG_FUNCPTR (gst_cv_dnn_mask_rcnn_change_state);

  // opencv_filter_class->cv_trans_ip_func =
  //   GST_DEBUG_FUNCPTR (gst_cv_dnn_mask_rcnn_transform_ip);

  g_object_class_install_property (gobject_class, PROP_CONFIDENCE_THRESHOLD,
      g_param_spec_double ("confidence-threshold", "Confidence threshold",
          "Confidence threshold for deciding there is an object",
          -G_MAXDOUBLE, G_MAXDOUBLE, DEFAULT_CONFIDENCE_THRESHOLD,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_DRAW,
      g_param_spec_boolean ("draw", "Draw",
          "Whether to draw bounding boxes and labels",
          DEFAULT_DRAW,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  gst_element_class_set_static_metadata (element_class,
      "cvcvdnnmaskrcnn",
      "Filter/Effect/Video",
      "Performs DNN masking of objects on videos and images",
      "Stian Selnes <stian@pexip.com.com>");

  gst_element_class_add_static_pad_template (element_class, &src_template);
  gst_element_class_add_static_pad_template (element_class, &sink_template);
}

gboolean
gst_cv_dnn_mask_rcnn_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_cv_dnn_mask_rcnn_debug, "cvdnnmaskrcnn",
      0, "Mask objects using OpenCV's DNN module");

  return gst_element_register (plugin, "cvdnnmaskrcnn", GST_RANK_NONE,
      GST_TYPE_CV_DNN_MASK_RCNN);
}
