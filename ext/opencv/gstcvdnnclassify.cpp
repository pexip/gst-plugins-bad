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

/**
 * SECTION:element-cvdnnclassify
 *
` * Performs classification on videos and images using OpenCV's Deep Neural
 * Net module.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 autovideosrc ! decodebin ! colorspace ! cvdnnclassify ! videoconvert ! xvimagesink
 * ]| Classification on images and draw labels
 *
 * </refsect2>
 */

/*
 * Add examples:
 * gst-launch-1.0 v4l2src ! videoconvert ! cvdnnclassify model=/share/models/opencv_face_classifyor.caffemodel config=/share/models/opencv_face_classifyor.prototxt width=300 height=300 channel-order=bgr mean-red=123 mean-green=177 mean-blue=104 scale=1.0 ! videoconvert ! ximagesink sync=false
 *
 * Set channel-order=rgb
 * gst-launch-1.0 v4l2src ! videoconvert ! cvdnnclassify model=/share/yolov3-tiny/yolov3-tiny.weights config=/share/yolov3-tiny/yolov3-tiny.cfg classes=/share/yolov3-tiny/coco.names width=416 height=416 scale=0.00392  ! videoconvert ! ximagesink sync=false
 *
 * See https://github.com/opencv/opencv/tree/master/samples/dnn
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include "gstcvdnnclassify.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

struct _GstCvDnnClassify
{
  GstOpencvDnnVideoFilter element;

  gboolean draw;
};

GST_DEBUG_CATEGORY_STATIC (gst_cv_dnn_classify_debug);
#define GST_CAT_DEFAULT gst_cv_dnn_classify_debug

#define DEFAULT_DRAW TRUE

enum
{
  PROP_0,
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

G_DEFINE_TYPE (GstCvDnnClassify, gst_cv_dnn_classify, GST_TYPE_OPENCV_DNN_VIDEO_FILTER);
#define parent_class gst_cv_dnn_classify_parent_class


static void
gst_cv_dnn_classify_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCvDnnClassify *classify = GST_CV_DNN_CLASSIFY_CAST (object);
  (void) classify;
  switch (prop_id) {
    case PROP_DRAW:
      classify->draw = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_cv_dnn_classify_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstCvDnnClassify *classify = GST_CV_DNN_CLASSIFY_CAST (object);
  (void) classify;

  switch (prop_id) {
    case PROP_DRAW:
      g_value_set_boolean (value, classify->draw);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_cv_dnn_classify_post_process (GstOpencvDnnVideoFilter * dnnfilter,
    vector<Mat> & outs, Mat & frame)
{
  GstCvDnnClassify *classify = GST_CV_DNN_CLASSIFY (dnnfilter);
  Net& net = dnnfilter->net;
  Mat& prob = outs[0];

  Point classIdPoint;
  double confidence;

  // Get class with highest score
  minMaxLoc (prob.reshape (1, 1), 0, &confidence, 0, &classIdPoint);
  int classId = classIdPoint.x;

  if (classify->draw) {
    const gchar * class_str = dnnfilter->classes.empty () ?
      format ("Class #%d", classId).c_str () : dnnfilter->classes[classId].c_str ();
    string label = format ("%.4f %s", confidence, class_str);

    // Draw half transparent background with label on top
    int base_line;
    Size text_size = getTextSize ("Xg", FONT_HERSHEY_DUPLEX, 0.5, 1,
        &base_line);
    int bg_width = frame.cols;
    int bg_height = text_size.height + base_line;
    Mat roi (frame, Rect(0, frame.rows - bg_height, bg_width, bg_height));
    Mat bg (roi.size(), CV_8UC3, Scalar::all(0));
    double alpha = 0.4;
    addWeighted (bg, alpha, roi, 1 - alpha, 0, roi);
    // FIXME: May clip the text
    putText (frame, label, Point(0, frame.rows - base_line), FONT_HERSHEY_DUPLEX, 0.5, Scalar::all(255), 1, LINE_AA);
  }
}

static void
gst_cv_dnn_classify_dispose (GObject * obj)
{
  G_OBJECT_CLASS (parent_class)->dispose (obj);
}

static void
gst_cv_dnn_classify_init (GstCvDnnClassify * classify)
{
  gst_opencv_video_filter_set_in_place (GST_OPENCV_VIDEO_FILTER_CAST (classify),
      TRUE);
}

static void
gst_cv_dnn_classify_class_init (GstCvDnnClassifyClass * klass)
{
  GObjectClass *gclass;
  GstElementClass *element_class;
  //GstOpencvVideoFilterClass *opencv_filter_class;
  GstOpencvDnnVideoFilterClass *opencv_dnnfilter_class;

  element_class = GST_ELEMENT_CLASS (klass);
  gclass = G_OBJECT_CLASS (klass);
/*   opencv_filter_class = GST_OPENCV_VIDEO_FILTER_CLASS (klass); */
  opencv_dnnfilter_class = GST_OPENCV_DNN_VIDEO_FILTER_CLASS (klass);

  gclass->dispose =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_classify_dispose);
  gclass->set_property =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_classify_set_property);
  gclass->get_property =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_classify_get_property);

  opencv_dnnfilter_class->post_process_ip =
    GST_DEBUG_FUNCPTR (gst_cv_dnn_classify_post_process);

  // element_class->change_state =
  //   GST_DEBUG_FUNCPTR (gst_cv_dnn_classify_change_state);

  // opencv_filter_class->cv_trans_ip_func =
  //   GST_DEBUG_FUNCPTR (gst_cv_dnn_classify_transform_ip);

  // g_object_class_install_property (gclass, PROP_CONFIDENCE_THRESHOLD,
  //     g_param_spec_double ("confidence-threshold", "Confidence threshold",
  //         "Confidence threshold for deciding there is an object",
  //         -G_MAXDOUBLE, G_MAXDOUBLE, DEFAULT_CONFIDENCE_THRESHOLD,
  //         (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gclass, PROP_DRAW,
      g_param_spec_boolean ("draw", "Draw",
          "Whether to draw labels",
          DEFAULT_DRAW,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  gst_element_class_set_static_metadata (element_class,
      "dnnclassify",
      "Filter/Effect/Video",
      "Performs DNN classification on videos and images",
      "Stian Selnes <stian@pexip.com.com>");

  gst_element_class_add_static_pad_template (element_class, &src_template);
  gst_element_class_add_static_pad_template (element_class, &sink_template);
}

gboolean
gst_cv_dnn_classify_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_cv_dnn_classify_debug, "cvdnnclassify",
      0, "Classification using OpenCV's DNN module");

  return gst_element_register (plugin, "cvdnnclassify", GST_RANK_NONE,
      GST_TYPE_CV_DNN_CLASSIFY);
}
