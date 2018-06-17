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

#include "gstdnnobjectdetect.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

struct _GstDnnObjectDetect
{
  GstOpencvDnnVideoFilter element;

  gdouble confidence_threshold;
  gboolean draw;
  //vector<string> classes;
};

GST_DEBUG_CATEGORY_STATIC (gst_dnn_object_detect_debug);
#define GST_CAT_DEFAULT gst_dnn_object_detect_debug

#define DEFAULT_CONFIDENCE_THRESHOLD 0.5
#define DEFAULT_DRAW TRUE

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

G_DEFINE_TYPE (GstDnnObjectDetect, gst_dnn_object_detect, GST_TYPE_OPENCV_DNN_VIDEO_FILTER);
#define parent_class gst_dnn_object_detect_parent_class


static void
gst_dnn_object_detect_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDnnObjectDetect *detect = GST_DNN_OBJECT_DETECT_CAST (object);
  (void) detect;
  switch (prop_id) {
    case PROP_CONFIDENCE_THRESHOLD:
      detect->confidence_threshold = g_value_get_double (value);
      break;
    case PROP_DRAW:
      detect->draw = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_dnn_object_detect_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDnnObjectDetect *detect = GST_DNN_OBJECT_DETECT_CAST (object);
  (void) detect;

  switch (prop_id) {
    case PROP_CONFIDENCE_THRESHOLD:
      g_value_set_double (value, detect->confidence_threshold);
      break;
    case PROP_DRAW:
      g_value_set_boolean (value, detect->draw);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
draw_bounding_box (GstDnnObjectDetect * detect, int class_id, float conf, int left,
    int top, int right, int bottom, Mat & frame)
{
  GstOpencvDnnVideoFilter *dnnfilter =
    GST_OPENCV_DNN_VIDEO_FILTER_CAST (detect);

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
      GST_WARNING_OBJECT (detect,
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

static void
gst_dnn_object_detect_post_process (GstOpencvDnnVideoFilter * dnnfilter,
    vector<Mat> & outs, Mat & frame)
{
  GstDnnObjectDetect *detect = GST_DNN_OBJECT_DETECT (dnnfilter);
  Net& net = dnnfilter->net;

  // FIXME: No static, put in object (parent?)
  static vector<int> out_layers = net.getUnconnectedOutLayers();
  static string out_layer_type = net.getLayer(out_layers[0])->type;

  if (net.getLayer(0)->outputNameToIndex("im_info") != -1) {
    // Faster-RCNN or R-FCN.
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batch_id, class_id, confidence, left, top, right, bottom]
    g_assert_cmpint (outs.size(), ==, 1);
    float* data = (float*)outs[0].data;
    for (size_t i = 0; i < outs[0].total(); i += 7) {
      float confidence = data[i + 2];
      if (confidence > detect->confidence_threshold) {
        int left = (int)data[i + 3];
        int top = (int)data[i + 4];
        int right = (int)data[i + 5];
        int bottom = (int)data[i + 6];
        int class_id = (int)(data[i + 1]) - 1;  // Skip 0th background class id.
        draw_bounding_box (detect, class_id, confidence, left, top, right, bottom, frame);
      }
    }
  }
  else if (out_layer_type == "DetectionOutput") {
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batch_id, class_id, confidence, left, top, right, bottom]
    g_assert_cmpint (outs.size(), ==, 1);
    float* data = (float*)outs[0].data;
    for (size_t i = 0; i < outs[0].total(); i += 7) {
      float confidence = data[i + 2];
      if (confidence > detect->confidence_threshold) {
        int left = (int)(data[i + 3] * frame.cols);
        int top = (int)(data[i + 4] * frame.rows);
        int right = (int)(data[i + 5] * frame.cols);
        int bottom = (int)(data[i + 6] * frame.rows);
        int class_id = (int)(data[i + 1]) - 1;  // Skip 0th background class id.
        draw_bounding_box (detect, class_id, confidence, left, top, right, bottom, frame);
      }
    }
  } else if (out_layer_type == "Region") {
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i)
    {
      // Network produces output blob with a shape NxC where N is a number of
      // detected objects and C is a number of classes + 4 where the first 4
      // numbers are [center_x, center_y, width, height]
      float* data = (float*)outs[i].data;
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
      {
        Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        Point class_id_point;
        double confidence;
        minMaxLoc (scores, 0, &confidence, 0, &class_id_point);
        if (confidence > detect->confidence_threshold)
        {
          int centerX = (int)(data[0] * frame.cols);
          int centerY = (int)(data[1] * frame.rows);
          int width = (int)(data[2] * frame.cols);
          int height = (int)(data[3] * frame.rows);
          int left = centerX - width / 2;
          int top = centerY - height / 2;

          class_ids.push_back (class_id_point.x);
          confidences.push_back ((float) confidence);
          boxes.push_back (Rect (left, top, width, height));
        }
      }
    }
    vector<int> indices;
    NMSBoxes (boxes, confidences, detect->confidence_threshold, 0.4f, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
      int idx = indices[i];
      Rect box = boxes[idx];
      draw_bounding_box (detect, class_ids[idx], confidences[idx], box.x, box.y,
          box.x + box.width, box.y + box.height, frame);
    }
  } else {
    GST_ERROR_OBJECT (detect, "Unknown output layer type: %s", out_layer_type.c_str());
  }
}

#if 0
static void
load_model (GstDnnObjectDetect * detect)
{
  // fixme: leak?
  detect->net = readNet (detect->model_fn, detect->config_fn, detect->framework);
  // FIXME
  detect->net.setPreferableBackend(0);
  detect->net.setPreferableTarget(0);

  /* Parse and store classes */
  detect->classes.clear();
  if (detect->classes_fn != NULL) {
    ifstream ifs (detect->classes_fn);
    if (ifs.is_open ()) {
      string line;
      while (getline(ifs, line))
        detect->classes.push_back (line);
    } else {
      GST_ERROR_OBJECT (detect, "Could not open '%s'", detect->classes_fn);
    }
  }
}

static void
clear_model (GstDnnObjectDetect * detect)
{
  (void) detect;
}

static GstStateChangeReturn
gst_dnn_object_detect_change_state (GstElement * element,
    GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstDnnObjectDetect *detect = GST_DNN_OBJECT_DETECT_CAST (element);

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      load_model (detect);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      clear_model (detect);
      break;
    default:
      break;
  }
  return ret;
}
#endif


static void
gst_dnn_object_detect_dispose (GObject * obj)
{
  G_OBJECT_CLASS (parent_class)->dispose (obj);
}

static void
gst_dnn_object_detect_init (GstDnnObjectDetect * detect)
{
  gst_opencv_video_filter_set_in_place (GST_OPENCV_VIDEO_FILTER_CAST (detect),
      TRUE);
}

static void
gst_dnn_object_detect_class_init (GstDnnObjectDetectClass * klass)
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
    GST_DEBUG_FUNCPTR (gst_dnn_object_detect_dispose);
  gobject_class->set_property =
    GST_DEBUG_FUNCPTR (gst_dnn_object_detect_set_property);
  gobject_class->get_property =
    GST_DEBUG_FUNCPTR (gst_dnn_object_detect_get_property);

  opencv_dnnfilter_class->post_process_ip =
    GST_DEBUG_FUNCPTR (gst_dnn_object_detect_post_process);

  // element_class->change_state =
  //   GST_DEBUG_FUNCPTR (gst_dnn_object_detect_change_state);

  // opencv_filter_class->cv_trans_ip_func =
  //   GST_DEBUG_FUNCPTR (gst_dnn_object_detect_transform_ip);

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
      "dnnobjectdetect",
      "Filter/Effect/Video",
      "Performs DNN object detection on videos and images",
      "Stian Selnes <stian@pexip.com.com>");

  gst_element_class_add_static_pad_template (element_class, &src_template);
  gst_element_class_add_static_pad_template (element_class, &sink_template);
}

gboolean
gst_dnn_object_detect_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dnn_object_detect_debug, "dnnobjectdetect",
      0, "Object detection using OpenCV's DNN module");

  return gst_element_register (plugin, "dnnobjectdetect", GST_RANK_NONE,
      GST_TYPE_DNN_OBJECT_DETECT);
}
