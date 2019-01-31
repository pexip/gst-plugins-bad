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

#include "gstopencvdnnvideofilter.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

GType
gst_opencv_dnn_channel_order_get_type (void)
{
  static GType dnn_channel_order_type = 0;
  static const GEnumValue dnn_channel_order[] = {
    {GST_OPENCV_DNN_CHANNEL_ORDER_RGB, "Model takes RGB input", "rgb"},
    {GST_OPENCV_DNN_CHANNEL_ORDER_BGR, "Model takes BGR input", "bgr"},
    {0, NULL, NULL},
  };

  if (!dnn_channel_order_type) {
    dnn_channel_order_type =
        g_enum_register_static ("GstOpencvDnnChannelOrder", dnn_channel_order);
  }
  return dnn_channel_order_type;
}

GType
gst_opencv_dnn_backend_get_type (void)
{
  static GType dnn_backend_type = 0;
  static const GEnumValue dnn_backend[] = {
    { DNN_BACKEND_DEFAULT, "Default C++ backend", "default" },
    { DNN_BACKEND_HALIDE, "Halide language", "halide" },
    { DNN_BACKEND_INFERENCE_ENGINE, "Intel's Deep Learning Inference Engine", "inference-engine" },
    { 0, NULL, NULL },
  };

  if (!dnn_backend_type) {
    dnn_backend_type =
        g_enum_register_static ("GstOpencvDnnBackend", dnn_backend);
  }
  return dnn_backend_type;
}

GType
gst_opencv_dnn_target_get_type (void)
{
  static GType dnn_target_type = 0;
  static const GEnumValue dnn_target[] = {
    { DNN_TARGET_CPU, "CPU", "cpu" },
    { DNN_TARGET_OPENCL, "OpenCL", "opencl" },
    { DNN_TARGET_OPENCL_FP16, "OpenCL FP16", "opencl-fp16" },
    { DNN_TARGET_MYRIAD, "Myriad", "myriad" }, // FIXME: What's this? When is it available
    { 0, NULL, NULL },
  };

  if (!dnn_target_type) {
    dnn_target_type =
        g_enum_register_static ("GstOpencvDnnTarget", dnn_target);
  }
  return dnn_target_type;
}

GST_DEBUG_CATEGORY_STATIC (gst_opencv_dnn_video_filter_debug);
#define GST_CAT_DEFAULT gst_opencv_dnn_video_filter_debug

#define DEFAULT_MODEL NULL
#define DEFAULT_CONFIG NULL
#define DEFAULT_FRAMEWORK NULL
#define DEFAULT_CLASSES NULL
#define DEFAULT_WIDTH (-1)
#define DEFAULT_HEIGHT (-1)
#define DEFAULT_MEAN_RED 0.0
#define DEFAULT_MEAN_GREEN 0.0
#define DEFAULT_MEAN_BLUE 0.0
#define DEFAULT_SCALE 1.0
#define DEFAULT_BACKEND DNN_BACKEND_DEFAULT
#define DEFAULT_TARGET DNN_TARGET_CPU

enum
{
  PROP_0,
  PROP_MODEL,
  PROP_CONFIG,
  PROP_FRAMEWORK,
  PROP_CLASSES,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_CHANNEL_ORDER,
  PROP_MEAN_RED,
  PROP_MEAN_GREEN,
  PROP_MEAN_BLUE,
  PROP_SCALE,
  PROP_BACKEND,
  PROP_TARGET,
};


G_DEFINE_ABSTRACT_TYPE (GstOpencvDnnVideoFilter, gst_opencv_dnn_video_filter, GST_TYPE_OPENCV_VIDEO_FILTER);
#define parent_class gst_opencv_dnn_video_filter_parent_class


Scalar
gst_opencv_dnn_video_filter_get_mean_values (GstOpencvDnnVideoFilter * dnn)
{
  if (dnn->channel_order == GST_OPENCV_DNN_CHANNEL_ORDER_RGB)
    return Scalar (dnn->mean_red, dnn->mean_green, dnn->mean_blue);
  else
    return Scalar (dnn->mean_blue, dnn->mean_green, dnn->mean_red);
}


static void
gst_opencv_dnn_video_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstOpencvDnnVideoFilter *dnn = GST_OPENCV_DNN_VIDEO_FILTER (object);

  switch (prop_id) {
    case PROP_MODEL:
      g_free (dnn->model_fn);
      dnn->model_fn = g_value_dup_string (value);
      break;
    case PROP_CONFIG:
      g_free (dnn->config_fn);
      dnn->config_fn = g_value_dup_string (value);
      break;
    case PROP_FRAMEWORK:
      g_free (dnn->framework);
      dnn->framework = g_value_dup_string (value);
      break;
    case PROP_CLASSES:
      g_free (dnn->classes_fn);
      dnn->classes_fn = g_value_dup_string (value);
      break;
    case PROP_WIDTH:
      dnn->width = g_value_get_int (value);
      break;
    case PROP_HEIGHT:
      dnn->height = g_value_get_int (value);
      break;
    case  PROP_CHANNEL_ORDER:
      dnn->channel_order = (GstOpencvDnnChannelOrder) g_value_get_enum (value);
      break;
    case PROP_MEAN_RED:
      dnn->mean_red = g_value_get_double (value);
      break;
    case PROP_MEAN_GREEN:
      dnn->mean_green = g_value_get_double (value);
      break;
    case PROP_MEAN_BLUE:
      dnn->mean_blue = g_value_get_double (value);
      break;
    case PROP_SCALE:
      dnn->scale = g_value_get_double (value);
      break;
    case  PROP_BACKEND:
      dnn->backend = (Backend) g_value_get_enum (value);
      break;
    case  PROP_TARGET:
      dnn->target = (Target) g_value_get_enum (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static void
gst_opencv_dnn_video_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstOpencvDnnVideoFilter *dnn = GST_OPENCV_DNN_VIDEO_FILTER (object);

  switch (prop_id) {
    case PROP_MODEL:
      g_value_set_string (value, dnn->model_fn);
      break;
    case PROP_CONFIG:
      g_value_set_string (value, dnn->config_fn);
      break;
    case PROP_FRAMEWORK:
      g_value_set_string (value, dnn->framework);
      break;
    case PROP_CLASSES:
      g_value_set_string (value, dnn->classes_fn);
      break;
    case PROP_WIDTH:
      g_value_set_int (value, dnn->width);
      break;
    case PROP_HEIGHT:
      g_value_set_int (value, dnn->height);
      break;
    case PROP_CHANNEL_ORDER:
      g_value_set_enum (value, dnn->channel_order);
      break;
    case PROP_MEAN_RED:
      g_value_set_double (value, dnn->mean_red);
      break;
    case PROP_MEAN_GREEN:
      g_value_set_double (value, dnn->mean_green);
      break;
    case PROP_MEAN_BLUE:
      g_value_set_double (value, dnn->mean_blue);
      break;
    case PROP_SCALE:
      g_value_set_double (value, dnn->scale);
      break;
    case PROP_BACKEND:
      g_value_set_enum (value, dnn->backend);
      break;
    case PROP_TARGET:
      g_value_set_enum (value, dnn->target);
      break;
default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static vector<String>
get_out_blob_names(GstOpencvDnnVideoFilter * dnn)
{
#if 1
  static std::vector<String> names(2);
  static gboolean first = TRUE;
  if (first) {
    names[0] = "detection_out_final";
    names[1] = "detection_masks";
  }
  return names;
#else

// FIXME: Store names in object
  std::vector<cv::String> names;

  if (names.empty()) {
    vector<int> out_layers = dnn->net.getUnconnectedOutLayers();
    vector<String> layer_names = dnn->net.getLayerNames();
    names.resize(out_layers.size());
    for (size_t i = 0; i < out_layers.size(); ++i)
      names[i] = layer_names[out_layers[i] - 1];
  }
#endif

  return names;
}


static Mat
gst_opencv_dnn_video_filter_pre_process_default (GstOpencvDnnVideoFilter * dnn,
    Mat & frame, gint width, gint height)
{
  /* Create a 4D blob form the input image */
  Scalar mean;
  bool swap_rb;
  Size size (width, height);

  // FIXME: Check caps if input is RGB or BGR
  if (dnn->channel_order == GST_OPENCV_DNN_CHANNEL_ORDER_RGB) {
    /* Model takes same as image, no need to swap */
    swap_rb = false;
    mean = Scalar (dnn->mean_red, dnn->mean_green, dnn->mean_blue);
  } else {
    /* Mmodel takes BGR while input is RGB, swap needed */
    swap_rb = true;
    mean = Scalar (dnn->mean_blue, dnn->mean_green, dnn->mean_red);
  }

  return blobFromImage (frame, dnn->scale, size, mean, swap_rb, false);
}


static vector<Mat>
run_inference (GstOpencvDnnVideoFilter * dnn, Mat & blob, Mat & frame, int width, int height)
{
  Net &net = dnn->net;

  /* Run the model_fn */
  net.setInput (blob);
  
  if (net.getLayer(0)->outputNameToIndex("im_info") != -1) {
    // Faster-RCNN or R-FCN
    resize (frame, frame, Size (width, height));
    Mat im_info = (Mat_<float>(1, 3) << height, width, 1.6f);
    net.setInput(im_info, "im_info");
  }

  std::vector<Mat> outs;
  net.forward (outs, get_out_blob_names(dnn));
  return outs;
}


static void
draw_inference_time (GstOpencvDnnVideoFilter * dnn, Mat & frame)
{
  vector<double> layers_times;
  double freq = getTickFrequency () / 1000;
  double t = dnn->net.getPerfProfile (layers_times) / freq;
  string label = format ("Inference time: %.2f ms", t);

  // Draw text on semi-transparent background.
  // Get size of the "max" text we expect to draw so that the background box
  // doesn't jump around in size.
  int base_line;
  Size text_size = getTextSize ("Inference time: 9999.99 ms", FONT_HERSHEY_DUPLEX, 0.5, 1,
     &base_line);
  int bg_width = text_size.width;
  int bg_height = text_size.height + base_line;
  Mat roi (frame, Rect(0, 0, bg_width, bg_height));
  Mat bg (roi.size(), CV_8UC3, Scalar::all(0));
  double alpha = 0.4;
  addWeighted (bg, alpha, roi, 1 - alpha, 0, roi);
  putText (frame, label, Point(0, text_size.height), FONT_HERSHEY_DUPLEX, 0.5, Scalar::all(255), 1, LINE_AA);
}


static GstFlowReturn
gst_opencv_dnn_video_filter_transform (GstOpencvVideoFilter * base,
   GstBuffer * buffer, IplImage * img, GstBuffer * outbuf, IplImage * outimg)
{
  GstOpencvDnnVideoFilter *dnn = GST_OPENCV_DNN_VIDEO_FILTER (base);
  GstOpencvDnnVideoFilterClass *dnnclass =
    GST_OPENCV_DNN_VIDEO_FILTER_GET_CLASS (base);

  Net& net = dnn->net;
  Mat frame = cvarrToMat (img);
  Mat outframe = cvarrToMat (outimg);  
  int width = dnn->width > 0 ? dnn->width : frame.cols;
  int height = dnn->height > 0 ? dnn->height : frame.rows;

  Mat blob = gst_opencv_dnn_video_filter_pre_process_default (dnn, frame, width, height);
  vector<Mat> outs = run_inference (dnn, blob, frame, width, height);

  if (dnnclass->post_process)
    dnnclass->post_process (dnn, outs, frame, outframe);

  draw_inference_time (dnn, outframe);

  return GST_FLOW_OK;
}


static GstFlowReturn
gst_opencv_dnn_video_filter_transform_ip (GstOpencvVideoFilter * base, GstBuffer * buf,
    IplImage * img)
{
  GstOpencvDnnVideoFilter *dnn = GST_OPENCV_DNN_VIDEO_FILTER (base);
  GstOpencvDnnVideoFilterClass *dnnclass =
    GST_OPENCV_DNN_VIDEO_FILTER_GET_CLASS (base);

  Net& net = dnn->net;
  Mat frame = cvarrToMat (img);
  int width = dnn->width > 0 ? dnn->width : frame.cols;
  int height = dnn->height > 0 ? dnn->height : frame.rows;

  Mat blob = gst_opencv_dnn_video_filter_pre_process_default (dnn, frame, width, height);
  vector<Mat> outs = run_inference (dnn, blob, frame, width, height);

  if (dnnclass->post_process_ip)
    dnnclass->post_process_ip (dnn, outs, frame);

  draw_inference_time (dnn, frame);

  return GST_FLOW_OK;
}


static void
load_model (GstOpencvDnnVideoFilter * dnn)
{
  dnn->net = readNet (dnn->model_fn, dnn->config_fn, dnn->framework);
  dnn->net.setPreferableBackend(dnn->backend);
  dnn->net.setPreferableTarget(dnn->target);

  /* Parse and store classes */
  dnn->classes.clear();
  if (dnn->classes_fn != NULL) {
    std::ifstream ifs (dnn->classes_fn);
    if (ifs.is_open ()) {
      std::string line;
      while (std::getline(ifs, line))
        dnn->classes.push_back (line);
    } else {
      GST_ERROR_OBJECT (dnn, "Could not open '%s'", dnn->classes_fn);
    }
  }
}


static void
clear_model (GstOpencvDnnVideoFilter * dnn)
{
  (void) dnn;
}


static GstStateChangeReturn
gst_opencv_dnn_video_filter_change_state (GstElement * element,
    GstStateChange transition)
{
  GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
  GstOpencvDnnVideoFilter *dnn = GST_OPENCV_DNN_VIDEO_FILTER_CAST (element);

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      load_model (dnn);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      clear_model (dnn);
      break;
    default:
      break;
  }
  return ret;
}


static void
gst_opencv_dnn_video_filter_dispose (GObject * obj)
{
  G_OBJECT_CLASS (parent_class)->dispose (obj);
}


static void
gst_opencv_dnn_video_filter_init (GstOpencvDnnVideoFilter * dnn)
{
  (void) dnn;
}


static void
gst_opencv_dnn_video_filter_class_init (GstOpencvDnnVideoFilterClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstOpencvVideoFilterClass *gstopencvbasefilter_class;

  element_class = GST_ELEMENT_CLASS (klass);
  gobject_class = G_OBJECT_CLASS (klass);
  gstopencvbasefilter_class = GST_OPENCV_VIDEO_FILTER_CLASS (klass);

  GST_DEBUG_CATEGORY_INIT (gst_opencv_dnn_video_filter_debug,
      "opencvdnnvideofilter", 0, "OpenCV DNN video filter");

  gobject_class->dispose =
    GST_DEBUG_FUNCPTR (gst_opencv_dnn_video_filter_dispose);
  gobject_class->set_property =
    GST_DEBUG_FUNCPTR (gst_opencv_dnn_video_filter_set_property);
  gobject_class->get_property =
    GST_DEBUG_FUNCPTR (gst_opencv_dnn_video_filter_get_property);

  element_class->change_state =
    GST_DEBUG_FUNCPTR (gst_opencv_dnn_video_filter_change_state);


  gstopencvbasefilter_class->cv_trans_func =
    GST_DEBUG_FUNCPTR (gst_opencv_dnn_video_filter_transform);
  gstopencvbasefilter_class->cv_trans_ip_func =
    GST_DEBUG_FUNCPTR (gst_opencv_dnn_video_filter_transform_ip);

  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model",
          "Path to a file of model containing trained weights (required)",
          DEFAULT_MODEL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG,
      g_param_spec_string ("config", "Config",
          "Path to a file of model containing network configuration (required)",
          DEFAULT_CONFIG,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_FRAMEWORK,
      g_param_spec_string ("framework", "Framework",
          "Name tag of origin of model to override automatic detection",
          DEFAULT_FRAMEWORK,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CLASSES,
      g_param_spec_string ("classes", "Classes",
          "Path to text file containing class labels (one per line)",
          DEFAULT_CLASSES,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_WIDTH,
      g_param_spec_int ("width", "Width",
          "Preprocess input image by resizing to specified width",
          -1, G_MAXINT, DEFAULT_WIDTH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_HEIGHT,
      g_param_spec_int ("height", "Height",
          "Preprocess input image by resizing to specified height",
          -1, G_MAXINT, DEFAULT_HEIGHT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_CHANNEL_ORDER,
      g_param_spec_enum ("channel-order", "Channel order",
          "Channel order of model",
          GST_TYPE_OPENCV_DNN_CHANNEL_ORDER, GST_OPENCV_DNN_CHANNEL_ORDER_RGB,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_MEAN_RED,
      g_param_spec_double ("mean-red", "Mean red",
          "Subtract mean from the red channel",
          -G_MAXDOUBLE, G_MAXDOUBLE, DEFAULT_MEAN_RED,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_MEAN_GREEN,
      g_param_spec_double ("mean-green", "Mean green",
          "Subtract mean from the green channel",
          -G_MAXDOUBLE, G_MAXDOUBLE, DEFAULT_MEAN_GREEN,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_MEAN_BLUE,
      g_param_spec_double ("mean-blue", "Mean blue",
          "Subtract mean from the blue channel",
          -G_MAXDOUBLE, G_MAXDOUBLE, DEFAULT_MEAN_BLUE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_SCALE,
      g_param_spec_double ("scale", "Scale",
          "Scale factor to multiply with all channels",
          -G_MAXDOUBLE, G_MAXDOUBLE, DEFAULT_SCALE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));

  g_object_class_install_property (gobject_class, PROP_BACKEND,
      g_param_spec_enum ("backend", "Backend",
          "Computation backend",
          GST_TYPE_OPENCV_DNN_BACKEND, DEFAULT_BACKEND,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_TARGET,
      g_param_spec_enum ("target", "Target",
          "Target computation device",
          GST_TYPE_OPENCV_DNN_TARGET, DEFAULT_TARGET,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT | GST_PARAM_MUTABLE_READY)));
}
