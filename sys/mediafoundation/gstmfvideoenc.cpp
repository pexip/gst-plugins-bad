/* GStreamer
 * Copyright (C) 2020 Seungha Yang <seungha.yang@navercorp.com>
 * Copyright (C) 2020 Seungha Yang <seungha@centricular.com>
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include "gstmfvideoenc.h"
#include <wrl.h>
#include "gstmfvideobuffer.h"
#include <string.h>

#if GST_MF_HAVE_D3D11
#include <d3d10.h>
#endif

using namespace Microsoft::WRL;

G_BEGIN_DECLS

GST_DEBUG_CATEGORY_EXTERN (gst_mf_video_enc_debug);
#define GST_CAT_DEFAULT gst_mf_video_enc_debug

G_END_DECLS

#define gst_mf_video_enc_parent_class parent_class
G_DEFINE_ABSTRACT_TYPE (GstMFVideoEnc, gst_mf_video_enc,
    GST_TYPE_VIDEO_ENCODER);

static void gst_mf_video_enc_dispose (GObject * object);
static void gst_mf_video_enc_set_context (GstElement * element,
    GstContext * context);
static gboolean gst_mf_video_enc_open (GstVideoEncoder * enc);
static gboolean gst_mf_video_enc_close (GstVideoEncoder * enc);
static gboolean gst_mf_video_enc_set_format (GstVideoEncoder * enc,
    GstVideoCodecState * state);
static GstFlowReturn gst_mf_video_enc_handle_frame (GstVideoEncoder * enc,
    GstVideoCodecFrame * frame);
static GstFlowReturn gst_mf_video_enc_finish (GstVideoEncoder * enc);
static gboolean gst_mf_video_enc_flush (GstVideoEncoder * enc);
static gboolean gst_mf_video_enc_propose_allocation (GstVideoEncoder * enc,
    GstQuery * query);
static gboolean gst_mf_video_enc_sink_query (GstVideoEncoder * enc,
    GstQuery * query);
static gboolean gst_mf_video_enc_src_query (GstVideoEncoder * enc,
    GstQuery * query);

static HRESULT gst_mf_video_on_new_sample (GstMFTransform * object,
    IMFSample * sample, GstMFVideoEnc * self);

static void
gst_mf_video_enc_class_init (GstMFVideoEncClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstVideoEncoderClass *videoenc_class = GST_VIDEO_ENCODER_CLASS (klass);

  gobject_class->dispose = gst_mf_video_enc_dispose;

  element_class->set_context = GST_DEBUG_FUNCPTR (gst_mf_video_enc_set_context);

  videoenc_class->open = GST_DEBUG_FUNCPTR (gst_mf_video_enc_open);
  videoenc_class->close = GST_DEBUG_FUNCPTR (gst_mf_video_enc_close);
  videoenc_class->set_format = GST_DEBUG_FUNCPTR (gst_mf_video_enc_set_format);
  videoenc_class->handle_frame =
      GST_DEBUG_FUNCPTR (gst_mf_video_enc_handle_frame);
  videoenc_class->finish = GST_DEBUG_FUNCPTR (gst_mf_video_enc_finish);
  videoenc_class->flush = GST_DEBUG_FUNCPTR (gst_mf_video_enc_flush);
  videoenc_class->propose_allocation =
      GST_DEBUG_FUNCPTR (gst_mf_video_enc_propose_allocation);
  videoenc_class->sink_query =
      GST_DEBUG_FUNCPTR (gst_mf_video_enc_sink_query);
  videoenc_class->src_query =
      GST_DEBUG_FUNCPTR (gst_mf_video_enc_src_query);

  gst_type_mark_as_plugin_api (GST_TYPE_MF_VIDEO_ENC, (GstPluginAPIFlags) 0);
}

static void
gst_mf_video_enc_init (GstMFVideoEnc * self)
{
}

static void
gst_mf_video_enc_dispose (GObject * object)
{
#if GST_MF_HAVE_D3D11
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (object);

  gst_clear_object (&self->d3d11_device);
  gst_clear_object (&self->other_d3d11_device);
#endif

  G_OBJECT_CLASS (parent_class)->dispose (object);
}

static void
gst_mf_video_enc_set_context (GstElement * element, GstContext * context)
{
#if GST_MF_HAVE_D3D11
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (element);

  gst_d3d11_handle_set_context (element, context, 0, &self->other_d3d11_device);
#endif

  GST_ELEMENT_CLASS (parent_class)->set_context (element, context);
}

static gboolean
gst_mf_video_enc_open (GstVideoEncoder * enc)
{
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);
  GstMFVideoEncClass *klass = GST_MF_VIDEO_ENC_GET_CLASS (enc);
  GstMFVideoEncDeviceCaps *device_caps = &klass->device_caps;
  GstMFTransformEnumParams enum_params = { 0, };
  gint64 adapter_luid = 0;
  MFT_REGISTER_TYPE_INFO output_type;
  gboolean ret;

#if GST_MF_HAVE_D3D11
  if (device_caps->d3d11_aware) {
    HRESULT hr;
    ID3D11Device *device_handle;
    ComPtr<ID3D10Multithread> multi_thread;
    GstD3D11Device *device;

    if (!gst_d3d11_ensure_element_data (GST_ELEMENT_CAST (self),
            device_caps->adapter, &self->other_d3d11_device)) {
      GST_ERROR_OBJECT (self, "Other d3d11 device is unavailable");
      return FALSE;
    }

    /* Create our own device with D3D11_CREATE_DEVICE_VIDEO_SUPPORT */
    self->d3d11_device = gst_d3d11_device_new (device_caps->adapter,
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT);
    if (!self->d3d11_device) {
      GST_ERROR_OBJECT (self, "Couldn't create internal d3d11 device");
      gst_clear_object (&self->other_d3d11_device);
      return FALSE;
    }

    device = self->d3d11_device;

    hr = MFCreateDXGIDeviceManager (&self->reset_token,
        &self->device_manager);
    if (!gst_mf_result (hr)) {
      GST_ERROR_OBJECT (self, "Couldn't create DXGI device manager");
      gst_clear_object (&self->other_d3d11_device);
      gst_clear_object (&self->d3d11_device);
      return FALSE;
    }

    device_handle = gst_d3d11_device_get_device_handle (device);
    /* Enable multi thread protection as this device will be shared with
     * MFT */
    hr = device_handle->QueryInterface (IID_PPV_ARGS (&multi_thread));
    if (!gst_d3d11_result (hr, device)) {
      GST_WARNING_OBJECT (self,
          "device doesn't suport ID3D10Multithread interface");
      gst_clear_object (&self->other_d3d11_device);
      gst_clear_object (&self->d3d11_device);
    }

    multi_thread->SetMultithreadProtected (TRUE);

    hr = self->device_manager->ResetDevice ((IUnknown *) device_handle,
        self->reset_token);
    if (!gst_mf_result (hr)) {
      GST_ERROR_OBJECT (self,
          "Couldn't reset device with given d3d11 device");
      gst_clear_object (&self->other_d3d11_device);
      gst_clear_object (&self->d3d11_device);
      return FALSE;
    }

    g_object_get (self->d3d11_device, "adapter-luid", &adapter_luid, NULL);
  }
#endif

  output_type.guidMajorType = MFMediaType_Video;
  output_type.guidSubtype = klass->codec_id;

  enum_params.category = MFT_CATEGORY_VIDEO_ENCODER;
  enum_params.enum_flags = klass->enum_flags;
  enum_params.output_typeinfo = &output_type;
  enum_params.device_index = klass->device_index;

  if (device_caps->d3d11_aware)
    enum_params.adapter_luid = adapter_luid;

  GST_DEBUG_OBJECT (self,
      "Create MFT with enum flags: 0x%x, device index: %d, d3d11 aware: %d, "
      "adapter-luid %" G_GINT64_FORMAT, klass->enum_flags, klass->device_index,
      device_caps->d3d11_aware, adapter_luid);

  self->transform = gst_mf_transform_new (&enum_params);
  ret = !!self->transform;

  if (!ret) {
    GST_ERROR_OBJECT (self, "Cannot create MFT object");
    return FALSE;
  }

  /* In case of hardware MFT, it will be running on async mode.
   * And new output sample callback will be called from Media Foundation's
   * internal worker queue thread */
  if (self->transform &&
      (enum_params.enum_flags & MFT_ENUM_FLAG_HARDWARE) ==
          MFT_ENUM_FLAG_HARDWARE) {
    self->async_mft = TRUE;
    gst_mf_transform_set_new_sample_callback (self->transform,
        (GstMFTransformNewSampleCallback) gst_mf_video_on_new_sample,
        self);
  } else {
    self->async_mft = FALSE;
  }

  return ret;
}

static gboolean
gst_mf_video_enc_close (GstVideoEncoder * enc)
{
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);

  gst_clear_object (&self->transform);

  if (self->input_state) {
    gst_video_codec_state_unref (self->input_state);
    self->input_state = NULL;
  }

#if GST_MF_HAVE_D3D11
  if (self->device_manager) {
    self->device_manager->Release ();
    self->device_manager = nullptr;
  }

  if (self->mf_allocator) {
    self->mf_allocator->UninitializeSampleAllocator ();
    self->mf_allocator->Release ();
    self->mf_allocator = NULL;
  }

  gst_clear_object (&self->other_d3d11_device);
  gst_clear_object (&self->d3d11_device);
#endif

  return TRUE;
}

static gboolean
gst_mf_video_enc_set_format (GstVideoEncoder * enc, GstVideoCodecState * state)
{
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);
  GstMFVideoEncClass *klass = GST_MF_VIDEO_ENC_GET_CLASS (enc);
  GstVideoInfo *info = &state->info;
  ComPtr<IMFMediaType> in_type;
  ComPtr<IMFMediaType> out_type;
  GList *input_types = NULL;
  GList *iter;
  HRESULT hr;
  gint fps_n, fps_d;

  GST_DEBUG_OBJECT (self, "Set format");

  gst_mf_video_enc_finish (enc);

  if (self->input_state)
    gst_video_codec_state_unref (self->input_state);
  self->input_state = gst_video_codec_state_ref (state);

  if (!gst_mf_transform_open (self->transform)) {
    GST_ERROR_OBJECT (self, "Failed to open MFT");
    return FALSE;
  }

  if (self->device_manager) {
    if (!gst_mf_transform_set_device_manager (self->transform,
        self->device_manager)) {
      GST_ERROR_OBJECT (self, "Couldn't set device manager");
      return FALSE;
    } else {
      GST_DEBUG_OBJECT (self, "set device manager done");
    }
  }

  hr = MFCreateMediaType (out_type.GetAddressOf ());
  if (!gst_mf_result (hr))
    return FALSE;

  hr = out_type->SetGUID (MF_MT_MAJOR_TYPE, MFMediaType_Video);
  if (!gst_mf_result (hr))
    return FALSE;

  if (klass->set_option) {
    if (!klass->set_option (self, self->input_state, out_type.Get ())) {
      GST_ERROR_OBJECT (self, "subclass failed to set option");
      return FALSE;
    }
  }

  fps_n = GST_VIDEO_INFO_FPS_N (info);
  fps_d = GST_VIDEO_INFO_FPS_D (info);
  if (fps_n == 0 || fps_d == 0) {
    fps_n = 0;
    fps_d = 1;
  }

  hr = MFSetAttributeRatio (out_type.Get (), MF_MT_FRAME_RATE, fps_n, fps_d);
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self,
        "Couldn't set framerate %d/%d, hr: 0x%x", (guint) hr);
    return FALSE;
  }

  hr = MFSetAttributeSize (out_type.Get (), MF_MT_FRAME_SIZE,
      GST_VIDEO_INFO_WIDTH (info), GST_VIDEO_INFO_HEIGHT (info));
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self,
        "Couldn't set resolution %dx%d, hr: 0x%x", GST_VIDEO_INFO_WIDTH (info),
        GST_VIDEO_INFO_HEIGHT (info), (guint) hr);
    return FALSE;
  }

  hr = MFSetAttributeRatio (out_type.Get (), MF_MT_PIXEL_ASPECT_RATIO,
      GST_VIDEO_INFO_PAR_N (info), GST_VIDEO_INFO_PAR_D (info));
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self, "Couldn't set par %d/%d",
        GST_VIDEO_INFO_PAR_N (info), GST_VIDEO_INFO_PAR_D (info));
    return FALSE;
  }

  hr = out_type->SetUINT32 (MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self,
        "Couldn't set interlace mode, hr: 0x%x", (guint) hr);
    return FALSE;
  }

  if (!gst_mf_transform_set_output_type (self->transform, out_type.Get ())) {
    GST_ERROR_OBJECT (self, "Couldn't set output type");
    return FALSE;
  }

  if (!gst_mf_transform_get_input_available_types (self->transform,
      &input_types)) {
    GST_ERROR_OBJECT (self, "Couldn't get available input types");
    return FALSE;
  }

  for (iter = input_types; iter; iter = g_list_next (iter)) {
    GstVideoFormat format;
    GUID subtype;
    IMFMediaType *type = (IMFMediaType *) iter->data;

    hr = type->GetGUID (MF_MT_SUBTYPE, &subtype);
    if (!gst_mf_result (hr))
      continue;

    format = gst_mf_video_subtype_to_video_format (&subtype);
    if (format != GST_VIDEO_INFO_FORMAT (info))
      continue;

    in_type = type;
  }

  g_list_free_full (input_types, (GDestroyNotify) gst_mf_media_type_release);

  if (!in_type) {
    GST_ERROR_OBJECT (self,
        "Couldn't convert input caps %" GST_PTR_FORMAT " to media type",
        state->caps);
    return FALSE;
  }

  hr = MFSetAttributeSize (in_type.Get (), MF_MT_FRAME_SIZE,
      GST_VIDEO_INFO_WIDTH (info), GST_VIDEO_INFO_HEIGHT (info));
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self, "Couldn't set frame size %dx%d",
        GST_VIDEO_INFO_WIDTH (info), GST_VIDEO_INFO_HEIGHT (info));
    return FALSE;
  }

  hr = in_type->SetUINT32 (MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self,
        "Couldn't set interlace mode, hr: 0x%x", (guint) hr);
    return FALSE;
  }

  hr = MFSetAttributeRatio (in_type.Get (), MF_MT_PIXEL_ASPECT_RATIO,
      GST_VIDEO_INFO_PAR_N (info), GST_VIDEO_INFO_PAR_D (info));
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self, "Couldn't set par %d/%d",
        GST_VIDEO_INFO_PAR_N (info), GST_VIDEO_INFO_PAR_D (info));
    return FALSE;
  }

  hr = MFSetAttributeRatio (in_type.Get (), MF_MT_FRAME_RATE, fps_n, fps_d);
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self, "Couldn't set framerate ratio %d/%d", fps_n, fps_d);
    return FALSE;
  }

  hr = in_type->SetUINT32 (MF_MT_DEFAULT_STRIDE,
      GST_VIDEO_INFO_PLANE_STRIDE (info, 0));
  if (!gst_mf_result (hr)) {
    GST_ERROR_OBJECT (self, "Couldn't set default stride");
    return FALSE;
  }

  if (!gst_mf_transform_set_input_type (self->transform, in_type.Get ())) {
    GST_ERROR_OBJECT (self, "Couldn't set input media type");
    return FALSE;
  }

  g_assert (klass->set_src_caps != NULL);
  if (!klass->set_src_caps (self, self->input_state, out_type.Get ())) {
    GST_ERROR_OBJECT (self, "subclass couldn't set src caps");
    return FALSE;
  }

#if GST_MF_HAVE_D3D11
  if (self->mf_allocator) {
    self->mf_allocator->UninitializeSampleAllocator ();
    self->mf_allocator->Release ();
    self->mf_allocator = NULL;
  }

  /* Check whether upstream is d3d11 element */
  if (state->caps) {
    GstCapsFeatures *features;
    ComPtr<IMFVideoSampleAllocatorEx> allocator;

    features = gst_caps_get_features (state->caps, 0);

    if (features &&
        gst_caps_features_contains (features,
            GST_CAPS_FEATURE_MEMORY_D3D11_MEMORY)) {
      GST_DEBUG_OBJECT (self, "found D3D11 memory feature");

      hr = MFCreateVideoSampleAllocatorEx (IID_PPV_ARGS (&allocator));
      if (!gst_mf_result (hr))
        GST_WARNING_OBJECT (self, "IMFVideoSampleAllocatorEx interface is unavailable");
    }

    if (allocator) {
      do {
        ComPtr<IMFAttributes> attr;

        hr = MFCreateAttributes (&attr, 4);
        if (!gst_mf_result (hr))
          break;

        /* Only one buffer per sample
        * (multiple sample is usually for multi-view things) */
        hr = attr->SetUINT32 (GST_GUID_MF_SA_BUFFERS_PER_SAMPLE, 1);
        if (!gst_mf_result (hr))
          break;

        hr = attr->SetUINT32 (GST_GUID_MF_SA_D3D11_USAGE, D3D11_USAGE_DEFAULT);
        if (!gst_mf_result (hr))
          break;

        /* TODO: Check if we need to use keyed-mutex */
        hr = attr->SetUINT32 (GST_GUID_MF_SA_D3D11_SHARED_WITHOUT_MUTEX, TRUE);
        if (!gst_mf_result (hr))
          break;

        hr = attr->SetUINT32 (GST_GUID_MF_SA_D3D11_BINDFLAGS,
            D3D11_BIND_VIDEO_ENCODER);
        if (!gst_mf_result (hr))
          break;

        hr = allocator->SetDirectXManager (self->device_manager);
        if (!gst_mf_result (hr))
          break;

        hr = allocator->InitializeSampleAllocatorEx (
          /* min samples, since we are running on async mode,
          * at least 2 samples would be required */
          2,
          /* max samples, why 16 + 2? it's just magic number
          * (H264 max dpb size 16 + our min sample size 2) */
          16 + 2,
          attr.Get (),
          in_type.Get ()
        );

        if (!gst_mf_result (hr))
          break;

        GST_DEBUG_OBJECT (self, "IMFVideoSampleAllocatorEx is initialized");

        self->mf_allocator = allocator.Detach ();
      } while (0);
    }
  }
#endif

  return TRUE;
}

static void
gst_mf_video_buffer_free (GstVideoFrame * frame)
{
  if (!frame)
    return;

  gst_video_frame_unmap (frame);
  g_free (frame);
}

static gboolean
gst_mf_video_enc_frame_needs_copy (GstVideoFrame * vframe)
{
  /* Single plane data can be used without copy */
  if (GST_VIDEO_FRAME_N_PLANES (vframe) == 1)
    return FALSE;

  switch (GST_VIDEO_FRAME_FORMAT (vframe)) {
    case GST_VIDEO_FORMAT_I420:
    {
      guint8 *data, *other_data;
      guint size;

      /* Unexpected stride size, Media Foundation doesn't provide API for
       * per plane stride information */
      if (GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 0) !=
          2 * GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 1) ||
          GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 1) !=
          GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 2)) {
        return TRUE;
      }

      size = GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 0) *
          GST_VIDEO_FRAME_HEIGHT (vframe);
      if (size + GST_VIDEO_FRAME_PLANE_OFFSET (vframe, 0) !=
          GST_VIDEO_FRAME_PLANE_OFFSET (vframe, 1))
        return TRUE;

      data = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (vframe, 0);
      other_data = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (vframe, 1);
      if (data + size != other_data)
        return TRUE;

      size = GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 1) *
          GST_VIDEO_FRAME_COMP_HEIGHT (vframe, 1);
      if (size + GST_VIDEO_FRAME_PLANE_OFFSET (vframe, 1) !=
          GST_VIDEO_FRAME_PLANE_OFFSET (vframe, 2))
        return TRUE;

      data = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (vframe, 1);
      other_data = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (vframe, 2);
      if (data + size != other_data)
        return TRUE;

      return FALSE;
    }
    case GST_VIDEO_FORMAT_NV12:
    case GST_VIDEO_FORMAT_P010_10LE:
    case GST_VIDEO_FORMAT_P016_LE:
    {
      guint8 *data, *other_data;
      guint size;

      /* Unexpected stride size, Media Foundation doesn't provide API for
       * per plane stride information */
      if (GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 0) !=
          GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 1)) {
        return TRUE;
      }

      size = GST_VIDEO_FRAME_PLANE_STRIDE (vframe, 0) *
          GST_VIDEO_FRAME_HEIGHT (vframe);

      /* Unexpected padding */
      if (size + GST_VIDEO_FRAME_PLANE_OFFSET (vframe, 0) !=
          GST_VIDEO_FRAME_PLANE_OFFSET (vframe, 1))
        return TRUE;

      data = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (vframe, 0);
      other_data = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (vframe, 1);
      if (data + size != other_data)
        return TRUE;

      return FALSE;
    }
    default:
      g_assert_not_reached ();
      return TRUE;
  }

  return TRUE;
}

typedef struct
{
  GstClockTime mf_pts;
} GstMFVideoEncFrameData;

static gboolean
gst_mf_video_enc_process_input (GstMFVideoEnc * self,
    GstVideoCodecFrame * frame, IMFSample * sample)
{
  GstMFVideoEncClass *klass = GST_MF_VIDEO_ENC_GET_CLASS (self);
  HRESULT hr;
  gboolean unset_force_keyframe = FALSE;
  GstMFVideoEncFrameData *frame_data = NULL;
  gboolean res;

  frame_data = g_new0 (GstMFVideoEncFrameData, 1);
  frame_data->mf_pts = frame->pts / 100;

  gst_video_codec_frame_set_user_data (frame,
      frame_data, (GDestroyNotify) g_free);

  hr = sample->SetSampleTime (frame_data->mf_pts);
  if (!gst_mf_result (hr))
    return FALSE;

  hr = sample->SetSampleDuration (
      GST_CLOCK_TIME_IS_VALID (frame->duration) ? frame->duration / 100 : 0);
  if (!gst_mf_result (hr))
    return FALSE;

  if (GST_VIDEO_CODEC_FRAME_IS_FORCE_KEYFRAME (frame)) {
    if (klass->device_caps.force_keyframe) {
      unset_force_keyframe =
          gst_mf_transform_set_codec_api_uint32 (self->transform,
          &CODECAPI_AVEncVideoForceKeyFrame, TRUE);
    } else {
      GST_WARNING_OBJECT (self, "encoder does not support force keyframe");
    }
  }

  /* Unlock temporary so that we can output frame from Media Foundation's
   * worker thread.
   * While we are processing input, MFT might notify
   * METransformHaveOutput event from Media Foundation's internal worker queue
   * thread. Then we will output encoded data from the thread synchroniously,
   * not from streaming (this) thread */
  if (self->async_mft)
    GST_VIDEO_ENCODER_STREAM_UNLOCK (self);
  res = gst_mf_transform_process_input (self->transform, sample);
  if (self->async_mft)
      GST_VIDEO_ENCODER_STREAM_LOCK (self);

  if (unset_force_keyframe) {
    gst_mf_transform_set_codec_api_uint32 (self->transform,
        &CODECAPI_AVEncVideoForceKeyFrame, FALSE);
  }

  if (!res) {
    GST_ERROR_OBJECT (self, "Failed to process input");
    return FALSE;
  }

  return TRUE;
}

static GstVideoCodecFrame *
gst_mf_video_enc_find_output_frame (GstMFVideoEnc * self, UINT64 mf_dts,
    UINT64 mf_pts)
{
  GList *l, *walk = gst_video_encoder_get_frames (GST_VIDEO_ENCODER (self));
  GstVideoCodecFrame *ret = NULL;

  for (l = walk; l; l = l->next) {
    GstVideoCodecFrame *frame = (GstVideoCodecFrame *) l->data;
    GstMFVideoEncFrameData *data = (GstMFVideoEncFrameData *)
        gst_video_codec_frame_get_user_data (frame);

    if (!data)
      continue;

    if (mf_dts == data->mf_pts) {
      ret = frame;
      break;
    }
  }

  /* find target with pts */
  if (!ret) {
    for (l = walk; l; l = l->next) {
      GstVideoCodecFrame *frame = (GstVideoCodecFrame *) l->data;
      GstMFVideoEncFrameData *data = (GstMFVideoEncFrameData *)
          gst_video_codec_frame_get_user_data (frame);

      if (!data)
        continue;

      if (mf_pts == data->mf_pts) {
        ret = frame;
        break;
      }
    }
  }

  if (ret) {
    gst_video_codec_frame_ref (ret);
  } else {
    /* just return the oldest one */
    ret = gst_video_encoder_get_oldest_frame (GST_VIDEO_ENCODER (self));
  }

  if (walk)
    g_list_free_full (walk, (GDestroyNotify) gst_video_codec_frame_unref);

  return ret;
}

static HRESULT
gst_mf_video_enc_finish_sample (GstMFVideoEnc * self, IMFSample * sample)
{
  HRESULT hr = S_OK;
  BYTE *data;
  ComPtr<IMFMediaBuffer> media_buffer;
  GstBuffer *buffer;
  GstFlowReturn res = GST_FLOW_ERROR;
  GstVideoCodecFrame *frame;
  LONGLONG sample_timestamp;
  LONGLONG sample_duration;
  UINT32 keyframe = FALSE;
  UINT64 mf_dts = GST_CLOCK_TIME_NONE;
  DWORD buffer_len;

  hr = sample->GetBufferByIndex (0, media_buffer.GetAddressOf ());
  if (!gst_mf_result (hr))
    goto done;

  hr = media_buffer->Lock (&data, NULL, &buffer_len);
  if (!gst_mf_result (hr))
    goto done;

  buffer = gst_buffer_new_allocate (NULL, buffer_len, NULL);
  gst_buffer_fill (buffer, 0, data, buffer_len);
  media_buffer->Unlock ();

  sample->GetSampleTime (&sample_timestamp);
  sample->GetSampleDuration (&sample_duration);
  sample->GetUINT32 (MFSampleExtension_CleanPoint, &keyframe);

  hr = sample->GetUINT64 (MFSampleExtension_DecodeTimestamp, &mf_dts);
  if (FAILED (hr)) {
    mf_dts = sample_timestamp;
    hr = S_OK;
  }

  frame = gst_mf_video_enc_find_output_frame (self,
      mf_dts, (UINT64) sample_timestamp);

  if (frame) {
    if (keyframe) {
      GST_DEBUG_OBJECT (self, "Keyframe pts %" GST_TIME_FORMAT,
          GST_TIME_ARGS (frame->pts));
      GST_VIDEO_CODEC_FRAME_SET_SYNC_POINT (frame);
      GST_BUFFER_FLAG_UNSET (buffer, GST_BUFFER_FLAG_DELTA_UNIT);
    } else {
      GST_BUFFER_FLAG_SET (buffer, GST_BUFFER_FLAG_DELTA_UNIT);
    }

    frame->pts = sample_timestamp * 100;
    frame->dts = mf_dts * 100;
    frame->duration = sample_duration * 100;
    frame->output_buffer = buffer;

    res = gst_video_encoder_finish_frame (GST_VIDEO_ENCODER (self), frame);
  } else {
    GST_BUFFER_DTS (buffer) = mf_dts * 100;
    GST_BUFFER_PTS (buffer) = sample_timestamp * 100;
    GST_BUFFER_DURATION (buffer) = sample_duration * 100;

    if (keyframe) {
      GST_DEBUG_OBJECT (self, "Keyframe pts %" GST_TIME_FORMAT,
          GST_BUFFER_PTS (buffer));
      GST_BUFFER_FLAG_UNSET (buffer, GST_BUFFER_FLAG_DELTA_UNIT);
    } else {
      GST_BUFFER_FLAG_SET (buffer, GST_BUFFER_FLAG_DELTA_UNIT);
    }

    res = gst_pad_push (GST_VIDEO_ENCODER_SRC_PAD (self), buffer);
  }

done:
  self->last_ret = res;

  return hr;
}

static GstFlowReturn
gst_mf_video_enc_process_output (GstMFVideoEnc * self)
{
  ComPtr<IMFSample> sample;
  GstFlowReturn res = GST_FLOW_ERROR;

  res = gst_mf_transform_get_output (self->transform, &sample);

  if (res != GST_FLOW_OK)
    return res;

  gst_mf_video_enc_finish_sample (self, sample.Get ());

  return self->last_ret;
}

static gboolean
gst_mf_video_enc_create_input_sample (GstMFVideoEnc * self,
    GstVideoCodecFrame * frame, IMFSample ** sample)
{
  HRESULT hr;
  ComPtr<IMFSample> new_sample;
  ComPtr<IMFMediaBuffer> media_buffer;
  ComPtr<IGstMFVideoBuffer> video_buffer;
  GstVideoInfo *info = &self->input_state->info;
  gint i, j;
  GstVideoFrame *vframe = NULL;
  BYTE *data = NULL;
  gboolean need_copy;

  vframe = g_new0 (GstVideoFrame, 1);

  if (!gst_video_frame_map (vframe, info, frame->input_buffer, GST_MAP_READ)) {
    GST_ERROR_OBJECT (self, "Couldn't map input frame");
    g_free (vframe);
    return FALSE;
  }

  hr = MFCreateSample (&new_sample);
  if (!gst_mf_result (hr))
    goto error;

  /* Check if we can forward this memory to Media Foundation without copy */
  need_copy = gst_mf_video_enc_frame_needs_copy (vframe);
  if (need_copy) {
    GST_TRACE_OBJECT (self, "Copy input buffer into Media Foundation memory");
    hr = MFCreateMemoryBuffer (GST_VIDEO_INFO_SIZE (info), &media_buffer);
  } else {
    GST_TRACE_OBJECT (self, "Can use input buffer without copy");
    hr = IGstMFVideoBuffer::CreateInstanceWrapped (&vframe->info,
      (BYTE *) GST_VIDEO_FRAME_PLANE_DATA (vframe, 0),
      GST_VIDEO_INFO_SIZE (&vframe->info), &media_buffer);
  }

  if (!gst_mf_result (hr))
    goto error;

  if (!need_copy) {
    hr = media_buffer.As (&video_buffer);
    if (!gst_mf_result (hr))
      goto error;
  } else {
    hr = media_buffer->Lock (&data, NULL, NULL);
    if (!gst_mf_result (hr))
      goto error;

    for (i = 0; i < GST_VIDEO_INFO_N_PLANES (info); i++) {
      guint8 *src, *dst;
      gint src_stride, dst_stride;
      gint width;

      src = (guint8 *) GST_VIDEO_FRAME_PLANE_DATA (vframe, i);
      dst = data + GST_VIDEO_INFO_PLANE_OFFSET (info, i);

      src_stride = GST_VIDEO_FRAME_PLANE_STRIDE (vframe, i);
      dst_stride = GST_VIDEO_INFO_PLANE_STRIDE (info, i);

      width = GST_VIDEO_INFO_COMP_WIDTH (info, i)
          * GST_VIDEO_INFO_COMP_PSTRIDE (info, i);

      for (j = 0; j < GST_VIDEO_INFO_COMP_HEIGHT (info, i); j++) {
        memcpy (dst, src, width);
        src += src_stride;
        dst += dst_stride;
      }
    }

    media_buffer->Unlock ();
  }

  hr = media_buffer->SetCurrentLength (GST_VIDEO_INFO_SIZE (info));
  if (!gst_mf_result (hr))
    goto error;

  hr = new_sample->AddBuffer (media_buffer.Get ());
  if (!gst_mf_result (hr))
    goto error;

  if (!need_copy) {
    /* IGstMFVideoBuffer will hold GstVideoFrame (+ GstBuffer), then it will be
     * cleared when it's no more referenced by Media Foundation internals */
    hr = video_buffer->SetUserData ((gpointer) vframe,
        (GDestroyNotify) gst_mf_video_buffer_free);
    if (!gst_mf_result (hr))
      goto error;
  } else {
    gst_video_frame_unmap (vframe);
    g_free (vframe);
    vframe = NULL;
  }

  *sample = new_sample.Detach ();

  return TRUE;

error:
  if (vframe) {
    gst_video_frame_unmap (vframe);
    g_free (vframe);
  }

  return FALSE;
}

#if GST_MF_HAVE_D3D11
static gboolean
gst_mf_video_enc_create_input_sample_d3d11 (GstMFVideoEnc * self,
    GstVideoCodecFrame * frame, IMFSample ** sample)
{
  HRESULT hr;
  ComPtr<IMFSample> new_sample;
  ComPtr<IMFMediaBuffer> mf_buffer;
  ComPtr<IMFDXGIBuffer> dxgi_buffer;
  ComPtr<ID3D11Texture2D> mf_texture;
  ComPtr<IDXGIResource> dxgi_resource;
  ComPtr<ID3D11Texture2D> shared_texture;
  ComPtr<ID3D11Query> query;
  D3D11_QUERY_DESC query_desc;
  BOOL sync_done = FALSE;
  HANDLE shared_handle;
  GstMemory *mem;
  GstD3D11Memory *dmem;
  ID3D11Texture2D *texture;
  ID3D11Device *device_handle;
  ID3D11DeviceContext *context_handle;
  GstMapInfo info;
  D3D11_BOX src_box = { 0, };
  D3D11_TEXTURE2D_DESC dst_desc, src_desc;
  guint subidx;

  if (!self->mf_allocator) {
    GST_WARNING_OBJECT (self, "IMFVideoSampleAllocatorEx was configured");
    return FALSE;
  }

  mem = gst_buffer_peek_memory (frame->input_buffer, 0);
  if (!gst_is_d3d11_memory (mem)) {
    GST_WARNING_OBJECT (self, "Non-d3d11 memory");
    return FALSE;
  }

  dmem = (GstD3D11Memory * ) mem;
  device_handle = gst_d3d11_device_get_device_handle (dmem->device);
  context_handle = gst_d3d11_device_get_device_context_handle (dmem->device);

  /* 1) Allocate new encoding surface */
  hr = self->mf_allocator->AllocateSample (&new_sample);
  if (!gst_mf_result (hr)) {
    GST_WARNING_OBJECT (self,
        "Couldn't allocate new sample via IMFVideoSampleAllocatorEx");
    return FALSE;
  }

  hr = new_sample->GetBufferByIndex (0, &mf_buffer);
  if (!gst_mf_result (hr)) {
    GST_WARNING_OBJECT (self, "Couldn't get IMFMediaBuffer from sample");
    return FALSE;
  }

  hr = mf_buffer.As (&dxgi_buffer);
  if (!gst_mf_result (hr)) {
    GST_WARNING_OBJECT (self, "Couldn't get IMFDXGIBuffer from IMFMediaBuffer");
    return FALSE;
  }

  hr = dxgi_buffer->GetResource (IID_PPV_ARGS (&mf_texture));
  if (!gst_mf_result (hr)) {
    GST_WARNING_OBJECT (self,
        "Couldn't get ID3D11Texture2D from IMFDXGIBuffer");
    return FALSE;
  }

  hr = mf_texture.As (&dxgi_resource);
  if (!gst_mf_result (hr)) {
    GST_WARNING_OBJECT (self,
        "Couldn't get IDXGIResource from ID3D11Texture2D");
    return FALSE;
  }

  hr = dxgi_resource->GetSharedHandle (&shared_handle);
  if (!gst_mf_result (hr)) {
    GST_WARNING_OBJECT (self,
        "Couldn't get shared handle from IDXGIResource");
    return FALSE;
  }

  /* Allocation succeeded. Now open shared texture to access it from
   * other device */
  hr = device_handle->OpenSharedResource (shared_handle,
      IID_PPV_ARGS (&shared_texture));
  if (!gst_mf_result (hr)) {
    GST_WARNING_OBJECT (self, "Couldn't open shared resource");
    return FALSE;
  }

  /* 2) Copy upstream texture to mf's texture */
  /* Map memory so that ensure pending upload from staging texture */
  if (!gst_memory_map (mem, &info, (GstMapFlags) (GST_MAP_READ | GST_MAP_D3D11))) {
    GST_ERROR_OBJECT (self, "Couldn't map d3d11 memory");
    return FALSE;
  }

  texture = (ID3D11Texture2D *) info.data;
  texture->GetDesc (&src_desc);
  shared_texture->GetDesc (&dst_desc);
  subidx = gst_d3d11_memory_get_subresource_index (dmem);

  /* src/dst texture size might be different if padding was used.
   * select smaller size */
  src_box.left = 0;
  src_box.top = 0;
  src_box.front = 0;
  src_box.back = 1;
  src_box.right = MIN (src_desc.Width, dst_desc.Width);
  src_box.bottom = MIN (src_desc.Height, dst_desc.Height);

  /* CopySubresourceRegion() might not be able to guarantee
   * copying. To ensure it, we will make use of d3d11 query */
  query_desc.Query = D3D11_QUERY_EVENT;
  query_desc.MiscFlags = 0;

  hr = device_handle->CreateQuery (&query_desc, &query);
  if (!gst_d3d11_result (hr, dmem->device)) {
    GST_ERROR_OBJECT (self, "Couldn't Create event query");
    return FALSE;
  }

  gst_d3d11_device_lock (dmem->device);
  context_handle->CopySubresourceRegion (shared_texture.Get (), 0, 0, 0, 0,
      texture, subidx, &src_box);
  context_handle->End (query.Get());

  /* Wait until all issued GPU commands are finished */
  do {
    context_handle->GetData (query.Get(), &sync_done, sizeof (BOOL), 0);
  } while (!sync_done && (hr == S_OK || hr == S_FALSE));

  if (!gst_d3d11_result (hr, dmem->device)) {
    GST_ERROR_OBJECT (self, "Couldn't sync GPU operation");
    gst_d3d11_device_unlock (dmem->device);
    gst_memory_unmap (mem, &info);

    return FALSE;
  }

  gst_d3d11_device_unlock (dmem->device);
  gst_memory_unmap (mem, &info);

  *sample = new_sample.Detach ();

  return TRUE;
}
#endif

static GstFlowReturn
gst_mf_video_enc_handle_frame (GstVideoEncoder * enc,
    GstVideoCodecFrame * frame)
{
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);
  GstFlowReturn ret = GST_FLOW_OK;
  ComPtr<IMFSample> sample;

#if GST_MF_HAVE_D3D11
  if (self->mf_allocator &&
      !gst_mf_video_enc_create_input_sample_d3d11 (self, frame, &sample)) {
    GST_WARNING_OBJECT (self, "Failed to create IMFSample for D3D11");
    sample = nullptr;
  }
#endif

  if (!sample && !gst_mf_video_enc_create_input_sample (self, frame, &sample)) {
    GST_ERROR_OBJECT (self, "Failed to create IMFSample");
    return GST_FLOW_ERROR;
  }

  if (!gst_mf_video_enc_process_input (self, frame, sample.Get ())) {
    GST_ERROR_OBJECT (self, "Failed to process input");
    ret = GST_FLOW_ERROR;
    goto done;
  }

  /* Don't call process_output for async (hardware) MFT. We will output
   * encoded data from gst_mf_video_on_new_sample() callback which is called
   * from Media Foundation's internal worker queue thread */
  if (!self->async_mft) {
    do {
      ret = gst_mf_video_enc_process_output (self);
    } while (ret == GST_FLOW_OK);
  }

  if (ret == GST_MF_TRANSFORM_FLOW_NEED_DATA)
    ret = GST_FLOW_OK;

done:
  gst_video_codec_frame_unref (frame);

  return ret;
}

static GstFlowReturn
gst_mf_video_enc_finish (GstVideoEncoder * enc)
{
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);
  GstFlowReturn ret = GST_FLOW_OK;

  if (!self->transform)
    return GST_FLOW_OK;

  /* Unlock temporary so that we can output frame from Media Foundation's
   * worker thread */
  if (self->async_mft)
    GST_VIDEO_ENCODER_STREAM_UNLOCK (enc);

  gst_mf_transform_drain (self->transform);

  if (self->async_mft)
    GST_VIDEO_ENCODER_STREAM_LOCK (enc);

  if (!self->async_mft) {
    do {
      ret = gst_mf_video_enc_process_output (self);
    } while (ret == GST_FLOW_OK);
  }

  if (ret == GST_MF_TRANSFORM_FLOW_NEED_DATA)
    ret = GST_FLOW_OK;

  return ret;
}

static gboolean
gst_mf_video_enc_flush (GstVideoEncoder * enc)
{
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);

  if (!self->transform)
    return TRUE;

  /* Unlock while flushing, while flushing, new sample callback might happen */
  if (self->async_mft)
    GST_VIDEO_ENCODER_STREAM_UNLOCK (enc);

  gst_mf_transform_flush (self->transform);

  if (self->async_mft)
    GST_VIDEO_ENCODER_STREAM_LOCK (enc);

  return TRUE;
}

static gboolean
gst_mf_video_enc_propose_allocation (GstVideoEncoder * enc,
    GstQuery * query)
{
#if GST_MF_HAVE_D3D11
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);
  GstVideoInfo info;
  GstBufferPool *pool = NULL;
  GstCaps *caps;
  guint size;
  GstD3D11Device *device = self->other_d3d11_device;

  gst_query_parse_allocation (query, &caps, NULL);

  if (caps == NULL)
    return FALSE;

  if (!gst_video_info_from_caps (&info, caps))
    return FALSE;

  if (gst_query_get_n_allocation_pools (query) == 0) {
    GstCapsFeatures *features;
    GstStructure *config;
    gboolean is_d3d11 = FALSE;

    features = gst_caps_get_features (caps, 0);

    if (features && gst_caps_features_contains (features,
            GST_CAPS_FEATURE_MEMORY_D3D11_MEMORY)) {
      GST_DEBUG_OBJECT (self, "Allocation caps supports d3d11 memory");
      pool = gst_d3d11_buffer_pool_new (device);
      is_d3d11 = TRUE;
    } else {
      pool = gst_video_buffer_pool_new ();
    }

    config = gst_buffer_pool_get_config (pool);

    gst_buffer_pool_config_add_option (config,
        GST_BUFFER_POOL_OPTION_VIDEO_META);

    /* d3d11 pool does not support video alignment */
    if (!is_d3d11) {
      gst_buffer_pool_config_add_option (config,
          GST_BUFFER_POOL_OPTION_VIDEO_ALIGNMENT);
    } else {
      GstD3D11AllocationParams *d3d11_params;
      guint misc_flags = 0;
      gboolean is_hardware = FALSE;
      gint i;

      g_object_get (device, "hardware", &is_hardware, NULL);

      /* In case of hardware, set D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX flag
       * so that it can be shared with other d3d11 devices */
      if (is_hardware)
        misc_flags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

      d3d11_params =
          gst_buffer_pool_config_get_d3d11_allocation_params (config);
      if (!d3d11_params) {
        d3d11_params = gst_d3d11_allocation_params_new (device, &info,
            (GstD3D11AllocationFlags) 0, 0);
      } else {
        for (i = 0; i < GST_VIDEO_INFO_N_PLANES (&info); i++)
          d3d11_params->desc[i].MiscFlags |= misc_flags;
      }

      gst_buffer_pool_config_set_d3d11_allocation_params (config, d3d11_params);
      gst_d3d11_allocation_params_free (d3d11_params);
    }

    size = GST_VIDEO_INFO_SIZE (&info);
    gst_buffer_pool_config_set_params (config, caps, size, 0, 0);

    if (!gst_buffer_pool_set_config (pool, config))
      goto config_failed;

    /* d3d11 buffer pool might update buffer size by self */
    if (is_d3d11)
      size = GST_D3D11_BUFFER_POOL (pool)->buffer_size;

    gst_query_add_allocation_pool (query, pool, size, 0, 0);
    gst_object_unref (pool);
  }

  gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);

  return TRUE;

  /* ERRORS */
config_failed:
  {
    GST_ERROR_OBJECT (self, "failed to set config");
    gst_object_unref (pool);
    return FALSE;
  }

#else
  return GST_VIDEO_ENCODER_CLASS (parent_class)->propose_allocation (enc,
        query);
#endif
}

static gboolean
gst_mf_video_enc_sink_query (GstVideoEncoder * enc, GstQuery *query)
{
#if GST_MF_HAVE_D3D11
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CONTEXT:
      if (gst_d3d11_handle_context_query (GST_ELEMENT (self),
              query, self->other_d3d11_device)) {
        return TRUE;
      }
      break;
    default:
      break;
  }
#endif

  return GST_VIDEO_ENCODER_CLASS (parent_class)->sink_query (enc, query);
}

static gboolean
gst_mf_video_enc_src_query (GstVideoEncoder * enc,  GstQuery * query)
{
#if GST_MF_HAVE_D3D11
  GstMFVideoEnc *self = GST_MF_VIDEO_ENC (enc);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CONTEXT:
      if (gst_d3d11_handle_context_query (GST_ELEMENT (self),
              query, self->other_d3d11_device)) {
        return TRUE;
      }
      break;
    default:
      break;
  }
#endif

  return GST_VIDEO_ENCODER_CLASS (parent_class)->src_query (enc, query);
}

static HRESULT
gst_mf_video_on_new_sample (GstMFTransform * object,
    IMFSample * sample, GstMFVideoEnc * self)
{
  GST_LOG_OBJECT (self, "New Sample callback");

  /* NOTE: this callback will be called from Media Foundation's internal
   * worker queue thread */
  GST_VIDEO_ENCODER_STREAM_LOCK (self);
  gst_mf_video_enc_finish_sample (self, sample);
  GST_VIDEO_ENCODER_STREAM_UNLOCK (self);

  return S_OK;
}

typedef struct
{
  guint profile;
  const gchar *profile_str;
} GstMFVideoEncProfileMap;

static void
gst_mf_video_enc_enum_internal (GstMFTransform * transform, GUID &subtype,
    GstObject * d3d11_device, GstMFVideoEncDeviceCaps * device_caps,
    GstCaps ** sink_template, GstCaps ** src_template)
{
  HRESULT hr;
  MFT_REGISTER_TYPE_INFO *infos;
  UINT32 info_size;
  gint i;
  GstCaps *src_caps = NULL;
  GstCaps *sink_caps = NULL;
  GstCaps *d3d11_caps = NULL;
  GValue *supported_formats = NULL;
  GValue *profiles = NULL;
  gboolean have_I420 = FALSE;
  gboolean have_NV12 = FALSE;
  gboolean have_P010 = FALSE;
  gboolean d3d11_aware = FALSE;
  gchar *device_name = NULL;
  IMFActivate *activate;
  IMFTransform *encoder;
  ICodecAPI *codec_api;
  ComPtr<IMFMediaType> out_type;
  GstMFVideoEncProfileMap h264_profile_map[] = {
    { eAVEncH264VProfile_High, "high" },
    { eAVEncH264VProfile_Main, "main" },
    { eAVEncH264VProfile_Base, "baseline" },
    { 0, NULL },
  };
  GstMFVideoEncProfileMap hevc_profile_map[] = {
    { eAVEncH265VProfile_Main_420_8, "main" },
    { eAVEncH265VProfile_Main_420_10, "main-10" },
    { 0, NULL },
  };
  GstMFVideoEncProfileMap *profile_to_check = NULL;
  static gchar *h264_caps_str =
      "video/x-h264, stream-format=(string) byte-stream, alignment=(string) au";
  static gchar *hevc_caps_str =
      "video/x-h265, stream-format=(string) byte-stream, alignment=(string) au";
  static gchar *vp9_caps_str = "video/x-vp9";
  static gchar *codec_caps_str = NULL;

  /* NOTE: depending on environment,
   * some enumerated h/w MFT might not be usable (e.g., multiple GPU case) */
  if (!gst_mf_transform_open (transform))
    return;

  activate = gst_mf_transform_get_activate_handle (transform);
  if (!activate) {
    GST_WARNING_OBJECT (transform, "No IMFActivate interface available");
    return;
  }

  encoder = gst_mf_transform_get_transform_handle (transform);
  if (!encoder) {
    GST_WARNING_OBJECT (transform, "No IMFTransform interface available");
    return;
  }

  codec_api = gst_mf_transform_get_codec_api_handle (transform);
  if (!codec_api) {
    GST_WARNING_OBJECT (transform, "No ICodecAPI interface available");
    return;
  }

  g_object_get (transform, "device-name", &device_name, NULL);
  if (!device_name) {
    GST_WARNING_OBJECT (transform, "Unknown device name");
    return;
  }
  g_free (device_name);

  hr = activate->GetAllocatedBlob (MFT_INPUT_TYPES_Attributes,
      (UINT8 **) & infos, &info_size);
  if (!gst_mf_result (hr))
    return;

  for (i = 0; i < info_size / sizeof (MFT_REGISTER_TYPE_INFO); i++) {
    GstVideoFormat format;
    GValue val = G_VALUE_INIT;

    format = gst_mf_video_subtype_to_video_format (&infos[i].guidSubtype);
    if (format == GST_VIDEO_FORMAT_UNKNOWN)
      continue;

    if (!supported_formats) {
      supported_formats = g_new0 (GValue, 1);
      g_value_init (supported_formats, GST_TYPE_LIST);
    }

    switch (format) {
      /* media foundation has duplicated formats IYUV and I420 */
      case GST_VIDEO_FORMAT_I420:
        if (have_I420)
          continue;

        have_I420 = TRUE;
        break;
      case GST_VIDEO_FORMAT_NV12:
        have_NV12 = TRUE;
        break;
      case GST_VIDEO_FORMAT_P010_10LE:
        have_P010 = TRUE;
        break;
      default:
        break;
    }

    g_value_init (&val, G_TYPE_STRING);
    g_value_set_static_string (&val, gst_video_format_to_string (format));
    gst_value_list_append_and_take_value (supported_formats, &val);
  }
  CoTaskMemFree (infos);

  if (!supported_formats) {
    GST_WARNING_OBJECT (transform, "Couldn't figure out supported format");
    return;
  }

  if (IsEqualGUID (MFVideoFormat_H264, subtype)) {
    profile_to_check = h264_profile_map;
    codec_caps_str = h264_caps_str;
  } else if (IsEqualGUID (MFVideoFormat_HEVC, subtype)) {
    profile_to_check = hevc_profile_map;
    codec_caps_str = hevc_caps_str;
  } else if (IsEqualGUID (MFVideoFormat_VP90, subtype)) {
    codec_caps_str = vp9_caps_str;
  } else {
    g_assert_not_reached ();
    return;
  }

  if (profile_to_check) {
    hr = MFCreateMediaType (&out_type);
    if (!gst_mf_result (hr))
      return;

    hr = out_type->SetGUID (MF_MT_MAJOR_TYPE, MFMediaType_Video);
    if (!gst_mf_result (hr))
      return;

    hr = out_type->SetGUID (MF_MT_SUBTYPE, subtype);
    if (!gst_mf_result (hr))
      return;

    hr = out_type->SetUINT32 (MF_MT_AVG_BITRATE, 2048000);
    if (!gst_mf_result (hr))
      return;

    hr = MFSetAttributeRatio (out_type.Get (), MF_MT_FRAME_RATE, 30, 1);
    if (!gst_mf_result (hr))
      return;

    hr = out_type->SetUINT32 (MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    if (!gst_mf_result (hr))
      return;

    hr = MFSetAttributeSize (out_type.Get (), MF_MT_FRAME_SIZE, 1920, 1080);
    if (!gst_mf_result (hr))
      return;

    i = 0;
    do {
      GValue profile_val = G_VALUE_INIT;
      guint mf_profile = profile_to_check[i].profile;
      const gchar *profile_str = profile_to_check[i].profile_str;

      i++;

      if (mf_profile == 0)
        break;

      g_assert (profile_str != NULL);

      hr = out_type->SetUINT32 (MF_MT_MPEG2_PROFILE, mf_profile);
      if (!gst_mf_result (hr))
        return;

      if (!gst_mf_transform_set_output_type (transform, out_type.Get ()))
        continue;

      if (!profiles) {
        profiles = g_new0 (GValue, 1);
        g_value_init (profiles, GST_TYPE_LIST);
      }

      g_value_init (&profile_val, G_TYPE_STRING);
      g_value_set_static_string (&profile_val, profile_str);
      gst_value_list_append_and_take_value (profiles, &profile_val);
    } while (1);

    if (!profiles) {
      GST_WARNING_OBJECT (transform, "Couldn't query supported profile");
      return;
    }
  }

  src_caps = gst_caps_from_string (codec_caps_str);
  if (profiles) {
    gst_caps_set_value (src_caps, "profile", profiles);
    g_value_unset (profiles);
    g_free (profiles);
  }

  sink_caps = gst_caps_new_empty_simple ("video/x-raw");
  /* FIXME: don't hardcode max resolution, but MF doesn't provide
   * API for querying supported max resolution... */
  gst_caps_set_simple (sink_caps,
      "width", GST_TYPE_INT_RANGE, 64, 8192,
      "height", GST_TYPE_INT_RANGE, 64, 8192, NULL);
  gst_caps_set_simple (src_caps,
      "width", GST_TYPE_INT_RANGE, 64, 8192,
      "height", GST_TYPE_INT_RANGE, 64, 8192, NULL);

#if GST_MF_HAVE_D3D11
  /* Check whether this MFT can support D3D11 */
  if (d3d11_device && (have_NV12 || have_P010)) {
    g_object_get (transform, "d3d11-aware", &d3d11_aware, NULL);
    GST_DEBUG_OBJECT (transform, "d3d11 aware %d", d3d11_aware);
  }

  if (d3d11_device && (have_NV12 || have_P010) && d3d11_aware) {
    guint adapter = 0;
    GValue d3d11_formats = G_VALUE_INIT;

    g_object_get (d3d11_device, "adapter", &adapter, NULL);

    d3d11_caps = gst_caps_copy (sink_caps);

    g_value_init (&d3d11_formats, GST_TYPE_LIST);
    if (have_NV12) {
      GValue val = G_VALUE_INIT;
      g_value_init (&val, G_TYPE_STRING);
      g_value_set_static_string (&val, "NV12");
      gst_value_list_append_and_take_value (&d3d11_formats, &val);
    }

    if (have_P010) {
      GValue val = G_VALUE_INIT;
      g_value_init (&val, G_TYPE_STRING);
      g_value_set_static_string (&val, "P010_10LE");
      gst_value_list_append_and_take_value (&d3d11_formats, &val);
    }

    gst_caps_set_value (d3d11_caps, "format", &d3d11_formats);
    g_value_unset (&d3d11_formats);
    gst_caps_set_features_simple (d3d11_caps,
        gst_caps_features_from_string (GST_CAPS_FEATURE_MEMORY_D3D11_MEMORY));
    device_caps->d3d11_aware = TRUE;
    device_caps->adapter = adapter;
  }
#endif

  gst_caps_set_value (sink_caps, "format", supported_formats);
  g_value_unset (supported_formats);
  g_free (supported_formats);

  if (d3d11_caps)
    gst_caps_append (sink_caps, d3d11_caps);

  *sink_template = sink_caps;
  *src_template = src_caps;

#define CHECK_DEVICE_CAPS(codec_obj,api,val) \
  if (SUCCEEDED((codec_obj)->IsSupported(&(api)))) {\
    (device_caps)->val = TRUE; \
  }

  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncCommonRateControlMode, rc_mode);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncCommonQuality, quality);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncAdaptiveMode, adaptive_mode);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncCommonBufferSize, buffer_size);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncCommonMaxBitRate, max_bitrate);
  CHECK_DEVICE_CAPS (codec_api,
      CODECAPI_AVEncCommonQualityVsSpeed, quality_vs_speed);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncH264CABACEnable, cabac);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncH264SPSID, sps_id);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncH264PPSID, pps_id);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncMPVDefaultBPictureCount, bframes);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncMPVGOPSize, gop_size);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncNumWorkerThreads, threads);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncVideoContentType, content_type);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncVideoEncodeQP, qp);
  CHECK_DEVICE_CAPS (codec_api,
      CODECAPI_AVEncVideoForceKeyFrame, force_keyframe);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVLowLatencyMode, low_latency);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncVideoMinQP, min_qp);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncVideoMaxQP, max_qp);
  CHECK_DEVICE_CAPS (codec_api,
      CODECAPI_AVEncVideoEncodeFrameTypeQP, frame_type_qp);
  CHECK_DEVICE_CAPS (codec_api, CODECAPI_AVEncVideoMaxNumRefFrame, max_num_ref);
  if (device_caps->max_num_ref) {
    VARIANT min;
    VARIANT max;
    VARIANT step;

    hr = codec_api->GetParameterRange (&CODECAPI_AVEncVideoMaxNumRefFrame,
        &min, &max, &step);
    if (SUCCEEDED (hr)) {
      device_caps->max_num_ref_high = max.uiVal;
      device_caps->max_num_ref_low = min.uiVal;
      VariantClear (&min);
      VariantClear (&max);
      VariantClear (&step);
    } else {
      device_caps->max_num_ref = FALSE;
    }
  }
#undef CHECK_DEVICE_CAPS

  return;
}

static GstMFTransform *
gst_mf_video_enc_enum (guint enum_flags, GUID * subtype, guint device_index,
    GstMFVideoEncDeviceCaps * device_caps, GstObject * d3d11_device,
    GstCaps ** sink_template, GstCaps ** src_template)
{
  GstMFTransformEnumParams enum_params = { 0, };
  MFT_REGISTER_TYPE_INFO output_type;
  GstMFTransform *transform;
  gint64 adapter_luid = 0;

  *sink_template = NULL;
  *src_template = NULL;
  memset (device_caps, 0, sizeof (GstMFVideoEncDeviceCaps));

  if (!IsEqualGUID (MFVideoFormat_H264, *subtype) &&
      !IsEqualGUID (MFVideoFormat_HEVC, *subtype) &&
      !IsEqualGUID (MFVideoFormat_VP90, *subtype)) {
    GST_ERROR ("Unknown subtype GUID");

    return NULL;
  }

  if (d3d11_device) {
    g_object_get (d3d11_device, "adapter-luid", &adapter_luid, NULL);
    if (!adapter_luid) {
      GST_ERROR ("Couldn't get adapter LUID");
      return NULL;
    }
  }

  output_type.guidMajorType = MFMediaType_Video;
  output_type.guidSubtype = *subtype;

  enum_params.category = MFT_CATEGORY_VIDEO_ENCODER;
  enum_params.output_typeinfo = &output_type;
  enum_params.device_index = device_index;
  enum_params.enum_flags = enum_flags;
  enum_params.adapter_luid = adapter_luid;

  transform = gst_mf_transform_new (&enum_params);
  if (!transform)
    return NULL;

  gst_mf_video_enc_enum_internal (transform, output_type.guidSubtype,
      d3d11_device, device_caps, sink_template, src_template);

  return transform;
}

static void
gst_mf_video_enc_register_internal (GstPlugin * plugin, guint rank,
    GUID * subtype, GTypeInfo * type_info,
    const GstMFVideoEncDeviceCaps * device_caps,
    guint32 enum_flags, guint device_index, GstMFTransform * transform,
    GstCaps * sink_caps, GstCaps * src_caps)
{
  GType type;
  GTypeInfo local_type_info;
  gchar *type_name;
  gchar *feature_name;
  gint i;
  GstMFVideoEncClassData *cdata;
  gboolean is_default = TRUE;
  gchar *device_name = NULL;
  static gchar *type_name_prefix = NULL;
  static gchar *feature_name_prefix = NULL;

  if (IsEqualGUID (MFVideoFormat_H264, *subtype)) {
    type_name_prefix = "H264";
    feature_name_prefix = "h264";
  } else if (IsEqualGUID (MFVideoFormat_HEVC, *subtype)) {
    type_name_prefix = "H265";
    feature_name_prefix = "h265";
  } else if (IsEqualGUID (MFVideoFormat_VP90, *subtype)) {
    type_name_prefix = "VP9";
    feature_name_prefix = "vp9";
  } else {
    g_assert_not_reached ();
    return;
  }

  /* Must be checked already */
  g_object_get (transform, "device-name", &device_name, NULL);
  g_assert (device_name != NULL);

  cdata = g_new0 (GstMFVideoEncClassData, 1);
  cdata->sink_caps = gst_caps_copy (sink_caps);
  cdata->src_caps = gst_caps_copy (src_caps);
  cdata->device_name = device_name;
  cdata->device_caps = *device_caps;
  cdata->enum_flags = enum_flags;
  cdata->device_index = device_index;

  local_type_info = *type_info;
  local_type_info.class_data = cdata;

  GST_MINI_OBJECT_FLAG_SET (cdata->sink_caps,
      GST_MINI_OBJECT_FLAG_MAY_BE_LEAKED);
  GST_MINI_OBJECT_FLAG_SET (cdata->src_caps,
      GST_MINI_OBJECT_FLAG_MAY_BE_LEAKED);

  type_name = g_strdup_printf ("GstMF%sEnc", type_name_prefix);
  feature_name = g_strdup_printf ("mf%senc", feature_name_prefix);

  i = 1;
  while (g_type_from_name (type_name) != 0) {
    g_free (type_name);
    g_free (feature_name);
    type_name = g_strdup_printf ("GstMF%sDevice%dEnc", type_name_prefix, i);
    feature_name = g_strdup_printf ("mf%sdevice%denc", feature_name_prefix, i);
    is_default = FALSE;
    i++;
  }

  cdata->is_default = is_default;

  type =
      g_type_register_static (GST_TYPE_MF_VIDEO_ENC, type_name,
      &local_type_info, (GTypeFlags) 0);

  /* make lower rank than default device */
  if (rank > 0 && !is_default)
    rank--;

  if (!gst_element_register (plugin, feature_name, rank, type))
    GST_WARNING ("Failed to register plugin '%s'", type_name);

  g_free (type_name);
  g_free (feature_name);
}

void
gst_mf_video_enc_register (GstPlugin * plugin, guint rank, GUID * subtype,
    GTypeInfo * type_info, GList * d3d11_device)
{
  GstMFTransform *transform = NULL;
  GstCaps *sink_template = NULL;
  GstCaps *src_template = NULL;
  guint enum_flags;
  GstMFVideoEncDeviceCaps device_caps;
  guint i;

  /* register hardware encoders first */
  enum_flags = (MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_ASYNCMFT |
      MFT_ENUM_FLAG_SORTANDFILTER | MFT_ENUM_FLAG_SORTANDFILTER_APPROVED_ONLY);

  if (d3d11_device) {
    GList *iter;
    for (iter = d3d11_device; iter; iter = g_list_next (iter)) {
      GstObject *device = (GstObject *) iter->data;

      transform = gst_mf_video_enc_enum (enum_flags, subtype, 0, &device_caps,
          device, &sink_template, &src_template);

      /* No more MFT to enumerate */
      if (!transform)
        break;

      /* Failed to open MFT */
      if (!sink_template) {
        gst_clear_object (&transform);
        continue;
      }

      gst_mf_video_enc_register_internal (plugin, rank, subtype, type_info,
          &device_caps, enum_flags, 0, transform, sink_template, src_template);
      gst_clear_object (&transform);
      gst_clear_caps (&sink_template);
      gst_clear_caps (&src_template);
    }
  } else {
    /* AMD seems to be able to support up to 12 GPUs */
    for (i = 0; i < 12; i++) {
      transform = gst_mf_video_enc_enum (enum_flags, subtype, i, &device_caps,
          NULL, &sink_template, &src_template);

      /* No more MFT to enumerate */
      if (!transform)
        break;

      /* Failed to open MFT */
      if (!sink_template) {
        gst_clear_object (&transform);
        continue;
      }

      gst_mf_video_enc_register_internal (plugin, rank, subtype, type_info,
          &device_caps, enum_flags, i, transform, sink_template, src_template);
      gst_clear_object (&transform);
      gst_clear_caps (&sink_template);
      gst_clear_caps (&src_template);
    }
  }

  /* register software encoders */
  enum_flags = (MFT_ENUM_FLAG_SYNCMFT |
      MFT_ENUM_FLAG_SORTANDFILTER | MFT_ENUM_FLAG_SORTANDFILTER_APPROVED_ONLY);

  transform = gst_mf_video_enc_enum (enum_flags, subtype, 0, &device_caps,
      NULL, &sink_template, &src_template);

  if (!transform)
    goto done;

  if (!sink_template)
    goto done;

  gst_mf_video_enc_register_internal (plugin, rank, subtype, type_info,
      &device_caps, enum_flags, 0, transform, sink_template, src_template);

done:
  gst_clear_object (&transform);
  gst_clear_caps (&sink_template);
  gst_clear_caps (&src_template);
}