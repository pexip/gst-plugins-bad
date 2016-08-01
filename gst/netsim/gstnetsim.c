/*
 * GStreamer
 *
 *  Copyright 2006 Collabora Ltd,
 *  Copyright 2006 Nokia Corporation
 *   @author: Philippe Kalaf <philippe.kalaf@collabora.co.uk>.
 *  Copyright 2012-2016 Pexip
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
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstnetsim.h"
#include <string.h>
#include <math.h>
#include <float.h>

GST_DEBUG_CATEGORY (netsim_debug);
#define GST_CAT_DEFAULT (netsim_debug)


typedef enum
{
  DISTRIBUTION_UNIFORM,
  DISTRIBUTION_NORMAL
} GstNetSimDistribution;

static GType
distribution_get_type (void)
{
  static volatile gsize g_define_type_id__volatile = 0;
  if (g_once_init_enter (&g_define_type_id__volatile)) {
    static const GEnumValue values[] = {
      {DISTRIBUTION_UNIFORM, "uniform", "uniform"},
      {DISTRIBUTION_NORMAL, "normal", "normal"},
      {0, NULL, NULL}
    };
    GType g_define_type_id =
        g_enum_register_static ("GstNetSimDistribution", values);
    g_once_init_leave (&g_define_type_id__volatile, g_define_type_id);
  }
  return g_define_type_id__volatile;
}

enum
{
  PROP_0,
  PROP_MIN_DELAY,
  PROP_MAX_DELAY,
  PROP_DELAY_DISTRIBUTION,
  PROP_DELAY_PROBABILITY,
  PROP_DROP_PROBABILITY,
  PROP_DUPLICATE_PROBABILITY,
  PROP_DROP_PACKETS,
  PROP_MAX_KBPS,
  PROP_MAX_BUCKET_SIZE,
};

typedef struct
{
  gboolean generate;
  gdouble z0;
  gdouble z1;
} NormalDistributionState;

struct _GstNetSimPrivate
{
  GstPad *sinkpad, *srcpad;

  GMutex loop_mutex;
  GCond start_cond;
  GMainLoop *main_loop;
  gboolean running;
  GRand *rand_seed;
  gsize bucket_size;
  GstClockTime prev_time;
  NormalDistributionState delay_state;

  /* properties */
  gint min_delay;
  gint max_delay;
  GstNetSimDistribution delay_distribution;
  gfloat delay_probability;
  gfloat drop_probability;
  gfloat duplicate_probability;
  guint drop_packets;
  gint max_kbps;
  gint max_bucket_size;
};

/* these numbers are nothing but wild guesses and dont reflect any reality */
#define DEFAULT_MIN_DELAY 200
#define DEFAULT_MAX_DELAY 400
#define DEFAULT_DELAY_DISTRIBUTION DISTRIBUTION_UNIFORM
#define DEFAULT_DELAY_PROBABILITY 0.0
#define DEFAULT_DROP_PROBABILITY 0.0
#define DEFAULT_DUPLICATE_PROBABILITY 0.0
#define DEFAULT_DROP_PACKETS 0
#define DEFAULT_MAX_KBPS -1
#define DEFAULT_MAX_BUCKET_SIZE -1

#define GST_NET_SIM_GET_PRIVATE(o) \
  (G_TYPE_INSTANCE_GET_PRIVATE ((o), GST_TYPE_NET_SIM, GstNetSimPrivate))

static GstStaticPadTemplate gst_net_sim_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate gst_net_sim_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

G_DEFINE_TYPE (GstNetSim, gst_net_sim, GST_TYPE_ELEMENT);

static void
gst_net_sim_loop (GstNetSim * netsim)
{
  GstNetSimPrivate *priv = netsim->priv;
  GMainLoop *loop;

  GST_TRACE_OBJECT (netsim, "TASK: begin");

  g_mutex_lock (&priv->loop_mutex);
  loop = g_main_loop_ref (priv->main_loop);
  priv->running = TRUE;
  GST_TRACE_OBJECT (netsim, "TASK: signal start");
  g_cond_signal (&priv->start_cond);
  g_mutex_unlock (&priv->loop_mutex);

  GST_TRACE_OBJECT (netsim, "TASK: run");
  g_main_loop_run (loop);
  g_main_loop_unref (loop);

  g_mutex_lock (&priv->loop_mutex);
  GST_TRACE_OBJECT (netsim, "TASK: pause");
  gst_pad_pause_task (priv->srcpad);
  priv->running = FALSE;
  GST_TRACE_OBJECT (netsim, "TASK: signal end");
  g_cond_signal (&priv->start_cond);
  g_mutex_unlock (&priv->loop_mutex);
  GST_TRACE_OBJECT (netsim, "TASK: end");
}

static gboolean
_main_loop_quit_and_remove_source (gpointer user_data)
{
  GMainLoop *main_loop = user_data;
  GST_DEBUG ("MAINLOOP: Quit %p", main_loop);
  g_main_loop_quit (main_loop);
  g_assert (!g_main_loop_is_running (main_loop));
  return FALSE;                 /* Remove source */
}

static gboolean
gst_net_sim_src_activatemode (GstPad * pad, GstObject * parent,
    GstPadMode mode, gboolean active)
{
  GstNetSim *netsim = GST_NET_SIM (parent);
  GstNetSimPrivate *priv = netsim->priv;
  gboolean result = FALSE;
  (void) pad;
  (void) mode;

  g_mutex_lock (&priv->loop_mutex);
  if (active) {
    if (priv->main_loop == NULL) {
      GMainContext *main_context = g_main_context_new ();
      priv->main_loop = g_main_loop_new (main_context, FALSE);
      g_main_context_unref (main_context);

      GST_TRACE_OBJECT (netsim, "ACT: Starting task on srcpad");
      result = gst_pad_start_task (priv->srcpad,
          (GstTaskFunction) gst_net_sim_loop, netsim, NULL);

      GST_TRACE_OBJECT (netsim, "ACT: Wait for task to start");
      g_assert (!priv->running);
      while (!priv->running)
        g_cond_wait (&priv->start_cond, &priv->loop_mutex);
      GST_TRACE_OBJECT (netsim, "ACT: Task on srcpad started");
    }
  } else {
    if (priv->main_loop != NULL) {
      GSource *source;
      guint id;

      /* Adds an Idle Source which quits the main loop from within.
       * This removes the possibility for run/quit race conditions. */
      GST_TRACE_OBJECT (netsim, "DEACT: Stopping main loop on deactivate");
      source = g_idle_source_new ();
      g_source_set_callback (source, _main_loop_quit_and_remove_source,
          g_main_loop_ref (priv->main_loop),
          (GDestroyNotify) g_main_loop_unref);
      id = g_source_attach (source, g_main_loop_get_context (priv->main_loop));
      g_source_unref (source);
      g_assert_cmpuint (id, >, 0);
      g_main_loop_unref (priv->main_loop);
      priv->main_loop = NULL;

      GST_TRACE_OBJECT (netsim, "DEACT: Wait for mainloop and task to pause");
      g_assert (priv->running);
      while (priv->running)
        g_cond_wait (&priv->start_cond, &priv->loop_mutex);

      GST_TRACE_OBJECT (netsim, "DEACT: Stopping task on srcpad");
      result = gst_pad_stop_task (priv->srcpad);
      GST_TRACE_OBJECT (netsim, "DEACT: Mainloop and GstTask stopped");
    }
  }
  g_mutex_unlock (&priv->loop_mutex);

  return result;
}

typedef struct
{
  GstPad *pad;
  GstBuffer *buf;
} PushBufferCtx;

G_INLINE_FUNC PushBufferCtx *
push_buffer_ctx_new (GstPad * pad, GstBuffer * buf)
{
  PushBufferCtx *ctx = g_slice_new (PushBufferCtx);
  ctx->pad = gst_object_ref (pad);
  ctx->buf = gst_buffer_ref (buf);
  return ctx;
}

G_INLINE_FUNC void
push_buffer_ctx_free (PushBufferCtx * ctx)
{
  if (G_LIKELY (ctx != NULL)) {
    gst_buffer_unref (ctx->buf);
    gst_object_unref (ctx->pad);
    g_slice_free (PushBufferCtx, ctx);
  }
}

static gboolean
push_buffer_ctx_push (PushBufferCtx * ctx)
{
  GST_DEBUG_OBJECT (ctx->pad, "Pushing buffer now");
  gst_pad_push (ctx->pad, gst_buffer_ref (ctx->buf));
  return FALSE;
}

static gint
get_random_value_uniform (GRand * rand_seed, gint32 min_value, gint32 max_value)
{
  return g_rand_int_range (rand_seed, min_value, max_value);
}

/* Generate a value from a normal distributation with 95% confidense interval
 * between LOW and HIGH, using the Box-Muller transform. */
static gint
get_random_value_normal (GRand * rand_seed, gint32 low, gint32 high,
    NormalDistributionState * state)
{
  gdouble u1, u2, t1, t2;
  gdouble mu = (high + low) / 2.0;
  gdouble sigma = (high - low) / (2 * 1.96);    /* 95% confidence interval */

  state->generate = !state->generate;

  if (!state->generate)
    return round (state->z1 * sigma + mu);

  do {
    u1 = g_rand_double (rand_seed);
    u2 = g_rand_double (rand_seed);
  } while (u1 <= DBL_EPSILON);

  t1 = sqrt (-2.0 * log (u1));
  t2 = 2.0 * G_PI * u2;
  state->z0 = t1 * cos (t2);
  state->z1 = t1 * sin (t2);

  return round (state->z0 * sigma + mu);
}


static GstFlowReturn
gst_net_sim_delay_buffer (GstNetSim * netsim, GstBuffer * buf)
{
  GstNetSimPrivate *priv = netsim->priv;
  GstFlowReturn ret = GST_FLOW_OK;

  g_mutex_lock (&priv->loop_mutex);
  if (priv->main_loop != NULL && priv->delay_probability > 0 &&
      g_rand_double (priv->rand_seed) < priv->delay_probability) {
    gint delay;
    PushBufferCtx *ctx;
    GSource *source;

    switch (priv->delay_distribution) {
      case DISTRIBUTION_UNIFORM:
        delay = get_random_value_uniform (priv->rand_seed, priv->min_delay,
            priv->max_delay);
        break;
      case DISTRIBUTION_NORMAL:
        delay = get_random_value_normal (priv->rand_seed, priv->min_delay,
            priv->max_delay, &priv->delay_state);
        break;
      default:
        g_assert_not_reached ();
        break;
    }

    if (delay < 0)
      delay = 0;

    ctx = push_buffer_ctx_new (priv->srcpad, buf);
    source = g_timeout_source_new (delay);

    GST_DEBUG_OBJECT (netsim, "Delaying packet by %d", delay);
    g_source_set_callback (source, (GSourceFunc) push_buffer_ctx_push,
        ctx, (GDestroyNotify) push_buffer_ctx_free);
    g_source_attach (source, g_main_loop_get_context (priv->main_loop));
    g_source_unref (source);
  } else {
    ret = gst_pad_push (priv->srcpad, gst_buffer_ref (buf));
  }
  g_mutex_unlock (&priv->loop_mutex);

  return ret;
}

static gsize
get_buffer_size_in_bits (GstBuffer * buf)
{
  GstMapInfo map = GST_MAP_INFO_INIT;
  gsize size;
  gst_buffer_map (buf, &map, GST_MAP_READ);
  size = map.size * 8;
  gst_buffer_unmap (buf, &map);
  return size;
}

static gint
gst_net_sim_get_tokens (GstNetSim * netsim)
{
  GstNetSimPrivate *priv = netsim->priv;
  gint tokens = 0;
  GstClockTimeDiff elapsed_time = 0;
  GstClockTime current_time = 0;
  GstClockTimeDiff token_time;
  GstClock *clock;

  /* check for umlimited kbps and fill up the bucket if that is the case,
     if not, calculate the number of tokens to add based on the elapsed time
   */
  if (priv->max_kbps == -1)
    return priv->max_bucket_size * 1000 - priv->bucket_size;

  /* get the current time */
  clock = gst_element_get_clock (GST_ELEMENT_CAST (netsim));
  if (clock == NULL) {
    GST_WARNING_OBJECT (netsim, "No clock, can't get the time");
  } else {
    current_time = gst_clock_get_time (clock);
  }

  /* get the elapsed time */
  if (GST_CLOCK_TIME_IS_VALID (priv->prev_time)) {
    if (current_time < priv->prev_time) {
      GST_WARNING_OBJECT (netsim, "Clock is going backwards!!");
    } else {
      elapsed_time = GST_CLOCK_DIFF (priv->prev_time, current_time);
    }
  } else {
    priv->prev_time = current_time;
  }

  /* calculate number of tokens and how much time is "spent" by these tokens */
  tokens =
      gst_util_uint64_scale_int (elapsed_time, priv->max_kbps * 1000,
      GST_SECOND);
  token_time =
      gst_util_uint64_scale_int (GST_SECOND, tokens, priv->max_kbps * 1000);

  /* increment the time with how much we spent in terms of whole tokens */
  priv->prev_time += token_time;
  gst_object_unref (clock);
  return tokens;
}

static gboolean
gst_net_sim_token_bucket (GstNetSim * netsim, GstBuffer * buf)
{
  GstNetSimPrivate *priv = netsim->priv;
  gsize buffer_size;
  gint tokens;

  /* with an unlimited bucket-size, we have nothing to do */
  if (priv->max_bucket_size == -1)
    return TRUE;

  buffer_size = get_buffer_size_in_bits (buf);
  tokens = gst_net_sim_get_tokens (netsim);

  priv->bucket_size = MIN (G_MAXINT, priv->bucket_size + tokens);
  GST_LOG_OBJECT (netsim, "Adding %d tokens to bucket (contains %lu tokens)",
      tokens, priv->bucket_size);

  if (priv->max_bucket_size != -1 && priv->bucket_size >
      priv->max_bucket_size * 1000)
    priv->bucket_size = priv->max_bucket_size * 1000;

  if (buffer_size > priv->bucket_size) {
    GST_DEBUG_OBJECT (netsim, "Buffer size (%lu) exeedes bucket size (%lu)",
        buffer_size, priv->bucket_size);
    return FALSE;
  }

  priv->bucket_size -= buffer_size;
  GST_LOG_OBJECT (netsim, "Buffer taking %lu tokens (%lu left)",
      buffer_size, priv->bucket_size);
  return TRUE;
}

static GstFlowReturn
gst_net_sim_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstNetSim *netsim = GST_NET_SIM (parent);
  GstNetSimPrivate *priv = netsim->priv;
  GstFlowReturn ret = GST_FLOW_OK;
  (void) pad;

  if (!gst_net_sim_token_bucket (netsim, buf))
    goto done;

  if (priv->drop_packets > 0) {
    priv->drop_packets--;
    GST_DEBUG_OBJECT (netsim, "Dropping packet (%d left)", priv->drop_packets);
  } else if (priv->drop_probability > 0
      && g_rand_double (priv->rand_seed) < (gdouble) priv->drop_probability) {
    GST_DEBUG_OBJECT (netsim, "Dropping packet");
  } else if (priv->duplicate_probability > 0 &&
      g_rand_double (priv->rand_seed) < (gdouble) priv->duplicate_probability) {
    GST_DEBUG_OBJECT (netsim, "Duplicating packet");
    gst_net_sim_delay_buffer (netsim, buf);
    ret = gst_net_sim_delay_buffer (netsim, buf);
  } else {
    ret = gst_net_sim_delay_buffer (netsim, buf);
  }

done:
  gst_buffer_unref (buf);
  return ret;
}


static void
gst_net_sim_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstNetSim *netsim = GST_NET_SIM (object);
  GstNetSimPrivate *priv = netsim->priv;

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    case PROP_MIN_DELAY:
      priv->min_delay = g_value_get_int (value);
      break;
    case PROP_MAX_DELAY:
      priv->max_delay = g_value_get_int (value);
      break;
    case PROP_DELAY_DISTRIBUTION:
      priv->delay_distribution = g_value_get_enum (value);
      break;
    case PROP_DELAY_PROBABILITY:
      priv->delay_probability = g_value_get_float (value);
      break;
    case PROP_DROP_PROBABILITY:
      priv->drop_probability = g_value_get_float (value);
      break;
    case PROP_DUPLICATE_PROBABILITY:
      priv->duplicate_probability = g_value_get_float (value);
      break;
    case PROP_DROP_PACKETS:
      priv->drop_packets = g_value_get_uint (value);
      break;
    case PROP_MAX_KBPS:
      priv->max_kbps = g_value_get_int (value);
      break;
    case PROP_MAX_BUCKET_SIZE:
      priv->max_bucket_size = g_value_get_int (value);
      if (priv->max_bucket_size != -1)
        priv->bucket_size = priv->max_bucket_size * 1000;
      break;
  }
}

static void
gst_net_sim_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstNetSim *netsim = GST_NET_SIM (object);
  GstNetSimPrivate *priv = netsim->priv;

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    case PROP_MIN_DELAY:
      g_value_set_int (value, priv->min_delay);
      break;
    case PROP_MAX_DELAY:
      g_value_set_int (value, priv->max_delay);
      break;
    case PROP_DELAY_DISTRIBUTION:
      g_value_set_enum (value, priv->delay_distribution);
      break;
    case PROP_DELAY_PROBABILITY:
      g_value_set_float (value, priv->delay_probability);
      break;
    case PROP_DROP_PROBABILITY:
      g_value_set_float (value, priv->drop_probability);
      break;
    case PROP_DUPLICATE_PROBABILITY:
      g_value_set_float (value, priv->duplicate_probability);
      break;
    case PROP_DROP_PACKETS:
      g_value_set_uint (value, priv->drop_packets);
      break;
    case PROP_MAX_KBPS:
      g_value_set_int (value, priv->max_kbps);
      break;
    case PROP_MAX_BUCKET_SIZE:
      g_value_set_int (value, priv->max_bucket_size);
      break;
  }
}

static void
gst_net_sim_init (GstNetSim * netsim)
{
  GstNetSimPrivate *priv;
  priv = netsim->priv = GST_NET_SIM_GET_PRIVATE (netsim);

  priv->srcpad =
      gst_pad_new_from_static_template (&gst_net_sim_src_template, "src");
  priv->sinkpad =
      gst_pad_new_from_static_template (&gst_net_sim_sink_template, "sink");

  gst_element_add_pad (GST_ELEMENT (netsim), priv->srcpad);
  gst_element_add_pad (GST_ELEMENT (netsim), priv->sinkpad);

  g_mutex_init (&priv->loop_mutex);
  g_cond_init (&priv->start_cond);
  priv->rand_seed = g_rand_new ();
  priv->main_loop = NULL;
  priv->prev_time = GST_CLOCK_TIME_NONE;

  GST_OBJECT_FLAG_SET (priv->sinkpad,
      GST_PAD_FLAG_PROXY_CAPS | GST_PAD_FLAG_PROXY_ALLOCATION);

  gst_pad_set_chain_function (priv->sinkpad,
      GST_DEBUG_FUNCPTR (gst_net_sim_chain));
  gst_pad_set_activatemode_function (priv->srcpad,
      GST_DEBUG_FUNCPTR (gst_net_sim_src_activatemode));
}

static void
gst_net_sim_finalize (GObject * object)
{
  GstNetSim *netsim = GST_NET_SIM (object);
  GstNetSimPrivate *priv = netsim->priv;

  g_rand_free (priv->rand_seed);
  g_mutex_clear (&priv->loop_mutex);
  g_cond_clear (&priv->start_cond);

  G_OBJECT_CLASS (gst_net_sim_parent_class)->finalize (object);
}

static void
gst_net_sim_dispose (GObject * object)
{
  GstNetSim *netsim = GST_NET_SIM (object);

  g_assert (netsim->priv->main_loop == NULL);

  G_OBJECT_CLASS (gst_net_sim_parent_class)->dispose (object);
}

static void
gst_net_sim_class_init (GstNetSimClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);

  g_type_class_add_private (klass, sizeof (GstNetSimPrivate));

  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_net_sim_src_template);
  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_net_sim_sink_template);

  gst_element_class_set_metadata (gstelement_class,
      "Network Simulator",
      "Filter/Network",
      "An element that simulates network jitter, "
      "packet loss and packet duplication",
      "Philippe Kalaf <philippe.kalaf@collabora.co.uk>, "
      "Havard Graff <havard@pexip.com>");

  gobject_class->dispose = GST_DEBUG_FUNCPTR (gst_net_sim_dispose);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_net_sim_finalize);

  gobject_class->set_property = gst_net_sim_set_property;
  gobject_class->get_property = gst_net_sim_get_property;

  g_object_class_install_property (gobject_class, PROP_MIN_DELAY,
      g_param_spec_int ("min-delay", "Minimum delay (ms)",
          "The minimum delay in ms to apply to buffers",
          G_MININT, G_MAXINT, DEFAULT_MIN_DELAY,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MAX_DELAY,
      g_param_spec_int ("max-delay", "Maximum delay (ms)",
          "The maximum delay in ms to apply to buffers",
          G_MININT, G_MAXINT, DEFAULT_MAX_DELAY,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DELAY_DISTRIBUTION,
      g_param_spec_enum ("delay-distribution", "Delay GstNetSimDistribution",
          "Distribution for the amount of delay",
          distribution_get_type (), DEFAULT_DELAY_DISTRIBUTION,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DELAY_PROBABILITY,
      g_param_spec_float ("delay-probability", "Delay Probability",
          "The Probability a buffer is delayed",
          0.0, 1.0, DEFAULT_DELAY_PROBABILITY,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DROP_PROBABILITY,
      g_param_spec_float ("drop-probability", "Drop Probability",
          "The Probability a buffer is dropped",
          0.0, 1.0, DEFAULT_DROP_PROBABILITY,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DUPLICATE_PROBABILITY,
      g_param_spec_float ("duplicate-probability", "Duplicate Probability",
          "The Probability a buffer is duplicated",
          0.0, 1.0, DEFAULT_DUPLICATE_PROBABILITY,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DROP_PACKETS,
      g_param_spec_uint ("drop-packets", "Drop Packets",
          "Drop the next n packets",
          0, G_MAXUINT, DEFAULT_DROP_PACKETS,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MAX_KBPS,
      g_param_spec_int ("max-kbps", "Maximum Kbps",
          "The maximum number of kilobits to let through per second "
          "(-1 = unlimited)", -1, G_MAXINT, DEFAULT_MAX_KBPS,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MAX_BUCKET_SIZE,
      g_param_spec_int ("max-bucket-size", "Maximum Bucket Size (Kb)",
          "The size of the token bucket, related to burstiness resilience "
          "(-1 = unlimited)", -1, G_MAXINT, DEFAULT_MAX_BUCKET_SIZE,
          G_PARAM_READWRITE | G_PARAM_CONSTRUCT | G_PARAM_STATIC_STRINGS));

  GST_DEBUG_CATEGORY_INIT (netsim_debug, "netsim", 0, "Network simulator");
}

static gboolean
gst_net_sim_plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "netsim",
      GST_RANK_MARGINAL, GST_TYPE_NET_SIM);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    netsim,
    "Network Simulator",
    gst_net_sim_plugin_init, PACKAGE_VERSION, "LGPL", GST_PACKAGE_NAME,
    GST_PACKAGE_ORIGIN)
