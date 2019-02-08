#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#define GLIB_DISABLE_DEPRECATION_WARNINGS

#include "pcapstats.h"

typedef struct {
  GstClockTime first_ts;
  guint packets;
  guint bytes;
} StreamStats;

typedef struct {
  guint32 ssrc;
  gint payload_type;
  StreamStats rtp_stats;
  StreamStats rtcp_stats;
} SSRCStats;

struct _PcapStats {
  gchar * id_str;
  gchar * src_ip;
  guint src_port;
  gchar * dst_ip;
  guint dst_port;

  GstClockTime cur_ts;
  StreamStats stream_stats;

  GHashTable * ssrc_to_stats;
};

PcapStats *
pcap_stats_new (const gchar * id_str,
    const gchar * src_ip, guint src_port,
    const char * dst_ip, guint dst_port)
{
  PcapStats * stats = g_new0 (PcapStats, 1);

  stats->id_str = g_strdup (id_str);
  stats->src_ip = g_strdup (src_ip);
  stats->src_port = src_port;
  stats->dst_ip = g_strdup (dst_ip);
  stats->dst_port = dst_port;

  stats->cur_ts = GST_CLOCK_TIME_NONE;
  stats->stream_stats.first_ts = GST_CLOCK_TIME_NONE;

  stats->ssrc_to_stats = g_hash_table_new_full (NULL, NULL, NULL,
      (GDestroyNotify) g_free);

  return stats;
}

void
pcap_stats_free (PcapStats * stats)
{
  g_hash_table_destroy (stats->ssrc_to_stats);
  g_free (stats->dst_ip);
  g_free (stats->src_ip);
  g_free (stats->id_str);
  g_free (stats);
}

void
pcap_stats_update (PcapStats * stats,
    GstClockTime cur_ts, gint payload_size)
{
  stats->cur_ts = cur_ts;
  if (stats->stream_stats.first_ts == GST_CLOCK_TIME_NONE)
    stats->stream_stats.first_ts = cur_ts;

  stats->stream_stats.packets++;
  stats->stream_stats.bytes += payload_size;
}

static SSRCStats *
ssrc_stats_new (guint32 ssrc)
{
  SSRCStats * ssrc_stats = g_new0 (SSRCStats, 1);

  ssrc_stats->ssrc = ssrc;
  ssrc_stats->rtp_stats.first_ts = GST_CLOCK_TIME_NONE;
  ssrc_stats->rtcp_stats.first_ts = GST_CLOCK_TIME_NONE;

  return ssrc_stats;
}

void
pcap_stats_update_rtcp (PcapStats * stats,
    guint32 ssrc, gint payload_size)
{
  SSRCStats * ssrc_stats = g_hash_table_lookup (stats->ssrc_to_stats,
    GUINT_TO_POINTER (ssrc));
  if (ssrc_stats == NULL) {
    ssrc_stats = ssrc_stats_new (ssrc);
    g_hash_table_insert (stats->ssrc_to_stats, GUINT_TO_POINTER (ssrc), ssrc_stats);
  }

  if (ssrc_stats->rtcp_stats.first_ts == GST_CLOCK_TIME_NONE)
    ssrc_stats->rtcp_stats.first_ts = stats->cur_ts;
  ssrc_stats->rtcp_stats.packets++;
  ssrc_stats->rtcp_stats.bytes += payload_size;
}

void
pcap_stats_update_rtp (PcapStats * stats,
    guint32 ssrc, guint payload_type, gint payload_size)
{
  SSRCStats * ssrc_stats = g_hash_table_lookup (stats->ssrc_to_stats,
    GUINT_TO_POINTER (ssrc));
  if (ssrc_stats == NULL) {
    ssrc_stats = ssrc_stats_new (ssrc);
    g_hash_table_insert (stats->ssrc_to_stats, GUINT_TO_POINTER (ssrc), ssrc_stats);
  }

  if (ssrc_stats->rtp_stats.first_ts == GST_CLOCK_TIME_NONE)
    ssrc_stats->rtp_stats.first_ts = stats->cur_ts;
  /* Assume that an SSRC doesn't change PT */
  ssrc_stats->payload_type = payload_type;
  ssrc_stats->rtp_stats.packets++;
  ssrc_stats->rtp_stats.bytes += payload_size;
}

guint
pcap_stats_count (PcapStats * stats)
{
  GList * ssrc_stats_list, * walk;
  guint len, i, count = 0;

  len = g_hash_table_size (stats->ssrc_to_stats);
  if (len <= 1) {
    /* Backwards compatibility for simple pcaps: everything in one blob */
    return 1;
  }

  walk = ssrc_stats_list = g_hash_table_get_values (stats->ssrc_to_stats);
  for (i = 0; i < len; i++) {
    SSRCStats * ssrc_stats = walk->data;

    if (ssrc_stats->rtcp_stats.first_ts != GST_CLOCK_TIME_NONE)
      count++;
    if (ssrc_stats->rtp_stats.first_ts != GST_CLOCK_TIME_NONE)
      count++;

    walk = walk->next;
  }
  g_list_free (ssrc_stats_list);

  return count;
}

GstStructure *
pcap_stats_nth_to_structure (PcapStats * stats, guint index)
{
  GList * ssrc_stats_list, * walk;
  guint len;
  GstStructure * s;

  g_assert (index < pcap_stats_count (stats));

  s = gst_structure_new ("stats",
      "id-str", G_TYPE_STRING, stats->id_str,
      "src-ip", G_TYPE_STRING, stats->src_ip,
      "src-port", G_TYPE_INT, stats->src_port,
      "dst-ip", G_TYPE_STRING, stats->dst_ip,
      "dst-port", G_TYPE_INT, stats->dst_port,
      /* Default these to the transport-wide stats */
      "first-ts", G_TYPE_UINT64, stats->stream_stats.first_ts,
      "packets", G_TYPE_INT, stats->stream_stats.packets,
      "bytes", G_TYPE_INT, stats->stream_stats.bytes,
      NULL);

  len = g_hash_table_size (stats->ssrc_to_stats);
  walk = ssrc_stats_list = g_hash_table_get_values (stats->ssrc_to_stats);
  if (len == 0) {
    /* No RT(C)P stats */
  } else if (len == 1) {
    /* Backwards compatibility for simple pcaps: everything in one blob */
    SSRCStats * ssrc_stats = walk->data;

    if (ssrc_stats->rtcp_stats.first_ts != GST_CLOCK_TIME_NONE)
      gst_structure_set (s,
          "has-rtcp", G_TYPE_BOOLEAN, TRUE,
          "ssrc", G_TYPE_UINT, ssrc_stats->ssrc,
          NULL);

    if (ssrc_stats->rtp_stats.first_ts != GST_CLOCK_TIME_NONE)
      gst_structure_set (s,
          "has-rtp", G_TYPE_BOOLEAN, TRUE,
          "payload-type", G_TYPE_INT, ssrc_stats->payload_type,
          "ssrc", G_TYPE_UINT, ssrc_stats->ssrc,
          NULL);
  } else {
    while (1) {
      SSRCStats * ssrc_stats = walk->data;

      if (ssrc_stats->rtcp_stats.first_ts != GST_CLOCK_TIME_NONE) {
        if (index == 0) {
          gst_structure_set (s,
              "has-rtcp", G_TYPE_BOOLEAN, TRUE,
              "ssrc", G_TYPE_UINT, ssrc_stats->ssrc,
              "first-ts", G_TYPE_UINT64, ssrc_stats->rtcp_stats.first_ts,
              "packets", G_TYPE_INT, ssrc_stats->rtcp_stats.packets,
              "bytes", G_TYPE_INT, ssrc_stats->rtcp_stats.bytes,
              NULL);
          break;
	}
        index--;
      }

      if (ssrc_stats->rtp_stats.first_ts != GST_CLOCK_TIME_NONE) {
        if (index == 0) {
          gst_structure_set (s,
              "has-rtp", G_TYPE_BOOLEAN, TRUE,
              "payload-type", G_TYPE_INT, ssrc_stats->payload_type,
              "ssrc", G_TYPE_UINT, ssrc_stats->ssrc,
              "first-ts", G_TYPE_UINT64, ssrc_stats->rtp_stats.first_ts,
              "packets", G_TYPE_INT, ssrc_stats->rtp_stats.packets,
              "bytes", G_TYPE_INT, ssrc_stats->rtp_stats.bytes,
              NULL);
          break;
        }
        index--;
      }

      walk = walk->next;
    }
  }

  g_list_free (ssrc_stats_list);

  return s;
}
