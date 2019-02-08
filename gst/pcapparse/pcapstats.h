#ifndef __GST_PCAP_STATS_H__
#define __GST_PCAP_STATS_H__

#include <gst/gst.h>

typedef struct _PcapStats PcapStats;

PcapStats * pcap_stats_new (const gchar * id_str,
    const gchar * src_ip, guint src_port,
    const char * dst_ip, guint dst_port);
void pcap_stats_free (PcapStats * stats);

void pcap_stats_update (PcapStats * stats,
    GstClockTime cur_ts, gint payload_size);
void pcap_stats_update_rtcp (PcapStats * stats,
    guint32 ssrc, gint payload_size);
void pcap_stats_update_rtp (PcapStats * stats,
    guint32 ssrc, guint payload_type, gint payload_size);
guint pcap_stats_count (PcapStats * stats);
GstStructure * pcap_stats_nth_to_structure (PcapStats * stats,
    guint index);


#endif /* __GST_PCAP_STATS_H__ */
