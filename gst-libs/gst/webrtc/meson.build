webrtc_sources = [
  'dtlstransport.c',
  'icetransport.c',
  'rtcsessiondescription.c',
  'rtpreceiver.c',
  'rtpsender.c',
  'rtptransceiver.c',
  'datachannel.c',
  'sctptransport.c',
]

webrtc_headers = [
  'dtlstransport.h',
  'icetransport.h',
  'rtcsessiondescription.h',
  'rtpreceiver.h',
  'rtpsender.h',
  'rtptransceiver.h',
  'datachannel.h',
  'webrtc_fwd.h',
  'webrtc.h',
  'sctptransport.h',
]

webrtc_enumtypes_headers = [
  'dtlstransport.h',
  'icetransport.h',
  'rtptransceiver.h',
  'webrtc_fwd.h',
]

webrtc_enums = gnome.mkenums_simple('webrtc-enumtypes',
  sources : webrtc_enumtypes_headers,
  body_prefix : '#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif',
  header_prefix : '#include <gst/webrtc/webrtc_fwd.h>',
  decorator: 'GST_WEBRTC_API',
  install_header: true,
  install_dir : join_paths(get_option('includedir'), 'gstreamer-1.0/gst/webrtc/'))

gstwebrtc_c = webrtc_enums[0]
gstwebrtc_h = webrtc_enums[1]

webrtc_gen_sources = [gstwebrtc_h]

gstwebrtc_dependencies = [gstbase_dep, gstsdp_dep]

gstwebrtc = library('gstwebrtc-' + api_version,
  webrtc_sources, gstwebrtc_c, gstwebrtc_h,
  c_args : gst_plugins_bad_args + ['-DGST_USE_UNSTABLE_API', '-DBUILDING_GST_WEBRTC'],
  include_directories : [configinc, libsinc],
  version : libversion,
  soversion : soversion,
  darwin_versions : osxversion,
  install : true,
  dependencies : gstwebrtc_dependencies,
)

pkgconfig.generate(gstwebrtc,
  libraries : [gst_dep, gstbase_dep],
  variables : pkgconfig_variables,
  subdirs : pkgconfig_subdirs,
  name : 'gstreamer-webrtc-1.0',
  description : 'GStreamer WebRTC support',
)

if build_gir
  webrtc_gir = gnome.generate_gir(gstwebrtc,
    sources : webrtc_sources + webrtc_headers + [gstwebrtc_h],
    namespace : 'GstWebRTC',
    nsversion : api_version,
    identifier_prefix : 'Gst',
    symbol_prefix : 'gst',
    export_packages : 'gstreamer-webrtc-1.0',
    includes : ['Gst-1.0', 'GstSdp-1.0'],
    install : true,
    extra_args : gir_init_section + ['-DGST_USE_UNSTABLE_API'] + ['--c-include=gst/webrtc/webrtc.h'],
    dependencies : [gstbase_dep, gstsdp_dep]
  )
  webrtc_gen_sources += webrtc_gir
endif

install_headers(webrtc_headers, subdir : 'gstreamer-1.0/gst/webrtc')

gstwebrtc_dep = declare_dependency(link_with: gstwebrtc,
  include_directories : libsinc,
  sources: webrtc_gen_sources,
  dependencies: gstwebrtc_dependencies)

meson.override_dependency('gstreamer-webrtc-1.0', gstwebrtc_dep)
