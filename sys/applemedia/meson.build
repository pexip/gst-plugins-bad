applemedia_sources = [
    'plugin.m',
    'vtutil.c',
    'corevideomemory.c',
    'corevideobuffer.c',
    'coremediabuffer.c',
    'videotexturecache.m',
    'videotexturecache-gl.m',
    'atdec.c',
    'glcontexthelper.c'
]

applemedia_args = [
    '-Dgst_core_media_buffer_new=gst_core_media_buffer_priv_new',
    '-Dgst_core_media_buffer_get_type=gst_core_media_buffer_priv_get_type',
    '-Dgst_core_media_buffer_get_pixel_buffer=gst_core_media_buffer_priv_get_pixel_buffer',
    '-Dgst_core_video_buffer_new=gst_core_video_buffer_priv_new',
    '-Dgst_core_video_buffer_get_type=gst_core_video_buffer_priv_get_type'
]

applemedia_objc_args = []
applemedia_option = get_option('applemedia')
if not ['darwin', 'ios'].contains(host_system) or applemedia_option.disabled()
  subdir_done()
endif

if ['darwin', 'ios'].contains(host_system)
  objc = meson.get_compiler('objc')
#  if not objc.has_argument('-fobjc-arc')
#    error('ARC is required for building')
#  endif

  applemedia_objc_args += ['-fobjc-arc']
endif

applemedia_frameworks = []
corefoundation_dep = dependency('appleframeworks', modules : ['CoreFoundation'], required : applemedia_option)
coremedia_dep = dependency('appleframeworks', modules : ['CoreMedia'], required : applemedia_option)
corevideo_dep = dependency('appleframeworks', modules : ['CoreVideo'], required : applemedia_option)
audiotoolbox_dep = dependency('appleframeworks', modules : ['AudioToolbox'], required : applemedia_option)
applemedia_frameworks += [corefoundation_dep, coremedia_dep, corevideo_dep, audiotoolbox_dep]

avfoundation_dep = dependency('appleframeworks', modules : ['AVFoundation'], required : applemedia_option)
if avfoundation_dep.found()
    applemedia_sources += [
        'avfvideosrc.m',
        'avfassetsrc.m',
        'avsamplevideosink.m',
        'avfdeviceprovider.m',
    ]
    applemedia_frameworks += [avfoundation_dep]
endif

videotoolbox_dep = dependency('appleframeworks', modules : ['VideoToolbox'], required : applemedia_option)
if videotoolbox_dep.found()
    applemedia_sources += [
        'vtenc.c',
        'vtdec.c'
    ]
    applemedia_frameworks += [videotoolbox_dep]
endif

if host_system == 'ios'
    applemedia_sources += [
        'iosassetsrc.m',
        'iosglmemory.c'
    ]
    applemedia_objc_args += ['-fobjc-abi-version=2', '-fobjc-legacy-dispatch']

    ios_media_dep = dependency('appleframeworks', modules : ['Foundation', 'AssetsLibrary'], required : applemedia_option)
    iosurface_dep = dependency('IOSurface', required : applemedia_option)
    applemedia_frameworks += [ios_media_dep, iosurface_dep]
else
    applemedia_sources += [
        'iosurfaceglmemory.c'
    ]
    cocoa_dep = dependency('appleframeworks', modules : ['Cocoa'], required : applemedia_option)
    iosurface_dep = dependency('appleframeworks', modules : ['IOSurface'], required : applemedia_option)
    applemedia_opengl_dep = dependency('appleframeworks', modules : ['OpenGL'], required : applemedia_option)
    applemedia_frameworks += [cocoa_dep, applemedia_opengl_dep, iosurface_dep]
endif

applemedia_found_deps = true
foreach framework : applemedia_frameworks
    if not framework.found()
        applemedia_found_deps = false
    endif
endforeach

if gstvulkan_dep.found() and have_objcpp
  moltenvk_dep = cc.find_library('MoltenVK', required : false)
  metal_dep = dependency('appleframeworks', modules : ['Metal'], required : false)
  if metal_dep.found() and moltenvk_dep.found() and cc.has_header ('MoltenVK/vk_mvk_moltenvk.h')
    applemedia_frameworks += [moltenvk_dep, gstvulkan_dep, metal_dep]
    applemedia_sources += [
      'videotexturecache-vulkan.mm',
      'iosurfacevulkanmemory.c',
    ]
    applemedia_args += ['-DAPPLEMEDIA_MOLTENVK']
  endif
endif

if applemedia_found_deps
    gstapplemedia = library('gstapplemedia',
        applemedia_sources,
        c_args : gst_plugins_bad_args + applemedia_args,
        objc_args : gst_plugins_bad_args + applemedia_args + applemedia_objc_args,
        objcpp_args : gst_plugins_bad_args + applemedia_args + applemedia_objc_args,
        link_args : noseh_link_args,
        include_directories : [configinc, libsinc],
        dependencies : [gstvideo_dep, gstaudio_dep, gstpbutils_dep, gst_dep, gstbase_dep, gstgl_dep] + applemedia_frameworks,
        install : true,
        install_dir : plugins_install_dir,
    )
    pkgconfig.generate(gstapplemedia, install_dir : plugins_pkgconfig_install_dir)
    plugins += [gstapplemedia]
endif
