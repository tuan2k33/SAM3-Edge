"""Fake RTSP cameras from one H264 mp4, for the DeepStream pipelines: rtsp://127.0.0.1:8554/cam0 .. cam{N-1}. Uses the system's libgstrtspserver
through ctypes (no dev package, no gir, no download needed). Each client gets the file from its start; make the file long enough:

    ffmpeg -stream_loop 8 -i test/data/pedestrians.mp4 -c copy test/data/pedestrians_2min.mp4
    python3 test/rtsp_serve.py test/data/pedestrians_2min.mp4 2
"""
import ctypes, os, sys

path, n = os.path.abspath(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2
gst, glib, rtsp = (ctypes.CDLL(f"lib{l}.so.0") for l in ("gstreamer-1.0", "glib-2.0", "gstrtspserver-1.0"))
for f in (rtsp.gst_rtsp_server_new, rtsp.gst_rtsp_server_get_mount_points, rtsp.gst_rtsp_media_factory_new, glib.g_main_loop_new):
    f.restype = ctypes.c_void_p
P = ctypes.c_void_p
gst.gst_init(None, None)
server = P(rtsp.gst_rtsp_server_new())
mounts = P(rtsp.gst_rtsp_server_get_mount_points(server))
for i in range(n):
    factory = P(rtsp.gst_rtsp_media_factory_new())
    rtsp.gst_rtsp_media_factory_set_launch(factory, f"( filesrc location={path} ! qtdemux ! h264parse config-interval=1 ! rtph264pay name=pay0 pt=96 )".encode())
    rtsp.gst_rtsp_mount_points_add_factory(mounts, f"/cam{i}".encode(), factory)
assert rtsp.gst_rtsp_server_attach(server, None), "port 8554 is busy"
print(f"serving rtsp://127.0.0.1:8554/cam0..cam{n - 1} from {path}", flush=True)
glib.g_main_loop_run(P(glib.g_main_loop_new(None, 0)))
