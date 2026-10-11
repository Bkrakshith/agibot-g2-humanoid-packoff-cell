"""Compressed image transports: compressed (JPEG/PNG), compressedDepth (PNG), zstd.

JPEG needs cv2 or Pillow (skipped in .venv-tools, which has neither; run this file with
.venv-learn too). PNG has a pure numpy fallback, so compressedDepth works everywhere.
"""
import io
import struct
import zlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from g2cell_recorder import image_codec as ic


# ---------------------------------------------------------------- test PNG encoder

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def _filter_row(ftype, row, prev, bpp):
    out = bytearray(len(row))
    for i, x in enumerate(row):
        a = row[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        pred = (0, a, b, (a + b) // 2, _paeth(a, b, c))[ftype]
        out[i] = (x - pred) & 0xFF
    return bytes(out)


def encode_png(arr: np.ndarray, filters=(0, 1, 2, 3, 4)) -> bytes:
    """Minimal PNG writer that cycles through all five filter types row by row."""
    arr = np.asarray(arr)
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    depth = 16 if arr.dtype == np.uint16 else 8
    ctype = {1: 0, 3: 2, 4: 6}[ch]
    bpp = ch * depth // 8
    raw = arr.astype(">u2" if depth == 16 else np.uint8).tobytes()
    stride = w * bpp
    prev = bytes(stride)
    body = bytearray()
    for y in range(h):
        row = raw[y * stride:(y + 1) * stride]
        f = filters[y % len(filters)]
        body += bytes([f]) + _filter_row(f, row, prev, bpp)
        prev = row

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(body)))
            + chunk(b"IEND", b""))


def compressed_depth_payload(depth_m: np.ndarray, z0=100.0, depth_max=10.0) -> bytes:
    """What compressed_depth_image_transport publishes for a 32FC1 image (png mode)."""
    a = z0 * (z0 + 1.0)
    b = 1.0 - a / depth_max
    ok = np.isfinite(depth_m) & (depth_m > 0) & (depth_m < depth_max)
    inv = np.zeros(depth_m.shape, np.uint16)
    inv[ok] = (a / depth_m[ok] + b).astype(np.uint16)
    return struct.pack("<iff", 0, a, b) + encode_png(inv)


RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- PNG

@pytest.mark.parametrize("shape,dtype", [((7, 9, 3), np.uint8), ((5, 6, 4), np.uint8),
                                         ((6, 5), np.uint16), ((4, 3), np.uint8)])
def test_numpy_png_decoder_handles_all_filters(shape, dtype):
    hi = 65535 if dtype == np.uint16 else 255
    arr = RNG.integers(0, hi + 1, size=shape).astype(dtype)
    out = ic.png_decode_numpy(encode_png(arr))
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, arr)


def test_png_decoder_rejects_interlaced():
    png = bytearray(encode_png(np.zeros((2, 2), np.uint8)))
    png[8 + 8 + 12] = 1  # IHDR interlace byte
    with pytest.raises(ValueError, match="interlac"):
        ic.png_decode_numpy(bytes(png))


# ---------------------------------------------------------------- compressedDepth

def test_compressed_depth_32fc1_decodes_to_metres_within_quantisation():
    d = RNG.uniform(0.3, 2.8, size=(8, 10)).astype(np.float32)
    d[0, 0] = np.nan          # invalid -> stays NaN
    msg = NS(format="32FC1; compressedDepth png", data=compressed_depth_payload(d))
    out = ic.decode_compressed(msg, depth=True)
    assert out.dtype == np.float32 and out.shape == d.shape
    assert np.isnan(out[0, 0])
    ok = np.isfinite(d)
    # inverse-depth quantisation: |err| <= z^2 / (z0 (z0+1)) ~ 0.8 mm at 2.8 m
    np.testing.assert_allclose(out[ok], d[ok], atol=1e-3)


def test_compressed_depth_16uc1_is_millimetres():
    mm = np.array([[0, 1500], [250, 65000]], np.uint16)
    msg = NS(format="16UC1; compressedDepth png",
             data=struct.pack("<iff", 0, 0.0, 0.0) + encode_png(mm))
    out = ic.decode_compressed(msg, depth=True)
    np.testing.assert_allclose(out, mm.astype(np.float32) * 1e-3)


def test_compressed_depth_rvl_is_reported_clearly():
    msg = NS(format="32FC1; compressedDepth rvl", data=b"\0" * 20)
    with pytest.raises(NotImplementedError, match="rvl"):
        ic.decode_compressed(msg, depth=True)


# ---------------------------------------------------------------- compressed (JPEG / PNG)

def test_compressed_png_rgb_and_rgba_drop_alpha():
    rgba = RNG.integers(0, 256, size=(6, 8, 4)).astype(np.uint8)
    out = ic.decode_compressed(NS(format="rgba8; png compressed bgra8", data=encode_png(rgba)))
    np.testing.assert_array_equal(out, rgba[..., :3])
    rgb = rgba[..., :3].copy()
    out = ic.decode_compressed(NS(format="png", data=encode_png(rgb)))
    np.testing.assert_array_equal(out, rgb)


def test_compressed_png_mono_is_replicated():
    g = RNG.integers(0, 256, size=(3, 4)).astype(np.uint8)
    out = ic.decode_compressed(NS(format="mono8; png compressed mono8", data=encode_png(g)))
    np.testing.assert_array_equal(out, np.repeat(g[..., None], 3, axis=2))


def test_compressed_jpeg_decodes_to_rgb():
    PIL = pytest.importorskip("PIL.Image")
    img = np.zeros((32, 48, 3), np.uint8)
    img[:, :16] = (250, 10, 10)     # red block: catches a BGR/RGB swap
    img[:, 16:32] = (10, 250, 10)
    img[:, 32:] = (10, 10, 250)
    buf = io.BytesIO()
    PIL.fromarray(img).save(buf, "JPEG", quality=95)
    out = ic.decode_compressed(NS(format="rgba8; jpeg compressed bgr8", data=buf.getvalue()))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img).mean() < 8      # chroma subsampling at block edges
    assert [int(np.argmax(out[16, x])) for x in (8, 24, 40)] == [0, 1, 2]


def test_jpeg_without_decoder_names_the_fix(monkeypatch):
    monkeypatch.setattr(ic, "_BACKENDS", ())
    jpeg = b"\xff\xd8\xff\xe0" + b"\0" * 16
    with pytest.raises(ImportError, match="opencv|Pillow"):
        ic.decode_compressed(NS(format="jpeg", data=jpeg))


def test_unknown_compressed_payload_raises():
    with pytest.raises(ValueError, match="format"):
        ic.decode_compressed(NS(format="theora", data=b"abcdefgh"))


# ---------------------------------------------------------------- zstd transport (CDR Image)

IMAGE_DEF = """std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def _cdr_image(h, w, enc, data, frame_id="cam"):
    """Serialise sensor_msgs/Image with mcap_ros2 (independent of image_codec's parser)."""
    pytest.importorskip("mcap_ros2")
    from mcap_ros2._dynamic import serialize_dynamic
    enc_fn = serialize_dynamic("sensor_msgs/msg/Image", IMAGE_DEF)["sensor_msgs/msg/Image"]
    return enc_fn({"header": {"stamp": {"sec": 3, "nanosec": 7}, "frame_id": frame_id},
                   "height": h, "width": w, "encoding": enc, "is_bigendian": 0,
                   "step": len(data) // h, "data": data})


@pytest.mark.parametrize("frame_id", ["", "c", "camera_link"])   # string padding cases
def test_parse_image_cdr(frame_id):
    rgba = RNG.integers(0, 256, size=(3, 5, 4)).astype(np.uint8)
    msg = ic.parse_image_cdr(_cdr_image(3, 5, "rgba8", rgba.tobytes(), frame_id))
    assert (msg.height, msg.width, msg.encoding, msg.step) == (3, 5, "rgba8", 20)
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (3, 7)
    assert bytes(msg.data) == rgba.tobytes()


def test_zstd_transport_rgb_and_depth():
    zstd = pytest.importorskip("zstandard")
    rgba = RNG.integers(0, 256, size=(4, 6, 4)).astype(np.uint8)
    blob = zstd.ZstdCompressor(level=1).compress(_cdr_image(4, 6, "rgba8", rgba.tobytes()))
    out = ic.decode_compressed(NS(format="zstd", data=blob))
    np.testing.assert_array_equal(out, rgba[..., :3])
    d = RNG.uniform(0.2, 3.0, size=(4, 6)).astype(np.float32)
    blob = zstd.ZstdCompressor(level=1).compress(_cdr_image(4, 6, "32FC1", d.tobytes()))
    np.testing.assert_array_equal(ic.decode_compressed(NS(format="zstd", data=blob), depth=True), d)


# ---------------------------------------------------------------- payloads (picklable work items)

def test_payload_roundtrip_raw_and_compressed():
    rgba = RNG.integers(0, 256, size=(2, 3, 4)).astype(np.uint8)
    raw = NS(height=2, width=3, encoding="rgba8", step=12, data=memoryview(rgba.tobytes()))
    p = ic.frame_payload(raw, depth=False)
    assert isinstance(p["data"], bytes)
    np.testing.assert_array_equal(ic.decode_payload(p), rgba[..., :3])
    assert ic.transport_of(raw) == "raw"

    comp = NS(format="png", data=encode_png(rgba[..., :3].copy()))
    p = ic.frame_payload(comp, depth=False)
    np.testing.assert_array_equal(ic.decode_payload(p), rgba[..., :3])
    assert ic.transport_of(comp) == "png"
    assert ic.transport_of(NS(format="32FC1; compressedDepth png", data=b"")) == "compressedDepth"
    assert ic.transport_of(NS(format="rgba8; jpeg compressed bgr8", data=b"")) == "jpeg"
