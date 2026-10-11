"""Decode image_transport payloads into the arrays episode.py stores.

Transports handled (all published by Genie Sim's render node through image_transport):

  raw              sensor_msgs/Image                               .../image_raw
  jpeg / png       sensor_msgs/CompressedImage, compressed plugin  .../image_raw/compressed
                   (format e.g. "rgba8; jpeg compressed bgr8")
  compressedDepth  CompressedImage, compressed_depth plugin       .../image_raw/compressedDepth
                   (format "32FC1; compressedDepth png": 12-byte ConfigHeader + 16-bit PNG of
                   quantised inverse depth; "16UC1; ..." stores millimetres directly)
  zstd             CompressedImage, zstd plugin                    .../image_raw/zstd
                   (zstd frame around a CDR-serialised sensor_msgs/Image)

Decoders: cv2, else Pillow. PNG also has a pure-numpy fallback (slow, ~0.5 s per 640x400
frame) so compressedDepth converts in any venv; JPEG needs cv2 or Pillow.

Work items are plain dicts of bytes/ints (`frame_payload`) so they can be sent to worker
processes; `decode_payload` turns one into uint8 [H,W,3] RGB or float32 [H,W] metres.
"""
from __future__ import annotations

import struct
import zlib
from types import SimpleNamespace as NS

import numpy as np


def _available_backends() -> tuple[str, ...]:
    out = []
    try:
        import cv2  # noqa: F401
        out.append("cv2")
    except ImportError:
        pass
    try:
        import PIL.Image  # noqa: F401
        out.append("PIL")
    except ImportError:
        pass
    return tuple(out)


_BACKENDS = _available_backends()

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8"


# ---------------------------------------------------------------- raw payloads

_RGB_ENCODINGS = {  # encoding -> (channels, channel order to take for R,G,B)
    "rgba8": (4, [0, 1, 2]),
    "rgb8": (3, [0, 1, 2]),
    "bgra8": (4, [2, 1, 0]),
    "bgr8": (3, [2, 1, 0]),
    "mono8": (1, [0, 0, 0]),
}


def _rows(data, height: int, step: int) -> np.ndarray:
    try:  # bytes, bytearray, array.array('B'), memoryview
        buf = np.frombuffer(data, dtype=np.uint8)
    except TypeError:  # list of ints
        buf = np.asarray(data, dtype=np.uint8)
    return buf[: height * step].reshape(height, step)


def image_to_rgb(data, height: int, width: int, encoding: str, step: int) -> np.ndarray:
    """sensor_msgs/Image payload -> uint8 [H,W,3] RGB."""
    enc = encoding.lower()
    if enc not in _RGB_ENCODINGS:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    ch, order = _RGB_ENCODINGS[enc]
    rows = _rows(data, height, step)[:, : width * ch]
    px = rows.reshape(height, width, ch)
    return np.ascontiguousarray(px[..., order])


def depth_to_float32(data, height: int, width: int, encoding: str, step: int) -> np.ndarray:
    """sensor_msgs/Image depth payload -> float32 [H,W] in metres (16UC1 is millimetres)."""
    enc = encoding.upper()
    if enc == "32FC1":
        dt, scale = np.dtype("<f4"), None
    elif enc == "16UC1":
        dt, scale = np.dtype("<u2"), 1e-3
    else:
        raise ValueError(f"unsupported depth encoding {encoding!r}")
    rows = _rows(data, height, step)[:, : width * dt.itemsize]
    d = np.ascontiguousarray(rows).view(dt).reshape(height, width).astype(np.float32)
    return d * np.float32(scale) if scale else d


# ---------------------------------------------------------------- PNG (numpy fallback)

def _unfilter(ftype: int, row: bytearray, prev: bytes, bpp: int) -> bytearray:
    if ftype == 0:
        return row
    if ftype == 2:
        return bytearray(((np.frombuffer(row, np.uint8).astype(np.uint16)
                           + np.frombuffer(prev, np.uint8)) & 0xFF).astype(np.uint8).tobytes())
    if ftype == 1:
        a = np.frombuffer(bytes(row), np.uint8).reshape(-1, bpp)
        return bytearray(np.cumsum(a, axis=0, dtype=np.uint8).tobytes())
    n = len(row)
    if ftype == 3:
        for i in range(n):
            left = row[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
        return row
    if ftype == 4:
        for i in range(n):
            if i >= bpp:
                a, c = row[i - bpp], prev[i - bpp]
            else:
                a = c = 0
            b = prev[i]
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            row[i] = (row[i] + pred) & 0xFF
        return row
    raise ValueError(f"bad PNG filter type {ftype}")


def png_decode_numpy(buf: bytes) -> np.ndarray:
    """Non-interlaced 8/16-bit gray/gray+alpha/RGB/RGBA PNG -> ndarray (uint8 or uint16)."""
    buf = bytes(buf)
    if not buf.startswith(_PNG_MAGIC):
        raise ValueError("not a PNG")
    pos, idat, hdr = 8, [], None
    while pos < len(buf):
        (length,) = struct.unpack(">I", buf[pos:pos + 4])
        tag = buf[pos + 4:pos + 8]
        data = buf[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", data)
        elif tag == b"IDAT":
            idat.append(data)
        elif tag == b"IEND":
            break
    if hdr is None:
        raise ValueError("PNG without IHDR")
    w, h, depth, ctype, _, _, interlace = hdr
    if interlace:
        raise ValueError("interlaced PNG is not supported")
    ch = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    if ch is None or depth not in (8, 16):
        raise ValueError(f"unsupported PNG colour type {ctype} / bit depth {depth}")
    bpp = ch * depth // 8
    stride = w * bpp
    raw = zlib.decompress(b"".join(idat))
    out = bytearray(h * stride)
    prev = bytes(stride)
    for y in range(h):
        off = y * (stride + 1)
        row = _unfilter(raw[off], bytearray(raw[off + 1:off + 1 + stride]), prev, bpp)
        out[y * stride:(y + 1) * stride] = row
        prev = bytes(row)
    arr = np.frombuffer(bytes(out), ">u2" if depth == 16 else np.uint8)
    arr = arr.astype(np.uint16) if depth == 16 else arr
    return arr.reshape(h, w, ch) if ch > 1 else arr.reshape(h, w)


# ---------------------------------------------------------------- generic decoders

def _decode_png_any(buf: bytes, backends) -> np.ndarray:
    """PNG -> native array (keeps 16-bit), channel order RGB(A)."""
    if "cv2" in backends:
        import cv2
        a = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
        if a is not None:
            if a.ndim == 3:
                a = a[..., [2, 1, 0, 3][: a.shape[2]]] if a.shape[2] >= 3 else a
            return np.ascontiguousarray(a)
    if "PIL" in backends:
        import io
        import PIL.Image
        im = PIL.Image.open(io.BytesIO(buf))
        a = np.asarray(im)
        if im.mode.startswith("I"):
            a = a.astype(np.uint16)
        return a
    return png_decode_numpy(buf)


def _to_rgb(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.uint8:
        raise ValueError(f"expected an 8-bit colour image, got {a.dtype}")
    if a.ndim == 2:
        return np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] == 2:   # gray + alpha
        return np.repeat(a[..., :1], 3, axis=2)
    return np.ascontiguousarray(a[..., :3])


def _decode_jpeg(buf: bytes, backends) -> np.ndarray:
    if "cv2" in backends:
        import cv2
        a = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if a is not None:
            return np.ascontiguousarray(a[..., ::-1])
    if "PIL" in backends:
        import io
        import PIL.Image
        return np.asarray(PIL.Image.open(io.BytesIO(buf)).convert("RGB"))
    raise ImportError("JPEG frames need opencv-python (cv2) or Pillow; install one of them "
                      "(the ROS container has cv2) or record with --rgb raw")


def _decode_compressed_depth(fmt: str, data: bytes, backends) -> np.ndarray:
    if "rvl" in fmt.lower():
        raise NotImplementedError("compressedDepth 'rvl' format is not supported; set the "
                                  "publisher's compressedDepth.format to png")
    _, qa, qb = struct.unpack("<iff", data[:12])
    inv = _decode_png_any(data[12:], backends)
    if inv.ndim != 2:
        raise ValueError("compressedDepth PNG must be single-channel")
    enc = fmt.split(";")[0].strip().upper()
    if enc == "16UC1":
        return inv.astype(np.float32) * np.float32(1e-3)
    if enc != "32FC1":
        raise ValueError(f"unsupported compressedDepth encoding {enc!r}")
    out = np.full(inv.shape, np.nan, np.float32)
    nz = inv > 0
    out[nz] = np.float32(qa) / (inv[nz].astype(np.float32) - np.float32(qb))
    return out


# ---------------------------------------------------------------- CDR sensor_msgs/Image

def parse_image_cdr(buf: bytes) -> NS:
    """Serialised sensor_msgs/msg/Image (CDR with 4-byte encapsulation header) -> namespace."""
    buf = memoryview(bytes(buf))
    if len(buf) < 4 or buf[0] != 0 or buf[1] not in (0, 1):
        raise ValueError("not a CDR payload")
    e = "<" if buf[1] == 1 else ">"
    body = buf[4:]
    pos = 0

    def align(n):
        nonlocal pos
        pos += (-pos) % n

    def u32():
        nonlocal pos
        align(4)
        (v,) = struct.unpack_from(e + "I", body, pos)
        pos += 4
        return v

    def i32():
        nonlocal pos
        align(4)
        (v,) = struct.unpack_from(e + "i", body, pos)
        pos += 4
        return v

    def string():
        nonlocal pos
        n = u32()
        s = bytes(body[pos:pos + n]).rstrip(b"\0").decode()
        pos += n
        return s

    sec, nsec = i32(), u32()
    frame_id = string()
    height, width = u32(), u32()
    encoding = string()
    is_bigendian = body[pos]
    pos += 1
    step = u32()
    n = u32()
    data = bytes(body[pos:pos + n])
    return NS(header=NS(stamp=NS(sec=sec, nanosec=nsec), frame_id=frame_id), height=height,
              width=width, encoding=encoding, is_bigendian=is_bigendian, step=step, data=data)


def _unzstd(data: bytes) -> bytes:
    try:
        import zstandard
    except ImportError as e:
        raise ImportError("zstd image transport needs the 'zstandard' package") from e
    d = zstandard.ZstdDecompressor()
    try:
        return d.decompress(data)
    except zstandard.ZstdError:   # frame without content size
        return d.decompressobj().decompress(data)


# ---------------------------------------------------------------- public API

def is_compressed(msg) -> bool:
    return hasattr(msg, "format") and not hasattr(msg, "height")


def transport_of(msg) -> str:
    if not is_compressed(msg):
        return "raw"
    fmt = str(msg.format).lower()
    if "compresseddepth" in fmt:
        return "compressedDepth"
    if "zstd" in fmt:
        return "zstd"
    if "jpeg" in fmt or "jpg" in fmt:
        return "jpeg"
    if "png" in fmt:
        return "png"
    return "compressed"


def decode_compressed(msg, depth: bool = False, backends=None) -> np.ndarray:
    """sensor_msgs/CompressedImage -> uint8 [H,W,3] RGB, or float32 [H,W] metres if depth."""
    backends = _BACKENDS if backends is None else backends
    fmt = str(msg.format)
    data = bytes(msg.data)
    t = transport_of(msg)
    if t == "compressedDepth":
        return _decode_compressed_depth(fmt, data, backends)
    if t == "zstd":
        img = parse_image_cdr(_unzstd(data))
        conv = depth_to_float32 if depth else image_to_rgb
        return conv(img.data, img.height, img.width, img.encoding, img.step)
    if depth:
        raise ValueError(f"depth over the {fmt!r} transport is not supported "
                         "(use compressedDepth or zstd)")
    if data.startswith(_JPEG_MAGIC):
        return _decode_jpeg(data, backends)
    if data.startswith(_PNG_MAGIC):
        return _to_rgb(_decode_png_any(data, backends))
    raise ValueError(f"unknown compressed image format {fmt!r}")


def _as_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    try:
        return memoryview(data).tobytes()
    except TypeError:  # list of ints
        return np.asarray(data, dtype=np.uint8).tobytes()


def frame_payload(msg, depth: bool) -> dict:
    """Picklable copy of one image message (copies the data out of reader buffers)."""
    if is_compressed(msg):
        return {"transport": transport_of(msg), "depth": depth, "format": str(msg.format),
                "data": _as_bytes(msg.data)}
    return {"transport": "raw", "depth": depth, "height": int(msg.height),
            "width": int(msg.width), "encoding": str(msg.encoding), "step": int(msg.step),
            "data": _as_bytes(msg.data)}


def decode_payload(p: dict) -> np.ndarray:
    if p["transport"] == "raw":
        conv = depth_to_float32 if p["depth"] else image_to_rgb
        return conv(p["data"], p["height"], p["width"], p["encoding"], p["step"])
    return decode_compressed(NS(format=p["format"], data=p["data"]), depth=p["depth"])


def payload_encoding(p: dict) -> str:
    return p["encoding"] if p["transport"] == "raw" else p["format"]
