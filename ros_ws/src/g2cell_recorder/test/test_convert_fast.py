"""Compressed-transport input, HDF5 compression choices and parallel conversion."""
import struct
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest

from g2cell_recorder.episode import DEFAULT_TOPICS, convert_messages

from test_episode import SEC, _fake_bag, _stamp
from test_image_codec import compressed_depth_payload, encode_png

HEAD = "/genie_sim/head_front_camera_rgb/image_raw"
DEPTH = "/genie_sim/head_front_camera_depth/image_raw"


def _png_img(ns, value, h=4, w=5):
    rgb = np.full((h, w, 3), value, np.uint8)
    rgb[0, 0] = (value, 0, 255 - value)
    return NS(header=NS(stamp=_stamp(ns)), format="rgba8; png compressed bgra8",
              data=encode_png(rgb))


def _cdepth(ns, h=4, w=5):
    return NS(header=NS(stamp=_stamp(ns)), format="32FC1; compressedDepth png",
              data=compressed_depth_payload(np.full((h, w), 1.5, np.float32)))


def _compressed_bag():
    """The fake bag, with head RGB and depth over compressed / compressedDepth instead."""
    out = []
    for topic, t, msg in _fake_bag():
        if topic == HEAD:
            out.append((HEAD + "/compressed", t, _png_img(t, msg.data[0])))
        elif topic == DEPTH:
            out.append((DEPTH + "/compressedDepth", t, _cdepth(t)))
        else:
            out.append((topic, t, msg))
    return out


def test_default_topics_map_compressed_transports_to_the_same_streams():
    for suffix in ("/compressed", "/zstd"):
        assert DEFAULT_TOPICS[HEAD + suffix] == DEFAULT_TOPICS[HEAD]
        free = "/genie_sim/free_camera_rgb/image_raw"
        assert DEFAULT_TOPICS[free + suffix] == DEFAULT_TOPICS[free]
    for suffix in ("/compressedDepth", "/zstd"):
        assert DEFAULT_TOPICS[DEPTH + suffix] == DEFAULT_TOPICS[DEPTH]
    assert DEFAULT_TOPICS["/g2cell/rec/head_front_camera_depth/compressedDepth"] == \
        DEFAULT_TOPICS[DEPTH]


def test_compressed_bag_converts_to_the_same_schema(tmp_path):
    raw, comp = tmp_path / "raw.hdf5", tmp_path / "comp.hdf5"
    convert_messages(iter(_fake_bag()), raw, source="b")
    s = convert_messages(iter(_compressed_bag()), comp, source="b")
    assert s["transports"]["images/head_front_rgb"] == "png"
    assert s["transports"]["depth/head_front"] == "compressedDepth"
    with h5py.File(raw) as a, h5py.File(comp) as b:
        np.testing.assert_array_equal(a["images/head_front_rgb/data"][:, 1:, 1:],
                                      b["images/head_front_rgb/data"][:, 1:, 1:])
        assert b["images/head_front_rgb/data"][1, 0, 0].tolist() == [2, 0, 253]
        np.testing.assert_allclose(b["depth/head_front/data"][:], 1.5, atol=1e-3)
        g = b["images/head_front_rgb"]
        assert g.attrs["topic"] == HEAD + "/compressed"
        assert g.attrs["transport"] == "png"
        assert g.attrs["encoding"] == "rgba8; png compressed bgra8"
        assert a["images/head_front_rgb"].attrs["transport"] == "raw"
        np.testing.assert_array_equal(a["aligned/images/head_front_rgb/index"][:],
                                      b["aligned/images/head_front_rgb/index"][:])


def test_second_topic_for_the_same_stream_is_ignored(tmp_path):
    msgs = _fake_bag() + [(HEAD + "/compressed", t, _png_img(t, 9))
                          for t in (5, SEC // 10 + 5)]
    msgs.sort(key=lambda m: m[1])
    s = convert_messages(iter(msgs), tmp_path / "e.hdf5", source="b")
    assert s["ignored_topics"] == {HEAD + "/compressed": 2}
    with h5py.File(tmp_path / "e.hdf5") as f:
        assert f["images/head_front_rgb/data"].shape[0] == 3


@pytest.mark.parametrize("compression,level", [("gzip", 1), ("gzip", 6), ("lzf", None),
                                               ("none", None)])
def test_compression_choices_roundtrip(tmp_path, compression, level):
    ref = tmp_path / "ref.hdf5"
    convert_messages(iter(_fake_bag()), ref, source="b", compression="none")
    out = tmp_path / "o.hdf5"
    convert_messages(iter(_fake_bag()), out, source="b", compression=compression,
                     compression_level=level)
    with h5py.File(ref) as a, h5py.File(out) as b:
        for k in ("images/head_front_rgb/data", "depth/head_front/data"):
            np.testing.assert_array_equal(a[k][:], b[k][:])
            ds = b[k]
            assert ds.chunks == (1, *ds.shape[1:])          # one frame per chunk
            assert ds.compression == (None if compression == "none" else compression)
            if compression == "gzip":
                assert ds.compression_opts == level
        assert b["depth/head_front/data"].shuffle == (compression != "none")
        assert b.attrs["image_compression"] == compression


def test_parallel_workers_match_serial(tmp_path):
    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    convert_messages(iter(_compressed_bag()), a, source="b", workers=0, aligned_images=True)
    convert_messages(iter(_compressed_bag()), b, source="b", workers=2, aligned_images=True)
    with h5py.File(a) as fa, h5py.File(b) as fb:
        keys = []
        fa.visititems(lambda n, o: keys.append(n) if isinstance(o, h5py.Dataset) else None)
        for k in keys:
            np.testing.assert_array_equal(fa[k][:], fb[k][:], err_msg=k)
        assert fb["aligned/images/head_front_rgb/data"][-1, 1, 1].tolist() == [3, 3, 3]


def test_bad_compression_name_raises(tmp_path):
    with pytest.raises(ValueError, match="compression"):
        convert_messages(iter(_fake_bag()), tmp_path / "x.hdf5", source="b", compression="zip")


def test_failed_frame_decode_leaves_no_partial_file(tmp_path):
    bad = NS(header=NS(stamp=_stamp(0)), format="png", data=b"\x89PNG\r\n\x1a\n" + b"\0" * 8)
    msgs = [(HEAD + "/compressed", 0, bad)] + _fake_bag()
    with pytest.raises(Exception):
        convert_messages(iter(msgs), tmp_path / "x.hdf5", source="b", workers=2)
    assert not list(tmp_path.iterdir())


def test_header_struct_is_twelve_bytes():
    assert struct.calcsize("<iff") == 12
