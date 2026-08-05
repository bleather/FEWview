import unittest
from pathlib import Path

from fewview.cli import cluster_job, cluster_segment


class ClusterCameraFlagTests(unittest.TestCase):
    """The job generator must emit only flags the segment worker accepts."""

    def _job_args(self, extra):
        argv = ["modes.npz", "--frames", "120", "--segments", "4"] + extra
        return cluster_job.build_parser().parse_args(argv)

    def test_worker_arguments_forward_camera_flight_to_segment_parser(self):
        args = self._job_args(
            [
                "--camera-latitude", "10",
                "--camera-longitude", "0",
                "--camera-latitude-end", "70",
                "--camera-zoom-end", "1.8",
                "--camera-loop",
                "--camera-azimuth", "5",
                "--camera-elevation", "-3",
                "--no-waveform-transparent",
            ]
        )
        worker = cluster_job._worker_arguments(args, Path("modes.npz"), Path("segs"))
        # The segment CLI must parse exactly what the job CLI emits.
        seg = cluster_segment.build_parser().parse_args(
            worker + ["--segment-index", "0"]
        )
        self.assertEqual(seg.camera_latitude, 10.0)
        self.assertEqual(seg.camera_longitude, 0.0)
        self.assertEqual(seg.camera_latitude_end, 70.0)
        self.assertIsNone(seg.camera_longitude_end)
        self.assertEqual(seg.camera_zoom_end, 1.8)
        self.assertTrue(seg.camera_loop)
        self.assertEqual(seg.camera_azimuth, 5.0)
        self.assertEqual(seg.camera_elevation, -3.0)
        self.assertFalse(seg.waveform_transparent)

    def test_defaults_round_trip_and_omit_absolute_angles(self):
        args = self._job_args([])
        worker = cluster_job._worker_arguments(args, Path("modes.npz"), Path("segs"))
        # With no absolute angle requested, the optional flags are not emitted.
        self.assertNotIn("--camera-latitude", worker)
        self.assertNotIn("--camera-longitude-end", worker)
        self.assertNotIn("--camera-zoom-end", worker)
        seg = cluster_segment.build_parser().parse_args(
            worker + ["--segment-index", "0"]
        )
        self.assertIsNone(seg.camera_latitude)
        self.assertIsNone(seg.camera_zoom_end)
        self.assertFalse(seg.camera_loop)
        self.assertTrue(seg.waveform_transparent)  # transparent panel by default

    def test_camera_flight_settings_change_the_run_fingerprint(self):
        # Different camera options must land in a different segment directory,
        # so re-runs do not reuse another shot's completed segments.
        base = self._job_args([])
        loop = self._job_args(["--camera-loop", "--camera-latitude-end", "70"])
        keys = (
            "camera_loop",
            "camera_latitude_end",
            "camera_latitude",
            "camera_longitude",
            "waveform_transparent",
        )
        self.assertNotEqual(
            tuple(getattr(base, k) for k in keys),
            tuple(getattr(loop, k) for k in keys),
        )


if __name__ == "__main__":
    unittest.main()
