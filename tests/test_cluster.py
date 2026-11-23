import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fewview.cli import SEGMENT_RENDER_VERSION, cluster_job, cluster_merge, cluster_segment


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
                "--logo", "icon",
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
        self.assertEqual(seg.logo, "icon")

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

    def test_glow_has_its_own_cache_and_reaches_the_worker(self):
        classic = self._job_args([])
        glow = self._job_args(["--source-glow"])
        modes = Path("modes.npz")
        file_stat = Path(__file__).stat()
        keys = cluster_job.RENDER_KEYS
        legacy_keys = tuple(key for key in keys if key != "source_glow")
        fingerprint = cluster_job._run_fingerprint
        default_id = fingerprint(classic, keys, modes, file_stat)
        self.assertEqual(default_id, fingerprint(classic, legacy_keys, modes, file_stat))
        self.assertNotEqual(default_id, fingerprint(glow, keys, modes, file_stat))
        worker = cluster_job._worker_arguments(glow, modes, Path("segs"))
        segment = cluster_segment.build_parser().parse_args(
            worker + ["--segment-index", "0"]
        )
        self.assertTrue(segment.source_glow)

    def test_waveform_glow_round_trips_and_preserves_older_run_ids(self):
        modes = Path("modes.npz")
        file_stat = Path(__file__).stat()
        keys = cluster_job.RENDER_KEYS
        older_keys = tuple(key for key in keys if key != "waveform_glow")
        for flags in ([], ["--source-glow"]):
            args = self._job_args(flags)
            fingerprint = cluster_job._run_fingerprint
            older_id = fingerprint(args, older_keys, modes, file_stat)
            self.assertEqual(fingerprint(args, keys, modes, file_stat), older_id)
            args = self._job_args([*flags, "--waveform-glow"])
            self.assertNotEqual(fingerprint(args, keys, modes, file_stat), older_id)
            worker = cluster_job._worker_arguments(args, modes, Path("segs"))
            segment = cluster_segment.build_parser().parse_args(
                worker + ["--segment-index", "0"]
            )
            self.assertTrue(segment.waveform_glow)

    def test_camera_flight_settings_change_the_run_fingerprint(self):
        # Different camera options must produce a different run fingerprint,
        # so re-runs do not reuse another shot's completed segments.
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            tmp.write(b"dummy")
            tmp.flush()
            modes = Path(tmp.name)
            mode_stat = modes.stat()

            base = self._job_args([])
            loop = self._job_args(["--camera-loop", "--camera-latitude-end", "70"])

            fp_base = cluster_job._run_fingerprint(
                base, cluster_job.RENDER_KEYS, modes, mode_stat
            )
            fp_loop = cluster_job._run_fingerprint(
                loop, cluster_job.RENDER_KEYS, modes, mode_stat
            )
            self.assertNotEqual(fp_base, fp_loop)

    def test_merge_catches_segments_from_before_the_aligned_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "segment_0000.mp4").touch()
            (folder / "segment_0000.done").write_text("complete\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "Render segment_0000 again"):
                cluster_merge.main([tmp, str(folder / "movie.mp4"), "--segments", "1"])

    def test_custom_style_is_frozen_for_each_queued_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "modes.npz").touch()
            style = project / "paper.mplstyle"
            job = project / "job"
            for size in (10, 12):
                style.write_text(f"font.size: {size}\n")
                cluster_job.main(["modes.npz", "--project-dir", tmp, "--job-dir", str(job),
                                  "--waveform-style-file", str(style)])
                script = (job / "render_array.slurm").read_text()
                snapshots = list((job / "segments").glob("*/waveform.mplstyle"))
                current = next(p for p in snapshots if p.read_text() == style.read_text())
                self.assertIn(str(current), script)
            self.assertEqual(len(snapshots), 2)
            self.assertEqual({p.read_text() for p in snapshots}, {"font.size: 10\n", "font.size: 12\n"})

    def test_failed_merge_preserves_previous_movie(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "segment_0000.mp4").touch()
            (folder / "segment_0000.done").write_text(f"{SEGMENT_RENDER_VERSION}:icon\n")
            output = folder / "movie.mp4"
            output.write_bytes(b"previous movie")

            def fail_after_open(command, *, check):
                Path(command[-1]).write_bytes(b"partial")
                raise subprocess.CalledProcessError(1, command)

            with patch.object(subprocess, "run", side_effect=fail_after_open):
                with self.assertRaises(subprocess.CalledProcessError):
                    cluster_merge.main([tmp, str(output), "--segments", "1"])
            self.assertEqual(output.read_bytes(), b"previous movie")

    def test_merge_catches_different_logos(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for index, logo in enumerate(("wordmark", "icon")):
                segment = folder / f"segment_{index:04d}"
                segment.with_suffix(".mp4").touch()
                segment.with_suffix(".done").write_text(f"{SEGMENT_RENDER_VERSION}:{logo}\n")
            with self.assertRaisesRegex(SystemExit, "same --logo"):
                cluster_merge.main([tmp, str(folder / "movie.mp4"), "--segments", "2"])


if __name__ == "__main__":
    unittest.main()
