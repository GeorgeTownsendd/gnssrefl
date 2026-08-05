from unittest import mock

import os
import shutil
import urllib.error
from pathlib import Path

import ncompress
import pytest
import wget

from gnssrefl.gps import *


def test_get_sopac_navfile_working(tmp_path, monkeypatch, mocker):
    """
    SOPAC serves the nav file .Z compressed. Assert the outcome - the nav file
    is on disk with its contents intact and the compressed original is gone -
    rather than the argv of the shell tool that used to do the decompressing.
    """
    monkeypatch.chdir(tmp_path)
    nav_contents = b"     2.11           N: GPS NAV DATA                     RINEX VERSION / TYPE\n"

    def download_a_compressed_navfile(url, out):
        Path(out).write_bytes(ncompress.compress(nav_contents))

    mocker.patch("wget.download", side_effect=download_a_compressed_navfile)

    assert (
        get_sopac_navfile("auto1050.20n", "2020", "20", "105")
        == "auto1050.20n"
    )
    wget.download.assert_called_once_with(
        "ftp://garner.ucsd.edu/pub/rinex/2020/105/auto1050.20n.Z",
        "auto1050.20n.Z",
    )
    assert (tmp_path / "auto1050.20n").read_bytes() == nav_contents
    assert not (tmp_path / "auto1050.20n.Z").exists()


def test_get_sopac_navfile_download_fails(tmp_path, monkeypatch, mocker):
    """a download that never happens leaves nothing behind and does not raise"""
    monkeypatch.chdir(tmp_path)
    mocker.patch("wget.download", side_effect=urllib.error.URLError("404"))

    assert (
        get_sopac_navfile("p1031050.20.snr66", "2020", "20", "105")
        == "p1031050.20.snr66"  # should this return None instead?
    )
    wget.download.assert_called_once_with(
        "ftp://garner.ucsd.edu/pub/rinex/2020/105/p1031050.20.snr66.Z",
        "p1031050.20.snr66.Z",
    )
    assert list(tmp_path.iterdir()) == []


# this should be changed to run on january 1, 2021 which probably
# has a corrupt file on the sopac database
def test_get_sopac_navfile_corrupt_download(tmp_path, monkeypatch, mocker):
    """
    A download that arrives corrupt cannot be decompressed, so the partial file
    is cleaned up. Asserting that the corpse is really gone tests more than the
    old assertion on the argv of the rm did.
    """
    monkeypatch.chdir(tmp_path)

    def download_a_corrupt_navfile(url, out):
        Path(out).write_bytes(b"this is not LZW compressed data")

    mocker.patch("wget.download", side_effect=download_a_corrupt_navfile)

    assert (
        get_sopac_navfile("p1031050.20.snr66", "2020", "20", "105")
        == "p1031050.20.snr66"
    )
    assert list(tmp_path.iterdir()) == []


def test_store_snrfile_moves_across_filesystems(tmp_path):
    """
    Regression test for issue 417 (bug introduced in commit 165cd4a7).

    Checks that store_snrfile moves an SNR file when the file and REFL_CODE
    are on two different filesystems, which is what the Docker bind-mount
    layout produces and what os.replace cannot do.
    """
    second_device = Path("/dev/shm")
    if not second_device.is_dir() or second_device.stat().st_dev == tmp_path.stat().st_dev:
        pytest.skip("no second filesystem available to exercise a cross-device move")

    refl_code = second_device / f"refl_code_{os.getpid()}"
    refl_code.mkdir()
    try:
        snrfile = tmp_path / "p1010010.25.snr66"
        snrfile.write_text("snr data")
        with mock.patch.dict(os.environ, {"REFL_CODE": str(refl_code)}):
            store_snrfile(str(snrfile), 2025, "p101")
        assert (refl_code / "2025" / "snr" / "p101" / "p1010010.25.snr66").is_file()
    finally:
        shutil.rmtree(refl_code, ignore_errors=True)


def test_checkEGM_creates_missing_files_directory(tmp_path, mocker):
    """
    Regression test for issue 418.

    checkEGM must create REFL_CODE/Files when it does not already exist, which
    is the case in a fresh Docker container. The old code passed the directory
    as the bufsize argument of subprocess.call, raising TypeError.
    """
    mocker.patch("wget.download")
    refl_code = tmp_path / "refl_code"
    refl_code.mkdir()
    with mock.patch.dict(os.environ, {"REFL_CODE": str(refl_code)}):
        checkEGM()
    assert (refl_code / "Files").is_dir()
