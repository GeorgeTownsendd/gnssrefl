"""
Unit tests for gnssrefl.fileops, the portable replacements for the Unix shell
tools (gunzip, unxz, uncompress, gzip, mv, rm) that gnssrefl calls through
subprocess.

These run on Linux, macOS and Windows and need no network access.
"""

import errno
import gzip
import lzma
import os
import shutil
import string
from pathlib import Path

import ncompress
import pytest

from gnssrefl import fileops

# a little bigger than one buffer and deliberately not valid text, so a
# truncated or newline translated result would not compare equal
PAYLOAD = (b'RINEX observation data \r\n\x00\xff' * 5000)


def _second_filesystem(reference):
    """
    Find a writable directory on a filesystem other than the one holding
    reference, or None if there isn't one. Used to exercise the cross device
    move that os.replace cannot do (issue 417).
    """
    this_device = os.stat(reference).st_dev

    if os.name == 'nt':
        candidates = [Path(letter + ':\\') for letter in string.ascii_uppercase]
    else:
        candidates = [Path('/dev/shm'), Path('/tmp'), Path('/var/tmp'), Path.home()]

    for candidate in candidates:
        try:
            if (candidate.is_dir() and os.stat(candidate).st_dev != this_device
                    and os.access(candidate, os.W_OK)):
                return candidate
        except OSError:
            continue

    return None


# decompression


@pytest.mark.parametrize('suffix, compress, func', [
    ('.gz', gzip.compress, fileops.gunzip),
    ('.xz', lzma.compress, fileops.unxz),
    ('.Z', ncompress.compress, fileops.uncompress),
])
def test_decompress_round_trip(tmp_path, suffix, compress, func):
    """the contents come back byte for byte and the original is deleted"""
    compressed = tmp_path / ('rinexfile.o' + suffix)
    compressed.write_bytes(compress(PAYLOAD))

    assert func(str(compressed)) is True

    decompressed = tmp_path / 'rinexfile.o'
    assert decompressed.read_bytes() == PAYLOAD
    assert not compressed.exists()


@pytest.mark.parametrize('suffix, func', [
    ('.gz', fileops.gunzip),
    ('.xz', fileops.unxz),
    ('.Z', fileops.uncompress),
])
def test_decompress_corrupt_input(tmp_path, suffix, func):
    """corrupt input fails without raising and leaves no truncated output"""
    compressed = tmp_path / ('rinexfile.o' + suffix)
    compressed.write_bytes(b'this is not compressed data')

    assert func(str(compressed)) is False

    assert not (tmp_path / 'rinexfile.o').exists()
    assert compressed.exists()


@pytest.mark.parametrize('func', [fileops.gunzip, fileops.unxz, fileops.uncompress])
def test_decompress_wrong_suffix(tmp_path, func):
    """a file that does not carry the expected extension is refused"""
    plain = tmp_path / 'rinexfile.o'
    plain.write_bytes(PAYLOAD)

    assert func(str(plain)) is False

    assert plain.read_bytes() == PAYLOAD


@pytest.mark.parametrize('suffix, func', [
    ('.gz', fileops.gunzip),
    ('.xz', fileops.unxz),
    ('.Z', fileops.uncompress),
])
def test_decompress_missing_input(tmp_path, suffix, func):
    """a file that is not there fails without raising"""
    assert func(str(tmp_path / ('missing.o' + suffix))) is False

    assert not (tmp_path / 'missing.o').exists()


# compression


def test_gzip_file_round_trip(tmp_path):
    """gzip_file writes filename.gz, deletes the original, and gunzip undoes it"""
    plain = tmp_path / 'rinexfile.o'
    plain.write_bytes(PAYLOAD)

    assert fileops.gzip_file(str(plain)) is True

    compressed = tmp_path / 'rinexfile.o.gz'
    assert not plain.exists()
    assert gzip.decompress(compressed.read_bytes()) == PAYLOAD

    assert fileops.gunzip(str(compressed)) is True
    assert plain.read_bytes() == PAYLOAD


def test_gzip_file_missing_input(tmp_path):
    """compressing a file that is not there fails and leaves no .gz behind"""
    assert fileops.gzip_file(str(tmp_path / 'missing.o')) is False

    assert not (tmp_path / 'missing.o.gz').exists()


# move


def test_move_to_directory(tmp_path):
    """a directory destination keeps the basename, as mv does"""
    source = tmp_path / 'auto1050.20n'
    source.write_bytes(PAYLOAD)
    destination = tmp_path / 'nav'
    destination.mkdir()

    assert fileops.move(str(source), str(destination)) is True

    assert (destination / 'auto1050.20n').read_bytes() == PAYLOAD
    assert not source.exists()


def test_move_to_file(tmp_path):
    """a file destination renames"""
    source = tmp_path / 'temporary.o'
    source.write_bytes(PAYLOAD)
    destination = tmp_path / 'p1010010.25o'

    assert fileops.move(str(source), str(destination)) is True

    assert destination.read_bytes() == PAYLOAD
    assert not source.exists()


def test_move_overwrites_existing_destination(tmp_path):
    """
    mv silently overwrites an existing destination file. shutil.move does not,
    on Windows, and six call sites depend on the overwrite.
    """
    source = tmp_path / 'temporary.o'
    source.write_bytes(PAYLOAD)
    destination = tmp_path / 'p1010010.25o'
    destination.write_bytes(b'stale contents')

    assert fileops.move(str(source), str(destination)) is True

    assert destination.read_bytes() == PAYLOAD
    assert not source.exists()


def test_move_overwrites_in_destination_directory(tmp_path):
    """the same, when the destination is given as a directory"""
    source = tmp_path / 'auto1050.20n'
    source.write_bytes(PAYLOAD)
    destination = tmp_path / 'nav'
    destination.mkdir()
    (destination / 'auto1050.20n').write_bytes(b'stale contents')

    assert fileops.move(str(source), str(destination)) is True

    assert (destination / 'auto1050.20n').read_bytes() == PAYLOAD
    assert not source.exists()


def test_move_missing_source(tmp_path):
    """moving a file that is not there fails without raising"""
    assert fileops.move(str(tmp_path / 'missing.o'), str(tmp_path / 'destination.o')) is False


def test_move_across_filesystems(tmp_path):
    """
    os.replace cannot move between filesystems, which is what the Docker bind
    mount layout produces (issue 417), so move has to fall back to shutil.move.
    """
    second_device = _second_filesystem(tmp_path)
    if second_device is None:
        pytest.skip('no second filesystem available to exercise a cross-device move')

    destination_dir = second_device / ('gnssrefl_fileops_%d' % os.getpid())
    destination_dir.mkdir()
    try:
        source = tmp_path / 'p1010010.25.snr66'
        source.write_bytes(PAYLOAD)

        assert fileops.move(str(source), str(destination_dir)) is True

        assert (destination_dir / 'p1010010.25.snr66').read_bytes() == PAYLOAD
        assert not source.exists()
    finally:
        shutil.rmtree(destination_dir, ignore_errors=True)


def test_move_falls_back_when_replace_fails(tmp_path, monkeypatch):
    """
    The same fallback, forced. os.replace raises EXDEV across filesystems, so
    simulating that exercises the shutil.move path (and its overwrite) on every
    platform, not only where a second volume happens to be mounted.
    """
    def refuse_to_replace(src, dst):
        raise OSError(errno.EXDEV, 'Invalid cross-device link')

    monkeypatch.setattr(os, 'replace', refuse_to_replace)

    source = tmp_path / 'p1010010.25.snr66'
    source.write_bytes(PAYLOAD)
    destination = tmp_path / 'stored.snr66'
    destination.write_bytes(b'stale contents')

    assert fileops.move(str(source), str(destination)) is True

    assert destination.read_bytes() == PAYLOAD
    assert not source.exists()


# remove


def test_remove(tmp_path):
    """an existing file is deleted"""
    target = tmp_path / 'p1010010.25o'
    target.write_bytes(PAYLOAD)

    assert fileops.remove(str(target)) is True

    assert not target.exists()


def test_remove_missing_file(tmp_path):
    """rm -f tolerance: about twenty call sites remove a file without checking"""
    assert fileops.remove(str(tmp_path / 'missing.o')) is True


# remove_glob


def test_remove_glob(tmp_path):
    """every match goes, and nothing else does"""
    for name in ['p1010010.25otmp1', 'p1010010.25otmp2', 'p1010010.25otmp3']:
        (tmp_path / name).write_bytes(PAYLOAD)
    keep = tmp_path / 'p1010010.25o'
    keep.write_bytes(PAYLOAD)

    assert fileops.remove_glob(str(tmp_path / 'p1010010.25otmp*')) is True

    assert sorted(p.name for p in tmp_path.iterdir()) == ['p1010010.25o']
    assert keep.exists()


def test_remove_glob_no_matches(tmp_path):
    """rm -f on a pattern that matches nothing is not an error"""
    assert fileops.remove_glob(str(tmp_path / 'nothing_matches_this*')) is True


def test_remove_glob_leaves_directories(tmp_path):
    """remove_glob is rm -f, not rm -rf, so a matching directory is not removed"""
    (tmp_path / 'p1010010.25otmp1').write_bytes(PAYLOAD)
    (tmp_path / 'p1010010.25otmpdir').mkdir()

    assert fileops.remove_glob(str(tmp_path / 'p1010010.25otmp*')) is False

    assert (tmp_path / 'p1010010.25otmpdir').is_dir()
    assert not (tmp_path / 'p1010010.25otmp1').exists()


# remove_tree


def test_remove_tree(tmp_path):
    """rm -rf takes directories, recursively, as well as files"""
    nested = tmp_path / '25d' / 'deep' / 'deeper'
    nested.mkdir(parents=True)
    (nested / 'buried.rnx').write_bytes(PAYLOAD)
    (tmp_path / '25d' / 'loose.rnx').write_bytes(PAYLOAD)
    keep = tmp_path / 'keep.rnx'
    keep.write_bytes(PAYLOAD)

    assert fileops.remove_tree(str(tmp_path / '25*')) is True

    assert not (tmp_path / '25d').exists()
    assert keep.exists()


def test_remove_tree_no_matches(tmp_path):
    """rm -rf on a pattern that matches nothing is not an error"""
    assert fileops.remove_tree(str(tmp_path / 'nothing_matches_this*')) is True


# copy


def test_copy_to_file(tmp_path):
    """a file destination copies and leaves the original in place"""
    source = tmp_path / 'p1010010.25o'
    source.write_bytes(PAYLOAD)
    destination = tmp_path / 'copy.25o'

    assert fileops.copy(str(source), str(destination)) is True

    assert destination.read_bytes() == PAYLOAD
    assert source.exists()


def test_copy_to_directory(tmp_path):
    """a directory destination keeps the basename, as cp does"""
    source = tmp_path / 'p1010010.25o'
    source.write_bytes(PAYLOAD)
    destination = tmp_path / 'input'
    destination.mkdir()

    assert fileops.copy(str(source), str(destination)) is True

    assert (destination / 'p1010010.25o').read_bytes() == PAYLOAD


def test_copy_onto_itself(tmp_path):
    """
    cp reports that source and destination are the same file and carries on.
    shutil.copy raises SameFileError, which would turn a silent no-op on Linux
    into a crash, so copy has to swallow it.
    """
    source = tmp_path / 'p1010010.25o'
    source.write_bytes(PAYLOAD)

    assert fileops.copy(str(source), str(tmp_path)) is False

    assert source.read_bytes() == PAYLOAD


def test_copy_missing_source(tmp_path):
    """copying a file that is not there fails without raising"""
    assert fileops.copy(str(tmp_path / 'missing.o'), str(tmp_path / 'dst.o')) is False
