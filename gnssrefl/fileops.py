"""
Portable replacements for the Unix shell tools gnssrefl calls through subprocess.

gnssrefl currently reaches for gunzip, unxz, uncompress, gzip, mv and rm via
subprocess.call. None of those are executables on Windows, so those calls raise
FileNotFoundError instead of returning a non-zero status. This module provides
the same operations in pure Python plus ncompress, which makes those call sites
work identically on Linux, macOS and Windows.

This is deliberately a leaf module: it imports the standard library and
ncompress and nothing from gnssrefl, so gps.py can use it without dragging in
utils.py (numpy, FileManagement) and creating an import cycle.

Every function returns True on success and False on failure, and none of them
raise. That matches the subprocess.call behaviour they replace - a failed shell
tool returned a non-zero status rather than raising, and no existing call site
inspects the return value.
"""
import glob
import gzip
import io
import lzma
import os
import shutil
import zipfile

# gzip raises BadGzipFile (an OSError) on a non-gzip file and EOFError on a
# truncated one, lzma raises LZMAError, and ncompress raises ValueError on
# input that is not in LZW format or is empty.
_DECOMPRESSION_ERRORS = (OSError, EOFError, ValueError, lzma.LZMAError)


def _open_gz(filename):
    """Open a gzip file for binary reading."""
    return gzip.open(filename, 'rb')


def _open_xz(filename):
    """Open an xz file for binary reading."""
    return lzma.open(filename, 'rb')


def _open_z(filename):
    """
    Read a .Z (LZW, Unix compress) file and return its contents as a stream.

    There is no LZW decoder in the standard library, so this uses ncompress.
    ncompress documents bytes and BytesIO arguments only, so the whole file is
    read into memory rather than passing it an open file handle. RINEX files
    are small enough for that not to matter.
    """
    from ncompress import decompress as lzw_decompress

    with open(filename, 'rb') as f_in:
        return io.BytesIO(lzw_decompress(f_in.read()))


def _decompress(filename, suffix, opener):
    """
    Decompress filename in place, i.e. write the decompressed contents to the
    name without suffix and delete the compressed original, as the command line
    decompression tools do.

    Parameters
    ----------
    filename : str
        name of the compressed file
    suffix : str
        extension the compressed file is required to have
    opener : callable
        returns an open binary stream of the decompressed contents

    Returns
    -------
    bool
        True if the file was decompressed and the original removed
    """
    if not filename.endswith(suffix):
        print('Cannot decompress ' + filename + ' because it does not end in ' + suffix)
        return False

    outfile = filename[:-len(suffix)]
    try:
        with opener(filename) as f_in, open(outfile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    except _DECOMPRESSION_ERRORS as e:
        print('Could not decompress ' + filename + ' : ' + str(e))
        # do not leave a truncated file behind for the caller to find
        remove(outfile)
        return False

    return remove(filename)


def gunzip(filename):
    """
    Decompress a .gz file in place and delete the original.

    Parameters
    ----------
    filename : str
        name of the .gz file

    Returns
    -------
    bool
        True on success
    """
    return _decompress(filename, '.gz', _open_gz)


def unxz(filename):
    """
    Decompress a .xz file in place and delete the original.

    Parameters
    ----------
    filename : str
        name of the .xz file

    Returns
    -------
    bool
        True on success
    """
    return _decompress(filename, '.xz', _open_xz)


def uncompress(filename):
    """
    Decompress a .Z file in place and delete the original.

    Parameters
    ----------
    filename : str
        name of the .Z file

    Returns
    -------
    bool
        True on success
    """
    try:
        import ncompress  # noqa: F401
    except ImportError:
        print('Reading .Z files requires the ncompress package : pip install ncompress')
        return False

    return _decompress(filename, '.Z', _open_z)


def decompress(filename):
    """
    Decompress a file in place, picking the method from its extension.

    For call sites that are handed either a .Z or a .gz depending on how old
    the data is, and would otherwise have to keep track of which.

    Parameters
    ----------
    filename : str
        name of the compressed file

    Returns
    -------
    bool
        True on success
    """
    if filename.endswith('.gz'):
        return gunzip(filename)
    if filename.endswith('.xz'):
        return unxz(filename)
    if filename.endswith('.Z'):
        return uncompress(filename)

    print('Do not know how to decompress ' + filename)
    return False


def unzip(filename, destination='.'):
    """
    Extract a zip archive, as unzip does.

    Unlike the decompression functions the archive is kept, again as unzip
    does. Note that unzip extracts into the working directory, not next to the
    archive, so that is the default here too.

    Parameters
    ----------
    filename : str
        name of the .zip file
    destination : str, optional
        directory to extract into, the working directory by default

    Returns
    -------
    bool
        True on success
    """
    try:
        with zipfile.ZipFile(filename) as archive:
            archive.extractall(destination)
    except (OSError, zipfile.BadZipFile) as e:
        print('Could not unzip ' + filename + ' : ' + str(e))
        return False

    return True


def gzip_file(filename):
    """
    Compress a file to filename.gz and delete the original, as gzip does.

    compresslevel 6 is the gzip command line default.

    Parameters
    ----------
    filename : str
        name of the file to compress

    Returns
    -------
    bool
        True on success
    """
    outfile = filename + '.gz'
    try:
        with open(filename, 'rb') as f_in, gzip.open(outfile, 'wb', compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    except OSError as e:
        print('Could not compress ' + filename + ' : ' + str(e))
        remove(outfile)
        return False

    return remove(filename)


def move(src, dst):
    """
    Move a file, overwriting the destination if it exists, as mv -f does.

    If dst is an existing directory the file keeps its name, again as mv does.
    os.replace is tried first because it is atomic, but it cannot move between
    filesystems, which is exactly what the Docker bind mount layout asks for
    (issue 417), so it falls back to shutil.move. shutil.move will not overwrite
    an existing file on Windows, so the destination is removed first.

    Parameters
    ----------
    src : str
        file to move
    dst : str
        destination file or directory

    Returns
    -------
    bool
        True on success
    """
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))

    try:
        os.replace(src, dst)
        return True
    except OSError:
        pass

    try:
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
    except OSError as e:
        print('Could not move ' + src + ' to ' + dst + ' : ' + str(e))
        return False

    return True


def copy(src, dst):
    """
    Copy a file, as cp does. dst may be a directory.

    shutil.copy raises where cp merely returned a non-zero status, including
    SameFileError when the two paths are the same file, so the exceptions are
    swallowed here to keep the behaviour these call sites already had.

    Parameters
    ----------
    src : str
        file to copy
    dst : str
        destination file or directory

    Returns
    -------
    bool
        True on success
    """
    try:
        shutil.copy(src, dst)
    except (OSError, shutil.SameFileError) as e:
        print('Could not copy ' + src + ' to ' + dst + ' : ' + str(e))
        return False

    return True


def remove(filename):
    """
    Delete a file, tolerating one that is not there, as rm -f does.

    Parameters
    ----------
    filename : str
        file to delete

    Returns
    -------
    bool
        True if the file is gone
    """
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
    except OSError as e:
        print('Could not remove ' + filename + ' : ' + str(e))
        return False

    return True


def remove_glob(pattern):
    """
    Delete every file matching a wildcard pattern, as rm -f pattern does.

    Matching zero files is not an error.

    Parameters
    ----------
    pattern : str
        wildcard pattern, e.g. p1010010.25o*

    Returns
    -------
    bool
        True if every matching file was removed
    """
    allgood = True
    for filename in glob.glob(pattern):
        if not remove(filename):
            allgood = False

    return allgood


def remove_tree(pattern):
    """
    Delete every file or directory matching a wildcard pattern, as rm -rf does.

    Directories go recursively. Matching zero paths is not an error.

    Parameters
    ----------
    pattern : str
        wildcard pattern, e.g. gnss/data/highrate/2025/011/25d/*

    Returns
    -------
    bool
        True if everything that matched was removed
    """
    allgood = True
    for name in glob.glob(pattern):
        if os.path.isdir(name) and not os.path.islink(name):
            try:
                shutil.rmtree(name)
            except OSError as e:
                print('Could not remove directory ' + name + ' : ' + str(e))
                allgood = False
        elif not remove(name):
            allgood = False

    return allgood
