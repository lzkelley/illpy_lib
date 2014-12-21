

import os


def getFileSize(fname, precision=1):
    byteSize = os.path.getsize(fname)
    return bytesString(byteSize, precision)


def bytesString(bytes, precision=1):
    """
    Return a humanized string representation of a number of bytes.
    
    Arguments
    ---------
    bytes : scalar, number of bytes
    precision : int, target precision in number of decimal places

    Examples
    --------
    >> humanize_bytes(1024*12342,2)
    '12.05 MB'

    """

    abbrevs = (
        (1<<50L, 'PB'),
        (1<<40L, 'TB'),
        (1<<30L, 'GB'),
        (1<<20L, 'MB'),
        (1<<10L, 'kB'),
        (1, 'bytes')
    )

    for factor, suffix in abbrevs:
        if bytes >= factor: break

    return '%.*f %s' % (precision, bytes / factor, suffix)
