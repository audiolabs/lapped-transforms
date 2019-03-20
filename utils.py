import numpy as np
import scipy.signal as ss

def mdct(N):
    """
    Create MDCT matrix

    Parameters
    ---------
    N : int
        Framelength, number of output coefficients (or rows).

    Returns
    -------
    kernel : array_like
        MDCT matrix

    """
    M = N * 2
    shift = -N // 2
    n, k = np.meshgrid(
        np.arange(M, dtype=float), np.arange(N, dtype=float)
    )
    return np.cos(
        np.pi / N * (n + shift + 1 / 2) * (k + 1 / 2)
    ) / np.sqrt(N / 2)


def dct4(N, M=None):
    """
    Create DCT-IV matrix

    Parameters
    ---------
    N : int
        Framelength, number of output coefficients (or rows).

    Returns
    -------
    kernel : array_like
        DCT-IV matrix

    """
    if M is None:
        M = N
    n, k = np.meshgrid(
        np.arange(M, dtype=float), np.arange(N, dtype=float)
    )
    return np.cos(
        np.pi / N * (n + 1 / 2) * (k + 1 / 2)
    ) / np.sqrt(N / 2)


def freq(data, axis=-1):
    """
    Calculate frequency response of filterbank by analysing the FFT of the
    matrix.

    Parameters
    ---------
    data : array_like
        Transform matrix

    Returns
    -------
    out : array_like
        Frequency response of matrix

    """
    return np.abs(np.fft.rfft(data, axis=axis, n=data.shape[axis] * 2))


def env(data, axis=-1):
    """
    Calculate impulse response of filterbank by analysing the Hilbert transform
    of the matrix. Be cautious with discontinuities at the edges of the matrix.

    Parameters
    ---------
    data : array_like
        Transform matrix

    Returns
    -------a
    out : array_like
        Ipmulse response of matrix

    """
    return np.abs(ss.hilbert(data, axis=axis))


def make_twoframe(data, trim=False):
    """
    Create two-frame variant of transform matrix.

    Parameters
    ---------
    data : array_like
        Non-square transform matrix. Usually a MDCT folding matrix
    trim : boolean
        Set to :code:`True` to only create the "center" of the matrix.

    Returns
    -------
    out : array_like
        Two-frame variant of transform matrix.

    """
    N = data.shape[0] // 2
    out = np.eye(data.shape[1])
    out[N:2 * N, N:3 * N] = data[N:, 2 * N:]
    out[2 * N:3 * N, N:3 * N] = data[:N, :2 * N]

    if trim:
        out = out[N:-N, N:-N]
    return out


def lap(x, L=2, copy=True):
    """
    Create lapped view of array. By default a copy of `x` is taken prior to
    lapping, as to not accidentally change the input array values.

    Parameters
    ---------
    x : array_like
        Framed input signal. Framelength is inferred from the last dimension.
    L : int
        Overlap factor. The factor by how much the last dimension will be
        virtually extended.
    copy : boolean
        Create copy before lapping signal. Useful if you don't want the following
        transform operations to have an effect on the input array, too.

    Returns
    -------
    out : array_like
        Lapped view on input signal. None of the values are duplicated in
        space, but instead L elements are referring to the same value.

    """
    if copy:
        x = x.copy()

    return np.lib.stride_tricks.as_strided(
        x,
        shape=(x.shape[0] - L + 1, x.shape[1] * L),
        strides=x.strides
    )


def unlap(x, L=2, copy=True):
    """
    Unlap array by resetting strides to non-overlapping frames.
    By default a copy of `x` is created, as to not accidentally change the
    input array values.

    Parameters
    ---------
    x : array_like
        Lapped input signal. Framelength is inferred from the last dimension and L.
    L : int
        Overlap factor. The factor by how much the last dimension will be
        shrunk.
    copy : boolean
        Create copy after unlapping signal. Useful if you don't want the following
        transform operations to have an effect on the input array, too.

    Returns
    -------
    out : array_like
        Unlapped view on input signal. Instead of L elements referring to the same
        value in memory, only one element will.

    """
    outval = np.lib.stride_tricks.as_strided(
        x,
        shape=(x.shape[0] + L - 1, x.shape[1] // L),
        strides=x.strides
    )

    if copy:
        outval = outval.copy()
    return outval



def transform(x, T):
    """
    Transform array. This is basically a matrix-matrix product,
    but because `x` may be lapped, we have to perform this in a loop.

    Parameters
    ---------
    x : array_like
        Input signal. May be lapped or unlapped. The last dimension is transformed.
        Be aware that the input signal is transformed in place, so the input array
        will be changed too.
    T : array_like
        Transform matrix to apply to the last dimension.

    Returns
    -------
    x : array_like
        The input signal.

    """
    for i in range(len(x)):
        x[i, :] = T @ x[i, :]
    return x


def flatten(x, L=2, copy=True):
    """
    Completely flatten lapped array without duplicating data.

    Parameters
    ---------
    x : array_like
        Lapped input signal. Framelength is inferred from the last dimension and L.
    L : int
        Overlap factor. The factor by how much the last dimension will be
        shrunk.
    copy : boolean
        Create copy after unlapping signal. Useful if you don't want the following
        transform operations to have an effect on the input array, too.

    Returns
    -------
    out : array_like
        Unlapped and flattened 1D view of input signal.

    """
    return unlap(x, L, copy=copy).ravel()


def lap_like(x, y):
    """
    Lap unlapped array to match other array.

    Parameters
    ---------
    x : array_like
        Unlapped signal.
    y : array_like
        Lapped signal. The overlap factor of this array will be used to create a
        lapped view of :code:`x`

    Returns
    -------
    out : array_like
        Lapped view of input signal.

    """
    return np.lib.stride_tricks.as_strided(
        x,
        shape=y.shape,
        strides=y.strides
    )


def copy(x, L=2):
    """
    Create a copy of a lapped array without duplicating data by flattening
    it first.

    Parameters
    ---------
    x : array_like
        Lapped input signal. Framelength is inferred from the last dimension and L.
    L : int
        Overlap factor.

    Returns
    -------
    out : array_like
        Lapped copy of input signal.

    """
    new = flatten(x, copy=True)
    return lap_like(new, x)
