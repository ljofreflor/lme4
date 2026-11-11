"""Minimal reproduction of R's default RNG (Mersenne-Twister with R's
``set.seed()`` scrambling, src/main/RNG.c) to recreate simulated test data
from lme4's R tests: ``runif()`` and ``sample(..., replace=TRUE)`` (both
the pre-3.6 "Rounding" and the default "Rejection" sample.kind)."""
import math

import numpy as np

_N, _M = 624, 397
_MATRIX_A = 0x9908b0df
_UPPER, _LOWER = 0x80000000, 0x7fffffff
_I2_32M1 = 2.328306437080797e-10


class RRNG:
    def __init__(self, seed, sample_kind="Rejection"):
        self.sample_kind = sample_kind
        self.set_seed(seed)

    def set_seed(self, seed):
        s = seed & 0xffffffff
        for _ in range(50):
            s = (69069 * s + 1) & 0xffffffff
        iseed = []
        for _ in range(625):
            s = (69069 * s + 1) & 0xffffffff
            iseed.append(s)
        self.mt = iseed[1:]
        self.mti = _N            # FixupSeeds: dummy[0] = 624

    def _genrand(self):
        mt = self.mt
        if self.mti >= _N:
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)
        return (y & 0xffffffff) * 2.3283064365386963e-10

    def unif_rand(self):
        v = self._genrand()
        if v <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return v

    def runif(self, n, min=0.0, max=1.0):
        return np.array([min + (max - min) * self.unif_rand() for _ in range(n)])

    def _rbits(self, bits):
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return v & ((1 << bits) - 1)

    def unif_index(self, dn):
        if self.sample_kind == "Rounding":
            return int(math.floor(dn * self.unif_rand()))
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample(self, x, size, replace=True):
        if not replace:
            raise NotImplementedError
        n = len(x)
        return [x[self.unif_index(n)] for _ in range(size)]
