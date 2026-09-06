"""Metal-compatible stubs for CUDA libdevice functions.

Metal has no libdevice. These are Triton JIT composites over tl.* and
tl.math.* primitives.
"""
import triton
import triton.language as tl

# ── Direct mappings (tl.* → libdevice) ──────────────────────────────────

DIRECT = {
    'exp': tl.exp,
    'exp2': tl.exp2,
    'log': tl.log,
    'log2': tl.log2,
    'sin': tl.sin,
    'cos': tl.cos,
    'sqrt': tl.sqrt,
    'abs': tl.abs,
    'fabs': tl.abs,
}

for _name in ['floor', 'ceil', 'rsqrt', 'erf', 'exp2', 'log2', 'div_rn']:
    _fn = getattr(tl.math, _name, None)
    if _fn is not None:
        DIRECT[_name] = _fn

# ── Composite stubs ─────────────────────────────────────────────────────


@triton.jit
def _log1p(x):
    # log(1+x) loses the whole mantissa once x is small enough that 1+x rounds:
    # rescaling by u = (1+x)-x recovers the discarded low bits.
    u = 1.0 + x
    d = u - 1.0
    val = tl.where(u == 1.0, x, tl.log(u) * (x / tl.where(d == 0.0, 1.0, d)))
    # Once u saturates the rescaling reads inf/inf; the limit is the log itself.
    return tl.where(u == float("inf"), float("inf"), val)


@triton.jit
def _exp10(x):
    return tl.exp2(x * 3.321928094887362)


@triton.jit
def _log10(x):
    return tl.log2(x) * 0.30102999566398120


@triton.jit
def _cbrt(x):
    # exp(log(x)/3) is nan for x < 0; cbrt is defined on all reals.
    ax = tl.abs(x)
    r = tl.exp(tl.log(tl.where(ax == 0.0, 1.0, ax)) / 3.0)
    r = tl.where(ax == 0.0, 0.0, r)
    return tl.where(x < 0.0, -r, r)


@triton.jit
def _round_to_int(y):
    # Round-half-away-from-zero to the nearest integral float value.
    return tl.where(y >= 0.0, tl.floor(y + 0.5), tl.ceil(y - 0.5))


@triton.jit
def _pow_mag(ax, y):
    # |ax|**y via compensated product (Metal has no f64). ax**y =
    # 2**(y*log2(ax)); t = y*log2(ax) kept as a (hi, lo) double-single pair,
    # then 2**t = 2**ki * 2**frac to keep the exp2 argument small.
    lg = tl.log2(ax)
    hi = y * lg
    e = tl.math.fma(y, lg, -hi)
    t = hi + e
    ki = _round_to_int(t)
    tf = (hi - ki) + e
    return tl.math.exp2(tf) * tl.math.exp2(ki)


@triton.jit
def _pow(x, y):
    # x < 0: |x|**y, negated for odd integer y, NaN for non-integer y.
    ax = tl.abs(x)
    mag = _pow_mag(ax, y)
    # x == 0: _pow_mag would compute fma(y, -inf, +inf) = NaN for every y.
    zero_mag = tl.where(y == 0.0, 1.0, tl.where(y < 0.0, float("inf"), 0.0))
    mag = tl.where(ax == 0.0, zero_mag, mag)
    # x == +-inf: same hole, signs flipped.
    inf_mag = tl.where(y < 0.0, 0.0, float("inf"))
    mag = tl.where(ax == float("inf"), inf_mag, mag)
    mag = tl.where(y == 0.0, 1.0, mag)
    y_rounded = _round_to_int(y)
    is_int = y_rounded == y
    odd = (y_rounded * 0.5 - tl.floor(y_rounded * 0.5)) != 0.0
    neg_sign = tl.where(odd, -1.0, 1.0)
    nan = float("nan")
    neg_result = tl.where(is_int, neg_sign * mag, nan)
    return tl.where(x < 0.0, neg_result, mag)


@triton.jit
def _tan(x):
    return tl.sin(x) / tl.cos(x)


@triton.jit
def _tanh(x):
    # Sign-folded form overflows for large |x|; small |x| uses u/(2+u) with
    # u = exp(2|x|)-1, cancellation-free as u -> 0.
    ax = tl.abs(x)
    e = tl.exp(ax + ax)
    big = 1.0 - 2.0 / (e + 1.0)
    u = e - 1.0
    small = u / (u + 2.0)
    t = tl.where(ax > 0.5, big, small)
    t = tl.where(ax < 1e-4, ax, t)
    return tl.where(x < 0.0, -t, t)


@triton.jit
def _sinh(x):
    # expm1's rescaling keeps small |x| exact; sign fold avoids overflow.
    ax = tl.abs(x)
    u = _expm1(ax)
    # u/(u+1) tends to 1, but spelled directly it is inf/inf once u saturates.
    ratio = tl.where(u == float("inf"), 1.0, u / (u + 1.0))
    mag = 0.5 * (u + ratio)
    return tl.where(x < 0.0, -mag, mag)


@triton.jit
def _cosh(x):
    ax = tl.abs(x)
    e = tl.exp(ax)
    return 0.5 * (e + 1.0 / e)


@triton.jit
def _asinh(x):
    # log1p on x + x^2/(sqrt(x^2+1)+1), cancellation-free as x -> 0.
    ax = tl.abs(x)
    # ax*ax overflows long before asinh does, and its limit is log(2x).
    big = ax > 1.0e18
    axs = tl.where(big, 1.0, ax)
    s = tl.sqrt(axs * axs + 1.0)
    mag = tl.where(big,
                   tl.log(ax) + 0.6931471805599453,
                   _log1p(axs + axs * axs / (s + 1.0)))
    return tl.where(x < 0.0, -mag, mag)


@triton.jit
def _acosh(x):
    # Domain x >= 1; log1p on (x-1) + sqrt((x-1)*(x+1)) avoids precision loss
    # near x -> 1.
    # d*(x+1) overflows long before acosh does, and its limit is log(2x).
    big = x > 1.0e18
    xs = tl.where(big, 2.0, x)
    d = xs - 1.0
    val = tl.where(big,
                   tl.log(x) + 0.6931471805599453,
                   _log1p(d + tl.sqrt(d * (xs + 1.0))))
    return tl.where(x < 1.0, float("nan"), val)


@triton.jit
def _atanh(x):
    # Domain |x| < 1; log1p of 2x/(1-x) avoids cancellation for small |x|.
    return 0.5 * _log1p(2.0 * x / (1.0 - x))


@triton.jit
def _atan(x):
    # Range-reduce to |x| <= 1 via atan(x) = pi/2 - atan(1/x), minimax poly.
    ax = tl.abs(x)
    inv = ax > 1.0
    z = tl.where(inv, 1.0 / ax, ax)
    z2 = z * z
    p = 0.0028662257
    p = -0.0161657367 + p * z2
    p = 0.0429096138 + p * z2
    p = -0.0752896400 + p * z2
    p = 0.1065626393 + p * z2
    p = -0.1420889944 + p * z2
    p = 0.1999355085 + p * z2
    p = -0.3333314528 + p * z2
    r = z + z * z2 * p
    r = tl.where(inv, 1.5707963267948966 - r, r)
    return tl.where(x < 0.0, -r, r)


@triton.jit
def _atan2(y, x):
    pi = 3.141592653589793
    halfpi = 1.5707963267948966
    return tl.where(
        x > 0, _atan(y / x),
        tl.where(x < 0,
                 _atan(y / x) + tl.where(y >= 0, pi, -pi),
                 tl.where(y > 0, halfpi, tl.where(y < 0, -halfpi, 0.0))))


@triton.jit
def _asin(x):
    return _atan2(x, tl.sqrt(tl.maximum(1.0 - x * x, 0.0)))


@triton.jit
def _acos(x):
    return _atan2(tl.sqrt(tl.maximum(1.0 - x * x, 0.0)), x)


@triton.jit
def _fmod(x, y):
    # C fmod: quotient truncated toward zero.
    q = x / y
    t = tl.where(q >= 0.0, tl.math.floor(q), tl.math.ceil(q))
    return x - t * y


@triton.jit
def _rint(x):
    # Round half to even.
    f = tl.math.floor(x)
    d = x - f
    up = tl.where(d == 0.5, (f - 2.0 * tl.math.floor(f * 0.5)) == 1.0, d > 0.5)
    return tl.where(up, f + 1.0, f)


@triton.jit
def _erfc(x):
    # 1 - erf(x) cancels once erf(x) -> 1, so it is used only for
    # |x| <= 0.927734375. The tail is a degree-9 Remez fit.
    ax = tl.abs(x)
    t = tl.minimum(ax, 10.5)
    q = (t - 4.0) / (t + 4.0)
    p = 5.271875e-03
    p = p * q + -1.6534764e-02
    p = p * q + 3.702093e-02
    p = p * q + -6.6275224e-02
    p = p * q + 9.375815e-02
    p = p * q + -1.01042934e-01
    p = p * q + 6.809548e-02
    p = p * q + 1.5379757e-02
    p = p * q + -1.396211e-01
    p = p * q + 2.3299512e-01
    s = t * t
    r = tl.math.fma(t, t, -s)
    e = tl.exp(-s)
    e = tl.math.fma(-e, r, e)
    tail = (1.0 + p) / (2.0 * t + 1.0) * e
    near = 1.0 - tl.math.erf(x)
    big = tl.where(x >= 0.0, tail, 2.0 - tail)
    # min(nan, 10.5) is 10.5 on Metal, so NaN must be special-cased here.
    return tl.where(x != x, x, tl.where(ax <= 0.927734375, near, big))


@triton.jit
def _expm1(x):
    u = tl.exp(x)
    d = u - 1.0
    lg = tl.log(tl.where(u == 0.0, 1.0, u))
    val = tl.where(u == 1.0, x, d * (x / tl.where(lg == 0.0, 1.0, lg)))
    # u saturates at both ends: `d * x` is +inf where the limit is -1, and
    # `inf * (inf / inf)` is nan where it is +inf.
    val = tl.where(u == 0.0, -1.0, val)
    return tl.where(u == float("inf"), float("inf"), val)


@triton.jit
def _erfcx(x):
    # exp(x^2)*erfc(x); positive tail folds the exponential into erfc's own
    # scaled form since exp(x*x) overflows before erfcx does.
    ax = tl.abs(x)
    t = 1.0 / (1.0 + 0.5 * ax)
    poly = (-1.26551223 + t * (1.00002368 + t *
                               (0.37409196 + t *
                                (0.09678418 + t *
                                 (-0.18628806 + t *
                                  (0.27886807 + t *
                                   (-1.13520398 + t *
                                    (1.48851587 + t *
                                     (-0.82215223 + t * 0.17087277)))))))))
    scaled = t * tl.exp(poly)
    return tl.where(x >= 0.5, scaled, tl.exp(x * x) * _erfc(x))


@triton.jit
def _lgamma_pos(x):
    # Lanczos approximation (g=5) for x > 0; log|Gamma(x)|.
    c0 = 1.000000000190015
    c1 = 76.18009172947146
    c2 = -86.50532032941677
    c3 = 24.01409824083091
    c4 = -1.231739572450155
    c5 = 0.1208650973866179e-2
    c6 = -0.5395239384953e-5
    xm1 = x - 1.0
    ser = (c0 + c1 / (xm1 + 1.0) + c2 / (xm1 + 2.0) + c3 / (xm1 + 3.0) + c4 /
           (xm1 + 4.0) + c5 / (xm1 + 5.0) + c6 / (xm1 + 6.0))
    tmp = xm1 + 5.5
    return (xm1 + 0.5) * tl.log(tmp) - tmp + tl.log(2.5066282746310005 * ser)


@triton.jit
def _lgamma(x):
    pi = 3.141592653589793
    pos = _lgamma_pos(x)
    refl = tl.log(pi / tl.abs(tl.sin(pi * x))) - _lgamma_pos(1.0 - x)
    # Gamma has a pole at every non-positive integer. sin(pi*x) only rounds to
    # zero there, so the reflection returns a finite value where inf is due.
    pole = (x <= 0.0) & (x == tl.floor(x))
    return tl.where(pole, float("inf"), tl.where(x > 0.0, pos, refl))


@triton.jit
def _erfinv(x):
    # Rational approximation (Giles 2010).
    w = -tl.log((1.0 - x) * (1.0 + x))
    w1 = w - 2.5
    p1 = 2.81022636e-08
    p1 = 3.43273939e-07 + p1 * w1
    p1 = -3.5233877e-06 + p1 * w1
    p1 = -4.39150654e-06 + p1 * w1
    p1 = 0.00021858087 + p1 * w1
    p1 = -0.00125372503 + p1 * w1
    p1 = -0.00417768164 + p1 * w1
    p1 = 0.246640727 + p1 * w1
    p1 = 1.50140941 + p1 * w1
    ws = tl.sqrt(w) - 3.0
    p2 = -0.000200214257
    p2 = 0.000100950558 + p2 * ws
    p2 = 0.00134934322 + p2 * ws
    p2 = -0.00367342844 + p2 * ws
    p2 = 0.00573950773 + p2 * ws
    p2 = -0.0076224613 + p2 * ws
    p2 = 0.00943887047 + p2 * ws
    p2 = 1.00167406 + p2 * ws
    p2 = 2.83297682 + p2 * ws
    p = tl.where(w < 5.0, p1, p2)
    # At |x| == 1 the log is +inf and p2 diverges negative, so `p * x` lands on
    # the wrong signed infinity.
    edge = tl.where(x > 0.0, float("inf"), float("-inf"))
    return tl.where(tl.abs(x) == 1.0, edge, p * x)


@triton.jit
def _trunc(x):
    return tl.where(x >= 0, tl.math.floor(x), tl.math.ceil(x))


@triton.jit
def _signbit(x):
    # 1/(-0.0) = -inf < 0 catches -0.0 too.
    return ((x < 0.0) | (1.0 / x == float('-inf'))).to(tl.int32)


@triton.jit
def _isinf(x):
    # Bit-level test: fast-math folds away x==inf / x!=x.
    bits = x.to(tl.float32).to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    return bits == 0x7F800000


@triton.jit
def _isnan(x):
    bits = x.to(tl.float32).to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    return bits > 0x7F800000


@triton.jit
def _finitef(x):
    bits = x.to(tl.float32).to(tl.uint32, bitcast=True) & 0x7F800000
    return bits != 0x7F800000


@triton.jit
def _div_rz(x, y):
    return _trunc(x / y)


@triton.jit
def _fast_dividef(x, y):
    return x / y


@triton.jit
def _mul_rn(x, y):
    # +0.0 prevents contraction into an FMA with any later add.
    return tl.fma(x, y, 0.0)


@triton.jit
def _fast_gelu(x):
    return 0.5 * x * (1.0 + _tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


@triton.jit
def _cyl_bessel_i0(x):
    # Modified Bessel I0, A&S 9.8.1/9.8.2. Split at 3.75: polynomial for
    # small, asymptotic exp(ax)/sqrt(ax)*poly for large.
    ax = tl.abs(x)
    small = ax < 3.75
    ts = (ax / 3.75) * (ax / 3.75)
    ps = 0.0045813
    ps = 0.0360768 + ps * ts
    ps = 0.2659732 + ps * ts
    ps = 1.2067492 + ps * ts
    ps = 3.0899424 + ps * ts
    ps = 3.5156229 + ps * ts
    ps = 1.0 + ps * ts
    # Guard the division so the unused branch never sees ax==0.
    axl = tl.where(small, 3.75, ax)
    tl_ = 3.75 / axl
    pl = 0.00392377
    pl = -0.01647633 + pl * tl_
    pl = 0.02635537 + pl * tl_
    pl = -0.02057706 + pl * tl_
    pl = 0.00916281 + pl * tl_
    pl = -0.00157565 + pl * tl_
    pl = 0.00225319 + pl * tl_
    pl = 0.01328592 + pl * tl_
    pl = 0.39894228 + pl * tl_
    large = (tl.exp(axl) / tl.sqrt(axl)) * pl
    return tl.where(small, ps, large)


@triton.jit
def _cyl_bessel_i1(x):
    # Modified Bessel I1, A&S 9.8.3/9.8.4. Odd: compute for |x|, restore sign.
    ax = tl.abs(x)
    small = ax < 3.75
    ts = (ax / 3.75) * (ax / 3.75)
    ps = 0.00032411
    ps = 0.00301532 + ps * ts
    ps = 0.02658733 + ps * ts
    ps = 0.15084934 + ps * ts
    ps = 0.51498869 + ps * ts
    ps = 0.87890594 + ps * ts
    ps = 0.5 + ps * ts
    small_val = ax * ps
    axl = tl.where(small, 3.75, ax)
    tl_ = 3.75 / axl
    pl = -0.00420059
    pl = 0.01787654 + pl * tl_
    pl = -0.02895312 + pl * tl_
    pl = 0.02282967 + pl * tl_
    pl = -0.01031555 + pl * tl_
    pl = 0.00163801 + pl * tl_
    pl = -0.00362018 + pl * tl_
    pl = -0.03988024 + pl * tl_
    pl = 0.39894228 + pl * tl_
    large_val = (tl.exp(axl) / tl.sqrt(axl)) * pl
    mag = tl.where(small, small_val, large_val)
    return tl.where(x < 0.0, -mag, mag)


@triton.jit
def _hypot(x, y):
    return tl.sqrt(x * x + y * y)


@triton.jit
def _copysign(x, y):
    ax = tl.abs(x)
    neg = (y < 0.0) | (1.0 / y == float('-inf'))
    return tl.where(neg, -ax, ax)


@triton.jit
def _j0(x):
    # Bessel J0, A&S 9.4.1/9.4.3. Small |x|<8: rational polynomial.
    # Large: amplitude/phase asymptotic.
    ax = tl.abs(x)
    small = ax < 8.0
    y = x * x
    p1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y *
                                                    (-11214424.18 + y *
                                                     (77392.33017 + y *
                                                      (-184.9052456)))))
    q1 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y *
                                                  (59272.64853 + y *
                                                   (267.8532712 + y))))
    small_val = p1 / q1
    axl = tl.where(small, 8.0, ax)
    z = 8.0 / axl
    y2 = z * z
    xx = axl - 0.785398164
    pa = 1.0 + y2 * (-0.1098628627e-2 + y2 *
                     (0.2734510407e-4 + y2 *
                      (-0.2073370639e-5 + y2 * 0.2093887211e-6)))
    pb = -0.1562499995e-1 + y2 * (0.1430488765e-3 + y2 *
                                  (-0.6911147651e-5 + y2 *
                                   (0.7621095161e-6 + y2 * (-0.934935152e-7))))
    large_val = tl.sqrt(
        0.636619772 / axl) * (tl.cos(xx) * pa - z * tl.sin(xx) * pb)
    return tl.where(small, small_val, large_val)


@triton.jit
def _j1(x):
    # Bessel J1, A&S 9.4.4/9.4.6. Odd function.
    ax = tl.abs(x)
    small = ax < 8.0
    y = x * x
    p1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y *
                                                        (-2972611.439 + y *
                                                         (15704.48260 + y *
                                                          (-30.16036606))))))
    q1 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y *
                                                   (99447.43394 + y *
                                                    (376.9991397 + y))))
    small_val = p1 / q1
    axl = tl.where(small, 8.0, ax)
    z = 8.0 / axl
    y2 = z * z
    xx = axl - 2.356194491
    pa = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4 + y2 *
                                         (0.2457520174e-5 + y2 *
                                          (-0.240337019e-6))))
    pb = 0.04687499995 + y2 * (-0.2002690873e-3 + y2 *
                               (0.8449199096e-5 + y2 *
                                (-0.88228987e-6 + y2 * 0.105787412e-6)))
    mag = tl.sqrt(0.636619772 / axl) * (tl.cos(xx) * pa - z * tl.sin(xx) * pb)
    large_val = tl.where(x < 0.0, -mag, mag)
    return tl.where(small, small_val, large_val)


@triton.jit
def _y0(x):
    # Bessel Y0 (second kind), A&S 9.4.2/9.4.3. Defined for x > 0.
    small = x < 8.0
    y = x * x
    p1 = -2957821389.0 + y * (7062834065.0 + y *
                              (-512359803.6 + y *
                               (10879881.29 + y *
                                (-86327.92757 + y * 228.4622733))))
    q1 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y *
                                                 (47447.26470 + y *
                                                  (226.1030244 + y))))
    xs = tl.where(small, x, 1.0)
    small_val = (p1 / q1) + 0.636619772 * _j0(x) * tl.log(xs)
    xl = tl.where(small, 8.0, x)
    z = 8.0 / xl
    y2 = z * z
    xx = xl - 0.785398164
    pa = 1.0 + y2 * (-0.1098628627e-2 + y2 *
                     (0.2734510407e-4 + y2 *
                      (-0.2073370639e-5 + y2 * 0.2093887211e-6)))
    pb = -0.1562499995e-1 + y2 * (0.1430488765e-3 + y2 *
                                  (-0.6911147651e-5 + y2 *
                                   (0.7621095161e-6 + y2 * (-0.934935152e-7))))
    large_val = tl.sqrt(
        0.636619772 / xl) * (tl.sin(xx) * pa + z * tl.cos(xx) * pb)
    return tl.where(small, small_val, large_val)


@triton.jit
def _y1(x):
    # Bessel Y1 (second kind), A&S 9.4.5/9.4.6. x > 0.
    small = x < 8.0
    y = x * x
    p1 = x * (-0.4900604943e13 + y *
              (0.1275274390e13 + y *
               (-0.5153438139e11 + y *
                (0.7349264551e9 + y *
                 (-0.4237922726e7 + y * 0.8511937935e4)))))
    q1 = 0.2499580570e14 + y * (0.4244419664e12 + y *
                                (0.3733650367e10 + y *
                                 (0.2245904002e8 + y *
                                  (0.1020426050e6 + y *
                                   (0.3549632885e3 + y)))))
    xs = tl.where(small, x, 1.0)
    small_val = (p1 / q1) + 0.636619772 * (_j1(x) * tl.log(xs) - 1.0 / xs)
    xl = tl.where(small, 8.0, x)
    z = 8.0 / xl
    y2 = z * z
    xx = xl - 2.356194491
    pa = 1.0 + y2 * (0.183105e-2 + y2 * (-0.3516396496e-4 + y2 *
                                         (0.2457520174e-5 + y2 *
                                          (-0.240337019e-6))))
    pb = 0.04687499995 + y2 * (-0.2002690873e-3 + y2 *
                               (0.8449199096e-5 + y2 *
                                (-0.88228987e-6 + y2 * 0.105787412e-6)))
    large_val = tl.sqrt(
        0.636619772 / xl) * (tl.sin(xx) * pa + z * tl.cos(xx) * pb)
    # j1(0) is 0 and log(0) is -inf, so the small-x arm evaluates 0 * -inf and
    # poisons the pole to nan.
    val = tl.where(small, small_val, large_val)
    return tl.where(x == 0.0, float("-inf"), val)


@triton.jit
def _nextafter(x, y):
    # Step one ULP via the integer bit pattern; sign-magnitude means the
    # negative side steps in the opposite direction.
    xb = x.to(tl.float32).to(tl.int32, bitcast=True)
    up = y > x
    # -0 and +0 are adjacent encodings 0x80000000 and 0, so zero-crossing
    # can't be handled by stepping the magnitude.
    step = tl.where(x < 0.0, tl.where(up, -1, 1), tl.where(up, 1, -1))
    stepped = (xb + step).to(tl.float32, bitcast=True)
    denorm = tl.full(stepped.shape, 1.4012984643e-45, tl.float32)
    from_zero = tl.where(up, denorm, -denorm)
    r = tl.where(x == 0.0, from_zero, stepped)
    return tl.where(x == y, y.to(tl.float32), r).to(x.dtype)


@triton.jit
def _ilogb(x):
    # Read the exponent field directly. A subnormal carries a zero field, so it
    # is scaled into the normal range first and the shift taken back off.
    ax = tl.abs(x)
    sub = (ax < 1.1754943508222875e-38) & (ax > 0.0)
    scaled = tl.where(sub, ax * 16777216.0, ax)
    bits = scaled.to(tl.float32, bitcast=True).to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127
    return tl.where(sub, e - 24, e).to(tl.int32)


@triton.jit
def _ldexp(x, n):
    # Build the scale by writing the exponent field directly (exact; exp2
    # would round). Apply in two halves so large |n| doesn't overflow the
    # intermediate even when the final product is finite.
    k = tl.minimum(tl.maximum(n, -252), 252)
    h = k // 2
    s1 = ((h + 127) << 23).to(tl.float32, bitcast=True)
    s2 = ((k - h + 127) << 23).to(tl.float32, bitcast=True)
    return x * s1 * s2


@triton.jit
def _popc(x):
    u = x.to(tl.uint32)
    u = u - ((u >> 1) & 0x55555555)
    u = (u & 0x33333333) + ((u >> 2) & 0x33333333)
    u = (u + (u >> 4)) & 0x0F0F0F0F
    return (((u * 0x01010101) >> 24) & 0x3F).to(tl.int32)


@triton.jit
def _clz(x):
    u = x.to(tl.uint32)
    u = u | (u >> 1)
    u = u | (u >> 2)
    u = u | (u >> 4)
    u = u | (u >> 8)
    u = u | (u >> 16)
    return _popc((~u).to(tl.uint32))


COMPOSITES = {
    'clz': _clz,
    'popc': _popc,
    'log1p': _log1p,
    'exp10': _exp10,
    'log10': _log10,
    'cbrt': _cbrt,
    'pow': _pow,
    'tan': _tan,
    'tanh': _tanh,
    'sinh': _sinh,
    'cosh': _cosh,
    'asinh': _asinh,
    'acosh': _acosh,
    'atanh': _atanh,
    'acos': _acos,
    'asin': _asin,
    'atan': _atan,
    'atan2': _atan2,
    'fmod': _fmod,
    'hypot': _hypot,
    'copysign': _copysign,
    'nextafter': _nextafter,
    'ldexp': _ldexp,
    'ilogb': _ilogb,
    'rint': _rint,
    'nearbyint': _rint,
    'llrint': _rint,
    'lrint': _rint,
    'erfc': _erfc,
    'erfcx': _erfcx,
    'expm1': _expm1,
    'lgamma': _lgamma,
    'erfinv': _erfinv,
    'trunc': _trunc,
    'signbit': _signbit,
    'isinf': _isinf,
    'isnan': _isnan,
    'finitef': _finitef,
    'isfinited': _finitef,
    'div_rz': _div_rz,
    'mul_rn': _mul_rn,
    'fast_dividef': _fast_dividef,
    'fast_tanh': _tanh,
    'fast_erf': DIRECT.get('erf', _tanh),
    'fast_gelu': _fast_gelu,
    'cyl_bessel_i0': _cyl_bessel_i0,
    'cyl_bessel_i1': _cyl_bessel_i1,
    'j0': _j0,
    'j1': _j1,
    'y0': _y0,
    'y1': _y1,
}

ALL_STUBS = {**DIRECT, **COMPOSITES}
