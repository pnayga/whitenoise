"""
core/models.py — Theoretical MSD and PDF formulas for all SWNA models.

Implemented models (10): fbm, sin_half, cos_half, exponential, sine, cosine,
                          inc_gamma, bessel_j0_cos, bessel_jmu_nu, dna
Stubs (9):               exp_whittaker, bessel_K, hypergeom_F1, bessel_I,
                          hypergeom_3F2, csc_power, cot_power, bessel_pair,
                          bessel_pair2

References:
  Bernido non-Markovian book (draft), Table 2.1
  Violanda et al. (2019), Phys. Scr. 94, 125006  [dna model]
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, jv, gammainc

# ── Shared helpers ────────────────────────────────────────────────────────────

def _to_array(T) -> tuple[np.ndarray, bool]:
    """Return (T as 1-D float array, was_scalar)."""
    scalar = np.ndim(T) == 0
    return np.atleast_1d(np.asarray(T, dtype=float)), scalar


def _finalize_msd(val: np.ndarray, scalar: bool):
    """
    Replace non-positive or non-finite entries with nan.
    Return float if scalar input, else ndarray.
    """
    result = np.where(np.isfinite(val) & (val > 0), val, np.nan)
    if scalar:
        return float(result.flat[0])
    return result


def _pdf_from_sigma2(dx, sigma2: float) -> np.ndarray:
    """
    Gaussian PDF evaluated at dx with variance sigma2.
    Always returns ndarray. Returns all-nan if sigma2 is invalid.
    """
    dx_arr = np.atleast_1d(np.asarray(dx, dtype=float))   # ensure array input
    if not (np.isfinite(sigma2) and sigma2 > 0):           # guard: σ² must be positive and finite
        return np.full_like(dx_arr, np.nan, dtype=float)   # return all-NaN if MSD is invalid
    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma2)            # 1/√(2πσ²) — Gaussian normalization constant
    return norm * np.exp(-dx_arr ** 2 / (2.0 * sigma2))   # Gaussian PDF: N(0, σ²)


_NOT_IMPL_MSG = (
    "Model '{name}' is not yet implemented.\n"
    "Implemented: cosine, exponential, sine, fbm, dna, "
    "sin_half, cos_half, inc_gamma, bessel_j0_cos, bessel_jmu_nu\n"
    "Run wn.list_models() to see all 18 models and their status."
)


# ── Priority Model: cosine (Table 2.1 row 10) ─────────────────────────────────
# Published applications in Bernido group:
#   Elnar et al. (2021) — GBR coral bleaching, μ ≈ 4.64 (hyperballistic)
#   Elnar et al. (2024) — CO₂ Keeling curve, μ ≈ 0.91–0.97 (subdiffusive)
#   Calotes thesis (2024) — X-ray binary light curves, μ ∈ [0.50, 1.39]

def msd_cosine(T, mu: float, nu: float):
    """
    Theoretical MSD for the cosine model (Table 2.1 row 10).

    Formula::

        MSD(T) = sqrt(pi) * Gamma(mu) * cos(nu*T/2)
                 * J_{mu-1/2}(nu*T/2) * (T/nu)^(mu - 1/2)

    Parameters
    ----------
    T : float or np.ndarray
        Time lag(s). Positive values expected.
    mu : float
        Memory parameter.  mu < 1 subdiffusive, mu = 1 Brownian,
        mu > 1 superdiffusive.
    nu : float
        Characteristic frequency (rad per time unit).

    Returns
    -------
    float or np.ndarray
        MSD value(s).  Returns nan where cos(nu*T/2) <= 0 or where
        the result is non-positive / non-finite.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 10.
    """
    # Step 1: Normalize input — accept scalar, list, or array; remember if scalar
    T_arr, scalar = _to_array(T)

    # Step 2: Compute νT/2 — this is the shared argument passed to both cos() and the Bessel function
    arg = nu * T_arr / 2.0

    # Step 3: Evaluate cos(νT/2) — the oscillatory envelope of the cosine model
    cos_val = np.cos(arg)

    # Step 4: Pre-fill output with NaN — lags where cos(νT/2) ≤ 0 are physically undefined
    #         (the formula produces a negative MSD there, which has no meaning)
    result = np.full_like(T_arr, np.nan, dtype=float)
    mask = cos_val > 0   # only compute where cosine is strictly positive

    # Step 5: For valid lags, evaluate the full MSD formula:
    #         MSD(T) = √π · Γ(μ) · cos(νT/2) · J_{μ-1/2}(νT/2) · (T/ν)^(μ-1/2)
    if np.any(mask):
        T_m = T_arr[mask]
        bv  = jv(mu - 0.5, arg[mask])   # Bessel function of the first kind, order μ-1/2
        val = (
            np.sqrt(np.pi) * gamma(mu)   # √π · Γ(μ): amplitude prefactor from the white noise integral
            * cos_val[mask] * bv          # oscillatory × Bessel modulation
            * (T_m / nu) ** (mu - 0.5)   # power-law scaling controlled by memory parameter μ
        )
        # Step 6: Accept only positive finite values; replace anything else with NaN
        result[mask] = np.where(np.isfinite(val) & (val > 0), val, np.nan)

    # Step 7: Return a plain float if the original input was a scalar, else return the full array
    return float(result.flat[0]) if scalar else result


def pdf_cosine(dx, T: float, mu: float, nu: float) -> np.ndarray:
    """
    Theoretical PDF for the cosine model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = MSD_cosine(T, mu, nu))

    Parameters
    ----------
    dx : float or np.ndarray
        Displacement value(s).
    T : float
        Evaluation time (scalar).
    mu, nu : float
        Model parameters.

    Returns
    -------
    np.ndarray
        PDF values.  All-nan if sigma^2 is invalid.
    """
    sigma2 = msd_cosine(float(T), mu, nu)
    return _pdf_from_sigma2(dx, sigma2)


# ── Priority Model: exponential (Table 2.1 row 4) ────────────────────────────
# Published applications in Bernido group:
#   Roque et al. (2024) — Philippine earthquake inter-event times, μ ≈ 1.00–1.19
#   Toledo et al. (2024) — Solar sunspot number time series, μ ≈ 1.15

def msd_exponential(T, mu: float, beta: float):
    """
    Theoretical MSD for the exponential model (Table 2.1 row 4).

    Formula::

        MSD(T) = Gamma(mu) * beta^(-mu) * T^(mu - 1) * exp(-beta / T)

    Parameters
    ----------
    T : float or np.ndarray
        Time lag(s). Positive values expected.
    mu : float
        Memory parameter.
    beta : float
        Exponential decay rate (inverse time unit).

    Returns
    -------
    float or np.ndarray
        MSD value(s).  Returns nan for non-positive or non-finite results.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 4.
    """
    # Step 1: Normalize input — accept scalar, list, or array; remember if scalar
    T_arr, scalar = _to_array(T)

    # Step 2: Evaluate the full MSD formula in one expression:
    #         MSD(T) = Γ(μ) · β^(-μ) · T^(μ-1) · exp(-β/T)
    #
    #         Γ(μ)       — amplitude prefactor (from the white noise integral)
    #         β^(-μ)     — rescales amplitude by the decay rate β
    #         T^(μ-1)    — power-law memory: μ>1 superdiffusive, μ<1 subdiffusive
    #         exp(-β/T)  — suppresses MSD at very small T (short-lag regularization)
    val = (
        gamma(mu)
        * beta ** (-mu)
        * T_arr ** (mu - 1.0)
        * np.exp(-beta / T_arr)
    )

    # Step 3: Replace any non-positive or non-finite values with NaN, return result
    return _finalize_msd(val, scalar)


def pdf_exponential(dx, T: float, mu: float, beta: float) -> np.ndarray:
    """
    Theoretical PDF for the exponential model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = MSD_exponential(T, mu, beta))

    Parameters
    ----------
    dx : float or np.ndarray
    T : float
    mu, beta : float

    Returns
    -------
    np.ndarray
    """
    sigma2 = msd_exponential(float(T), mu, beta)
    return _pdf_from_sigma2(dx, sigma2)


# ── Priority Model: sine (Table 2.1 row 9) ───────────────────────────────────
# Published applications in Bernido group:
#   Tuñaco et al. (unpublished) — planktonic predator–prey system;
#     role of memory in population resilience.
# Structurally analogous to the cosine model (row 10) — sin replaces cos in the
# oscillatory envelope. Available as an alternative when cosine fit is poor.

def msd_sine(T, mu: float, nu: float):
    """
    Theoretical MSD for the sine model (Table 2.1 row 9).

    Formula::

        MSD(T) = sqrt(pi) * Gamma(mu) * sin(nu*T/2)
                 * J_{mu-1/2}(nu*T/2) * (T/nu)^(mu - 1/2)

    Parameters
    ----------
    T : float or np.ndarray
    mu : float
        Memory parameter.
    nu : float
        Characteristic frequency.

    Returns
    -------
    float or np.ndarray
        Returns nan where sin(nu*T/2) <= 0 or result is invalid.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 9.
    """
    # Step 1: Normalize input — accept scalar, list, or array; remember if scalar
    T_arr, scalar = _to_array(T)

    # Step 2: Compute νT/2 — shared argument for both sin() and the Bessel function
    arg = nu * T_arr / 2.0

    # Step 3: Evaluate sin(νT/2) — the oscillatory envelope of the sine model
    sin_val = np.sin(arg)

    # Step 4: Pre-fill output with NaN — lags where sin(νT/2) ≤ 0 are physically undefined
    result = np.full_like(T_arr, np.nan, dtype=float)
    mask = sin_val > 0   # only compute where sine is strictly positive

    # Step 5: For valid lags, evaluate the full MSD formula:
    #         MSD(T) = √π · Γ(μ) · sin(νT/2) · J_{μ-1/2}(νT/2) · (T/ν)^(μ-1/2)
    #         Same structure as cosine model — only the envelope (sin vs cos) differs
    if np.any(mask):
        T_m = T_arr[mask]
        bv  = jv(mu - 0.5, arg[mask])   # Bessel function of the first kind, order μ-1/2
        val = (
            np.sqrt(np.pi) * gamma(mu)   # √π · Γ(μ): amplitude prefactor
            * sin_val[mask] * bv          # sine oscillatory × Bessel modulation
            * (T_m / nu) ** (mu - 0.5)   # power-law scaling controlled by μ
        )
        # Step 6: Accept only positive finite values; replace anything else with NaN
        result[mask] = np.where(np.isfinite(val) & (val > 0), val, np.nan)

    # Step 7: Return a plain float if the original input was a scalar, else return the full array
    return float(result.flat[0]) if scalar else result


def pdf_sine(dx, T: float, mu: float, nu: float) -> np.ndarray:
    """
    Theoretical PDF for the sine model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = MSD_sine(T, mu, nu))

    Parameters
    ----------
    dx : float or np.ndarray
    T : float
    mu, nu : float

    Returns
    -------
    np.ndarray
    """
    sigma2 = msd_sine(float(T), mu, nu)
    return _pdf_from_sigma2(dx, sigma2)


# ── Priority Model: fbm (Table 2.1 row 1) ────────────────────────────────────
# Published applications in Bernido group: used as a classical benchmark model.
# Original fBm theory: Mandelbrot & Van Ness (1968), SIAM Rev. 10(4), 422–437.
# No specific Bernido-group dataset study identified using this model.

def msd_fbm(T, H: float):
    """
    Theoretical MSD for fractional Brownian motion (Table 2.1 row 1).

    Formula::

        MSD(T) = T^(2H)

    Parameters
    ----------
    T : float or np.ndarray
    H : float
        Hurst exponent.
        H = 0.5  -> ordinary Brownian motion (MSD = T).
        H > 0.5  -> superdiffusive (persistent).
        H < 0.5  -> subdiffusive (anti-persistent).

    Returns
    -------
    float or np.ndarray

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 1.
    """
    # Step 1: Normalize input — accept scalar, list, or array; remember if scalar
    T_arr, scalar = _to_array(T)

    # Step 2: Evaluate MSD(T) = T^(2H)
    #         H = 0.5 → T^1 = T (ordinary Brownian motion)
    #         H > 0.5 → grows faster than T (superdiffusive / persistent)
    #         H < 0.5 → grows slower than T (subdiffusive / anti-persistent)
    val = T_arr ** (2.0 * H)

    # Step 3: Replace any non-positive or non-finite values with NaN, return result
    return _finalize_msd(val, scalar)


def pdf_fbm(dx, T: float, H: float) -> np.ndarray:
    """
    Theoretical PDF for fractional Brownian motion.

    P(dx; T) = Gaussian(mean=0, sigma^2 = T^(2H))

    Parameters
    ----------
    dx : float or np.ndarray
    T : float
    H : float

    Returns
    -------
    np.ndarray
    """
    sigma2 = msd_fbm(float(T), H)
    return _pdf_from_sigma2(dx, sigma2)


# ── Published DNA model (Violanda et al. 2019) ───────────────────────────────
# Published applications in Bernido group:
#   Violanda, Bernido & Carpio-Bernido (2019) — nucleotide separation distances
#   in Synechococcus elongatus PCC 7942 bacterial genome.
#   Phys. Scr. 94, 125006.  DOI: 10.1088/1402-4896/ab2920
#   Results: a ≈ 5.21, b ≈ 0.0024, c ≈ 3.81 (nucleotide A in that organism).

def msd_dna(L, a: float, b: float, c: float):
    """
    MSD model for DNA nucleotide separation distances.

    Derived from the stochastic integral with exponentially decaying memory::

        x(L) = x₀ + ∫₀ᴸ exp(-b(L-s)/2) · ω(s) ds

    The resulting MSD takes the shifted exponential (plateau) form::

        MSD(L) = a - c · exp(-b · L)

    The curve rises from ``(a - c)`` at ``L = 0`` and asymptotes to the
    plateau ``a`` as ``L → ∞``, characteristic of restricted diffusion.

    Parameters
    ----------
    L : float or np.ndarray
        Occurrence number (analogous to time) — the number of intervening
        bases between successive occurrences of a nucleotide.
    a : float
        Plateau height.  MSD approaches ``a`` as L → ∞.  Must satisfy
        ``a > c > 0``.
    b : float
        Exponential decay rate (memory decay parameter, analogous to β
        in other models).
    c : float
        Amplitude of the exponential term.  Controls the rate of rise to
        the plateau.  Must satisfy ``c < a``.

    Returns
    -------
    float or np.ndarray
        MSD values.  Returns nan for non-positive or non-finite results.

    Notes
    -----
    Physical interpretation:

    * Small L:  MSD rises from ``(a - c)`` toward plateau ``a``.
    * Large L:  MSD ≈ ``a``  (restricted diffusion plateau).
    * ``b`` controls the rate of rise; larger ``b`` → faster saturation.

    Validated parameters from Violanda et al. (2019),
    nucleotide A in *Synechococcus elongatus* PCC 7942:

        a ≈ 5.21,  b ≈ 0.0024,  c ≈ 3.81

    (Values vary by nucleotide identity and genome species.)

    References
    ----------
    Violanda, Bernido & Carpio-Bernido (2019),
    "White noise functional integral for exponentially decaying memory:
    nucleotide distribution in bacterial genomes",
    Physica Scripta 94, 125006.

    Examples
    --------
    >>> L = np.arange(1, 500)
    >>> msd = msd_dna(L, a=5.21, b=0.0024, c=3.81)
    """
    # Step 1: Normalize input — accept scalar, list, or array; remember if scalar
    L_arr, scalar = _to_array(L)

    # Step 2: Evaluate the plateau MSD formula:
    #         MSD(L) = a - c · exp(-b · L)
    #
    #         At L=0:   MSD = a - c          (starting value; must be > 0, so a > c)
    #         As L→∞:   MSD → a              (plateau — restricted diffusion ceiling)
    #         b controls how fast the plateau is reached (larger b = faster saturation)
    val = a - c * np.exp(-b * L_arr)

    # Step 3: Replace any non-positive or non-finite values with NaN, return result
    return _finalize_msd(val, scalar)


def pdf_dna(dx, L: float, a: float, b: float, c: float) -> np.ndarray:
    """
    PDF for DNA nucleotide separation distances.

    Gaussian with variance equal to the plateau MSD::

        P(dx; L) = 1/√(2π·σ²) · exp(-dx²/(2σ²))
        where  σ² = a - c · exp(-b · L)

    Parameters
    ----------
    dx : float or np.ndarray
        Displacement values.
    L : float
        Occurrence number at which the PDF is evaluated (scalar).
    a, b, c : float
        Same parameters as :func:`msd_dna`.

    Returns
    -------
    np.ndarray
        PDF values.  All-nan if σ² is non-positive or non-finite.

    References
    ----------
    Violanda et al. (2019), Phys. Scr. 94, 125006.
    """
    sigma2 = msd_dna(float(L), a, b, c)
    return _pdf_from_sigma2(dx, sigma2)


# ── Extended model stubs ──────────────────────────────────────────────────────

def _stub(name: str):
    raise NotImplementedError(_NOT_IMPL_MSG.format(name=name))


# ── Implemented: sin_half (Table 2.1 row 2) ──────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Formula from Table 2.1 row 2 — verify against source before applying to data.

def msd_sin_half(T):
    """
    Theoretical MSD for the sin^(1/2) memory kernel model (Table 2.1 row 2).

    Memory function: f(t-t') = [sin(T-t)]^(1/2), h(t') = 1
    Formula::

        MSD(T) = integral_0^T sin(T-t) dt = 1 - cos(T)

    No free parameters.

    Returns
    -------
    float or np.ndarray
        MSD values. Returns nan where result is non-positive (cos(T) >= 1).

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 2.
    """
    # Step 1: Normalize input
    T_arr, scalar = _to_array(T)

    # Step 2: MSD(T) = ∫₀ᵀ sin(T-t) dt = [-cos(T-t)]₀ᵀ = 1 - cos(T)
    #         Derivation: squaring f gives sin(T-t); integrating over [0,T] gives 1 - cos(T).
    #         Always ≥ 0 (since cos(T) ≤ 1), but equals 0 at T = 0, 2π, 4π...
    val = 1.0 - np.cos(T_arr)

    # Step 3: Replace non-positive / non-finite with NaN, return
    return _finalize_msd(val, scalar)


def pdf_sin_half(dx, T: float) -> np.ndarray:
    """
    Theoretical PDF for the sin^(1/2) model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = 1 - cos(T))
    """
    sigma2 = msd_sin_half(float(T))
    return _pdf_from_sigma2(dx, sigma2)


# ── Implemented: cos_half (Table 2.1 row 3) ──────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Formula from Table 2.1 row 3 — verify against source before applying to data.

def msd_cos_half(T):
    """
    Theoretical MSD for the cos^(1/2) memory kernel model (Table 2.1 row 3).

    Memory function: f(t-t') = [cos(T-t)]^(1/2), h(t') = 1
    Formula::

        MSD(T) = integral_0^T cos(T-t) dt = sin(T)

    No free parameters.

    Returns
    -------
    float or np.ndarray
        MSD values. Returns nan where sin(T) <= 0 (i.e. T outside (0, pi), (2pi, 3pi), ...).

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 3.
    """
    # Step 1: Normalize input
    T_arr, scalar = _to_array(T)

    # Step 2: MSD(T) = ∫₀ᵀ cos(T-t) dt = [sin(T-t)/(-1)·(-1)]₀ᵀ = sin(T)
    #         Substituting u = T-t: ∫₀ᵀ cos(u) du = sin(T).
    #         Positive only for T ∈ (0,π), (2π,3π), ... — elsewhere NaN.
    val = np.sin(T_arr)

    # Step 3: Replace non-positive / non-finite with NaN, return
    return _finalize_msd(val, scalar)


def pdf_cos_half(dx, T: float) -> np.ndarray:
    """
    Theoretical PDF for the cos^(1/2) model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = sin(T))
    """
    sigma2 = msd_cos_half(float(T))
    return _pdf_from_sigma2(dx, sigma2)


# ── Stub: exp_whittaker (Table 2.1 row 5) ────────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: Whittaker W — implementable via scipy.special.hyperu
#   W_{k,m}(z) = exp(-z/2) * z^(m+1/2) * hyperu(m-k+1/2, 2m+1, z)
# Provide exact MSD formula from Table 2.1 row 5 to complete the implementation.

def msd_exp_whittaker(T, mu: float, beta: float, nu: float):
    """
    MSD for the exponential-Whittaker model (Table 2.1 row 5).
    Involves the Whittaker W function.
    Parameters: mu, beta, nu.
    Status: not yet implemented — provide exact formula from Table 2.1.
    """
    _stub('exp_whittaker')


# ── Stub: bessel_K (Table 2.1 row 6) ─────────────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: modified Bessel K — available as scipy.special.kv(v, z).
# Provide exact MSD formula from Table 2.1 row 6 to complete the implementation.

def msd_bessel_K(T, mu: float, beta: float):
    """
    MSD for the modified Bessel K model (Table 2.1 row 6).
    Parameters: mu, beta.
    Status: not yet implemented — provide exact formula from Table 2.1.
    """
    _stub('bessel_K')


# ── Stub: hypergeom_F1 (Table 2.1 row 7) ─────────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: Appell F₁ (two-variable hypergeometric).
#   NOT in scipy — requires mpmath: mpmath.appellf1(a, b1, b2, c, x, y)
# Provide exact MSD formula from Table 2.1 row 7 to complete the implementation.

def msd_hypergeom_F1(T, mu: float, beta: float, nu: float):
    """
    MSD for the Appell hypergeometric F1 model (Table 2.1 row 7).
    Parameters: mu, beta, nu.
    Status: not yet implemented — requires mpmath for Appell F1.
    """
    _stub('hypergeom_F1')


# ── Stub: bessel_I (Table 2.1 row 8) ─────────────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: modified Bessel I — available as scipy.special.iv(v, z).
# Provide exact MSD formula from Table 2.1 row 8 to complete the implementation.

def msd_bessel_I(T, mu: float, beta: float):
    """
    MSD for the modified Bessel I model (Table 2.1 row 8).
    Parameters: mu, beta.
    Status: not yet implemented — provide exact formula from Table 2.1.
    """
    _stub('bessel_I')


# ── Stub: hypergeom_3F2 (Table 2.1 row 11) ───────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: generalized hypergeometric ₃F₂.
#   NOT in scipy (only ₂F₁ available) — requires mpmath: mpmath.hyp3f2(a1,a2,a3,b1,b2,z)
# Provide exact MSD formula from Table 2.1 row 11 to complete the implementation.

def msd_hypergeom_3F2(T, mu: float, beta: float, lam: float):
    """
    MSD for the generalized hypergeometric 3F2 model (Table 2.1 row 11).
    Parameters: mu, beta, lam (lambda).
    Status: not yet implemented — requires mpmath for ₃F₂.
    """
    _stub('hypergeom_3F2')


# ── Stub: csc_power (Table 2.1 row 12) ───────────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: numpy only (csc = 1/sin) — no special library needed.
# Provide exact MSD formula from Table 2.1 row 12 to complete the implementation.

def msd_csc_power(T, nu: float, c: float):
    """
    MSD for the cosecant power-law model (Table 2.1 row 12).
    Parameters: nu, c.
    Status: not yet implemented — provide exact formula from Table 2.1.
    """
    _stub('csc_power')


# ── Stub: cot_power (Table 2.1 row 13) ───────────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: numpy only (cot = cos/sin) — no special library needed.
# Provide exact MSD formula from Table 2.1 row 13 to complete the implementation.

def msd_cot_power(T, nu: float, c: float):
    """
    MSD for the cotangent power-law model (Table 2.1 row 13).
    Parameters: nu, c.
    Status: not yet implemented — provide exact formula from Table 2.1.
    """
    _stub('cot_power')


# ── Implemented: inc_gamma (Table 2.1 row 14) ────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Formula derived from SWNA integral with power-law × exponential kernel.
# MSD = Γ(ν)·P(ν,μT)/μ^ν — models restricted (plateau) diffusion.
# Verify against Table 2.1 row 14 before applying to real data.

def msd_inc_gamma(T, nu: float, mu: float):
    """
    Theoretical MSD for the incomplete gamma model (Table 2.1 row 14).

    Memory function: f(t-t') = (T-t)^(nu/2 - 1) * exp(-mu*(T-t)), h(t') = 1
    Formula::

        MSD(T) = integral_0^T t^(nu-1) * exp(-mu*t) dt
               = gamma(nu) * P(nu, mu*T) / mu^nu

    where P(nu, z) = gammainc(nu, z) is the regularized lower incomplete gamma.

    Parameters
    ----------
    T : float or np.ndarray
        Time lag(s).
    nu : float
        Shape parameter (controls diffusion exponent). nu > 0.
    mu : float
        Rate parameter (sets the saturation timescale). mu > 0.

    Returns
    -------
    float or np.ndarray
        MSD values. Monotonically rises from 0 to Gamma(nu)/mu^nu as T -> inf.

    Notes
    -----
    This model describes restricted diffusion with a smooth plateau.
    For large mu, saturation occurs early; for small mu, the MSD rises
    slowly before levelling off.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 14.
    """
    # Step 1: Normalize input
    T_arr, scalar = _to_array(T)

    # Step 2: Evaluate MSD(T) = Γ(ν) · P(ν, μT) / μ^ν
    #
    #   Γ(ν)          — complete gamma; sets the plateau height Γ(ν)/μ^ν as T → ∞
    #   P(ν, μT)      — regularized lower incomplete gamma (scipy.gammainc);
    #                   rises from 0 to 1 as T → ∞, encoding the memory decay
    #   μ^(-ν)        — rate scaling; larger μ → faster saturation → lower plateau
    val = gamma(nu) * gammainc(nu, mu * T_arr) / mu ** nu

    # Step 3: Replace non-positive / non-finite with NaN, return
    return _finalize_msd(val, scalar)


def pdf_inc_gamma(dx, T: float, nu: float, mu: float) -> np.ndarray:
    """
    Theoretical PDF for the incomplete gamma model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = MSD_inc_gamma(T, nu, mu))
    """
    sigma2 = msd_inc_gamma(float(T), nu, mu)
    return _pdf_from_sigma2(dx, sigma2)


# ── Stub: bessel_pair (Table 2.1 row 15) ─────────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: Bessel J pair — available as scipy.special.jv(v, z).
# Provide exact MSD formula from Table 2.1 row 15 to complete the implementation.

def msd_bessel_pair(T, nu: float):
    """
    MSD for the Bessel function pair model (Table 2.1 row 15).
    Parameter: nu.
    Status: not yet implemented — provide exact formula from Table 2.1.
    """
    _stub('bessel_pair')


# ── Stub: bessel_pair2 (Table 2.1 row 16) ────────────────────────────────────
# Published applications in Bernido group: none documented yet.
# Special function required: Bessel J pair — available as scipy.special.jv(v, z).
# Provide exact MSD formula from Table 2.1 row 16 to complete the implementation.

def msd_bessel_pair2(T, nu: float, mu: float):
    """
    MSD for the Bessel function pair with memory model (Table 2.1 row 16).
    Parameters: nu, mu.
    Status: not yet implemented — provide exact formula from Table 2.1.
    """
    _stub('bessel_pair2')


# ── Implemented: bessel_j0_cos (Table 2.1 row 17) ───────────────────────────
# Published applications in Bernido group: none documented yet.
# Formula: MSD = J_0(T) − cos(T)  (Bernido non-Markovian book draft, Eq. 11.3.38)
# Verify against source equation before applying to real data.

def msd_bessel_j0_cos(T):
    """
    Theoretical MSD for the J_0 minus cosine model (Table 2.1 row 17).

    Memory functions: f(t-t') = sqrt(J_{1-nu}(t-t')), h(t') = sqrt(J_nu(t'))
    Formula (Eq. 11.3.38)::

        MSD(T) = J_0(T) - cos(T)

    No free parameters. The nu in the kernel cancels in the MSD integral
    via the Bessel convolution identity:
        integral_0^T J_{1-nu}(T-t) * J_nu(t) dt = J_0(T) - cos(T)

    Returns
    -------
    float or np.ndarray
        MSD values. Returns nan where J_0(T) - cos(T) <= 0.

    Notes
    -----
    For small T: J_0(T) ~ 1 - T^2/4 and cos(T) ~ 1 - T^2/2,
    so MSD ~ T^2/4 > 0. The MSD oscillates and can dip negative for
    large T (e.g. near T = 2*pi where J_0(2*pi) < cos(2*pi)); those
    lags are masked as NaN.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 17, Eq. 11.3.38.
    """
    # Step 1: Normalize input
    T_arr, scalar = _to_array(T)

    # Step 2: MSD(T) = J_0(T) - cos(T)
    #   J_0(T) — Bessel function of the first kind, order 0 (scipy.jv(0, T))
    #   cos(T) — ordinary cosine
    #   Their difference is the result of the Bessel convolution identity
    #   which cancels the ν dependence from the kernel.
    val = jv(0, T_arr) - np.cos(T_arr)

    # Step 3: Replace non-positive / non-finite with NaN, return
    return _finalize_msd(val, scalar)


def pdf_bessel_j0_cos(dx, T: float) -> np.ndarray:
    """
    Theoretical PDF for the J_0 minus cosine model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = J_0(T) - cos(T))
    """
    sigma2 = msd_bessel_j0_cos(float(T))
    return _pdf_from_sigma2(dx, sigma2)


# ── Implemented: bessel_jmu_nu (Table 2.1 row 18) ───────────────────────────
# Published applications in Bernido group: none documented yet.
# Formula: MSD = J_{μ+ν}(T)/μ  (Bernido non-Markovian book draft, Eq. 11.3.40)
# Verify against source equation before applying to real data.

def msd_bessel_jmu_nu(T, mu: float, nu: float):
    """
    Theoretical MSD for the Bessel J_{mu+nu} model (Table 2.1 row 18).

    Memory functions: f(t-t') = sqrt(J_nu(t-t')), h(t') = sqrt(t'^{-1} J_mu(t'))
    Formula (Eq. 11.3.40)::

        MSD(T) = J_{mu+nu}(T) / mu

    Parameters
    ----------
    T : float or np.ndarray
        Time lag(s).
    mu : float
        Memory parameter (denominator). mu > 0.
    nu : float
        Secondary order parameter. mu + nu > -1 for J_{mu+nu} to be regular.

    Returns
    -------
    float or np.ndarray
        MSD values. Returns nan where J_{mu+nu}(T)/mu <= 0 or non-finite.

    Notes
    -----
    For small T: J_{mu+nu}(T) ~ (T/2)^(mu+nu) / Gamma(mu+nu+1),
    so MSD ~ T^(mu+nu) / (mu * 2^(mu+nu) * Gamma(mu+nu+1)) > 0.
    The MSD oscillates for large T (Bessel functions oscillate) —
    oscillatory lags are masked as NaN.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 2.1 row 18, Eq. 11.3.40.
    """
    # Step 1: Normalize input
    T_arr, scalar = _to_array(T)

    # Step 2: MSD(T) = J_{μ+ν}(T) / μ
    #   jv(mu + nu, T_arr) — Bessel J of order μ+ν evaluated at T
    #   divide by μ — scales amplitude; μ is the memory parameter
    val = jv(mu + nu, T_arr) / mu

    # Step 3: Replace non-positive / non-finite with NaN, return
    return _finalize_msd(val, scalar)


def pdf_bessel_jmu_nu(dx, T: float, mu: float, nu: float) -> np.ndarray:
    """
    Theoretical PDF for the J_{mu+nu} model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = J_{mu+nu}(T) / mu)
    """
    sigma2 = msd_bessel_jmu_nu(float(T), mu, nu)
    return _pdf_from_sigma2(dx, sigma2)


# ── MODELS registry ───────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    # fBm — general subdiffusive/superdiffusive benchmark; no specific
    # Bernido-group publication listed, but widely used in diffusion literature.
    'fbm': {
        'msd':         msd_fbm,
        'pdf':         pdf_fbm,
        'params':      ['H'],
        'n_params':    1,
        'row':         1,
        'status':      'available',
        'description': 'Fractional Brownian motion (Hurst exponent)',
        'reference':   'Table 2.1 row 1, Bernido & Carpio-Bernido (2015)',
    },
    'sin_half': {
        'msd':         msd_sin_half,
        'pdf':         pdf_sin_half,
        'params':      [],
        'n_params':    0,
        'row':         2,
        'status':      'available',
        'description': 'MSD = 1 - cos(T)  (no free parameters)',
        'reference':   'Table 2.1 row 2',
    },
    'cos_half': {
        'msd':         msd_cos_half,
        'pdf':         pdf_cos_half,
        'params':      [],
        'n_params':    0,
        'row':         3,
        'status':      'available',
        'description': 'MSD = sin(T)  (no free parameters)',
        'reference':   'Table 2.1 row 3',
    },
    # exponential — used in:
    #   Roque et al. (2024): Philippine earthquake time series, mu ≈ 1.00–1.19
    #   Toledo et al. (2024): solar sunspot number, mu ≈ 1.15
    'exponential': {
        'msd':         msd_exponential,
        'pdf':         pdf_exponential,
        'params':      ['mu', 'beta'],
        'n_params':    2,
        'row':         4,
        'status':      'available',
        'description': 'Power-law memory with exponential modulation',
        'reference':   'Table 2.1 row 4, Bernido & Carpio-Bernido (2015)',
    },
    'exp_whittaker': {
        'msd':         msd_exp_whittaker,
        'pdf':         None,
        'params':      ['mu', 'beta', 'nu'],
        'n_params':    3,
        'row':         5,
        'status':      'not_implemented',
        'description': 'Power-law memory with Whittaker W function',
        'reference':   'Table 2.1 row 5',
    },
    'bessel_K': {
        'msd':         msd_bessel_K,
        'pdf':         None,
        'params':      ['mu', 'beta'],
        'n_params':    2,
        'row':         6,
        'status':      'not_implemented',
        'description': 'Power-law memory with modified Bessel K',
        'reference':   'Table 2.1 row 6',
    },
    'hypergeom_F1': {
        'msd':         msd_hypergeom_F1,
        'pdf':         None,
        'params':      ['mu', 'beta', 'nu'],
        'n_params':    3,
        'row':         7,
        'status':      'not_implemented',
        'description': 'Power-law memory with Appell hypergeometric F1',
        'reference':   'Table 2.1 row 7',
    },
    'bessel_I': {
        'msd':         msd_bessel_I,
        'pdf':         None,
        'params':      ['mu', 'beta'],
        'n_params':    2,
        'row':         8,
        'status':      'not_implemented',
        'description': 'Power-law memory with modified Bessel I',
        'reference':   'Table 2.1 row 8',
    },
    # sine — structurally similar to cosine; no specific published benchmark
    # in Bernido group yet, but available for datasets where sine fits better.
    'sine': {
        'msd':         msd_sine,
        'pdf':         pdf_sine,
        'params':      ['mu', 'nu'],
        'n_params':    2,
        'row':         9,
        'status':      'available',
        'description': 'Power-law memory with sine modulation',
        'reference':   'Table 2.1 row 9, Bernido & Carpio-Bernido (2015)',
    },
    # cosine — used in:
    #   Elnar et al. (2021): GBR coral bleaching, mu ≈ 4.64 (hyperballistic)
    #   Elnar et al. (2024): CO2 Keeling curve, mu ≈ 0.91–0.97 (subdiffusive)
    #   Calotes thesis (2024): X-ray binary light curves, mu ∈ [0.50, 1.39]
    'cosine': {
        'msd':         msd_cosine,
        'pdf':         pdf_cosine,
        'params':      ['mu', 'nu'],
        'n_params':    2,
        'row':         10,
        'status':      'available',
        'description': 'Power-law memory with cosine modulation',
        'reference':   'Table 2.1 row 10, Bernido & Carpio-Bernido (2015)',
    },
    'hypergeom_3F2': {
        'msd':         msd_hypergeom_3F2,
        'pdf':         None,
        'params':      ['mu', 'beta', 'lam'],
        'n_params':    3,
        'row':         11,
        'status':      'not_implemented',
        'description': 'Power-law with generalized hypergeometric 3F2',
        'reference':   'Table 2.1 row 11',
    },
    'csc_power': {
        'msd':         msd_csc_power,
        'pdf':         None,
        'params':      ['nu', 'c'],
        'n_params':    2,
        'row':         12,
        'status':      'not_implemented',
        'description': 'Cosecant power-law modulation',
        'reference':   'Table 2.1 row 12',
    },
    'cot_power': {
        'msd':         msd_cot_power,
        'pdf':         None,
        'params':      ['nu', 'c'],
        'n_params':    2,
        'row':         13,
        'status':      'not_implemented',
        'description': 'Cotangent power-law modulation',
        'reference':   'Table 2.1 row 13',
    },
    'inc_gamma': {
        'msd':         msd_inc_gamma,
        'pdf':         pdf_inc_gamma,
        'params':      ['nu', 'mu'],
        'n_params':    2,
        'row':         14,
        'status':      'available',
        'description': 'MSD = Gamma(nu)*P(nu,mu*T)/mu^nu  (restricted diffusion)',
        'reference':   'Table 2.1 row 14',
    },
    'bessel_pair': {
        'msd':         msd_bessel_pair,
        'pdf':         None,
        'params':      ['nu'],
        'n_params':    1,
        'row':         15,
        'status':      'not_implemented',
        'description': 'Bessel function pair (no memory parameter)',
        'reference':   'Table 2.1 row 15',
    },
    'bessel_pair2': {
        'msd':         msd_bessel_pair2,
        'pdf':         None,
        'params':      ['nu', 'mu'],
        'n_params':    2,
        'row':         16,
        'status':      'not_implemented',
        'description': 'Bessel function pair with memory parameter',
        'reference':   'Table 2.1 row 16',
    },
    'bessel_j0_cos': {
        'msd':         msd_bessel_j0_cos,
        'pdf':         pdf_bessel_j0_cos,
        'params':      [],
        'n_params':    0,
        'row':         17,
        'status':      'available',
        'description': 'MSD = J_0(T) - cos(T)  (no free parameters)',
        'reference':   'Table 2.1 row 17, Eq. 11.3.38',
    },
    'bessel_jmu_nu': {
        'msd':         msd_bessel_jmu_nu,
        'pdf':         pdf_bessel_jmu_nu,
        'params':      ['mu', 'nu'],
        'n_params':    2,
        'row':         18,
        'status':      'available',
        'description': 'MSD = J_{mu+nu}(T)/mu  (Eq. 11.3.40)',
        'reference':   'Table 2.1 row 18, Eq. 11.3.40',
    },
    # dna — used in:
    #   Violanda, Bernido & Carpio-Bernido (2019): nucleotide separation
    #   distances in bacterial genomes (Synechococcus elongatus PCC 7942);
    #   plateau-shaped MSD (restricted diffusion), not a Table 2.1 power-law model.
    'dna': {
        'msd':         msd_dna,
        'pdf':         pdf_dna,
        'params':      ['a', 'b', 'c'],
        'n_params':    3,
        'row':         None,
        'status':      'available',
        'description': 'Exponentially decaying memory — DNA nucleotide '
                       'separation distances (plateau MSD shape)',
        'reference':   'Violanda et al. (2019), Phys. Scr. 94, 125006',
    },
}

_AVAILABLE_NAMES = [n for n, v in MODELS.items() if v['status'] == 'available']
_ALL_NAMES = list(MODELS.keys())

# ── Registry accessors ────────────────────────────────────────────────────────

def get_model(name: str) -> dict:
    """
    Return the model entry from the registry for a given model name.

    Parameters
    ----------
    name : str
        Model name (e.g. ``'cosine'``, ``'exponential'``).

    Returns
    -------
    dict
        Registry entry with keys ``'msd'``, ``'pdf'``, ``'params'``,
        ``'n_params'``, ``'row'``, ``'status'``, ``'description'``,
        ``'reference'``.

    Raises
    ------
    ValueError
        If ``name`` is not one of the 18 registered models.
    NotImplementedError
        If the model is registered but not yet implemented (stub).

    Examples
    --------
    >>> m = wn.get_model('cosine')
    >>> msd_fn = m['msd']
    """
    if name not in MODELS:
        raise ValueError(
            f"✗ Unknown model '{name}'.\n"
            f"Available: {', '.join(_ALL_NAMES)}\n"
            f"Run wn.list_models() for details."
        )
    info = MODELS[name]
    if info['status'] == 'not_implemented':
        raise NotImplementedError(
            f"Model '{name}' (row {info['row']}) is not yet implemented.\n"
            f"Implemented: cosine, exponential, sine, fbm, dna, "
            f"sin_half, cos_half, inc_gamma, bessel_j0_cos, bessel_jmu_nu\n"
            f"Run wn.list_models() to see all 18 models."
        )
    return info


def list_models() -> None:
    """
    Print a formatted table of all 18 models from Bernido & Carpio-Bernido (2015), Table 2.1.

    Shows name, row number, parameters, implementation status,
    and description for each model.

    Examples
    --------
    >>> wn.list_models()
    """
    _PSYM = {
        'mu': 'mu', 'nu': 'nu', 'beta': 'beta',
        'H': 'H', 'lam': 'lam', 'c': 'c',
    }

    def _fmt_params(params: list[str]) -> str:
        return ', '.join(_PSYM.get(p, p) for p in params) or '(none)'

    def _fmt_status(status: str) -> str:
        return 'available' if status == 'available' else 'stub'

    # Column specs: (header, width)
    cols = [
        ('Name',        16),
        ('Row',          4),
        ('Params',      14),
        ('Status',      10),
        ('Description', 40),
    ]

    def _hline(left='+-', mid='-+-', right='-+') -> str:
        return left[0] + mid[1].join('-' * (w + 2) for _, w in cols) + right[-1]

    def _row(*values) -> str:
        parts = [f' {str(v):<{w}} ' for v, (_, w) in zip(values, cols)]
        return '|' + '|'.join(parts) + '|'

    sep = _hline(left='+-', mid='-+-', right='-+')
    print(sep)
    print(_row(*[h for h, _ in cols]))
    print(sep)
    # Table 2.1 models first (sorted by row), then extras (row=None) at the end
    def _sort_key(kv):
        r = kv[1]['row']
        return (r is None, r or 0)

    for name, info in sorted(MODELS.items(), key=_sort_key):
        p = _fmt_params(info['params'])
        s = _fmt_status(info['status'])
        d = info['description']
        if len(d) > 40:
            d = d[:37] + '...'
        row_display = 'N/A' if info['row'] is None else info['row']
        print(_row(name, row_display, p, s, d))
    print(sep)
