from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math

@dataclass
class StockInputs:
    # --- Quality / profitability ---
    roic: Optional[float] = None          # e.g., 0.14 for 14%
    gross_margin: Optional[float] = None  # 0.55
    op_margin: Optional[float] = None     # 0.35

    # --- Financial strength ---
    current_ratio: Optional[float] = None
    net_debt_to_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None  # EBIT / Interest
    fcf_positive: Optional[bool] = None

    # --- Growth ---
    rev_cagr_5y: Optional[float] = None   # 0.12 for 12%
    eps_cagr_5y: Optional[float] = None
    fcf_cagr_5y: Optional[float] = None

    # --- Valuation inputs ---
    price: Optional[float] = None
    shares_outstanding: Optional[float] = None  # in same units as you prefer
    fcf_ttm: Optional[float] = None             # trailing-twelve-month free cash flow (total)
    net_debt: Optional[float] = None            # total net debt (debt - cash). Can be negative.

    # Optional “relative valuation” signals
    pe: Optional[float] = None
    p_fcf: Optional[float] = None
    ev_ebitda: Optional[float] = None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_quality(inp: StockInputs) -> Tuple[float, Dict[str, float]]:
    # 0–10
    s = 0.0
    parts: Dict[str, float] = {}

    # ROIC
    if inp.roic is not None:
        # 8% -> 0 pts, 20% -> 4 pts
        roic_pts = 4 * clamp((inp.roic - 0.08) / (0.20 - 0.08), 0, 1)
        parts["roic"] = roic_pts
        s += roic_pts

    # Gross margin
    if inp.gross_margin is not None:
        # 25% -> 0 pts, 60% -> 3 pts
        gm_pts = 3 * clamp((inp.gross_margin - 0.25) / (0.60 - 0.25), 0, 1)
        parts["gross_margin"] = gm_pts
        s += gm_pts

    # Operating margin
    if inp.op_margin is not None:
        # 5% -> 0 pts, 30% -> 3 pts
        om_pts = 3 * clamp((inp.op_margin - 0.05) / (0.30 - 0.05), 0, 1)
        parts["op_margin"] = om_pts
        s += om_pts

    return clamp(s, 0, 10), parts


def score_financial_strength(inp: StockInputs) -> Tuple[float, Dict[str, float]]:
    # 0–10
    s = 0.0
    parts: Dict[str, float] = {}

    if inp.current_ratio is not None:
        # 1.0 -> 0, 2.0 -> 2
        pts = 2 * clamp((inp.current_ratio - 1.0) / (2.0 - 1.0), 0, 1)
        parts["current_ratio"] = pts
        s += pts

    if inp.net_debt_to_ebitda is not None:
        # <=0.5 -> 3, 2.5 -> 0
        pts = 3 * clamp((2.5 - inp.net_debt_to_ebitda) / (2.5 - 0.5), 0, 1)
        parts["net_debt_to_ebitda"] = pts
        s += pts

    if inp.interest_coverage is not None:
        # 2 -> 0, 10 -> 3
        pts = 3 * clamp((inp.interest_coverage - 2.0) / (10.0 - 2.0), 0, 1)
        parts["interest_coverage"] = pts
        s += pts

    if inp.fcf_positive is not None:
        pts = 2.0 if inp.fcf_positive else 0.0
        parts["fcf_positive"] = pts
        s += pts

    return clamp(s, 0, 10), parts


def score_growth(inp: StockInputs) -> Tuple[float, Dict[str, float]]:
    # 0–10
    s = 0.0
    parts: Dict[str, float] = {}

    # Revenue CAGR: 0% -> 0, 15% -> 4
    if inp.rev_cagr_5y is not None:
        pts = 4 * clamp(inp.rev_cagr_5y / 0.15, 0, 1)
        parts["rev_cagr_5y"] = pts
        s += pts

    # EPS CAGR: 0% -> 0, 18% -> 3
    if inp.eps_cagr_5y is not None:
        pts = 3 * clamp(inp.eps_cagr_5y / 0.18, 0, 1)
        parts["eps_cagr_5y"] = pts
        s += pts

    # FCF CAGR: 0% -> 0, 18% -> 3
    if inp.fcf_cagr_5y is not None:
        pts = 3 * clamp(inp.fcf_cagr_5y / 0.18, 0, 1)
        parts["fcf_cagr_5y"] = pts
        s += pts

    return clamp(s, 0, 10), parts


def dcf_lite_equity_value(
    fcf0: float,
    years: int = 5,
    growth: float = 0.10,
    discount: float = 0.10,
    terminal_growth: float = 0.025,
    net_debt: float = 0.0,
) -> float:
    """
    Returns *equity value* (not per share).
    fcf0 = trailing FCF (total)
    """
    if discount <= terminal_growth:
        raise ValueError("discount rate must be > terminal growth rate")

    pv = 0.0
    fcf = fcf0
    for t in range(1, years + 1):
        fcf *= (1 + growth)
        pv += fcf / ((1 + discount) ** t)

    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount) ** years)

    enterprise_value = pv + pv_terminal
    equity_value = enterprise_value - net_debt
    return equity_value


def score_valuation(inp: StockInputs, intrinsic_upside_threshold: float = 0.30) -> Tuple[float, Dict[str, float]]:
    """
    0–10 valuation score.
    If you supply price, shares_outstanding, fcf_ttm (and optionally net_debt),
    we compute an intrinsic value via DCF-lite and score the upside.
    """
    s = 0.0
    parts: Dict[str, float] = {}

    if inp.price is not None and inp.shares_outstanding and inp.fcf_ttm:
        equity_value = dcf_lite_equity_value(
            fcf0=inp.fcf_ttm,
            years=5,
            growth=0.10,          # you can tune
            discount=0.10,        # EM-risk-adjusted default
            terminal_growth=0.025,
            net_debt=inp.net_debt or 0.0
        )
        intrinsic_per_share = equity_value / inp.shares_outstanding
        upside = (intrinsic_per_share - inp.price) / inp.price
        parts["intrinsic_per_share"] = intrinsic_per_share
        parts["upside_pct"] = upside

        # upside <= -20% -> 0 pts ; 0% -> 5 pts ; +30% or more -> 10 pts
        if upside <= -0.20:
            pts = 0.0
        elif upside >= intrinsic_upside_threshold:
            pts = 10.0
        else:
            # map -20%..+30% into 0..10
            pts = 10 * (upside + 0.20) / (intrinsic_upside_threshold + 0.20)
        parts["dcf_pts"] = pts
        s += pts
    else:
        # fallback: a mild neutral if no DCF inputs
        parts["note"] = "No DCF inputs provided; valuation score defaults to 5."
        s = 5.0

    return clamp(s, 0, 10), parts


def total_weighted_score(inp: StockInputs) -> Dict[str, object]:
    # weights: Business Quality 25, Financial Strength 20, Growth 20, Valuation 25, Management 10
    q, q_parts = score_quality(inp)
    fs, fs_parts = score_financial_strength(inp)
    g, g_parts = score_growth(inp)
    v, v_parts = score_valuation(inp)

    # Management is hard to quantify with raw statements; you can plug your own score 0–10.
    management = 7.5

    total = (
        0.25 * q +
        0.20 * fs +
        0.20 * g +
        0.25 * v +
        0.10 * management
    )

    def label(x: float) -> str:
        if x >= 8.0: return "Strong Buy (by this rubric)"
        if x >= 6.5: return "Buy/Watch"
        if x >= 5.0: return "Hold"
        return "Avoid"

    return {
        "scores": {"quality": q, "financial_strength": fs, "growth": g, "valuation": v, "management": management},
        "breakdown": {"quality": q_parts, "financial_strength": fs_parts, "growth": g_parts, "valuation": v_parts},
        "total_weighted_score": total,
        "decision_band": label(total),
    }


# ------------------ Example usage (replace with real numbers) ------------------
if __name__ == "__main__":
    example = StockInputs(
        roic=0.15, gross_margin=0.52, op_margin=0.38,
        current_ratio=2.0, net_debt_to_ebitda=0.5, interest_coverage=20, fcf_positive=True,
        rev_cagr_5y=0.12, eps_cagr_5y=0.15, fcf_cagr_5y=0.12,
        price=100.0, shares_outstanding=5_000_000_000, fcf_ttm=60_000_000_000, net_debt=-10_000_000_000
    )
    print(total_weighted_score(example))
