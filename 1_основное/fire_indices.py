"""Two standard fire-danger indices, used as baselines (no machine learning).

* Canadian Fire Weather Index (FWI), Van Wagner (1987): the index used by the European
  and Global Wildfire Information Systems (EFFIS / GWIS).
* Nesterov index: the fire-danger index traditionally used by weather services in
  Kazakhstan and Russia.

Every function works on numpy arrays of shape (cells,) for one day, so a whole country
is updated day by day.
"""
import numpy as np

# day-length factors for DMC and DC (northern hemisphere, by month 1..12)
DMC_LE = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])
DC_LF = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])


def ffmc_step(ffmc0, T, H, W, r):
    mo = 147.2 * (101.0 - ffmc0) / (59.5 + ffmc0)
    rf = r - 0.5
    wet = r > 0.5
    mo_r = np.where(
        mo > 150,
        mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / np.maximum(rf, 1e-6)))
        + 0.0015 * (mo - 150) ** 2 * np.sqrt(np.maximum(rf, 0)),
        mo + 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / np.maximum(rf, 1e-6))),
    )
    mo = np.where(wet, np.minimum(mo_r, 250), mo)
    ed = 0.942 * H**0.679 + 11 * np.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    ew = 0.618 * H**0.753 + 10 * np.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    ko = 0.424 * (1 - (H / 100) ** 1.7) + 0.0694 * np.sqrt(W) * (1 - (H / 100) ** 8)
    kd = ko * 0.581 * np.exp(0.0365 * T)
    k1 = 0.424 * (1 - ((100 - H) / 100) ** 1.7) + 0.0694 * np.sqrt(W) * (1 - ((100 - H) / 100) ** 8)
    kw = k1 * 0.581 * np.exp(0.0365 * T)
    m = np.where(mo > ed, ed + (mo - ed) * 10 ** (-kd), np.where(mo < ew, ew - (ew - mo) * 10 ** (-kw), mo))
    return np.clip(59.5 * (250 - m) / (147.2 + m), 0, 101)


def dmc_step(dmc0, T, H, r, month):
    T = np.maximum(T, -1.1)
    rk = 1.894 * (T + 1.1) * (100 - H) * DMC_LE[month - 1] * 1e-4
    re = 0.92 * r - 1.27
    mo = 20 + np.exp(5.6348 - dmc0 / 43.43)
    b = np.where(dmc0 <= 33, 100 / (0.5 + 0.3 * dmc0),
                 np.where(dmc0 <= 65, 14 - 1.3 * np.log(np.maximum(dmc0, 1e-6)),
                          6.2 * np.log(np.maximum(dmc0, 1e-6)) - 17.2))
    mr = mo + 1000 * re / (48.77 + b * re)
    pr = np.maximum(244.72 - 43.43 * np.log(np.maximum(mr - 20, 1e-6)), 0)
    dmc = np.where(r > 1.5, pr, dmc0)
    return np.maximum(dmc + rk, 0)


def dc_step(dc0, T, r, month):
    T = np.maximum(T, -2.8)
    pe = np.maximum((0.36 * (T + 2.8) + DC_LF[month - 1]) / 2, 0)
    rd = 0.83 * r - 1.27
    qo = 800 * np.exp(-dc0 / 400)
    qr = qo + 3.937 * rd
    dr = np.maximum(400 * np.log(800 / np.maximum(qr, 1e-6)), 0)
    dc = np.where(r > 2.8, dr, dc0)
    return np.maximum(dc + pe, 0)


def isi_bui_fwi(ffmc, dmc, dc, W):
    mo = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    ff = 19.115 * np.exp(-0.1386 * mo) * (1 + mo**5.31 / 4.93e7)
    isi = ff * np.exp(0.05039 * W)
    bui = np.where(dmc <= 0.4 * dc,
                   0.8 * dmc * dc / np.maximum(dmc + 0.4 * dc, 1e-6),
                   dmc - (1 - 0.8 * dc / np.maximum(dmc + 0.4 * dc, 1e-6)) * (0.92 + (0.0114 * dmc) ** 1.7))
    bui = np.maximum(bui, 0)
    fd = np.where(bui <= 80, 0.626 * bui**0.809 + 2, 1000 / (25 + 108.64 * np.exp(-0.023 * bui)))
    b = 0.1 * isi * fd
    fwi = np.where(b > 1, np.exp(2.72 * (0.434 * np.log(np.maximum(b, 1e-6))) ** 0.647), b)
    return isi, bui, fwi
