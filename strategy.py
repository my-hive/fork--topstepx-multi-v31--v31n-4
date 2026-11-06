"""v31n-4 candidate A: ES 5-day-drop mean reversion.

Mechanism: after 5-session close-to-close drop in bottom quartile of trailing
250-session distribution (lagged), go long 1 ES at 09:30 CT. Hard stop 6 points
($300 entry risk). Time exit after 3 sessions at 15:09. Re-enter while true.

Causal: features from completed prior sessions' RTH closes only. Entries 09:30.
"""
import numpy as np


def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol != 'ES':
            result[symbol] = out
            continue
        day = df[(df.ct_sec >= 30600) & (df.ct_sec < 54600)]
        g = day.groupby('session', sort=False).close
        c = g.last()  # ~15:09 CT close of sessions that have RTH bars
        mom5 = c.shift(1) - c.shift(6)
        thr = mom5.rolling(250, min_periods=100).quantile(0.25).shift(1)
        sig = ((mom5 <= thr) & thr.notna()).to_dict()

        session_rows = df.groupby('session', sort=False).indices
        close_all = df.close.to_numpy()
        low_all = df.low.to_numpy()
        sec_all = df.ct_sec.to_numpy()
        pos = 0
        entry = np.nan
        days_left = 0
        for s, rows in session_rows.items():
            sec = sec_all[rows]
            has_rth = ((sec >= 30600) & (sec < 54600)).any()
            s_sig = bool(sig.get(s, False))
            if has_rth and s_sig and pos == 0:
                # plan entry at first bar >= 09:30 CT
                pass
            for j in range(len(rows)):
                r = rows[j]
                t = pos
                if pos == 0 and has_rth and s_sig and 34200 <= sec[j] < 54000:
                    pos = 1
                    entry = close_all[r]
                    days_left = 2
                    t = 1
                elif pos == 1:
                    if sec[j] >= 34200 and low_all[r] <= entry - 6.0:
                        pos = 0
                        entry = np.nan
                        t = 0
                    elif days_left <= 0 and sec[j] >= 53400:
                        pos = 0
                        entry = np.nan
                        t = 0
                    else:
                        t = 1
                out[r] = t
            if pos == 1 and has_rth:
                days_left -= 1
        result[symbol] = out
    return result
