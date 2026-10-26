"""The numerical example of the Deflated Sharpe Ratio paper, from its summary numbers.

Bailey and López de Prado (2014): 100 independent trials, the variance of
their annualised Sharpe ratios is 0.5, the best annualised Sharpe ratio is
2.5, and it was measured on 1250 daily values with skewness -3 and raw
kurtosis 10. The paper reports a DSR of about 0.9004.
"""

import math

from luckgate import dsr, expected_max_sharpe

DAYS_PER_YEAR = 250

# luckgate works per period, so turn the annual numbers into daily ones first.
sr_daily = 2.5 / math.sqrt(DAYS_PER_YEAR)
sr_std_daily = math.sqrt(0.5 / DAYS_PER_YEAR)  # a standard deviation, not a variance

benchmark = expected_max_sharpe(100, sr_std_daily)
right = dsr(sr_daily, n_obs=1250, n_trials=100, sr_std=sr_std_daily, skew=-3, kurtosis=10)
same = dsr(
    sr_daily,
    n_obs=1250,
    n_trials=100,
    sr_std=sr_std_daily,
    skew=-3,
    kurtosis=7,
    kurtosis_kind="excess",
)
annual_by_mistake = dsr(2.5, n_obs=1250, n_trials=100, sr_std=math.sqrt(0.5), skew=-3, kurtosis=10)

print(f"daily Sharpe of the winner:          {sr_daily:.4f}")
print(f"expected best of 100 noise trials:   {benchmark:.4f}")
print(f"DSR (raw kurtosis 10):               {right:.4f}")
print(f"DSR (same data, excess kurtosis 7):  {same:.4f}")
print(f"DSR with annual numbers (a mistake): {annual_by_mistake:.4f}")
