import numpy as np
import scipy
from scipy.stats import norm as Normal
import matplotlib.pyplot as plt
import yfinance as yf

class BSM_BaseModel(object):
    def __init__(self, pr_spot, pr_strike, interest, day_expire, vol):
        assert pr_strike > 0
        self.pr_spot = pr_spot # spot price
        self.pr_strike = pr_strike # option strike
        self.interest = interest # interest rate
        self.day_expire = day_expire # days to option expiration
        self.vol = vol # volatility

        a = self.vol * np.sqrt(self.day_expire)
        self.d1 = (np.log(self.pr_spot / self.pr_strike) + (self.interest + np.power(self.vol, 2) / 2) * self.day_expire) / a
        self.d2 = self.d1 - a
        self.decay_exp = np.exp(-self.interest * self.day_expire)

        self.option_prices = {}
        self.greeks = {}
        self.option_prices["pr_call"] = self.option_prices["pr_put"] = None
        self.greeks["delta_call"] = self.greeks["delta_put"] = None
        self.greeks["theta_call"] = self.greeks["theta_put"] = None
        self.greeks["rho_call"] = self.greeks["rho_put"] = None
        self.greeks["vega"] = None
        self.greeks["gamma"] = None

    def comp_option_prices(self):
        raise NotImplementedError()

    def comp_greeks(self):
        raise NotImplementedError()

class BSM_EurOption(BSM_BaseModel):
    def __init__(self, pr_spot, pr_strike, interest, day_expire, vol):
        super(BSM_EurOption, self).__init__(pr_spot, pr_strike, interest, day_expire, vol)
        a = self.vol * np.sqrt(self.day_expire)
        self.d1 = (np.log(self.pr_spot / self.pr_strike) + (self.interest + np.power(self.vol, 2) / 2) * self.day_expire) / a
        self.d2 = self.d1 - a
        self.decay_exp = np.exp(-self.interest * self.day_expire)

    def comp_option_prices(self):
        if np.abs(self.vol) < 1e-6 or np.abs(self.day_expire) < 1e-6:
            self.option_prices["pr_call"] = np.maximum(0.0, self.pr_spot - self.pr_strike)
            self.option_prices["pr_put"] = np.maximum(0.0, self.pr_strike - self.pr_spot)
        else:
            self.option_prices["pr_call"] = self.pr_spot * Normal.cdf(self.d1) - self.pr_strike * self.decay_exp * Normal.cdf(self.d2)
            self.option_prices["pr_put"] = self.pr_strike * self.decay_exp * Normal.cdf(-self.d2) - self.pr_spot * Normal.cdf(-self.d1)

        return self.option_prices

    def comp_greeks(self):
        if np.abs(self.vol) < 1e-6 or np.abs(self.day_expire) < 1e-6:
            self.greeks["delta_call"] = 1.0 if self.pr_spot > self.pr_strike else 0.0
            self.greeks["delta_put"] = -1.0 if self.pr_spot < self.pr_strike else 0.0
        else:
            self.greeks["delta_call"] = Normal.cdf(self.d1)
            self.greeks["delta_put"] = -Normal.cdf(-self.d1)

        tmp1 = -self.pr_spot * Normal.pdf(self.d1) * self.vol / (2 * np.sqrt(self.day_expire))
        tmp2 = self.interest * self.pr_strike * self.decay_exp
        self.greeks["theta_call"] = (tmp1 - tmp2 * Normal.cdf(self.d2)) / 365
        self.greeks["theta_put"] = (tmp1 + tmp2 * Normal.cdf(-self.d2)) / 365

        tmp3 = self.pr_strike * self.day_expire * self.decay_exp / 100
        self.greeks["rho_call"] = tmp3 * Normal.cdf(self.d2)
        self.greeks["rho_put"] = -tmp3 * Normal.cdf(-self.d2)

        if np.abs(self.vol) < 1e-6 or np.abs(self.day_expire) < 1e-6:
            self.greeks["vega"] = 0.0
        else:
            self.greeks["vega"] = self.pr_spot * Normal.pdf(self.d1) * np.sqrt(self.day_expire) / 100

        self.greeks["gamma"] = Normal.pdf(self.d1) / (self.pr_spot * self.vol * np.sqrt(self.day_expire))

        return self.greeks

if __name__ == "__main__":
    pr_spot = 100
    pr_strike = 100
    interest = 0.05
    day_expire = 1
    vol = 0.2
    bsm_eur_option = BSM_EurOption(pr_spot, pr_strike, interest, day_expire, vol)

    option_prices = bsm_eur_option.comp_option_prices()
    for k, v in option_prices.items():
        print(f"{k} = {v}")
    greeks = bsm_eur_option.comp_greeks()
    for k, v in greeks.items():
        print(f"{k} = {v}")
