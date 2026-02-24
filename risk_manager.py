import numpy as np
from enum import Enum
from config import RiskManagementConfig as Config


class RiskMode(Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    RISK_OFF = "RISK_OFF"
    EMERGENCY = "EMERGENCY"


class RiskManager:
    """Professional risk engine with asymmetric transitions and sticky recovery."""

    _SEVERITY = {
        RiskMode.NORMAL: 0,
        RiskMode.CAUTION: 1,
        RiskMode.RISK_OFF: 2,
        RiskMode.EMERGENCY: 3,
    }

    def __init__(self):
        self.current_mode = RiskMode.NORMAL
        self.recovery_counter = 0

        self.portfolio_values = []
        self.returns = []

        self.last_drawdown = 0.0
        self.last_volatility = 0.0
        self.last_var_pct = 0.0

    def update_returns(self, daily_return: float) -> None:
        self.returns.append(float(daily_return))
        if len(self.returns) > Config.VAR_LOOKBACK:
            self.returns.pop(0)

    def _update_nav_window(self, nav: float) -> None:
        self.portfolio_values.append(float(nav))
        if len(self.portfolio_values) > Config.ROLLING_PEAK_WINDOW:
            self.portfolio_values.pop(0)

    def _rolling_peak(self) -> float:
        if not self.portfolio_values:
            return 0.0
        return float(max(self.portfolio_values))

    def _drawdown(self, nav: float) -> float:
        peak = self._rolling_peak()
        if peak <= 0:
            return 0.0
        return (nav - peak) / peak

    def _volatility(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        daily_vol = float(np.std(self.returns, ddof=0))
        return daily_vol * np.sqrt(252)

    def _var_loss_pct(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        mean_ret = float(np.mean(self.returns))
        std_ret = float(np.std(self.returns, ddof=0))
        var_ret = mean_ret - Config.VAR_ZSCORE * std_ret
        return max(0.0, -var_ret)

    def _mode_from_dd(self, dd: float) -> RiskMode:
        if dd <= Config.DD_THRESHOLDS["EMERGENCY"]:
            return RiskMode.EMERGENCY
        if dd <= Config.DD_THRESHOLDS["RISK_OFF"]:
            return RiskMode.RISK_OFF
        if dd <= Config.DD_THRESHOLDS["CAUTION"]:
            return RiskMode.CAUTION
        return RiskMode.NORMAL

    def _mode_from_vol(self, vol: float) -> RiskMode:
        if vol >= Config.VOL_THRESHOLDS["EMERGENCY"]:
            return RiskMode.EMERGENCY
        if vol >= Config.VOL_THRESHOLDS["RISK_OFF"]:
            return RiskMode.RISK_OFF
        if vol >= Config.VOL_THRESHOLDS["CAUTION"]:
            return RiskMode.CAUTION
        return RiskMode.NORMAL

    def _mode_from_var(self, var_loss_pct: float) -> RiskMode:
        if var_loss_pct > Config.VAR_THRESHOLDS["EMERGENCY"]:
            return RiskMode.EMERGENCY
        if var_loss_pct > Config.VAR_THRESHOLDS["RISK_OFF"]:
            return RiskMode.RISK_OFF
        if var_loss_pct > Config.VAR_THRESHOLDS["CAUTION"]:
            return RiskMode.CAUTION
        return RiskMode.NORMAL

    def _max_mode(self, *modes: RiskMode) -> RiskMode:
        return max(modes, key=lambda mode: self._SEVERITY[mode])

    def _recovery_target(self, mode: RiskMode):
        if mode == RiskMode.EMERGENCY:
            return RiskMode.RISK_OFF, "EMERGENCY"
        if mode == RiskMode.RISK_OFF:
            return RiskMode.CAUTION, "RISK_OFF"
        if mode == RiskMode.CAUTION:
            return RiskMode.NORMAL, "CAUTION"
        return RiskMode.NORMAL, None

    def _recovery_ok(self, mode_key: str, dd: float, vol: float, var_loss_pct: float) -> bool:
        t = Config.RECOVERY_THRESHOLDS[mode_key]
        return dd > t["dd"] and vol < t["vol"] and var_loss_pct < t["var"]

    def evaluate(self, nav: float, daily_return: float | None = None) -> dict:
        if daily_return is not None:
            self.update_returns(daily_return)

        self._update_nav_window(nav)

        dd = self._drawdown(nav)
        vol = self._volatility()
        var_loss_pct = self._var_loss_pct()

        dd_mode = self._mode_from_dd(dd)
        vol_mode = self._mode_from_vol(vol)
        var_mode = self._mode_from_var(var_loss_pct)
        entry_mode = self._max_mode(dd_mode, vol_mode, var_mode)

        if self._SEVERITY[entry_mode] > self._SEVERITY[self.current_mode]:
            self.current_mode = entry_mode
            self.recovery_counter = 0
        elif self.current_mode != RiskMode.NORMAL:
            target_mode, mode_key = self._recovery_target(self.current_mode)
            if mode_key and self._recovery_ok(mode_key, dd, vol, var_loss_pct):
                self.recovery_counter += 1
                if self.recovery_counter >= Config.RECOVERY_DAYS[mode_key]:
                    self.current_mode = target_mode
                    self.recovery_counter = 0
            else:
                self.recovery_counter = 0

        self.last_drawdown = dd
        self.last_volatility = vol
        self.last_var_pct = var_loss_pct

        return {
            "mode": self.current_mode.value,
            "dd": dd,
            "vol": vol,
            "var_loss_pct": var_loss_pct,
            "dd_mode": dd_mode.value,
            "vol_mode": vol_mode.value,
            "var_mode": var_mode.value,
            "entry_mode": entry_mode.value,
            "recovery_counter": self.recovery_counter,
            "mode_cap": self.mode_cap(),
        }

    def mode_cap(self) -> float:
        return Config.MODE_CAPS[self.current_mode.value]

    def get_volatility(self) -> float:
        return self.last_volatility

    def calculate_var(self, confidence: float = 0.99) -> float:
        _ = confidence
        return -self.last_var_pct
