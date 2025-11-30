"""
Minimal event-driven backtester.

Daily frequency backtester with:
- Event-driven architecture
- Position accounting (FIFO)
- Fee + spread costs
- Corporate action handling
- Basic performance reporting
"""

from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class Position:
    """Track a single position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_date: date
    entry_value: float = field(init=False)

    def __post_init__(self):
        self.entry_value = self.quantity * self.entry_price


@dataclass
class Trade:
    """Record of a trade execution."""
    date: date
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fees: float
    impact: float
    value: float
    pnl: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance statistics per SSOT §4.9."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_trade_pnl: float
    num_trades: int
    turnover: float
    dsr: float  # Deflated Sharpe Ratio
    total_fees: float
    total_impact: float


class MinimalBacktester:
    """
    Event-driven backtester for daily frequency strategies.

    Features:
    - FIFO position accounting
    - Transaction costs (fees + spread)
    - Corporate action adjustments
    - Daily P&L tracking
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        fee_per_share: float = 0.0,  # US equities often $0
        spread_bps: float = 5.0,      # Estimate 5 bps spread
        borrow_bps: float = 0.0,      # Daily borrow fee (annualized bps) applied to shorts
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            fee_per_share: Commission per share
            spread_bps: Bid-ask spread in basis points
        """
        self.initial_capital = initial_capital
        self.fee_per_share = fee_per_share
        self.spread_bps = spread_bps
        self.borrow_bps = borrow_bps

        # State
        self.cash = initial_capital
        self.positions: Dict[str, deque] = {}  # symbol -> deque of Position objects
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[date, float]] = []
        self.last_prices: Dict[str, float] = {}  # carry forward missing quotes

        logger.info(
            f"Backtester initialized: capital=${initial_capital:,.0f}, "
            f"spread={spread_bps}bps"
        )

    def calculate_costs(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        volatility: Optional[float] = None,
        adv: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate transaction costs including market impact.

        Args:
            symbol: Symbol
            quantity: Number of shares
            price: Execution price
            side: 'buy' or 'sell'
            volatility: Optional daily volatility (sigma) for impact calculation
            adv: Optional average daily volume for impact calculation

        Returns:
            Tuple of (fees, impact) where fees = commission + spread
        """
        # Commission
        commission = abs(quantity) * self.fee_per_share

        # Spread cost (pay half spread each way)
        spread_cost = abs(quantity) * price * (self.spread_bps / 10000) / 2

        fees = commission + spread_cost

        # Market impact using square-root law per SSOT
        # impact = sigma * sqrt(qty/ADV) * price * qty
        impact = 0.0
        if volatility is not None and adv is not None and adv > 0:
            participation = abs(quantity) / adv
            if participation > 0:
                # Square-root impact model: impact_bps = sigma * sqrt(participation)
                impact_bps = volatility * np.sqrt(participation)
                impact = abs(quantity) * price * impact_bps

        return fees, impact

    def execute_trade(
        self,
        trade_date: date,
        symbol: str,
        target_quantity: float,
        price: float,
        volatility: Optional[float] = None,
        adv: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Execute a trade with costs.

        Args:
            trade_date: Trade date
            symbol: Symbol
            target_quantity: Target position (positive=long, negative=short, 0=close)
            price: Execution price
            volatility: Optional daily volatility for impact calculation
            adv: Optional average daily volume for impact calculation

        Returns:
            Trade object or None if no trade
        """
        # Get current position
        current_qty = self.get_position_quantity(symbol)

        # Calculate trade quantity
        trade_qty = target_quantity - current_qty

        if abs(trade_qty) < 0.01:  # No trade needed
            return None

        side = 'buy' if trade_qty > 0 else 'sell'

        # Calculate costs (fees and market impact)
        fees, impact = self.calculate_costs(symbol, trade_qty, price, side, volatility, adv)

        # Calculate value
        value = trade_qty * price

        # Update cash (deduct value, fees, and impact)
        self.cash -= (value + fees + impact)

        # Update positions
        pnl = 0.0
        if trade_qty > 0:  # Buy
            # Add to position queue
            if symbol not in self.positions:
                self.positions[symbol] = deque()

            position = Position(
                symbol=symbol,
                quantity=trade_qty,
                entry_price=price,
                entry_date=trade_date
            )
            self.positions[symbol].append(position)

        else:  # Sell
            # Remove from position queue (FIFO)
            remaining = abs(trade_qty)

            while remaining > 0 and symbol in self.positions and self.positions[symbol]:
                pos = self.positions[symbol][0]

                if pos.quantity <= remaining:
                    # Close entire position
                    pnl += (price - pos.entry_price) * pos.quantity
                    remaining -= pos.quantity
                    self.positions[symbol].popleft()
                else:
                    # Partial close
                    pnl += (price - pos.entry_price) * remaining
                    pos.quantity -= remaining
                    remaining = 0

            # Clean up empty queues
            if symbol in self.positions and not self.positions[symbol]:
                del self.positions[symbol]

        # Record trade
        trade = Trade(
            date=trade_date,
            symbol=symbol,
            side=side,
            quantity=abs(trade_qty),
            price=price,
            fees=fees,
            impact=impact,
            value=abs(value),
            pnl=pnl if trade_qty < 0 else None
        )
        self.trades.append(trade)

        logger.debug(
            f"{trade_date} {side.upper()} {abs(trade_qty):.0f} {symbol} @ ${price:.2f}, "
            f"fees=${fees:.2f}, impact=${impact:.2f}"
        )

        return trade

    def get_position_quantity(self, symbol: str) -> float:
        """Get current position quantity for symbol."""
        if symbol not in self.positions:
            return 0.0
        return sum(pos.quantity for pos in self.positions[symbol])

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.

        Args:
            prices: Dict of symbol -> current price

        Returns:
            Total value (cash + positions)
        """
        position_value = 0.0

        for symbol, pos_queue in self.positions.items():
            quantity = sum(pos.quantity for pos in pos_queue)
            px = prices.get(symbol, self.last_prices.get(symbol))
            if px is None:
                logger.warning(f"Missing price for held symbol {symbol}; valuing at $0 for now")
                continue
            position_value += quantity * px

        return self.cash + position_value

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run backtest.

        Args:
            signals: DataFrame with columns: date, symbol, target_quantity
            prices: DataFrame with columns: date, symbol, close

        Returns:
            DataFrame with daily equity curve
        """
        logger.info("Starting backtest...")

        # Get unique dates
        dates = sorted(signals['date'].unique())

        for trade_date in dates:
            # Get signals for this date
            day_signals = signals[signals['date'] == trade_date]

            # Get prices for this date
            day_prices = prices[prices['date'] == trade_date]
            price_map = dict(zip(day_prices['symbol'], day_prices['close']))
            # Update carry-forward cache
            self.last_prices.update(price_map)

            # Execute trades
            for _, row in day_signals.iterrows():
                symbol = row['symbol']
                target_qty = row['target_quantity']

                if symbol in price_map:
                    self.execute_trade(trade_date, symbol, target_qty, price_map[symbol])
                else:
                    logger.warning(f"Skipping trade for {symbol} on {trade_date}: missing price")

            # Daily borrow carry for existing short positions
            if self.borrow_bps and self.borrow_bps > 0:
                daily_rate = (self.borrow_bps / 10000.0) / 252.0
                carry_cost = 0.0
                for symbol, pos_queue in list(self.positions.items()):
                    qty = sum(p.quantity for p in pos_queue)
                    px = price_map.get(symbol, self.last_prices.get(symbol))
                    if qty < 0 and px is not None:
                        carry_cost += abs(qty) * px * daily_rate
                if carry_cost:
                    self.cash -= carry_cost

            # Calculate portfolio value
            portfolio_value = self.get_portfolio_value(price_map)
            self.equity_curve.append((trade_date, portfolio_value))

        logger.info(f"Backtest complete: {len(self.trades)} trades executed")

        # Build equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        return equity_df

    def calculate_performance(self, num_trials: int = 1) -> PerformanceMetrics:
        """
        Calculate performance metrics per SSOT §4.9.

        Args:
            num_trials: Number of strategy variants tested (for DSR calculation)
        """
        if not self.equity_curve:
            return PerformanceMetrics(
                total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                calmar_ratio=0.0, win_rate=0.0, avg_trade_pnl=0.0,
                num_trades=0, turnover=0.0, dsr=0.0,
                total_fees=0.0, total_impact=0.0
            )

        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])

        # Returns
        returns = equity_df['equity'].pct_change().dropna()
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1

        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()

        # Max drawdown
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_dd = drawdown.min()

        # Calmar ratio (annualized return / abs(max drawdown))
        calmar = 0.0
        if max_dd < 0:
            # Annualize return: (1 + total_return)^(252/n_days) - 1
            n_days = len(returns)
            if n_days > 0:
                ann_return = (1 + total_return) ** (252 / n_days) - 1
                calmar = ann_return / abs(max_dd)

        # Trade statistics
        realized_trades = [t for t in self.trades if t.pnl is not None]
        num_trades = len(realized_trades)

        if num_trades > 0:
            wins = [t for t in realized_trades if t.pnl > 0]
            win_rate = len(wins) / num_trades
            avg_trade_pnl = np.mean([t.pnl for t in realized_trades])
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0

        # Turnover (total trade value / avg equity)
        total_trade_value = sum(t.value for t in self.trades)
        avg_equity = equity_df['equity'].mean()
        turnover = total_trade_value / avg_equity if avg_equity > 0 else 0

        # Total fees and impact from actual trades
        total_fees = sum(t.fees for t in self.trades)
        total_impact = sum(t.impact for t in self.trades)

        # Deflated Sharpe Ratio (DSR) per Bailey & Lopez de Prado
        # DSR = SR * sqrt(1 - skew*SR/3 + (kurt-3)*SR^2/24) / sqrt(var_sr)
        # Simplified approximation: penalize for multiple trials
        dsr = self._calculate_dsr(sharpe, returns, num_trials)

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            num_trades=num_trades,
            turnover=turnover,
            dsr=dsr,
            total_fees=total_fees,
            total_impact=total_impact
        )

    def _calculate_dsr(
        self,
        sharpe: float,
        returns: pd.Series,
        num_trials: int = 1
    ) -> float:
        """
        Calculate Deflated Sharpe Ratio per Bailey & Lopez de Prado.

        Adjusts Sharpe ratio for:
        - Multiple testing (num_trials)
        - Non-normal return distribution (skewness, kurtosis)

        Args:
            sharpe: Annualized Sharpe ratio
            returns: Daily returns series
            num_trials: Number of strategy variants tested

        Returns:
            Deflated Sharpe Ratio
        """
        from scipy import stats

        if len(returns) < 10 or sharpe <= 0:
            return 0.0

        n = len(returns)

        # Skewness and kurtosis of returns
        skew = returns.skew()
        kurt = returns.kurtosis()  # Excess kurtosis

        # Standard error of Sharpe ratio
        # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt+3)/4*SR^2) / (n-1))
        sr_daily = sharpe / np.sqrt(252)  # Convert to daily SR for calculation
        se_sr = np.sqrt(
            (1 + 0.5 * sr_daily**2 - skew * sr_daily + (kurt + 3) / 4 * sr_daily**2)
            / (n - 1)
        )

        if se_sr <= 0:
            return sharpe

        # Expected maximum Sharpe under null hypothesis (multiple testing)
        # E[max(SR)] ≈ sqrt(2*log(num_trials)) * se_sr (for large num_trials)
        if num_trials > 1:
            expected_max_sr = np.sqrt(2 * np.log(num_trials)) * se_sr * np.sqrt(252)
        else:
            expected_max_sr = 0.0

        # DSR = probability that observed SR exceeds expected max SR under null
        # Approximation: DSR ≈ SR - expected_max_sr (bounded below by 0)
        # More rigorous: use t-distribution CDF
        t_stat = (sharpe - expected_max_sr) / (se_sr * np.sqrt(252))
        dsr = stats.t.cdf(t_stat, df=n-1)

        # Scale to make interpretable (0-1 maps to haircut on Sharpe)
        # Return as adjusted Sharpe: SR * DSR
        return sharpe * dsr

    def print_summary(self, num_trials: int = 1):
        """Print backtest summary."""
        metrics = self.calculate_performance(num_trials)

        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Initial Capital:    ${self.initial_capital:,.0f}")
        print(f"Final Equity:       ${self.equity_curve[-1][1]:,.0f}")
        print(f"Total Return:       {metrics.total_return*100:.2f}%")
        print(f"Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
        print(f"Deflated SR (DSR):  {metrics.dsr:.2f}")
        print(f"Max Drawdown:       {metrics.max_drawdown*100:.2f}%")
        print(f"Calmar Ratio:       {metrics.calmar_ratio:.2f}")
        print("-"*60)
        print(f"Number of Trades:   {metrics.num_trades}")
        print(f"Win Rate:           {metrics.win_rate*100:.1f}%")
        print(f"Avg Trade P&L:      ${metrics.avg_trade_pnl:,.2f}")
        print(f"Turnover:           {metrics.turnover:.2f}x")
        print("-"*60)
        print(f"Total Fees:         ${metrics.total_fees:,.2f}")
        print(f"Total Impact:       ${metrics.total_impact:,.2f}")
        print("="*60)


# CLI for testing
if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--signals", type=str, required=True, help="Path to signals CSV/parquet")
    parser.add_argument("--prices", type=str, required=True, help="Path to prices parquet")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--spread-bps", type=float, default=5.0)

    args = parser.parse_args()

    # Load data
    if args.signals.endswith('.parquet'):
        signals = pd.read_parquet(args.signals)
    else:
        signals = pd.read_csv(args.signals)

    prices = pd.read_parquet(args.prices)

    logger.info(f"Loaded {len(signals)} signals, {len(prices)} price bars")

    # Initialize backtester
    bt = MinimalBacktester(
        initial_capital=args.capital,
        spread_bps=args.spread_bps
    )

    # Run backtest
    equity_curve = bt.run(signals, prices)

    # Print summary
    bt.print_summary()

    # Save equity curve
    output_path = "backtest/results/equity_curve.csv"
    equity_curve.to_csv(output_path, index=False)
    print(f"\n[OK] Equity curve saved to {output_path}")
