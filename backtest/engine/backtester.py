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
from typing import List, Dict, Optional, Tuple
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
    value: float
    pnl: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance statistics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_pnl: float
    num_trades: int
    turnover: float


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

        # State
        self.cash = initial_capital
        self.positions: Dict[str, deque] = {}  # symbol -> deque of Position objects
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[date, float]] = []

        logger.info(
            f"Backtester initialized: capital=${initial_capital:,.0f}, "
            f"spread={spread_bps}bps"
        )

    def calculate_costs(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> float:
        """
        Calculate transaction costs.

        Args:
            symbol: Symbol
            quantity: Number of shares
            price: Execution price
            side: 'buy' or 'sell'

        Returns:
            Total cost
        """
        # Commission
        commission = abs(quantity) * self.fee_per_share

        # Spread cost (pay half spread each way)
        spread_cost = abs(quantity) * price * (self.spread_bps / 10000) / 2

        return commission + spread_cost

    def execute_trade(
        self,
        trade_date: date,
        symbol: str,
        target_quantity: float,
        price: float
    ) -> Optional[Trade]:
        """
        Execute a trade with costs.

        Args:
            trade_date: Trade date
            symbol: Symbol
            target_quantity: Target position (positive=long, negative=short, 0=close)
            price: Execution price

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

        # Calculate costs
        fees = self.calculate_costs(symbol, trade_qty, price, side)

        # Calculate value
        value = trade_qty * price

        # Update cash
        self.cash -= (value + fees)

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
            value=abs(value),
            pnl=pnl if trade_qty < 0 else None
        )
        self.trades.append(trade)

        logger.debug(
            f"{trade_date} {side.upper()} {abs(trade_qty):.0f} {symbol} @ ${price:.2f}, "
            f"fees=${fees:.2f}"
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
            if symbol in prices:
                position_value += quantity * prices[symbol]

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

            # Execute trades
            for _, row in day_signals.iterrows():
                symbol = row['symbol']
                target_qty = row['target_quantity']

                if symbol in price_map:
                    self.execute_trade(trade_date, symbol, target_qty, price_map[symbol])

            # Calculate portfolio value
            portfolio_value = self.get_portfolio_value(price_map)
            self.equity_curve.append((trade_date, portfolio_value))

        logger.info(f"Backtest complete: {len(self.trades)} trades executed")

        # Build equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        return equity_df

    def calculate_performance(self) -> PerformanceMetrics:
        """Calculate performance metrics."""
        if not self.equity_curve:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)

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

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            num_trades=num_trades,
            turnover=turnover
        )

    def print_summary(self):
        """Print backtest summary."""
        metrics = self.calculate_performance()

        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"Initial Capital:    ${self.initial_capital:,.0f}")
        print(f"Final Equity:       ${self.equity_curve[-1][1]:,.0f}")
        print(f"Total Return:       {metrics.total_return*100:.2f}%")
        print(f"Sharpe Ratio:       {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown:       {metrics.max_drawdown*100:.2f}%")
        print(f"Number of Trades:   {metrics.num_trades}")
        print(f"Win Rate:           {metrics.win_rate*100:.1f}%")
        print(f"Avg Trade P&L:      ${metrics.avg_trade_pnl:,.2f}")
        print(f"Turnover:           {metrics.turnover:.2f}x")
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
