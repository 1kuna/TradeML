"""
Test Phase 2 end-to-end pipeline:
1. Load data
2. Build features & labels
3. Run CPCV with Ridge baseline
4. Build portfolio
5. Backtest with costs
6. Calculate DSR & PBO
7. Generate trade blotter
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ops.pipelines.equities_xs import PipelineConfig, run_pipeline
from loguru import logger

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("PHASE 2 END-TO-END PIPELINE TEST")
    logger.info("=" * 80)

    # Test universe: 10 liquid tech stocks
    TEST_UNIVERSE = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "AMD", "INTC", "QCOM"
    ]

    # Configure pipeline
    cfg = PipelineConfig(
        start_date="2022-06-01",  # Start after sufficient lookback
        end_date="2024-06-01",  # 2 years of trading
        universe=TEST_UNIVERSE,
        label_type="horizon",  # Use horizon returns
        horizon_days=5,  # 5-day forward returns
        n_folds=5,  # 5-fold CV (faster than 8)
        embargo_days=10,
        initial_capital=1_000_000.0,
        spread_bps=5.0,
        gross_cap=1.0,  # 100% gross exposure (long-only for now)
        max_name=0.10,  # Max 10% per name
        kelly_fraction=0.25,  # Conservative 25% Kelly
    )

    logger.info(f"Universe: {len(cfg.universe)} symbols")
    logger.info(f"Date range: {cfg.start_date} to {cfg.end_date}")
    logger.info(f"Label: {cfg.label_type} (k={cfg.horizon_days})")
    logger.info(f"CV: {cfg.n_folds} folds, embargo={cfg.embargo_days} days")

    try:
        logger.info("\n[1/7] Loading data & building features...")
        results = run_pipeline(cfg)

        logger.info("\n[2/7] ✓ Feature engineering complete")
        logger.info(f"Features shape: {results['target_weights'].shape}")

        logger.info("\n[3/7] ✓ CPCV training complete")
        logger.info(f"Folds completed: {cfg.n_folds}")

        logger.info("\n[4/7] ✓ Portfolio construction complete")
        logger.info(f"Target weights generated: {len(results['target_weights'])}")

        logger.info("\n[5/7] ✓ Backtest complete")
        metrics = results['backtest_metrics']
        logger.info(f"Total Return: {metrics.total_return*100:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
        logger.info(f"Win Rate: {metrics.win_rate*100:.1f}%")
        logger.info(f"Turnover: {metrics.turnover:.2f}")

        logger.info("\n[6/7] ✓ Anti-overfitting metrics calculated")
        dsr = results['dsr']
        pbo = results['pbo']
        logger.info(f"DSR: {dsr['dsr']:.4f} (threshold: > 0)")
        logger.info(f"PBO: {pbo['pbo']*100:.1f}% (threshold: <= 5%)")

        logger.info("\n[7/7] ✓ Trade blotter generated")
        logger.info(f"Report saved to: ops/reports/")

        # Check go/no-go criteria
        logger.info("\n" + "=" * 80)
        logger.info("GO/NO-GO EVALUATION (Blueprint Section 2)")
        logger.info("=" * 80)

        go_nogo = {
            "Sharpe >= 1.0": metrics.sharpe_ratio >= 1.0,
            "Max DD <= 20%": metrics.max_drawdown <= 0.20,
            "DSR > 0": dsr['dsr'] > 0,
            "PBO <= 5%": pbo['pbo'] <= 0.05,
        }

        all_pass = all(go_nogo.values())

        for criterion, passed in go_nogo.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{criterion:<20} {status}")

        if all_pass:
            logger.success("\n🎉 ALL GO/NO-GO CRITERIA MET! Ready for Phase 3.")
        else:
            logger.warning("\n⚠️  Some criteria not met. Iteration needed before Phase 3.")

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2 TEST COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
