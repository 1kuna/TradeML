-- TradeML Database Schema
-- PostgreSQL initialization script for metadata and run logs

-- Data ingestion metadata
CREATE TABLE IF NOT EXISTS data_ingestion_log (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) NOT NULL,
    source_uri TEXT NOT NULL,
    data_type VARCHAR(50) NOT NULL,  -- 'ticks', 'bars', 'options', 'corpactions', etc.
    ingestion_start_ts TIMESTAMP WITH TIME ZONE NOT NULL,
    ingestion_end_ts TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL,  -- 'running', 'completed', 'failed'
    records_ingested BIGINT DEFAULT 0,
    file_hash VARCHAR(64),  -- SHA256 checksum
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ingestion_source_date ON data_ingestion_log(source_name, ingestion_start_ts);
CREATE INDEX idx_ingestion_status ON data_ingestion_log(status);

-- Data quality checks log
CREATE TABLE IF NOT EXISTS data_quality_log (
    id SERIAL PRIMARY KEY,
    check_name VARCHAR(100) NOT NULL,
    check_type VARCHAR(50) NOT NULL,  -- 'schema', 'monotonicity', 'outlier', 'coverage', etc.
    data_source VARCHAR(100) NOT NULL,
    check_ts TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) NOT NULL,  -- 'pass', 'fail', 'warning'
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_qc_source_ts ON data_quality_log(data_source, check_ts);
CREATE INDEX idx_qc_status ON data_quality_log(status);

-- Model training runs
CREATE TABLE IF NOT EXISTS model_runs (
    run_id VARCHAR(100) PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,  -- 'equities_xs', 'equities_ts', 'options_vol'
    model_name VARCHAR(100) NOT NULL,
    train_start_date DATE NOT NULL,
    train_end_date DATE NOT NULL,
    validation_method VARCHAR(50) NOT NULL,  -- 'cpcv', 'walk_forward', etc.
    config_hash VARCHAR(64) NOT NULL,
    config JSONB NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL,  -- 'running', 'completed', 'failed'
    mlflow_run_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_runs_model_type ON model_runs(model_type, started_at);
CREATE INDEX idx_runs_status ON model_runs(status);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL REFERENCES model_runs(run_id),
    fold_id INTEGER,  -- NULL for overall metrics
    metric_name VARCHAR(50) NOT NULL,
    metric_value NUMERIC NOT NULL,
    is_oos BOOLEAN NOT NULL,  -- Out-of-sample vs in-sample
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_metrics_run_fold ON model_metrics(run_id, fold_id);
CREATE INDEX idx_metrics_name ON model_metrics(metric_name);

-- Champion-Challenger tracking
CREATE TABLE IF NOT EXISTS model_champions (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,  -- 'equities_xs', etc.
    champion_run_id VARCHAR(100) NOT NULL REFERENCES model_runs(run_id),
    promoted_at TIMESTAMP WITH TIME ZONE NOT NULL,
    promoted_by VARCHAR(100),  -- 'auto' or username
    promotion_reason TEXT,
    previous_champion_run_id VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_champions_type_active ON model_champions(model_type, is_active);

-- Shadow trading log
CREATE TABLE IF NOT EXISTS shadow_trades (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL REFERENCES model_runs(run_id),
    trade_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL,  -- 'long', 'short', 'close'
    target_weight NUMERIC,
    expected_alpha_bps NUMERIC,
    confidence NUMERIC,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_shadow_run_date ON shadow_trades(run_id, trade_date);

-- Live trading blotter
CREATE TABLE IF NOT EXISTS trade_blotter (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    instrument_type VARCHAR(20) NOT NULL,  -- 'equity', 'option'
    side VARCHAR(10) NOT NULL,  -- 'buy', 'sell'
    quantity NUMERIC NOT NULL,
    price NUMERIC,
    fill_time TIMESTAMP WITH TIME ZONE,
    fees NUMERIC,
    slippage_bps NUMERIC,
    model_run_id VARCHAR(100),
    rationale TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_blotter_date_symbol ON trade_blotter(trade_date, symbol);
CREATE INDEX idx_blotter_model ON trade_blotter(model_run_id);

-- Daily performance metrics
CREATE TABLE IF NOT EXISTS daily_performance (
    date DATE PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    pnl NUMERIC NOT NULL,
    sharpe NUMERIC,
    max_drawdown NUMERIC,
    turnover NUMERIC,
    exposure_gross NUMERIC,
    exposure_net NUMERIC,
    realized_impact_bps NUMERIC,
    modeled_impact_bps NUMERIC,
    factor_exposures JSONB,
    greeks JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_perf_date_type ON daily_performance(date, model_type);

-- Drift monitoring
CREATE TABLE IF NOT EXISTS drift_metrics (
    id SERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    drift_type VARCHAR(50) NOT NULL,  -- 'psi', 'kl_divergence', 'mean_shift', etc.
    drift_value NUMERIC NOT NULL,
    threshold NUMERIC,
    is_alert BOOLEAN NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_drift_date_feature ON drift_metrics(metric_date, feature_name);
CREATE INDEX idx_drift_alert ON drift_metrics(is_alert);

-- System health / tripwires
CREATE TABLE IF NOT EXISTS system_health (
    id SERIAL PRIMARY KEY,
    check_ts TIMESTAMP WITH TIME ZONE NOT NULL,
    component VARCHAR(100) NOT NULL,  -- 'data_freshness', 'model_performance', 'execution', etc.
    status VARCHAR(20) NOT NULL,  -- 'ok', 'warning', 'critical'
    message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_health_ts_component ON system_health(check_ts, component);
CREATE INDEX idx_health_status ON system_health(status);

-- Corporate actions reference
CREATE TABLE IF NOT EXISTS corporate_actions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    event_type VARCHAR(20) NOT NULL,  -- 'split', 'dividend', 'merger', etc.
    ex_date DATE NOT NULL,
    record_date DATE,
    pay_date DATE,
    ratio NUMERIC,
    amount NUMERIC,
    source VARCHAR(50) NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_corpaction_symbol_exdate ON corporate_actions(symbol, ex_date);
CREATE INDEX idx_corpaction_type ON corporate_actions(event_type);

-- Delistings reference
CREATE TABLE IF NOT EXISTS delistings (
    symbol VARCHAR(20) PRIMARY KEY,
    delist_date DATE NOT NULL,
    reason VARCHAR(200),
    source VARCHAR(50) NOT NULL,
    ingested_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_delist_date ON delistings(delist_date);

-- COMMENT statements for documentation
COMMENT ON TABLE data_ingestion_log IS 'Audit log of all data ingestion jobs with checksums and lineage';
COMMENT ON TABLE data_quality_log IS 'QC checks results for schema validation, outliers, coverage gaps';
COMMENT ON TABLE model_runs IS 'Complete audit trail of model training with config hashing for reproducibility';
COMMENT ON TABLE model_metrics IS 'Per-fold and overall metrics from CPCV; PBO and DSR stored here';
COMMENT ON TABLE model_champions IS 'Champion-challenger promotion history with governance trail';
COMMENT ON TABLE shadow_trades IS 'Shadow trading signals from challenger models during evaluation period';
COMMENT ON TABLE trade_blotter IS 'Live/paper trade executions with slippage tracking and model attribution';
COMMENT ON TABLE daily_performance IS 'Daily rollup of PnL, risk metrics, exposures, and Greeks';
COMMENT ON TABLE drift_metrics IS 'Feature drift monitoring (PSI, KL) for regime change detection';
COMMENT ON TABLE system_health IS 'Tripwire and health check log for alerting and kill switches';
COMMENT ON TABLE corporate_actions IS 'Point-in-time corporate actions for survivorship-bias-free backtesting';
COMMENT ON TABLE delistings IS 'Delisting database to include defunct tickers in historical universes';
