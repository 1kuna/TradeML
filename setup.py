"""
TradeML setup and validation script.

Run this after setting up your environment to verify all components are working.
"""

import os
import sys
from pathlib import Path
from datetime import date, timedelta
import subprocess


def check_python_version():
    """Verify Python version is 3.11+."""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]

    if major < 3 or (major == 3 and minor < 11):
        print(f"❌ Python {major}.{minor} detected. Requires Python 3.11+")
        return False

    print(f"✅ Python {major}.{minor} OK")
    return True


def check_env_file():
    """Verify .env file exists."""
    print("\nChecking environment configuration...")

    if not Path(".env").exists():
        print("❌ .env file not found")
        print("   Run: cp .env.template .env")
        print("   Then edit .env with your API keys")
        return False

    print("✅ .env file exists")

    # Check for required keys
    from dotenv import load_dotenv
    load_dotenv()

    required_keys = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
    ]

    missing = []
    for key in required_keys:
        if not os.getenv(key):
            missing.append(key)

    if missing:
        print(f"⚠️  Missing API keys in .env: {', '.join(missing)}")
        print("   Add these keys to .env file")
        return False

    print("✅ Required API keys found")
    return True


def check_docker_services():
    """Verify Docker services are running."""
    print("\nChecking Docker services...")

    try:
        result = subprocess.run(
            ["docker-compose", "-f", "infra/docker-compose.yml", "ps"],
            capture_output=True,
            text=True,
            check=True
        )

        # Check if key services are up
        required_services = ["postgres", "minio"]
        output = result.stdout.lower()

        missing = []
        for service in required_services:
            if service not in output or "up" not in output:
                missing.append(service)

        if missing:
            print(f"❌ Services not running: {', '.join(missing)}")
            print("   Run: cd infra && docker-compose up -d")
            return False

        print("✅ Docker services running")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not running or docker-compose not found")
        print("   Start services: cd infra && docker-compose up -d")
        return False


def check_database_connection():
    """Test PostgreSQL connection."""
    print("\nChecking database connection...")

    try:
        import psycopg2
        from dotenv import load_dotenv
        load_dotenv()

        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
            database=os.getenv("POSTGRES_DB", "trademl"),
            user=os.getenv("POSTGRES_USER", "trademl"),
            password=os.getenv("POSTGRES_PASSWORD", "trademl_dev_pass")
        )

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
        table_count = cursor.fetchone()[0]

        conn.close()

        print(f"✅ Database connected ({table_count} tables)")
        return True

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def check_minio_connection():
    """Test MinIO/S3 connection."""
    print("\nChecking MinIO/S3 storage...")

    try:
        import boto3
        from dotenv import load_dotenv
        load_dotenv()

        s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{os.getenv('MINIO_ENDPOINT', 'localhost:9000')}",
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
        )

        # List buckets
        response = s3_client.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]

        required_buckets = ['raw', 'curated', 'reference']
        missing = [b for b in required_buckets if b not in buckets]

        if missing:
            print(f"⚠️  Missing buckets: {', '.join(missing)}")
            print("   Buckets will be created on first use")
        else:
            print(f"✅ MinIO connected ({len(buckets)} buckets)")

        return True

    except Exception as e:
        print(f"❌ MinIO connection failed: {e}")
        return False


def test_alpaca_connector():
    """Test Alpaca data fetching."""
    print("\nTesting Alpaca connector...")

    try:
        from data_layer.connectors.alpaca_connector import AlpacaConnector

        connector = AlpacaConnector()

        # Fetch 1 week of data for AAPL
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        df = connector.fetch_bars(
            symbols=["AAPL"],
            start_date=start_date,
            end_date=end_date,
            timeframe="1Day"
        )

        if df.empty:
            print("⚠️  No data returned (may be weekend/holiday)")
            return True

        print(f"✅ Fetched {len(df)} bars for AAPL")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        return True

    except Exception as e:
        print(f"❌ Alpaca connector test failed: {e}")
        return False


def test_calendar():
    """Test exchange calendar."""
    print("\nTesting exchange calendar...")

    try:
        from data_layer.reference.calendars import get_calendar

        cal = get_calendar("XNYS")

        # Check if today is a trading day
        is_trading = cal.is_trading_day(date.today())

        # Get next trading day
        next_day = cal.next_trading_day(date.today())

        print(f"✅ Calendar working")
        print(f"   Today is trading day: {is_trading}")
        print(f"   Next trading day: {next_day}")
        return True

    except Exception as e:
        print(f"❌ Calendar test failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("TradeML Setup Validation")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Docker Services", check_docker_services),
        ("Database Connection", check_database_connection),
        ("MinIO Storage", check_minio_connection),
        ("Exchange Calendar", test_calendar),
        ("Alpaca Connector", test_alpaca_connector),
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")

    print(f"\nScore: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! System is ready.")
        print("\nNext steps:")
        print("1. Review QUICKSTART.md for usage examples")
        print("2. Fetch historical data: python -m data_layer.connectors.alpaca_connector")
        print("3. Continue with Phase 1 implementation")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        print("See QUICKSTART.md for troubleshooting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
