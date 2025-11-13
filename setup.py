"""
Setup file for algo-trading-system
Makes the package importable
"""
from setuptools import setup, find_packages

setup(
    name="algo-trading-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "yfinance>=0.2.32",
        "alpha-vantage>=2.3.1",
        "requests>=2.32.5",
        "websocket-client>=1.6.4",
        "pandas>=2.1.3",
        "numpy>=1.26.0,<2.0.0",
        "pytz>=2023.3",
        "psycopg2-binary>=2.9.9",
        "SQLAlchemy>=2.0.23",
        "redis>=5.0.1",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "schedule>=1.2.0",
        "loguru>=0.7.2",
        "pytest>=7.4.3",
    ],
    python_requires=">=3.9",
)
