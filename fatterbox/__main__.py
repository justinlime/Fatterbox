"""Allow running the package as: python -m fatterbox"""
import asyncio

from .main import main

if __name__ == "__main__":
    asyncio.run(main())
