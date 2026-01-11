#!/usr/bin/env python3
"""
Check which OpenAI models are available to your API key.

Usage:
    python scripts/check_openai_models.py

Environment:
    OPENAI_API_KEY: Your OpenAI API key (required)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("❌ Error: openai package not installed")
    print("Run: pip install openai")
    sys.exit(1)


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it in your .env file or export it")
        sys.exit(1)

    print("🔍 Checking available OpenAI models...")
    print(f"Using API key: {api_key[:20]}...{api_key[-4:]}\n")

    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()

        # Filter for GPT models
        gpt_models = [m for m in models if 'gpt' in m.id.lower()]
        gpt_models.sort(key=lambda m: m.id)

        print(f"✅ Found {len(gpt_models)} GPT models:\n")

        # Categorize models
        gpt4_models = []
        gpt5_models = []
        other_models = []

        for model in gpt_models:
            if model.id.startswith('gpt-4'):
                gpt4_models.append(model.id)
            elif model.id.startswith('gpt-5'):
                gpt5_models.append(model.id)
            else:
                other_models.append(model.id)

        if gpt4_models:
            print("📦 GPT-4 Series (Available):")
            for model_id in gpt4_models:
                marker = "✨" if model_id in ['gpt-4o-mini', 'gpt-4o'] else "  "
                print(f"   {marker} {model_id}")
            print()

        if gpt5_models:
            print("🚀 GPT-5 Series (Available):")
            for model_id in gpt5_models:
                print(f"   ✨ {model_id}")
            print()
        else:
            print("⏳ GPT-5 Series: Not yet available")
            print("   GPT-5 models are not publicly released yet")
            print()

        if other_models:
            print("📋 Other GPT Models:")
            for model_id in other_models:
                print(f"      {model_id}")
            print()

        # Recommendations
        print("💡 Recommendations for Voice Agent:")
        if 'gpt-4o-mini' in gpt4_models:
            print("   ✅ Use: OPENAI_MODEL=gpt-4o-mini (best balance of speed & quality)")
        elif 'gpt-4o' in gpt4_models:
            print("   ✅ Use: OPENAI_MODEL=gpt-4o (high quality, slower)")
        elif gpt4_models:
            print(f"   ✅ Use: OPENAI_MODEL={gpt4_models[0]}")

        if gpt5_models:
            print(f"   🚀 Try: OPENAI_MODEL={gpt5_models[0]} (GPT-5 available!)")

    except Exception as e:
        print(f"❌ Error querying OpenAI API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
