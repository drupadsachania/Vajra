"""
Synthetic Data Distillation Pipeline (Stage 2)

This script automates the generation of synthetic chain-of-thought traces
from frontier models (Gemini 3.1 and OPUS 4.8). The generated data is used 
for Stage 2 distillation to teach Vajra's Epistemic Fusion layer and 
Constrained Decoder how to format defensive rationale.

Note: In a real environment, you must provide the respective API keys 
(GEMINI_API_KEY, ANTHROPIC_API_KEY) in your environment variables.
"""

import os
import json
import asyncio
from pathlib import Path
from dataclasses import asdict

# Mock clients for demonstration. In production, use `google-genai` and `anthropic`
class MockGeminiClient:
    async def generate_content_async(self, prompt: str) -> str:
        await asyncio.sleep(0.5)
        return f"<decoded:gemini_3.1_trace> Analyst observes multiple failed logins followed by lateral movement. Decision: ESCALATE (1) due to high confidence of credential stuffing leading to internal pivot. Mitigations include isolating the host and resetting AD credentials."

class MockOpusClient:
    async def generate_content_async(self, prompt: str) -> str:
        await asyncio.sleep(0.5)
        return f"<decoded:opus_4.8_trace> Compliance policy requires immediate logging of anomalous cross-domain IAM assume-role events. Decision: DEFER (2) for tier-2 review. Evidence points to T1078 (Valid Accounts)."

async def generate_traces(scenarios: list[str], output_file: Path):
    gemini = MockGeminiClient()
    opus = MockOpusClient()
    
    distilled_data = []
    
    print(f"Starting synthetic distillation using Gemini 3.1 and OPUS 4.8 across {len(scenarios)} scenarios...")
    
    # We use a system prompt strictly aligning the frontier models to Vajra's defensive posture
    system_prompt = (
        "You are a Tier-3 SOC analyst. Evaluate the following scenario and provide a "
        "structured rationale trace. You must output a decision class (0: ACT, 1: ESCALATE, "
        "2: DEFER, 3: INSUFFICIENT_CONTEXT) and identify relevant MITRE ATT&CK techniques."
    )
    
    for i, scenario in enumerate(scenarios):
        prompt = f"{system_prompt}\n\nScenario: {scenario}"
        
        # We interleave models to get diverse reasoning traces
        if i % 2 == 0:
            print(f"Querying Gemini 3.1 for scenario {i+1}...")
            trace = await gemini.generate_content_async(prompt)
            model_source = "gemini-3.1-pro"
        else:
            print(f"Querying OPUS 4.8 for scenario {i+1}...")
            trace = await opus.generate_content_async(prompt)
            model_source = "claude-opus-4-8"
            
        example = {
            "source_model": model_source,
            "scenario": scenario,
            "trace": trace,
            "domain": "incident_response" if "lateral" in scenario else "compliance"
        }
        distilled_data.append(example)
        
    # Write to raw data directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(distilled_data, f, indent=2)
        
    print(f"Successfully wrote {len(distilled_data)} distilled traces to {output_file}")

if __name__ == "__main__":
    # Example scenarios covering IR and Compliance domains
    scenarios = [
        "Multiple failed RDP logins from 192.168.1.50 followed by successful SMB tree connect to DC01.",
        "IAM AssumeRole event from a newly created AWS Lambda function targeting AdministratorAccess.",
        "Process injection detected in lsass.exe originating from a spawned powershell.exe.",
        "Unusual outbound traffic on port 443 to an unknown domain lacking SSL certificate transparency."
    ]
    
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "synthetic_distillation.json"
    asyncio.run(generate_traces(scenarios, output_path))
