"""
Colab/Kaggle Inference Playbook for Vajra

Demonstrates how to invoke the defensive posture inference using Vajra's 
native structured types (VajraInferenceRequest) and parse the results,
instead of relying on standard conversational LM formats.
"""

import json
from vajra.config import VajraConfig
from vajra.interface.pipeline import VajraInferencePipeline
from vajra.interface.types import VajraInferenceRequest, SecurityEvent

def generate_cyber_response(pipeline: VajraInferencePipeline, user_query: str, domain: str = "incident_response"):
    """
    Mimic the defensive reasoning logic by wrapping the user query into 
    Vajra's strict typed SecurityEvent stream.
    """
    
    # Vajra expects telemetry/events, not raw chat messages. 
    # We map the text query into an 'incident_response' event structure.
    event = SecurityEvent(
        event_id="evt_colab_1",
        timestamp=1680000000.0, # Dummy timestamp
        domain=domain,
        event_type="analyst_query",
        data={"query_content": user_query}
    )
    
    request = VajraInferenceRequest(
        request_id="req_playbook_1",
        events=[event],
        request_justification_trace=True,
        require_attack_graph=True
    )
    
    # Execute the structured inference pipeline
    # The pipeline inherently enforces the system defensive posture through 
    # the BaNEL loss trained weights and the Epistemic Fusion alignment.
    response = pipeline.run(request)
    
    return response

if __name__ == "__main__":
    print("Loading Vajra Inference Pipeline...")
    # Initialize with tiny=True just for playbook structural validation
    cfg = VajraConfig()
    pipeline = VajraInferencePipeline(cfg, tiny=True)
    pipeline.eval()
    
    # Example query directly modeled after Fenrir-v2.1 scenarios
    query = "Compare mitigations for Reflected vs Stored XSS in a modern SPA."
    print(f"\nQuery: {query}")
    
    try:
        response = generate_cyber_response(pipeline, query)
        print("\n--- Vajra Inference Response ---")
        print(f"Decision Class: {response.decision_class}")
        print(f"Confidence: {response.confidence:.4f}")
        print(f"ATT&CK Techniques Identified: {response.technique_ids}")
        print(f"AFN Activation Score Count: {len(response.afn_scores)}")
        
        if response.justification_trace:
            print(f"Decoder Trace: {response.justification_trace}")
    except NotImplementedError as e:
        print("\nNote: Full inference wiring (tokenization paths) is required to run the payload.")
        print(f"Exception caught as expected: {e}")
