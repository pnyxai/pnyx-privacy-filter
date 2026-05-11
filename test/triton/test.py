import argparse
import numpy as np
import tritonclient.http as httpclient
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor

# Sample text for testing redaction
DEFAULT_TEXT = "My name is Alice Smith. I live at 123 Main St, New York. My email is alice.smith@example.com and my phone is 555-0123."

def send_request(url, model_name, text, verbose=False):
    try:
        # Create a client
        client = httpclient.InferenceServerClient(url=url)
        
        # Create inputs: PROMPT is shaped [batch_size, 1] for BYTES
        # For a single request, batch_size is 1.
        prompt_data = np.array([[text.encode('utf-8')]], dtype=object)
        
        input_prompt = httpclient.InferInput("PROMPT", [1, 1], "BYTES")
        input_prompt.set_data_from_numpy(prompt_data)
        
        inputs = [input_prompt]
        outputs = [httpclient.InferRequestedOutput("LABELS")]

        start_time = time.time()
        results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        latency = time.time() - start_time
        
        # LABELS contains the JSON string of RedactionResult
        output_data = results.as_numpy("LABELS")
        
        if output_data is not None and len(output_data) > 0:
            # Handle list of results and extract the first element
            if isinstance(output_data, np.ndarray):
                json_val = output_data.flatten()[0]
            else:
                json_val = output_data[0]
                
            if isinstance(json_val, bytes):
                json_val = json_val.decode('utf-8')

            if verbose:
                print(f"DEBUG: Content of first output element: {json_val}")
            
            # Parse response for validation
            response = json.loads(str(json_val))
            
            if verbose:
                print(f"\n--- Inference Result ---")
                print(f"Latency: {latency*1000:.2f}ms")
                print(f"Redacted Text: {response.get('redacted_text', 'N/A')}")
                print(f"Detections found: {response.get('summary', {}).get('span_count', 0)}")
                for det in response.get('detected_spans', []):
                    print(f" - [{det['label']}] '{det['text']}' at {det['start']}:{det['end']}")
            
            return True, latency, response
        
        return False, latency, None
    except Exception as e:
        print(f"Request failed: {e}")
        return False, 0, None

def main():
    parser = argparse.ArgumentParser(description="Test and Benchmark Pnyx Privacy Filter Triton")
    parser.add_argument("--url", type=str, default="localhost:8000", help="Triton URL")
    parser.add_argument("--model", type=str, default="ensemble_model", help="Model name (ensemble_model, tokenizer_model, etc.)")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Text to process")
    parser.add_argument("--requests", type=int, default=5, help="Total number of requests for benchmarking")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent requests")
    parser.add_argument("--verbose", action="store_true", help="Print detailed inference results")
    args = parser.parse_args()
    
    # 1. Check Model Readiness
    try:
        initial_client = httpclient.InferenceServerClient(url=args.url)
        if not initial_client.is_model_ready(args.model):
            print(f"Error: Model '{args.model}' is not ready at {args.url}")
            sys.exit(1)
        print(f"Model '{args.model}' is READY.")
    except Exception as e:
        print(f"Error connecting to Triton: {e}")
        sys.exit(1)

    # 2. Single Diagnostic Request
    print(f"\nPerforming single diagnostic request...")
    success, lat, result = send_request(args.url, args.model, args.text, verbose=True)
    if not success:
        print("Initial request failed. Aborting.")
        sys.exit(1)

    # 3. Simple Benchmark (if requests > 1)
    if args.requests > 1:
        print(f"\nBenchmarking {args.model}...")
        print(f"Requests: {args.requests} | Concurrency: {args.concurrency}")

        start_bench = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(send_request, args.url, args.model, args.text)
                for _ in range(args.requests)
            ]
            results = [f.result() for f in futures]

        total_time = time.time() - start_bench
        successes = [r for r in results if r[0]]
        latencies = [r[1] for r in results if r[0] and r[1] > 0]

        print("\n--- Benchmark Results ---")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {len(successes) / total_time:.2f} requests/sec")
        if latencies:
            print(f"Avg Latency: {np.mean(latencies)*1000:.2f}ms")
            print(f"P95 Latency: {np.percentile(latencies, 95)*1000:.2f}ms")
        print(f"Success rate: {len(successes)}/{args.requests}")

if __name__ == "__main__":
    main()
