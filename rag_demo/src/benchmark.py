# benchmark.py

import asyncio
import time
from query_engine import answer_query_async, batch_answer_queries

async def run_benchmark():
    # Test queries
    test_queries = [
        "What is Chloe's educational background?",
        "What programming languages does she know?",
        "What was her role at Snarkify?",
        "What kind of projects has she worked on?",
        "What database technologies is she familiar with?"
    ]
    
    # 1. Sequential test (using async functions one at a time)
    print("Testing sequential query processing...")
    sync_start = time.time()
    
    for query in test_queries:
        # Directly using the async function, not a synchronous wrapper
        answer = await answer_query_async(query)
        print(f"Query: {query[:30]}... - Completed")
    
    sync_time = time.time() - sync_start
    print(f"Total time for sequential processing: {sync_time:.2f} seconds")
    
    # 2. Asynchronous concurrent test
    print("\nTesting asynchronous concurrent query processing...")
    async_start = time.time()
    
    answers = await batch_answer_queries(test_queries)
    for i, answer in enumerate(answers):
        print(f"Query: {test_queries[i][:30]}... - Completed")
    
    async_time = time.time() - async_start
    print(f"\nPerformance Comparison:")
    print(f"Sequential processing: {sync_time:.2f} seconds")
    print(f"Asynchronous processing: {async_time:.2f} seconds")
    print(f"Speed improvement: {(sync_time/async_time):.2f}x")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
