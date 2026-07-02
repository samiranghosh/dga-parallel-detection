import time
import math
import random
from typing import Iterator, Tuple, List


def uniform_load(domains: List[str], rate: float, batch_size: int = 100) -> Iterator[Tuple[float, List[str]]]:
    """Generate uniform traffic at a constant rate.
    
    Args:
        domains: List of domains to serve.
        rate: Target arrival rate (domains per second).
        batch_size: Number of domains per yielded batch.
        
    Yields:
        (scheduled_time, list_of_domains)
    """
    start_time = time.time()
    for i in range(0, len(domains), batch_size):
        batch = domains[i:i + batch_size]
        # Calculate when this batch should arrive to maintain the target rate
        # Cumulative domains processed so far (including this batch) is i + len(batch)
        expected_time = start_time + (i + len(batch)) / rate
        yield expected_time, batch


def poisson_bursty_load(domains: List[str], mean_rate: float, 
                        burst_factor: float = 3.0, batch_size: int = 100) -> Iterator[Tuple[float, List[str]]]:
    """Generate bursty traffic using a Markov-modulated Poisson process (simplified).
    
    Alternates between a low rate and a high (burst) rate, keeping the overall 
    mean rate approximately equal to `mean_rate`.
    
    Args:
        domains: List of domains to serve.
        mean_rate: Overall target arrival rate.
        burst_factor: Multiplier for the burst rate relative to the mean.
        batch_size: Number of domains per yielded batch.
        
    Yields:
        (scheduled_time, list_of_domains)
    """
    low_rate = mean_rate / burst_factor
    burst_rate = mean_rate * burst_factor
    
    current_time = time.time()
    in_burst = False
    
    # Simple state machine: flip state randomly, weighting time spent
    for i in range(0, len(domains), batch_size):
        batch = domains[i:i + batch_size]
        
        # 10% chance to flip state
        if random.random() < 0.1:
            in_burst = not in_burst
            
        current_rate = burst_rate if in_burst else low_rate
        
        delay = len(batch) / current_rate
        current_time += delay
        yield current_time, batch


def ramp_load(domains: List[str], start_rate: float, end_rate: float, 
              batch_size: int = 100) -> Iterator[Tuple[float, List[str]]]:
    """Generate traffic with a linearly increasing or decreasing arrival rate.
    
    Args:
        domains: List of domains to serve.
        start_rate: Initial arrival rate.
        end_rate: Final arrival rate when domains are exhausted.
        batch_size: Number of domains per yielded batch.
        
    Yields:
        (scheduled_time, list_of_domains)
    """
    total_batches = math.ceil(len(domains) / batch_size)
    if total_batches == 0:
        return
        
    rate_step = (end_rate - start_rate) / total_batches
    current_time = time.time()
    
    for step, i in enumerate(range(0, len(domains), batch_size)):
        batch = domains[i:i + batch_size]
        current_rate = start_rate + (step * rate_step)
        # Avoid division by zero if rate drops to 0
        current_rate = max(current_rate, 1.0)
        
        delay = len(batch) / current_rate
        current_time += delay
        yield current_time, batch
