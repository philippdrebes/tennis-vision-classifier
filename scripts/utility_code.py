#!/usr/bin/env python3

# Utility functions to log the time taken for each step of a process, helpful for profiling and improving perfomance and processing time

import logging
logging.basicConfig(level=logging.INFO)   # I changed this to INFO becaue DEBUG logging was getting too messy with readouts from requests
formatter = logging.Formatter(f"%(levelname)s Line %(lineno)3d  %(asctime)s  →   %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.setFormatter(formatter)

import functools, time, pympler
def profiler_mem_time(func):
    """Print the runtime and memory consumption of the decorated function, inspired by https://realpython.com/primer-on-python-decorators/ """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)   # run the wrapped function
        end_time = time.perf_counter()
        run_duration = end_time - start_time
        mem_usage = pympler.asizeof.asizeof(value)
        logging.info(f"{run_duration:.2f}s {mem_usage / (1024 ** 2):.2f}mb for {func.__name__!r}")
        return value
    return wrapper_timer

import functools, time, logging, cProfile, pstats, io, pympler, re
from pympler import asizeof   # strangely you need to manually import this for it work inside modelling_fetch_datasets()
def profiler_mem_time_detailed(func):
    @functools.wraps(func)
    def wrapper_profiler(*args, **kwargs):
        pr, s = cProfile.Profile(), io.StringIO()
        pr.enable()
        start_time, value = time.perf_counter(), func(*args, **kwargs)
        pr.disable()

        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        max_cumtime = next((float(re.search(r'^\s*\d+/?\d*\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)\s+', line).group(1)) for line in s.getvalue().split('\n') if re.search(r'^\s*\d+/?\d*\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+', line)), 1)

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats(30)
        lines = [line for line in s.getvalue().split('\n') if not line.startswith('   Ordered by') and not line.startswith('   List reduced')]

        def process_line(line):
            match = re.match(r'^(\s*\d+/?\d*\s+)(\d+\.\d+)', line)
            if match:
                tottime = float(match.group(2))
                try:
                    percentage = tottime / max_cumtime * 100 if max_cumtime > 0 else 0
                except ZeroDivisionError:
                    percentage = 0
                return f"{match.group(1)}{percentage:4.1f}%     {tottime:.3f} " + line[len(match.group(0)):], percentage
            return line, 0

        processed_lines, percentages = zip(*[process_line(line) for line in lines])

        cumusum = 0
        filtered_lines = []
        for line, perc in zip(processed_lines, percentages):
            if cumusum <= 95:
                filtered_lines.append(line)
                cumusum += perc

        header_index = next(i for i, line in enumerate(filtered_lines) if 'ncalls' in line)
        filtered_lines[header_index] = '   ncalls  tot_time%  tot_time  percall  cumtime  percall filename:lineno(function)'

        logging.info(re.sub(r'\s(/.*/python3\.\d+(/site-packages)?)/', ' ', '\n'.join(filtered_lines)))
        logging.info(f"{time.perf_counter() - start_time:.2f}s {pympler.asizeof.asizeof(value) / (1024 ** 2):.2f}mb to finish {func.__name__!r}\n\n")
        return value
    return wrapper_profiler

from concurrent.futures import ThreadPoolExecutor
import functools, re
def threaded_openai_call(max_workers=10):
    def decorator_openai_call(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result()
        return wrapper
    return decorator_openai_call
