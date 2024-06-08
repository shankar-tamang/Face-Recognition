import cProfile
import pstats
from pathlib import Path

def run_profiler():
    import model

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()
    run_profiler()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('profile_results.prof')