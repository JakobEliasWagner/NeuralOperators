import asyncio
import sys


class ProgressMonitor:
    @staticmethod
    def monitor_pool(result, queue, total_size, prefix: str = "", suffix: str = "", update_interval: float = 1.0):
        asyncio.run(ProgressMonitor.monitor_pool_async(result, queue, total_size, prefix, suffix, update_interval))

    @staticmethod
    async def monitor_pool_async(
        result, queue, total_size, prefix: str = "", suffix: str = "", update_interval: float = 1.0
    ):
        while True:
            if result.ready():
                ProgressMonitor.print_progress_bar(1, 1, prefix, suffix)
                break
            else:
                size = queue.qsize()
                ProgressMonitor.print_progress_bar(size, total_size, prefix, suffix)
            await asyncio.sleep(update_interval)

    @staticmethod
    def print_progress_bar(iteration, total, prefix="", suffix="", length=50, fill="█"):
        """Call in a loop to create terminal progress bar.

        :param iteration: current iteration
        :param total: total iterations
        :param prefix: prefix
        :param suffix: suffix
        :param length: character length of bar
        :param fill: bar fill character
        :return:
        """
        percent = f"{100 * (iteration / float(total)):.1f}"
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)
        sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
        sys.stdout.flush()
