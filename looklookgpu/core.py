import os
import time
from collections import deque
from itertools import cycle

import psutil
import pynvml
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Color palette for charts
COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red", "white"]


class Looklook:
    def __init__(
        self,
        total_epochs=None,
        cpu=True,
        gpu=True,
        layout_ratio=(1, 2),
        charts=None,
        smoothing=0.6,
        height=10,
        refresh_rate=4,
    ):
        self.total_epochs = total_epochs
        self.show_cpu = cpu
        # 严格判定 gpu=0
        self.show_gpu = gpu is not False

        self.l_ratio, self.r_ratio = layout_ratio
        self.smoothing = max(0.0, min(0.99, smoothing))
        self.graph_height = height
        self.refresh_rate = refresh_rate
        self.start_time = time.time()
        self.history_len = 120

        self.console = Console()
        self.gpu_status_msg = "Init..."

        if isinstance(charts, dict):
            self.chart_config = charts
        elif isinstance(charts, list):
            self.chart_config = {f"Chart {i + 1}": g for i, g in enumerate(charts)}
        else:
            self.chart_config = {}

        self.metrics_history = {}
        self.metrics_smoothed = {}
        self.sys_history = {
            "GPU Util Avg": deque([0], maxlen=self.history_len),
            "GPU Mem Avg": deque([0], maxlen=self.history_len),
            "CPU Util": deque([0], maxlen=self.history_len),
        }
        self.gpu_history_data = {}

        # --- GPU Initialization ---
        self.nvml_handles = {}
        self.target_gpu_ids = []
        self.total_phys_gpus = 0

        if self.show_gpu:
            try:
                pynvml.nvmlInit()
                self.total_phys_gpus = pynvml.nvmlDeviceGetCount()
                self.target_gpu_ids = self._get_target_gpus(gpu, self.total_phys_gpus)

                if not self.target_gpu_ids:
                    self.show_gpu = False
                    self.gpu_status_msg = "[red]NONE[/]"
                else:
                    for i in self.target_gpu_ids:
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        self.nvml_handles[i] = h
                        self.gpu_history_data[f"GPU {i} Util"] = deque(
                            [0], maxlen=self.history_len
                        )
                        self.gpu_history_data[f"GPU {i} Mem"] = deque(
                            [0], maxlen=self.history_len
                        )

                    # Python 3.9 安全拼接
                    ids_str = ",".join(map(str, self.target_gpu_ids))
                    if len(self.target_gpu_ids) == 1:
                        self.gpu_status_msg = "[green]OK[/]"
                    else:
                        self.gpu_status_msg = f"[green]{ids_str}[/]"
            except Exception:
                self.gpu_status_msg = "[red]ERR[/]"
                self.show_gpu = False

        # --- Layout ---
        self.layout = Layout()
        self.layout.split(Layout(name="header", size=4), Layout(name="body", ratio=1))
        self.layout["body"].split_row(
            Layout(name="left", ratio=self.l_ratio),
            Layout(name="right", ratio=self.r_ratio),
        )

        # 动态计算表格高度以确保多卡显示
        table_size = 2 + (1 if self.show_cpu else 0) + len(self.target_gpu_ids)
        l_els = [Layout(name="sys_table", size=table_size)]
        if self.show_cpu:
            l_els.append(Layout(name="cpu_chart"))
        if self.show_gpu:
            l_els.append(Layout(name="gpu_chart"))
        self.layout["body"]["left"].split_column(*l_els)

        self.layout_right_initialized = False

    def _get_target_gpus(self, gpu_config, phys_count):
        if isinstance(gpu_config, int):
            return [gpu_config] if gpu_config < phys_count else []
        if isinstance(gpu_config, list):
            return [i for i in gpu_config if i < phys_count]
        env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env:
            try:
                return [
                    int(i.strip())
                    for i in env.split(",")
                    if i.strip() and int(i.strip()) < phys_count
                ]
            except:
                pass
        return [0] if phys_count > 0 else []

    def _init_right_layout(self):
        elements = [Layout(name="metrics_table", size=6)]
        for title in self.chart_config.keys():
            elements.append(Layout(name=f"chart_{abs(hash(title))}"))
        self.layout["body"]["right"].split_column(*elements)
        self.layout_right_initialized = True

    def _get_system_stats(self):
        if self.show_cpu:
            self.sys_history["CPU Util"].append(psutil.cpu_percent())
        if self.show_gpu and self.nvml_handles:
            u_sum, m_sum, count = 0, 0, 0
            for i, h in self.nvml_handles.items():
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    mu = (m.used / m.total) * 100
                    self.gpu_history_data[f"GPU {i} Util"].append(u)
                    self.gpu_history_data[f"GPU {i} Mem"].append(mu)
                    u_sum += u
                    m_sum += mu
                    count += 1
                except:
                    pass
            if count > 0:
                self.sys_history["GPU Util Avg"].append(u_sum / count)
                self.sys_history["GPU Mem Avg"].append(m_sum / count)

    def _draw_solid_bars(self, data_deque, title, color, available_width):
        """Hardware Bars (Python 3.9 Friendly)"""
        if not data_deque:
            return Panel("Waiting...", title=title)
        data = list(data_deque)[-(max(10, available_width - 10)) :]
        lines, bars = [""] * self.graph_height, "  ▂▃▄▅▆▇█"
        for val in data:
            h_idx = int((val / 100) * self.graph_height * 8)
            for r in range(self.graph_height):
                bot = (self.graph_height - 1 - r) * 8
                if h_idx >= bot + 8:
                    lines[r] += f"[{color}]█[/]"
                elif h_idx <= bot:
                    lines[r] += " "
                else:
                    lines[r] += f"[{color}]{bars[max(0, min(7, h_idx - bot))]}[/]"

        # 反斜杠处理：外部 join
        top_l, bot_l = f"[{color}]100%[/] {lines[0]}", f"[{color}]  0%[/] {lines[-1]}"
        mid_ls = [f"     {l}" for l in lines[1:-1]]
        all_ls = [top_l] + mid_ls + [bot_l]
        chart_str = "\n".join(all_ls)

        curr, avg = data[-1], sum(data) / len(data)
        footer = f"[dim]Cur:[/][{color}] {curr:>5.1f}%[/] [dim]| Avg:[/][{color}] {avg:>5.1f}%[/]"

        content = Text.from_markup(f"{chart_str}\n{footer}")
        content.no_wrap = True
        return Panel(content, title=title, border_style=color, box=box.HORIZONTALS)

    def _draw_braille_lines(self, series_list, labels, colors, title, available_width):
        """Training Curves (Python 3.9 Friendly)"""
        if not series_list or not series_list[0]:
            return Panel("Waiting...", title=title)
        max_pts = max(10, available_width - 12)
        sliced = [list(s)[-max_pts:] for s in series_list]
        all_vals = [x for s in sliced for x in s]
        min_v, max_v = min(all_vals), max(all_vals)
        rng = (max_v - min_v) if max_v != min_v else 1e-5

        lines = [""] * self.graph_height
        for x in range(len(sliced[0])):
            dots, row_cols = [0] * self.graph_height, [""] * self.graph_height
            for i, series in enumerate(sliced):
                y = int((series[x] - min_v) / rng * (self.graph_height * 4 - 1))
                r = (self.graph_height - 1) - (y // 4)
                if 0 <= r < self.graph_height:
                    if not row_cols[r]:
                        row_cols[r] = colors[i]
                    dots[r] |= {0: 0x40, 1: 0x4, 2: 0x2, 3: 0x1}[y % 4]
            for r in range(self.graph_height):
                lines[r] += f"[{row_cols[r] or colors[0]}]{chr(0x2800 + dots[r])}[/]"

        # 反斜杠处理：外部 join
        c_col = colors[0]
        max_t, min_t = f"[{c_col}]{max_v:>7.4f}[/]", f"[{c_col}]{min_v:>7.4f}[/]"
        lab_ls = (
            [f"{max_t} {lines[0]}"]
            + [f"{' ' * 7} {l}" for l in lines[1:-1]]
            + [f"{min_t} {lines[-1]}"]
        )
        chart_str = "\n".join(lab_ls)

        legend = "  ".join([f"[{c}]■ {l}[/]" for l, c in zip(labels, colors)])
        foot_items = [
            f"[{c}]{l}[/] = [white]{s[-1]:.4f}[/]"
            for l, s, c in zip(labels, sliced, colors)
        ]
        footer = " | ".join(foot_items)

        content = Text.from_markup(f"{chart_str}\n{legend}\n[dim]{footer}[/]")
        content.no_wrap = True
        return Panel(content, title=title, border_style=c_col, box=box.HORIZONTALS)

    def update(self, epoch, metrics):
        self._get_system_stats()
        term_w = self.console.width
        for k, v in metrics.items():
            if k not in self.metrics_history:
                self.metrics_history[k] = deque(maxlen=self.history_len)
            if k not in self.metrics_smoothed:
                self.metrics_smoothed[k] = v
            self.metrics_smoothed[k] = self.metrics_smoothed[k] * self.smoothing + v * (
                1 - self.smoothing
            )
            self.metrics_history[k].append(self.metrics_smoothed[k])
        if not self.layout_right_initialized:
            self._init_right_layout()

        # 1. Header (两行分别居中)
        duration = time.time() - self.start_time
        target_ids_str = ",".join(map(str, self.target_gpu_ids))
        top_line = f"╼ [bold white]L O O K L O O K[/] ╾ | GPU: {self.gpu_status_msg} | Time: [yellow]{duration:.0f}s[/yellow]"
        p_len = 25
        if self.total_epochs:
            val = int((epoch / self.total_epochs) * p_len)
            p_bar = f"Progress: [bold green][{'█' * val}{' ' * (p_len - val)}][/] {int((epoch / self.total_epochs) * 100)}% · Epoch: {epoch}/{self.total_epochs}"
        else:
            p_bar = f"Epoch: [bold green]{epoch}[/]"

        header_text = Text.from_markup(f"{top_line}\n{p_bar}", justify="center")
        self.layout["header"].update(
            Panel(Align.center(header_text), box=box.HORIZONTALS, padding=(0, 1))
        )

        # 2. Left
        l_w = int(term_w * (self.l_ratio / (self.l_ratio + self.r_ratio)))
        sys_t = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        if self.show_cpu:
            sys_t.add_row(
                "[bold]CPU Util[/]",
                f"[cyan]{self.sys_history['CPU Util'][-1]:>5.1f}%[/]",
            )
        for i in self.target_gpu_ids:
            u, m = (
                self.gpu_history_data[f"GPU {i} Util"][-1],
                self.gpu_history_data[f"GPU {i} Mem"][-1],
            )
            sys_t.add_row(
                f"[bold magenta]GPU {i}[/]",
                f"Load: [magenta]{u:>5.1f}%[/] | Mem: [magenta]{m:>5.1f}%[/]",
            )
        self.layout["body"]["left"]["sys_table"].update(
            Panel(sys_t, title="💻 System", border_style="cyan", box=box.HORIZONTALS)
        )
        if self.show_cpu:
            self.layout["body"]["left"]["cpu_chart"].update(
                self._draw_solid_bars(
                    self.sys_history["CPU Util"], "CPU Load", "cyan", l_w
                )
            )
        if self.show_gpu:
            title = (
                f"GPU Load (ID:{self.target_gpu_ids[0]})"
                if len(self.target_gpu_ids) == 1
                else "GPU Avg Load"
            )
            self.layout["body"]["left"]["gpu_chart"].update(
                self._draw_solid_bars(
                    self.sys_history["GPU Util Avg"], title, "magenta", l_w
                )
            )

        # 3. Right
        r_w = term_w - l_w
        t_t = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        items = list(metrics.items())
        for i in range(0, len(items), 3):
            cells = [f"[bold]{k}[/]\n[yellow]{v:.4f}[/]" for k, v in items[i : i + 3]]
            t_t.add_row(*cells)
        self.layout["body"]["right"]["metrics_table"].update(
            Panel(t_t, title="🔥 Metrics", border_style="green", box=box.HORIZONTALS)
        )
        for title, group in self.chart_config.items():
            s_list, labs, cols, c_gen = [], [], [], cycle(COLORS)
            for m in group:
                if m in self.metrics_history:
                    s_list.append(self.metrics_history[m])
                    labs.append(m)
                    cols.append(next(c_gen))
            if s_list:
                try:
                    self.layout["body"]["right"][f"chart_{abs(hash(title))}"].update(
                        self._draw_braille_lines(s_list, labs, cols, title, r_w)
                    )
                except:
                    pass
        return self.layout

    def live(self):
        return Live(self.layout, refresh_per_second=self.refresh_rate, screen=True)

    def close(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass
