"""Tkinter desktop editor for a simplified 2D thermal room model.

Run: python main.py
Dependencies: numpy, scipy, matplotlib, pillow (Tkinter comes with Python).
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from matplotlib import colormaps

from model import Heater, Project, Wall, Window, example_project
from physics import ModelError, Solution, solve


class ThermalApp:
    TOOLS = [
        ("Стена", "wall"),
        ("Окно", "window"),
        ("Улица", "street"),
        ("Нагреватель", "heater"),
        ("Ластик", "erase"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Тепловая карта комнаты — 2D")
        root.geometry("1190x810")
        root.minsize(920, 620)
        self.project = example_project()
        self.solution: Solution | None = None
        self.preview_start: tuple[float, float] | None = None
        self.preview_end: tuple[float, float] | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.filename: str | None = None

        self.tool = tk.StringVar(value="wall")
        self.wall_thickness = tk.StringVar(value="0.25")
        self.window_length = tk.StringVar(value="1.50")
        self.heater_radius = tk.StringVar(value="0.55")
        self.heater_temperature = tk.StringVar(value="40")
        self.outdoor_temperature = tk.StringVar(value="-10")
        self.grid_step = tk.StringVar(value="0.10")
        self.air_k = tk.StringVar(value="0.10")
        self.concrete_k = tk.StringVar(value="1.40")
        self.window_k = tk.StringVar(value="0.80")
        self.show_field = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Готово. Выберите инструмент, чтобы изменить план.")
        self.stats = tk.StringVar(value="Расчёт ещё не выполнен")

        self._build_ui()
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_move)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)
        self.root.bind("<Control-s>", lambda _: self.save_project())
        self.root.bind("<Control-o>", lambda _: self.open_project())
        self.root.bind("<Return>", lambda _: self.compute())
        self.root.after(100, self.draw)

    def _build_ui(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Arial", 15, "bold"))
        style.configure("Section.TLabel", font=("Arial", 10, "bold"))
        style.configure("Run.TButton", font=("Arial", 11, "bold"), padding=(12, 9))

        toolbar = ttk.Frame(self.root, padding=(12, 9))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(toolbar, text="Тепловая карта комнаты", style="Title.TLabel").pack(side=tk.LEFT)
        for text, command in (("Пример", self.load_example), ("Новый", self.new_project),
                              ("Открыть…", self.open_project), ("Сохранить…", self.save_project)):
            ttk.Button(toolbar, text=text, command=command).pack(side=tk.RIGHT, padx=(5, 0))

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)
        side_canvas = tk.Canvas(body, width=300, highlightthickness=0, bg="#f4f6f9")
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=side_canvas.yview)
        side_canvas.configure(yscrollcommand=scrollbar.set)
        side_canvas.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        side = ttk.Frame(side_canvas, padding=13)
        side_window = side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda _: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda e: side_canvas.itemconfigure(side_window, width=e.width))

        def section(text: str):
            ttk.Label(side, text=text, style="Section.TLabel").pack(anchor="w", pady=(13, 7))

        section("Рисование")
        for caption, key in self.TOOLS:
            ttk.Radiobutton(side, text=caption, variable=self.tool, value=key).pack(anchor="w", pady=2)
        self._entry(side, "Толщина стены, м", self.wall_thickness)
        self._entry(side, "Длина окна, м", self.window_length)
        self._entry(side, "Радиус нагревателя, м", self.heater_radius)
        self._entry(side, "Температура нагревателя, °C", self.heater_temperature)
        ttk.Label(
            side, text="Стена: протянуть мышью. Окно: нажать на стену.\n"
                       "Улица: нажать вне комнаты. Нагреватель:\n"
                       "нажать внутри. Правая кнопка — удалить.",
            wraplength=274, justify="left", foreground="#586576",
        ).pack(anchor="w", pady=(7, 0))

        section("Физические параметры")
        self._entry(side, "Температура улицы, °C", self.outdoor_temperature)
        self._entry(side, "Шаг сетки, м", self.grid_step)
        self._entry(side, "λ воздуха (эфф.), Вт/(м·К)", self.air_k)
        self._entry(side, "λ бетона, Вт/(м·К)", self.concrete_k)
        self._entry(side, "λ окна, Вт/(м·К)", self.window_k)
        ttk.Button(side, text="Вычислить температуру", command=self.compute,
                   style="Run.TButton").pack(fill=tk.X, pady=(13, 5))
        ttk.Checkbutton(side, text="Показывать тепловую карту", variable=self.show_field,
                        command=self.draw).pack(anchor="w")

        section("Результат")
        ttk.Label(side, textvariable=self.stats, justify="left", wraplength=274).pack(anchor="w")
        ttk.Separator(side).pack(fill=tk.X, pady=15)
        ttk.Label(
            side,
            text="Модель: стационарная 2D-теплопроводность.\n"
                 "Улица и нагреватель — фиксированные температуры.\n"
                 "На внешней кромке сетки нет теплового потока.\n"
                 "Не учитывает конвекцию и излучение.",
            justify="left", wraplength=274, foreground="#697487",
        ).pack(anchor="w")

        self.canvas = tk.Canvas(body, bg="#edf1f5", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        statusbar = ttk.Label(self.root, textvariable=self.status, padding=(12, 7),
                              relief="sunken")
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    @staticmethod
    def _entry(parent, caption, value):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=caption).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=value, width=9, justify="right").pack(side=tk.RIGHT)

    def _view(self):
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        width, height = self.project.width, self.project.height
        scale = min((cw - 70) / width, (ch - 70) / height)
        scale = max(scale, 1.0)
        ox = (cw - width * scale) / 2
        oy = (ch - height * scale) / 2
        return ox, oy, scale

    def to_canvas(self, x, y):
        ox, oy, scale = self._view()
        return ox + x * scale, oy + y * scale

    def to_world(self, x, y, clamp=True):
        ox, oy, scale = self._view()
        px, py = (x - ox) / scale, (y - oy) / scale
        if clamp:
            px = min(max(px, 0), self.project.width)
            py = min(max(py, 0), self.project.height)
        return px, py

    def _snap(self, x, y):
        step = 0.1
        for wall in self.project.walls:
            for ex, ey in ((wall.x1, wall.y1), (wall.x2, wall.y2)):
                if math.hypot(x - ex, y - ey) < 0.20:
                    return ex, ey
        return round(x / step) * step, round(y / step) * step

    @staticmethod
    def _wall_projection(x, y, wall):
        dx, dy = wall.x2 - wall.x1, wall.y2 - wall.y1
        ll = dx * dx + dy * dy
        if ll == 0:
            return math.inf, 0
        t = max(0, min(1, ((x - wall.x1) * dx + (y - wall.y1) * dy) / ll))
        distance = math.hypot(x - wall.x1 - t * dx, y - wall.y1 - t * dy)
        return distance, t

    def _nearest_wall(self, x, y):
        if not self.project.walls:
            return None, 0.0, math.inf
        nearest = min(
            ((wall, *self._wall_projection(x, y, wall)) for wall in self.project.walls),
            key=lambda item: item[1],
        )
        return nearest

    def invalidate(self):
        self.solution = None
        self.stats.set("Параметры изменены — запустите расчёт")
        self.draw()

    def read_settings(self):
        try:
            values = {
                "outdoor_temperature": float(self.outdoor_temperature.get()),
                "step": float(self.grid_step.get()),
                "air_conductivity": float(self.air_k.get()),
                "concrete_conductivity": float(self.concrete_k.get()),
                "window_conductivity": float(self.window_k.get()),
            }
            if not (0.02 <= values["step"] <= 0.5):
                raise ValueError("Шаг сетки должен быть от 0.02 до 0.5 м")
            for k in ("air_conductivity", "concrete_conductivity", "window_conductivity"):
                if not (0 < values[k] <= 100):
                    raise ValueError("Теплопроводности должны быть положительными")
            for k, v in values.items():
                setattr(self.project, k, v)
        except ValueError as exc:
            raise ModelError(f"Проверьте физические параметры: {exc}") from exc

    def on_press(self, event):
        x, y = self.to_world(event.x, event.y)
        if self.tool.get() == "wall":
            self.preview_start = self._snap(x, y)
            self.preview_end = self.preview_start
            return
        try:
            if self.tool.get() == "street":
                self.project.street_seed = (min(x, self.project.width - 1e-8),
                                            min(y, self.project.height - 1e-8))
                self.status.set("Точка улицы установлена. Нажмите «Вычислить».")
            elif self.tool.get() == "heater":
                radius = float(self.heater_radius.get())
                temperature = float(self.heater_temperature.get())
                if not (0.02 <= radius <= 2):
                    raise ValueError("Радиус должен быть от 0.02 до 2 м")
                self.project.heaters.append(Heater(x, y, radius, temperature))
            elif self.tool.get() == "window":
                wall, distance, t = self._nearest_wall(x, y)
                if wall is None or distance > max(0.25, wall.thickness):
                    self.status.set("Нажмите на существующую стену, чтобы добавить окно.")
                    return
                length = float(self.window_length.get())
                length_wall = math.hypot(wall.x2 - wall.x1, wall.y2 - wall.y1)
                if length <= 0 or length > length_wall:
                    raise ValueError("Длина окна должна быть положительной и не длиннее стены")
                t = min(max(t, length / (2 * length_wall)),
                        1 - length / (2 * length_wall))
                wall.windows.append(Window(t, length))
            elif self.tool.get() == "erase":
                self.erase(x, y)
                return
            self.invalidate()
        except ValueError as exc:
            messagebox.showerror("Некорректное значение", str(exc))

    def on_drag(self, event):
        if self.preview_start is not None:
            self.preview_end = self._snap(*self.to_world(event.x, event.y))
            self.draw()

    def on_release(self, event):
        if self.preview_start is None:
            return
        end = self._snap(*self.to_world(event.x, event.y))
        start = self.preview_start
        self.preview_start = None
        self.preview_end = None
        if math.dist(start, end) < 0.1:
            self.status.set("Протяните мышью, чтобы задать отрезок стены.")
            self.draw()
            return
        try:
            thickness = float(self.wall_thickness.get())
            if not (0.03 <= thickness <= 2.0):
                raise ValueError("Толщина стены должна быть от 0.03 до 2 м")
            self.project.walls.append(Wall(*start, *end, thickness))
            self.invalidate()
        except ValueError as exc:
            messagebox.showerror("Некорректное значение", str(exc))

    def on_right_click(self, event):
        self.erase(*self.to_world(event.x, event.y))

    def erase(self, x, y):
        for idx in range(len(self.project.heaters) - 1, -1, -1):
            heater = self.project.heaters[idx]
            if math.hypot(x - heater.x, y - heater.y) < max(heater.radius, 0.2):
                self.project.heaters.pop(idx)
                self.invalidate()
                return
        wall, distance, t = self._nearest_wall(x, y)
        if wall is not None and distance < max(0.25, wall.thickness):
            length = math.hypot(wall.x2 - wall.x1, wall.y2 - wall.y1)
            for idx, window in enumerate(wall.windows):
                if abs(t - window.centre) * length <= window.length / 2:
                    wall.windows.pop(idx)
                    self.invalidate()
                    return
            self.project.walls.remove(wall)
            self.invalidate()
            return
        if self.project.street_seed and math.dist((x, y), self.project.street_seed) < 0.3:
            self.project.street_seed = None
            self.invalidate()

    def on_move(self, event):
        x, y = self.to_world(event.x, event.y)
        if self.solution is not None:
            raster = self.solution.raster
            j = min(max(int(x / raster.hx), 0), raster.materials.shape[1] - 1)
            i = min(max(int(y / raster.hy), 0), raster.materials.shape[0] - 1)
            label = ("воздух", "бетон", "окно")[raster.materials[i, j]]
            self.status.set(
                f"x = {x:.2f} м, y = {y:.2f} м   •   "
                f"{label}   •   T = {self.solution.temperature[i, j]:.2f} °C"
            )
        else:
            self.status.set(f"x = {x:.2f} м, y = {y:.2f} м   •   "
                            "Выберите инструмент и отредактируйте геометрию")

    def compute(self):
        try:
            self.read_settings()
        except ModelError as exc:
            messagebox.showerror("Ошибка параметров", str(exc))
            return
        self.status.set("Решение СЛАУ…")
        self.root.update_idletasks()
        self.root.after(10, self._compute_now)

    def _compute_now(self):
        try:
            import time
            start = time.perf_counter()
            self.solution = solve(self.project)
            elapsed = time.perf_counter() - start
            t = self.solution.temperature
            self.stats.set(
                f"Неизвестных: {self.solution.unknown_count:,}\n"
                f"Ненулевых в A: {self.solution.matrix.nnz:,}\n"
                f"Tмин = {t.min():.2f} °C\n"
                f"Tмакс = {t.max():.2f} °C\n"
                f"Невязка (∞): {self.solution.residual:.2e}\n"
                f"Время: {elapsed:.3f} с"
            )
            self.status.set("Решение получено. Наведите мышь для температуры в точке.")
            self.draw()
        except (ModelError, ValueError, RuntimeError) as exc:
            self.solution = None
            messagebox.showerror("Не удалось вычислить температуру", str(exc))
            self.status.set("Расчёт не выполнен")

    def draw(self):
        if not hasattr(self, "canvas"):
            return
        canvas = self.canvas
        canvas.delete("all")
        ox, oy, s = self._view()
        x2, y2 = self.to_canvas(self.project.width, self.project.height)
        canvas.create_rectangle(ox, oy, x2, y2, fill="#fafcfe", outline="#9da8b7")

        if self.solution is not None and self.show_field.get():
            temperature = self.solution.temperature
            lo, hi = float(temperature.min()), float(temperature.max())
            denominator = max(hi - lo, 1e-10)
            rgba = colormaps["coolwarm"]((temperature - lo) / denominator, bytes=True)
            image = Image.fromarray(rgba[:, :, :3], "RGB")
            image = image.resize(
                (max(1, int(self.project.width * s)),
                 max(1, int(self.project.height * s))), Image.Resampling.BILINEAR,
            )
            self.photo = ImageTk.PhotoImage(image)
            canvas.create_image(ox, oy, image=self.photo, anchor=tk.NW)
            canvas.create_rectangle(ox, oy, x2, y2, outline="#68788a")
            # Lightweight color scale at the bottom of the drawing canvas.
            n = 55
            scale_y = min(canvas.winfo_height() - 19, y2 + 19)
            scale_x = ox
            scale_w = min(240, self.project.width * s)
            cmap = colormaps["coolwarm"]
            for idx in range(n):
                canvas.create_rectangle(scale_x + idx * scale_w / n, scale_y,
                                        scale_x + (idx + 1) * scale_w / n + 1,
                                        scale_y + 9, fill=self._rgb(cmap(idx / (n - 1))),
                                        width=0)
            canvas.create_text(scale_x, scale_y + 15, text=f"{lo:.1f} °C",
                               anchor=tk.NW, font=("Arial", 9), fill="#344256")
            canvas.create_text(scale_x + scale_w, scale_y + 15, text=f"{hi:.1f} °C",
                               anchor=tk.NE, font=("Arial", 9), fill="#344256")

        # Coordinate grid (1 m intervals) on top of the temperature raster.
        for i in range(1, int(self.project.width)):
            x, _ = self.to_canvas(i, 0)
            canvas.create_line(x, oy, x, y2, fill="#93a0af", dash=(2, 5))
            canvas.create_text(x, y2 + 7, text=str(i), anchor=tk.N,
                               font=("Arial", 9), fill="#586577")
        for i in range(1, int(self.project.height)):
            _, y = self.to_canvas(0, i)
            canvas.create_line(ox, y, x2, y, fill="#93a0af", dash=(2, 5))
            canvas.create_text(ox - 8, y, text=str(i), anchor=tk.E,
                               font=("Arial", 9), fill="#586577")
        canvas.create_text(ox, oy - 13, text="м", anchor=tk.W,
                           font=("Arial", 9), fill="#586577")

        # Draw geometry on top of heatmap.
        for wall in self.project.walls:
            x1, y1 = self.to_canvas(wall.x1, wall.y1)
            x2w, y2w = self.to_canvas(wall.x2, wall.y2)
            px = max(2, round(wall.thickness * s))
            canvas.create_line(x1, y1, x2w, y2w, width=px,
                               fill="#505b6c", capstyle=tk.ROUND)
            for window in wall.windows:
                length_wall = math.hypot(wall.x2 - wall.x1, wall.y2 - wall.y1)
                if length_wall < 1e-8:
                    continue
                half = window.length / (2 * length_wall)
                start_t, end_t = max(0.0, window.centre - half), min(1.0, window.centre + half)
                p1 = self.to_canvas(wall.x1 + (wall.x2 - wall.x1) * start_t,
                                    wall.y1 + (wall.y2 - wall.y1) * start_t)
                p2 = self.to_canvas(wall.x1 + (wall.x2 - wall.x1) * end_t,
                                    wall.y1 + (wall.y2 - wall.y1) * end_t)
                canvas.create_line(*p1, *p2, width=px, fill="#4eb8e8", capstyle=tk.BUTT)
                canvas.create_line(*p1, *p2, width=max(1, px // 5), fill="#e5f6ff")

        for heater in self.project.heaters:
            x, y = self.to_canvas(heater.x, heater.y)
            r = heater.radius * s
            canvas.create_oval(x - r, y - r, x + r, y + r,
                               fill="#e35a3b", outline="#8e3020", width=2,
                               stipple="gray50")
            canvas.create_text(x, y, text=f"{heater.temperature:g}°",
                               fill="#1e1919", font=("Arial", 10, "bold"))

        if self.project.street_seed is not None:
            x, y = self.to_canvas(*self.project.street_seed)
            canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#1380c9", width=3)
            canvas.create_rectangle(x + 8, y - 18, x + 79, y + 1,
                                    fill="#eef8ff", outline="#5ea5d5")
            canvas.create_text(x + 13, y - 9, text="УЛИЦА", anchor=tk.W,
                               font=("Arial", 10, "bold"), fill="#176ca9")

        if self.preview_start is not None and self.preview_end is not None:
            x1, y1 = self.to_canvas(*self.preview_start)
            x2p, y2p = self.to_canvas(*self.preview_end)
            try:
                px = max(3, float(self.wall_thickness.get()) * s)
            except ValueError:
                px = 7
            canvas.create_line(x1, y1, x2p, y2p,
                               fill="#b37a33", width=px, dash=(4, 3))

    @staticmethod
    def _rgb(rgba):
        return "#%02x%02x%02x" % tuple(int(v * 255) for v in rgba[:3])

    def load_example(self):
        self.project = example_project()
        self.filename = None
        self.update_settings()
        self.invalidate()
        self.status.set("Пример загружен. Нажмите «Вычислить температуру».")

    def new_project(self):
        self.project = Project()
        self.filename = None
        self.update_settings()
        self.invalidate()
        self.status.set("Новый проект. Нарисуйте замкнутую комнату и отметьте улицу.")

    def update_settings(self):
        for var, value in (
            (self.outdoor_temperature, self.project.outdoor_temperature),
            (self.grid_step, self.project.step),
            (self.air_k, self.project.air_conductivity),
            (self.concrete_k, self.project.concrete_conductivity),
            (self.window_k, self.project.window_conductivity),
        ):
            var.set(str(value))

    def save_project(self):
        try:
            self.read_settings()
        except ModelError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("Проект комнаты", "*.json")],
            initialfile="room.json" if not self.filename else None,
        )
        if path:
            self.project.save(path)
            self.filename = path
            self.status.set(f"Сохранено: {path}")

    def open_project(self):
        path = filedialog.askopenfilename(filetypes=[("Проект комнаты", "*.json")])
        if not path:
            return
        try:
            self.project = Project.load(path)
            self.filename = path
            self.update_settings()
            self.invalidate()
            self.status.set(f"Открыто: {path}")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            messagebox.showerror("Не удалось открыть файл", str(exc))


def main():
    root = tk.Tk()
    ThermalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
