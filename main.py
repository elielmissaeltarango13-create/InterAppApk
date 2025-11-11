# -*- coding: utf-8 -*-
import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.clock import Clock

from kivy_garden.matplotlib import FigureCanvasKivyAgg
import matplotlib
matplotlib.use("module://kivy_garden.matplotlib.backend_kivy")
import matplotlib.pyplot as plt

KV = r"""
<Root>:
    orientation: "vertical"
    canvas.before:
        Color: rgba: (.12,.14,.16,1)
        Rectangle: pos: self.pos; size: self.size

    BoxLayout:
        size_hint_y: None
        height: "56dp"
        padding: "10dp"
        Label:
            text: "Interpolación Lineal y Cuadrática"
            color: 1, 1, 1, 1
            bold: True
            font_size: "20sp"

    BoxLayout:
        size_hint_y: None
        height: "56dp"
        padding: "8dp"
        spacing: "8dp"
        Label:
            text: "Número de puntos (filas):"
            size_hint_x: None
            width: self.texture_size[0] + dp(12)
        TextInput:
            id: rows_input
            text: str(root.n_rows)
            input_filter: "int"
            hint_text: ">= 2"
            multiline: False
            size_hint_x: None
            width: "80dp"
            on_text_validate: root.reload_table()
        Button:
            text: "Crear / Reiniciar tabla"
            on_release: root.reload_table()

    BoxLayout:
        size_hint_y: None
        height: "44dp"
        padding: "8dp"
        spacing: "14dp"
        ToggleButton:
            text: "Lineal (2 puntos)"
            state: "down" if root.use_lineal else "normal"
            on_state: root.use_lineal = (self.state=="down")
        ToggleButton:
            text: "Cuadrática (3 puntos)"
            state: "down" if root.use_cuad else "normal"
            on_state: root.use_cuad = (self.state=="down")

    BoxLayout:
        padding: "8dp"
        spacing: "8dp"

        BoxLayout:
            orientation: "vertical"
            ScrollView:
                id: scroller
                do_scroll_x: False
                GridLayout:
                    id: table
                    cols: 3
                    size_hint_y: None
                    height: self.minimum_height
                    row_default_height: "36dp"
                    padding: "6dp"
                    spacing: "6dp"

        BoxLayout:
            orientation: "vertical"
            size_hint_x: None
            width: "220dp"
            spacing: "8dp"
            Button:
                text: "Calcular"
                on_release: root.do_calculate()
            Button:
                text: "Generar gráfica"
                on_release: root.do_plot()

    BoxLayout:
        orientation: "vertical"
        size_hint_y: None
        height: "180dp"
        padding: "8dp"
        Label:
            text: "Resultados"
            size_hint_y: None
            height: "24dp"
            bold: True
        TextInput:
            id: results
            readonly: True

    BoxLayout:
        size_hint_y: None
        height: "28dp"
        padding: "8dp"
        Label:
            text: "Integrantes: Samantha | Adriel | Elian | Eliel"
            font_size: "12sp"
"""

class Root(BoxLayout):
    n_rows = NumericProperty(5)
    use_lineal = BooleanProperty(True)
    use_cuad = BooleanProperty(True)
    x_inputs = ListProperty([])
    y_inputs = ListProperty([])

    def on_kv_post(self, *args):
        self.ids.results.text = "Tabla creada. Ingresa valores numéricos para x e y.\n"
        self.reload_table()

    def popup(self, title, msg):
        Popup(title=title, content=Label(text=msg), size_hint=(0.85,0.35)).open()

    def reload_table(self):
        try:
            n = int(self.ids.rows_input.text)
        except Exception:
            n = 5
        if n < 2:
            n = 2
        self.n_rows = n

        table = self.ids.table
        table.clear_widgets()
        self.x_inputs = []
        self.y_inputs = []

        for txt in ("#", "x", "y"):
            table.add_widget(Label(text=f"[b]{txt}[/b]", markup=True, size_hint_y=None, height="28dp"))

        for i in range(n):
            table.add_widget(Label(text=str(i+1), size_hint_y=None, height="36dp"))
            ex = TextInput(text="", multiline=False, halign="center", size_hint_y=None, height="36dp")
            ey = TextInput(text="", multiline=False, halign="center", size_hint_y=None, height="36dp")
            table.add_widget(ex); table.add_widget(ey)
            self.x_inputs.append(ex); self.y_inputs.append(ey)

        table.add_widget(Label(text="", size_hint_y=None, height="1dp"))
        table.add_widget(Label(text="[i]Consejo: deja vacías las filas que no uses.[/i]", markup=True, size_hint_y=None, height="28dp"))
        table.add_widget(Label(text="", size_hint_y=None, height="1dp"))
        Clock.schedule_once(lambda dt: setattr(self.ids.scroller, "scroll_y", 1))

    @staticmethod
    def interp_lineal_2pts(x2, y2):
        x2 = np.asarray(x2, dtype=float); y2 = np.asarray(y2, dtype=float)
        if len(x2) != 2 or len(y2) != 2:
            raise ValueError("Se requieren exactamente 2 puntos para interpolación lineal.")
        if x2[1] == x2[0]:
            raise ValueError("Para interpolación lineal, x1 y x2 no deben ser iguales.")
        m = (y2[1] - y2[0]) / (x2[1] - x2[0]); b = y2[0] - m * x2[0]
        return m, b

    @staticmethod
    def interp_cuadratica_3pts(x3, y3):
        x3 = np.asarray(x3, dtype=float); y3 = np.asarray(y3, dtype=float)
        if len(x3) != 3 or len(y3) != 3:
            raise ValueError("Se requieren exactamente 3 puntos para interpolación cuadrática.")
        A = np.vstack([x3**2, x3, np.ones(3)]).T
        a, b, c = np.linalg.solve(A, y3)
        return a, b, c

    def leer_datos(self):
        xs, ys = [], []
        for i, (ex, ey) in enumerate(zip(self.x_inputs, self.y_inputs), start=1):
            sx = ex.text.strip(); sy = ey.text.strip()
            if sx == "" and sy == "": continue
            if sx == "" or sy == "": raise ValueError(f"Fila {i}: x o y vacío.")
            try:
                xv = float(sx.replace(",", ".")); yv = float(sy.replace(",", "."))
            except Exception:
                raise ValueError(f"Fila {i}: valor no numérico.")
            xs.append(xv); ys.append(yv)
        if len(xs) == 0: raise ValueError("No hay datos. Ingresa al menos 2 puntos.")
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def do_calculate(self):
        self.ids.results.text = ""
        try:
            X, Y = self.leer_datos()
        except Exception as e:
            self.popup("Error de entrada", str(e)); return
        if not self.use_lineal and not self.use_cuad:
            self.popup("Selección requerida", "Activa Lineal y/o Cuadrática para calcular."); return
        r = []
        if self.use_lineal:
            if len(X) < 2: r.append("⚠️ Lineal: al menos 2 puntos.")
            else:
                try:
                    m, b = self.interp_lineal_2pts(X[:2], Y[:2])
                    r.append("— Interpolación lineal (primeros 2 puntos):")
                    r.append(f"   y = {m:.6g}·x + {b:.6g}")
                except Exception as e: r.append(f"   Error lineal: {e}")
        if self.use_cuad:
            if len(X) < 3: r.append("⚠️ Cuadrática: al menos 3 puntos.")
            else:
                try:
                    a, b, c = self.interp_cuadratica_3pts(X[:3], Y[:3])
                    r.append("— Interpolación cuadrática (primeros 3 puntos):")
                    r.append(f"   y = {a:.6g}·x² + {b:.6g}·x + {c:.6g}")
                except Exception as e: r.append(f"   Error cuadrática: {e}")
        if len(X) > 3: r.append("ℹ️ Nota: si ingresas >3 puntos, se usan solo los primeros.")
        self.ids.results.text = "\n".join(r) + "\n"

    def do_plot(self):
        try:
            X, Y = self.leer_datos()
        except Exception as e:
            self.popup("Error de entrada", str(e)); return
        if not self.use_lineal and not self.use_cuad:
            self.popup("Selección requerida", "Activa Lineal y/o Cuadrática para graficar."); return
        xmin, xmax = float(np.min(X)), float(np.max(X))
        if xmin == xmax:
            xmin -= 1.0; xmax += 1.0
        x_eval = np.linspace
