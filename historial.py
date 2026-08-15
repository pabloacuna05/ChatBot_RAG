"""
Almacenamiento del historial de conversacion.

La CLI tiene una sola conversacion y le basta un JSON; la API atiende a
muchos usuarios a la vez y necesita separarlas por session_id. Ambas usan
la misma interfaz, asi que cambiar SQLite por Postgres o Redis mas adelante
solo implica escribir otra subclase, sin tocar ni la CLI ni los endpoints.
"""

import json
import sqlite3
import threading
from pathlib import Path


class AlmacenHistorial:
    """Interfaz comun. Un historial es una lista de {"role", "text"}."""

    def cargar(self, session_id):
        raise NotImplementedError

    def guardar(self, session_id, historial):
        raise NotImplementedError

    def borrar(self, session_id):
        raise NotImplementedError


class HistorialJSON(AlmacenHistorial):
    """Un unico archivo JSON. Lo que usa la CLI: simple, legible a mano y
    facil de borrar. Ignora el session_id porque solo hay una conversacion."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)

    def cargar(self, session_id=None):
        if not self.ruta.exists():
            return []
        try:
            with open(self.ruta, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print("Aviso: no se pudo leer el historial guardado, empiezo de cero.")
            return []

    def guardar(self, session_id, historial):
        try:
            with open(self.ruta, "w", encoding="utf-8") as f:
                json.dump(historial, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"Aviso: no se pudo guardar el historial ({e}).")

    def borrar(self, session_id=None):
        if self.ruta.exists():
            try:
                self.ruta.unlink()
            except OSError as e:
                print(f"Aviso: no se pudo borrar el archivo de historial ({e}).")


class HistorialSQLite(AlmacenHistorial):
    """Historial por sesion en SQLite. Lo que usa la API.

    Cada turno es una fila, no un blob JSON por sesion: asi dos peticiones
    concurrentes de la misma sesion no se pisan la una a la otra reescribiendo
    el historial entero."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._crear_esquema()

    def _conexion(self):
        # SQLite no permite compartir una conexion entre hilos, y el servidor
        # atiende peticiones en varios; una por hilo lo resuelve.
        if not hasattr(self._local, "conexion"):
            self._local.conexion = sqlite3.connect(str(self.ruta))
            self._local.conexion.execute("PRAGMA journal_mode=WAL")
        return self._local.conexion

    def _crear_esquema(self):
        with self._conexion() as conexion:
            conexion.execute(
                """
                CREATE TABLE IF NOT EXISTS turnos (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    rol        TEXT NOT NULL,
                    texto      TEXT NOT NULL,
                    creado_en  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conexion.execute(
                "CREATE INDEX IF NOT EXISTS idx_turnos_sesion ON turnos(session_id, id)"
            )

    def cargar(self, session_id, limite=None):
        with self._conexion() as conexion:
            filas = conexion.execute(
                "SELECT rol, texto FROM turnos WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        historial = [{"role": rol, "text": texto} for rol, texto in filas]
        return historial[-limite:] if limite else historial

    def guardar(self, session_id, historial):
        """Reemplaza el historial completo de la sesion. Se usa poco: lo
        habitual es añadir turnos con añadir_turno."""
        with self._conexion() as conexion:
            conexion.execute("DELETE FROM turnos WHERE session_id = ?", (session_id,))
            conexion.executemany(
                "INSERT INTO turnos (session_id, rol, texto) VALUES (?, ?, ?)",
                [(session_id, t["role"], t["text"]) for t in historial],
            )

    def añadir_turno(self, session_id, rol, texto):
        with self._conexion() as conexion:
            conexion.execute(
                "INSERT INTO turnos (session_id, rol, texto) VALUES (?, ?, ?)",
                (session_id, rol, texto),
            )

    def borrar(self, session_id):
        with self._conexion() as conexion:
            conexion.execute("DELETE FROM turnos WHERE session_id = ?", (session_id,))

    def sesiones(self):
        with self._conexion() as conexion:
            filas = conexion.execute(
                "SELECT session_id, COUNT(*) FROM turnos GROUP BY session_id"
            ).fetchall()
        return [{"session_id": s, "turnos": n} for s, n in filas]
