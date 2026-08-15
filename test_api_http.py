"""
Tests de los endpoints HTTP. Se ejecutan con:

    python -m unittest test_api_http

Se saltan solos si FastAPI no esta instalado, porque es una dependencia
opcional. No llaman a Gemini: se sustituye la funcion que genera la
respuesta.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    HAY_FASTAPI = True
except ImportError:
    HAY_FASTAPI = False


@unittest.skipUnless(HAY_FASTAPI, "FastAPI no instalado (dependencia opcional)")
class TestEndpoints(unittest.TestCase):
    def setUp(self):
        import api
        import colecciones
        from historial import HistorialSQLite

        self.api = api
        self.colecciones = colecciones
        self.tmp = Path(tempfile.mkdtemp())

        # Se redirige todo el estado a un directorio temporal.
        self._data_dir_original = colecciones.DATA_DIR
        colecciones.DATA_DIR = self.tmp
        api._cache = colecciones.CacheBuscadores(data_dir=self.tmp)
        api._almacen = HistorialSQLite(self.tmp / "historial.db")

        # Nada de llamadas reales al modelo.
        self._responder_original = api.conversacion.responder
        api.conversacion.responder = lambda *a, **k: (
            "respuesta de prueba",
            [{"source": "d.txt", "page": 1, "seccion": None}],
        )

        self.cliente = TestClient(api.app)

    def tearDown(self):
        self.colecciones.DATA_DIR = self._data_dir_original
        self.api.conversacion.responder = self._responder_original
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _subir(self, coleccion, nombre="doc.txt", contenido=b"contenido de prueba"):
        return self.cliente.post(
            f"/colecciones/{coleccion}/documentos",
            files={"archivo": (nombre, contenido, "text/plain")},
        )

    def test_salud_sin_auth(self):
        respuesta = self.cliente.get("/salud")
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.json()["estado"], "ok")

    def test_listar_colecciones_vacio(self):
        respuesta = self.cliente.get("/colecciones")
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.json()["colecciones"], [])

    def test_subir_y_listar(self):
        self.assertEqual(self._subir("catalogo").status_code, 200)
        datos = self.cliente.get("/colecciones").json()
        self.assertEqual(datos["colecciones"][0]["nombre"], "catalogo")
        self.assertEqual(datos["colecciones"][0]["documentos"], ["doc.txt"])

    def test_subir_rechaza_nombre_de_coleccion_invalido(self):
        respuesta = self.cliente.post(
            "/colecciones/..%2F..%2Fetc/documentos",
            files={"archivo": ("d.txt", b"x", "text/plain")},
        )
        self.assertIn(respuesta.status_code, (400, 404))

    def test_subir_rechaza_formato_no_soportado(self):
        respuesta = self._subir("c", nombre="hoja.xlsx")
        self.assertEqual(respuesta.status_code, 400)
        self.assertIn("Formato no soportado", respuesta.json()["detail"])

    def test_subir_rechaza_archivo_vacio(self):
        self.assertEqual(self._subir("c", contenido=b"").status_code, 400)

    def test_el_nombre_del_archivo_no_escapa_de_la_carpeta(self):
        """Aunque el cliente mande una ruta, solo se usa el nombre base."""
        respuesta = self._subir("c", nombre="../../fuera.txt")
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(respuesta.json()["archivo"], "fuera.txt")
        self.assertFalse((self.tmp.parent / "fuera.txt").exists())
        docs, _ = self.colecciones.rutas("c", data_dir=self.tmp)
        self.assertTrue((docs / "fuera.txt").exists())

    def test_chat_en_coleccion_inexistente(self):
        respuesta = self.cliente.post(
            "/chat",
            json={"mensaje": "hola", "session_id": "s1", "coleccion": "fantasma"},
        )
        self.assertEqual(respuesta.status_code, 404)

    def test_chat_responde_y_guarda_historial(self):
        self._subir("catalogo")
        self.colecciones.crear("catalogo", data_dir=self.tmp)
        respuesta = self.cliente.post(
            "/chat",
            json={"mensaje": "hola", "session_id": "s1", "coleccion": "catalogo"},
        )
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        datos = respuesta.json()
        self.assertEqual(datos["respuesta"], "respuesta de prueba")
        self.assertEqual(datos["fuentes"][0]["documento"], "d.txt")
        self.assertEqual(len(self.api._almacen.cargar("s1")), 2)

    def test_las_sesiones_estan_separadas(self):
        self._subir("catalogo")
        for sesion in ("ana", "luis"):
            self.cliente.post(
                "/chat",
                json={"mensaje": "hola", "session_id": sesion, "coleccion": "catalogo"},
            )
        self.assertEqual(len(self.api._almacen.cargar("ana")), 2)
        self.assertEqual(len(self.api._almacen.cargar("luis")), 2)

    def test_borrar_sesion(self):
        self._subir("catalogo")
        self.cliente.post(
            "/chat",
            json={"mensaje": "hola", "session_id": "s1", "coleccion": "catalogo"},
        )
        self.assertEqual(self.cliente.delete("/sesiones/s1").status_code, 200)
        self.assertEqual(self.api._almacen.cargar("s1"), [])

    def test_chat_valida_la_entrada(self):
        for cuerpo in (
            {"mensaje": "", "session_id": "s", "coleccion": "c"},
            {"mensaje": "hola", "session_id": "", "coleccion": "c"},
            {"mensaje": "hola", "session_id": "s"},
        ):
            self.assertEqual(self.cliente.post("/chat", json=cuerpo).status_code, 422)

    def test_api_key_protege_los_endpoints(self):
        self.api.API_KEY = "secreta"
        try:
            self.assertEqual(self.cliente.get("/colecciones").status_code, 401)
            con_key = self.cliente.get(
                "/colecciones", headers={"X-API-Key": "secreta"}
            )
            self.assertEqual(con_key.status_code, 200)
            self.assertEqual(self.cliente.get("/salud").status_code, 200)
        finally:
            self.api.API_KEY = ""


if __name__ == "__main__":
    unittest.main()
