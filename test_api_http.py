"""
Tests de los endpoints HTTP, incluido el hardening del punto 18.

    python -m unittest test_api_http

Se saltan solos si FastAPI no esta instalado, porque es una dependencia
opcional. No llaman a Gemini: se sustituye la funcion que genera respuesta.
"""

import shutil
import tempfile
import time
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient

    HAY_FASTAPI = True
except ImportError:
    HAY_FASTAPI = False


@unittest.skipUnless(HAY_FASTAPI, "FastAPI no instalado (dependencia opcional)")
class BaseAPI(unittest.TestCase):
    def setUp(self):
        import api
        import colecciones
        import seguridad
        from historial import AlmacenSesiones, HistorialSQLite

        self.api = api
        self.colecciones = colecciones
        self.seguridad = seguridad
        self.tmp = Path(tempfile.mkdtemp())

        # Todo el estado, a un directorio temporal.
        self._data_dir_original = colecciones.DATA_DIR
        colecciones.DATA_DIR = self.tmp
        api._cache = colecciones.CacheBuscadores(data_dir=self.tmp)
        api._almacen = HistorialSQLite(self.tmp / "h.db")
        api._sesiones = AlmacenSesiones(self.tmp / "h.db")
        api._limitador_sesion = seguridad.LimitadorPeticiones([(1000, 60)])
        api._limitador_ip = seguridad.LimitadorPeticiones([(1000, 60)])
        api.COOKIES_SEGURAS = False  # el TestClient habla por http

        # Nada de llamadas reales al modelo.
        self._responder_original = api.conversacion.responder
        api.conversacion.responder = lambda *a, **k: (
            "respuesta de prueba",
            [{"source": "d.txt", "page": 1, "seccion": None}],
        )

        colecciones.crear("catalogo", data_dir=self.tmp)
        self.cliente = TestClient(api.app)

    def tearDown(self):
        self.colecciones.DATA_DIR = self._data_dir_original
        self.api.conversacion.responder = self._responder_original
        shutil.rmtree(self.tmp, ignore_errors=True)

    def abrir_sesion(self, coleccion="catalogo"):
        return self.cliente.post("/sesiones", json={"coleccion": coleccion})


class TestSesiones(BaseAPI):
    def test_abrir_sesion_devuelve_cookie_firmada(self):
        respuesta = self.abrir_sesion()
        self.assertEqual(respuesta.status_code, 200)
        cookie = respuesta.cookies.get(self.api.NOMBRE_COOKIE)
        self.assertIsNotNone(cookie)
        self.assertIn(".", cookie, "la cookie debe llevar firma")

    def test_la_cookie_es_httponly(self):
        respuesta = self.abrir_sesion()
        cabecera = respuesta.headers.get("set-cookie", "")
        self.assertIn("HttpOnly", cabecera)
        self.assertIn("SameSite=lax", cabecera.replace("samesite", "SameSite"))

    def test_sesion_en_coleccion_inexistente(self):
        self.assertEqual(self.abrir_sesion("fantasma").status_code, 404)

    def test_chat_sin_sesion_es_401(self):
        respuesta = self.cliente.post("/chat", json={"mensaje": "hola"})
        self.assertEqual(respuesta.status_code, 401)

    def test_no_se_puede_inventar_una_sesion(self):
        """El ataque directo: poner un session_id cualquiera en la cookie."""
        self.cliente.cookies.set(self.api.NOMBRE_COOKIE, "a" * 32)
        respuesta = self.cliente.post("/chat", json={"mensaje": "hola"})
        self.assertEqual(respuesta.status_code, 401)

    def test_no_se_puede_robar_la_sesion_de_otro(self):
        """Con el id de otra sesion pero sin su firma valida, no entra."""
        self.abrir_sesion()
        cookie = self.cliente.cookies.get(self.api.NOMBRE_COOKIE)
        session_id = cookie.split(".")[0]

        self.cliente.cookies.clear()
        self.cliente.cookies.set(self.api.NOMBRE_COOKIE, f"{session_id}.firmafalsa")
        self.assertEqual(
            self.cliente.post("/chat", json={"mensaje": "hola"}).status_code, 401
        )

    def test_chat_funciona_con_sesion_valida(self):
        self.abrir_sesion()
        respuesta = self.cliente.post("/chat", json={"mensaje": "hola"})
        self.assertEqual(respuesta.status_code, 200, respuesta.text)
        self.assertEqual(respuesta.json()["respuesta"], "respuesta de prueba")
        self.assertEqual(respuesta.json()["coleccion"], "catalogo")

    def test_la_sesion_queda_ligada_a_su_coleccion(self):
        """El cuerpo de /chat no acepta 'coleccion': sale de la sesion."""
        self.colecciones.crear("otra", data_dir=self.tmp)
        self.abrir_sesion("catalogo")
        respuesta = self.cliente.post(
            "/chat", json={"mensaje": "hola", "coleccion": "otra"}
        )
        self.assertEqual(respuesta.status_code, 200)
        self.assertEqual(
            respuesta.json()["coleccion"], "catalogo", "no debe cambiar de corpus"
        )

    def test_cerrar_sesion_borra_el_historial(self):
        self.abrir_sesion()
        self.cliente.post("/chat", json={"mensaje": "hola"})
        cookie = self.cliente.cookies.get(self.api.NOMBRE_COOKIE)
        session_id = cookie.split(".")[0]

        self.assertEqual(self.cliente.delete("/sesiones/actual").status_code, 200)
        self.assertEqual(self.api._almacen.cargar(session_id), [])
        self.assertIsNone(self.api._sesiones.obtener(session_id))

    def test_sesion_caducada_por_inactividad(self):
        self.abrir_sesion()
        cookie = self.cliente.cookies.get(self.api.NOMBRE_COOKIE)
        session_id = cookie.split(".")[0]
        # Se envejece la sesion a mano.
        self.api._sesiones.registrar_uso(
            session_id, ahora=time.time() - self.api.SEGUNDOS_EXPIRACION_SESION - 10
        )
        self.assertEqual(
            self.cliente.post("/chat", json={"mensaje": "hola"}).status_code, 401
        )

    def test_tope_de_mensajes_por_sesion(self):
        self.abrir_sesion()
        cookie = self.cliente.cookies.get(self.api.NOMBRE_COOKIE)
        session_id = cookie.split(".")[0]
        for _ in range(self.api.MAX_MENSAJES_POR_SESION):
            self.api._sesiones.registrar_uso(session_id)
        respuesta = self.cliente.post("/chat", json={"mensaje": "hola"})
        self.assertEqual(respuesta.status_code, 429)


class TestLimitesDeEntrada(BaseAPI):
    def test_mensaje_demasiado_largo(self):
        self.abrir_sesion()
        largo = "x" * (self.api.MAX_CHARS_PREGUNTA + 1)
        respuesta = self.cliente.post("/chat", json={"mensaje": largo})
        self.assertEqual(respuesta.status_code, 422, "pydantic corta antes")

    def test_mensaje_vacio(self):
        self.abrir_sesion()
        self.assertEqual(
            self.cliente.post("/chat", json={"mensaje": ""}).status_code, 422
        )

    def test_mensaje_solo_espacios(self):
        self.abrir_sesion()
        self.assertEqual(
            self.cliente.post("/chat", json={"mensaje": "   "}).status_code, 400
        )

    def test_rate_limit_devuelve_retry_after(self):
        self.api._limitador_sesion = self.seguridad.LimitadorPeticiones([(2, 60)])
        self.abrir_sesion()
        self.cliente.post("/chat", json={"mensaje": "una"})
        self.cliente.post("/chat", json={"mensaje": "dos"})
        respuesta = self.cliente.post("/chat", json={"mensaje": "tres"})
        self.assertEqual(respuesta.status_code, 429)
        self.assertIn("retry-after", {k.lower() for k in respuesta.headers})

    def test_rate_limit_por_ip_en_abrir_sesion(self):
        self.api._limitador_ip = self.seguridad.LimitadorPeticiones([(2, 60)])
        self.abrir_sesion()
        self.abrir_sesion()
        self.assertEqual(self.abrir_sesion().status_code, 429)


class TestSuperficieDeError(BaseAPI):
    def test_los_errores_llevan_id_de_correlacion(self):
        respuesta = self.cliente.post("/chat", json={"mensaje": "hola"})
        cuerpo = respuesta.json()
        self.assertIn("correlacion", cuerpo["detail"])
        self.assertIn("error", cuerpo["detail"])

    def test_un_fallo_interno_no_filtra_el_detalle(self):
        """Un error de Gemini puede llevar dentro datos del servidor."""

        def explotar(*a, **k):
            raise RuntimeError("clave secreta AIzaSyXXXX y ruta /home/user/app")

        self.abrir_sesion()
        cookies = dict(self.cliente.cookies)
        self.api.conversacion.responder = explotar

        # raise_server_exceptions=False es del constructor, no de la
        # peticion: sin esto el TestClient relanzaria la excepcion en vez de
        # dejar que responda el handler global, que es justo lo que se prueba.
        cliente = TestClient(self.api.app, raise_server_exceptions=False)
        cliente.cookies.update(cookies)
        respuesta = cliente.post("/chat", json={"mensaje": "hola"})
        self.assertEqual(respuesta.status_code, 500)
        texto = respuesta.text
        self.assertNotIn("AIzaSy", texto)
        self.assertNotIn("/home/user", texto)
        self.assertNotIn("Traceback", texto)
        self.assertIn("correlacion", texto)

    def test_la_api_key_nunca_aparece_en_las_respuestas(self):
        self.abrir_sesion()
        for respuesta in (
            self.cliente.post("/chat", json={"mensaje": "hola"}),
            self.cliente.get("/colecciones"),
            self.cliente.get("/salud"),
        ):
            self.assertNotIn("GEMINI_API_KEY", respuesta.text)
            self.assertNotIn("AIzaSy", respuesta.text)


class TestCabecerasYSalud(BaseAPI):
    def test_cabeceras_de_seguridad(self):
        respuesta = self.cliente.get("/salud")
        self.assertEqual(respuesta.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(respuesta.headers["X-Frame-Options"], "DENY")
        self.assertEqual(respuesta.headers["Referrer-Policy"], "no-referrer")
        self.assertIn("frame-ancestors 'none'", respuesta.headers["Content-Security-Policy"])

    def test_salud_no_expone_nada_interno(self):
        datos = self.cliente.get("/salud").json()
        self.assertEqual(datos, {"estado": "ok"})

    def test_cors_nunca_es_comodin(self):
        self.assertNotIn("*", self.api.ORIGENES_CORS)

    def test_la_documentacion_no_queda_bloqueada_por_su_propia_csp(self):
        """Swagger carga su JS de un CDN: con la CSP estricta de la API se
        veria en blanco pero devolviendo 200, un fallo mudo."""
        respuesta = self.cliente.get("/docs")
        self.assertEqual(respuesta.status_code, 200)
        csp = respuesta.headers["Content-Security-Policy"]
        self.assertIn("cdn.jsdelivr.net", csp)
        self.assertIn("frame-ancestors 'none'", csp, "sigue sin poder embeberse")

    def test_los_endpoints_de_datos_mantienen_la_csp_estricta(self):
        """La excepcion es solo para la documentacion."""
        for ruta in ("/salud", "/colecciones"):
            csp = self.cliente.get(ruta).headers["Content-Security-Policy"]
            self.assertEqual(csp, self.api.CSP_API, f"{ruta} no deberia relajarla")


class TestSubidaDeDocumentos(BaseAPI):
    def _subir(self, coleccion, nombre, contenido, tipo="application/octet-stream"):
        return self.cliente.post(
            f"/colecciones/{coleccion}/documentos",
            files={"archivo": (nombre, contenido, tipo)},
        )

    def test_subida_valida(self):
        respuesta = self._subir("catalogo", "doc.txt", b"contenido de prueba")
        self.assertEqual(respuesta.status_code, 200, respuesta.text)

    def test_el_nombre_del_usuario_no_se_conserva(self):
        respuesta = self._subir("catalogo", "mi documento.txt", b"contenido")
        guardado = respuesta.json()["documento"]
        self.assertNotIn("mi documento", guardado)
        self.assertTrue(guardado.endswith(".txt"))

    def test_path_traversal_en_el_nombre_del_archivo(self):
        respuesta = self._subir("catalogo", "../../fuera.txt", b"contenido")
        self.assertEqual(respuesta.status_code, 200)
        self.assertNotIn("..", respuesta.json()["documento"])
        self.assertFalse((self.tmp.parent / "fuera.txt").exists())

    def test_path_traversal_en_el_nombre_de_la_coleccion(self):
        respuesta = self._subir("..%2F..%2Fetc", "d.txt", b"x")
        self.assertIn(respuesta.status_code, (400, 404))

    def test_ejecutable_disfrazado_de_pdf(self):
        respuesta = self._subir("catalogo", "malo.pdf", b"MZ\x90\x00ejecutable")
        self.assertEqual(respuesta.status_code, 400)
        self.assertIn("rechazado", respuesta.json()["detail"]["error"].lower())

    def test_content_type_mentido_no_sirve(self):
        """El Content-Type lo elige quien sube: se ignora."""
        respuesta = self._subir("catalogo", "malo.pdf", b"no soy pdf", "application/pdf")
        self.assertEqual(respuesta.status_code, 400)

    def test_formato_no_admitido(self):
        self.assertEqual(self._subir("catalogo", "a.exe", b"MZ").status_code, 400)

    def test_archivo_vacio(self):
        self.assertEqual(self._subir("catalogo", "a.txt", b"").status_code, 400)

    def test_archivo_demasiado_grande(self):
        original = self.api.MAX_BYTES_SUBIDA
        self.api.MAX_BYTES_SUBIDA = 10
        try:
            respuesta = self._subir("catalogo", "a.txt", b"x" * 100)
            self.assertEqual(respuesta.status_code, 413)
        finally:
            self.api.MAX_BYTES_SUBIDA = original


class TestAdmin(BaseAPI):
    def test_la_api_key_protege_administracion(self):
        self.api.ADMIN_API_KEY = "secreta"
        try:
            self.assertEqual(self.cliente.get("/colecciones").status_code, 401)
            con_key = self.cliente.get("/colecciones", headers={"X-API-Key": "secreta"})
            self.assertEqual(con_key.status_code, 200)
        finally:
            self.api.ADMIN_API_KEY = ""

    def test_una_key_incorrecta_no_pasa(self):
        self.api.ADMIN_API_KEY = "secreta"
        try:
            respuesta = self.cliente.get(
                "/colecciones", headers={"X-API-Key": "secretb"}
            )
            self.assertEqual(respuesta.status_code, 401)
        finally:
            self.api.ADMIN_API_KEY = ""

    def test_el_chat_no_necesita_api_key_de_admin(self):
        """Los usuarios finales se autentican por sesion, no con la key."""
        self.api.ADMIN_API_KEY = "secreta"
        try:
            self.abrir_sesion()
            respuesta = self.cliente.post("/chat", json={"mensaje": "hola"})
            self.assertEqual(respuesta.status_code, 200)
        finally:
            self.api.ADMIN_API_KEY = ""


if __name__ == "__main__":
    unittest.main()
