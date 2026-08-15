"""
Tests de la generacion paralela de embeddings. Se ejecutan con:

    python -m unittest test_embeddings

La garantia critica: el vector i tiene que corresponder al texto i. Con
lotes en paralelo que terminan en orden arbitrario, un fallo aqui asignaria
a cada fragmento el vector de otro sin dar ningun error.
"""

import random
import threading
import time
import unittest

import rag


class _APIErrorFalso(Exception):
    """Imita google.genai.errors.APIError: lo unico que mira el codigo es
    el atributo .code."""

    def __init__(self, code):
        super().__init__(f"error {code}")
        self.code = code


class _Embedding:
    def __init__(self, values):
        self.values = values


class _Resultado:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _Modelos:
    """Devuelve un vector que codifica el texto recibido, para poder
    comprobar despues que cada vector acabo en su sitio. Con retardo
    aleatorio, para que los lotes terminen desordenados a proposito."""

    def __init__(self, fallos_429=0, retardo=True):
        self.fallos_pendientes = fallos_429
        self.retardo = retardo
        self.lock = threading.Lock()
        self.llamadas = 0
        self.concurrencia_maxima = 0
        self._activos = 0

    def embed_content(self, model, contents, config):
        with self.lock:
            self.llamadas += 1
            self._activos += 1
            self.concurrencia_maxima = max(self.concurrencia_maxima, self._activos)
            if self.fallos_pendientes > 0:
                self.fallos_pendientes -= 1
                self._activos -= 1
                raise _APIErrorFalso(429)
        try:
            if self.retardo:
                time.sleep(random.uniform(0, 0.02))
            return _Resultado([_Embedding([float(len(t)), float(hash(t) % 1000)]) for t in contents])
        finally:
            with self.lock:
                self._activos -= 1


class ClienteFalso:
    def __init__(self, fallos_429=0, retardo=True):
        self.models = _Modelos(fallos_429, retardo)


def _vector_esperado(texto):
    return [float(len(texto)), float(hash(texto) % 1000)]


class TestOrdenDeLosEmbeddings(unittest.TestCase):
    def setUp(self):
        self._api_error_original = rag.APIError
        rag.APIError = _APIErrorFalso

    def tearDown(self):
        rag.APIError = self._api_error_original

    def test_el_orden_se_conserva_con_muchos_lotes(self):
        """Con lotes que acaban desordenados, cada vector debe seguir en su
        posicion original."""
        textos = [f"texto numero {i} con longitud variable {'x' * (i % 17)}" for i in range(95)]
        cliente = ClienteFalso()
        vectores = rag._embed_textos(cliente, textos, "RETRIEVAL_DOCUMENT")

        self.assertEqual(len(vectores), len(textos))
        for texto, vector in zip(textos, vectores):
            self.assertEqual(vector, _vector_esperado(texto))

    def test_usa_varios_hilos(self):
        textos = [f"t{i}" for i in range(80)]
        cliente = ClienteFalso()
        rag._embed_textos(cliente, textos, "RETRIEVAL_DOCUMENT")
        self.assertGreater(
            cliente.models.concurrencia_maxima, 1, "deberia paralelizar"
        )

    def test_respeta_el_limite_de_workers(self):
        textos = [f"t{i}" for i in range(200)]
        cliente = ClienteFalso()
        rag._embed_textos(cliente, textos, "RETRIEVAL_DOCUMENT")
        self.assertLessEqual(
            cliente.models.concurrencia_maxima, rag.WORKERS_EMBEDDING
        )

    def test_un_solo_lote_no_paraleliza(self):
        cliente = ClienteFalso()
        vectores = rag._embed_textos(cliente, ["uno", "dos"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(cliente.models.llamadas, 1)
        self.assertEqual(vectores[0], _vector_esperado("uno"))

    def test_lista_vacia(self):
        cliente = ClienteFalso()
        self.assertEqual(rag._embed_textos(cliente, [], "RETRIEVAL_DOCUMENT"), [])

    def test_reintenta_tras_429_y_mantiene_el_orden(self):
        textos = [f"texto {i}" for i in range(60)]
        cliente = ClienteFalso(fallos_429=3)
        vectores = rag._embed_textos(cliente, textos, "RETRIEVAL_DOCUMENT")
        for texto, vector in zip(textos, vectores):
            self.assertEqual(vector, _vector_esperado(texto))

    def test_un_error_no_429_se_propaga(self):
        """No queremos guardar un indice a medias sin enterarnos."""

        class _ClienteRoto:
            class models:
                @staticmethod
                def embed_content(model, contents, config):
                    raise _APIErrorFalso(500)

        with self.assertRaises(_APIErrorFalso):
            rag._embed_textos(_ClienteRoto(), [f"t{i}" for i in range(40)], "X")


class TestControlDeRitmo(unittest.TestCase):
    def test_baja_la_concurrencia_tras_varios_429(self):
        control = rag._ControlDeRitmo(4)
        self.assertFalse(control.registrar_429(), "el primero no reduce")
        self.assertTrue(control.registrar_429(), "el segundo si")
        self.assertEqual(control._permitidos, 2)

    def test_un_exito_reinicia_la_cuenta(self):
        control = rag._ControlDeRitmo(4)
        control.registrar_429()
        control.registrar_exito()
        self.assertFalse(control.registrar_429(), "la cuenta se habia reiniciado")

    def test_nunca_baja_de_un_worker(self):
        control = rag._ControlDeRitmo(2)
        for _ in range(10):
            control.registrar_429()
        self.assertGreaterEqual(control._permitidos, 1)


class TestBarraDeProgreso(unittest.TestCase):
    def test_no_falla_con_total_cero(self):
        rag._barra_progreso(0, 0)

    def test_dibuja_el_avance(self):
        import io
        import contextlib

        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            rag._barra_progreso(5, 10)
        self.assertIn("5/10", salida.getvalue())


if __name__ == "__main__":
    unittest.main()
