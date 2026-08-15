"""
Tests de la reescritura de consulta de chatbot.py. Se ejecutan con:

    python -m unittest test_chatbot

Lo importante aqui es que la reescritura nunca rompa la conversacion: ante
cualquier fallo del modelo hay que seguir adelante con la pregunta original.
"""

import unittest

import chatbot
import conversacion


class _Respuesta:
    def __init__(self, text):
        self.text = text


class _Modelos:
    def __init__(self, respuesta=None, error=None):
        self._respuesta = respuesta
        self._error = error
        self.llamadas = 0
        self.ultimo_prompt = None

    def generate_content(self, model, contents, config=None):
        self.llamadas += 1
        self.ultimo_prompt = contents[0].parts[0].text
        if self._error is not None:
            raise self._error
        return self._respuesta


class ClienteFalso:
    def __init__(self, respuesta=None, error=None):
        self.models = _Modelos(respuesta, error)


HISTORIAL = [
    {"role": "user", "text": "Que modelos de zapatilla hay?"},
    {"role": "model", "text": "Estan la Kjerag y la Tomir."},
]


class TestReescribirConsulta(unittest.TestCase):
    def test_devuelve_la_reescritura_del_modelo(self):
        cliente = ClienteFalso(_Respuesta("Cuanto pesa la Kjerag?"))
        resultado = chatbot.reescribir_consulta(cliente, HISTORIAL, "y cuanto pesa?")
        self.assertEqual(resultado, "Cuanto pesa la Kjerag?")

    def test_sin_historial_no_llama_al_modelo(self):
        cliente = ClienteFalso(_Respuesta("lo que sea"))
        resultado = chatbot.reescribir_consulta(cliente, [], "hola")
        self.assertEqual(resultado, "hola")
        self.assertEqual(cliente.models.llamadas, 0)

    def test_error_de_api_cae_a_la_pregunta_original(self):
        cliente = ClienteFalso(error=RuntimeError("api caida"))
        resultado = chatbot.reescribir_consulta(cliente, HISTORIAL, "y cuanto pesa?")
        self.assertEqual(resultado, "y cuanto pesa?")

    def test_respuesta_vacia_cae_a_la_pregunta_original(self):
        for vacio in ("", "   ", None):
            with self.subTest(vacio=vacio):
                cliente = ClienteFalso(_Respuesta(vacio))
                resultado = chatbot.reescribir_consulta(
                    cliente, HISTORIAL, "y cuanto pesa?"
                )
                self.assertEqual(resultado, "y cuanto pesa?")

    def test_limpia_comillas_y_texto_de_mas(self):
        cliente = ClienteFalso(_Respuesta('"Cuanto pesa la Kjerag?"\n\nEspero que ayude.'))
        resultado = chatbot.reescribir_consulta(cliente, HISTORIAL, "y cuanto pesa?")
        self.assertEqual(resultado, "Cuanto pesa la Kjerag?")

    def test_solo_usa_los_ultimos_turnos(self):
        historial = [
            {"role": "user", "text": f"pregunta antigua {i}"} for i in range(20)
        ]
        cliente = ClienteFalso(_Respuesta("reescrita"))
        chatbot.reescribir_consulta(cliente, historial, "y eso?")
        prompt = cliente.models.ultimo_prompt
        self.assertIn("pregunta antigua 19", prompt)
        self.assertNotIn("pregunta antigua 0", prompt)

    def test_se_puede_desactivar(self):
        original = conversacion.REESCRIBIR_CONSULTA
        conversacion.REESCRIBIR_CONSULTA = False
        try:
            cliente = ClienteFalso(_Respuesta("reescrita"))
            resultado = chatbot.reescribir_consulta(cliente, HISTORIAL, "y eso?")
            self.assertEqual(resultado, "y eso?")
            self.assertEqual(cliente.models.llamadas, 0)
        finally:
            conversacion.REESCRIBIR_CONSULTA = original


class TestMensajeConContexto(unittest.TestCase):
    def test_no_queda_ninguna_referencia_al_proyecto_anterior(self):
        mensaje = chatbot.construir_mensaje_con_contexto(
            "una pregunta", [{"text": "un fragmento"}]
        )
        self.assertNotIn("NNormal", mensaje)
        self.assertIn("un fragmento", mensaje)
        self.assertIn("una pregunta", mensaje)

    def test_contexto_vacio_lo_indica(self):
        mensaje = chatbot.construir_mensaje_con_contexto("una pregunta", [])
        self.assertIn("No se ha encontrado informacion relevante", mensaje)


class TestCitasOpcionales(unittest.TestCase):
    CONTEXTO = [
        {"text": "La Kjerag pesa 215 g.", "source": "catalogo.pdf", "page": 4,
         "seccion": "CARACTERISTICAS"},
        {"text": "La Tomir pesa 290 g.", "source": "catalogo.pdf", "page": 7},
    ]

    def test_sin_fuentes_no_numera_ni_pide_citar(self):
        mensaje = chatbot.construir_mensaje_con_contexto(
            "pregunta", self.CONTEXTO, mostrar_fuentes=False
        )
        self.assertNotIn("[1]", mensaje)
        self.assertIn("no la cites", mensaje)

    def test_con_fuentes_numera_los_fragmentos(self):
        mensaje = chatbot.construir_mensaje_con_contexto(
            "pregunta", self.CONTEXTO, mostrar_fuentes=True
        )
        self.assertIn("[1] La Kjerag", mensaje)
        self.assertIn("[2] La Tomir", mensaje)
        self.assertIn("citando cada afirmacion", mensaje)

    def test_las_dos_reglas_no_conviven_en_el_mismo_prompt(self):
        sin = chatbot.construir_system_prompt("Bot", "el tema", mostrar_fuentes=False)
        con = chatbot.construir_system_prompt("Bot", "el tema", mostrar_fuentes=True)

        self.assertIn("No menciones de donde sacas", sin)
        self.assertNotIn("Cita de donde sale", sin)
        self.assertIn("Cita de donde sale", con)
        self.assertNotIn("No menciones de donde sacas", con)

    def test_el_resto_del_prompt_es_identico(self):
        """La plantilla no se duplica: solo cambia la regla 3."""
        sin = chatbot.construir_system_prompt("Bot", "el tema", False)
        con = chatbot.construir_system_prompt("Bot", "el tema", True)
        for fragmento in ("Tu rol:", "Fuente de verdad", "Alcance del tema", "Tono:"):
            self.assertIn(fragmento, sin)
            self.assertIn(fragmento, con)

    def test_leyenda_de_fuentes(self):
        leyenda = chatbot.formatear_leyenda_fuentes(self.CONTEXTO)
        self.assertIn("[1] catalogo.pdf, p. 4 (CARACTERISTICAS)", leyenda)
        self.assertIn("[2] catalogo.pdf, p. 7", leyenda)

    def test_leyenda_vacia_sin_contexto(self):
        self.assertEqual(chatbot.formatear_leyenda_fuentes([]), "")

    def test_leyenda_tolera_metadatos_ausentes(self):
        leyenda = chatbot.formatear_leyenda_fuentes([{"text": "x"}])
        self.assertIn("documento desconocido", leyenda)


class TestFlagEnv(unittest.TestCase):
    def test_valores_verdaderos(self):
        import os

        for valor in ("1", "true", "TRUE", "si", "yes", "on"):
            os.environ["_FLAG_TEST"] = valor
            self.assertTrue(chatbot._flag_env("_FLAG_TEST"), valor)
        del os.environ["_FLAG_TEST"]

    def test_valores_falsos_y_ausencia(self):
        import os

        for valor in ("0", "false", "no", "", "   "):
            os.environ["_FLAG_TEST"] = valor
            self.assertFalse(chatbot._flag_env("_FLAG_TEST"), valor)
        del os.environ["_FLAG_TEST"]
        self.assertFalse(chatbot._flag_env("_FLAG_TEST"))


class _APIErrorFalso(Exception):
    def __init__(self, code):
        super().__init__(f"error {code}")
        self.code = code


class _Fragmento:
    def __init__(self, text):
        self.text = text


class _ModelosStream:
    """Emite fragmentos y, opcionalmente, falla tras N de ellos."""

    def __init__(self, fragmentos, error=None, fallar_tras=0, exitos_tras=None):
        self.fragmentos = fragmentos
        self.error = error
        self.fallar_tras = fallar_tras
        self.exitos_tras = exitos_tras
        self.llamadas = 0

    def generate_content_stream(self, model, config, contents):
        self.llamadas += 1
        if self.exitos_tras is not None and self.llamadas > self.exitos_tras:
            self.error = None
        for i, texto in enumerate(self.fragmentos):
            if self.error is not None and i == self.fallar_tras:
                raise self.error
            yield _Fragmento(texto)
        if self.error is not None and self.fallar_tras >= len(self.fragmentos):
            raise self.error


class ClienteStream:
    def __init__(self, fragmentos, error=None, fallar_tras=0, exitos_tras=None):
        self.models = _ModelosStream(fragmentos, error, fallar_tras, exitos_tras)


class TestStreaming(unittest.TestCase):
    def _capturar(self, cliente, **kw):
        import contextlib
        import io

        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            texto = chatbot.enviar_mensaje_en_streaming(cliente, [], "sp", **kw)
        return texto, salida.getvalue()

    def test_acumula_el_texto_completo(self):
        cliente = ClienteStream(["Hola ", "que ", "tal"])
        texto, impreso = self._capturar(cliente)
        self.assertEqual(texto, "Hola que tal")
        self.assertIn("Hola que tal", impreso)

    def test_imprime_el_prefijo_una_sola_vez(self):
        cliente = ClienteStream(["uno ", "dos ", "tres"])
        _, impreso = self._capturar(cliente, prefijo="Bot: ")
        self.assertEqual(impreso.count("Bot: "), 1)

    def test_ignora_fragmentos_vacios(self):
        cliente = ClienteStream(["Hola", "", None, " mundo"])
        texto, _ = self._capturar(cliente)
        self.assertEqual(texto, "Hola mundo")

    def test_sin_contenido_devuelve_none(self):
        cliente = ClienteStream([])
        texto, _ = self._capturar(cliente)
        self.assertIsNone(texto)

    def test_429_antes_de_imprimir_nada_reintenta(self):
        cliente = ClienteStream(
            ["Respuesta"], error=_APIErrorFalso(429), fallar_tras=0, exitos_tras=1
        )
        original = conversacion.APIError
        conversacion.APIError = _APIErrorFalso
        try:
            texto, _ = self._capturar(cliente)
        finally:
            conversacion.APIError = original
        self.assertEqual(texto, "Respuesta")
        self.assertEqual(cliente.models.llamadas, 2, "deberia haber reintentado")

    def test_429_a_mitad_de_stream_no_reintenta(self):
        """Reintentar aqui duplicaria el principio ya impreso."""
        cliente = ClienteStream(
            ["Primera parte. ", "segunda"], error=_APIErrorFalso(429), fallar_tras=1
        )
        original = conversacion.APIError
        conversacion.APIError = _APIErrorFalso
        try:
            texto, impreso = self._capturar(cliente)
        finally:
            conversacion.APIError = original
        self.assertEqual(cliente.models.llamadas, 1, "no debe reintentar")
        self.assertEqual(texto, "Primera parte. ")
        self.assertIn("se corto", impreso.lower())

    def test_error_no_429_sin_texto_devuelve_none(self):
        cliente = ClienteStream(["x"], error=RuntimeError("boom"), fallar_tras=0)
        texto, _ = self._capturar(cliente)
        self.assertIsNone(texto)

if __name__ == "__main__":
    unittest.main()
