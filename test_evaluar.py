"""
Tests de las metricas de evaluar.py. Se ejecutan con:

    python -m unittest test_evaluar

Son logica pura, sin API. Merecen test porque un error aqui no se ve: las
metricas seguirian saliendo, solo que mintiendo, y con ellas se decide que
configuracion es mejor.
"""

import unittest

import evaluar


def _chunk(text, source="a.pdf", page=1):
    return {"text": text, "source": source, "page": page}


class TestNormalizar(unittest.TestCase):
    def test_quita_tildes_y_mayusculas(self):
        self.assertEqual(evaluar.normalizar("Drop de 6 MM"), "drop de 6 mm")
        self.assertEqual(evaluar.normalizar("Kjerag pesa 215 g"), "kjerag pesa 215 g")
        self.assertEqual(evaluar.normalizar("ámbito ÉPICO"), "ambito epico")

    def test_colapsa_espacios_y_saltos(self):
        self.assertEqual(evaluar.normalizar("215\n  g\tpor  pie"), "215 g por pie")

    def test_acepta_none(self):
        self.assertEqual(evaluar.normalizar(None), "")


class TestRelevancia(unittest.TestCase):
    def test_fragmento_esperado_ignora_tildes_y_saltos(self):
        caso = {"esperado": "pesa 215 g"}
        self.assertTrue(evaluar._chunk_es_relevante(_chunk("La Kjerag pesa\n215 g."), caso))

    def test_fragmento_ausente(self):
        caso = {"esperado": "pesa 215 g"}
        self.assertFalse(evaluar._chunk_es_relevante(_chunk("La Tomir pesa 250 g"), caso))

    def test_documento_y_pagina_deben_coincidir(self):
        caso = {"esperado": "215 g", "documento": "a.pdf", "pagina": 4}
        self.assertTrue(
            evaluar._chunk_es_relevante(_chunk("pesa 215 g", "a.pdf", 4), caso)
        )
        self.assertFalse(
            evaluar._chunk_es_relevante(_chunk("pesa 215 g", "b.pdf", 4), caso),
            "documento distinto no deberia contar",
        )
        self.assertFalse(
            evaluar._chunk_es_relevante(_chunk("pesa 215 g", "a.pdf", 9), caso),
            "pagina distinta no deberia contar",
        )

    def test_caso_sin_criterios_nunca_acierta(self):
        """Un caso mal escrito no puede puntuar como acierto gratis."""
        self.assertFalse(evaluar._chunk_es_relevante(_chunk("lo que sea"), {}))

    def test_posicion_del_primer_acierto(self):
        caso = {"esperado": "215 g"}
        contexto = [_chunk("nada"), _chunk("tampoco"), _chunk("pesa 215 g")]
        self.assertEqual(evaluar._posicion_primer_acierto(contexto, caso), 3)
        self.assertIsNone(evaluar._posicion_primer_acierto([_chunk("nada")], caso))


class TestAgregar(unittest.TestCase):
    def _resultado(self, **kwargs):
        base = {
            "id": "x",
            "tipo": "positivo",
            "posicion": None,
            "acierto_recuperacion": False,
            "latencia_recuperacion": 0.1,
            "fiel": None,
            "responde": None,
            "nota": None,
            "rechaza": None,
        }
        base.update(kwargs)
        return base

    def test_recall_y_mrr(self):
        resultados = [
            self._resultado(posicion=1, acierto_recuperacion=True),
            self._resultado(posicion=2, acierto_recuperacion=True),
            self._resultado(posicion=None, acierto_recuperacion=False),
        ]
        metricas = evaluar.agregar(resultados)
        self.assertAlmostEqual(metricas["recall"], 2 / 3)
        # (1/1 + 1/2 + 0) / 3
        self.assertAlmostEqual(metricas["mrr"], 0.5)

    def test_mrr_premia_la_primera_posicion(self):
        primero = evaluar.agregar([self._resultado(posicion=1, acierto_recuperacion=True)])
        ultimo = evaluar.agregar([self._resultado(posicion=4, acierto_recuperacion=True)])
        self.assertEqual(primero["recall"], ultimo["recall"])
        self.assertGreater(primero["mrr"], ultimo["mrr"])

    def test_los_negativos_no_entran_en_recall(self):
        resultados = [
            self._resultado(posicion=1, acierto_recuperacion=True),
            self._resultado(tipo="negativo", rechaza=True),
            self._resultado(tipo="negativo", rechaza=False),
        ]
        metricas = evaluar.agregar(resultados)
        self.assertEqual(metricas["recall"], 1.0, "el negativo no debe penalizar recall")
        self.assertEqual(metricas["positivos"], 1)
        self.assertEqual(metricas["negativos"], 2)
        self.assertAlmostEqual(metricas["rechazo"], 0.5)

    def test_sin_juez_las_metricas_de_respuesta_son_none(self):
        metricas = evaluar.agregar([self._resultado(posicion=1, acierto_recuperacion=True)])
        self.assertIsNone(metricas["fidelidad"])
        self.assertIsNone(metricas["nota_media"])


class TestRechazo(unittest.TestCase):
    def test_detecta_frases_de_rechazo(self):
        self.assertTrue(evaluar._parece_rechazo("No tengo esa información."))
        self.assertTrue(evaluar._parece_rechazo("Lo siento, no dispongo de ese dato"))

    def test_no_marca_una_respuesta_normal(self):
        self.assertFalse(evaluar._parece_rechazo("La Kjerag pesa 215 g y tiene 6 mm."))


if __name__ == "__main__":
    unittest.main()
