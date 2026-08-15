"""
Tests de la busqueda lexica (BM25), la fusion RRF y los tres modos de
recuperacion. Se ejecutan con:

    python -m unittest test_busqueda

No gastan API: el modo lexico no necesita embeddings, y para los modos que
si los usan se inyecta un cliente falso.
"""

import unittest

import numpy as np

import rag

CORPUS = [
    "La zapatilla Kjerag pesa 215 g y tiene un drop de 6 mm.",
    "La Tomir es un modelo mas robusto, pensado para terreno tecnico.",
    "El modelo con referencia NN-2024-KJ esta disponible en talla 42.",
    "Las plantillas se venden por separado y son compatibles con todos los modelos.",
    "El drop de la Tomir es de 8 mm y su peso ronda los 290 g.",
]


def _chunks(textos=CORPUS):
    return [
        {"text": t, "source": "catalogo.pdf", "page": i + 1}
        for i, t in enumerate(textos)
    ]


class _Embedding:
    def __init__(self, values):
        self.values = values


class _Resultado:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _Modelos:
    """Devuelve un vector fijo para la consulta, de forma que la parte densa
    sea determinista y podamos razonar sobre el resultado."""

    def __init__(self, vector):
        self.vector = vector
        self.llamadas = 0

    def embed_content(self, model, contents, config):
        self.llamadas += len(contents)
        return _Resultado([_Embedding(self.vector) for _ in contents])


class ClienteFalso:
    def __init__(self, vector):
        self.models = _Modelos(vector)


class TestTokenizacion(unittest.TestCase):
    def test_minusculas_sin_tildes_sin_puntuacion(self):
        self.assertEqual(
            rag._tokenizar("¡Máximo, 6 mm de drop!"),
            ["maximo", "6", "mm", "de", "drop"],
        )

    def test_no_altera_el_texto_original(self):
        chunks = _chunks()
        original = chunks[0]["text"]
        rag.crear_buscador(chunks, None, modo="lexica")
        self.assertEqual(chunks[0]["text"], original)


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.indice = rag.IndiceBM25([rag._tokenizar(t) for t in CORPUS])

    def test_encuentra_una_referencia_exacta(self):
        """El caso que la busqueda densa hace peor: un codigo de producto."""
        resultados = self.indice.buscar(rag._tokenizar("NN-2024-KJ"), top_n=3)
        self.assertTrue(resultados)
        self.assertEqual(resultados[0][0], 2)

    def test_termino_inexistente_no_devuelve_nada(self):
        self.assertEqual(self.indice.buscar(rag._tokenizar("bicicleta")), [])

    def test_ordena_por_relevancia(self):
        resultados = self.indice.buscar(rag._tokenizar("Kjerag 215"), top_n=5)
        self.assertEqual(resultados[0][0], 0)

    def test_scores_no_negativos(self):
        resultados = self.indice.buscar(rag._tokenizar("drop modelo"), top_n=5)
        for _, score in resultados:
            self.assertGreaterEqual(score, 0.0)

    def test_corpus_vacio_no_revienta(self):
        self.assertEqual(rag.IndiceBM25([]).buscar(["lo", "que", "sea"]), [])


class TestFusionRRF(unittest.TestCase):
    def test_premia_lo_que_aparece_en_ambas_listas(self):
        # 7 va segundo en las dos; 1 y 2 van primeros pero solo en una.
        orden, _ = rag._fusion_rrf([[1, 7, 3], [2, 7, 4]])
        self.assertEqual(orden[0], 7)

    def test_conserva_todos_los_candidatos(self):
        orden, _ = rag._fusion_rrf([[1, 2], [3, 4]])
        self.assertEqual(set(orden), {1, 2, 3, 4})

    def test_una_sola_lista_mantiene_el_orden(self):
        orden, _ = rag._fusion_rrf([[5, 6, 7]])
        self.assertEqual(orden, [5, 6, 7])


class TestModosDeBusqueda(unittest.TestCase):
    def setUp(self):
        self.chunks = _chunks()
        # Matriz densa donde el chunk 3 es el mas parecido a la consulta.
        matriz = np.zeros((len(self.chunks), 4), dtype=np.float32)
        for i in range(len(self.chunks)):
            matriz[i][i % 4] = 1.0
        self.matriz = matriz
        self.vector_consulta = [0.0, 0.0, 0.0, 1.0]  # apunta al chunk 3

    def test_modo_lexico_no_gasta_embeddings(self):
        buscador = rag.crear_buscador(self.chunks, self.matriz, modo="lexica")
        cliente = ClienteFalso(self.vector_consulta)
        resultados = rag.recuperar_contexto(cliente, "NN-2024-KJ", buscador, top_k=2)
        self.assertEqual(cliente.models.llamadas, 0, "no deberia embeber la consulta")
        self.assertTrue(resultados)
        self.assertIn("NN-2024-KJ", resultados[0]["text"])

    def test_modo_denso_usa_el_coseno(self):
        buscador = rag.crear_buscador(self.chunks, self.matriz, modo="densa")
        cliente = ClienteFalso(self.vector_consulta)
        resultados = rag.recuperar_contexto(
            cliente, "cualquier cosa", buscador, top_k=1, umbral_similitud=0.5
        )
        self.assertEqual(len(resultados), 1)
        self.assertEqual(resultados[0]["text"], CORPUS[3])
        self.assertAlmostEqual(resultados[0]["score"], 1.0, places=5)

    def test_modo_hibrido_mezcla_ambas_vias(self):
        buscador = rag.crear_buscador(self.chunks, self.matriz, modo="hibrida")
        cliente = ClienteFalso(self.vector_consulta)
        resultados = rag.recuperar_contexto(
            cliente, "NN-2024-KJ", buscador, top_k=4, umbral_similitud=0.0
        )
        textos = [r["text"] for r in resultados]
        # El chunk lexicamente exacto entra aunque su vector denso no destaque.
        self.assertTrue(any("NN-2024-KJ" in t for t in textos))

    def test_umbral_descarta_lo_poco_parecido(self):
        buscador = rag.crear_buscador(self.chunks, self.matriz, modo="densa")
        cliente = ClienteFalso(self.vector_consulta)
        resultados = rag.recuperar_contexto(
            cliente, "x", buscador, top_k=5, umbral_similitud=0.99
        )
        self.assertEqual(len(resultados), 1, "solo el vector identico supera 0.99")

    def test_modo_invalido_falla_pronto(self):
        with self.assertRaises(ValueError):
            rag.crear_buscador(self.chunks, self.matriz, modo="magica")

    def test_buscador_vacio_devuelve_lista_vacia(self):
        buscador = rag.crear_buscador([], None)
        self.assertEqual(rag.recuperar_contexto(None, "hola", buscador), [])


class _ModelosRerank:
    """Cliente falso para el reranker: devuelve el texto que se le indique."""

    def __init__(self, texto=None, error=None):
        self.texto = texto
        self.error = error
        self.llamadas = 0

    def generate_content(self, model, contents, config=None):
        self.llamadas += 1
        if self.error:
            raise self.error

        class _R:
            pass

        r = _R()
        r.text = self.texto
        return r


class ClienteRerank:
    def __init__(self, texto=None, error=None):
        self.models = _ModelosRerank(texto, error)


class TestReranker(unittest.TestCase):
    def setUp(self):
        self.candidatos = [dict(c) for c in _chunks()]

    def test_reordena_segun_el_modelo(self):
        cliente = ClienteRerank('[{"id": 2, "puntuacion": 9}, {"id": 0, "puntuacion": 5}]')
        resultado, _ = rag.rerankear(cliente, "pregunta", self.candidatos, top_k=2)
        self.assertEqual(len(resultado), 2)
        self.assertEqual(resultado[0]["text"], CORPUS[2])
        self.assertEqual(resultado[0]["score_rerank"], 9)

    def test_acepta_json_envuelto_en_bloque_de_codigo(self):
        cliente = ClienteRerank('```json\n[{"id": 1, "puntuacion": 7}]\n```')
        resultado, _ = rag.rerankear(cliente, "pregunta", self.candidatos, top_k=2)
        self.assertEqual(resultado[0]["text"], CORPUS[1])

    def test_ignora_ids_fuera_de_rango_y_repetidos(self):
        cliente = ClienteRerank('[{"id": 99}, {"id": 1}, {"id": 1}, {"id": -3}]')
        resultado, _ = rag.rerankear(cliente, "pregunta", self.candidatos, top_k=5)
        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0]["text"], CORPUS[1])

    def test_json_ilegible_cae_al_orden_original(self):
        cliente = ClienteRerank("lo siento, no puedo")
        resultado, _ = rag.rerankear(cliente, "pregunta", self.candidatos, top_k=3)
        self.assertEqual(resultado, self.candidatos)

    def test_error_de_api_cae_al_orden_original(self):
        cliente = ClienteRerank(error=RuntimeError("caido"))
        resultado, _ = rag.rerankear(cliente, "pregunta", self.candidatos, top_k=3)
        self.assertEqual(resultado, self.candidatos)

    def test_backend_desconocido_no_revienta(self):
        cliente = ClienteRerank("[]")
        resultado, _ = rag.rerankear(
            cliente, "pregunta", self.candidatos, top_k=3, backend="inventado"
        )
        self.assertEqual(resultado, self.candidatos)

    def test_mide_la_latencia(self):
        cliente = ClienteRerank('[{"id": 0}]')
        _, latencia = rag.rerankear(cliente, "pregunta", self.candidatos, top_k=1)
        self.assertGreaterEqual(latencia, 0.0)
        self.assertEqual(rag.ULTIMA_LATENCIA_RERANK, latencia)

    def test_no_se_llama_si_caben_todos_los_candidatos(self):
        buscador = rag.crear_buscador(_chunks(), None, modo="lexica")
        cliente = ClienteRerank('[{"id": 0}]')
        rag.recuperar_contexto(
            cliente, "NN-2024-KJ", buscador, top_k=10, usar_reranker=True
        )
        self.assertEqual(cliente.models.llamadas, 0)



class TestUmbralAdaptativo(unittest.TestCase):
    def _cands(self, *scores):
        return [{"text": f"c{i}", "score": s} for i, s in enumerate(scores)]

    def test_conserva_los_cercanos_al_mejor(self):
        # margen 0.85 sobre 0.90 -> corte en 0.765
        resultado = rag._filtrar_por_umbral(self._cands(0.90, 0.80, 0.50))
        self.assertEqual(len(resultado), 2)

    def test_descarta_todo_si_nada_supera_el_suelo(self):
        """Una pregunta fuera del corpus no debe colar el 'menos malo'."""
        resultado = rag._filtrar_por_umbral(self._cands(0.20, 0.19, 0.05))
        self.assertEqual(resultado, [])

    def test_el_mejor_siempre_entra_si_supera_el_suelo(self):
        resultado = rag._filtrar_por_umbral(self._cands(0.35, 0.10))
        self.assertEqual(len(resultado), 1)
        self.assertAlmostEqual(resultado[0]["score"], 0.35)

    def test_umbral_fijo_sigue_disponible(self):
        resultado = rag._filtrar_por_umbral(self._cands(0.90, 0.80, 0.50), umbral=0.55)
        self.assertEqual(len(resultado), 2)

    def test_margen_configurable(self):
        cands = self._cands(0.90, 0.80, 0.70)
        estricto = rag._filtrar_por_umbral(cands, margen=0.95)
        laxo = rag._filtrar_por_umbral(cands, margen=0.5)
        self.assertEqual(len(estricto), 1)
        self.assertEqual(len(laxo), 3)

    def test_sin_scores_no_filtra(self):
        """En modo lexico no hay coseno: no se puede filtrar por umbral."""
        cands = [{"text": "a"}, {"text": "b"}]
        self.assertEqual(rag._filtrar_por_umbral(cands), cands)

    def test_adaptativo_por_defecto_en_recuperar_contexto(self):
        chunks = _chunks()
        matriz = np.zeros((len(chunks), 4), dtype=np.float32)
        matriz[0][0] = 1.0
        for i in range(1, len(chunks)):
            matriz[i][1] = 1.0
        buscador = rag.crear_buscador(chunks, matriz, modo="densa")
        cliente = ClienteFalso([1.0, 0.0, 0.0, 0.0])
        resultados = rag.recuperar_contexto(cliente, "x", buscador, top_k=5)
        self.assertEqual(len(resultados), 1, "solo el alineado supera el corte")

class TestIndiceVectorial(unittest.TestCase):
    def setUp(self):
        matriz = np.zeros((6, 4), dtype=np.float32)
        for i in range(6):
            matriz[i][i % 4] = 1.0
        matriz[0] = [1.0, 0.0, 0.0, 0.0]
        self.matriz = matriz

    def test_numpy_devuelve_indices_y_coseno(self):
        indice = rag.IndiceNumpy(self.matriz)
        resultados = indice.buscar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 2)
        self.assertEqual(len(resultados), 2)
        self.assertEqual(resultados[0][0], 0)
        self.assertAlmostEqual(resultados[0][1], 1.0, places=5)

    def test_ordena_de_mas_a_menos(self):
        indice = rag.IndiceNumpy(self.matriz)
        resultados = indice.buscar(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 6)
        scores = [s for _, s in resultados]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_auto_elige_numpy_con_corpus_pequeno(self):
        indice = rag.crear_indice_vectorial(self.matriz, backend="auto")
        self.assertEqual(indice.nombre, "numpy")

    def test_matriz_vacia_devuelve_none(self):
        self.assertIsNone(rag.crear_indice_vectorial(None))
        self.assertIsNone(rag.crear_indice_vectorial(np.zeros((0, 4), dtype=np.float32)))

    def test_backend_hnsw_cae_a_numpy_si_no_esta_instalado(self):
        """No queremos que falte una dependencia opcional y se caiga el bot."""
        indice = rag.crear_indice_vectorial(self.matriz, backend="hnsw")
        self.assertIsNotNone(indice)
        self.assertIn(indice.nombre, ("hnsw", "numpy"))

    def test_el_buscador_usa_el_indice_vectorial(self):
        buscador = rag.crear_buscador(_chunks(), self.matriz[:5], modo="densa", index_dir=None)
        self.assertIsNotNone(buscador.indice_vectorial)

    def test_modo_lexico_no_construye_indice_vectorial(self):
        buscador = rag.crear_buscador(_chunks(), self.matriz[:5], modo="lexica", index_dir=None)
        self.assertIsNone(buscador.indice_vectorial)

    def test_benchmark_devuelve_metricas(self):
        matriz = rag._normalizar_filas(
            np.random.default_rng(1).normal(size=(120, 8)).astype(np.float32)
        )
        resultado = rag.benchmark_vectorial(matriz, top_k=4, repeticiones=1)
        self.assertIn("numpy", resultado)
        self.assertEqual(resultado["numpy"]["recall@4"], 1.0)
        self.assertGreater(resultado["numpy"]["latencia_ms"], 0.0)

    def test_benchmark_con_matriz_vacia(self):
        self.assertEqual(rag.benchmark_vectorial(None), {})


class TestHibridoConScoresDispersos(unittest.TestCase):
    """En hibrido, un candidato que solo aporta BM25 no viene con score
    denso; hay que calcularselo o el filtro lo tiraria injustamente."""

    def test_el_candidato_lexico_recibe_score(self):
        chunks = _chunks()
        matriz = np.zeros((len(chunks), 4), dtype=np.float32)
        for i in range(len(chunks)):
            matriz[i][i % 4] = 1.0
        buscador = rag.crear_buscador(chunks, matriz, modo="hibrida", index_dir=None)
        cliente = ClienteFalso([1.0, 0.0, 0.0, 0.0])
        resultados = rag.recuperar_contexto(
            cliente, "NN-2024-KJ", buscador, top_k=5, umbral_similitud=0.0
        )
        for r in resultados:
            self.assertIsNotNone(r.get("score"), f"sin score: {r['text'][:30]}")

if __name__ == "__main__":
    unittest.main()
