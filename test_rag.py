"""
Tests del indexado incremental de rag.py. Se ejecutan con:

    python -m unittest test_rag

No hacen ninguna llamada real a la API: usan un cliente falso que devuelve
un vector determinista por texto y registra que se le ha pedido embeber,
que es justo lo que hay que vigilar (que no se reembeba de mas y que
chunks y embeddings no se desalineen).
"""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rag


def _vector_determinista(texto):
    """Mismo texto -> mismo vector, para poder comparar entre pasadas."""
    semilla = int.from_bytes(hashlib.sha256(texto.encode("utf-8")).digest()[:4], "big")
    generador = np.random.default_rng(semilla)
    return generador.normal(size=rag.EMBEDDING_DIM).tolist()


def _vector_esperado_normalizado(texto):
    vector = np.array(_vector_determinista(texto), dtype=np.float32)
    return vector / np.linalg.norm(vector)


class _Embedding:
    def __init__(self, values):
        self.values = values


class _Resultado:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _Modelos:
    def __init__(self, registro):
        self._registro = registro

    def embed_content(self, model, contents, config):
        self._registro.extend(contents)
        return _Resultado([_Embedding(_vector_determinista(t)) for t in contents])


class ClienteFalso:
    def __init__(self):
        self.textos_embebidos = []
        self.models = _Modelos(self.textos_embebidos)


def _leer_pdf_falso(pdf_path):
    """Extractor de prueba: trata el archivo como texto plano, para no
    tener que generar PDFs reales en los tests."""
    return [(1, Path(pdf_path).read_text(encoding="utf-8"))]


def _texto_largo(etiqueta, repeticiones=12):
    """Texto suficientemente largo como para producir varios chunks."""
    return " ".join(
        f"{etiqueta} frase numero {i} con contenido de relleno suficiente "
        f"para que el troceado genere mas de un fragmento."
        for i in range(repeticiones)
    )


class TestReindexadoIncremental(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.docs = self.tmp / "docs"
        self.docs.mkdir()
        self.index = self.tmp / "rag_index"

        # Se parchea la entrada de EXTRACTORES, no rag._leer_paginas_pdf:
        # el despacho por extension guarda una referencia a la funcion, asi
        # que sustituir el atributo del modulo no tendria ningun efecto.
        self._extractor_original = rag.EXTRACTORES[".pdf"]
        rag.EXTRACTORES[".pdf"] = _leer_pdf_falso

        self.textos = {
            "a.pdf": _texto_largo("Alfa"),
            "b.pdf": _texto_largo("Bravo"),
            "c.pdf": _texto_largo("Charlie"),
        }
        for nombre, texto in self.textos.items():
            (self.docs / nombre).write_text(texto, encoding="utf-8")

    def tearDown(self):
        rag.EXTRACTORES[".pdf"] = self._extractor_original
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _construir(self):
        cliente = ClienteFalso()
        rag.construir_indice(cliente, docs_dir=self.docs, index_dir=self.index)
        chunks, matriz = rag.cargar_indice(self.index)
        return cliente, chunks, matriz

    def _filas_de(self, chunks, matriz, nombre):
        posiciones = [i for i, c in enumerate(chunks) if c["source"] == nombre]
        return [chunks[i] for i in posiciones], matriz[posiciones]

    def test_modificar_un_pdf_conserva_los_embeddings_de_los_demas(self):
        _, chunks_ini, matriz_ini = self._construir()
        self.assertGreater(len(chunks_ini), 3, "esperaba varios chunks por documento")

        a_ini = self._filas_de(chunks_ini, matriz_ini, "a.pdf")
        c_ini = self._filas_de(chunks_ini, matriz_ini, "c.pdf")

        # Se modifica solo b.pdf.
        (self.docs / "b.pdf").write_text(_texto_largo("Bravo revisado"), encoding="utf-8")
        cliente, chunks_fin, matriz_fin = self._construir()

        # a.pdf y c.pdf conservan exactamente sus chunks y sus vectores.
        for nombre, (chunks_antes, matriz_antes) in (("a.pdf", a_ini), ("c.pdf", c_ini)):
            chunks_ahora, matriz_ahora = self._filas_de(chunks_fin, matriz_fin, nombre)
            self.assertEqual(chunks_antes, chunks_ahora, f"{nombre}: chunks alterados")
            np.testing.assert_array_equal(
                matriz_antes, matriz_ahora, err_msg=f"{nombre}: embeddings alterados"
            )

        # Y no se ha vuelto a pedir el embedding de ninguno de sus fragmentos.
        textos_intactos = {c["text"] for c in a_ini[0]} | {c["text"] for c in c_ini[0]}
        self.assertTrue(cliente.textos_embebidos, "b.pdf deberia haberse reembebido")
        self.assertFalse(
            textos_intactos & set(cliente.textos_embebidos),
            "se han reembebido fragmentos de documentos que no cambiaron",
        )

    def test_sin_cambios_no_se_embebe_nada(self):
        self._construir()
        cliente, _, _ = self._construir()
        self.assertEqual(cliente.textos_embebidos, [])

    def test_borrar_un_pdf_elimina_solo_sus_chunks(self):
        _, chunks_ini, _ = self._construir()
        (self.docs / "c.pdf").unlink()
        cliente, chunks_fin, matriz_fin = self._construir()

        fuentes = {c["source"] for c in chunks_fin}
        self.assertEqual(fuentes, {"a.pdf", "b.pdf"})
        self.assertEqual(len(chunks_fin), matriz_fin.shape[0])
        self.assertEqual(cliente.textos_embebidos, [], "no habia que reembeber nada")
        self.assertLess(len(chunks_fin), len(chunks_ini))

    def test_cada_chunk_conserva_su_propio_embedding(self):
        """La garantia critica: la fila i de la matriz corresponde al chunk i.
        Se comprueba tras varias altas, bajas y modificaciones."""
        self._construir()
        (self.docs / "b.pdf").write_text(_texto_largo("Bravo v2"), encoding="utf-8")
        (self.docs / "d.pdf").write_text(_texto_largo("Delta"), encoding="utf-8")
        (self.docs / "a.pdf").unlink()
        _, chunks, matriz = self._construir()

        self.assertEqual(len(chunks), matriz.shape[0])
        for i, chunk in enumerate(chunks):
            np.testing.assert_allclose(
                matriz[i],
                _vector_esperado_normalizado(chunk["text"]),
                atol=1e-5,
                err_msg=f"el chunk {i} ({chunk['source']}) no cuadra con su vector",
            )

    def test_embeddings_normalizados_al_guardar(self):
        _, _, matriz = self._construir()
        normas = np.linalg.norm(matriz, axis=1)
        np.testing.assert_allclose(normas, np.ones_like(normas), atol=1e-5)

    def test_dimension_reducida(self):
        _, _, matriz = self._construir()
        self.assertEqual(matriz.shape[1], rag.EMBEDDING_DIM)


class TestManifestIncompatible(unittest.TestCase):
    """Un manifest de formato antiguo, corrupto o generado con otra
    configuracion tiene que provocar una reconstruccion completa, nunca un
    error ni la reutilizacion de vectores invalidos."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.docs = self.tmp / "docs"
        self.docs.mkdir()
        self.index = self.tmp / "rag_index"
        # Se parchea la entrada de EXTRACTORES, no rag._leer_paginas_pdf:
        # el despacho por extension guarda una referencia a la funcion, asi
        # que sustituir el atributo del modulo no tendria ningun efecto.
        self._extractor_original = rag.EXTRACTORES[".pdf"]
        rag.EXTRACTORES[".pdf"] = _leer_pdf_falso
        (self.docs / "a.pdf").write_text(_texto_largo("Alfa"), encoding="utf-8")

    def tearDown(self):
        rag.EXTRACTORES[".pdf"] = self._extractor_original
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _reconstruye(self):
        cliente = ClienteFalso()
        rag.construir_indice(cliente, docs_dir=self.docs, index_dir=self.index)
        return bool(cliente.textos_embebidos)

    def test_manifest_formato_antiguo(self):
        self._reconstruye()
        _, _, manifest_path = rag._rutas_indice(self.index)
        # Formato v1: mapa plano {nombre: hash}, sin version ni dimension.
        manifest_path.write_text(
            json.dumps({"a.pdf": "hash-cualquiera"}), encoding="utf-8"
        )
        self.assertTrue(self._reconstruye(), "deberia reindexar desde cero")
        self.assertTrue(rag.indice_disponible(self.index))

    def test_manifest_corrupto(self):
        self._reconstruye()
        _, _, manifest_path = rag._rutas_indice(self.index)
        manifest_path.write_text("{ esto no es json", encoding="utf-8")
        self.assertTrue(self._reconstruye(), "deberia reindexar desde cero")

    def test_cambio_de_dimension_regenera(self):
        self._reconstruye()
        _, _, manifest_path = rag._rutas_indice(self.index)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["embedding_dim"] = rag.EMBEDDING_DIM * 2
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        desactualizado, _ = rag.indice_desactualizado(self.docs, self.index)
        self.assertTrue(desactualizado, "un cambio de dimension debe invalidar")
        self.assertTrue(self._reconstruye())


if __name__ == "__main__":
    unittest.main()
