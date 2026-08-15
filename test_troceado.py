"""
Tests del troceado estructurado. Se ejecutan con:

    python -m unittest test_troceado

Lo que se vigila aqui: que no se pierda texto al trocear, que la pagina y
la seccion asignadas a cada chunk sean las correctas, y que el contenido
que cruza un salto de pagina deje de partirse.
"""

import unittest

import rag


class TestConcatenarPaginas(unittest.TestCase):
    def test_offsets_apuntan_a_la_pagina_correcta(self):
        paginas = [(1, "primera pagina"), (2, "segunda pagina"), (3, "tercera")]
        texto, offsets = rag._concatenar_paginas(paginas)

        self.assertEqual(rag._pagina_de_offset(offsets, 0), 1)
        self.assertEqual(rag._pagina_de_offset(offsets, texto.index("segunda")), 2)
        self.assertEqual(rag._pagina_de_offset(offsets, texto.index("tercera")), 3)

    def test_conserva_todo_el_texto(self):
        paginas = [(1, "alfa"), (2, "bravo")]
        texto, _ = rag._concatenar_paginas(paginas)
        self.assertIn("alfa", texto)
        self.assertIn("bravo", texto)

    def test_sin_paginas(self):
        texto, offsets = rag._concatenar_paginas([])
        self.assertEqual(texto, "")
        self.assertEqual(rag._pagina_de_offset(offsets, 0), 1)


class TestDeteccionDeSecciones(unittest.TestCase):
    def test_titulos_numerados(self):
        self.assertTrue(rag._es_titulo_de_seccion("3.2 Materiales"))
        self.assertTrue(rag._es_titulo_de_seccion("1. Introduccion"))
        self.assertTrue(rag._es_titulo_de_seccion("4.1.2) Suela"))

    def test_titulos_en_mayusculas(self):
        self.assertTrue(rag._es_titulo_de_seccion("CARACTERISTICAS TECNICAS"))

    def test_no_confunde_texto_normal(self):
        self.assertFalse(rag._es_titulo_de_seccion("La zapatilla pesa 215 g."))
        self.assertFalse(rag._es_titulo_de_seccion(""))
        self.assertFalse(
            rag._es_titulo_de_seccion("ESTA LINEA ES DEMASIADO LARGA " * 5),
            "una linea larga no es un titulo aunque este en mayusculas",
        )

    def test_no_confunde_una_cifra_suelta(self):
        self.assertFalse(rag._es_titulo_de_seccion("215"))

    def test_seccion_mas_cercana_por_encima(self):
        texto = "INTRO\ntexto uno\n\n2. DETALLES\ntexto dos\n"
        secciones = rag._secciones_con_offset(texto)
        self.assertEqual(rag._seccion_de_offset(secciones, texto.index("texto uno")), "INTRO")
        self.assertEqual(
            rag._seccion_de_offset(secciones, texto.index("texto dos")), "2. DETALLES"
        )

    def test_sin_secciones_devuelve_none(self):
        self.assertIsNone(rag._seccion_de_offset([], 10))


class TestTroceadoJerarquico(unittest.TestCase):
    def test_respeta_los_parrafos_si_caben(self):
        texto = "Primer parrafo corto.\n\nSegundo parrafo corto."
        unidades = rag._unidades_con_offset(texto, chunk_size=100)
        self.assertEqual([u[0] for u in unidades], ["Primer parrafo corto.", "Segundo parrafo corto."])

    def test_parrafo_largo_se_parte_por_frases(self):
        texto = "Frase una que ocupa. Frase dos que ocupa. Frase tres que ocupa."
        unidades = rag._unidades_con_offset(texto, chunk_size=25)
        self.assertGreater(len(unidades), 1)
        for texto_unidad, _ in unidades:
            self.assertLessEqual(len(texto_unidad), 25)
        # Ninguna frase se ha partido por la mitad de una palabra.
        self.assertTrue(all(u[0].endswith(".") for u in unidades))

    def test_frase_gigante_se_corta_a_lo_bruto(self):
        texto = "palabra " * 60  # una sola frase sin puntuacion
        unidades = rag._unidades_con_offset(texto, chunk_size=50)
        self.assertGreater(len(unidades), 1)
        for texto_unidad, _ in unidades:
            self.assertLessEqual(len(texto_unidad), 50)

    def test_los_offsets_apuntan_al_texto_real(self):
        """Si los offsets mienten, la pagina y la seccion del chunk mienten."""
        texto = "ALFA\n\nParrafo uno aqui.\n\nParrafo dos alla."
        for texto_unidad, offset in rag._unidades_con_offset(texto, chunk_size=100):
            self.assertEqual(
                texto[offset : offset + len(texto_unidad)],
                texto_unidad,
                f"offset incorrecto para {texto_unidad!r}",
            )


class TestTroceadoEstructurado(unittest.TestCase):
    def test_contenido_que_cruza_paginas_no_se_parte(self):
        """El fallo del troceado anterior: al ir pagina a pagina, una frase
        partida por el salto quedaba cortada en dos chunks sin sentido."""
        paginas = [(1, "La zapatilla Kjerag pesa"), (2, "215 gramos en talla 42.")]
        chunks = rag._trocear_estructurado(paginas, "catalogo.pdf", chunk_size=500)
        unido = " ".join(c["cuerpo"] for c in chunks)
        self.assertIn("pesa", unido)
        self.assertIn("215 gramos", unido)
        self.assertEqual(len(chunks), 1, "deberia caber todo en un solo chunk")

    def test_cabecera_en_el_texto_y_en_campo_aparte(self):
        paginas = [(1, "CARACTERISTICAS\n\nLa suela es de goma.")]
        chunks = rag._trocear_estructurado(paginas, "catalogo.pdf", chunk_size=500)
        chunk = chunks[0]
        self.assertIn("catalogo.pdf", chunk["text"])
        self.assertEqual(chunk["cabecera"], "catalogo.pdf > CARACTERISTICAS")
        self.assertEqual(chunk["seccion"], "CARACTERISTICAS")
        self.assertNotIn("[catalogo.pdf", chunk["cuerpo"], "el cuerpo va sin cabecera")

    def test_sin_seccion_la_cabecera_es_solo_el_documento(self):
        paginas = [(1, "Un texto cualquiera sin titulos.")]
        chunks = rag._trocear_estructurado(paginas, "notas.pdf", chunk_size=500)
        self.assertEqual(chunks[0]["cabecera"], "notas.pdf")
        self.assertIsNone(chunks[0]["seccion"])

    def test_asigna_la_pagina_correcta(self):
        paginas = [(1, "Alfa " * 60), (2, "Bravo " * 60), (3, "Charlie " * 60)]
        chunks = rag._trocear_estructurado(paginas, "d.pdf", chunk_size=200, overlap=0)
        paginas_vistas = {c["page"] for c in chunks}
        self.assertTrue(paginas_vistas.issubset({1, 2, 3}))
        self.assertIn(1, paginas_vistas)
        self.assertIn(3, paginas_vistas)

    def test_no_pierde_contenido(self):
        paginas = [(1, "uno dos tres.\n\ncuatro cinco seis.\n\nsiete ocho nueve.")]
        chunks = rag._trocear_estructurado(paginas, "d.pdf", chunk_size=30, overlap=0)
        unido = " ".join(c["cuerpo"] for c in chunks)
        for palabra in ("uno", "cuatro", "siete", "nueve"):
            self.assertIn(palabra, unido)

    def test_documento_vacio_no_genera_chunks(self):
        self.assertEqual(rag._trocear_estructurado([(1, "   ")], "vacio.pdf"), [])
        self.assertEqual(rag._trocear_estructurado([], "vacio.pdf"), [])

    def test_solapamiento_repite_contexto_entre_chunks(self):
        paginas = [(1, "Alfa uno.\n\nBravo dos.\n\nCharlie tres.\n\nDelta cuatro.")]
        con_solape = rag._trocear_estructurado(
            paginas, "d.pdf", chunk_size=25, overlap=15
        )
        sin_solape = rag._trocear_estructurado(paginas, "d.pdf", chunk_size=25, overlap=0)
        largo_con = sum(len(c["cuerpo"]) for c in con_solape)
        largo_sin = sum(len(c["cuerpo"]) for c in sin_solape)
        self.assertGreater(largo_con, largo_sin, "el solapamiento debe repetir texto")

    def test_los_chunks_respetan_el_tamaño_pedido(self):
        paginas = [(1, " ".join(f"Frase de relleno numero {i}." for i in range(40)))]
        chunks = rag._trocear_estructurado(paginas, "d.pdf", chunk_size=200, overlap=20)
        for chunk in chunks:
            # Margen: se empaqueta por unidades completas, no se corta a medias.
            self.assertLessEqual(len(chunk["cuerpo"]), 260)


if __name__ == "__main__":
    unittest.main()
