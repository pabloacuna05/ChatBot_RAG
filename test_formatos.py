"""
Tests de los extractores de formato. Se ejecutan con:

    python -m unittest test_formatos

Todos devuelven la misma forma [(pagina, texto)], que es lo que permite que
el resto del pipeline no sepa de que formato viene el contenido.
"""

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import rag

DOCX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Primer parrafo del documento.</w:t></w:r></w:p>
    <w:p><w:r><w:t>Segundo </w:t></w:r><w:r><w:t>parrafo partido.</w:t></w:r></w:p>
    <w:p></w:p>
  </w:body>
</w:document>"""


class TestExtractores(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _crear(self, nombre, contenido):
        ruta = self.tmp / nombre
        ruta.write_text(contenido, encoding="utf-8")
        return ruta

    def test_texto_plano(self):
        ruta = self._crear("notas.txt", "Contenido de prueba.")
        self.assertEqual(rag._leer_documento(ruta), [(1, "Contenido de prueba.")])

    def test_markdown(self):
        ruta = self._crear("guia.md", "# Titulo\n\nParrafo de texto.")
        paginas = rag._leer_documento(ruta)
        self.assertIn("Titulo", paginas[0][1])
        self.assertIn("Parrafo de texto", paginas[0][1])

    def test_html_ignora_etiquetas_y_scripts(self):
        ruta = self._crear(
            "pagina.html",
            "<html><head><style>p{color:red}</style></head>"
            "<body><script>var x=1;</script>"
            "<h1>Titulo</h1><p>Texto visible.</p></body></html>",
        )
        texto = rag._leer_documento(ruta)[0][1]
        self.assertIn("Titulo", texto)
        self.assertIn("Texto visible", texto)
        self.assertNotIn("var x", texto, "el script no deberia indexarse")
        self.assertNotIn("color:red", texto, "el css no deberia indexarse")

    def test_docx(self):
        ruta = self.tmp / "informe.docx"
        with zipfile.ZipFile(ruta, "w") as z:
            z.writestr("word/document.xml", DOCX_XML)
        texto = rag._leer_documento(ruta)[0][1]
        self.assertIn("Primer parrafo del documento.", texto)
        self.assertIn("Segundo parrafo partido.", texto, "los runs deben unirse")

    def test_docx_corrupto_no_revienta(self):
        ruta = self._crear("roto.docx", "esto no es un zip")
        self.assertEqual(rag._leer_documento(ruta), [])

    def test_extension_desconocida_se_ignora(self):
        ruta = self._crear("hoja.xlsx", "lo que sea")
        self.assertEqual(rag._leer_documento(ruta), [])

    def test_archivo_vacio(self):
        ruta = self._crear("vacio.txt", "   ")
        self.assertEqual(rag._leer_documento(ruta), [])


class TestDescubrimientoDeDocumentos(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for nombre in ("a.pdf", "b.txt", "c.md", "d.html", "e.docx", "f.xlsx", "g.jpg"):
            (self.tmp / nombre).write_text("x", encoding="utf-8")
        (self.tmp / "subcarpeta").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_solo_recoge_formatos_soportados(self):
        nombres = {r.name for r in rag._documentos_en(self.tmp)}
        self.assertEqual(nombres, {"a.pdf", "b.txt", "c.md", "d.html", "e.docx"})

    def test_ignora_directorios(self):
        self.assertNotIn("subcarpeta", {r.name for r in rag._documentos_en(self.tmp)})

    def test_carpeta_inexistente(self):
        self.assertEqual(rag._documentos_en(self.tmp / "no_existe"), [])

    def test_el_estado_cubre_todos_los_formatos(self):
        estado = rag._estado_actual_docs(self.tmp)
        self.assertEqual(len(estado), 5)
        for valor in estado.values():
            self.assertEqual(len(valor), 64, "sha256 en hexadecimal")


class TestPdfEscaneado(unittest.TestCase):
    """Un PDF sin texto extraible tiene que avisar por consola con el nombre
    del archivo: en silencio, el bot diria 'no tengo esa informacion' a todo
    y nadie sabria por que."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ruta = self.tmp / "escaneado.pdf"
        self.ruta.write_text("x", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_avisa_si_no_hay_texto(self):
        import contextlib
        import io

        class _PaginaVacia:
            def extract_text(self):
                return ""

        class _LectorFalso:
            pages = [_PaginaVacia(), _PaginaVacia()]

        original = rag.PdfReader
        rag.PdfReader = lambda ruta: _LectorFalso()
        try:
            salida = io.StringIO()
            with contextlib.redirect_stdout(salida):
                paginas = rag._leer_paginas_pdf(self.ruta)
        finally:
            rag.PdfReader = original

        self.assertEqual(paginas, [])
        mensaje = salida.getvalue()
        self.assertIn("escaneado.pdf", mensaje)
        self.assertIn("escaneado", mensaje.lower())
        self.assertIn("USAR_OCR", mensaje)


if __name__ == "__main__":
    unittest.main()
