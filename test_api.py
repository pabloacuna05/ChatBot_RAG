"""
Tests de las piezas de la API que no necesitan FastAPI instalado: validacion
de nombres de coleccion, cache LRU de buscadores y almacenes de historial.

    python -m unittest test_api

La validacion de nombres es lo mas importante de este archivo: el nombre de
la coleccion acaba siendo parte de una ruta en disco.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import colecciones
from historial import HistorialJSON, HistorialSQLite


class TestValidacionDeNombres(unittest.TestCase):
    def test_acepta_nombres_razonables(self):
        for nombre in ("catalogo", "apuntes-2024", "manual_v2", "a", "A1"):
            self.assertEqual(colecciones.validar_nombre(nombre), nombre)

    def test_rechaza_path_traversal(self):
        peligrosos = [
            "..",
            "../otro",
            "../../etc/passwd",
            "..\\windows",
            "/etc/passwd",
            "C:\\Windows",
            "coleccion/../..",
            "a/b",
            "a\\b",
        ]
        for nombre in peligrosos:
            with self.subTest(nombre=nombre):
                with self.assertRaises(colecciones.NombreInvalido):
                    colecciones.validar_nombre(nombre)

    def test_rechaza_caracteres_raros_y_vacios(self):
        for nombre in ("", " ", "con espacio", "punto.com", "nul\x00o", "ñandu", "-empieza"):
            with self.subTest(nombre=nombre):
                with self.assertRaises(colecciones.NombreInvalido):
                    colecciones.validar_nombre(nombre)

    def test_rechaza_no_cadenas(self):
        for valor in (None, 3, [], {}):
            with self.assertRaises(colecciones.NombreInvalido):
                colecciones.validar_nombre(valor)

    def test_rechaza_nombres_demasiado_largos(self):
        with self.assertRaises(colecciones.NombreInvalido):
            colecciones.validar_nombre("a" * 65)

    def test_las_rutas_quedan_dentro_de_data(self):
        base = Path("/tmp/data").resolve()
        docs, indice = colecciones.rutas("catalogo", data_dir=base)
        self.assertTrue(str(docs.resolve()).startswith(str(base)))
        self.assertTrue(str(indice.resolve()).startswith(str(base)))


class TestColecciones(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_crear_y_listar(self):
        colecciones.crear("uno", data_dir=self.tmp)
        colecciones.crear("dos", data_dir=self.tmp)
        docs, _ = colecciones.rutas("uno", data_dir=self.tmp)
        (docs / "a.txt").write_text("hola", encoding="utf-8")

        listado = colecciones.listar(data_dir=self.tmp)
        nombres = [c["nombre"] for c in listado]
        self.assertEqual(nombres, ["dos", "uno"])
        una = next(c for c in listado if c["nombre"] == "uno")
        self.assertEqual(una["documentos"], ["a.txt"])
        self.assertFalse(una["indexada"])

    def test_listar_ignora_carpetas_con_nombre_invalido(self):
        (self.tmp / "no valida").mkdir()
        colecciones.crear("valida", data_dir=self.tmp)
        nombres = [c["nombre"] for c in colecciones.listar(data_dir=self.tmp)]
        self.assertEqual(nombres, ["valida"])

    def test_listar_sin_data_dir(self):
        self.assertEqual(colecciones.listar(data_dir=self.tmp / "no_existe"), [])

    def test_existe(self):
        colecciones.crear("uno", data_dir=self.tmp)
        self.assertTrue(colecciones.existe("uno", data_dir=self.tmp))
        self.assertFalse(colecciones.existe("otra", data_dir=self.tmp))
        self.assertFalse(colecciones.existe("../fuera", data_dir=self.tmp))

    def test_eliminar(self):
        colecciones.crear("uno", data_dir=self.tmp)
        colecciones.eliminar("uno", data_dir=self.tmp)
        self.assertFalse(colecciones.existe("uno", data_dir=self.tmp))
        with self.assertRaises(colecciones.ColeccionNoEncontrada):
            colecciones.eliminar("uno", data_dir=self.tmp)


class TestCacheBuscadores(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for nombre in ("a", "b", "c"):
            colecciones.crear(nombre, data_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reutiliza_el_buscador_entre_llamadas(self):
        cache = colecciones.CacheBuscadores(maxsize=2, data_dir=self.tmp)
        primero = cache.obtener("a")
        segundo = cache.obtener("a")
        self.assertIs(primero, segundo, "no deberia recargar el indice")

    def test_expulsa_el_menos_usado(self):
        cache = colecciones.CacheBuscadores(maxsize=2, data_dir=self.tmp)
        cache.obtener("a")
        cache.obtener("b")
        cache.obtener("a")  # 'a' pasa a ser la mas reciente
        cache.obtener("c")  # expulsa a 'b', no a 'a'
        self.assertEqual(sorted(cache.cargadas), ["a", "c"])

    def test_invalidar_fuerza_recarga(self):
        cache = colecciones.CacheBuscadores(maxsize=2, data_dir=self.tmp)
        primero = cache.obtener("a")
        cache.invalidar("a")
        self.assertNotIn("a", cache.cargadas)
        self.assertIsNot(primero, cache.obtener("a"))

    def test_coleccion_inexistente(self):
        cache = colecciones.CacheBuscadores(data_dir=self.tmp)
        with self.assertRaises(colecciones.ColeccionNoEncontrada):
            cache.obtener("fantasma")

    def test_nombre_invalido_no_llega_al_disco(self):
        cache = colecciones.CacheBuscadores(data_dir=self.tmp)
        with self.assertRaises(colecciones.NombreInvalido):
            cache.obtener("../../etc")


class TestHistorialSQLite(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.almacen = HistorialSQLite(self.tmp / "historial.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sesion_vacia(self):
        self.assertEqual(self.almacen.cargar("nueva"), [])

    def test_las_sesiones_no_se_mezclan(self):
        self.almacen.añadir_turno("ana", "user", "hola desde ana")
        self.almacen.añadir_turno("luis", "user", "hola desde luis")

        de_ana = self.almacen.cargar("ana")
        self.assertEqual(len(de_ana), 1)
        self.assertEqual(de_ana[0]["text"], "hola desde ana")
        self.assertEqual(len(self.almacen.cargar("luis")), 1)

    def test_conserva_el_orden(self):
        for i in range(10):
            self.almacen.añadir_turno("s", "user" if i % 2 == 0 else "model", f"t{i}")
        textos = [t["text"] for t in self.almacen.cargar("s")]
        self.assertEqual(textos, [f"t{i}" for i in range(10)])

    def test_limite_devuelve_los_ultimos(self):
        for i in range(10):
            self.almacen.añadir_turno("s", "user", f"t{i}")
        textos = [t["text"] for t in self.almacen.cargar("s", limite=3)]
        self.assertEqual(textos, ["t7", "t8", "t9"])

    def test_guardar_reemplaza(self):
        self.almacen.añadir_turno("s", "user", "viejo")
        self.almacen.guardar("s", [{"role": "user", "text": "nuevo"}])
        self.assertEqual([t["text"] for t in self.almacen.cargar("s")], ["nuevo"])

    def test_borrar_solo_afecta_a_su_sesion(self):
        self.almacen.añadir_turno("a", "user", "de a")
        self.almacen.añadir_turno("b", "user", "de b")
        self.almacen.borrar("a")
        self.assertEqual(self.almacen.cargar("a"), [])
        self.assertEqual(len(self.almacen.cargar("b")), 1)

    def test_persiste_entre_instancias(self):
        self.almacen.añadir_turno("s", "user", "persistente")
        otro = HistorialSQLite(self.tmp / "historial.db")
        self.assertEqual(otro.cargar("s")[0]["text"], "persistente")

    def test_uso_desde_varios_hilos(self):
        """El servidor atiende peticiones en varios hilos y SQLite no deja
        compartir conexion entre ellos."""
        import threading

        errores = []

        def escribir(n):
            try:
                for i in range(20):
                    self.almacen.añadir_turno(f"sesion{n}", "user", f"{n}-{i}")
            except Exception as e:
                errores.append(e)

        hilos = [threading.Thread(target=escribir, args=(n,)) for n in range(4)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()

        self.assertEqual(errores, [])
        for n in range(4):
            self.assertEqual(len(self.almacen.cargar(f"sesion{n}")), 20)


class TestHistorialJSON(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ruta = self.tmp / "historial.json"
        self.almacen = HistorialJSON(self.ruta)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ida_y_vuelta(self):
        historial = [{"role": "user", "text": "hola"}]
        self.almacen.guardar(None, historial)
        self.assertEqual(self.almacen.cargar(), historial)

    def test_sin_archivo_devuelve_vacio(self):
        self.assertEqual(self.almacen.cargar(), [])

    def test_json_corrupto_no_revienta(self):
        self.ruta.write_text("{ roto", encoding="utf-8")
        self.assertEqual(self.almacen.cargar(), [])

    def test_borrar(self):
        self.almacen.guardar(None, [{"role": "user", "text": "x"}])
        self.almacen.borrar()
        self.assertFalse(self.ruta.exists())
        self.almacen.borrar()  # borrar dos veces no debe fallar

    def test_ambos_almacenes_cumplen_la_interfaz(self):
        from historial import AlmacenHistorial

        for clase in (HistorialJSON, HistorialSQLite):
            for metodo in ("cargar", "guardar", "borrar"):
                self.assertTrue(hasattr(clase, metodo))
                self.assertIsNot(
                    getattr(clase, metodo),
                    getattr(AlmacenHistorial, metodo),
                    f"{clase.__name__}.{metodo} deberia estar implementado",
                )


if __name__ == "__main__":
    unittest.main()
