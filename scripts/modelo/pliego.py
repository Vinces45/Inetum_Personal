import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DIR_RESPALDO = PROJECT_ROOT / "datos" / "borradores_pliego" / "borrador_actual.json"

class NodoSeccion:
    def __init__(self, titulo, contenido=""):
        self.titulo = titulo
        self.contenido = contenido
        self.subsecciones = {} 

    def to_dict(self):
        """Prepara el nodo para ser guardado en JSON."""
        return {
            "contenido": self.contenido,
            "subsecciones": {t: sub.to_dict() for t, sub in self.subsecciones.items()}
        }

    @classmethod
    def from_dict(cls, titulo, datos):
        """Reconstruye el nodo al leer el JSON."""
        nodo = cls(titulo, datos.get("contenido", ""))
        for sub_tit, sub_datos in datos.get("subsecciones", {}).items():
            nodo.subsecciones[sub_tit] = cls.from_dict(sub_tit, sub_datos)
        return nodo

class BorradorPliego:
    def __init__(self, archivo_respaldo=DIR_RESPALDO):
        self.archivo_respaldo = archivo_respaldo
        self.secciones = self.cargar_respaldo()
        
    # def actualizar_seccion(self, titulo_principal, contenido, titulo_subseccion=None):
    #     """Actualiza o crea una seccion, soportando un nivel de subseccion."""
    #     if titulo_principal not in self.secciones:
    #         self.secciones[titulo_principal] = NodoSeccion(titulo_principal)
            
    #     if titulo_subseccion:
    #         nodo_principal = self.secciones[titulo_principal]
    #         nodo_principal.subsecciones[titulo_subseccion] = NodoSeccion(titulo_subseccion, contenido)
    #     else:
    #         self.secciones[titulo_principal].contenido = contenido
            
    #     self.guardar_respaldo()

    def actualizar_seccion_infinita(self, ruta_titulos, contenido):
        """
        Navega por el arbol y guarda el contenido en el ultimo nodo.
        """
        if not ruta_titulos:
            return
            
        titulo_raiz = ruta_titulos[0]
        if titulo_raiz not in self.secciones:
            self.secciones[titulo_raiz] = NodoSeccion(titulo_raiz)
            
        nodo_actual = self.secciones[titulo_raiz]

        for titulo in ruta_titulos[1:]:
            if titulo not in nodo_actual.subsecciones:
                nodo_actual.subsecciones[titulo] = NodoSeccion(titulo)
            nodo_actual = nodo_actual.subsecciones[titulo]
            
        nodo_actual.contenido = contenido
        self.guardar_respaldo()
        
    # def eliminar_seccion(self, titulo):
    #     """Elimina una seccion principal."""
    #     if titulo in self.secciones:
    #         del self.secciones[titulo]
    #         self.guardar_respaldo()
    #         return True
    #     return False

    # def obtener_seccion(self, titulo):
    #     """Obtiene el texto de una seccion principal."""
    #     nodo = self.secciones.get(titulo)
    #     return nodo.contenido if nodo else None
        
    def mostrar_documento(self):
        """Recorre el arbol para mostrar el documento tabulado."""
        if not self.secciones:
            return "El documento esta vacio."
            
        texto_doc = "\n" + "="*50 + "\nBORRADOR ACTUAL\n" + "="*50 + "\n"
        for tit_prin, nodo_prin in self.secciones.items():
            texto_doc += f"\n# {tit_prin.upper()}\n"
            if nodo_prin.contenido:
                texto_doc += f"{nodo_prin.contenido}\n"
                
            for tit_sub, nodo_sub in nodo_prin.subsecciones.items():
                texto_doc += f"\n  ## {tit_sub}\n"
                if nodo_sub.contenido:
                    texto_doc += f"  {nodo_sub.contenido}\n"
                    
        return texto_doc + "\n" + "="*50

    def guardar_respaldo(self):
        """ Guarda el borrador actual para que no se pierda"""
        try:
            datos = {t: nodo.to_dict() for t, nodo in self.secciones.items()}
            with open(self.archivo_respaldo, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR PERSISTENCIA]: {e}")

    def cargar_respaldo(self):
        """Carga el borrador que esta guardado"""
        if os.path.exists(self.archivo_respaldo):
            try:
                with open(self.archivo_respaldo, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                return {t: NodoSeccion.from_dict(t, d) for t, d in datos.items()}
            except Exception:
                pass
        return {}
    
    # def buscar_texto_seccion_recursivo(self, titulo_buscar, nodos=None):
    #     """Busca recursivamente una seccion o subseccion por titulo y devuelve su texto."""
    #     if nodos is None:
    #         nodos = self.secciones
            
    #     for tit, nodo in nodos.items():
    #         if tit.lower() == titulo_buscar.lower():
    #             return nodo.contenido
            
    #         texto_hijo = self.buscar_texto_seccion_recursivo(titulo_buscar, nodo.subsecciones)
    #         if texto_hijo is not None:
    #             return texto_hijo
                
    #     return None
    
    def obtener_rutas_secciones(self, nodos=None, ruta_actual=""):
        """Devuelve una lista con las rutas completas de todas las secciones."""
        if nodos is None:
            nodos = self.secciones
            
        rutas = []
        for tit, nodo in nodos.items():
            nueva_ruta = f"{ruta_actual} > {tit}" if ruta_actual else tit
            rutas.append(nueva_ruta)
            
            rutas.extend(self.obtener_rutas_secciones(nodo.subsecciones, nueva_ruta))
            
        return rutas
    
    def buscar_texto_por_ruta(self, ruta_completa):
        """Busca el texto siguiendo la ruta exacta generada por el orquestador."""
        titulos = [t.strip() for t in ruta_completa.split(">")]
        
        nodos_actuales = self.secciones
        nodo_destino = None
        
        for tit in titulos:
            if tit in nodos_actuales:
                nodo_destino = nodos_actuales[tit]
                nodos_actuales = nodo_destino.subsecciones
            else:
                return None 
                
        return nodo_destino.contenido if nodo_destino else None
    
    def eliminar_por_ruta(self, ruta_completa):
        """Elimina una seccion o subseccion especifica siguiendo su ruta exacta."""
        titulos = [t.strip() for t in ruta_completa.split(">")]
        
        if not titulos:
            return False
            
        if len(titulos) == 1:
            titulo_raiz = titulos[0]
            if titulo_raiz in self.secciones:
                del self.secciones[titulo_raiz]
                self.guardar_respaldo()
                return True
            return False
            
        nodos_actuales = self.secciones
        nodo_padre = None
        
        for tit in titulos[:-1]: 
            if tit in nodos_actuales:
                nodo_padre = nodos_actuales[tit]
                nodos_actuales = nodo_padre.subsecciones
            else:
                return False
                
        titulo_a_borrar = titulos[-1]
        
        if nodo_padre and titulo_a_borrar in nodo_padre.subsecciones:
            del nodo_padre.subsecciones[titulo_a_borrar]
            self.guardar_respaldo()
            return True
            
        return False