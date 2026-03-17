import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DIR_RESPALDO = PROJECT_ROOT / "datos" / "borradores_pliego" / "borrador_actual.json"

class NodoSeccion:
    """Implementacion para soportar arboles de secciones."""
    def __init__(self, titulo, contenido=""):
        self.titulo = titulo
        self.contenido = contenido
        self.subsecciones = {}  # Diccionario {titulo_subseccion: NodoSeccion}

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
        
    def actualizar_seccion(self, titulo_principal, contenido, titulo_subseccion=None):
        """Actualiza o crea una seccion, soportando un nivel de subseccion."""
        if titulo_principal not in self.secciones:
            self.secciones[titulo_principal] = NodoSeccion(titulo_principal)
            
        if titulo_subseccion:
            # Insertar en la subseccion
            nodo_principal = self.secciones[titulo_principal]
            nodo_principal.subsecciones[titulo_subseccion] = NodoSeccion(titulo_subseccion, contenido)
        else:
            # Insertar en la seccion principal
            self.secciones[titulo_principal].contenido = contenido
            
        self.guardar_respaldo()
        
    def eliminar_seccion(self, titulo):
        """Elimina una seccion principal."""
        if titulo in self.secciones:
            del self.secciones[titulo]
            self.guardar_respaldo()
            return True
        return False

    def obtener_seccion(self, titulo):
        """Obtiene el texto de una seccion principal."""
        nodo = self.secciones.get(titulo)
        return nodo.contenido if nodo else None
        
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
        try:
            datos = {t: nodo.to_dict() for t, nodo in self.secciones.items()}
            with open(self.archivo_respaldo, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR PERSISTENCIA]: {e}")

    def cargar_respaldo(self):
        if os.path.exists(self.archivo_respaldo):
            try:
                with open(self.archivo_respaldo, 'r', encoding='utf-8') as f:
                    datos = json.load(f)
                return {t: NodoSeccion.from_dict(t, d) for t, d in datos.items()}
            except Exception:
                pass
        return {}