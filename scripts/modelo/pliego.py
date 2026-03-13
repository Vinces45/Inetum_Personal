import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DIR_RESPALDO = PROJECT_ROOT / "datos" / "borradores_pliego" / "borrador_actual.json"

class BorradorPliego:
    def __init__(self, archivo_respaldo=DIR_RESPALDO):
        self.archivo_respaldo = archivo_respaldo
        self.secciones = self.cargar_respaldo()
        
    def actualizar_seccion(self, titulo, contenido):
        self.secciones[titulo] = contenido
        self.guardar_respaldo()
        
    def obtener_seccion(self, titulo):
        return self.secciones.get(titulo, None)
        
    def mostrar_documento(self):
        doc = "\n" + "="*50 + "\nBORRADOR ACTUAL DEL PLIEGO\n" + "="*50 + "\n"
        if not self.secciones:
            return doc + "El documento esta vacio.\n" + "="*50
            
        for titulo, texto in self.secciones.items():
            doc += f"\n--- {titulo.upper()} ---\n{texto}\n"
        return doc + "="*50
    
    def guardar_respaldo(self):
        """Serializa el diccionario y lo guarda en disco de forma segura."""
        try:
            with open(self.archivo_respaldo, 'w', encoding='utf-8') as f:
                json.dump(self.secciones, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR PERSISTENCIA] No se pudo guardar el borrador: {e}")

    def cargar_respaldo(self):
        """Lee el archivo JSON del disco para restaurar el estado en caso de caida."""
        if os.path.exists(self.archivo_respaldo):
            try:
                with open(self.archivo_respaldo, 'r', encoding='utf-8') as f:
                    datos_recuperados = json.load(f)
                    print("[SISTEMA] Borrador anterior recuperado con exito.")
                    return datos_recuperados
            except Exception as e:
                print(f"[ERROR PERSISTENCIA] Archivo corrupto, iniciando vacio: {e}")
        return {}
        
    def limpiar_borrador(self):
        """Borra el estado cuando el pliego se da por finalizado y exportado."""
        self.secciones = {}
        if os.path.exists(self.archivo_respaldo):
            os.remove(self.archivo_respaldo)