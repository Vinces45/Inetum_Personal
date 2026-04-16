import os
import json
import glob
import shutil
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DIR_RESPALDO = PROJECT_ROOT / "datos" / "borradores_pliego" / "borrador_actual.json"
DIR_HISTORIAL = PROJECT_ROOT / "datos" / "borradores_pliego" / "historial"

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
    def __init__(self, archivo_respaldo=DIR_RESPALDO, max_versiones = 5):
        self.archivo_respaldo = archivo_respaldo
        
        self.max_versiones = max_versiones
        os.makedirs(DIR_HISTORIAL, exist_ok=True)

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


    def limpiar_borrador(self):
        """Vacia el documento entero y guarda el json vacio."""
        self.secciones = {}
        self.guardar_respaldo()

    def actualizar_desde_texto(self, texto_plano):
        """Parsea el texto plano del text_area y reconstruye el arbol de nodos."""
        self.secciones = {}
        lineas = texto_plano.split('\n')
        
        nodo_prin = None
        nodo_sub = None
        
        for linea in lineas:
            linea_limpia = linea.strip()
            
            # 1. Ignoramos la cabecera visual del documento
            if linea_limpia.startswith('===') or linea_limpia == 'BORRADOR ACTUAL':
                continue
                
            # 2. Detectamos subsecciones (##)
            if linea_limpia.startswith('## '):
                tit_sub = linea_limpia.replace('## ', '').strip()
                if nodo_prin:
                    nodo_prin.subsecciones[tit_sub] = NodoSeccion(tit_sub)
                    nodo_sub = nodo_prin.subsecciones[tit_sub]
                    
            # 3. Detectamos secciones principales (#)
            elif linea.startswith('# '):
                tit_prin = linea.replace('# ', '').strip()
                self.secciones[tit_prin] = NodoSeccion(tit_prin)
                nodo_prin = self.secciones[tit_prin]
                nodo_sub = None
                
            # 4. Es contenido normal
            elif linea_limpia != "":
                if nodo_sub:
                    nodo_sub.contenido += linea_limpia + "\n"
                elif nodo_prin:
                    nodo_prin.contenido += linea_limpia + "\n"
                    
        # 5. Limpiamos saltos de linea sobrantes y guardamos
        for np in self.secciones.values():
            np.contenido = np.contenido.strip()
            for ns in np.subsecciones.values():
                ns.contenido = ns.contenido.strip()
                
        self.guardar_respaldo()


        
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

            self.crear_punto_restauracion()

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


    def crear_punto_restauracion(self):
        """Copia el borrador actual al historial con marca de tiempo."""
        if not os.path.exists(self.archivo_respaldo):
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_version = DIR_HISTORIAL / f"borrador_{timestamp}.json"
        
        shutil.copy2(self.archivo_respaldo, ruta_version)
        self.limpiar_historial()
        

    def limpiar_historial(self):
        """Borra las versiones antiguas manteniendo solo el limite (max_versiones)."""
        # Obtenemos todos los archivos ordenados del mas antiguo al mas nuevo
        archivos = sorted(glob.glob(str(DIR_HISTORIAL / "borrador_*.json")))
        
        while len(archivos) > self.max_versiones:
            archivo_a_borrar = archivos.pop(0) 
            try:
                os.remove(archivo_a_borrar)
            except Exception as e:
                print(f"[ERROR HISTORIAL]: No se pudo borrar {archivo_a_borrar}: {e}")

    def listar_versiones(self):
        """Devuelve las versiones disponibles para que el LLM o usuario las vea."""
        # Obtenemos archivos ordenados del mas NUEVO al mas VIEJO
        archivos = sorted(glob.glob(str(DIR_HISTORIAL / "borrador_*.json")), reverse=True)
        versiones = []
        
        for i, ruta in enumerate(archivos):
            nombre_archivo = os.path.basename(ruta)
            # Extraemos la fecha del string del archivo
            fecha_str = nombre_archivo.replace("borrador_", "").replace(".json", "")
            try:
                fecha_obj = datetime.strptime(fecha_str, "%Y%m%d_%H%M%S")
                fecha_formateada = fecha_obj.strftime("%d/%m/%Y a las %H:%M:%S")
            except ValueError:
                fecha_formateada = "Fecha desconocida"
                
            versiones.append({
                "id": i, # El ID 0 siempre es la version inmediatamente anterior
                "ruta": ruta,
                "fecha": fecha_formateada
            })
            
        return versiones
        
    def restaurar_version(self, id_version):
        """Carga una version anterior y sobreescribe el estado actual."""
        versiones = self.listar_versiones()
        
        if 0 <= id_version < len(versiones):
            ruta_historica = versiones[id_version]["ruta"]
            
            # 1. Sobreescribir el archivo principal con la version historica
            shutil.copy2(ruta_historica, self.archivo_respaldo)
            
            # 2. Recargar el arbol de nodos en memoria
            self.secciones = self.cargar_respaldo()
            
            return True, versiones[id_version]["fecha"]
            
        return False, None