from pathlib import Path
import pymupdf4llm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ruta_pdf = PROJECT_ROOT / "datos" / "pdfs" / "ppt"/ "03_PPT_TechFablab.pdf"
ruta_salida = PROJECT_ROOT / "datos" / "archivos_markdown" / "analisis_visual.md"

print("Extrayendo con modelo avanzado de layout...")
texto_md = pymupdf4llm.to_markdown(ruta_pdf)

with open(ruta_salida, "w", encoding="utf-8") as archivo_md:
    archivo_md.write(texto_md)
    
print(f"Archivo guardado exitosamente en {ruta_salida}. ¡Abrelo y echale un ojo!")