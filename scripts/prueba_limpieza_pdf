import PyPDF2
from pathlib import Path
from reportlab.pdfgen import canvas

#Obtencion de ruta
directorio_actual = Path(__file__).parent
ruta_pdf = directorio_actual.parent / 'datos' / 'prueba1.pdf'
ruta_pdf_guardar = directorio_actual.parent / 'datos' / 'prueba2.pdf'


#Abrir un PDF

try:
    with open(ruta_pdf, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_paginas = len(pdf_reader.pages)
        
        for pagina in range(int(num_paginas-num_paginas)):
            print("--------------------PAGINA",pagina,"---------------------")
            objeto_pagina = pdf_reader.pages[pagina]
            print(objeto_pagina.extract_text())

    #     obj_pag = pdf_reader.pages[3]
    #     print(obj_pag.extract_text())

    new_pdf = canvas.Canvas(str(ruta_pdf_guardar))
    new_pdf.drawString(100, 750, "Hello World")
    new_pdf.save()

except Exception as e:
    print("Excepción: " + str(e))
