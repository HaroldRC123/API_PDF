import os
import base64
import json
import re
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile
import pymupdf as fitz  
from PIL import Image
from openai import OpenAI

app = FastAPI(title="SGSST PDF Extractor Exacto", version="12.0")

client = OpenAI()

@app.post("/api/procesar-examen/")
async def procesar_examen(request: Request, file: UploadFile = None):
    try:
        pdf_bytes = None

        if file is not None:
            pdf_bytes = await file.read()
        else:
            body_bytes = await request.body()
            if not body_bytes or len(body_bytes) == 0:
                raise HTTPException(status_code=400, detail="El cuerpo de la solicitud llegó vacío.")
            
            try:
                body_json = json.loads(body_bytes.decode('utf-8'))
                if isinstance(body_json, dict):
                    file_content_base64 = body_json.get("$content") or body_json.get("content")
                    if file_content_base64:
                        pdf_bytes = base64.b64decode(file_content_base64)
            except Exception:
                pass
                
            if pdf_bytes is None:
                pdf_bytes = body_bytes

        if not pdf_bytes or len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="El contenido del archivo PDF está vacío.")

        # 1. Extracción de texto nativo con PyMuPDF (Respaldo)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texto_pdf_nativo = ""
        for pagina_doc in doc:
            texto_pdf_nativo += pagina_doc.get_text()

        if len(doc) > 0:
            pagina = doc[0]
            pix = pagina.get_pixmap(dpi=250) 
            img_path = "/tmp/certificado_temp.png"
            pix.save(img_path)
        else:
            raise HTTPException(status_code=400, detail="El PDF está vacío o corrupto.")

        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 2. Prompt ultra-estricto anti-resumen
        prompt_sistema = """
        Eres un transcriptor clínico estricto. Analiza el certificado de salud ocupacional de Sanitas. 
        REGLA DE ORO: NO resumas ni abrevies ningún campo. Copia y pega el texto EXACTO que ves en la imagen.
        Devuelve ÚNICAMENTE un JSON válido con estas llaves:
        {
          "nombre_empleado": "Nombre completo del trabajador en minúsculas",
          "tipo_documento": "CC o CE",
          "numero_documento": "Número exacto del paciente. LÉELO DÍGITO POR DÍGITO SIN OMITIR NINGUNO.",
          "empresa_cliente": "Nombre de la empresa cliente en mayúsculas",
          "tipo_examen": "Tipo de evaluación en minúsculas",
          "fecha_examen": "Fecha de atención en formato YYYY-MM-DD",
          "concepto_aptitud": "Copia LITERALMENTE el texto del concepto. NO uses la palabra 'apto' a menos que el documento diga explícitamente solo 'apto'. Si dice 'con hallazgos que requieren medidas...', copia todo eso de forma exacta.",
          "observaciones": "Copia el texto completo e íntegro del bloque de observaciones, sin acortarlo.",
          "enfasis": "Énfasis médico limpio (ej: osteomuscular)",
          "limitaciones": "Limitaciones o restricciones exactas indicadas",
          "ips_prestador": "Nombre de la IPS prestadora",
          "pruebas_apoyo": "Pruebas diagnósticas realizadas",
          "recomendaciones_medicas": "Lista completa de las recomendaciones marcadas"
        }
        """

        # 3. Cambio al modelo principal GPT-4o (Alta precisión)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_sistema},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.0
        )

        contenido_respuesta = response.choices[0].message.content.strip()
        if contenido_respuesta.startswith("```"):
            contenido_respuesta = contenido_respuesta.split("```")[1]
            if contenido_respuesta.startswith("json"):
                contenido_respuesta = contenido_respuesta[4:]
        contenido_respuesta = contenido_respuesta.strip()

        datos_extraidos = json.loads(contenido_respuesta)

        # 4. Respaldo por código para extracción de la cédula del texto digital
        match_cedulas_pdf = re.findall(r"(?:CC|CE|TI|NIT|PP)[\-\.\s]*(\d{6,12})", texto_pdf_nativo, re.IGNORECASE)
        # Se incluye la lista negra ampliada con los médicos recurrentes
        cedulas_doctores = ["1013609058", "46672834", "46072854", "4607285", "101360905", "1032363717", "554771", "52270442"]

        cedula_nativa_valida = None
        for c in match_cedulas_pdf:
            c_limpia = c.strip()
            if c_limpia not in cedulas_doctores:
                cedula_nativa_valida = c_limpia
                break

        if cedula_nativa_valida:
            num_ia = str(datos_extraidos.get("numero_documento", ""))
            # Si hay disparidad (ej: omisión de un dígito por la IA), prioriza la lectura del texto digital nativo
            if len(num_ia) < len(cedula_nativa_valida) or num_ia != cedula_nativa_valida:
                datos_extraidos["numero_documento"] = cedula_nativa_valida

        return {
            "status": "ok",
            "bytes_recibidos": len(pdf_bytes),
            "datos_extraidos": datos_extraidos
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        error_detallado = traceback.format_exc()
        print("--- ERROR INTERNO ---")
        print(error_detallado)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "online", "system": "Extractor GPT-4o Precisión Alta v12.0"}
