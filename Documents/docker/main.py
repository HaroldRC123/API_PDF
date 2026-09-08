import os
import base64
import json
import re
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile
import pymupdf as fitz  
from PIL import Image
from openai import OpenAI

app = FastAPI(title="SGSST PDF Extractor Exacto", version="13.0")

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

        # Prompt con reglas de frontera inflexibles
        prompt_sistema = """
        Eres un transcriptor de datos OCR estricto y literal. Tu único objetivo es transcribir exactamente los campos del certificado médico, respetando estas reglas inviolables:
        Devuelve ÚNICAMENTE un JSON válido con estas llaves:
        {
          "nombre_empleado": "Nombre completo del trabajador en minúsculas",
          "tipo_documento": "CC o CE",
          "numero_documento": "Busca en la sección 'DATOS DEL PACIENTE'. Extrae SOLO los números. IGNORA los números en la cabecera superior del documento.",
          "empresa_cliente": "Nombre de la empresa cliente en mayúsculas",
          "tipo_examen": "Tipo de evaluación en minúsculas",
          "fecha_examen": "Fecha de atención en formato YYYY-MM-DD",
          "concepto_aptitud": "Extrae SOLO el valor final, sin la etiqueta. Si el PDF dice 'CONCEPTO-EXAMEN PREINGRESO: CON HALLAZGOS QUE...', tú solo extraes 'con hallazgos que...'.",
          "observaciones": "CRÍTICO: Todo el texto físico que aparece entre 'OBSERVACIONES AL CONCEPTO:' y la palabra 'ENFASIS' pertenece a este campo. Si el médico escribió frases como 'RECOMENDACIONES NUTRICIONALES...' en este espacio, PERTENECEN A OBSERVACIONES. Cópialo todo exactamente como un solo bloque de texto.",
          "enfasis": "Extrae SOLO la especialidad médica (ej: 'osteomuscular', 'visual'). NO incluyas la palabra 'énfasis'.",
          "limitaciones": "Limitaciones o restricciones exactas indicadas",
          "ips_prestador": "Nombre de la IPS prestadora",
          "pruebas_apoyo": "Pruebas diagnósticas realizadas",
          "recomendaciones_medicas": "CRÍTICO: Solo extrae los ítems que aparecen debajo del gran encabezado central 'RECOMENDACIONES' (generalmente marcados con viñetas). Si debajo de 'RECOMENDACIONES' no hay viñetas y solo sigue la sección 'LIMITACIONES', pon 'ninguna'."
        }
        """

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

        # Respaldo en Python anclado a la palabra IDENTIFICACIÓN para evitar capturar basura del encabezado
        match_cedulas_pdf = re.findall(r"IDENTIFICACI[OÓ]N:[\s\n]*(?:CC|CE|TI|NIT|PP)?[\-\.\s]*(\d{6,12})", texto_pdf_nativo, re.IGNORECASE)
        
        # Si no lo encuentra con la etiqueta, busca el patrón general
        if not match_cedulas_pdf:
            match_cedulas_pdf = re.findall(r"(?:CC|CE|TI|NIT|PP)[\-\.\s]*(\d{6,12})", texto_pdf_nativo, re.IGNORECASE)

        cedulas_doctores = ["1013609058", "46672834", "46072854", "4607285", "101360905", "1032363717", "554771", "52270442", "830015429", "860006314"]

        cedula_nativa_valida = None
        for c in match_cedulas_pdf:
            c_limpia = c.strip()
            if c_limpia not in cedulas_doctores:
                cedula_nativa_valida = c_limpia
                break

        if cedula_nativa_valida:
            num_ia = str(datos_extraidos.get("numero_documento", ""))
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
    return {"status": "online", "system": "Extractor GPT-4o Reglas Estrictas v13.0"}
