import os
import base64
import json
import re
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile
import pymupdf as fitz  
from PIL import Image
from openai import OpenAI

app = FastAPI(title="SGSST PDF Extractor Literal", version="15.0")

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

      # Prompt ajustado con reglas de escaneo horizontal forzado para cuadrículas
        prompt_sistema = """
        Eres un transcriptor de datos OCR literal de altísima precisión. No interpretas ni resumes. Transcribe los campos del certificado respetando estas reglas espaciales restrictivas:
        Devuelve ÚNICAMENTE un JSON válido con estas llaves:
        {
          "nombre_empleado": "Nombre completo del trabajador en minúsculas",
          "tipo_documento": "CC o CE",
          "numero_documento": "Busca ESTRICTAMENTE en la sección 'DATOS DEL PACIENTE' junto a la etiqueta 'IDENTIFICACIÓN:'. Extrae todos los dígitos exactos. NO tomes la identificación de la cabecera superior ni NITs.",
          "empresa_cliente": "Nombre de la empresa",
          "tipo_examen": "Tipo de evaluación",
          "fecha_examen": "Fecha de atención en formato YYYY-MM-DD",
          "concepto_aptitud": "Copia TODO el texto que aparece después de los dos puntos (:) en la línea del concepto. Ignora la etiqueta inicial.",
          "observaciones": "REGLA ESTRICTA: Extrae TODO el texto ubicado físicamente entre 'OBSERVACIONES AL CONCEPTO:' y 'ENFASIS'.",
          "enfasis": "Especialidad médica limpia (ej: osteomuscular).",
          "limitaciones": "Limitaciones indicadas en minúsculas. Si no hay, pon 'ninguna'.",
          "ips_prestador": "Nombre de la IPS prestadora",
          "pruebas_apoyo": "Pruebas diagnósticas realizadas",
          "recomendaciones_medicas": "ESCANEO MULTICOLUMNA OBLIGATORIO: Ubica la sección 'RECOMENDACIONES' -> '» GENERALES'. Los ítems están distribuidos horizontalmente a lo ancho de la página. Debes escanear la imagen de extrema izquierda a extrema derecha. Extrae TODOS los textos que tengan una casilla negra marcada (☑) a su lado. PROHIBIDO detenerse en la primera columna; debes recorrer toda la fila hasta el margen derecho. Sepáralos por comas."
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

        # ---------------------------------------------------------
        # LIMPIEZA DETERMINISTA CON PYTHON (Filtros de Formato)
        # ---------------------------------------------------------

        # 1. Imponer mayúsculas en Empresa y Tipo de Examen
        if "empresa_cliente" in datos_extraidos and isinstance(datos_extraidos["empresa_cliente"], str):
            datos_extraidos["empresa_cliente"] = datos_extraidos["empresa_cliente"].upper()
            
        if "tipo_examen" in datos_extraidos and isinstance(datos_extraidos["tipo_examen"], str):
            datos_extraidos["tipo_examen"] = datos_extraidos["tipo_examen"].upper()

        # 2. Limpiar prefijos basura en Concepto Aptitud
        if "concepto_aptitud" in datos_extraidos and isinstance(datos_extraidos["concepto_aptitud"], str):
            concepto = datos_extraidos["concepto_aptitud"]
            # Detecta y borra frases como "CONCEPTO - EXAMEN PERIODICO:" o "EXAMEN PREINGRESO:"
            concepto = re.sub(r'^(?:CONCEPTO\s*[-–]?\s*)?EXAMEN\s+[A-Z]+\s*:\s*', '', concepto, flags=re.IGNORECASE)
            datos_extraidos["concepto_aptitud"] = concepto.strip()

        # 3. Respaldo de Identificación enfocado SOLO en la sección correcta
        match_cedula_paciente = re.search(r"DATOS DEL PACIENTE.*?IDENTIFICACI[OÓ]N:[\s\n]*(?:CC|CE|TI|NIT|PP)?[\-\.\s]*(\d{5,15})", texto_pdf_nativo, re.IGNORECASE | re.DOTALL)
        
        cedulas_doctores = ["1013609058", "46672834", "46072854", "4607285", "101360905", "1032363717", "554771", "52270442", "830015429", "860006314"]

        if match_cedula_paciente:
            cedula_encontrada = match_cedula_paciente.group(1).strip()
            if cedula_encontrada not in cedulas_doctores:
                num_ia = str(datos_extraidos.get("numero_documento", ""))
                # Si la IA omitió dígitos o difiere del texto digital exacto, sobreescribe
                if len(num_ia) < len(cedula_encontrada) or num_ia != cedula_encontrada:
                    datos_extraidos["numero_documento"] = cedula_encontrada

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
    return {"status": "online", "system": "Extractor GPT-4o Reglas Limpieza v15.0"}
