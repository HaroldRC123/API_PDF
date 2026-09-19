import os
import base64
import json
import re
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile
import fitz  # PyMuPDF
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="SGSST PDF Extractor Estructurado", version="17.0")

client = OpenAI()

# 1. Definimos el esquema estricto usando Pydantic (Structured Outputs)
class CertificadoMedico(BaseModel):
    nombre_empleado: str
    tipo_documento: str
    numero_documento: str
    empresa_cliente: str
    tipo_examen: str
    fecha_examen: str
    concepto_aptitud: str
    observaciones: str
    enfasis: str
    limitaciones: str
    ips_prestador: str
    pruebas_apoyo: str
    recomendaciones_medicas: str

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

        # 2. Extracción Híbrida: Sacamos el texto nativo para ayudar al LLM
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

        prompt_sistema = """
        Eres un transcriptor de datos OCR literal de altísima precisión. No interpretas ni resumes.
        REGLA ESTRICTA PARA RECOMENDACIONES: El documento tiene un formato multicolumna. Para evitar perder datos, apóyate en el texto extraído nativamente que se te proporciona. Extrae TODOS los textos que tengan una casilla negra marcada (☑) a su lado. Sepáralos por comas. Todo en minúsculas.
        """

        # 3. Llamada a la API usando el método 'parse' para forzar la salida estructurada
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06", # Modelo óptimo para Structured Outputs
            messages=[
                {
                    "role": "system",
                    "content": prompt_sistema
                },
                {
                    "role": "user",
                    "content": [
                        # Le pasamos la imagen de alta calidad
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        # Le pasamos el texto nativo extraído por PyMuPDF como "muleta" espacial
                        {
                            "type": "text",
                            "text": f"--- INICIO TEXTO NATIVO DE APOYO ---\n{texto_pdf_nativo}\n--- FIN TEXTO NATIVO ---\nUtiliza este texto para verificar que no saltaste ninguna columna en las recomendaciones."
                        }
                    ]
                }
            ],
            response_format=CertificadoMedico,
            temperature=0.0
        )

        # 4. Volcamos la respuesta validada a un diccionario de Python (sin hacer json.loads)
        datos_extraidos = response.choices[0].message.parsed.model_dump()

        # ---------------------------------------------------------
        # LIMPIEZA DETERMINISTA CON PYTHON (Filtros de Formato)
        # ---------------------------------------------------------

        if "empresa_cliente" in datos_extraidos and isinstance(datos_extraidos["empresa_cliente"], str):
            datos_extraidos["empresa_cliente"] = datos_extraidos["empresa_cliente"].upper()
            
        if "tipo_examen" in datos_extraidos and isinstance(datos_extraidos["tipo_examen"], str):
            datos_extraidos["tipo_examen"] = datos_extraidos["tipo_examen"].upper()

        if "nombre_empleado" in datos_extraidos and isinstance(datos_extraidos["nombre_empleado"], str):
            datos_extraidos["nombre_empleado"] = datos_extraidos["nombre_empleado"].title()

        if "concepto_aptitud" in datos_extraidos and isinstance(datos_extraidos["concepto_aptitud"], str):
            concepto = datos_extraidos["concepto_aptitud"]
            concepto = re.sub(r'^(?:CONCEPTO\s*[-–]?\s*)?EXAMEN\s+[A-Z]+\s*:\s*', '', concepto, flags=re.IGNORECASE)
            datos_extraidos["concepto_aptitud"] = concepto.strip().lower()

        if "observaciones" in datos_extraidos and isinstance(datos_extraidos["observaciones"], str):
            datos_extraidos["observaciones"] = datos_extraidos["observaciones"].strip().lower()
            
        if "recomendaciones_medicas" in datos_extraidos and isinstance(datos_extraidos["recomendaciones_medicas"], str):
            datos_extraidos["recomendaciones_medicas"] = datos_extraidos["recomendaciones_medicas"].strip().lower()

        if "fecha_examen" in datos_extraidos and isinstance(datos_extraidos["fecha_examen"], str):
            fecha_cruda = datos_extraidos["fecha_examen"].replace("/", "-").strip()
            match_yyyy = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", fecha_cruda)
            if match_yyyy:
                datos_extraidos["fecha_examen"] = f"{match_yyyy.group(3)}-{match_yyyy.group(2)}-{match_yyyy.group(1)}"
            else:
                datos_extraidos["fecha_examen"] = fecha_cruda

        # Respaldo de Identificación (Tu regex sigue siendo el mecanismo más seguro)
        match_cedula_paciente = re.search(r"DATOS DEL PACIENTE.*?IDENTIFICACI[OÓ]N:[\s\n]*(?:CC|CE|TI|NIT|PP)?[\-\.\s]*(\d{5,15})", texto_pdf_nativo, re.IGNORECASE | re.DOTALL)
        cedulas_doctores = ["1013609058", "46672834", "46072854", "4607285", "101360905", "1032363717", "554771", "52270442", "830015429", "860006314"]

        if match_cedula_paciente:
            cedula_encontrada = match_cedula_paciente.group(1).strip()
            if cedula_encontrada not in cedulas_doctores:
                num_ia = str(datos_extraidos.get("numero_documento", ""))
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
    return {"status": "online", "system": "Extractor Estructurado Híbrido v17.0"}
