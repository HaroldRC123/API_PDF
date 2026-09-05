import os
import base64
import json
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile
import pymupdf as fitz  # Importación actualizada recomendada
from PIL import Image
from openai import OpenAI

app = FastAPI(title="SGSST PDF Extractor con GPT-4o-mini", version="11.1")

# Inicializamos el cliente de OpenAI
client = OpenAI()

@app.post("/api/procesar-examen/")
async def procesar_examen(request: Request, file: UploadFile = None):
    try:
        pdf_bytes = None

        # 1. Recepción del archivo (Soporta Form-Data, JSON Base64 o Binario Puro)
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

        # 2. Abrir el PDF y rasterizar a 250 DPI (Garantiza definición perfecta de números)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(doc) > 0:
            pagina = doc[0]
            # Subimos a 250 DPI para que los dígitos (8, 3, 0, etc.) no se fusionen visualmente
            pix = pagina.get_pixmap(dpi=250) 
            
            img_path = "/tmp/certificado_temp.png"
            pix.save(img_path)
        else:
            raise HTTPException(status_code=400, detail="El PDF está vacío o corrupto.")

        # 3. Codificar la imagen a Base64
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 4. Prompt ultra-reforzado con atención especial en el número de documento
        prompt_sistema = """
        Eres un auditor experto en seguridad y salud ocupacional (SGSST) en Colombia. 
        Analiza la imagen de este certificado médico ocupacional de Sanitas con extrema precisión en los números 
        de identificación. Devuelve ÚNICAMENTE un objeto JSON válido (sin bloques markdown ni texto adicional) con estas llaves:
        {
          "nombre_empleado": "Nombre completo del trabajador en minúsculas",
          "tipo_documento": "CC o CE",
          "numero_documento": "Número exacto de la cédula o cédula de extranjería del paciente. ADVERTENCIA CRÍTICA: Lee dígito por dígito con sumo cuidado para no omitir ningún número (como los ceros o los ochos dobles). Ignora las cédulas de los médicos firmantes al pie de página.",
          "empresa_cliente": "Nombre de la empresa cliente en mayúsculas",
          "tipo_examen": "Tipo de evaluación en minúsculas (ej: periodico, preingreso)",
          "fecha_examen": "Fecha de atención en formato YYYY-MM-DD",
          "concepto_aptitud": "Texto exacto de la etiqueta de concepto de aptitud en minúsculas",
          "observaciones": "Texto completo y 100% íntegro de 'OBSERVACIONES AL CONCEPTO' en minúsculas, sin omitir partes",
          "enfasis": "Énfasis médico limpio en minúsculas (ej: osteomuscular, visual)",
          "limitaciones": "Limitaciones o restricciones indicadas en minúsculas (si no hay, coloca 'ninguna')",
          "ips_prestador": "Nombre de la IPS prestadora en minúsculas",
          "pruebas_apoyo": "Lista separada por comas de las pruebas diagnósticas realizadas en minúsculas",
          "recomendaciones_medicas": "Lista separada por comas de todas las recomendaciones de la sección 'RECOMENDACIONES' en minúsculas"
        }
        """

        # 5. Solicitud a GPT-4o-mini
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
            max_tokens=900,
            temperature=0.0
        )

        # 6. Procesamiento y limpieza del JSON
        contenido_respuesta = response.choices[0].message.content.strip()
        
        if contenido_respuesta.startswith("```"):
            contenido_respuesta = contenido_respuesta.split("```")[1]
            if contenido_respuesta.startswith("json"):
                contenido_respuesta = contenido_respuesta[4:]
        contenido_respuesta = contenido_respuesta.strip()

        datos_extraidos = json.loads(contenido_respuesta)

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
    return {"status": "online", "system": "Extractor OCR con GPT-4o-mini activo v11.1"}
