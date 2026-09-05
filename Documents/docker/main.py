import os
import base64
import json
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI

app = FastAPI(title="SGSST PDF Extractor API con GPT-4o-mini", version="11.0")

# Inicializamos el cliente de OpenAI (tomará automáticamente la API Key de las variables de entorno de Render)
client = OpenAI()

@app.post("/api/procesar-examen/")
async def procesar_examen(request: Request, file: UploadFile = None):
    try:
        pdf_bytes = None

        # OPCIÓN 1: Si se envía como Form-Data / UploadFile (Recomendado para Power Automate)
        if file is not None:
            pdf_bytes = await file.read()
            print("Archivo recibido por Form-Data (UploadFile).")
        else:
            # Obtenemos el cuerpo crudo de la solicitud como bytes
            body_bytes = await request.body()
            if not body_bytes or len(body_bytes) == 0:
                raise HTTPException(status_code=400, detail="El cuerpo de la solicitud llegó vacío.")
            
            # OPCIÓN 2: Intentar interpretar el cuerpo como un JSON de Power Automate {"$content": "..."}
            try:
                body_json = json.loads(body_bytes.decode('utf-8'))
                if isinstance(body_json, dict):
                    file_content_base64 = body_json.get("$content") or body_json.get("content")
                    if file_content_base64:
                        pdf_bytes = base64.b64decode(file_content_base64)
            except Exception:
                pass
                
            # OPCIÓN 3: Si no vino en JSON ni Form-Data, asumimos binario puro
            if pdf_bytes is None:
                pdf_bytes = body_bytes
                print("Archivo binario directo detectado.")

        if not pdf_bytes or len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="El contenido del archivo PDF está vacío.")

        # 1. Abrir el PDF y rasterizar la primera página a imagen (150 DPI para máxima nitidez visual)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(doc) > 0:
            pagina = doc[0]
            pix = pagina.get_pixmap(dpi=150) 
            
            # Guardamos temporalmente en el contenedor de Render para enviarla a OpenAI
            img_path = "/tmp/certificado_temp.png"
            pix.save(img_path)
        else:
            raise HTTPException(status_code=400, detail="El PDF está vacío o corrupto.")

        # 2. Codificar la imagen generada a Base64
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 3. Prompt estructurado y experto para GPT-4o-mini Vision
        prompt_sistema = """
        Eres un asistente experto en auditoría de salud ocupacional (SGSST) en Colombia. 
        Analiza la imagen de este certificado médico ocupacional de Sanitas y extrae la información requerida 
        devolviendo ÚNICAMENTE un objeto JSON válido (sin formato de bloques markdown ni texto adicional), estructurado exactamente con estas llaves:
        {
          "nombre_empleado": "Nombre completo del trabajador en minúsculas",
          "tipo_documento": "CC o CE",
          "numero_documento": "Número de cédula del paciente (ATENCIÓN: Ignora las cédulas de los médicos firmantes al pie de página, extrae estrictamente la del paciente de la sección superior)",
          "empresa_cliente": "Nombre de la empresa cliente en mayúsculas",
          "tipo_examen": "Tipo de evaluación en minúsculas (ej: periodico, preingreso)",
          "fecha_examen": "Fecha de atención en formato YYYY-MM-DD",
          "concepto_aptitud": "Texto exacto de la etiqueta de concepto de aptitud en minúsculas (ej: con restricciones para la labor)",
          "observaciones": "Texto completo y 100% íntegro de 'OBSERVACIONES AL CONCEPTO' en minúsculas. ADVERTENCIA: Captura todo el párrafo clínico completo de principio a fin, aun si contiene palabras como 'énfasis' o 'enfasis' a mitad de texto.",
          "enfasis": "Énfasis médico limpio en minúsculas (ej: osteomuscular, visual)",
          "limitaciones": "Limitaciones o restricciones indicadas en minúsculas (si no hay, coloca 'ninguna')",
          "ips_prestador": "Nombre de la IPS prestadora en minúsculas",
          "pruebas_apoyo": "Lista separada por comas de las pruebas diagnósticas realizadas (ej: audiometria, optometria) en minúsculas. NO incluyas encabezados fijos de tablas.",
          "recomendaciones_medicas": "Lista separada por comas de todas las recomendaciones marcadas o enlistadas en la sección 'RECOMENDACIONES' (ej: examen periodico ocupacional, continuar manejo medico, pausas activas, higiene postural) en minúsculas."
        }
        """

        # 4. Solicitud al modelo multimodal de OpenAI
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

        # 5. Procesamiento y limpieza del JSON devuelto por la IA
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
    return {"status": "online", "system": "Extractor OCR con GPT-4o-mini activo v11.0"}
