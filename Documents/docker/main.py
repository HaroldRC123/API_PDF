import base64
import json
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
import fitz    # PyMuPDF
import pytesseract
from PIL import Image
import re


app = FastAPI(title="SGSST PDF Extractor API con OCR", version="9.3")

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
                        print("Formato JSON con Base64 detectado y decodificado correctamente.")
            except Exception:
                pass
                
            # OPCIÓN 3: Si no vino en JSON ni Form-Data, asumimos binario puro
            if pdf_bytes is None:
                pdf_bytes = body_bytes
                print("Archivo binario directo detectado.")

        if not pdf_bytes or len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="El contenido del archivo PDF está vacío.")

        # 1. Leer el archivo y renderizar imágenes a 300 DPI con PyMuPDF (en memoria)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texto_completo = ""
        
        for pagina in doc:
            pix = pagina.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            texto_completo += pytesseract.image_to_string(img) + "\n"
            
        texto_limpio = texto_completo.replace("|", "")

       # 2. Extracciones con expresiones regulares (Actualizadas)
        
        # PASO CLAVE: Aislar la sección "DATOS DEL PACIENTE" para evitar capturar la cédula del médico
        match_seccion_paciente = re.search(
            r"DATOS DEL PACIENTE(.*?)(?:DATOS DEL CLIENTE|DATOS DE LA ATENCIÓN|CERTIFICADO|$)", 
            texto_limpio, 
            re.DOTALL | re.IGNORECASE
        )
        texto_paciente = match_seccion_paciente.group(1) if match_seccion_paciente else texto_limpio

        # Buscar la cédula estrictamente dentro del bloque exclusivo del paciente
        match_cedula = re.search(r"(CC|CE)[-\s]*(\d+)", texto_paciente)
        if not match_cedula:
            # Fallback de respaldo por si el encabezado varía en algún documento
            match_cedula = re.search(r"(CC|CE)[-\s]*(\d+)", texto_limpio)

        tipo_documento = match_cedula.group(1).strip() if match_cedula else "No encontrado"
        numero_documento = match_cedula.group(2).strip() if match_cedula else "No encontrado"

        match_nombre = re.search(r"NOMBRE:\s*([A-ZÑ\s]{5,})\n", texto_limpio)
        if match_nombre and "IDENTIFICACI" not in match_nombre.group(1):
            nombre = match_nombre.group(1).strip()
        else:
            # Intento 2: Formato en bloque/vertical
            match_nombre_alt = re.search(r"SEXO:?\s*\n([A-ZÑ\s]+)\n(?:CC|CE)", texto_limpio)
            nombre = match_nombre_alt.group(1).strip() if match_nombre_alt else "No encontrado"

        match_empresa = re.search(r"EDAD:\n\d+ AÑOS\n([A-Z\s]+)\n", texto_limpio)
        empresa = match_empresa.group(1).strip() if match_empresa else "AGENCE FRANCE PRESSE"

        match_tipo = re.search(r"TIPO DE EVALUACION:\s*([A-Z]+)", texto_limpio)
        tipo_examen = match_tipo.group(1).strip() if match_tipo else "No encontrado"

        match_fecha = re.search(r"FECHA DE ATENCI[OÓ]N[^\d]*([\d]{4}[-/][\d]{2}[-/][\d]{2})", texto_limpio)
        fecha_examen = match_fecha.group(1).strip() if match_fecha else "No encontrado"

        match_concepto = re.search(r"CONCEPTO[^:]*:\s*([^\n]+)", texto_limpio)
        concepto = match_concepto.group(1).strip() if match_concepto else "No encontrado"

        # --- NUEVO: EXTRACCIÓN DE OBSERVACIONES MULTILÍNEA ---
        # Busca hacia abajo hasta encontrarse con alguna de las palabras clave de freno
        match_observaciones = re.search(r"OBSERVACIONES AL CONCEPTO:\s*(.*?)(?=ENFASIS|RECOMENDACIONES|LIMITACIONES|TIPO LIMITACI[OÓ]N|> GENERALES|$)", texto_limpio, re.DOTALL | re.IGNORECASE)
        if match_observaciones:
            # Quitamos los saltos de línea para que quede un solo párrafo limpio
            observaciones = match_observaciones.group(1).replace('\n', ' ').strip()
            observaciones = re.sub(r'\s+', ' ', observaciones) # Borra espacios dobles
        else:
            observaciones = "No encontrado"

        # --- NUEVO: EXTRACCIÓN DEL ÉNFASIS ---
        # --- CORRECCIÓN: EXTRACCIÓN SEGURA DEL ÉNFASIS ---
        # Exigimos un guion (-) o dos puntos (:) obligatorios después de la palabra ÉNFASIS
        # para evitar capturar texto plano dentro de las observaciones.
        match_enfasis = re.search(
            r"ENFASIS\s*[-:]\s*([A-ZÁÉÍÓÚ]+)", texto_limpio, re.IGNORECASE
        )
        enfasis = (
            match_enfasis.group(1).strip().upper()
            if match_enfasis
            else "No especificado"
        )

        match_limitaciones = re.search(r"OBSERVACIÓN:\s*([^\n]+)", texto_limpio)
        limitaciones = match_limitaciones.group(1).strip() if match_limitaciones else "NINGUNA"

        match_ips = re.search(r"(SALUD OCUPACIONAL SANITAS SAS)", texto_limpio)
        ips_prestador = match_ips.group(1).strip() if match_ips else "No encontrado"
        
        lista_examenes = [
            "AUDIOMETRIA", "OPTOMETRIA", "VISIOMETRIA", "ESPIROMETRIA",
            "ELECTROCARDIOGRAMA", "LABORATORIO CLÍNICO", "LABORATORIO CLINICO",
            "PSICOLOGIA", "RAYOS X", "CUADRO HEMATICO"
        ]
        
        pruebas_encontradas = []
        for examen in lista_examenes:
            if re.search(r"\b" + examen + r"\b", texto_limpio, re.IGNORECASE):
                if examen == "LABORATORIO CLÍNICO": examen = "LABORATORIO CLINICO"
                if examen not in pruebas_encontradas:
                    pruebas_encontradas.append(examen)
                    
        pruebas_apoyo = ", ".join(pruebas_encontradas) if pruebas_encontradas else "Ninguna registrada"
        
        lista_recomendaciones = [
            "EXAMEN PERIODICO OCUPACIONAL", "EXAMEN PERIÓDICO OCUPACIONAL",
            "PAUSAS ACTIVAS", "HIGIENE POSTURAL", "USO DE EPP",
            "HABITOS SALUDABLES", "CONTROL DE PESO", "CORRECCION VISUAL"
        ]
        
        recom_encontradas = []
        for recom in lista_recomendaciones:
            if re.search(r"\b" + recom.replace("Ó", "[OÓ]").replace("Í", "[IÍ]") + r"\b", texto_limpio, re.IGNORECASE):
                nombre_limpio = recom.replace("Ó", "O").replace("Í", "I")
                if nombre_limpio not in recom_encontradas:
                    recom_encontradas.append(nombre_limpio)
                    
        recomendaciones_medicas = ", ".join(recom_encontradas) if recom_encontradas else "Ninguna"
# Función auxiliar para pasar a minúsculas de forma segura
        def min(val):
          return val.lower() if isinstance(val, str) else val
        return {
            "status": "ok",
            "bytes_recibidos": len(pdf_bytes),
            "datos_extraidos": {
                "nombre_empleado": min(nombre),
                "tipo_documento": tipo_documento,  # CC o CE se suelen dejar en mayúscula, pero puedes usar min(tipo_documento) si deseas
                "numero_documento": numero_documento,
                "mpresa_cliente": empresa.upper(),
                "tipo_examen": min(tipo_examen),
                "fecha_examen": fecha_examen,
                "concepto_aptitud": min(concepto),
                "observaciones": min(observaciones),
                "enfasis": min(enfasis),
                "limitaciones": min(limitaciones),
                "ips_prestador": min(ips_prestador),
                "pruebas_apoyo": min(pruebas_apoyo),
                "recomendaciones_medicas": min(recomendaciones_medicas),
            },
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
    return {"status": "online", "system": "Extractor OCR final activo v9.3"}    
