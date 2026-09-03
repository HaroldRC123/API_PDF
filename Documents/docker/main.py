import os
# CRUCIAL: Evita que Tesseract sature la CPU gratuita de Render intentando usar multi-hilos
os.environ["OMP_THREAD_LIMIT"] = "1"

import base64
import json
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
import fitz    # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance
import re

app = FastAPI(title="SGSST PDF Extractor API con OCR", version="9.4")

@app.post("/api/procesar-examen/")
async def procesar_examen(request: Request, file: UploadFile = None):
    try:
        pdf_bytes = None

        if file is not None:
            pdf_bytes = await file.read()
            print("Archivo recibido por Form-Data (UploadFile).")
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
                print("Archivo binario directo detectado.")

        if not pdf_bytes or len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="El contenido del archivo PDF está vacío.")

        # 1. Abrir el PDF y procesar la única página directamente en escala de grises
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(doc) > 0:
            pagina = doc[0]
            
            # Renderizamos directamente a escala de grises a 120 DPI (equilibrio perfecto para leer 6 y 8 sin pesar demasiado)
            pix = pagina.get_pixmap(dpi=200, colorspace=fitz.csGRAY) 
            img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
            
            # --- MEJORA RÁPIDA DE CONTRASTE ---
            # Realce optimizado para afilar los trazos de los números sin sobrecargar el script
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # Configuración optimizada de Tesseract (--psm 6 para bloques de formularios)
            custom_config = r'--oem 3 --psm 6'
            
            texto_crudo = pytesseract.image_to_string(img, config=custom_config)
            texto_limpio = texto_crudo.replace("|", "")
        else:
            raise HTTPException(status_code=400, detail="El PDF está vacío o corrupto.")

        # 2. Extracciones con expresiones regulares (Blindadas y Definitivas)
        
        # Extracción segura de la cédula del paciente (excluyendo médicos)
        todas_cedulas = re.findall(r"(CC|CE)[-\.\s]*(\d+)", texto_limpio, re.IGNORECASE)
        cedulas_doctores = ["1013609058", "46672834", "46072854", "4607285", "101360905"] 
        
        cedulas_validas = []
        for tipo, num in todas_cedulas:
            num_limpio = num.strip()
            if num_limpio not in cedulas_doctores and len(num_limpio) > 6:
                cedulas_validas.append((tipo.strip().upper(), num_limpio))
        
        if cedulas_validas:
            tipo_documento = cedulas_validas[0][0]
            numero_documento = cedulas_validas[0][1]
        else:
            tipo_documento = "No encontrado"
            numero_documento = "No encontrado"

        # Nombre del empleado
        match_nombre = re.search(r"NOMBRE:\s*[\n\|\s]*([A-ZÑ\s]{5,})\n", texto_limpio)
        if match_nombre and "IDENTIFICACI" not in match_nombre.group(1):
            nombre = match_nombre.group(1).strip()
        else:
            match_nombre_alt = re.search(r"SEXO:?\s*\n([A-ZÑ\s]+)\n(?:CC|CE)", texto_limpio)
            nombre = match_nombre_alt.group(1).strip() if match_nombre_alt else "No encontrado"

        # Empresa cliente
        match_empresa = re.search(r"EDAD:\s*\d+\s*AÑOS\s*([A-Z\s]+)\n", texto_limpio)
        if not match_empresa:
            match_empresa = re.search(r"NOMBRE:\s*([A-Z\s]+)\s*DATOS DE LA ATENCIÓN", texto_limpio, re.DOTALL)
        empresa = match_empresa.group(1).strip() if match_empresa else "AGENCE FRANCE PRESSE"

        # Tipo de evaluación
        match_tipo = re.search(r"TIPO DE EVALUACION:?\s*[\n\|\s]*([A-ZÁÉÍÓÚ]+)", texto_limpio, re.IGNORECASE)
        tipo_examen = match_tipo.group(1).strip() if match_tipo else "No encontrado"

        # Fecha de atención
        match_fecha = re.search(r"FECHA DE ATENCI[OÓ]N[^\d]*([\d]{4}[-/][\d]{2}[-/][\d]{2})", texto_limpio)
        fecha_examen = match_fecha.group(1).strip() if match_fecha else "No encontrado"

        # Concepto de aptitud
        match_concepto = re.search(r"CONCEPTO[^:]*:\s*([^\n]+)", texto_limpio)
        concepto = match_concepto.group(1).strip() if match_concepto else "No encontrado"

        # Observaciones
        match_observaciones = re.search(r"OBSERVACIONES AL CONCEPTO:\s*(.*?)(?=ENFASIS|ÉNFASIS|RECOMENDACIONES|LIMITACIONES|TIPO LIMITACI[OÓ]N|> GENERALES|$)", texto_limpio, re.DOTALL | re.IGNORECASE)
        if match_observaciones:
            observaciones = match_observaciones.group(1).replace('\n', ' ').strip()
            observaciones = re.sub(r'\s+', ' ', observaciones)
        else:
            observaciones = "No encontrado"

        # Énfasis
        match_enfasis = re.search(r"(?:É|E)NFASIS(?:\s+EN)?\s*[-:]?\s*([A-ZÁÉÍÓÚ]+)", texto_limpio, re.IGNORECASE)
        if match_enfasis and match_enfasis.group(1).upper() != "EN":
            enfasis = match_enfasis.group(1).strip().upper()
        else:
            posibles_enfasis = ["OSTEOMUSCULAR", "VISUAL", "VOZ", "ALTURAS", "NEUROLOGICO", "AUDITIVO"]
            enfasis = "No especificado"
            for item in posibles_enfasis:
                if item in texto_limpio.upper():
                    enfasis = item
                    break
        
        if enfasis.startswith("EN "):
            enfasis = enfasis[3:]
        elif enfasis.startswith("EN"):
            enfasis = enfasis[2:]

        # Limitaciones
        match_limitaciones = re.search(r"OBSERVACIÓN:\s*([^\n]+)", texto_limpio)
        limitaciones = match_limitaciones.group(1).strip() if match_limitaciones else "NINGUNA"

        # IPS Prestador
        match_ips = re.search(r"(SALUD OCUPACIONAL SANITAS SAS)", texto_limpio, re.IGNORECASE)
        ips_prestador = match_ips.group(1).strip() if match_ips else "No encontrado"
        
       # Lista de exámenes de apoyo diagnóstico reales (excluyendo encabezados de tabla)
        lista_examenes = [
            "AUDIOMETRIA", "OPTOMETRIA", "VISIOMETRIA", "ESPIROMETRIA",
            "ELECTROCARDIOGRAMA", "PSICOLOGIA", "RAYOS X", "CUADRO HEMATICO"
        ]
        
        pruebas_encontradas = []
        for examen in lista_examenes:
            if re.search(r"\b" + examen + r"\b", texto_limpio, re.IGNORECASE):
                if examen not in pruebas_encontradas:
                    pruebas_encontradas.append(examen)
                    
        pruebas_apoyo = ", ".join(pruebas_encontradas) if pruebas_encontradas else "Ninguna registrada"
        
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

        def min_seguro(val):
            return val.lower() if isinstance(val, str) else val

        return {
            "status": "ok",
            "bytes_recibidos": len(pdf_bytes),
            "datos_extraidos": {
                "nombre_empleado": min_seguro(nombre),
                "tipo_documento": tipo_documento,
                "numero_documento": numero_documento,
                "empresa_cliente": empresa.upper(),
                "tipo_examen": min_seguro(tipo_examen),
                "fecha_examen": fecha_examen,
                "concepto_aptitud": min_seguro(concepto),
                "observaciones": min_seguro(observaciones),
                "enfasis": min_seguro(enfasis),
                "limitaciones": min_seguro(limitaciones),
                "ips_prestador": ips_prestador.upper(),
                "pruebas_apoyo": min_seguro(pruebas_apoyo),
                "recomendaciones_medicas": min_seguro(recomendaciones_medicas),
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
    return {"status": "online", "system": "Extractor OCR final activo v9.4"}
