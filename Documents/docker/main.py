import base64
import json
import traceback
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
import fitz    # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance  # <-- SE IMPORTA AQUÍ ARRIBA
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

from PIL import ImageEnhance

        # 1. Abrir el PDF y procesar la única página con alta definición de caracteres numéricos
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(doc) > 0:
            pagina = doc[0]
            
            # Subimos ligeramente a 130 DPI para tener la definición perfecta en los dígitos
            pix = pagina.get_pixmap(dpi=130) 
            
            # Convertimos a escala de grises
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
            
            # --- MEJORA INTELIGENTE DE NÚMEROS (CONTRASTE Y NITIDEZ) ---
            # Esto afila los bordes y evita que los lazos del 6 y el 8 se fusionen o se cierren
            enhancer_contrast = ImageEnhance.Contrast(img)
            img = enhancer_contrast.enhance(1.8) # Aumenta el contraste de los trazos
            
            enhancer_sharpness = ImageEnhance.Sharpness(img)
            img = enhancer_sharpness.enhance(2.0) # Perfila las líneas para distinguir claramente las curvas
            
            # Configuración optimizada de Tesseract para formularios (--psm 6)
            custom_config = r'--oem 3 --psm 6'
            
            texto_crudo = pytesseract.image_to_string(img, config=custom_config)
            texto_limpio = texto_crudo.replace("|", "")
        else:
            raise HTTPException(status_code=400, detail="El PDF está vacío o corrupto.")
        # 2. Extracciones con expresiones regulares (Actualizadas y Definitivas)
        
# ESTRATEGIA 1: Buscar la cédula en "DATOS DEL PACIENTE"
# 2. Extracciones con expresiones regulares (Blindadas y Definitivas)
        
        # ESTRATEGIA 1: Buscar la cédula estrictamente en la sección superior "DATOS DEL PACIENTE"
        match_cedula = re.search(
            r"DATOS\s+DEL\s+PACIENTE.*?IDENTIFICACI[OÓ]N:?\s*[\n\|\s]*(CC|CE)[-\.\s]*(\d+)", 
            texto_limpio, 
            re.DOTALL | re.IGNORECASE
        )
        
        # ESTRATEGIA 2: Si no está arriba, buscar en el identificador inicial "ID: CC-..."
        if not match_cedula:
            match_cedula = re.search(r"ID:\s*(CC|CE)[-\.\s]*(\d+)", texto_limpio, re.IGNORECASE)

        if match_cedula:
            tipo_documento = match_cedula.group(1).strip().upper()
            numero_documento = match_cedula.group(2).strip()
        else:
            # Fallback seguro excluyendo categóricamente las cédulas de los médicos conocidos
            todas_cedulas = re.findall(r"(CC|CE)[-\.\s]*(\d+)", texto_limpio, re.IGNORECASE)
            cedulas_doctores = ["1013609058", "46672834", "46072854"] 
            cedulas_validas = [c for c in todas_cedulas if c[1] not in cedulas_doctores and len(c[1]) > 6]
            
            if cedulas_validas:
                tipo_documento = cedulas_validas[0][0].strip().upper()
                numero_documento = cedulas_validas[0][1].strip()
            else:
                tipo_documento = "No encontrado"
                numero_documento = "No encontrado"

        # Nombre del empleado (con soporte para saltos de línea del OCR)
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

        # --- CORREGIDO: Observaciones (Soporta ENFASIS con o sin tilde) ---
        match_observaciones = re.search(r"OBSERVACIONES AL CONCEPTO:\s*(.*?)(?=ENFASIS|ÉNFASIS|RECOMENDACIONES|LIMITACIONES|TIPO LIMITACI[OÓ]N|> GENERALES|$)", texto_limpio, re.DOTALL | re.IGNORECASE)
        if match_observaciones:
            observaciones = match_observaciones.group(1).replace('\n', ' ').strip()
            observaciones = re.sub(r'\s+', ' ', observaciones)
        else:
            observaciones = "No encontrado"

       # --- EXTRACCIÓN BLINDADA DEL ÉNFASIS ---
        match_enfasis = re.search(r"(?:É|E)NFASIS(?:\s+EN)?\s*[-:]?\s*([A-ZÁÉÍÓÚ]+)", texto_limpio, re.IGNORECASE)
        
        if match_enfasis and match_enfasis.group(1).upper() != "EN":
            enfasis = match_enfasis.group(1).strip().upper()
        else:
            # Fallback inteligente: Busca directamente los énfasis médicos comunes en todo el texto
            posibles_enfasis = ["OSTEOMUSCULAR", "VISUAL", "VOZ", "ALTURAS", "NEUROLOGICO", "AUDITIVO"]
            enfasis = "No especificado"
            for item in posibles_enfasis:
                if item in texto_limpio.upper():
                    enfasis = item
                    break

        # Limitaciones
        match_limitaciones = re.search(r"OBSERVACIÓN:\s*([^\n]+)", texto_limpio)
        limitaciones = match_limitaciones.group(1).strip() if match_limitaciones else "NINGUNA"

        # IPS Prestador
        match_ips = re.search(r"(SALUD OCUPACIONAL SANITAS SAS)", texto_limpio, re.IGNORECASE)
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
