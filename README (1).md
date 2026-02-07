# 🧠 NeuroBot API v2.0 — PostgreSQL Edition

Bot con redes neuronales que aprende de documentos, webs y texto. Diseñado para Render free tier.

## Despliegue en Render (paso a paso)

### 1. Sube a GitHub
```bash
git init
git add .
git commit -m "NeuroBot API v2.0"
git remote add origin https://github.com/TU_USER/neurobot-api.git
git push -u origin main
```

### 2. Crea la base de datos PostgreSQL en Render
1. Ve a https://dashboard.render.com
2. **New** → **PostgreSQL**
3. Name: `neurobot-db`
4. Plan: **Free**
5. Crea y copia el **Internal Database URL**

### 3. Crea el Web Service
1. **New** → **Web Service** → conecta tu repo
2. Configura:
   - **Build Command:** `chmod +x build.sh && ./build.sh`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
3. **Environment Variables:**
   - `DATABASE_URL` = la URL que copiaste del paso 2
   - `FLASK_DEBUG` = `false`

> **Alternativa:** Si usas `render.yaml`, Render detecta todo automáticamente con **New** → **Blueprint** → selecciona tu repo.

### 4. ¡Listo! Prueba con:
```bash
# Health check
curl https://tu-app.onrender.com/health

# Enseñarle algo
curl -X POST https://tu-app.onrender.com/learn/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Python fue creado por Guido van Rossum en 1991. Es un lenguaje interpretado de alto nivel.", "source": "manual"}'

# Preguntarle
curl -X POST https://tu-app.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Quién creó Python?"}'

# Subir PDF
curl -X POST https://tu-app.onrender.com/learn/document \
  -F "file=@mi_documento.pdf"

# Aprender de web
curl -X POST https://tu-app.onrender.com/learn/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://es.wikipedia.org/wiki/Inteligencia_artificial"}'

# Dar feedback (mejora el modelo)
curl -X POST https://tu-app.onrender.com/feedback \
  -H "Content-Type: application/json" \
  -d '{"interaction_id": 1, "score": 0.9}'

# Guardar modelo en BD (persiste entre reinicios)
curl -X POST https://tu-app.onrender.com/save

# Ver estadísticas
curl https://tu-app.onrender.com/stats
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Info del API |
| GET | `/health` | Health check |
| POST | `/learn/text` | Aprender de texto |
| POST | `/learn/document` | Aprender de PDF/DOCX/TXT |
| POST | `/learn/url` | Aprender de página web |
| POST | `/ask` | Hacer una pregunta |
| POST | `/feedback` | Dar feedback (-1 a 1) |
| GET | `/stats` | Estadísticas |
| GET | `/knowledge` | Listar conocimiento |
| POST | `/save` | Guardar modelo en PostgreSQL |
| POST | `/load` | Cargar modelo de PostgreSQL |

## Notas importantes

- **Los pesos del modelo se guardan EN PostgreSQL** (no en disco), así persisten entre reinicios de Render
- La BD gratuita de Render expira a los 90 días — recuerda renovarla
- El plan gratuito duerme el servicio tras 15 min de inactividad
- PyTorch funciona en CPU (no requiere GPU)
