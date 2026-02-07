# 🧠 NeuroBot API

**Bot con Redes Neuronales que Aprende Continuamente**

Un API REST inteligente que utiliza redes neuronales (PyTorch) para aprender de cualquier fuente de información — PDFs, documentos Word, páginas web, texto plano — y responder preguntas basándose en su conocimiento acumulado. Mejora con cada interacción gracias a un sistema de feedback con aprendizaje por refuerzo.

---

## 📐 Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                      USUARIO                                │
│   Pregunta / Documento / URL / Texto / Feedback             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    FLASK API                                 │
│  /learn/text  /learn/document  /learn/url  /ask  /feedback   │
└──────────────────────┬───────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
┌──────────────────┐    ┌──────────────────────┐
│  PROCESADORES    │    │   KNOWLEDGE ENGINE   │
│  - PDF (PyMuPDF) │    │                      │
│  - DOCX          │    │  ┌────────────────┐  │
│  - HTML/Web      │    │  │  TF-IDF        │  │
│  - Texto plano   │    │  │  Vectorizer    │  │
│                  │    │  └───────┬────────┘  │
└──────────────────┘    │          │            │
                        │          ▼            │
                        │  ┌────────────────┐  │
                        │  │ Knowledge      │  │
                        │  │ Encoder (NN)   │  │
                        │  │ Autoencoder    │  │
                        │  └───────┬────────┘  │
                        │          │            │
                        │          ▼            │
                        │  ┌────────────────┐  │
                        │  │ Attention      │  │
                        │  │ Ranker (NN)    │  │
                        │  │ + Feedback RL  │  │
                        │  └───────┬────────┘  │
                        │          │            │
                        └──────────┼────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │  SQLite DB       │
                        │  - knowledge     │
                        │  - interactions  │
                        │  - training_log  │
                        └──────────────────┘
```

---

## 🧬 Cómo Funciona la Red Neuronal

### 1. Knowledge Encoder (Autoencoder)
- Recibe vectores TF-IDF (5000 dimensiones) del texto
- Los comprime a embeddings densos de 256 dimensiones
- Entrenado para reconstruir el input (autoencoder)
- Los embeddings capturan el **significado semántico** del texto

### 2. Attention Ranker
- Cuando haces una pregunta, este módulo evalúa qué fragmentos son más relevantes
- Usa **mecanismo de atención** (como en los transformers)
- Se mejora con el **feedback del usuario** (aprendizaje por refuerzo)

### 3. Búsqueda Híbrida
- **70%** similitud coseno entre embeddings
- **30%** scores del attention ranker
- Esto permite que las respuestas mejoren con el tiempo

---

## 🚀 Instalación

```bash
# Clonar o copiar el proyecto
cd learning_bot

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python app.py
```

El servidor estará en: `http://localhost:5000`

---

## 📡 Endpoints

### `POST /learn/text` — Aprender de texto

```bash
curl -X POST http://localhost:5000/learn/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Python es un lenguaje de programación interpretado de alto nivel. Fue creado por Guido van Rossum y lanzado en 1991. Python destaca por su sintaxis limpia y legible.",
    "source": "manual_python",
    "metadata": {"topic": "programación"}
  }'
```

### `POST /learn/document` — Aprender de PDF/Word/TXT

```bash
curl -X POST http://localhost:5000/learn/document \
  -F "file=@manual_tecnico.pdf"

curl -X POST http://localhost:5000/learn/document \
  -F "file=@informe.docx"
```

### `POST /learn/url` — Aprender de página web

```bash
curl -X POST http://localhost:5000/learn/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://es.wikipedia.org/wiki/Inteligencia_artificial"}'
```

### `POST /ask` — Hacer una pregunta

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es Python y quién lo creó?"}'
```

**Respuesta:**
```json
{
  "answer_chunks": [
    {
      "id": "abc123",
      "content": "Python es un lenguaje de programación...",
      "source": "manual_python",
      "relevance_score": 0.8734
    }
  ],
  "confidence": 0.8734,
  "total_knowledge": 42,
  "synthesized_answer": "Python es un lenguaje de programación interpretado..."
}
```

### `POST /feedback` — Mejorar con feedback

```bash
curl -X POST http://localhost:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{"interaction_id": 1, "score": 0.8}'
```

Score: `-1` (terrible) a `1` (excelente). Esto **re-entrena la red neuronal**.

### `GET /stats` — Estadísticas

```bash
curl http://localhost:5000/stats
```

### `GET /knowledge` — Listar conocimiento

```bash
curl "http://localhost:5000/knowledge?page=1&per_page=10&type=pdf"
```

---

## 🏗️ Despliegue en Producción

### Con Gunicorn
```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --timeout 120
```

### Con Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1"]
```

### Variables de Entorno
| Variable | Default | Descripción |
|----------|---------|-------------|
| `PORT` | `5000` | Puerto del servidor |
| `FLASK_DEBUG` | `true` | Modo debug |
| `NEUROBOT_DB` | `neurobot_knowledge.db` | Ruta de la base de datos |

---

## 🔄 Flujo de Aprendizaje

```
1. Usuario envía documento/texto/URL
   ↓
2. Procesador extrae texto limpio
   ↓
3. Texto se divide en chunks inteligentes (por párrafos/oraciones)
   ↓
4. Chunks se almacenan en SQLite
   ↓
5. TF-IDF vectoriza todos los chunks
   ↓
6. Red neuronal (autoencoder) genera embeddings densos
   ↓
7. Cuando el usuario pregunta:
   - Se genera embedding de la pregunta
   - Similitud coseno encuentra chunks relevantes
   - Attention ranker refina el ranking
   - Se sintetiza una respuesta
   ↓
8. Usuario da feedback → Red neuronal se ajusta
   ↓
9. Próximas respuestas son mejores ✨
```

---

## 📋 Notas Técnicas

- **SQLite** como almacenamiento (simple, sin configuración). Para producción pesada, migrar a PostgreSQL.
- **PyTorch** para las redes neuronales (funciona en CPU, no requiere GPU).
- El modelo se re-entrena cada vez que se añade conocimiento nuevo.
- Los modelos se pueden guardar con `POST /save` y persistir entre reinicios.
- El chunking respeta límites de oraciones para mantener coherencia.
- Compatible con español e inglés (TF-IDF con n-gramas).

---

## 🛠️ Próximas Mejoras Posibles

- [ ] Integrar un LLM (como la API de Claude) para generar respuestas más naturales
- [ ] Añadir soporte para imágenes (OCR)
- [ ] Implementar RAG (Retrieval Augmented Generation) completo
- [ ] WebSocket para aprendizaje en tiempo real
- [ ] Panel web de administración
- [ ] Soporte para Excel/CSV
- [ ] Exportar/importar base de conocimiento

---

**Hecho con 🧠 + 🐍 + ⚡ PyTorch**
