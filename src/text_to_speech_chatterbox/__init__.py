import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando el modelo...")
modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

texto = "Hola, Bienvenidos a Inteligencia Artificial generativa vamos a crear imagenes, texto, audio y si el computador lo permite y aguanta, crearemos videos.  "

print("Generando el audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7, #Expresividad o emoción
    cfg_weight=0.4, #Afecta velocidad o ritmo
    temperature=0.8, #Controla la variabilidad
    repetition_penalty=1.2, # Penaliza repeticiones extrañas
    min_p=0.05, # filtra opciones improbables durante la generación
    top_p=1.0 #Controla el muestreo probabilistico
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)
print("Audio generado.")
