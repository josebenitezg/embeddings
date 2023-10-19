import gradio as gr
from utils import search
from process_dataset import unique_generes

# Define possible genres
genres = unique_generes.tolist()

# Create Interface
iface = gr.Interface(
    fn=search,
    inputs=[
        gr.Textbox(lines=5, placeholder="Escribe aqui tu consulta... ", label="Consulta"),
        gr.Dropdown(choices=genres, label="Género de la pelîcula"),
        gr.Slider(minimum=1, maximum=10, value=5, label="Puntuacion minima"),
        gr.Number(minimu=1, maximum=10, value=3, label="Número de resultados")
    ],
    outputs=gr.Dataframe(type="pandas", label="Resultados"),
    title = "Find your movie 🎞️",
    description = "Best place to find your movie 🍿"
)

iface.launch()
