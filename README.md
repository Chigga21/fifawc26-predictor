# WC26 PREDICTOR

<p align="center">
  <img src="static/fifa.png" alt="FIFA" width="600"/>
</p>

**WC26 Predictor** es un proyecto personal de Machine Learning orientado a la predicción de partidos internacionales, con especial enfoque en la *Copa Mundial* de la FIFA 2026. A partir del dataset [**international_results**](https://github.com/martj42/international_results), se estima la fuerza ofensiva y defensiva de cada selección mediante el modelo Dixon-Coles. Así, se entrenan dos modelos de goles esperados:

- Un modelo MCMC Bayesiano
- Un modelo de regresión con XGBoost

y convierte esas predicciones en una matriz de probabilidades de `Poisson` para calcular las probabilidades de victoria local, empate y victoria visitante (1X2).

## Ejecución

Crea y activa un entorno virtual:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Instala las dependencias y ejecuta la interfaz interactiva:

```bash
pip install -r requirements.txt
python main.py
```

Si así lo decides, puedes graficar los resultados con pyplot que posteriormente se almacenarán en `outputs/`.

Si tu terminal no soporta colores ANSI puedes usar el comando:

```bash
NO_COLOR=1 python main.py
```

> [!ADVERTENCIA]
> **Este proyecto fue desarrollado con fines educativos y de investigación.** Aunque el modelo puede ofrecer predicciones interesantes, **no se recomienda utilizarlo para realizar apuestas deportivas**, ya que ningún modelo puede predecir un partido con certeza y el fútbol contiene una gran cantidad de factores impredecibles.
>

<p align="center">
  <img src="static/apuesta.jpg" alt="Apuesta" width="500"/>
</p>

Dicho eso, personalmente ya se me ha dado más de una apuesta verdecita gracias a este programa. Tómalo como una anécdota y no como una garantía de resultados futuros.

## Testing

El proyecto cuenta con una *testsuite* automatizada que puedes ejecutar con:

```bash
pytest -v
```
