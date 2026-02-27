# XorIA-cortex: A General Purpose xLSTM and MinGRU

[Versión en Español abajo](#español)

XorIA-cortex is a high-performance implementation of the **xLSTM (Extended Long Short-Term Memory)** architecture, developed in Rust using the **Burn** deep learning framework. This project features a highly efficient MinGRU and functions as a highly versatile text generator for a wide range of use cases.

## Key Features

*   **xLSTM Architecture**: Implements both **sLSTM** (scalar LSTM with exponential gating) and **mLSTM** (matrix LSTM with covariance memory).
*   **Flexible Blocks**: Support for stacking blocks in sLSTM, mLSTM, or alternating patterns to capture different sequence dependencies.
*   **Burn Framework**: Leverages Rust's safety and performance with multiple backend support (NdArray, WGPU/Vulkan).
*   **Memory Optimized**: Efficient handling of the computation graph during validation and inference to prevent memory leaks.

## Getting Started / Compilations & Tests

**Compilations & Chats:**
```bash
cargo build --release --bin mainchat
cargo run --release --bin mlstmchat -- input.txt
cargo run --release --bin xlstmchat -- input.txt
cargo run --release --bin slstmchat -- input.txt
cargo run --release --bin mingruchat -- input.txt
```

**Stability Tests:**
```bash
cargo run --bin test_slstm
cargo run --bin test_slstm -- stability
cargo run --bin test_slstm -- monotonic
cargo run --bin test_slstm -- compare_lstm
cargo test test_mlstm_dual_equivalence -- --nocapture
```

## Text Generation and Efficiency: Shakespeare Output

This xLSTM implementation, featuring a highly efficient MinGRU, has proven incredibly capable as a generalized text generator for a wide range of applications.

### The mLSTM Triumph: Shakespeare & Convergence (Previous Run)

In a previous training session focused on **BPE-tokenized** text generation (Shakespeare dataset), the **mLSTM architecture** with only **2 blocks** achieved state-of-the-art results for this project scale:

*   **Accuracy**: **73.07%** (Phenomenal for sub-word level modeling).
*   **Loss**: **0.9877** (The "Holy Grail" threshold, breaking below 1.0).
*   **Final Status**: Cinematic `STATUS_CONTROL_C_EXIT` after total convergence.

#### The Lore
The model reached a level of understanding so profound that it decided to "retire" at the peak of its knowledge. Its last generated sequence before the exit was:
> *"MISTRESS OVERDONE: Heting art in the world a businesses with a lovely kiss!"*

This proves that matrix-based memory (**mLSTM**) is extremely efficient at capturing complex linguistic structures and character identities with minimal depth (only 2 blocks).

### 30-Epoch Run: Focus on Grammar (Current Run)

In a more recent test on **Linux**, we continued training with the **Shakespeare dataset** using the same configuration of **2 mLSTM layers with a hidden size of 256**. 

The training was manually stopped at **epoch 13** of the current run (adding up to **17 previous epochs**, for a total of **30 epochs**). 

**Final Metrics at 30 epochs:**
*   **Accuracy**: **~69.53%**
*   **Loss**: **~1.1086**

### Focus on Grammatical Quality over Memorization
While the model could have continued training for even lower loss and higher accuracy, it was decided to stop at 30 epochs to avoid **overfitting**. The priority is to achieve high **grammatical quality** with a touch of creative hallucination, ensuring the model generates completely correct structures without making up non-existent words, rather than simply memorizing the dataset verbatim. 

### The sLSTM Breakthrough: Superior Accuracy & Convergence

In recent experiments with the **sLSTM architecture**, we observed a massive performance leap. Even at early stages, sLSTM significantly outperformed the mLSTM variants:

*   **Epoch 30 (Significant Benchmark)**: Already reached **89.43% Accuracy** and **0.4225 Loss**, far surpassing the mLSTM's performance at the same stage.
*   **Epoch 50 (Final Result)**: Reached a staggering **96.73% Accuracy** and **0.12 Loss**.
*   **Performance**: The sLSTM demonstrated a much faster and more stable learning curve, achieving near-perfect coherence compared to previous architectures.

---

<a name="español"></a>
# XorIA-cortex: Un xLSTM y MinGRU para propósito general

XorIA-cortex es una implementación de alto rendimiento de la arquitectura **xLSTM (Extended Long Short-Term Memory)**, desarrollada en Rust utilizando el framework de deep learning **Burn**. Este proyecto incluye una MinGRU muy eficiente y funciona como un generador de texto altamente versátil para un amplio abanico de aplicaciones.

## Características Principales

*   **Arquitectura xLSTM**: Implementa tanto **sLSTM** (LSTM escalar con compuertas exponenciales) como **mLSTM** (LSTM matricial con memoria de covarianza).
*   **Bloques Flexibles**: Soporte para apilar bloques en patrones sLSTM, mLSTM o alternados para capturar diferentes dependencias de secuencias.
*   **Framework Burn**: Aprovecha la seguridad y el rendimiento de Rust con soporte para múltiples backends (NdArray, WGPU/Vulkan).
*   **Memoria Optimizada**: Gestión eficiente del grafo de computación durante la validación e inferencia para evitar fugas de memoria.

## Cómo empezar / Compilaciones y Tests de Estabilidad

**Compilaciones y Chats:**
```bash
cargo build --release --bin mainchat
cargo run --release --bin mlstmchat -- input.txt
cargo run --release --bin xlstmchat -- input.txt
cargo run --release --bin slstmchat -- input.txt
cargo run --release --bin mingruchat -- input.txt
```

**Tests de Estabilidad:**
```bash
cargo run --bin test_slstm
cargo run --bin test_slstm -- stability
cargo run --bin test_slstm -- monotonic
cargo run --bin test_slstm -- compare_lstm
cargo test test_mlstm_dual_equivalence -- --nocapture
```

## Generación de Texto y Eficiencia: Resultados con Shakespeare

Esta implementación de xLSTM, que incluye una MinGRU muy eficiente, ha demostrado ser increíblemente versátil como generador de texto aplicable a un amplio abanico de casos de uso.

### El Triunfo de mLSTM: Shakespeare y Convergencia (Prueba Anterior)

En una sesión de entrenamiento anterior enfocada en la generación de texto con **tokenización BPE** (dataset de Shakespeare), la arquitectura **mLSTM** con solo **2 bloques** logró resultados asombrosos para la escala de este proyecto:

*   **Precisión (Accuracy)**: **73.07%** (Fenomenal para modelado a nivel de sub-palabras).
*   **Pérdida (Loss)**: **0.9877** (Rompiendo la barrera mística del 1.0).
*   **Estado Final**: `STATUS_CONTROL_C_EXIT` cinematográfico tras la convergencia total.

#### El Lore
El modelo alcanzó un nivel de comprensión tan profundo que decidió "retirarse" en la cima de su conocimiento. Su última secuencia generada antes del cierre fue:
> *"MISTRESS OVERDONE: Heting art in the world a businesses with a lovely kiss!"*

Esto demuestra que la memoria basada en matrices (**mLSTM**) es extremadamente eficiente para capturar estructuras lingüísticas complejas e identidades de personajes con una profundidad mínima (solo 2 bloques).

### Prueba a 30 Épocas: Enfoque en Gramática (Prueba Actual)

En una prueba reciente en **Linux**, continuamos entrenando el modelo con el **dataset de Shakespeare** utilizando una configuración de **2 capas mLSTM de 256 de dimensión oculta (hidden size)**.

El entrenamiento se detuvo manualmente en la **época 13** de la ejecución actual (sumada a **17 épocas anteriores**, haciendo un total de **30 épocas** para este resultado final).

**Métricas Finales a las 30 épocas:**
*   **Precisión (Accuracy)**: **~69.53%**
*   **Pérdida (Loss)**: **~1.1086**

### Prioridad a la Calidad Gramatical sobre la Memorización
Aunque el modelo estaba en condiciones de seguir mejorando sus métricas, se decidió detener el entrenamiento a las 30 épocas para no entrar en **overfitting** (sobreajuste). El objetivo priorizado fue mantener la **calidad gramatical** y permitir cierta "alucinación" creativa, siendo muy correcto sintácticamente y sin inventar palabras, evitando así que el modelo simplemente memorizara el dataset.

### El Gran Avance de sLSTM: Precisión Superior y Convergencia

En pruebas recientes con la **arquitectura sLSTM**, observamos un salto masivo de rendimiento. Incluso en etapas tempranas, sLSTM superó significativamente a las variantes mLSTM:

*   **Época 30 (Hito Significativo)**: Ya alcanzaba el **89.43% de Precisión** y **0.4225 de Pérdida**, superando por mucho el rendimiento de mLSTM en la misma etapa.
*   **Época 50 (Resultado Final)**: Alcanzó un asombroso **96.73% de Precisión** y **0.12 de Pérdida**.
*   **Rendimiento**: sLSTM demostró una curva de aprendizaje mucho más rápida y estable, logrando una coherencia casi perfecta en comparación con arquitecturas anteriores.

## License / Licencia

**English**: The improvements and modifications made in this fork are dual-licensed under the **MIT License** and **Apache License 2.0**, following the Rust ecosystem standards. We encourage the original creator to define a global license for the project.

**Español**: Las mejoras y modificaciones realizadas en este fork tienen una licencia doble **MIT** y **Apache 2.0**, siguiendo los estándares del ecosistema de Rust. Se anima al creador original a definir una licencia global para el proyecto.



---

**xLSTM Python to Rust Port:** [github.com/emanuelbertey/xlstm](https://github.com/emanuelbertey/xlstm/tree/main)  
**Port de xLSTM de Python a Rust:** [github.com/emanuelbertey/xlstm](https://github.com/emanuelbertey/xlstm/tree/main)

---

**Official xLSTM Python Repository:** [github.com/NX-AI/xlstm](https://github.com/NX-AI/xlstm)

If you use this codebase, or otherwise find our work valuable, please cite the xLSTM paper:

```bibtex
@inproceedings{beck:24xlstm,
  title = {xLSTM: Extended Long Short-Term Memory}, 
  author = {Maximilian Beck and Korbinian Pöppel and Markus Spanring and Andreas Auer and Oleksandra Prudnikova and Michael Kopp and Günter Klambauer and Johannes Brandstetter and Sepp Hochreiter},
  booktitle = {Thirty-eighth Conference on Neural Information Processing Systems},
  year = {2024},
  url = {https://arxiv.org/abs/2405.04517}, 
}

@article{beck:25xlstm7b,
  title = {{xLSTM 7B}: A Recurrent LLM for Fast and Efficient Inference},
  author = {Maximilian Beck and Korbinian Pöppel and Phillip Lippe and Richard Kurle and Patrick M. Blies and Günter Klambauer and Sebastian Böck and Sepp Hochreiter},
  booktitle = {Forty-second International Conference on Machine Learning},
  year = {2025},
  url = {https://arxiv.org/abs/2503.13427}
}
```

---

### Personal Project / Proyecto Personal: **Laurelia**
For a more advanced implementation focused on LLMs and Chatbots using the **Hugging Face Candle** framework, check out:
Para una implementación más avanzada enfocada en LLMs y Chatbots usando el framework **Candle** de Hugging Face, visita:

**[Laurelia (emanuelbertey/LaurelIA)](https://github.com/emanuelbertey/LaurelIA.git)**
*Uses advanced optimization techniques such as fused projections, matrix memory normalization, and parallel kernels.*
*Utiliza técnicas avanzadas de optimización como proyecciones fusionadas, normalización de memoria matricial y kernels paralelos.*
