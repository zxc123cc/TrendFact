<h1 align="center">TrendFact: Un punto de referencia para la percepción de puntos calientes en la verificación automática de hechos</h1>

<p align="center">
  <a href="https://aclanthology.org/2026.acl-long.1219/"><img src="https://img.shields.io/badge/Paper-ACL%202026-b31b1b?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/zxc123cc/TrendFact"><img src="https://img.shields.io/badge/Dataset-HuggingFace-ffd21e?style=for-the-badge&logo=huggingface&logoColor=black" alt="Dataset"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-4c1?style=for-the-badge" alt="License"></a>
</p>


## Resumen general
<p align="center">
  <img src="figures/Comparison.png" width="90%" alt="Comparación de la asimetría del riesgo">
</p>

<p align="center">
  <img src="figures/TrendFact.png" width="90%" alt="Resumen general de TrendFact">
</p>

## FactISR

<p align="center">
  <img src="figures/FactISR.png" width="90%" alt="Marco de trabajo FactISR">
</p>

## Métricas

Hemos publicado la implementación de nuestras dos métricas propuestas, ECS y HCPI, en el directorio `metrics/`.

### ECS (Puntuación de Consistencia de Explicaciones)

ECS utiliza un modelo de lenguaje (LLM) como juez para evaluar la consistencia de una explicación generada con respecto a la explicación de referencia. Primero, debes configurar tus credenciales de API mediante variables de entorno:

```bash
export OPENAI_API_KEY=tu_clave
export OPENAI_API_BASE=https://api.openai.com/v1   # opcional
export OPENAI_MODEL=gpt-4o-2024-11-20              # opcional
```

```bash
python metrics/cal_ECS.py --input_file results.json --output_file results_ECS.json
```

El archivo de entrada es una lista JSON donde cada elemento debe contener al menos los siguientes campos: `claim`, `explanation` (la explicación de referencia) y la salida del modelo (`llm_response`, o `llm_response_parse` / `llm_think` + `llm_response`).

### HCPI (Índice de Percepción de Reclamos de Puntos Calientes)

HCPI combina la puntuación de influencia con ECS para medir la capacidad de percepción de puntos calientes. Toma como entrada el archivo de resultados de ECS generado previamente:

```bash
python metrics/cal_HCPI.py --input_file results_ECS.json
```

## Cita
```bibtex
@inproceedings{zhang2026trendfact,
  title={TrendFact: Un punto de referencia para la percepción de puntos calientes en la verificación automática de hechos},
  author={Zhang, Xiaocheng and Wang, Xi and Lu, Yifei and Wang, Jianing and Ye, Zhuangzhuang and Bao, Mengjiao and Yan, Peng and Su, Xiaohong},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={26494--26513},
  year={2026}
}
```
