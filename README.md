# Quadruped AI

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Box2D](https://img.shields.io/badge/Box-2.3.10-red.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-red.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="assets/logo.png" alt="logo">
</p>


## 📝 Project Description
This project is a try to understand how to use box2D with pygame. 🦊🦊🦊

To do so, I construct an AI able to control a quadruped, a fox with real physics, muscles, a world and a pretty low-poly design.

To learn data visualisation, I used Power BI to analyse in details !

🚨The project is not **finish** !🚨

---

## ⚙️ Features

Constructed :
- Real physique with muscles, interation with **box2D** library.
- A good-looking with pygame🦊.
- An algorithm to select the best choreography.

Project for the futur :
- A genetic algorithm
- PPO algorithm
- Add some commentary for each .py on the top for a better understanding of my code for AI (and me 🫠)


## Example Outputs

We can control the quadruped, the view (We can see clearly the parallax and the different mode - textured, skeleton and overlay) :
<p align="center">
  <img src="assets/Gif-human-controled.gif" alt="Example Outputs : Human controlled">
</p>

Here is the algorithm that select just the best choreography :
<p align="center">
  <img src="assets/Gif-select-choregraphy.gif" alt="Example Outputs : Select best choreography">
</p>

I'm currently working on other algorithm such as genetic neural network, PPO...

---

## ⚙️ How it works

Here it is juste a selection of the best choreography and adjusting time in consequence.



## 🗺️ Schema

![Overview](old_version/powerBI/img.png)

<details>
<summary>📸 See more data analyse</summary>

![Capture 1](old_version/powerBI/img_1.png)
![Capture 2](old_version/powerBI/img_2.png)
![Capture 3](old_version/powerBI/img_3.png)

</details>

---

## 📂 Repository structure
```bash
├── test1_physique.py
├── test2_physique.py
│
├── LICENSE
├── README.md
```

---

## 💻 Run it on Your PC
Clone the repository and install dependencies:
```bash
git clone https://github.com/Thibault-GAREL/test_box2D_pygame.git
cd Quadruped-AI

python -m venv .venv #if you don't have a virtual environnement
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install box2D pygame mlflow pydantic-settings pandas numpy

python main.py
```

---

## 🚀 Train the neuro-GA (tuned hyperparams)

Best effort hyperparams to escape early convergence (PowerShell, Windows):
```powershell
$env:NEURO_GA_POPULATION_SIZE       = "80"
$env:NEURO_GA_MUTATION_RATE         = "0.15"
$env:NEURO_GA_MUTATION_STRENGTH     = "0.35"
$env:NEURO_GA_ELITE_SIZE            = "3"
$env:NEURO_GA_TOURNAMENT_SIZE       = "5"
$env:NEURO_GA_CROSSOVER_RATE        = "0.75"
$env:NEURO_GA_REWARD_THRESHOLD_FOR_MAX_TIME = "3000"
$env:NEURO_GA_MAX_GENERATIONS       = "100000"
$env:NEURO_GA_AUTO_CONTINUE         = "False"

python main.py
```

Force a fresh from-scratch training (skip the loaded checkpoint):
```powershell
Move-Item outputs/models/fox_ai_neuro.pkl outputs/models/fox_ai_neuro.pkl.bak -Force
```

Kill a running training without killing your MLflow UI:
```powershell
Get-Process | Where-Object { $_.CommandLine -like "*main.py*" } | Stop-Process
```

Check progress of the latest run (works while training is live):
```powershell
python progress.py
```
Outputs : run name, status, duration, generations done, best fitness, gens/min.

---

## 📊 Visualize results with MLflow

Launch the UI (from the project root, with a venv that has mlflow):
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://localhost:5000 in your browser.

**Find the best runs** :
1. Open experiment `quadruped-neuro-ga`
2. Click the `Metrics` column header to add `best_distance_ever` (or `fitness_best`)
3. Sort by `best_distance_ever` descending → top row is your best run
4. Click any run → tab **Model metrics** → see fitness curve over generations

**Compare runs side by side** :
1. Check 2+ runs in the table
2. Click **Compare**
3. Tab **Parallel Coordinates** → see hyperparams vs final fitness

**Best saved model file** is at:
```
outputs/models/<name>_run-XX_date-YYYY-MM-DD/best_model.pkl
```

---

## 📖 Inspiration / Sources
I code it without any help 😆 !

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)