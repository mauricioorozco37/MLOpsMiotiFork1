
Instalar WSL2:

```wsl --install -d Ubuntu```


Instalar Miniconda en Ubuntu:


```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Crear el entorno del proyecto:

```
conda create -n masterclass_mlops python=3.10 -y
conda activate masterclass_mlops
pip install mlflow dvc dvc-s3 pandas scikit-learn fastapi uvicorn
```

Inicialización Git y DVC
```
git init
dvc init
git commit -m "Iniciando prroyecto MLOps"

```
Podemos crear el árbol de carpetas de nuestro experimento y el .gitignore
````
diabetes_mlops_masterclass/

├── data/                # CREAR
│   ├── raw/             # NO CREAR Dataset original (diabetes.csv controlado por DVC)
│   └── processed/       # NO CREAR Datos tras limpieza o ingeniería de características
├── src/
│── mlruns/ 
├── models/              # Almacén de binarios de modelos (.pkl) controlados por DVC
├── notebooks/           # Jupyter Notebooks para exploración inicial (EDA)
├
├
└── README.md            # Documentación general del proyecto
````

Configuración de repositorio remoto (en este caso lo hacemos local)
```
mkdir -p /tmp/dvc_storage
dvc remote add -d local_remote /tmp/dvc_storage
git add.dvc/config
git commit -m "Configurado storage remoto -local- de DVC"
```
Creamos un script src/load_data.py
```
import pandas as pd
from sklearn.datasets import load_diabetes
import os

def download_data():
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/diabetes.csv", index=False)
    print("Datos guardados en data/raw/diabetes.csv")

if __name__ == "__main__":
    download_data()
```

Ingestamos y versionamos los datos
```
python src/load_data.py
dvc add data/raw/diabetes.csv
git add data/raw/diabetes.csv.dvc .gitignore
git commit -m "Version 1 del dataset de diabetes"
dvc push
```
Nota: Cualquier persona que haga git checkout de este último commit y ejecute ``` dvc pull ``` tendrá exactamente el mismo CSV que hemos registrado en este commit

Creamos src/train.py. Usaremos la función autolog() de MLflow
```
import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

mlflow.set_experiment("Diabetes_Prediction")
mlflow.sklearn.autolog()

def run_training():
    df = pd.read_csv("data/raw/diabetes.csv")
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="Linear_Regression_Baseline"):
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Entrenamiento completado. MSE: {mse}")

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    run_training()
```

En la carpeta raiz (fuera de src) creamos el dvc.yaml - nuestro pipeline de ejecución del modelo
```
stages:
  load_data:
    cmd: python src/load_data.py
    deps:
      - src/load_data.py
    outs:
      - data/raw/diabetes.csv
  train:
    cmd: python src/train.py
    deps:
      - data/raw/diabetes.csv
      - src/train.py
    outs:
      - model.pkl  # DVC gestionará este binario
```
Para ejecutar el pipeline completo:

```
dvc repro
```
Concepto Clave: Idempotencia. Si se ejecuta ``` dvc repro ``` por segunda vez y nada ha cambiado (ni el código ni los datos), DVC no gastará CPU. Simplemente dirá que todo está al día.

Ejecutamos consola MLFlow para navegar a http://localhost:5000 y ver el historial
```
nohup mlflow ui --port 5000 &
```
