from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
import os

# 必要な関数やモデルクラスをすべて含める（省略部分は上のコードからコピー）
# - NutritionModel クラス
# - train_model 関数
# - predict_ideal_values 関数
# - save_data 関数
# - その他のサポート関数 (e.g., calculate_metrics, calculate_ideal_intake)

app = Flask(__name__)

# データ保存用
DATA_FILE = "dataset.csv"
MODEL_FILE = "model.pth"

if os.path.exists(DATA_FILE):
    data = pd.read_csv(DATA_FILE)
else:
    data = pd.DataFrame(columns=["age", "weight", "height", "sex", "rice", "meat", "ideal_carbs", "ideal_protein"])

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import tkinter as tk
from tkinter import messagebox
from sklearn.metrics import mean_squared_error, r2_score
global testscore

# 精度計算の関数
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# データ保存ファイル名
#DATA_FILE = "data_with_height.csv"
DATA_FILE = "dataset.csv"
MODEL_FILE = "model.pth"

def save_model(model, file_path=MODEL_FILE):
    """
    モデルの重みを保存する関数
    """
    torch.save(model.state_dict(), file_path)
    print(f"モデルが {file_path} に保存されました。")

# データの初期化または読み込み
if os.path.exists(DATA_FILE):
    data = pd.read_csv(DATA_FILE)
    
else:
    data = pd.DataFrame(columns=['age', 'weight', 'height', 'sex', 'rice', 'meat', 'ideal_carbs', 'ideal_protein'])

# PyTorchモデル定義
class NutritionModel(nn.Module):
    def __init__(self):
        super(NutritionModel, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # 入力次元→中間次元
        #self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # 出力次元
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# モデルの学習と保存
def train_model(optimizer_type='adam', scheduler_type='CosineAnnealing'):
    if len(data) < 10:  # データが10件未満の場合は学習をスキップ
        return None, None
    global testscore
    X = data[["age", "height","weight", "sex"]].replace({"male": 0, "female": 1}).values
    y = data[["ideal_carbs", "ideal_protein"]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = NutritionModel()
    #criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()  # Huber損失
    # 最適化手法の選択
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    # 学習率スケジューラの設定
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExzponentialLR(optimizer, gamma=0.9)
    
    elif scheduler_type == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    # 学習
    epochs = 4096
    best_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()


         # 学習率のスケジューリング
        scheduler.step()

        # 早期終了
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    # 最終的な精度の計算
    model.eval()
    final_train_predictions = model(X_train)
    final_test_predictions = model(X_test)

    train_mse, train_r2 = calculate_metrics(y_train.detach().numpy(), final_train_predictions.detach().numpy())
    test_mse, test_r2 = calculate_metrics(y_test.detach().numpy(), final_test_predictions.detach().numpy())
    
    print(train_mse,train_r2)
    print(test_mse,test_r2)
    testscore = test_r2

    # 最終精度のファイル保存
    with open("final_accuracy.txt", "w") as log_file:
        log_file.write(f"Final Train MSE: {train_mse:.4f}, Train R^2: {train_r2:.4f}\n")
        log_file.write(f"Final Test MSE: {test_mse:.4f}, Test R^2: {test_r2:.4f}\n")

    # モデルを保存
    torch.save(model.state_dict(), MODEL_FILE)
    return model, scaler

# モデルの予測
def predict_ideal_values(age, height,weight, sex):
    if os.path.exists(MODEL_FILE):
        print(f"モデル {MODEL_FILE} をロードしました。")
        model = NutritionModel()
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()

        # 新しいスケーラーを使うためtrain_model()を呼び出してスケーラーを取得
        model, scaler = train_model()  # モデル再学習後、スケーラーも更新される

    else:
        print("モデルファイルが見つかりません。")
        model, scaler = train_model()
        if model is None:
            return None, None

    sex_numeric = 0 if sex == "male" else 1
    input_data = scaler.transform([[age, height,weight, sex_numeric]])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    ideal_values = model(input_tensor).detach().numpy()
    return ideal_values[0][0], ideal_values[0][1]
 
# 入力データの保存
def save_data(age, height ,weight, sex, rice, meat, ideal_carbs, ideal_protein):
    global data
    new_entry = pd.DataFrame({
        "age": [age],
        "weight": [weight],
        "height": [height],
        "sex": [sex],
        "rice": [rice],
        "meat": [meat],
        "ideal_carbs": [ideal_carbs],
        "ideal_protein": [ideal_protein]
    })
    data = pd.concat([data, new_entry], ignore_index=True)
    data.to_csv(DATA_FILE, index=False)

    # 新しいデータが追加されたので、モデルを再学習して保存
    model, scaler = train_model()
    if model:
        save_model(model)  # モデルを保存
# BMIに基づいて理想的な炭水化物とタンパク質を計算
def calculate_ideal_intake(bmi):
    # 炭水化物量とタンパク質量の範囲を設定
    carbs_min, carbs_max = 150, 250  # 炭水化物量 (g)
    protein_min, protein_max = 50, 70  # タンパク質量 (g)

    # 標準BMI範囲
    bmi_min, bmi_max = 18.5, 24.9

    # 炭水化物量の計算
    if bmi < bmi_min:  # 低体重の場合
        ideal_carbs = carbs_max
        ideal_protein = protein_max
    elif bmi > bmi_max:  # 肥満の場合
        ideal_carbs = carbs_min
        ideal_protein = protein_min
    else:  # 標準範囲の場合は線形補間
        ideal_carbs = carbs_min + (carbs_max - carbs_min) * (bmi_max - bmi) / (bmi_max - bmi_min)
        ideal_protein = protein_min + (protein_max - protein_min) * (bmi_max - bmi) / (bmi_max - bmi_min)

    return ideal_carbs, ideal_protein

# ご飯とお肉をもとに栄養を計算
def calculate_nutrition(rice, meat):
    # ご飯150gで炭水化物55g
    carbs = rice * (55 / 150)
    # 鶏むね肉100gでタンパク質22g
    protein = meat * (22 / 100)
    return carbs, protein

# 不足分を補う提案
def suggest_food(carb_deficit, protein_deficit):
    FOOD_LIST = [
        {"name": "ご飯（150g）", "carbs": 55, "protein": 4},
        {"name": "鶏むね肉（100g）", "carbs": 0, "protein": 22},
        {"name": "卵（1個）", "carbs": 0.6, "protein": 6},
        {"name": "バナナ（1本）", "carbs": 27, "protein": 1},
        {"name": "パン（1枚）", "carbs": 20, "protein": 3},
    ]

    suggestions = []

    while carb_deficit > 0 or protein_deficit > 0:
        found = False
        for food in FOOD_LIST:
            # 必要量を超える食品は無視
            if carb_deficit > 0 and food["carbs"] <= carb_deficit:
                if(food["protein"] > protein_deficit):
                    continue
                suggestions.append(food["name"])
                carb_deficit -= food["carbs"]
                protein_deficit -= food["protein"]
                found = True
                break
            elif protein_deficit > 0 and food["protein"] <= protein_deficit:
                if(food["carbs"] > carb_deficit):
                    continue
                suggestions.append(food["name"])
                carb_deficit -= food["carbs"]
                protein_deficit -= food["protein"]
                found = True
                break

        # 提案可能な食品がなくなったら終了
        if not found:
            break

    return suggestions




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        # フォームからデータを取得
        age = int(request.form['age'])
        height = float(request.form['height']) / 100  # cmをmに変換
        weight = float(request.form['weight'])
        sex = request.form['sex']
        rice = float(request.form['rice'])
        meat = float(request.form['meat'])

        # BMIの計算
        bmi = weight / (height ** 2)
        ideal_weight = 18.5 * (height ** 2)

        # 深層学習モデルによる予測
        ideal_carbs, ideal_protein = predict_ideal_values(age, height * 100, weight, sex)

        # モデルが使えなければフォールバックとしてBMI基準を使用
        if ideal_carbs is None or ideal_protein is None:
            ideal_carbs, ideal_protein = calculate_ideal_intake(bmi)

        # 現在の栄養素計算
        carbs, protein = calculate_nutrition(rice, meat)
        carb_deficit = max(0, ideal_carbs - carbs)
        protein_deficit = max(0, ideal_protein - protein)

        # 提案する食事
        suggestions = suggest_food(carb_deficit, protein_deficit)

        # データ保存
        save_data(age, height * 100, weight, sex, rice, meat, ideal_carbs, ideal_protein)

        result = {
                "BMI": round(float(bmi), 1),
                "ideal_weight": round(float(ideal_weight), 1),
                "ideal_carbs": round(float(ideal_carbs), 1),
                "ideal_protein": round(float(ideal_protein), 1),
                "current_carbs": round(float(carbs), 1),
                "current_protein": round(float(protein), 1),
                "carb_deficit": round(float(carb_deficit), 1),
                "protein_deficit": round(float(protein_deficit), 1),
                "suggestions": suggestions
                }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)
