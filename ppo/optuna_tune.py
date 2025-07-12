import optuna
import subprocess

def objective(trial):
    # 하이퍼파라미터 샘플링
    gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.995])
    hid = trial.suggest_categorical("hid", [64, 128, 256])
    l = trial.suggest_categorical("l", [1, 2, 3])
    steps = trial.suggest_categorical("steps", [2000, 4000, 8000])
    exp_name = f"optuna_trial_{trial.number}"

    # PPO 실행 명령어
    cmd = [
        "python", "ppo_main.py",  # ← 당신의 PPO 학습 파일 이름으로 바꾸세요
        "--env", "BipedalWalker-v3",
        "--gamma", str(gamma),
        "--hid", str(hid),
        "--l", str(l),
        "--steps", str(steps),
        "--epochs", "50",
        "--exp_name", exp_name
    ]

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        for line in output.splitlines():
            if "AverageEpRet" in line or "EpRet" in line:
                try:
                    reward = float(line.strip().split()[-1])
                    return reward
                except ValueError:
                    continue
    except subprocess.CalledProcessError as e:
        print("Training failed:\n", e.output)
        return -1e6  # 실패한 실험은 낮은 값 반환

# Optuna 튜닝 실행
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_params)
