import time
import matplotlib.pyplot as plt

from Pipeline.Algorithm.ArtificialBeeColonyElm import ArtificialBeeColonyElm
# 导入你构建的组件
from Pipeline.Global.GallstoneDataSet import GallstoneDataSet
from Pipeline.Algorithm.EvaluationMatrix import EvaluationMatrix
from Pipeline.Global.GlobalSetting import GlobalSetting


def run_pipeline():
    print("=== [1] Data Ingestion & Preprocessing ===")
    dataset = GallstoneDataSet()
    dataset.fetch_data_path_1()

    x_train = dataset.x_train_scaled
    y_train = dataset.y_train
    x_test = dataset.x_test_scaled
    y_test = dataset.y_test

    print(f"Training features shape: {x_train.shape}")
    print(f"Testing features shape: {x_test.shape}\n")

    print("=== [2] ABC-ELM Optimization (K-Fold CV) ===")
    features_size = x_train.shape[1]
    hidden_size = 36

    # 实例化搭载了 CV 策略的 ABC-ELM
    abc_elm = ArtificialBeeColonyElm(
        features_size=features_size,
        hidden_size=hidden_size,
        activation_function=GlobalSetting.sigmoid,
        regularization_lambda=0.25,
        random_state= 42,
        fitness_function='F2-Score',  # 针对医疗数据（如胆结石），F2-Score 更看重 Recall
        solution_size=20,  # 蜜蜂种群数量
        trial_limit=20,
        max_iteration=100  # 最大迭代次数
    )

    start_time = time.time()
    # 启动训练，这里传入 cv_folds 触发内部交叉验证
    abc_elm.fit(x_train, y_train)
    training_time = time.time() - start_time

    print(f"\nOptimization Completed in {training_time:.2f} seconds.")
    print(f"Best CV Fitness (F2-Score): {abc_elm.best_fitness:.4f}\n")

    print("=== [3] Final Algorithm Testing ===")
    # 预测测试集
    y_pred = abc_elm.predict(x_test)

    # 评估指标
    eval_matrix = EvaluationMatrix(y_test, y_pred)
    report = eval_matrix.get_report()

    print("--- Confusion Matrix Counts ---")
    print(report["Counts"])
    print("--- Detailed Metrics ---")
    for metric, value in report["Metrics"].items():
        print(f"{metric.ljust(15)}: {value:.4f}")

    print("\n=== [4] Observability: Convergence Curve ===")


def plot_convergence(curve):
    if not curve:
        print("Warning: convergence_curve is empty. Did you add it to the base class?")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(curve) + 1), curve, marker='o', linestyle='-', color='b')
    plt.title('ABC-ELM Optimization Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (CV Score)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_pipeline()