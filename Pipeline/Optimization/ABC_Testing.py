import logging
import time

import pandas as pd

from Pipeline.Algorithm.EvaluationMatrix import EvaluationMatrix
from Pipeline.Global.GlobalSetting import GlobalSetting

logger = logging.getLogger(__name__)
def abc_testing(abc_model,x_train, y_train , x_test , y_test):

    monte_carlo_results = []
    convergence_history = {}
    scout_history = {}

    for seed in GlobalSetting.abc_testing_state_range:

        start_time = time.time()
        logger.info(f"Starting simulation for Seed: {seed}")

        abc_model_tested = abc_model
        abc_model_tested.init_random_state(seed)
        abc_model_tested.fit(x_train, y_train)

        convergence_history[seed] = abc_model_tested.convergence_curve
        scout_history[seed] = abc_model_tested.scout_trigger_history

        y_pred = abc_model_tested.predict(x_test)

        eval_metrics = EvaluationMatrix(y_true= y_test, y_pred=y_pred).get_all_metrics()
        eval_metrics['ABC_Seed'] = seed

        monte_carlo_results.append(eval_metrics)

        logger.info(f"Finished Seed: {seed} | Total Execution Time: {time.time() - start_time:.4f}s")

    return pd.DataFrame(monte_carlo_results), pd.DataFrame(convergence_history), pd.DataFrame(scout_history)
