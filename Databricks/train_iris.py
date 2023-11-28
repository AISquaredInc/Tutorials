from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import mlflow.sklearn
import mlflow

USER_DIRECTORY = '/Users/jacob.renn@squared.ai'

MODEL_NAME = 'IRIS_DECISION_TREE'
EXPERIMENT_NAME = f'{USER_DIRECTORY}/IRIS_DECISION_TREE_EXPERIMENT'

data = load_iris()
x = data['data']
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)


with mlflow.start_run():
    mlflow.sklearn.autolog()
    max_depth = 4
    min_samples_split = 4
    
    model = DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split)
    model.fit(x_train, y_train)
    
    predictions = model.predict(x_test)

    test_accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric('test_accuracy_score', test_accuracy)
    
    conf_mat = str(confusion_matrix(y_test, predictions))
    mlflow.log_text(conf_mat, 'test_confusion_matrix.txt')
    
    run_id = mlflow.active_run().info.run_id
    model_uri = f'runs:/{run_id}/model'

    model_details = mlflow.register_model(
        model_uri = model_uri,
        name = MODEL_NAME
    )
