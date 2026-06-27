# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")

    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth of trees")

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Step 2: read the train and test datasets
    train_df = pd.read_csv(os.path.join(args.train_data, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.test_data, "test.csv"))

    # Step 3: split features and target
    y_train = train_df["price"]
    X_train = train_df.drop(columns=["price"])
    y_test = test_df["price"]
    X_test = test_df.drop(columns=["price"])

    # Step 4: initialize and train the RandomForest Regressor
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Step 5: log hyperparameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Step 6: predict and compute MSE
    yhat = model.predict(X_test)
    mse = mean_squared_error(y_test, yhat)

    # Step 7: log MSE and save the model
    mlflow.log_metric("MSE", float(mse))
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":

    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

